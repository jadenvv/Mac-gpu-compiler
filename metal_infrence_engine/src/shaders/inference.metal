// Metal compute kernels for the Qwen3 forward pass.
//
// Compiled at runtime by MetalModel::load_model via device.new_library_with_source
// (include_str!), so no xcrun / Metal toolchain / build.rs is needed. A syntax error
// here shows up as a panic at load time with the compiler log.
//
// Every kernel mirrors a function in cpu_inference.rs; use those as the reference.
// The param structs MUST match the #[repr(C)] structs in metal_inference.rs
// field-for-field (only uint/float fields, so the layouts line up trivially).

#include <metal_stdlib>
using namespace metal;

// bf16 -> f32: same trick as bf16_to_f32 in cpu_inference.rs.
// weights are passed as `device const ushort*` rather than MSL's `bfloat` type,
// which is emulated on M1 anyway.
inline float bf16(ushort x) { return as_type<float>(uint(x) << 16); }

// ---------------------------------------------------------------------------
// param structs (keep in sync with metal_inference.rs)
// ---------------------------------------------------------------------------

struct MatvecParams {
    uint rows;          // out.len()
    uint cols;          // x.len()
};

struct NormParams {
    uint n;             // elements per vector (hidden_size, or head_dim for q/k norm)
    float eps;
};

struct RopeParams {
    uint head_dim;
    float eps;          // q/k norm is fused into this kernel
};

struct AttnParams {
    uint n_kv_heads;
    uint head_dim;
    uint pos;           // attend over cache positions 0..=pos
    float scale;        // 1 / sqrt(head_dim)
};

// ---------------------------------------------------------------------------
// threadgroup reductions
// ---------------------------------------------------------------------------
// Every thread in the threadgroup must call these (they contain barriers) and
// every thread gets the result back. `buf` needs >= 32 floats. The trailing
// barrier lets the caller reuse `buf` for the next reduction straight away.

inline float tg_sum(float v, threadgroup float* buf, uint lane, uint sg, uint tpg)
{
    uint nsg = (tpg + 31) / 32;
    v = simd_sum(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = simd_sum(lane < nsg ? buf[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float tg_max(float v, threadgroup float* buf, uint lane, uint sg, uint tpg)
{
    uint nsg = (tpg + 31) / 32;
    v = simd_max(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = simd_max(lane < nsg ? buf[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// same barrier contract as tg_sum, for the lowest index in argmax
inline uint tg_min(uint v, threadgroup uint* buf, uint lane, uint sg, uint tpg)
{
    uint nsg = (tpg + 31) / 32;
    v = simd_min(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = simd_min(lane < nsg ? buf[lane] : UINT_MAX);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// max head_dim supported by qk_norm_rope, and max context for attention. attention keeps
// scores for two q heads in threadgroup memory, so 2 * 4 * MAX_SEQ has to fit in 32 KB
// along with its reduction scratch.
constant uint MAX_HEAD_DIM = 256;
constant uint MAX_SEQ      = 4064;

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// out[i] = bf16(table[token * n + i])
// grid: n threads
kernel void embed(
    device const ushort* table  [[buffer(0)]],
    device float*        out    [[buffer(1)]],
    device const uint&   token  [[buffer(2)]],   // a buffer, so the previous step's argmax can feed it
    constant uint&       n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = bf16(table[token * n + gid]);
}

// out = x * rsqrt(mean(x^2) + eps) * w        (cpu: rmsnorm)
// grid: ONE threadgroup (n is only 1024), so the reduction stays inside it.
// hint: each thread sums a strided slice, then simd_sum(), then combine the
//       per-simdgroup partials through a threadgroup array.
kernel void rmsnorm(
    device const float*  x      [[buffer(0)]],
    device const ushort* w      [[buffer(1)]],
    device float*        out    [[buffer(2)]],
    constant NormParams& p      [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float buf[32];

    float ss = 0.0f;
    for (uint i = tid; i < p.n; i += tpg) ss += x[i] * x[i];
    float s = rsqrt(tg_sum(ss, buf, lane, sg, tpg) / float(p.n) + p.eps);

    for (uint i = tid; i < p.n; i += tpg) out[i] = x[i] * s * bf16(w[i]);
}

// out[r] = sum_c bf16(w[r * cols + c]) * x[c]    (cpu: matvec + dot_bf16)
// This is where ~all decode time goes, so it decides your MBU.
// mapping: each simdgroup (32 threads) owns ROWS_PER_SG rows; lanes stride over cols.
//       All of a simdgroup's rows advance through the same chunk together, so their
//       weight loads are in flight at once and share one x load. 256-thread
//       threadgroups = 8 simdgroups = 8 * ROWS_PER_SG rows each.
// ROWS_PER_SG is set at load time by metal_inference.rs (env MATVEC_ROWS_PER_SG).
#ifndef ROWS_PER_SG
#define ROWS_PER_SG 1
#endif

kernel void matvec_bf16(
    device const packed_uint4*   w    [[buffer(0)]],
    device const float4*    x    [[buffer(1)]],
    device float*          out  [[buffer(2)]],
    constant MatvecParams& p    [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
	const uint first_row = (tgid * sgs + sg) * ROWS_PER_SG;
	if (first_row >= p.rows) return;

	const uint chunks = p.cols / 8;   // one chunk = 16 bytes = 8 weights

	// Where each row starts. Past the end of the matrix (the last, partial simdgroup)
	// a slot re-reads the final row, which stays in bounds and keeps the loop below a
	// fixed size for the compiler to unroll. Those slots are never written out.
	uint row_start[ROWS_PER_SG];
	float acc[ROWS_PER_SG];
	for (uint r = 0; r < ROWS_PER_SG; r++) {
		row_start[r] = min(first_row + r, p.rows - 1) * chunks;
		acc[r] = 0.0f;
	}

	const uint hi = 0xFFFF0000u;
	for (uint i = lane; i < chunks; i += 32) {
		// shared by every row in this simdgroup
		float4 x0 = x[2 * i];
		float4 x1 = x[2 * i + 1];

		for (uint r = 0; r < ROWS_PER_SG; r++) {
			uint4 u = w[row_start[r] + i];
			float4 w0 = as_type<float4>(uint4(u.x << 16, u.x & hi, u.y << 16, u.y & hi));
			float4 w1 = as_type<float4>(uint4(u.z << 16, u.z & hi, u.w << 16, u.w & hi));
			acc[r] += dot(w0, x0) + dot(w1, x1);
		}
	}

	for (uint r = 0; r < ROWS_PER_SG; r++) {
		float total = simd_sum(acc[r]);
		if (lane == 0 && first_row + r < p.rows) out[first_row + r] = total;
	}
}

// in place, per head: rmsnorm(head, norm_w) then rope(head)   (cpu: the q_norm/k_norm +
// rope loops in forward). Used for both q and k. rope is this position's [cos, sin] for
// each i < head_dim/2, computed once on the CPU (MetalKVCache::new).
// grid: one threadgroup per head, head_dim threads. The rmsnorm reduction is
//       per-head, and rope pairs element i with i + head_dim/2, so you need a
//       threadgroup_barrier between normalizing and rotating.
kernel void qk_norm_rope(
    device float*        x      [[buffer(0)]],   // [n_heads * head_dim]
    device const ushort* norm_w [[buffer(1)]],   // [head_dim]
    device const float2* rope   [[buffer(2)]],   // [head_dim / 2]
    constant RopeParams& p      [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float buf[32];
    threadgroup float v[MAX_HEAD_DIM];

    const uint hd = p.head_dim;
    device float* h = x + head * hd;

    // rmsnorm over this head
    float xi = tid < hd ? h[tid] : 0.0f;
    float s = rsqrt(tg_sum(xi * xi, buf, lane, sg, tpg) / float(hd) + p.eps);
    if (tid < hd) v[tid] = xi * s * bf16(norm_w[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rope, HF rotate_half layout: i pairs with i + half
    const uint half_d = hd / 2;
    if (tid < half_d) {
        float2 r = rope[tid];
        float a = v[tid], b = v[tid + half_d];
        h[tid]          = a * r.x - b * r.y;
        h[tid + half_d] = b * r.x + a * r.y;
    }
}

// out[h] = softmax(q[h] . K[0..=pos] * scale) @ V[0..=pos]   with GQA: each kv head serves
// exactly two q heads (2 * kv_head and 2 * kv_head + 1), and both run together here so they
// share one read of that head's K and V.
// cache layout matches the CPU one: k_cache[t * kv_dim + kv_head * head_dim + i].
// grid: one threadgroup per KV head, head_dim threads (must be 128: 4 simdgroups of 32).
// Scores use the matvec pattern: a simdgroup owns a position and its lanes cover the head as
// 32 float4s, so one position's loads are adjacent instead of 4 KB apart.
kernel void attention(
    device const float4* q       [[buffer(0)]],
    device const float4* k_cache [[buffer(1)]],
    device const float*  v_cache [[buffer(2)]],
    device float*        out     [[buffer(3)]],
    constant AttnParams& p       [[buffer(4)]],
    uint kvh  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    threadgroup float buf[32];
    threadgroup float scores_a[MAX_SEQ];   // caller must keep pos < MAX_SEQ
    threadgroup float scores_b[MAX_SEQ];

    const uint hd     = p.head_dim;
    const uint kv_dim = p.n_kv_heads * hd;
    const uint n_t    = p.pos + 1;
    const uint chunks = hd / 4;

    const float4 qa = q[2 * kvh * chunks + lane];
    const float4 qb = q[(2 * kvh + 1) * chunks + lane];

    // scores[t] = q . k[t] * scale, one position per simdgroup
    float local_a = -INFINITY, local_b = -INFINITY;
    for (uint t = sg; t < n_t; t += sgs) {
        float4 k = k_cache[(t * kv_dim + kvh * hd) / 4 + lane];
        float a = simd_sum(dot(qa, k)) * p.scale;
        float b = simd_sum(dot(qb, k)) * p.scale;
        if (lane == 0) { scores_a[t] = a; scores_b[t] = b; }
        local_a = max(local_a, a);
        local_b = max(local_b, b);
    }
    const float max_a = tg_max(local_a, buf, lane, sg, tpg);
    const float max_b = tg_max(local_b, buf, lane, sg, tpg);

    // exp + sum
    float sum_a = 0.0f, sum_b = 0.0f;
    for (uint t = tid; t < n_t; t += tpg) {
        float ea = precise::exp(scores_a[t] - max_a);
        float eb = precise::exp(scores_b[t] - max_b);
        scores_a[t] = ea;
        scores_b[t] = eb;
        sum_a += ea;
        sum_b += eb;
    }
    // tg_sum's barriers also make all scores writes visible below
    const float inv_a = 1.0f / tg_sum(sum_a, buf, lane, sg, tpg);
    const float inv_b = 1.0f / tg_sum(sum_b, buf, lane, sg, tpg);

    // out[i] = sum_t p[t] * v[t][i], one output dim per thread
    device const float* v = v_cache + kvh * hd + tid;
    float acc_a = 0.0f, acc_b = 0.0f;
    for (uint t = 0; t < n_t; t++) {
        float vt = v[t * kv_dim];
        acc_a += scores_a[t] * vt;
        acc_b += scores_b[t] * vt;
    }
    out[2 * kvh * hd + tid]       = acc_a * inv_a;
    out[(2 * kvh + 1) * hd + tid] = acc_b * inv_b;
}

// *out = index of the largest x[i], lowest index on ties   (cpu: argmax)
// grid: ONE threadgroup. Lets the next step's embed read this token without the CPU.
kernel void argmax(
    device const float* x   [[buffer(0)]],
    device uint&        out [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float fbuf[32];
    threadgroup uint  ubuf[32];

    float best = -INFINITY;
    uint best_i = UINT_MAX;
    for (uint i = tid; i < n; i += tpg) {
        if (x[i] > best) { best = x[i]; best_i = i; }
    }
    float m = tg_max(best, fbuf, lane, sg, tpg);
    uint i = tg_min(best == m ? best_i : UINT_MAX, ubuf, lane, sg, tpg);
    // nothing compared greater (all NaN or -inf): the CPU argmax returns 0
    if (tid == 0) out = i == UINT_MAX ? 0 : i;
}

// gate[i] = silu(gate[i]) * up[i]
// grid: n threads
kernel void silu_mul(
    device float*       gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint&      n    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = gate[gid];
    gate[gid] = g / (1.0f + precise::exp(-g)) * up[gid];
}

// x[i] += y[i]   (residual)
// grid: n threads
kernel void add_inplace(
    device float*       x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] += y[gid];
}
