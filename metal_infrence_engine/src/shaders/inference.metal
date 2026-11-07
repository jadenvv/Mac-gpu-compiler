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
    uint pos;
    float theta;
    float eps;          // q/k norm is fused into this kernel
};

struct AttnParams {
    uint n_heads;
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

// max head_dim supported by qk_norm_rope / attention, and max context for attention
constant uint MAX_HEAD_DIM = 256;
constant uint MAX_SEQ      = 4096;

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// out[i] = bf16(table[token * n + i])
// grid: n threads
kernel void embed(
    device const ushort* table  [[buffer(0)]],
    device float*        out    [[buffer(1)]],
    constant uint&       token  [[buffer(2)]],
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
// suggested mapping: one simdgroup (32 threads) per row; lanes stride over cols,
//       then simd_sum(). Grid = rows simdgroups, e.g. threadgroups of 256 threads
//       = 8 rows each -> rows / 8 threadgroups.
// later: read ushort4 / x as float4 for wider loads, several rows per simdgroup.
kernel void matvec_bf16(
    device const ushort*   w    [[buffer(0)]],
    device const float*    x    [[buffer(1)]],
    device float*          out  [[buffer(2)]],
    constant MatvecParams& p    [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint sgs  [[simdgroups_per_threadgroup]])
{

	




}

// in place, per head: rmsnorm(head, norm_w) then rope(head, pos)   (cpu: the
// q_norm/k_norm + rope loops in forward). Used for both q and k.
// grid: one threadgroup per head, head_dim threads. The rmsnorm reduction is
//       per-head, and rope pairs element i with i + head_dim/2, so you need a
//       threadgroup_barrier between normalizing and rotating.
kernel void qk_norm_rope(
    device float*        x      [[buffer(0)]],   // [n_heads * head_dim]
    device const ushort* norm_w [[buffer(1)]],   // [head_dim]
    constant RopeParams& p      [[buffer(2)]],
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
        // precise:: so this matches the CPU powf / sin_cos for parity
        float freq = precise::pow(p.theta, -2.0f * float(tid) / float(hd));
        float ang  = float(p.pos) * freq;
        float sn = precise::sin(ang), cs = precise::cos(ang);
        float a = v[tid], b = v[tid + half_d];
        h[tid]          = a * cs - b * sn;
        h[tid + half_d] = b * cs + a * sn;
    }
}

// out[h] = softmax(q[h] . K[0..=pos] * scale) @ V[0..=pos]   with GQA:
// q head h reads kv head h / (n_heads / n_kv_heads).
// cache layout matches the CPU one: k_cache[t * kv_dim + kv_head * head_dim + i].
// grid: one threadgroup per q head.
// simple version: scores in a threadgroup array (caps pos at its size, e.g. 4096),
//       max-reduce, exp + sum-reduce, then each thread owns output dims.
// later: online softmax (flash-decoding style) to drop the length cap.
kernel void attention(
    device const float*  q       [[buffer(0)]],
    device const float*  k_cache [[buffer(1)]],
    device const float*  v_cache [[buffer(2)]],
    device float*        out     [[buffer(3)]],
    constant AttnParams& p       [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float buf[32];
    threadgroup float qs[MAX_HEAD_DIM];
    threadgroup float scores[MAX_SEQ];   // caller must keep pos < MAX_SEQ

    const uint hd     = p.head_dim;
    const uint kv_dim = p.n_kv_heads * hd;
    const uint kvh    = head / (p.n_heads / p.n_kv_heads);
    const uint n_t    = p.pos + 1;

    // stage this head's q in threadgroup memory; every thread reads all of it
    for (uint i = tid; i < hd; i += tpg) qs[i] = q[head * hd + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // scores[t] = q . k[t] * scale, one position per thread
    float local_max = -INFINITY;
    for (uint t = tid; t < n_t; t += tpg) {
        device const float* k = k_cache + t * kv_dim + kvh * hd;
        float d = 0.0f;
        for (uint i = 0; i < hd; i++) d += qs[i] * k[i];
        d *= p.scale;
        scores[t] = d;
        local_max = max(local_max, d);
    }
    float m = tg_max(local_max, buf, lane, sg, tpg);

    // exp + sum
    float local_sum = 0.0f;
    for (uint t = tid; t < n_t; t += tpg) {
        float e = precise::exp(scores[t] - m);
        scores[t] = e;
        local_sum += e;
    }
    // tg_sum's barriers also make all scores[] writes visible below
    float inv = 1.0f / tg_sum(local_sum, buf, lane, sg, tpg);

    // out[i] = sum_t p[t] * v[t][i], each thread owns output dims
    for (uint i = tid; i < hd; i += tpg) {
        device const float* v = v_cache + kvh * hd + i;
        float acc = 0.0f;
        for (uint t = 0; t < n_t; t++) acc += scores[t] * v[t * kv_dim];
        out[head * hd + i] = acc * inv;
    }
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
