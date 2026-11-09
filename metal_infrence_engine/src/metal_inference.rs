// Metal (GPU) version of cpu_inference.rs: same model, same math, same public shape
// (MetalModel::load_model / forward(token, pos, cache) -> logits), so the runner and
// tests can swap one for the other.
//
// Design:
//  - Weights are NOT copied. The whole safetensors mmap is wrapped as one MTLBuffer
//    (new_buffer_with_bytes_no_copy) and each tensor is bound with a byte offset.
//    Apple silicon has unified memory, so the GPU reads the same pages the CPU mapped.
//  - All activations live in StorageModeShared buffers allocated once (Scratch).
//  - The KV cache is preallocated per layer for max_seq positions.
//  - forward() encodes the whole token (all 28 layers) into ONE command buffer,
//    commits, waits, then reads logits straight out of the shared buffer.
//
// Kernels live in shaders/inference.metal.

use crate::cpu_inference::{Config, Model};
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::io;

const SHADER_SRC: &str = include_str!("shaders/inference.metal");

// no_copy buffers need a page-aligned pointer and length (16 KiB on Apple silicon)
const PAGE_SIZE: usize = 16 * 1024;
// the threadgroup scores[] array in the attention kernel; keep in sync with inference.metal
const MAX_SEQ: usize = 4096;
// simdgroup width the kernels hardcode (lane strides and tg_sum/tg_max)
const SIMD_WIDTH: u64 = 32;
// threads per threadgroup for matvec_bf16: 8 simdgroups
const MATVEC_TG: usize = 256;
const MATVEC_SIMDGROUPS_PER_TG: usize = MATVEC_TG / SIMD_WIDTH as usize;
// rows each matvec simdgroup handles, overridable with env MATVEC_ROWS_PER_SG
const DEFAULT_ROWS_PER_SG: usize = 1;
const MAX_ROWS_PER_SG: usize = 16;
// threads per threadgroup for the elementwise kernels (embed, silu_mul, add_inplace)
const ELEMENTWISE_TG: usize = 256;

// ---------------------------------------------------------------------------
// kernel params, must match the structs in inference.metal field-for-field
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatvecParams { pub rows: u32, pub cols: u32 }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NormParams { pub n: u32, pub eps: f32 }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RopeParams { pub head_dim: u32, pub pos: u32, pub theta: f32, pub eps: f32 }

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnParams { pub n_heads: u32, pub n_kv_heads: u32, pub head_dim: u32, pub pos: u32, pub scale: f32 }

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

// one compiled pipeline per kernel in inference.metal
struct Pipelines {
    embed: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    matvec_bf16: ComputePipelineState,
    qk_norm_rope: ComputePipelineState,
    attention: ComputePipelineState,
    silu_mul: ComputePipelineState,
    add_inplace: ComputePipelineState,
}

// f32 activation buffers, same names/sizes as the Vecs at the top of the CPU forward()
struct Scratch {
    x: Buffer,      // [hidden]
    xn: Buffer,     // [hidden]
    q: Buffer,      // [n_heads * head_dim]
    att: Buffer,    // [n_heads * head_dim]
    proj: Buffer,   // [hidden]
    gate: Buffer,   // [intermediate]
    up: Buffer,     // [intermediate]
    logits: Buffer, // [vocab]
    // no k/v here: write k_proj/v_proj output directly into the cache at
    // offset pos * kv_dim * 4, then run qk_norm_rope on that slice in place
}

pub struct MetalModel {
    // reuses the CPU loader: owns the mmap, tensor table and config
    pub cpu: Model,
    device: Device,
    queue: CommandQueue,
    pipelines: Pipelines,
    scratch: Scratch,
    // the entire mmap as one no-copy GPU buffer
    weights: Buffer,
    // tensor name -> byte offset into `weights` (8 + header_len + data_offsets[0])
    offsets: HashMap<String, u64>,
    // compiled into matvec_bf16 as ROWS_PER_SG, and sizes its dispatch
    rows_per_sg: usize,
}

pub struct MetalKVCache {
    // per layer, [max_seq * kv_heads * head_dim] f32, same layout as KVCache
    k: Vec<Buffer>,
    v: Vec<Buffer>,
    pub max_seq: usize,
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

impl MetalKVCache {
    pub fn new(model: &MetalModel, max_seq: usize) -> Self {
        assert!(max_seq <= MAX_SEQ, "max_seq {max_seq} exceeds the attention kernel's limit of {MAX_SEQ}");
        let c = model.config();
        let kv_dim = c.num_key_value_heads * c.head_dim;
        let alloc = || -> Vec<Buffer> {
            (0..c.num_hidden_layers).map(|_| new_f32_buffer(&model.device, max_seq * kv_dim)).collect()
        };
        MetalKVCache { k: alloc(), v: alloc(), max_seq }
    }
}

impl MetalModel {
    pub fn load_model(file: &str) -> io::Result<Self> {
        let cpu = Model::load_model(file)?;
        let device = Device::system_default()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no Metal device found"))?;
        let queue = device.new_command_queue();
        let rows_per_sg = rows_per_sg_from_env();
        let pipelines = compile_pipelines(&device, rows_per_sg);
        check_thread_limits(&pipelines, &cpu.config);
        let (weights, offsets) = wrap_weights(&device, &cpu);
        let scratch = Scratch::new(&device, &cpu.config);
        Ok(MetalModel { cpu, device, queue, pipelines, scratch, weights, offsets, rows_per_sg })
    }

    pub fn config(&self) -> &Config { &self.cpu.config }

    // (buffer, offset) to bind tensor `name`; asserts it is BF16 like Model::weight
    fn weight(&self, name: &str) -> (&Buffer, u64) {
        let off = *self.offsets.get(name).unwrap_or_else(|| panic!("missing BF16 tensor {name}"));
        (&self.weights, off)
    }
}

fn rows_per_sg_from_env() -> usize {
    let n = match std::env::var("MATVEC_ROWS_PER_SG") {
        Ok(v) => v.parse().unwrap_or_else(|_| panic!("MATVEC_ROWS_PER_SG must be a number, got {v:?}")),
        Err(_) => DEFAULT_ROWS_PER_SG,
    };
    assert!((1..=MAX_ROWS_PER_SG).contains(&n), "MATVEC_ROWS_PER_SG must be 1..={MAX_ROWS_PER_SG}, got {n}");
    n
}

// compile SHADER_SRC once and build a pipeline for every kernel
fn compile_pipelines(device: &Device, rows_per_sg: usize) -> Pipelines {
    // ROWS_PER_SG sizes arrays in the kernel, so it must be known at compile time.
    // #line keeps compiler error line numbers matching inference.metal.
    let src = format!("#define ROWS_PER_SG {rows_per_sg}\n#line 1\n{SHADER_SRC}");
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap_or_else(|e| panic!("shader compile failed:\n{e}"));
    let make = |name: &str| {
        let f = lib.get_function(name, None).unwrap_or_else(|e| panic!("missing kernel {name}: {e}"));
        device.new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| panic!("pipeline {name} failed: {e}"))
    };
    Pipelines {
        embed: make("embed"),
        rmsnorm: make("rmsnorm"),
        matvec_bf16: make("matvec_bf16"),
        qk_norm_rope: make("qk_norm_rope"),
        attention: make("attention"),
        silu_mul: make("silu_mul"),
        add_inplace: make("add_inplace"),
    }
}

// the dispatch sizes in the encode_* helpers rely on these; checked once at load so a
// mismatch fails loudly instead of silently producing wrong numbers
fn check_thread_limits(p: &Pipelines, c: &Config) {
    for (name, pso) in [("matvec_bf16", &p.matvec_bf16), ("rmsnorm", &p.rmsnorm),
                        ("qk_norm_rope", &p.qk_norm_rope), ("attention", &p.attention)] {
        assert_eq!(pso.thread_execution_width(), SIMD_WIDTH, "{name}: kernels assume 32-wide simdgroups");
    }
    // tg_sum/tg_max need whole simdgroups
    assert_eq!(c.hidden_size as u64 % SIMD_WIDTH, 0, "hidden_size must be a multiple of the simd width");
    assert_eq!(c.head_dim as u64 % SIMD_WIDTH, 0, "head_dim must be a multiple of the simd width");
    assert!(c.hidden_size as u64 <= p.rmsnorm.max_total_threads_per_threadgroup());
    assert!(c.head_dim as u64 <= p.qk_norm_rope.max_total_threads_per_threadgroup());
    assert!(c.head_dim as u64 <= p.attention.max_total_threads_per_threadgroup());
    assert!(MATVEC_TG as u64 <= p.matvec_bf16.max_total_threads_per_threadgroup());
}

// wrap the mmap as one MTLBuffer without copying, and compute per-tensor offsets
fn wrap_weights(device: &Device, cpu: &Model) -> (Buffer, HashMap<String, u64>) {
    // mmap is page aligned, and the mapping covers the rest of the last page,
    // so rounding the length up is safe
    let ptr = cpu.mmap.as_ptr() as *const std::ffi::c_void;
    let len = cpu.mmap.len().next_multiple_of(PAGE_SIZE) as u64;
    let weights = device.new_buffer_with_bytes_no_copy(ptr, len, MTLResourceOptions::StorageModeShared, None);

    let header_bytes: [u8; 8] = cpu.mmap[0..8].try_into().unwrap();
    let data_start = 8 + u64::from_le_bytes(header_bytes);

    // only BF16 tensors are listed, so MetalModel::weight rejects anything else
    let mut offsets = HashMap::new();
    for t in cpu.tensors.iter().filter(|t| t.dtype == "BF16") {
        let off = data_start + t.data_offsets[0] as u64;
        // matvec_bf16 reads packed_uint4, which needs 4-byte alignment
        assert_eq!(off % 4, 0, "{}: offset {off} is not 4-byte aligned", t.name);
        offsets.insert(t.name.clone(), off);
    }
    (weights, offsets)
}

impl Scratch {
    fn new(device: &Device, c: &Config) -> Self {
        let q_dim = c.num_attention_heads * c.head_dim;
        Scratch {
            x: new_f32_buffer(device, c.hidden_size),
            xn: new_f32_buffer(device, c.hidden_size),
            q: new_f32_buffer(device, q_dim),
            att: new_f32_buffer(device, q_dim),
            proj: new_f32_buffer(device, c.hidden_size),
            gate: new_f32_buffer(device, c.intermediate_size),
            up: new_f32_buffer(device, c.intermediate_size),
            logits: new_f32_buffer(device, c.vocab_size),
        }
    }
}

fn new_f32_buffer(device: &Device, len: usize) -> Buffer {
    device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared)
}

// read a shared f32 buffer back on the CPU (only valid after the command buffer completed)
fn read_f32(buf: &Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

impl MetalModel {
    // same contract as Model::forward: runs `token` at `pos`, writes its K/V into the
    // cache, returns logits. pos must be < cache.max_seq.
    pub fn forward(&self, token: u32, pos: usize, cache: &mut MetalKVCache) -> Vec<f32> {
        assert!(pos < cache.max_seq, "pos {pos} exceeds KV cache size {}", cache.max_seq);
        let c = self.config();
        let s = &self.scratch;
        let hidden = c.hidden_size;
        let inter = c.intermediate_size;
        let q_dim = c.num_attention_heads * c.head_dim;
        let kv_dim = c.num_key_value_heads * c.head_dim;
        // this token's slot in the KV cache, in bytes
        let kv_off = (pos * kv_dim * 4) as u64;

        // the command buffer and encoder are autoreleased, so without a pool
        // they would pile up for the life of the process
        objc::rc::autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            // Within one compute encoder, Metal tracks hazards on these buffers and
            // serializes dependent dispatches for you, so no manual barriers needed.
            self.encode_embed(enc, token);
            for l in 0..c.num_hidden_layers {
                let w = |n: &str| format!("model.layers.{l}.{n}");
                let (k_cache, v_cache) = (&cache.k[l], &cache.v[l]);

                // attention
                self.encode_rmsnorm(enc, &s.x, &w("input_layernorm.weight"), &s.xn, hidden);
                self.encode_matvec(enc, &w("self_attn.q_proj.weight"), &s.xn, &s.q, 0, q_dim, hidden);
                self.encode_matvec(enc, &w("self_attn.k_proj.weight"), &s.xn, k_cache, kv_off, kv_dim, hidden);
                self.encode_matvec(enc, &w("self_attn.v_proj.weight"), &s.xn, v_cache, kv_off, kv_dim, hidden);
                self.encode_qk_norm_rope(enc, &s.q, 0, &w("self_attn.q_norm.weight"), c.num_attention_heads, pos);
                self.encode_qk_norm_rope(enc, k_cache, kv_off, &w("self_attn.k_norm.weight"), c.num_key_value_heads, pos);
                self.encode_attention(enc, k_cache, v_cache, pos);
                self.encode_matvec(enc, &w("self_attn.o_proj.weight"), &s.att, &s.proj, 0, hidden, q_dim);
                self.encode_add(enc, &s.x, &s.proj, hidden);

                // swiglu mlp
                self.encode_rmsnorm(enc, &s.x, &w("post_attention_layernorm.weight"), &s.xn, hidden);
                self.encode_matvec(enc, &w("mlp.gate_proj.weight"), &s.xn, &s.gate, 0, inter, hidden);
                self.encode_matvec(enc, &w("mlp.up_proj.weight"), &s.xn, &s.up, 0, inter, hidden);
                self.encode_silu_mul(enc);
                self.encode_matvec(enc, &w("mlp.down_proj.weight"), &s.gate, &s.proj, 0, hidden, inter);
                self.encode_add(enc, &s.x, &s.proj, hidden);
            }
            self.encode_rmsnorm(enc, &s.x, "model.norm.weight", &s.xn, hidden);
            self.encode_matvec(enc, "lm_head.weight", &s.xn, &s.logits, 0, c.vocab_size, hidden);

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "GPU command buffer failed");
        });

        read_f32(&s.logits, c.vocab_size)
    }

    // ---- one helper per kernel; each binds buffers + params and dispatches ----

    fn encode_embed(&self, enc: &ComputeCommandEncoderRef, token: u32) {
        let n = self.config().hidden_size;
        let (table, off) = self.weight("model.embed_tokens.weight");
        enc.set_compute_pipeline_state(&self.pipelines.embed);
        enc.set_buffer(0, Some(table), off);
        enc.set_buffer(1, Some(&self.scratch.x), 0);
        set_params(enc, 2, &token);
        set_params(enc, 3, &(n as u32));
        let (groups, threads) = grid_1d(n, ELEMENTWISE_TG);
        enc.dispatch_thread_groups(groups, threads);
    }

    // one threadgroup with one thread per element, so the reduction stays inside it
    fn encode_rmsnorm(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, w: &str, out: &Buffer, n: usize) {
        let (wbuf, woff) = self.weight(w);
        enc.set_compute_pipeline_state(&self.pipelines.rmsnorm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(wbuf), woff);
        enc.set_buffer(2, Some(out), 0);
        set_params(enc, 3, &NormParams { n: n as u32, eps: self.config().rms_norm_eps });
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(n as u64, 1, 1));
    }

    // out_offset lets k/v projections write straight into the KV cache
    fn encode_matvec(&self, enc: &ComputeCommandEncoderRef, w: &str, x: &Buffer, out: &Buffer, out_offset: u64, rows: usize, cols: usize) {
        // the kernel reads 8 columns at a time and has no cleanup loop for leftovers
        assert_eq!(cols % 8, 0, "{w}: cols {cols} must be a multiple of 8");
        let pso = &self.pipelines.matvec_bf16;
        let (wbuf, woff) = self.weight(w);
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(wbuf), woff);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(out), out_offset);
        set_params(enc, 3, &MatvecParams { rows: rows as u32, cols: cols as u32 });
        enc.dispatch_thread_groups(
            MTLSize::new(rows.div_ceil(MATVEC_SIMDGROUPS_PER_TG * self.rows_per_sg) as u64, 1, 1),
            MTLSize::new(MATVEC_TG as u64, 1, 1),
        );
    }

    // one threadgroup per head, head_dim threads each
    fn encode_qk_norm_rope(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, x_offset: u64, norm_w: &str, n_heads: usize, pos: usize) {
        let c = self.config();
        let (wbuf, woff) = self.weight(norm_w);
        enc.set_compute_pipeline_state(&self.pipelines.qk_norm_rope);
        enc.set_buffer(0, Some(x), x_offset);
        enc.set_buffer(1, Some(wbuf), woff);
        set_params(enc, 2, &RopeParams {
            head_dim: c.head_dim as u32,
            pos: pos as u32,
            theta: c.rope_theta,
            eps: c.rms_norm_eps,
        });
        enc.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(c.head_dim as u64, 1, 1));
    }

    // one threadgroup per q head, head_dim threads each
    fn encode_attention(&self, enc: &ComputeCommandEncoderRef, k_cache: &Buffer, v_cache: &Buffer, pos: usize) {
        let c = self.config();
        enc.set_compute_pipeline_state(&self.pipelines.attention);
        enc.set_buffer(0, Some(&self.scratch.q), 0);
        enc.set_buffer(1, Some(k_cache), 0);
        enc.set_buffer(2, Some(v_cache), 0);
        enc.set_buffer(3, Some(&self.scratch.att), 0);
        set_params(enc, 4, &AttnParams {
            n_heads: c.num_attention_heads as u32,
            n_kv_heads: c.num_key_value_heads as u32,
            head_dim: c.head_dim as u32,
            pos: pos as u32,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
        });
        enc.dispatch_thread_groups(MTLSize::new(c.num_attention_heads as u64, 1, 1), MTLSize::new(c.head_dim as u64, 1, 1));
    }

    fn encode_silu_mul(&self, enc: &ComputeCommandEncoderRef) {
        let n = self.config().intermediate_size;
        enc.set_compute_pipeline_state(&self.pipelines.silu_mul);
        enc.set_buffer(0, Some(&self.scratch.gate), 0);
        enc.set_buffer(1, Some(&self.scratch.up), 0);
        set_params(enc, 2, &(n as u32));
        let (groups, threads) = grid_1d(n, ELEMENTWISE_TG);
        enc.dispatch_thread_groups(groups, threads);
    }

    fn encode_add(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, y: &Buffer, n: usize) {
        enc.set_compute_pipeline_state(&self.pipelines.add_inplace);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        set_params(enc, 2, &(n as u32));
        let (groups, threads) = grid_1d(n, ELEMENTWISE_TG);
        enc.dispatch_thread_groups(groups, threads);
    }
}

// set_bytes for a #[repr(C)] param struct (or a plain u32)
fn set_params<T: Copy>(enc: &ComputeCommandEncoderRef, index: u64, p: &T) {
    enc.set_bytes(index, std::mem::size_of::<T>() as u64, p as *const T as *const _);
}

// for elementwise kernels: n threads in groups of `tg`, last group may be partial
// (kernels must bounds-check gid < n)
fn grid_1d(n: usize, tg: usize) -> (MTLSize, MTLSize) {
    (MTLSize::new(n.div_ceil(tg) as u64, 1, 1), MTLSize::new(tg as u64, 1, 1))
}
