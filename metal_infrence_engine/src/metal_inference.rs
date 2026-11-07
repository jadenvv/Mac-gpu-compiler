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
// Fill in the todo!()s top to bottom; kernels live in shaders/inference.metal.

// scaffold: remove once everything is wired up
#![allow(dead_code, unused_variables)]

use crate::cpu_inference::{Config, Model};
use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
use std::collections::HashMap;
use std::io;

const SHADER_SRC: &str = include_str!("shaders/inference.metal");

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
        // TODO: for each layer allocate k and v with new_f32_buffer(&model.device, max_seq * kv_dim)
        todo!("MetalKVCache::new")
    }
}

impl MetalModel {
    pub fn load_model(file: &str) -> io::Result<Self> {
        // TODO:
        // 1. let cpu = Model::load_model(file)?;
        // 2. let device = Device::system_default() (panic/err if None), queue = device.new_command_queue()
        // 3. compile_pipelines(&device)
        // 4. wrap_weights(&device, &cpu) -> (weights, offsets)
        // 5. Scratch::new(&device, &cpu.config)
        todo!("MetalModel::load_model")
    }

    pub fn config(&self) -> &Config { &self.cpu.config }

    // (buffer, offset) to bind tensor `name`; asserts it is BF16 like Model::weight
    fn weight(&self, name: &str) -> (&Buffer, u64) {
        // TODO: look up offsets[name]; panic with the name if missing
        todo!("MetalModel::weight")
    }
}

// compile SHADER_SRC once and build a pipeline for every kernel
fn compile_pipelines(device: &Device) -> Pipelines {
    // TODO:
    //  let lib = device.new_library_with_source(SHADER_SRC, &metal::CompileOptions::new())
    //      .unwrap_or_else(|e| panic!("shader compile failed:\n{e}"));
    //  let make = |name| device.new_compute_pipeline_state_with_function(
    //      &lib.get_function(name, None).unwrap()).unwrap();
    todo!("compile_pipelines")
}

// wrap the mmap as one MTLBuffer without copying, and compute per-tensor offsets
fn wrap_weights(device: &Device, cpu: &Model) -> (Buffer, HashMap<String, u64>) {
    // TODO:
    //  - ptr = cpu.mmap.as_ptr(): mmap is page aligned, which no_copy requires
    //  - the LENGTH must also be a multiple of the page size (16 KiB on M1):
    //    round cpu.mmap.len() up. The mapping covers the rest of the last page.
    //  - new_buffer_with_bytes_no_copy(ptr, len, MTLResourceOptions::StorageModeShared, None)
    //  - header_len = u64 from mmap[0..8] (same as load_model); data starts at 8 + header_len
    //  - offsets[t.name] = 8 + header_len + t.data_offsets[0]
    //  - setBuffer offsets must be 4-byte aligned: assert every offset % 4 == 0
    todo!("wrap_weights")
}

impl Scratch {
    fn new(device: &Device, c: &Config) -> Self {
        // TODO: one new_f32_buffer per field, sized as in the comments on Scratch
        todo!("Scratch::new")
    }
}

fn new_f32_buffer(device: &Device, len: usize) -> Buffer {
    device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared)
}

// read a shared f32 buffer back on the CPU (only valid after the command buffer completed)
fn read_f32(buf: &Buffer, len: usize) -> Vec<f32> {
    // TODO: std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec()
    todo!("read_f32")
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

impl MetalModel {
    // same contract as Model::forward: runs `token` at `pos`, writes its K/V into the
    // cache, returns logits. pos must be < cache.max_seq.
    pub fn forward(&self, token: u32, pos: usize, cache: &mut MetalKVCache) -> Vec<f32> {
        assert!(pos < cache.max_seq, "pos {pos} exceeds KV cache size {}", cache.max_seq);
        // TODO:
        //  let cmd = self.queue.new_command_buffer();
        //  let enc = cmd.new_compute_command_encoder();
        //
        //  embed -> x
        //  for each layer l (mirror the CPU loop exactly):
        //      rmsnorm(x, input_layernorm) -> xn
        //      matvec q_proj -> q
        //      matvec k_proj -> k_cache[l] @ pos * kv_dim * 4
        //      matvec v_proj -> v_cache[l] @ pos * kv_dim * 4
        //      qk_norm_rope(q,  q_norm)  n_heads threadgroups
        //      qk_norm_rope(k slice, k_norm)  n_kv_heads threadgroups
        //      attention(q, k_cache[l], v_cache[l]) -> att
        //      matvec o_proj -> proj ; add_inplace(x, proj)
        //      rmsnorm(x, post_attention_layernorm) -> xn
        //      matvec gate_proj -> gate ; matvec up_proj -> up ; silu_mul(gate, up)
        //      matvec down_proj -> proj ; add_inplace(x, proj)
        //  rmsnorm(x, model.norm) -> xn ; matvec lm_head -> logits
        //
        //  enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
        //  read_f32(&self.scratch.logits, vocab)
        //
        // Within one compute encoder, Metal tracks hazards on these buffers and
        // serializes dependent dispatches for you, so no manual barriers needed.
        todo!("MetalModel::forward")
    }

    // ---- one helper per kernel; each binds buffers + params and dispatches ----

    fn encode_embed(&self, enc: &ComputeCommandEncoderRef, token: u32) {
        todo!("encode_embed")
    }

    fn encode_rmsnorm(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, w: &str, out: &Buffer, n: usize) {
        todo!("encode_rmsnorm")
    }

    // out_offset lets k/v projections write straight into the KV cache
    fn encode_matvec(&self, enc: &ComputeCommandEncoderRef, w: &str, x: &Buffer, out: &Buffer, out_offset: u64, rows: usize, cols: usize) {
        todo!("encode_matvec")
    }

    fn encode_qk_norm_rope(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, x_offset: u64, norm_w: &str, n_heads: usize, pos: usize) {
        todo!("encode_qk_norm_rope")
    }

    fn encode_attention(&self, enc: &ComputeCommandEncoderRef, k_cache: &Buffer, v_cache: &Buffer, pos: usize) {
        todo!("encode_attention")
    }

    fn encode_silu_mul(&self, enc: &ComputeCommandEncoderRef) {
        todo!("encode_silu_mul")
    }

    fn encode_add(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, y: &Buffer, n: usize) {
        todo!("encode_add")
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
