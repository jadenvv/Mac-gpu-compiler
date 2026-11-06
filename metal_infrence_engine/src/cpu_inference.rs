use memmap2::Mmap;
use std::io;
use serde_json::Value;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::thread;


#[derive(Deserialize, Debug)]
pub struct Tensor{
    #[serde(default)]
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i32>,
    pub data_offsets: Vec<i32>
}

// subset of config.json needed for the forward pass
#[derive(Deserialize, Debug)]
pub struct Config{
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub eos_token_id: u32,
}

pub struct Model{

    pub mmap: Mmap,
    pub tensors: Vec<Tensor>,
    pub config: Config,
    index: HashMap<String, usize>,
    raw_weights: * const u8,
}

// per-layer K/V history, each entry is [kv_heads * head_dim] per position
pub struct KVCache{
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KVCache{
    pub fn new(config: &Config) -> Self
    {
        KVCache { k: vec![Vec::new(); config.num_hidden_layers], v: vec![Vec::new(); config.num_hidden_layers] }
    }
}

impl Model{
    pub fn load_model(file: &str) -> io::Result<Self>
    {
    let f = File::open(file)?;
    let mmap = unsafe { Mmap::map(&f)? };
    let header_bytes: [u8;8] =  mmap[0..8].try_into().unwrap();
    let header_length: u64= u64::from_le_bytes(header_bytes);
    let json: &str =  str::from_utf8(&mmap[8..((header_length+8) as usize)]).unwrap();
    let parsed: HashMap<String, Value>= serde_json::from_str(&json)?;
    let mut tensors: Vec<Tensor> = Vec::new();
    for (key,value) in parsed
    {
        if key == "__metadata__" {continue};
        let mut tensor: Tensor = serde_json::from_value(value)?;
        tensor.name = key;
        tensors.push(tensor);
    }
    let raw_weights = unsafe {mmap.as_ptr().add((8+ header_length) as usize)};
    // weights are read in place as u16, so the data section must be 2-byte aligned
    assert!(raw_weights as usize % 2 == 0, "safetensors data section is not 2-byte aligned");

    let index = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();

    let config_path = Path::new(file).with_file_name("config.json");
    let config: Config = serde_json::from_reader(File::open(config_path)?)?;

    return Ok(Model { mmap, tensors, config, index, raw_weights })
    }

    // bf16 weights viewed straight out of the mmap
    fn weight(&self, name: &str) -> &[u16]
    {
        let t = &self.tensors[*self.index.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))];
        assert_eq!(t.dtype, "BF16", "{name}: only BF16 is supported");
        let start = t.data_offsets[0] as usize;
        let end = t.data_offsets[1] as usize;
        let numel: usize = t.shape.iter().map(|&d| d as usize).product();
        assert_eq!((end - start) / 2, numel);
        unsafe { std::slice::from_raw_parts(self.raw_weights.add(start) as *const u16, numel) }
    }

    // runs one token at position `pos` through the model, appending its K/V to the cache.
    // returns logits over the vocab
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KVCache) -> Vec<f32>
    {
        let c = &self.config;
        let hd = c.head_dim;
        let q_dim = c.num_attention_heads * hd;
        let kv_dim = c.num_key_value_heads * hd;
        let group = c.num_attention_heads / c.num_key_value_heads;
        let eps = c.rms_norm_eps;

        let embed = self.weight("model.embed_tokens.weight");
        let row = token as usize * c.hidden_size;
        let mut x: Vec<f32> = embed[row..row + c.hidden_size].iter().map(|&w| bf16_to_f32(w)).collect();

        let mut xn = vec![0.0; c.hidden_size];
        let mut q = vec![0.0; q_dim];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        let mut att = vec![0.0; q_dim];
        let mut proj = vec![0.0; c.hidden_size];
        let mut gate = vec![0.0; c.intermediate_size];
        let mut up = vec![0.0; c.intermediate_size];
        let mut scores = vec![0.0; pos + 1];

        for l in 0..c.num_hidden_layers
        {
            let p = format!("model.layers.{l}.");
            let w = |n: &str| self.weight(&format!("{p}{n}"));

            // attention
            rmsnorm(&mut xn, &x, w("input_layernorm.weight"), eps);
            matvec(&mut q, w("self_attn.q_proj.weight"), &xn);
            matvec(&mut k, w("self_attn.k_proj.weight"), &xn);
            matvec(&mut v, w("self_attn.v_proj.weight"), &xn);

            // qwen3 normalizes each q/k head before rope
            let q_norm = w("self_attn.q_norm.weight");
            let k_norm = w("self_attn.k_norm.weight");
            for h in q.chunks_exact_mut(hd) { rmsnorm_inplace(h, q_norm, eps); rope(h, pos, c.rope_theta); }
            for h in k.chunks_exact_mut(hd) { rmsnorm_inplace(h, k_norm, eps); rope(h, pos, c.rope_theta); }

            cache.k[l].extend_from_slice(&k);
            cache.v[l].extend_from_slice(&v);
            let (kc, vc) = (&cache.k[l], &cache.v[l]);

            let scale = 1.0 / (hd as f32).sqrt();
            for h in 0..c.num_attention_heads
            {
                let qh = &q[h * hd..(h + 1) * hd];
                let kv_off = (h / group) * hd;
                for t in 0..=pos
                {
                    let kt = &kc[t * kv_dim + kv_off..t * kv_dim + kv_off + hd];
                    scores[t] = dot(qh, kt) * scale;
                }
                softmax(&mut scores);
                let out = &mut att[h * hd..(h + 1) * hd];
                out.fill(0.0);
                for t in 0..=pos
                {
                    let vt = &vc[t * kv_dim + kv_off..t * kv_dim + kv_off + hd];
                    for i in 0..hd { out[i] += scores[t] * vt[i]; }
                }
            }
            matvec(&mut proj, w("self_attn.o_proj.weight"), &att);
            for i in 0..x.len() { x[i] += proj[i]; }

            // swiglu mlp
            rmsnorm(&mut xn, &x, w("post_attention_layernorm.weight"), eps);
            matvec(&mut gate, w("mlp.gate_proj.weight"), &xn);
            matvec(&mut up, w("mlp.up_proj.weight"), &xn);
            for i in 0..gate.len() { gate[i] = silu(gate[i]) * up[i]; }
            matvec(&mut proj, w("mlp.down_proj.weight"), &gate);
            for i in 0..x.len() { x[i] += proj[i]; }
        }

        rmsnorm(&mut xn, &x, self.weight("model.norm.weight"), eps);
        let mut logits = vec![0.0; c.vocab_size];
        matvec(&mut logits, self.weight("lm_head.weight"), &xn);
        logits
    }
}

#[inline(always)]
fn bf16_to_f32(x: u16) -> f32 { f32::from_bits((x as u32) << 16) }

fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn rmsnorm(out: &mut [f32], x: &[f32], w: &[u16], eps: f32)
{
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() { out[i] = x[i] * s * bf16_to_f32(w[i]); }
}

fn rmsnorm_inplace(x: &mut [f32], w: &[u16], eps: f32)
{
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() { x[i] = x[i] * s * bf16_to_f32(w[i]); }
}

// HF "rotate_half" layout: element i is paired with i + head_dim/2
fn rope(x: &mut [f32], pos: usize, theta: f32)
{
    let half = x.len() / 2;
    for i in 0..half
    {
        let freq = theta.powf(-2.0 * i as f32 / x.len() as f32);
        let (sin, cos) = (pos as f32 * freq).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos - b * sin;
        x[i + half] = b * cos + a * sin;
    }
}

fn softmax(x: &mut [f32])
{
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in x.iter_mut() { *v /= sum; }
}

// out[r] = sum_c w[r, c] * x[c], w is row-major bf16 [out.len(), x.len()].
// rows are split across threads
fn matvec(out: &mut [f32], w: &[u16], x: &[f32])
{
    let cols = x.len();
    assert_eq!(w.len(), out.len() * cols);
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let rows_per = out.len().div_ceil(threads);
    thread::scope(|s| {
        for (oc, wc) in out.chunks_mut(rows_per).zip(w.chunks(rows_per * cols))
        {
            s.spawn(move || {
                for (o, row) in oc.iter_mut().zip(wc.chunks_exact(cols)) { *o = dot_bf16(row, x); }
            });
        }
    });
}

fn dot_bf16(w: &[u16], x: &[f32]) -> f32
{
    // independent accumulators so the compiler can vectorize the reduction
    let mut acc = [0.0f32; 8];
    let wc = w.chunks_exact(8);
    let xc = x.chunks_exact(8);
    let (wr, xr) = (wc.remainder(), xc.remainder());
    for (a, b) in wc.zip(xc)
    {
        for i in 0..8 { acc[i] += bf16_to_f32(a[i]) * b[i]; }
    }
    let mut s: f32 = acc.iter().sum();
    for (a, b) in wr.iter().zip(xr) { s += bf16_to_f32(*a) * b; }
    s
}
