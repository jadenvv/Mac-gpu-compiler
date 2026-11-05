mod cpu_inference;
use std::io::{self, Write};
use std::time::Instant;
use cpu_inference::{Model, KVCache};

fn main() -> io::Result<()>{
    let model = Model::load_model("./qwen3-0.6b/model.safetensors")?;
    let mut cache = KVCache::new(&model.config);

    // "The capital of France is" (no tokenizer yet, ids taken from vocab.json)
    let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let max_new = 16;

    let start = Instant::now();
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate()
    {
        logits = model.forward(tok, pos, &mut cache);
    }

    let mut generated = Vec::new();
    for pos in prompt.len()..prompt.len() + max_new
    {
        let next = argmax(&logits);
        if next == model.config.eos_token_id { break; }
        generated.push(next);
        print!("{next} ");
        io::stdout().flush()?;
        logits = model.forward(next, pos, &mut cache);
    }
    println!();
    let n = prompt.len() + generated.len();
    println!("{n} tokens in {:.2?} ({:.2} tok/s)", start.elapsed(), n as f64 / start.elapsed().as_secs_f64());

    Ok(())
}

fn argmax(x: &[f32]) -> u32
{
    let mut best = 0;
    for i in 1..x.len() { if x[i] > x[best] { best = i; } }
    best as u32
}
