mod cpu_inference;
use std::io;
use cpu_inference::{Model};

fn main() -> io::Result<()>{
     let res = Model::load_model("./qwen3-0.6b/model.safetensors");
    
    println!("{:?}", res?.tensors);

    Ok(())
}
