use memmap2::Mmap;
use std::io::{self, Read};
use serde_json::Value; 
use serde::Deserialize; 
use std::collections::HashMap; 
use std::fs::File; 


#[derive(Deserialize, Debug)]
pub struct Tensor{
    #[serde(default)]
    name: String,
    dtype: String, 
    shape: Vec<i32>,
    data_offsets: Vec<i32>
}
pub struct Model{

    pub mmap: Mmap, 
    pub tensors: Vec<Tensor>,
    raw_weights: * const u8,
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
    
    return Ok(Model { mmap, tensors, raw_weights })
    } 
} 


