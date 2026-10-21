use memmap2::Mmap;
use std::io::{self, Read};
use serde_json::Value; 
use serde::Deserialize; 
use std::collections::HashMap; 
use std::fs::File; 

#[derive[Deserialize, Debug]]
struct tensor{
    shape: Vec<i32>,
    offset_start: i64,
    offset_end: i64
}
pub fn read_json() -> io::Result<()>
{
    let file = File::open("./qwen3-0.6b/model.safetensors")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let header_bytes: [u8;8] =  mmap[0..8].try_into().unwrap();
    let header_length: u64= u64::from_le_bytes(header_bytes);
    let json: &str =  str::from_utf8(&mmap[8..((header_length+8) as usize)]).unwrap();
    println!("json {:?}", json.len());



    let parsed: Value = serde_json::from_str(&json)?;
    let  obj = parsed.as_object().unwrap(); 
    let key_list: Vec<&String> = obj.keys().collect(); 
    println!("{:?}", key_list); 

        
        
    Ok(())
} 


