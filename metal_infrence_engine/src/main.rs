mod cpu_inference;
use std::io;
use cpu_inference::{read_json};

fn main() -> io::Result<()>{
    let res = read_json();
    return res;
}
