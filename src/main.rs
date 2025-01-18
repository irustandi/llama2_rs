use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Path to checkpoint
    #[arg(short, long)]
    checkpoint: String,

    // temperature, default 1.0
    #[arg(short, long, default_value_t=1.0)]
    temperature: f32,    

    // p value in top-p (nuclues) sample, default 0.9
    #[arg(short='p', long, default_value_t=0.9)]
    topp: f32,

    // seed rnd with time by default
    #[arg(short='s', long, default_value_t=0)]
    rng_seed: i64,

    // number of steps to run for, default 256
    #[arg(short='n', long, default_value_t=256)]
    steps: i32,

    // input prompt
    #[arg(short='i', long, value_name="PROMPT")]
    prompt: Option<String>,

    // optional path to custom tokenizer
    #[arg(short='z', long, value_name="TOKENIZER")]
    tokenizer: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    println!("Hello, world!");
}
