//! Rust-native HuggingFace dataset loader example.
//!
//! Downloads HF dataset shards in Rust and feeds them into the existing
//! corpus-native graph builder pipeline.
//!
//! Usage:
//!   cargo run --release --example corpus_hf_loader -- \
//!     --dataset code=bigcode/the-stack-smol:data/python \
//!     --vocab-size 5000 \
//!     --output /tmp/corpus_code_5k
//!
//! For a public JSON code dataset:
//!   cargo run --release --example corpus_hf_loader -- \
//!     --dataset code=nickrosh/Evol-Instruct-Code-80k-v1|instruction,output \
//!     --output /tmp/corpus_code_json

use anyhow::Result;
use geographdb_core::corpus::{HfDatasetLoader, HfDatasetSpec};
use std::env;

fn parse_dataset_args(args: &[String]) -> Vec<HfDatasetSpec> {
    let mut specs = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--dataset" && i + 1 < args.len() {
            match HfDatasetSpec::from_arg(&args[i + 1]) {
                Ok(spec) => specs.push(spec),
                Err(e) => eprintln!("Ignoring invalid dataset arg: {e}"),
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    specs
}

fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_output_dir(args: &[String]) -> String {
    args.windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "corpus_hf_loader".to_string())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let specs = parse_dataset_args(&args);
    if specs.is_empty() {
        eprintln!("Usage: corpus_hf_loader --dataset domain=repo[:subset][|col1,col2] [--dataset ...] --output dir [--vocab-size N]");
        std::process::exit(1);
    }

    let vocab_size: usize = parse_flag(&args, "--vocab-size", 5000);
    let output_dir = parse_output_dir(&args);

    println!("Rust-native HF Corpus Loader");
    println!("============================\n");

    for spec in &specs {
        println!("[{}] {} {:?}", spec.domain, spec.repo_id, spec.subset);
        let loader = HfDatasetLoader::new(spec.clone())?;
        let shards = loader.list_shards()?;
        println!("  shards: {}", shards.len());

        let mut total_texts = 0usize;
        let mut total_chars = 0usize;
        for path in loader.download_shards()? {
            let texts = loader.read_shard_texts(&path)?;
            total_texts += texts.len();
            total_chars += texts.iter().map(|s| s.len()).sum::<usize>();
        }
        println!("  texts: {total_texts}  chars: {total_chars}");
    }

    println!("\nOutput dir: {output_dir}");
    println!("Vocab size: {vocab_size}");
    println!("(Graph build integration coming in next phase.)");

    Ok(())
}
