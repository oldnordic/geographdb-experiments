//! E2: Train a CoordinateBranch on real Memoria activation coordinates.
//!
//! Reads token positions produced by Memoria's multilayer PCA pipeline
//! (`node_vector_fields.json`) and trains the pure-Rust CoordinateBranch MLP
//! to predict those 3D positions from token IDs.
//!
//! Example:
//!   cargo run --release --bin e2_memoria_coordinate_branch -- \
//!     --activations /home/feanor/Projects/Memoria/data_phase_token/layer23/n2349_spatial/node_vector_fields.json \
//!     --output /tmp/e2_memoria_coordinate_branch

use anyhow::{Context, Result};
use geographdb_core::corpus::CoordinateBranch;
use geographdb_core::{GraphNode4D, GraphProperties};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
struct NodeVectorFields {
    #[serde(rename = "node_fields")]
    node_fields: HashMap<String, MemoriaNode>,
}

#[derive(Debug, serde::Deserialize)]
struct MemoriaNode {
    pos: Vec<f32>,
    #[serde(rename = "dom_tid")]
    dom_tid: u64,
}

fn parse_arg(args: &[String], flag: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn parse_arg_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_arg_f32(args: &[String], flag: &str, default: f32) -> f32 {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<f32>().ok())
        .unwrap_or(default)
}

fn parse_arg_opt_usize(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<usize>().ok())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let activations_path = parse_arg(
        &args,
        "--activations",
        "/home/feanor/Projects/Memoria/data_phase_token/layer23/n2349_spatial/node_vector_fields.json",
    );
    let output_dir = parse_arg(&args, "--output", "/tmp/e2_memoria_coordinate_branch");
    let epochs = parse_arg_usize(&args, "--epochs", 200);
    let lr = parse_arg_f32(&args, "--lr", 0.05);
    let embed_dim = parse_arg_usize(&args, "--embed-dim", 64);
    let hidden_dim = parse_arg_usize(&args, "--hidden-dim", 128);
    let seed = parse_arg_usize(&args, "--seed", 42) as u32;
    let limit = parse_arg_opt_usize(&args, "--limit");

    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create output directory {output_dir}"))?;

    println!("E2 — CoordinateBranch on Memoria activation coordinates");
    println!("=======================================================");
    println!("  activations: {activations_path}");
    println!("  output:      {output_dir}");
    println!("  epochs:      {epochs}");
    println!("  lr:          {lr}");
    println!("  embed_dim:   {embed_dim}");
    println!("  hidden_dim:  {hidden_dim}");

    // 1. Load Memoria node fields.
    let file = std::fs::File::open(&activations_path)
        .with_context(|| format!("failed to open {activations_path}"))?;
    let fields: NodeVectorFields = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {activations_path}"))?;
    println!("\n[1/4] Loaded {} Memoria nodes", fields.node_fields.len());

    // 2. Build GraphNode4D training targets.
    //    node.id encodes the token ID using the same convention as
    //    corpus_native_graph.rs: id = token_id * 1000 + sense_index.
    //    These Memoria nodes are single-sense, so sense_index = 0.
    let mut nodes = Vec::with_capacity(fields.node_fields.len());
    let mut max_token_id: u64 = 0;
    for (_, node) in fields.node_fields {
        if node.pos.len() != 3 {
            continue;
        }
        let tid = node.dom_tid;
        max_token_id = max_token_id.max(tid);
        nodes.push(GraphNode4D {
            id: tid * 1000,
            x: node.pos[0],
            y: node.pos[1],
            z: node.pos[2],
            begin_ts: 0,
            end_ts: u64::MAX,
            properties: GraphProperties::new(),
            successors: Vec::new(),
        });
    }
    let vocab_size = (max_token_id + 1) as usize;
    if let Some(n) = limit {
        nodes.truncate(n);
        println!(
            "[2/4] Training targets: {} nodes (limited), vocab_size={}",
            nodes.len(),
            vocab_size
        );
    } else {
        println!(
            "[2/4] Training targets: {} nodes, vocab_size={}",
            nodes.len(),
            vocab_size
        );
    }

    // 3. Train CoordinateBranch.
    println!("[3/4] Training CoordinateBranch...");
    let mut branch = CoordinateBranch::new(vocab_size, embed_dim, hidden_dim, seed);
    let initial_loss = branch.train_epoch(&nodes, lr);
    println!("  epoch 0   loss={initial_loss:.6}");

    let mut last_loss = initial_loss;
    for e in 1..=epochs {
        last_loss = branch.train_epoch(&nodes, lr);
        if e % 10 == 0 || e == epochs {
            println!("  epoch {e:<3} loss={last_loss:.6}");
        }
    }

    // 4. Evaluate and save.
    println!("[4/4] Evaluating and saving...");
    let mut total_l2 = 0.0f32;
    let mut max_l2 = 0.0f32;
    let mut predictions = Vec::new();
    for node in &nodes {
        let tid = node.id / 1000;
        let pred = branch.predict(tid);
        let err = pred.distance(node.position());
        total_l2 += err;
        max_l2 = max_l2.max(err);
        predictions.push(serde_json::json!({
            "token_id": tid,
            "target": [node.x, node.y, node.z],
            "predicted": [pred.x, pred.y, pred.z],
            "l2_error": err,
        }));
    }
    let mean_l2 = total_l2 / nodes.len() as f32;

    let model_path = Path::new(&output_dir).join("coordinate_branch.json");
    branch.save(&model_path)?;

    let preds_path = Path::new(&output_dir).join("predictions.jsonl");
    let mut preds_file = std::fs::File::create(&preds_path)?;
    for p in predictions {
        serde_json::to_writer(&mut preds_file, &p)?;
        preds_file.write_all(b"\n")?;
    }

    let summary = serde_json::json!({
        "activations_path": activations_path,
        "nodes": nodes.len(),
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "initial_loss": initial_loss,
        "final_loss": last_loss,
        "mean_l2_error": mean_l2,
        "max_l2_error": max_l2,
    });
    let summary_path = Path::new(&output_dir).join("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    println!("\nResults");
    println!("  initial loss: {initial_loss:.6}");
    println!("  final loss:   {last_loss:.6}");
    println!("  mean L2:      {mean_l2:.6}");
    println!("  max L2:       {max_l2:.6}");
    println!("  model:        {}", model_path.display());
    println!("  predictions:  {}", preds_path.display());
    println!("  summary:      {}", summary_path.display());

    Ok(())
}
