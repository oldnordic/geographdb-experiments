//! Geometric graph walker — inference-time token generation via spatial traversal.
//!
//! Loads a corpus-native Graph4D built by `corpus_native_graph` and walks it
//! using the reusable `GeometricWalker` from `geographdb_core::corpus`. No
//! matrix multiplies, no attention heads — the graph topology IS the model.
//!
//! Usage:
//!   cargo run --release --example corpus_native_graph_walk -- [graph_dir] [seed] [steps] [temp] [plan_steps]

use anyhow::{Context, Result};
use geographdb_core::corpus::{
    build_edge_weights, build_node_index, build_octree, decode_node_id, GeometricWalker,
    TransitionMode, WalkerConfig,
};
use geographdb_core::{load_graph4d, GraphNode4D};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const DEFAULT_STEPS: usize = 50;
const DEFAULT_TEMP: f32 = 0.05;
const DEFAULT_PLAN_STEPS: usize = 8;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_start_node(
    graph: &[GraphNode4D],
    vocab: &HashMap<u64, String>,
    seed_word: &str,
) -> Option<usize> {
    let seed = seed_word.to_lowercase();
    let token_id = vocab.iter().find(|(_, v)| *v == &seed).map(|(k, _)| *k)?;
    graph.iter().position(|n| decode_node_id(n.id) == token_id)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let graph_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("corpus_native_graph");
    let seed_phrase = args.get(2).map(|s| s.as_str()).unwrap_or("the");
    let steps = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_STEPS);
    let temperature = args
        .get(4)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(DEFAULT_TEMP);
    let plan_interval = args
        .get(5)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PLAN_STEPS);

    println!("Geometric Graph Walker");
    println!("======================");
    println!("steps={steps} temperature={temperature} plan_interval={plan_interval}\n");

    // 1. Load graph
    println!("[1/4] Loading graph from {graph_dir}...");
    let graph = load_graph4d(Path::new(graph_dir))
        .with_context(|| format!("Failed to load graph from {graph_dir}"))?;
    println!("  Nodes: {}", graph.len());

    let node_index = build_node_index(&graph);

    // 2. Load vocab
    println!("[2/4] Loading vocab...");
    let vocab_path = Path::new(graph_dir).join("vocab.json");
    let vocab: HashMap<u64, String> = if vocab_path.exists() {
        let text = std::fs::read_to_string(&vocab_path)?;
        let map: HashMap<String, serde_json::Value> = serde_json::from_str(&text)?;
        map.into_iter()
            .filter_map(|(k, v)| {
                let id = k.parse::<u64>().ok()?;
                let word = v.as_str()?.to_string();
                Some((id, word))
            })
            .collect()
    } else {
        println!("  Warning: vocab.json not found; decoding will show numeric IDs");
        HashMap::new()
    };
    println!("  Vocab entries: {}", vocab.len());

    // 3. Build octree
    println!("[3/4] Building octree...");
    let octree = build_octree(&graph);
    println!("  Octree built");

    // 4. Walk
    println!("[4/4] Walking from seed '{seed_phrase}'...");
    let edge_weights = build_edge_weights(&graph);

    let start_idx = find_start_node(&graph, &vocab, seed_phrase).unwrap_or(0);
    let config = WalkerConfig {
        knn: 20,
        temperature,
        momentum: 0.7,
        step_size: 0.3,
        repetition_penalty: 0.3,
        recent_window: 8,
        plan_interval,
        context_weight: 0.0,
        goal_position: None,
        goal_weight: 1.5,
        transition_mode: TransitionMode::DistanceKnn,
    };
    let mut walker = GeometricWalker::new(&graph[start_idx], config);
    let trajectory = walker.walk(&graph, &node_index, &octree, &edge_weights, None, steps);

    let mut generated: Vec<String> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for (step, node_id) in trajectory.iter().enumerate() {
        let token_id = decode_node_id(*node_id);
        let word = vocab
            .get(&token_id)
            .cloned()
            .unwrap_or_else(|| format!("<{}>", token_id));

        if !seen.contains(&token_id) || step < 3 {
            generated.push(word);
        }
        seen.insert(token_id);
    }

    println!("\nGenerated text:");
    println!("{}", generated.join(" "));

    println!("\nWalker stats:");
    println!(
        "  Final position: ({:.3}, {:.3}, {:.3})",
        walker.position().x,
        walker.position().y,
        walker.position().z
    );
    println!("  Cumulative score: {:.3}", walker.cum_score());
    println!("  Unique tokens: {}", seen.len());

    Ok(())
}
