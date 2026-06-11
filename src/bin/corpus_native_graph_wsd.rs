//! Word Sense Disambiguation (WSD) probe.
//!
//! Tests whether the geometric graph captures contextual sense selection.
//!
//! For an input sentence, the probe walks token-by-token and selects the
//! sense-node for each word that is spatially closest to the running context
//! centroid (average position of previously selected senses).
//!
//! At a target ambiguous word, it reports the ranked sense candidates and
//! their distances to context. If the graph encodes meaning geometrically,
//! contrasting contexts should pull the same ambiguous word toward different
//! sense clusters.
//!
//! Usage:
//!   cargo run --release --example corpus_native_graph_wsd -- [graph_dir] [sentence] [target_word]
//!
//! Example:
//!   cargo run --release --example corpus_native_graph_wsd -- graph_dir \"the river bank was steep\" bank
//!   cargo run --release --example corpus_native_graph_wsd -- graph_dir \"the bank account was empty\" bank

use anyhow::{Context, Result};
use geographdb_core::{load_graph4d, GraphNode4D};
use glam::Vec3;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Simple whitespace tokenization (must match corpus_native_graph builder)
// ---------------------------------------------------------------------------
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn decode_node_id(node_id: u64) -> u64 {
    node_id / 1000
}

// ---------------------------------------------------------------------------
// WSD probe
// ---------------------------------------------------------------------------

/// For each token, select the sense-node closest to the running context.
/// Returns the sequence of selected node IDs and the target word's sense ranking.
fn disambiguate_sentence(
    graph: &[GraphNode4D],
    _vocab: &HashMap<u64, String>,
    reverse_vocab: &HashMap<String, u64>,
    sentence: &str,
    target_word: &str,
) -> Vec<(String, u64, Vec3)> {
    let tokens = tokenize(sentence);
    let mut context_centroid = Vec3::ZERO;
    let mut context_count = 0usize;
    let mut results = Vec::new();

    for (idx, word) in tokens.iter().enumerate() {
        let Some(&token_id) = reverse_vocab.get(word) else {
            // Unknown word
            results.push((word.clone(), u64::MAX, Vec3::ZERO));
            continue;
        };

        // Collect all sense-nodes for this token
        let sense_nodes: Vec<&GraphNode4D> = graph
            .iter()
            .filter(|n| decode_node_id(n.id) == token_id)
            .collect();

        if sense_nodes.is_empty() {
            results.push((word.clone(), u64::MAX, Vec3::ZERO));
            continue;
        }

        // Score each sense by distance to context centroid
        let mut scored: Vec<(u64, Vec3, f32)> = sense_nodes
            .iter()
            .map(|n| {
                let dist = if context_count > 0 {
                    n.position().distance(context_centroid)
                } else {
                    0.0 // first word: all senses tied
                };
                (n.id, n.position(), dist)
            })
            .collect();

        // Sort by ascending distance (closest to context wins)
        scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        let (selected_id, selected_pos, _selected_dist) = scored[0];

        // Update context centroid with exponential moving average
        if context_count == 0 {
            context_centroid = selected_pos;
        } else {
            context_centroid = context_centroid * 0.7 + selected_pos * 0.3;
        }
        context_count += 1;

        results.push((word.clone(), selected_id, selected_pos));

        // Print sense ranking for target word
        if word == &target_word.to_lowercase() {
            println!("\n  Target '{}' at position {}:", word, idx);
            println!(
                "    Context centroid: ({:.3}, {:.3}, {:.3})",
                context_centroid.x, context_centroid.y, context_centroid.z
            );
            println!("    Sense candidates (ranked by distance to context):");
            for (rank, (sid, pos, dist)) in scored.iter().enumerate().take(5) {
                let sense_idx = sid % 1000;
                println!(
                    "      #{}  sense={}  pos=({:.3}, {:.3}, {:.3})  dist={:.4}",
                    rank + 1,
                    sense_idx,
                    pos.x,
                    pos.y,
                    pos.z,
                    dist
                );
            }
        }
    }

    results
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

    // If no sentence provided, run a built-in battery
    let sentence = args.get(2).map(|s| s.as_str());
    let target_word = args.get(3).map(|s| s.as_str());

    println!("Word Sense Disambiguation Probe");
    println!("===============================\n");

    // 1. Load graph
    println!("[1/3] Loading graph from {graph_dir}...");
    let graph = load_graph4d(Path::new(graph_dir))
        .with_context(|| format!("Failed to load graph from {graph_dir}"))?;
    println!("  Nodes: {}", graph.len());

    // 2. Load vocab + build reverse mapping
    println!("[2/3] Loading vocab...");
    let vocab_path = Path::new(graph_dir).join("vocab.json");
    let (vocab, reverse_vocab): (HashMap<u64, String>, HashMap<String, u64>) =
        if vocab_path.exists() {
            let text = std::fs::read_to_string(&vocab_path)?;
            let map: HashMap<String, serde_json::Value> = serde_json::from_str(&text)?;
            let fwd: HashMap<u64, String> = map
                .into_iter()
                .filter_map(|(k, v)| {
                    let id = k.parse::<u64>().ok()?;
                    let word = v.as_str()?.to_string();
                    Some((id, word))
                })
                .collect();
            let rev: HashMap<String, u64> = fwd.iter().map(|(k, v)| (v.clone(), *k)).collect();
            (fwd, rev)
        } else {
            println!("  Warning: vocab.json not found");
            (HashMap::new(), HashMap::new())
        };
    println!("  Vocab entries: {}", vocab.len());

    // 3. Run WSD probe
    println!("[3/3] Running WSD probe...\n");

    let test_cases: Vec<(&str, &str)> = if let (Some(s), Some(t)) = (sentence, target_word) {
        vec![(s, t)]
    } else {
        vec![
            ("the river bank was steep and muddy", "bank"),
            ("the bank account was overdrawn", "bank"),
            ("he sat on the bank of the lake", "bank"),
            ("the investment bank collapsed", "bank"),
            ("the cricket match lasted five days", "cricket"),
            ("the cricket chirped all night", "cricket"),
            ("the python script crashed", "python"),
            ("the python swallowed the mouse", "python"),
            ("the star shone brightly", "star"),
            ("the movie star arrived late", "star"),
        ]
    };

    for (sent, target) in &test_cases {
        println!("────────────────────────────────────────────────────────────");
        println!("Sentence: \"{}\"", sent);
        println!("Target:   \"{}\"", target);

        let selections = disambiguate_sentence(&graph, &vocab, &reverse_vocab, sent, target);

        // Show full sense trace
        let trace: Vec<String> = selections
            .iter()
            .map(|(word, node_id, _)| {
                if word == &target.to_lowercase() {
                    let sense = node_id % 1000;
                    format!("{}[s{}]", word, sense)
                } else {
                    word.clone()
                }
            })
            .collect();
        println!("\n  Trace: {}", trace.join(" "));
        println!();
    }

    Ok(())
}
