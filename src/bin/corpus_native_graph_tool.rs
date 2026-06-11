//! Structure decoder — geometric tool-calling inference.
//!
//! Loads a corpus-native Graph4D (built with tool subgraphs) and runs a walker.
//! When the walker enters a tool anchor region, it emits a structured JSON tool
//! call instead of continuing to generate free text.
//!
//! Usage:
//!   cargo run --release --example corpus_native_graph_tool -- [graph_dir] "prompt text"

use anyhow::{Context, Result};
use geographdb_core::corpus::{
    build_edge_weights, build_node_index, build_octree, decode_node_id,
    load_persisted_tool_schemas, nearest_tool_anchor, nearest_tool_anchor_for_prompt,
    prompt_centroid, GeometricWalker, TransitionMode, WalkerConfig,
};
use geographdb_core::{load_graph4d, GraphNode4D};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const DEFAULT_STEPS: usize = 40;
const TOOL_REGION_RADIUS: f32 = 1.5;
const TOOL_REGION_HITS: usize = 1;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tokenize_prompt(prompt: &str) -> Vec<String> {
    prompt
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn find_start_node(
    graph: &[GraphNode4D],
    vocab: &HashMap<u64, String>,
    prompt: &str,
) -> Option<usize> {
    let words = tokenize_prompt(prompt);
    for word in words {
        let token_id = vocab.iter().find(|(_, v)| *v == &word).map(|(k, _)| *k)?;
        if let Some(idx) = graph.iter().position(|n| decode_node_id(n.id) == token_id) {
            return Some(idx);
        }
    }
    None
}

fn selected_tool_anchor_position(
    graph: &[GraphNode4D],
    position: glam::Vec3,
    radius: f32,
    prompt_tokens: &[u32],
    tokenizer: &Tokenizer,
) -> Option<glam::Vec3> {
    nearest_tool_anchor_for_prompt(graph, position, radius, prompt_tokens, tokenizer)
        .map(|(idx, _)| graph[idx].position())
        .or_else(|| {
            nearest_tool_anchor(graph, position, radius).map(|(idx, _)| graph[idx].position())
        })
}

fn build_args_for_tool(
    graph: &[GraphNode4D],
    anchor_idx: usize,
) -> serde_json::Map<String, serde_json::Value> {
    let mut args = serde_json::Map::new();
    for edge in &graph[anchor_idx].successors {
        let arg_idx = graph.iter().position(|n| n.id == edge.dst);
        let Some(idx) = arg_idx else { continue };
        if let Some(name) = graph[idx].properties.get("arg").and_then(|v| v.as_str()) {
            args.insert(name.to_string(), serde_json::Value::Null);
        }
    }
    args
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
    let prompt = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("convert 100 usd to eur");

    println!("Structure Decoder (Tool Calling)");
    println!("================================\n");
    println!("prompt: {prompt}\n");

    // 1. Load graph
    println!("[1/4] Loading graph from {graph_dir}...");
    let graph = load_graph4d(Path::new(graph_dir))
        .with_context(|| format!("Failed to load graph from {graph_dir}"))?;
    println!("  Nodes: {}", graph.len());

    let node_index = build_node_index(&graph);

    // 2. Load tokenizer, vocab, and tool schemas
    println!("[2/4] Loading tokenizer, vocab and tool schemas...");
    let tokenizer = Tokenizer::from_file(Path::new(graph_dir).join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

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
        println!("  Warning: vocab.json not found");
        HashMap::new()
    };

    let schemas = load_persisted_tool_schemas(&graph);
    println!(
        "  Vocab entries: {}  Tool schemas: {}",
        vocab.len(),
        schemas.len()
    );
    for schema in &schemas {
        println!("    - {}", schema.name);
    }

    // 3. Build octree
    println!("[3/4] Building octree...");
    let octree = build_octree(&graph);

    // 4. Walk and decode
    println!("[4/4] Decoding...\n");
    let edge_weights = build_edge_weights(&graph);

    // Encode the prompt as a centroid in graph space: this is the cognitive state
    // that routes the walker toward relevant semantic regions (text/code/math/tool).
    let prompt_centroid =
        prompt_centroid(&graph, &tokenizer, prompt).unwrap_or_else(|| graph[0].position());
    println!(
        "  Prompt centroid: ({:.3}, {:.3}, {:.3})",
        prompt_centroid.x, prompt_centroid.y, prompt_centroid.z
    );

    let prompt_encoding = tokenizer
        .encode(prompt.to_string(), false)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {}", e))?;
    let prompt_tokens: Vec<u32> = prompt_encoding.get_ids().to_vec();

    let start_idx = find_start_node(&graph, &vocab, prompt).unwrap_or(0);

    // Steer toward the most relevant tool anchor near the prompt centroid.
    // Relevance is measured by token overlap between the prompt and the tool name.
    let start_goal =
        selected_tool_anchor_position(&graph, prompt_centroid, 5.0, &prompt_tokens, &tokenizer);
    if let Some(pos) = start_goal {
        println!(
            "  Steering toward selected tool anchor at ({:.3}, {:.3}, {:.3})",
            pos.x, pos.y, pos.z
        );
    }
    let config = WalkerConfig {
        knn: 20,
        temperature: 0.05,
        momentum: 0.7,
        step_size: 0.3,
        repetition_penalty: 0.3,
        recent_window: 8,
        plan_interval: 8,
        context_weight: 0.0,
        goal_position: start_goal,
        goal_weight: 1.5,
        transition_mode: TransitionMode::DistanceKnn,
    };
    let mut walker = GeometricWalker::new(&graph[start_idx], config);
    walker.set_position(prompt_centroid);

    let mut generated: Vec<String> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut tool_hits = 0usize;
    let mut emitted_tool: Option<serde_json::Value> = None;

    for step in 0..DEFAULT_STEPS {
        let node_id = walker.current_node();
        let token_id = decode_node_id(node_id);
        let word = vocab
            .get(&token_id)
            .cloned()
            .unwrap_or_else(|| format!("<{}>", token_id));

        if !seen.contains(&token_id) || step < 3 {
            generated.push(word);
        }
        seen.insert(token_id);

        // Structure decoder: detect tool anchor proximity.
        if emitted_tool.is_none() {
            if let Some((anchor_idx, schema)) =
                nearest_tool_anchor(&graph, walker.position(), TOOL_REGION_RADIUS)
            {
                tool_hits += 1;
                if tool_hits >= TOOL_REGION_HITS {
                    let arguments = build_args_for_tool(&graph, anchor_idx);
                    emitted_tool = Some(json!({
                        "name": schema.name,
                        "arguments": arguments,
                    }));
                    break;
                }
            } else {
                tool_hits = 0;
            }
        }

        walker.step(&graph, &node_index, &octree, &edge_weights, None);
    }

    if let Some(tool_call) = emitted_tool {
        println!("Tool call detected:");
        println!("{}", serde_json::to_string_pretty(&tool_call)?);
    } else {
        println!("Generated text:");
        println!("{}", generated.join(" "));
    }

    println!("\nWalker stats:");
    println!(
        "  Final position: ({:.3}, {:.3}, {:.3})",
        walker.position().x,
        walker.position().y,
        walker.position().z
    );
    println!("  Cumulative score: {:.3}", walker.cum_score());

    Ok(())
}
