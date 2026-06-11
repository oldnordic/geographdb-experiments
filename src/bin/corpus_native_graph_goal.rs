//! GOAL-directed geometric graph engine.
//!
//! A non-classical transformer alternative where:
//!   - The graph topology IS the model (no weight matrices)
//!   - The GOAL drives computation (not a loss function)
//!   - Learning is graph rewiring (edge reinforcement), not backprop
//!
//! Architecture:
//!   1. ENCODE: input text → sense-nodes in graph space
//!   2. PROCESS: walker traverses graph toward a GOAL region
//!   3. DECODE: final node(s) → tokens / tool calls
//!   4. LEARN: strengthen edges on successful paths, weaken on failures
//!
//! The GOAL is a spatial region (e.g., "reach the tool-calling subgraph").
//! The walker scores candidates by local coherence AND progress toward goal.
//!
//! Usage:
//!   cargo run --release --example corpus_native_graph_goal -- [graph_dir] [seed] [steps] [goal_word]

use anyhow::{Context, Result};
use geographdb_core::spatial::octree::{BoundingBox, Octree};
use geographdb_core::storage::data_structures::NodePoint;
use geographdb_core::{load_graph4d, save_graph4d, GraphNode4D};
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const K_NEIGHBORS: usize = 20;
const MOMENTUM: f32 = 0.7;
const STEP_SIZE: f32 = 0.3;
const TEMPERATURE: f32 = 0.08;
const GOAL_WEIGHT: f32 = 1.5; // how much goal alignment influences scoring
const REPETITION_PENALTY: f32 = 0.3;
const LEARN_RATE: f32 = 0.05; // edge reinforcement per episode
const DECAY_RATE: f32 = 0.995; // edge decay per episode

// ---------------------------------------------------------------------------
// Goal-directed walker
// ---------------------------------------------------------------------------

struct GoalWalker {
    current_node: u64,
    position: Vec3,
    velocity: Vec3,
    time_step: u64,
    momentum: f32,
    temperature: f32,
    step_size: f32,
    goal_weight: f32,
    goal_position: Vec3,
    recent_tokens: Vec<u64>,
    recent_window: usize,
    trajectory: Vec<u64>, // node IDs visited
}

impl GoalWalker {
    fn new(start_node: &GraphNode4D, goal_pos: Vec3) -> Self {
        Self {
            current_node: start_node.id,
            position: start_node.position(),
            velocity: Vec3::ZERO,
            time_step: 0,
            momentum: MOMENTUM,
            temperature: TEMPERATURE,
            step_size: STEP_SIZE,
            goal_weight: GOAL_WEIGHT,
            goal_position: goal_pos,
            recent_tokens: Vec::new(),
            recent_window: 8,
            trajectory: vec![start_node.id],
        }
    }

    /// Single step: score by local coherence + goal alignment.
    fn step(
        &mut self,
        graph: &[GraphNode4D],
        node_index: &HashMap<u64, usize>,
        octree: &Octree,
        edge_weights: &HashMap<(u64, u64), f32>,
    ) -> u64 {
        let current_idx = node_index[&self.current_node];
        let current_successors: HashSet<u64> = graph[current_idx]
            .successors
            .iter()
            .map(|e| e.dst)
            .collect();

        let recent_set: HashSet<u64> = self
            .recent_tokens
            .iter()
            .rev()
            .take(self.recent_window)
            .copied()
            .collect();

        let knn = octree.query_knn(self.position, K_NEIGHBORS);
        let mut candidate_ids: HashSet<u64> = current_successors.clone();
        for (np, _) in &knn {
            if np.id != self.current_node {
                candidate_ids.insert(np.id);
            }
        }
        let mut candidate_positions: HashMap<u64, Vec3> = HashMap::new();
        for (np, _) in &knn {
            candidate_positions.insert(np.id, Vec3::new(np.x, np.y, np.z));
        }
        for &sid in &current_successors {
            if !candidate_positions.contains_key(&sid) {
                candidate_positions.insert(sid, graph[node_index[&sid]].position());
            }
        }

        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for &cid in &candidate_ids {
            let node_pos = candidate_positions[&cid];
            let dir = node_pos - self.position;
            let dist_sq = dir.length_squared();
            let dist = dist_sq.sqrt().max(1e-6);

            // Local coherence: spatial proximity
            let spatial_score = (-dist_sq / self.temperature).exp();

            // Velocity alignment
            let vel_norm = self.velocity.length().max(1e-6);
            let alignment = if vel_norm > 1e-6 {
                self.velocity.dot(dir) / (vel_norm * dist)
            } else {
                0.0
            };
            let alignment_score = (alignment * 0.5 + 0.5).max(0.0);

            // Sequential bias: graph successors
            let sequential_bonus = if current_successors.contains(&cid) {
                0.5
            } else {
                0.0
            };

            // Edge weight boost
            let edge_bonus = edge_weights
                .get(&(self.current_node, cid))
                .copied()
                .unwrap_or(0.0)
                * 0.05;

            // Repetition penalty
            let token_id = cid / 1000;
            let repetition_penalty = if recent_set.contains(&token_id) {
                REPETITION_PENALTY * recent_set.iter().filter(|&&t| t == token_id).count() as f32
            } else {
                0.0
            };

            // GOAL alignment: distance to goal region
            let goal_dist = node_pos.distance(self.goal_position);
            let goal_score = (-goal_dist / self.temperature).exp();

            let score = spatial_score * (1.0 + alignment_score + sequential_bonus)
                + edge_bonus
                + self.goal_weight * goal_score
                - repetition_penalty;
            candidates.push((cid, score.max(1e-6)));
        }

        if candidates.is_empty() {
            return self.current_node;
        }

        let max_score = candidates.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let exp_scores: Vec<f32> = candidates
            .iter()
            .map(|(_, s)| (s - max_score).exp())
            .collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        let best_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let next_id = candidates[best_idx].0;

        // Momentum update
        let next_node = &graph[node_index[&next_id]];
        let target = next_node.position();
        let dir = target - self.position;
        let dir_norm = dir.length().max(1e-6);
        let dir_unit = dir / dir_norm;
        self.velocity = self.velocity * self.momentum + dir_unit * (1.0 - self.momentum);
        self.position += self.velocity * self.step_size;

        self.current_node = next_id;
        self.time_step += 1;
        self.trajectory.push(next_id);
        self.recent_tokens.push(next_id / 1000);

        next_id
    }
}

// ---------------------------------------------------------------------------
// Graph rewiring (learning without backprop)
// ---------------------------------------------------------------------------

/// Hebbian reinforcement: strengthen edges that were traversed successfully.
fn reinforce_edges(graph: &mut [GraphNode4D], trajectory: &[u64], learn_rate: f32) {
    for window in trajectory.windows(2) {
        let from = window[0];
        let to = window[1];
        // Find the edge and boost its weight
        for node in graph.iter_mut() {
            if node.id == from {
                for edge in node.successors.iter_mut() {
                    if edge.dst == to {
                        edge.weight += learn_rate;
                        break;
                    }
                }
            }
        }
    }
}

/// Decay all edges to forget unused connections.
fn decay_edges(graph: &mut [GraphNode4D], decay: f32) {
    for node in graph.iter_mut() {
        for edge in node.successors.iter_mut() {
            edge.weight *= decay;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_edge_weights(graph: &[GraphNode4D]) -> HashMap<(u64, u64), f32> {
    let mut weights = HashMap::new();
    for node in graph {
        for edge in &node.successors {
            weights.insert((node.id, edge.dst), edge.weight);
        }
    }
    weights
}

fn find_node_by_word(
    graph: &[GraphNode4D],
    vocab: &HashMap<u64, String>,
    word: &str,
) -> Option<usize> {
    let word = word.to_lowercase();
    let token_id = vocab.iter().find(|(_, v)| v == &&word).map(|(k, _)| *k)?;
    graph.iter().position(|n| n.id / 1000 == token_id)
}

fn mean_edge_weight_along_path(
    graph: &[GraphNode4D],
    node_index: &HashMap<u64, usize>,
    trajectory: &[u64],
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for window in trajectory.windows(2) {
        let from = window[0];
        let to = window[1];
        let idx = node_index[&from];
        if let Some(edge) = graph[idx].successors.iter().find(|e| e.dst == to) {
            total += edge.weight;
            count += 1;
        }
    }
    if count > 0 {
        total / count as f32
    } else {
        0.0
    }
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
    let seed_word = args.get(2).map(|s| s.as_str()).unwrap_or("the");
    let steps = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(30);
    let goal_word = args.get(4).map(|s| s.as_str()).unwrap_or("game");
    let episodes = args
        .get(5)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    println!("GOAL-Directed Geometric Graph Engine");
    println!("====================================");
    println!(
        "seed='{}' goal='{}' steps={} episodes={}\n",
        seed_word, goal_word, steps, episodes
    );

    // 1. Load graph
    println!("[1/4] Loading graph...");
    let mut graph = load_graph4d(Path::new(graph_dir))
        .with_context(|| format!("Failed to load graph from {}", graph_dir))?;
    println!("  Nodes: {}", graph.len());

    let node_index: HashMap<u64, usize> =
        graph.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

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
        println!("  Warning: vocab.json not found");
        HashMap::new()
    };
    println!("  Vocab entries: {}", vocab.len());

    // 3. Build octree
    println!("[3/4] Building octree...");
    let mut min = graph[0].position();
    let mut max = graph[0].position();
    for node in &graph {
        min = min.min(node.position());
        max = max.max(node.position());
    }
    let span = (max - min).length().max(1.0);
    let pad = Vec3::splat(span * 0.25);
    let bounds = BoundingBox::new(min - pad, max + pad);
    let mut octree = Octree::new(bounds);
    for node in &graph {
        octree.insert(NodePoint {
            id: node.id,
            x: node.x,
            y: node.y,
            z: node.z,
        });
    }
    println!("  Octree built");

    // 4. Goal-directed episodes with learning
    println!("[4/4] Running {} goal-directed episodes...\n", episodes);

    let start_idx = find_node_by_word(&graph, &vocab, seed_word).unwrap_or(0);
    let goal_idx = find_node_by_word(&graph, &vocab, goal_word).unwrap_or(0);
    let goal_pos = graph[goal_idx].position();

    let mut episode_weights: Vec<f32> = Vec::new();
    let mut episode_final_dist: Vec<f32> = Vec::new();

    for ep in 0..episodes {
        // Decay old edges (forgetting)
        decay_edges(&mut graph, DECAY_RATE);

        // Run episode
        let mut walker = GoalWalker::new(&graph[start_idx], goal_pos);
        for _ in 0..steps {
            let edge_weights = build_edge_weights(&graph);
            walker.step(&graph, &node_index, &octree, &edge_weights);
        }

        // Evaluate: how close did we get to goal?
        let final_dist = walker.position.distance(goal_pos);
        let avg_weight = mean_edge_weight_along_path(&graph, &node_index, &walker.trajectory);

        // Reinforce the path (Hebbian learning)
        reinforce_edges(&mut graph, &walker.trajectory, LEARN_RATE);

        episode_weights.push(avg_weight);
        episode_final_dist.push(final_dist);

        if ep < 3 || ep == episodes - 1 {
            let words: Vec<String> = walker
                .trajectory
                .iter()
                .map(|&nid| {
                    let tid = nid / 1000;
                    vocab
                        .get(&tid)
                        .cloned()
                        .unwrap_or_else(|| format!("<{}>", tid))
                })
                .collect();
            println!(
                "Episode {:2}: final_dist={:.3} avg_pmi={:.3} | {}",
                ep,
                final_dist,
                avg_weight,
                words.join(" ")
            );
        }
    }

    // Summary statistics
    let initial_dist = episode_final_dist[0];
    let final_dist = episode_final_dist[episodes - 1];
    let initial_weight = episode_weights[0];
    let final_weight = episode_weights[episodes - 1];

    println!("\nLearning summary:");
    println!(
        "  Distance to goal:  {:.3} → {:.3} ({:.1}%)",
        initial_dist,
        final_dist,
        (initial_dist - final_dist) / initial_dist * 100.0
    );
    println!(
        "  Avg edge weight:   {:.3} → {:.3} ({:.1}%)",
        initial_weight,
        final_weight,
        (final_weight - initial_weight) / initial_weight * 100.0
    );

    // Save reinforced graph
    let out_dir = format!("{}_learned", graph_dir);
    save_graph4d(&graph, Path::new(&out_dir))?;
    println!("\nLearned graph saved to: {}", out_dir);

    Ok(())
}
