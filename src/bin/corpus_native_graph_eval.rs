//! Coherence evaluation: measure semantic continuity of walker trajectories.
//!
//! Generates multiple text sequences with and without A* path planning,
//! then computes the average graph edge weight along each trajectory.
//! Higher edge weights = stronger PMI co-occurrence = more semantically
//! coherent chains.
//!
//! This tests whether long-range graph planning (astar_find_path_4d)
//! produces more coherent token sequences than purely local spatial queries.
//!
//! Usage:
//!   cargo run --release --example corpus_native_graph_eval -- [graph_dir] [episodes] [steps]

use anyhow::{Context, Result};
use geographdb_core::spatial::octree::{BoundingBox, Octree};
use geographdb_core::storage::data_structures::NodePoint;
use geographdb_core::{astar_find_path_4d, TemporalWindow, TraversalContext4D};
use geographdb_core::{load_graph4d, GraphNode4D};
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ── Walker config ──────────────────────────────────────────────────────────
const K_NEIGHBORS: usize = 20;
const MOMENTUM: f32 = 0.7;
const STEP_SIZE: f32 = 0.3;
const REPETITION_PENALTY: f32 = 0.3;
const PLAN_SPATIAL_RADIUS: f32 = 2.0;

// ── Walker struct (same logic as corpus_native_graph_walk) ─────────────────

struct Walker {
    current_node: u64,
    position: Vec3,
    velocity: Vec3,
    time_step: u64,
    momentum: f32,
    temperature: f32,
    step_size: f32,
    recent_tokens: Vec<u64>,
    recent_window: usize,
    planned_path: Vec<u64>,
    path_index: usize,
    plan_replenish_interval: usize,
}

impl Walker {
    fn new(start_node: &GraphNode4D, temperature: f32, plan_interval: usize) -> Self {
        Self {
            current_node: start_node.id,
            position: start_node.position(),
            velocity: Vec3::ZERO,
            time_step: 0,
            momentum: MOMENTUM,
            temperature,
            step_size: STEP_SIZE,
            recent_tokens: Vec::new(),
            recent_window: 8,
            planned_path: Vec::new(),
            path_index: 0,
            plan_replenish_interval: plan_interval,
        }
    }

    fn replenish_plan(&mut self, graph: &[GraphNode4D], _node_index: &HashMap<u64, usize>) {
        if graph.len() < 2 {
            return;
        }
        let current_pos = self.position;
        let mut candidates: Vec<u64> = graph
            .iter()
            .filter(|n| {
                n.id != self.current_node
                    && n.position().distance(current_pos) < PLAN_SPATIAL_RADIUS * 2.0
            })
            .map(|n| n.id)
            .collect();
        if candidates.is_empty() {
            candidates = graph
                .iter()
                .filter(|n| n.id != self.current_node)
                .map(|n| n.id)
                .collect();
        }
        if candidates.is_empty() {
            return;
        }
        let goal_idx =
            ((self.current_node.wrapping_add(self.time_step)) as usize) % candidates.len();
        let goal_id = candidates[goal_idx];
        let ctx = TraversalContext4D {
            time_window: Some(TemporalWindow {
                start: self.time_step,
                end: self.time_step + self.plan_replenish_interval as u64 * 2,
            }),
            spatial_region: None,
            spatial_candidates: None,
            graph_weight: 1.0,
            spatial_weight: 0.0,
            temporal_weight: 0.5,
        };
        if let Some(path) = astar_find_path_4d(graph, self.current_node, goal_id, &ctx) {
            self.planned_path = path
                .node_ids
                .into_iter()
                .skip_while(|&id| id == self.current_node)
                .collect();
            self.path_index = 0;
        }
    }

    fn step(
        &mut self,
        graph: &[GraphNode4D],
        node_index: &HashMap<u64, usize>,
        octree: &Octree,
        edge_weights: &HashMap<(u64, u64), f32>,
    ) -> u64 {
        if self.plan_replenish_interval > 0
            && (self.planned_path.is_empty() || self.path_index >= self.planned_path.len())
        {
            self.replenish_plan(graph, node_index);
            self.path_index = 0;
        }

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
        let next_planned = self.planned_path.get(self.path_index).copied();

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
            let spatial_score = (-dist_sq / self.temperature).exp();
            let vel_norm = self.velocity.length().max(1e-6);
            let alignment = if vel_norm > 1e-6 {
                self.velocity.dot(dir) / (vel_norm * dist)
            } else {
                0.0
            };
            let alignment_score = (alignment * 0.5 + 0.5).max(0.0);
            let sequential_bonus = if current_successors.contains(&cid) {
                0.5
            } else {
                0.0
            };
            let edge_bonus = edge_weights
                .get(&(self.current_node, cid))
                .copied()
                .unwrap_or(0.0)
                * 0.05;
            let token_id = cid / 1000;
            let repetition_penalty = if recent_set.contains(&token_id) {
                REPETITION_PENALTY * recent_set.iter().filter(|&&t| t == token_id).count() as f32
            } else {
                0.0
            };
            let plan_bonus = if next_planned == Some(cid) { 2.0 } else { 0.0 };
            let score = spatial_score * (1.0 + alignment_score + sequential_bonus)
                + plan_bonus
                + edge_bonus
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

        let next_node = &graph[node_index[&next_id]];
        let target = next_node.position();
        let dir = target - self.position;
        let dir_norm = dir.length().max(1e-6);
        let dir_unit = dir / dir_norm;
        self.velocity = self.velocity * self.momentum + dir_unit * (1.0 - self.momentum);
        self.position += self.velocity * self.step_size;
        self.current_node = next_id;
        self.time_step += 1;
        self.path_index += 1;
        next_id
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_edge_weights(graph: &[GraphNode4D]) -> HashMap<(u64, u64), f32> {
    let mut weights = HashMap::new();
    for node in graph {
        for edge in &node.successors {
            weights.insert((node.id, edge.dst), edge.weight);
        }
    }
    weights
}

fn run_episode(
    graph: &[GraphNode4D],
    node_index: &HashMap<u64, usize>,
    octree: &Octree,
    edge_weights: &HashMap<(u64, u64), f32>,
    start_idx: usize,
    steps: usize,
    plan_interval: usize,
) -> (Vec<u64>, f32) {
    let mut walker = Walker::new(&graph[start_idx], 0.05, plan_interval);
    let mut trajectory: Vec<u64> = Vec::new();
    let mut total_edge_weight = 0.0f32;
    let mut edge_count = 0usize;

    for _ in 0..steps {
        let from = walker.current_node;
        trajectory.push(from / 1000);
        let to = walker.step(graph, node_index, octree, edge_weights);
        if let Some(&w) = edge_weights.get(&(from, to)) {
            total_edge_weight += w;
            edge_count += 1;
        }
        walker.recent_tokens.push(from / 1000);
    }

    let avg_weight = if edge_count > 0 {
        total_edge_weight / edge_count as f32
    } else {
        0.0
    };
    (trajectory, avg_weight)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let graph_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("corpus_native_graph");
    let episodes = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20);
    let steps = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(30);

    println!("Coherence Evaluation");
    println!("====================");
    println!("episodes={} steps={}\n", episodes, steps);

    let graph = load_graph4d(Path::new(graph_dir))
        .with_context(|| format!("Failed to load graph from {}", graph_dir))?;
    println!("Loaded {} nodes", graph.len());

    let node_index: HashMap<u64, usize> =
        graph.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

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
        HashMap::new()
    };

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

    let edge_weights = build_edge_weights(&graph);

    // Run episodes without planning
    println!("Running {} episodes WITHOUT planning...", episodes);
    let mut no_plan_weights: Vec<f32> = Vec::new();
    for ep in 0..episodes {
        let start_idx = (ep * 137) % graph.len();
        let (_, avg_w) = run_episode(
            &graph,
            &node_index,
            &octree,
            &edge_weights,
            start_idx,
            steps,
            0,
        );
        no_plan_weights.push(avg_w);
    }

    // Run episodes with planning
    println!("Running {} episodes WITH planning...", episodes);
    let mut plan_weights: Vec<f32> = Vec::new();
    for ep in 0..episodes {
        let start_idx = (ep * 137) % graph.len();
        let (_, avg_w) = run_episode(
            &graph,
            &node_index,
            &octree,
            &edge_weights,
            start_idx,
            steps,
            6,
        );
        plan_weights.push(avg_w);
    }

    // Statistics
    let no_plan_mean = no_plan_weights.iter().sum::<f32>() / no_plan_weights.len() as f32;
    let plan_mean = plan_weights.iter().sum::<f32>() / plan_weights.len() as f32;
    let no_plan_std = (no_plan_weights
        .iter()
        .map(|w| (w - no_plan_mean).powi(2))
        .sum::<f32>()
        / no_plan_weights.len() as f32)
        .sqrt();
    let plan_std = (plan_weights
        .iter()
        .map(|w| (w - plan_mean).powi(2))
        .sum::<f32>()
        / plan_weights.len() as f32)
        .sqrt();

    // Run episodes with pure random walk
    println!("Running {} episodes with RANDOM walk...", episodes);
    let mut random_weights: Vec<f32> = Vec::new();
    for ep in 0..episodes {
        let start_idx = (ep * 137) % graph.len();
        let (_, avg_w) = run_random_episode(&graph, &node_index, start_idx, steps);
        random_weights.push(avg_w);
    }

    let random_mean = random_weights.iter().sum::<f32>() / random_weights.len() as f32;
    let random_std = (random_weights
        .iter()
        .map(|w| (w - random_mean).powi(2))
        .sum::<f32>()
        / random_weights.len() as f32)
        .sqrt();

    println!("\nResults (average edge weight along trajectory):");
    println!(
        "  Random walk:   mean={:.4}  std={:.4}",
        random_mean, random_std
    );
    println!(
        "  No planning:   mean={:.4}  std={:.4}",
        no_plan_mean, no_plan_std
    );
    println!(
        "  With planning: mean={:.4}  std={:.4}",
        plan_mean, plan_std
    );
    println!(
        "  Walker vs random improvement: {:.1}%",
        (no_plan_mean - random_mean) / random_mean * 100.0
    );

    // Sample trajectories
    let start_idx = graph.iter().position(|n| n.id / 1000 == 0).unwrap_or(0);

    println!("\nSample: RANDOM walk");
    let (traj, _) = run_random_episode(&graph, &node_index, start_idx, 20);
    let words: Vec<String> = traj
        .iter()
        .map(|&tid| {
            vocab
                .get(&tid)
                .cloned()
                .unwrap_or_else(|| format!("<{}>", tid))
        })
        .collect();
    println!("  {}", words.join(" "));

    println!("\nSample: SPATIAL walker (no planning)");
    let (traj, _) = run_episode(
        &graph,
        &node_index,
        &octree,
        &edge_weights,
        start_idx,
        20,
        0,
    );
    let words: Vec<String> = traj
        .iter()
        .map(|&tid| {
            vocab
                .get(&tid)
                .cloned()
                .unwrap_or_else(|| format!("<{}>", tid))
        })
        .collect();
    println!("  {}", words.join(" "));

    println!("\nSample: SPATIAL walker (with A* planning)");
    let (traj, _) = run_episode(
        &graph,
        &node_index,
        &octree,
        &edge_weights,
        start_idx,
        20,
        6,
    );
    let words: Vec<String> = traj
        .iter()
        .map(|&tid| {
            vocab
                .get(&tid)
                .cloned()
                .unwrap_or_else(|| format!("<{}>", tid))
        })
        .collect();
    println!("  {}", words.join(" "));

    Ok(())
}

// Random walk baseline: pick a random successor each step
fn run_random_episode(
    graph: &[GraphNode4D],
    node_index: &HashMap<u64, usize>,
    start_idx: usize,
    steps: usize,
) -> (Vec<u64>, f32) {
    let mut current_id = graph[start_idx].id;
    let mut trajectory: Vec<u64> = Vec::new();
    let mut total_edge_weight = 0.0f32;
    let mut edge_count = 0usize;

    for _ in 0..steps {
        trajectory.push(current_id / 1000);
        let idx = node_index[&current_id];
        let successors: Vec<u64> = graph[idx].successors.iter().map(|e| e.dst).collect();
        if successors.is_empty() {
            break;
        }
        let next_id = successors
            [current_id.wrapping_add(trajectory.len() as u64) as usize % successors.len()];
        if let Some(edge) = graph[idx].successors.iter().find(|e| e.dst == next_id) {
            total_edge_weight += edge.weight;
            edge_count += 1;
        }
        current_id = next_id;
    }

    let avg_weight = if edge_count > 0 {
        total_edge_weight / edge_count as f32
    } else {
        0.0
    };
    (trajectory, avg_weight)
}
