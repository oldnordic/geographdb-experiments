//! D3 baseline comparison: distance-based walker vs geometric graph-attention walker.
//!
//! Runs both `TransitionMode::DistanceKnn` and
//! `TransitionMode::GraphAttentionTransport` for the same number of steps and
//! reports:
//!
//! - repetition rate = unique token IDs / total steps
//! - cross-community transitions
//! - first 20 nodes of each trajectory
//!
//! If a persisted graph directory is passed as the first argument, the graph is
//! loaded from disk; otherwise a small toy corpus graph with two communities is
//! used.
//!
//! This is the paper's core comparison: the geometry-derived attention should
//! traverse more coherently (lower repetition, purposeful cross-community moves)
//! even with W_Q = W_K = I.

use geographdb_core::algorithms::four_d::{GraphProperties, TemporalEdge};
use geographdb_core::corpus::{
    build_edge_weights, build_node_index, build_octree, GeometricWalker, TransitionMode,
    WalkerConfig,
};
use geographdb_core::{curvature_map_fast, load_graph4d, GraphNode4D};
use glam::Vec3;
use std::collections::HashMap;
use std::path::Path;

const COMMUNITY_A: u64 = 1;
const COMMUNITY_B: u64 = 2;

fn make_node(id: u64, pos: Vec3, community: u64) -> GraphNode4D {
    let mut props = GraphProperties::new();
    props.insert(
        "community".to_string(),
        serde_json::Value::String(community.to_string()),
    );
    GraphNode4D {
        id,
        x: pos.x,
        y: pos.y,
        z: pos.z,
        begin_ts: 0,
        end_ts: u64::MAX,
        properties: props,
        successors: Vec::new(),
    }
}

fn build_toy_corpus_graph() -> Vec<GraphNode4D> {
    // Community A: tokens 1000, 1001, 1002 clustered around (0,0,0)
    // Community B: tokens 2000, 2001, 2002 clustered around (2,0,0)
    let mut nodes = vec![
        make_node(1000, Vec3::new(0.0, 0.0, 0.0), COMMUNITY_A),
        make_node(1001, Vec3::new(0.1, 0.2, 0.0), COMMUNITY_A),
        make_node(1002, Vec3::new(-0.1, 0.1, 0.2), COMMUNITY_A),
        make_node(2000, Vec3::new(2.0, 0.0, 0.0), COMMUNITY_B),
        make_node(2001, Vec3::new(2.1, 0.2, 0.0), COMMUNITY_B),
        make_node(2002, Vec3::new(1.9, 0.1, 0.2), COMMUNITY_B),
    ];

    // Intra-community edges (dense local paths)
    let edges: Vec<(usize, usize)> = vec![
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        // Bridges between communities
        (2, 3),
        (1, 4),
    ];

    for (src, dst) in edges {
        let src_id = nodes[src].id;
        let dst_id = nodes[dst].id;
        nodes[src].successors.push(TemporalEdge {
            dst: dst_id,
            weight: 1.0,
            begin_ts: 0,
            end_ts: u64::MAX,
        });
        // Treat edges as undirected for this toy graph.
        nodes[dst].successors.push(TemporalEdge {
            dst: src_id,
            weight: 1.0,
            begin_ts: 0,
            end_ts: u64::MAX,
        });
    }

    nodes
}

fn community_assignments(graph: &[GraphNode4D]) -> HashMap<u64, u64> {
    // For real graphs, split the node population by the median x coordinate.
    // For the toy graph the hard-coded ranges below produce the same result.
    let mut xs: Vec<f32> = graph.iter().map(|n| n.x).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = xs[xs.len() / 2];

    graph
        .iter()
        .map(|n| {
            let community = if n.x < median || n.id < 2000 {
                COMMUNITY_A
            } else {
                COMMUNITY_B
            };
            (n.id, community)
        })
        .collect()
}

fn evaluate(name: &str, trajectory: &[u64], communities: &HashMap<u64, u64>) {
    let total = trajectory.len();
    let unique: std::collections::HashSet<u64> = trajectory.iter().copied().collect();
    let repetition_rate = if total > 0 {
        unique.len() as f32 / total as f32
    } else {
        0.0
    };

    let cross_community = trajectory
        .windows(2)
        .filter(|w| communities.get(&w[0]) != communities.get(&w[1]))
        .count();

    println!("\n=== {} ===", name);
    println!("  steps:            {}", total);
    println!("  unique nodes:     {}", unique.len());
    println!("  repetition rate:  {:.3}", repetition_rate);
    println!("  cross-community:  {}", cross_community);
    println!(
        "  trajectory:       {:?}",
        trajectory.iter().take(20).copied().collect::<Vec<_>>()
    );
}

fn run_walk(
    graph: &[GraphNode4D],
    mode: TransitionMode,
    steps: usize,
    curvature: Option<&std::collections::HashMap<(u64, u64), f32>>,
) -> Vec<u64> {
    let idx = build_node_index(graph);
    let octree = build_octree(graph);
    let weights = build_edge_weights(graph);

    let mut config = WalkerConfig::default();
    config.knn = 5;
    config.temperature = 0.05;
    config.plan_interval = 0;
    config.repetition_penalty = 0.0; // isolate geometry, not repetition suppression
    config.transition_mode = mode;

    let mut walker = GeometricWalker::new(&graph[0], config);
    walker.walk(graph, &idx, &octree, &weights, curvature, steps)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let graph = if args.len() > 1 {
        let path = Path::new(&args[1]);
        load_graph4d(path).expect("failed to load persisted graph")
    } else {
        build_toy_corpus_graph()
    };

    let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);

    let communities = community_assignments(&graph);

    let distance_traj = run_walk(&graph, TransitionMode::DistanceKnn, steps, None);
    let attention_traj = run_walk(
        &graph,
        TransitionMode::GraphAttentionTransport { ucb_c: 0.5 },
        steps,
        None,
    );

    // Compute fast Dice κ once and run the κ-weighted variant.
    // For count-weighted graphs pass a threshold (e.g. 10.0); for PMI-weighted
    // graphs 0.0 keeps every edge.
    let ricci_threshold = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0f32);
    let ricci = curvature_map_fast(&graph, ricci_threshold);
    let attention_ricci_traj = run_walk(
        &graph,
        TransitionMode::GraphAttentionTransport { ucb_c: 0.5 },
        steps,
        Some(&ricci),
    );

    evaluate("Distance-KNN", &distance_traj, &communities);
    evaluate("Graph attention + UCB", &attention_traj, &communities);
    evaluate(
        "Graph attention + UCB + κ",
        &attention_ricci_traj,
        &communities,
    );
}
