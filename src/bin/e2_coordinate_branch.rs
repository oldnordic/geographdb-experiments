//! E2 — coordinate branch demo: token ID → ℝ³.
//!
//! Trains a tiny CPU-only MLP to predict the 3D positions of tokens from their
//! IDs, supervised by positions measured from a PMI+SVD-style corpus graph.
//!
//! If a persisted graph directory is passed as the first argument, the graph is
//! loaded from disk; otherwise a small synthetic graph is used.
//!
//! Usage:
//!   cargo run --release --bin e2_coordinate_branch -- [graph_dir] [vocab_size]

use geographdb_core::algorithms::four_d::GraphProperties;
use geographdb_core::corpus::CoordinateBranch;
use geographdb_core::{load_graph4d, GraphNode4D};
use glam::Vec3;
use std::path::Path;

fn make_node(id: u64, pos: Vec3) -> GraphNode4D {
    GraphNode4D {
        id,
        x: pos.x,
        y: pos.y,
        z: pos.z,
        begin_ts: 0,
        end_ts: u64::MAX,
        properties: GraphProperties::new(),
        successors: Vec::new(),
    }
}

fn build_synthetic_graph() -> Vec<GraphNode4D> {
    // A tiny 3D embedding: tokens placed on the axes and the diagonal.
    vec![
        make_node(1000, Vec3::new(0.0, 0.0, 0.0)),
        make_node(2000, Vec3::new(1.0, 0.0, 0.0)),
        make_node(3000, Vec3::new(0.0, 1.0, 0.0)),
        make_node(4000, Vec3::new(0.0, 0.0, 1.0)),
        make_node(5000, Vec3::new(1.0, 1.0, 1.0)),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (graph, vocab_size) = if args.len() > 1 {
        let path = Path::new(&args[1]);
        let graph = load_graph4d(path).expect("failed to load persisted graph");
        let max_token = graph.iter().map(|n| n.id / 1000).max().unwrap_or(0) as usize;
        let vocab = args
            .get(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(max_token + 1);
        (graph, vocab)
    } else {
        (build_synthetic_graph(), 6)
    };

    let mut branch = CoordinateBranch::new(vocab_size, 8, 32, 42);

    println!("Training coordinate branch on {} nodes...", graph.len());
    let initial_loss = branch.train_epoch(&graph, 0.05);
    let final_loss = branch.fit(&graph, 500, 0.1);

    println!("Initial MSE loss: {:.6}", initial_loss);
    println!("Final MSE loss:   {:.6}", final_loss);

    println!("\nPredictions vs measured positions:");
    for node in &graph {
        let token_id = node.id / 1000;
        let pred = branch.predict(token_id);
        let target = node.position();
        let err = pred.distance(target);
        println!(
            "  token {}: pred=({:.3}, {:.3}, {:.3}) target=({:.3}, {:.3}, {:.3}) err={:.4}",
            token_id, pred.x, pred.y, pred.z, target.x, target.y, target.z, err
        );
    }
}
