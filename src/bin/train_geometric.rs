//! Train a pure-Rust geometric MLP classifier on a 3D corpus graph.
//!
//! Each graph node is treated as a training example:
//! - input = normalised 3D position `(x, y, z)`
//! - target = community label derived from the node property `community` (if
//!   present) or from the median `x` coordinate for toy graphs.
//!
//! The model is a two-layer MLP trained with Adam and cross-entropy. It uses
//! only the pure-Rust operators in `geographdb-core` (`matmul`, `softmax`,
//! cross-entropy, Adam).
//!
//! Usage:
//!   cargo run --release --bin train_geometric -- [graph_dir]
//!
//! Flags:
//!   --epochs N       training epochs (default 200)
//!   --lr LR          Adam learning rate (default 0.01)
//!   --hidden-dim H   hidden layer width (default 32)
//!   --seed S         random seed (default 42)
//!   --output DIR     directory for saved model and metrics (default /tmp/train_geometric)

use geographdb_core::algorithms::four_d::{GraphProperties, TemporalEdge};
use geographdb_core::{load_graph4d, GraphNode4D};
use geographdb_core::{Adam, MlpClassifier};
use glam::Vec3;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const COMMUNITY_A: u64 = 1;
const COMMUNITY_B: u64 = 2;

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
    // Two spatially separated communities.
    let mut nodes = vec![
        make_node(1000, Vec3::new(0.0, 0.0, 0.0), COMMUNITY_A),
        make_node(1001, Vec3::new(0.1, 0.2, 0.0), COMMUNITY_A),
        make_node(1002, Vec3::new(-0.1, 0.1, 0.2), COMMUNITY_A),
        make_node(1003, Vec3::new(0.2, -0.1, 0.1), COMMUNITY_A),
        make_node(2000, Vec3::new(2.0, 0.0, 0.0), COMMUNITY_B),
        make_node(2001, Vec3::new(2.1, 0.2, 0.0), COMMUNITY_B),
        make_node(2002, Vec3::new(1.9, 0.1, 0.2), COMMUNITY_B),
        make_node(2003, Vec3::new(2.0, -0.2, 0.1), COMMUNITY_B),
    ];

    let edges: Vec<(usize, usize)> = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (2, 5),
        (3, 6),
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
        nodes[dst].successors.push(TemporalEdge {
            dst: src_id,
            weight: 1.0,
            begin_ts: 0,
            end_ts: u64::MAX,
        });
    }

    nodes
}

fn community_of(node: &GraphNode4D, median_x: f32) -> u64 {
    if let Some(serde_json::Value::String(s)) = node.properties.get("community") {
        if let Ok(c) = s.parse::<u64>() {
            return c;
        }
    }
    if node.x < median_x {
        COMMUNITY_A
    } else {
        COMMUNITY_B
    }
}

fn compute_median_x(graph: &[GraphNode4D]) -> f32 {
    let mut xs: Vec<f32> = graph.iter().map(|n| n.x).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn normalize_positions(graph: &[GraphNode4D]) -> (Vec<[f32; 3]>, [f32; 3], [f32; 3]) {
    let mut mean = [0.0f32; 3];
    for node in graph {
        mean[0] += node.x;
        mean[1] += node.y;
        mean[2] += node.z;
    }
    let inv = 1.0 / graph.len() as f32;
    for i in 0..3 {
        mean[i] *= inv;
    }

    let mut var = [0.0f32; 3];
    for node in graph {
        let d = [node.x - mean[0], node.y - mean[1], node.z - mean[2]];
        for i in 0..3 {
            var[i] += d[i] * d[i];
        }
    }
    for i in 0..3 {
        var[i] *= inv;
    }

    let std = [
        var[0].sqrt().max(1e-6),
        var[1].sqrt().max(1e-6),
        var[2].sqrt().max(1e-6),
    ];

    let normalized: Vec<[f32; 3]> = graph
        .iter()
        .map(|n| {
            [
                (n.x - mean[0]) / std[0],
                (n.y - mean[1]) / std[1],
                (n.z - mean[2]) / std[2],
            ]
        })
        .collect();

    (normalized, mean, std)
}

fn evaluate(model: &MlpClassifier, inputs: &[[f32; 3]], targets: &[usize]) -> (f32, f32) {
    let mut loss = 0.0f32;
    let mut correct = 0usize;
    for (x, y) in inputs.iter().zip(targets.iter()) {
        loss += model.loss(x, *y);
        if model.predict(x) == *y {
            correct += 1;
        }
    }
    let count = inputs.len() as f32;
    (loss / count, correct as f32 / count)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let graph = if let Some(path) = args
        .windows(2)
        .find(|w| !w[0].starts_with("--"))
        .map(|w| &w[1])
    {
        load_graph4d(Path::new(path)).expect("failed to load persisted graph")
    } else {
        build_toy_corpus_graph()
    };

    let epochs = parse_arg_usize(&args, "--epochs", 200);
    let lr = parse_arg_f32(&args, "--lr", 0.01);
    let hidden_dim = parse_arg_usize(&args, "--hidden-dim", 32);
    let seed = parse_arg_usize(&args, "--seed", 42) as u32;
    let output_dir = parse_arg(&args, "--output", "/tmp/train_geometric");

    std::fs::create_dir_all(&output_dir).expect("failed to create output directory");

    println!("Training geometric MLP classifier");
    println!("=================================");
    println!("  nodes:       {}", graph.len());
    println!("  epochs:      {}", epochs);
    println!("  lr:          {}", lr);
    println!("  hidden_dim:  {}", hidden_dim);
    println!("  seed:        {}", seed);
    println!("  output:      {}", output_dir);

    let median_x = compute_median_x(&graph);
    let (inputs, mean, std) = normalize_positions(&graph);
    let targets: Vec<usize> = graph
        .iter()
        .map(|n| {
            let c = community_of(n, median_x);
            if c == COMMUNITY_A {
                0
            } else {
                1
            }
        })
        .collect();

    let class_counts = targets.iter().fold(HashMap::new(), |mut acc, &t| {
        *acc.entry(t).or_insert(0usize) += 1;
        acc
    });
    println!("  classes:     {:?}", class_counts);

    let mut model = MlpClassifier::new(3, hidden_dim, 2, seed);
    let total_params = model.flatten_params().len();
    let mut params = model.flatten_params();
    let mut opt = Adam::with_hyperparams(lr, 0.9, 0.999, 1e-8, total_params);

    let mut grad_accum = vec![0.0f32; total_params];

    for epoch in 1..=epochs {
        grad_accum.fill(0.0);

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let grad = model.backward(x, *y);
            let flat = model.flatten_grad(&grad);
            for i in 0..total_params {
                grad_accum[i] += flat[i];
            }
        }

        let inv_n = 1.0 / inputs.len() as f32;
        for g in grad_accum.iter_mut() {
            *g *= inv_n;
        }

        opt.step(&mut params, &grad_accum);
        model.load_flat_params(&params);

        if epoch == 1 || epoch % 20 == 0 || epoch == epochs {
            let (loss, acc) = evaluate(&model, &inputs, &targets);
            println!(
                "  epoch {:>3}  loss={:.6}  accuracy={:.2}%",
                epoch,
                loss,
                acc * 100.0
            );
        }
    }

    let (final_loss, final_acc) = evaluate(&model, &inputs, &targets);
    println!("\nFinal");
    println!("  loss:     {:.6}", final_loss);
    println!("  accuracy: {:.2}%", final_acc * 100.0);

    // Save model and metadata.
    let model_path = Path::new(&output_dir).join("model.json");
    let weights = serde_json::json!({
        "input_dim": 3,
        "hidden_dim": hidden_dim,
        "output_dim": 2,
        "w1": model.w1,
        "b1": model.b1,
        "w2": model.w2,
        "b2": model.b2,
        "mean": mean,
        "std": std,
    });
    std::fs::write(&model_path, serde_json::to_string_pretty(&weights).unwrap())
        .expect("failed to write model");

    let summary = serde_json::json!({
        "nodes": graph.len(),
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
    });
    let summary_path = Path::new(&output_dir).join("summary.json");
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("failed to write summary");

    let per_node_path = Path::new(&output_dir).join("predictions.jsonl");
    let mut file =
        std::fs::File::create(&per_node_path).expect("failed to create predictions file");
    for (i, node) in graph.iter().enumerate() {
        let pred = model.predict(&inputs[i]);
        let probs = model.probabilities(&inputs[i]);
        let line = serde_json::json!({
            "node_id": node.id,
            "position": [node.x, node.y, node.z],
            "target": targets[i],
            "predicted": pred,
            "probabilities": probs,
        });
        serde_json::to_writer(&mut file, &line).unwrap();
        file.write_all(b"\n").unwrap();
    }

    println!("  model:       {}", model_path.display());
    println!("  summary:     {}", summary_path.display());
    println!("  predictions: {}", per_node_path.display());
}
