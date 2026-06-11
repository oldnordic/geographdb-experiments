//! Multi-Formation Crop Circle Analysis using GeoGraphDB Core
//!
//! Analyzes 3 unexplained formations beyond the Julia Set:
//! - Milk Hill Spiral (2001): 409 circles in logarithmic spiral
//! - Pi Formation (2008): First 10 digits of π encoded in spiral arcs
//! - Arecibo Reply (2001): Binary-encoded message grid
//!
//! Each formation is modeled as a 4D graph, then analyzed with:
//! 1. Graph storage and connectivity
//! 2. Ricci curvature (sampled)
//! 3. Percolation analysis
//! 4. Tensor network encoding

use geographdb_core::algorithms::delay_embed::correlation_dimension;
use geographdb_core::algorithms::four_d::{
    reachable_4d, GraphNode4D, GraphProperties, TemporalEdge, TemporalWindow, TraversalContext4D,
};
use geographdb_core::algorithms::mps::{build_mps, get_tensor, mps_norm_sq};
use geographdb_core::algorithms::percolation::{find_critical_radius, percolation_sweep};
use geographdb_core::algorithms::ricci::ollivier_ricci;
use glam::Vec3;

// ── Milk Hill Spiral ─────────────────────────────────────────────────────────

/// Generate the Milk Hill 2001 spiral: 409 circles in logarithmic spiral.
/// r = a * e^(b*θ), with 409 points.
fn generate_milk_hill_spiral() -> Vec<Vec<f32>> {
    let n_circles = 409;
    let a = 0.5f32;
    let b = 0.08f32;
    let max_theta = 12.0 * std::f32::consts::PI;

    let mut points = Vec::with_capacity(n_circles);
    for i in 0..n_circles {
        let t = i as f32 / (n_circles as f32 - 1.0);
        let theta = t * max_theta;
        let r = a * (b * theta).exp();
        let x = r * theta.cos();
        let y = r * theta.sin();
        points.push(vec![x, y]);
    }
    points
}

// ── Pi Formation ─────────────────────────────────────────────────────────────

/// Generate the Pi Formation (Barbury Castle, 2008):
/// Spiral with 10 arcs, each arc length = digit of π.
fn generate_pi_formation() -> Vec<Vec<f32>> {
    let pi_digits: [usize; 10] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 4];
    let mut points = Vec::new();
    let mut theta_start = 0.0f32;

    for digit in pi_digits {
        let n_steps = digit * 20; // Scale for density
        let arc_length = digit as f32 * 0.3;
        let dtheta = arc_length / 5.0; // Approximate

        for i in 0..=n_steps {
            let t = i as f32 / n_steps as f32;
            let theta = theta_start + t * dtheta;
            let r = 0.1 + 0.15 * theta; // Expanding spiral
            let x = r * theta.cos();
            let y = r * theta.sin();
            points.push(vec![x, y]);
        }
        theta_start += dtheta + 0.05; // Gap between arcs
    }
    points
}

// ── Arecibo Reply Grid ───────────────────────────────────────────────────────

/// Generate the Arecibo Reply as a binary grid (23×73 = 1679).
/// We convert active pixels to 2D points.
fn generate_arecibo_reply() -> Vec<Vec<f32>> {
    let rows = 23usize;
    let cols = 73usize;
    let mut points = Vec::new();

    // Simplified representation: create patterns that encode the key features
    // Row 1-4: Numbers 1-10 (binary)
    for i in 0..4 {
        for j in 0..10 {
            if j < (i + 1) {
                let x = (j * 7 + 3) as f32 / cols as f32 * 2.0 - 1.0;
                let y = 1.0 - (i as f32 / rows as f32 * 2.0);
                points.push(vec![x, y]);
            }
        }
    }

    // Row 5-8: Atomic numbers (H=1, C=6, N=7, O=8, P=15, Si=14)
    let atomic = [1, 6, 7, 8, 15, 14];
    for (i, &atom) in atomic.iter().enumerate() {
        for j in 0..atom.min(20) {
            let x = (10 + i * 5 + j) as f32 / cols as f32 * 2.0 - 1.0;
            let y = 1.0 - ((4 + i % 4) as f32 / rows as f32 * 2.0);
            points.push(vec![x, y]);
        }
    }

    // Row 9-12: DNA structure (different from human)
    for i in 0..4 {
        for j in 0..20 {
            if j % 5 < 3 {
                let x = (40 + j) as f32 / cols as f32 * 2.0 - 1.0;
                let y = 1.0 - ((8 + i) as f32 / rows as f32 * 2.0);
                points.push(vec![x, y]);
            }
        }
    }

    // Row 13-16: Being height (~4 feet = 121 cm)
    for i in 0..4 {
        for j in 0..cols.min(30) {
            let x = j as f32 / cols as f32 * 2.0 - 1.0;
            let y = 1.0 - ((12 + i) as f32 / rows as f32 * 2.0);
            points.push(vec![x, y]);
        }
    }

    // Row 17-20: Population (~12.7 billion)
    for i in 0..4 {
        for j in 0..cols {
            let x = j as f32 / cols as f32 * 2.0 - 1.0;
            let y = 1.0 - ((16 + i) as f32 / rows as f32 * 2.0);
            points.push(vec![x, y]);
        }
    }

    // Row 21-23: Solar system (9 planets, 3rd emphasized)
    for i in 0..3 {
        for j in 0..9 {
            let base_x = j * 8 + 3;
            let x = base_x as f32 / cols as f32 * 2.0 - 1.0;
            let y = 1.0 - ((20 + i) as f32 / rows as f32 * 2.0);
            points.push(vec![x, y]);

            if j == 2 {
                // 3rd planet (Earth) emphasized
                let x2 = (base_x + 1) as f32 / cols as f32 * 2.0 - 1.0;
                points.push(vec![x2, y]);
            }
        }
    }

    points
}

// ── Shared utilities ─────────────────────────────────────────────────────────

fn print_results(name: &str, points: &[Vec<f32>], dim: f64) {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ {:<59} │", name);
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Point count: {:<46} │", points.len());
    println!("│ Estimated fractal dimension: {:<30.4} │", dim);
    println!(
        "│ Status: {:<51} │",
        if dim > 1.5 && dim < 2.0 {
            "TRUE FRACTAL (non-integer dimension)"
        } else if dim >= 2.0 {
            "FILLS 2D SPACE"
        } else if dim >= 1.0 {
            "LINE-LIKE / SPIRAL"
        } else {
            "DEGENERATE / DISCRETE"
        }
    );
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
}

fn point_cloud_to_graph_4d(points: &[Vec<f32>], edge_threshold: f32) -> Vec<GraphNode4D> {
    let mut nodes: Vec<GraphNode4D> = points
        .iter()
        .enumerate()
        .map(|(i, pt)| {
            let mut props = GraphProperties::new();
            props.insert("original_index".into(), serde_json::Value::Number(i.into()));
            GraphNode4D {
                id: i as u64,
                x: pt[0],
                y: pt[1],
                z: 0.0,
                begin_ts: 0,
                end_ts: 1,
                properties: props,
                successors: Vec::new(),
            }
        })
        .collect();

    for i in 0..nodes.len() {
        let pos_i = nodes[i].position();
        for j in (i + 1)..nodes.len() {
            let pos_j = nodes[j].position();
            let dist = pos_i.distance(pos_j);
            if dist <= edge_threshold {
                let weight = 1.0 / (dist + 0.001);
                nodes[i].successors.push(TemporalEdge {
                    dst: j as u64,
                    weight,
                    begin_ts: 0,
                    end_ts: 1,
                });
                nodes[j].successors.push(TemporalEdge {
                    dst: i as u64,
                    weight,
                    begin_ts: 0,
                    end_ts: 1,
                });
            }
        }
    }
    nodes
}

// ── Experiments ──────────────────────────────────────────────────────────────

fn experiment_1_graph_storage(name: &str, points: &[Vec<f32>]) {
    println!("EXPERIMENT 1: {} as 4D Graph Nodes", name);
    println!("───────────────────────────────────────────────────────────────");

    let nodes = point_cloud_to_graph_4d(points, 0.05);
    let total_edges: usize = nodes.iter().map(|n| n.successors.len()).sum();
    let avg_degree = total_edges as f64 / nodes.len().max(1) as f64;

    println!("  Total nodes: {}", nodes.len());
    println!("  Total edges: {}", total_edges);
    println!("  Average degree: {:.2}", avg_degree);

    if let Some(max_node) = nodes.iter().max_by_key(|n| n.successors.len()) {
        println!(
            "  Max degree: {} (node id: {})",
            max_node.successors.len(),
            max_node.id
        );
    }

    let center_id = (nodes.len() / 2) as u64;
    let ctx = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 0, end: 1 }),
        spatial_region: None,
        spatial_candidates: None,
        graph_weight: 1.0,
        spatial_weight: 0.0,
        temporal_weight: 0.0,
    };
    let reachable = reachable_4d(&nodes, center_id, &ctx);
    println!(
        "  Reachability from center: {} / {} ({:.1}%)",
        reachable.len(),
        nodes.len(),
        100.0 * reachable.len() as f64 / nodes.len().max(1) as f64
    );
    println!();
}

fn experiment_2_ricci_curvature(name: &str, nodes: &[GraphNode4D]) {
    println!("EXPERIMENT 2: Ricci Curvature on {}", name);
    println!("───────────────────────────────────────────────────────────────");

    let sample_size = nodes.len().min(200);
    let step = nodes.len() / sample_size.max(1);
    let sampled: Vec<GraphNode4D> = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| i % step == 0)
        .take(sample_size)
        .map(|(_, n)| n.clone())
        .collect();

    println!("  Sampled: {} nodes (from {})", sampled.len(), nodes.len());

    let ricci_edges = ollivier_ricci(&sampled, 0.5);
    if ricci_edges.is_empty() {
        println!("  No Ricci edges computed.\n");
        return;
    }

    let curvatures: Vec<f32> = ricci_edges.iter().map(|e| e.curvature).collect();
    let avg = curvatures.iter().sum::<f32>() / curvatures.len() as f32;
    let min_c = curvatures.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_c = curvatures.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!(
        "  Edges: {} | Avg: {:.4} | Min: {:.4} | Max: {:.4}",
        ricci_edges.len(),
        avg,
        min_c,
        max_c
    );

    if avg < -0.1 {
        println!("  → NEGATIVE curvature: hyperbolic-like / fractal branching");
    } else if avg > 0.1 {
        println!("  → POSITIVE curvature: compact / spherical-like");
    } else {
        println!("  → NEAR-ZERO curvature: flat / Euclidean-like");
    }
    println!();
}

fn experiment_3_percolation(name: &str, nodes: &[GraphNode4D]) {
    println!("EXPERIMENT 3: Percolation Analysis on {}", name);
    println!("───────────────────────────────────────────────────────────────");

    let center_id = (nodes.len() / 2) as u64;
    let sphere_center = Vec3::new(0.0, 0.0, 0.0);
    let radii: Vec<f32> = (1..=20).map(|i| i as f32 * 0.05).collect();
    let time_slices = vec![0u64];

    let percolation = percolation_sweep(nodes, center_id, sphere_center, &radii, &time_slices, 1);

    println!(
        "  {:<10} {:<10} {:<10} {:<12}",
        "Radius", "Active", "Reachable", "Fraction"
    );
    println!("  {}", "-".repeat(48));
    for p in &percolation {
        println!(
            "  {:<10.3} {:<10} {:<10} {:<12.4}",
            p.radius, p.active, p.reachable, p.fraction
        );
    }

    if let Some(cr) = find_critical_radius(&percolation, 0, 0.5, 1) {
        println!("  Critical radius (50%): {:.3}", cr);
    } else {
        println!("  No critical radius in range.");
    }

    let fractions: Vec<f32> = percolation.iter().map(|p| p.fraction).collect();
    if fractions.len() >= 2 {
        let max_jump = fractions
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, |a, b| a.max(b));
        println!("  Max connectivity jump: {:.4}", max_jump);
        if max_jump > 0.3 {
            println!("  → SHARP phase transition (fractal characteristic)");
        } else {
            println!("  → Gradual transition");
        }
    }
    println!();
}

fn experiment_4_tensor_network(name: &str, points: &[Vec<f32>]) {
    println!("EXPERIMENT 4: Tensor Network Encoding of {}", name);
    println!("───────────────────────────────────────────────────────────────");

    // Group points by distance from origin into concentric rings
    let n_rings = 10;
    let mut rings: Vec<Vec<(f32, f32)>> = vec![Vec::new(); n_rings];

    let max_r = points
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .fold(0.0f32, |a, b| a.max(b));

    for pt in points {
        let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
        let idx = ((r / max_r) * (n_rings as f32 - 1.0)).min(n_rings as f32 - 1.0) as usize;
        rings[idx].push((pt[0], pt[1]));
    }

    println!("  Points distributed into {} rings:", n_rings);
    for (i, ring) in rings.iter().enumerate() {
        println!("    Ring {}: {} points", i, ring.len());
    }

    // Build MPS from angular histograms
    let mut tensors: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();
    for ring in &rings {
        let n_bins = 8;
        let mut hist = vec![0.0f32; n_bins];
        for &(x, y) in ring {
            let angle = y.atan2(x);
            let bin = (((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI))
                * n_bins as f32)
                .min(n_bins as f32 - 1.0) as usize;
            hist[bin] += 1.0;
        }
        let sum: f32 = hist.iter().sum();
        if sum > 0.0 {
            for h in &mut hist {
                *h /= sum;
            }
        }
        tensors.push((vec![1, n_bins, 1], hist));
    }

    let tensor_refs: Vec<(&[usize], &[f32])> = tensors
        .iter()
        .map(|(s, d)| (s.as_slice(), d.as_slice()))
        .collect();

    let mps_nodes = build_mps(&tensor_refs);
    let norm_sq = mps_norm_sq(&mps_nodes);

    println!(
        "  MPS sites: {} | Norm²: {:.6} | Norm: {:.6}",
        mps_nodes.len(),
        norm_sq,
        norm_sq.sqrt()
    );

    println!("  Tensor verification:");
    for (i, node) in mps_nodes.iter().take(3).enumerate() {
        if let Some((shape, data)) = get_tensor(node) {
            println!(
                "    Site {}: shape={:?}, sum={:.4}",
                i,
                shape,
                data.iter().sum::<f32>()
            );
        }
    }
    println!();
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn analyze_formation(name: &str, points: &[Vec<f32>]) {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  ANALYZING: {:<47} ║", name);
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!();

    let dim = correlation_dimension(points, 0.001, 2.0, 25);
    print_results(name, points, dim);

    let graph_nodes = point_cloud_to_graph_4d(points, 0.05);
    experiment_1_graph_storage(name, points);
    experiment_2_ricci_curvature(name, &graph_nodes);
    experiment_3_percolation(name, &graph_nodes);
    experiment_4_tensor_network(name, points);

    println!("═══════════════════════════════════════════════════════════════");
    println!();
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  MULTI-FORMATION CROP CIRCLE ANALYSIS SUITE                ║");
    println!("║  Using GeoGraphDB Core                                     ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!();

    // Formation 1: Milk Hill Spiral
    let milk_hill = generate_milk_hill_spiral();
    analyze_formation("MILK HILL SPIRAL (2001)", &milk_hill);

    // Formation 2: Pi Formation
    let pi_formation = generate_pi_formation();
    analyze_formation("PI FORMATION (2008)", &pi_formation);

    // Formation 3: Arecibo Reply
    let arecibo = generate_arecibo_reply();
    analyze_formation("ARECIBO REPLY (2001)", &arecibo);

    // Summary
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  ALL FORMATIONS ANALYZED                                   ║");
    println!("╠═════════════════════════════════════════════════════════════╣");
    println!("║  1. Milk Hill Spiral  - 409 circles, logarithmic spiral    ║");
    println!("║  2. Pi Formation      - 10 digits of π encoded             ║");
    println!("║  3. Arecibo Reply     - Binary message grid                ║");
    println!("╠═════════════════════════════════════════════════════════════╣");
    println!("║  For each formation:                                       ║");
    println!("║    ✓ Fractal dimension estimation                          ║");
    println!("║    ✓ 4D graph storage + connectivity                       ║");
    println!("║    ✓ Ricci curvature analysis                              ║");
    println!("║    ✓ Percolation phase transitions                         ║");
    println!("║    ✓ Tensor network encoding                               ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
}
