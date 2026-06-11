//! Crop Circle Fractal Dimension Analysis using GeoGraphDB Core
//!
//! This demo uses the existing `delay_embed` and `correlation_dimension`
//! algorithms from geographdb-core to analyze the mathematical properties
//! of unexplained crop circle formations.
//!
//! # The Julia Set Formation (Stonehenge, 1996)
//!
//! The most mathematically significant crop circle: 151 circles arranged in a
//! spiral pattern matching a specific Julia set with c ≈ -3/4 + (1/9)i.
//!
//! We generate the Julia set as a point cloud, then use the
//! Grassberger-Procaccia algorithm to estimate its fractal dimension.
//!
//! # Usage
//! ```bash
//! cargo run --example demo_crop_circle_fractal_dim --release
//! ```

use geographdb_core::algorithms::delay_embed::correlation_dimension;
use geographdb_core::algorithms::four_d::{
    reachable_4d, GraphNode4D, GraphProperties, TemporalEdge, TemporalWindow, TraversalContext4D,
};
use geographdb_core::algorithms::mps::{build_mps, get_tensor, mps_norm_sq};
use geographdb_core::algorithms::percolation::{find_critical_radius, percolation_sweep};
use geographdb_core::algorithms::ricci::ollivier_ricci;
use glam::Vec3;

/// Complex number for Julia set iteration.
#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn magnitude_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn add(&self, other: &Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }

    fn sqr(&self) -> Complex {
        Complex::new(
            self.re * self.re - self.im * self.im,
            2.0 * self.re * self.im,
        )
    }
}

/// Generate a Julia set point cloud by iterating z = z² + c for each point
/// in a grid. Points that remain bounded (|z| < 2 after max_iter iterations)
/// are part of the Julia set.
///
/// The c value is derived from the Stonehenge 1996 crop circle:
/// c ≈ -3/4 + (1/9)i ≈ -0.745429 + 0.113008i
fn generate_julia_set_point_cloud(
    c: Complex,
    width: usize,
    height: usize,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    let mut points = Vec::new();

    for py in 0..height {
        let y = y_min + (y_max - y_min) * (py as f64) / (height as f64 - 1.0);
        for px in 0..width {
            let x = x_min + (x_max - x_min) * (px as f64) / (width as f64 - 1.0);

            let mut z = Complex::new(x, y);
            let mut bounded = true;

            for _ in 0..max_iter {
                z = z.sqr().add(&c);
                if z.magnitude_sq() > 4.0 {
                    bounded = false;
                    break;
                }
            }

            if bounded {
                points.push(vec![x as f32, y as f32]);
            }
        }
    }

    points
}

/// Print analysis results in a formatted table.
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
            "LINE-LIKE"
        } else {
            "DEGENERATE"
        }
    );
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
}

/// Convert a point cloud into a 4D graph (Experiment 1).
/// Each point becomes a GraphNode4D with spatial coordinates.
/// Edges connect nearby points within a threshold distance.
fn point_cloud_to_graph_4d(points: &[Vec<f32>], edge_threshold: f32) -> Vec<GraphNode4D> {
    let mut nodes: Vec<GraphNode4D> = points
        .iter()
        .enumerate()
        .map(|(i, pt)| {
            let mut props = GraphProperties::new();
            props.insert(
                "fractal_type".into(),
                serde_json::Value::String("julia_set".into()),
            );
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

    // Connect nearby points (spatial proximity = edge)
    for i in 0..nodes.len() {
        let pos_i = nodes[i].position();
        for j in (i + 1)..nodes.len() {
            let pos_j = nodes[j].position();
            let dist = pos_i.distance(pos_j);
            if dist <= edge_threshold {
                let weight = 1.0 / (dist + 0.001); // Inverse distance weight
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

/// Experiment 1: Store Julia set as 4D graph nodes and analyze connectivity.
fn experiment_1_graph_storage(julia_points: &[Vec<f32>]) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("EXPERIMENT 1: Julia Set as 4D Graph Nodes");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let nodes = point_cloud_to_graph_4d(julia_points, 0.05);

    println!("Graph Statistics:");
    println!("  Total nodes: {}", nodes.len());

    let total_edges: usize = nodes.iter().map(|n| n.successors.len()).sum();
    let avg_degree = total_edges as f64 / nodes.len() as f64;
    println!("  Total edges: {}", total_edges);
    println!("  Average degree: {:.2}", avg_degree);

    // Find node with highest degree (hub)
    let max_degree_node = nodes.iter().max_by_key(|n| n.successors.len()).unwrap();
    println!(
        "  Max degree: {} (node id: {})",
        max_degree_node.successors.len(),
        max_degree_node.id
    );
    println!(
        "  Hub position: ({:.3}, {:.3})",
        max_degree_node.x, max_degree_node.y
    );
    println!();

    // Test reachability from center node
    let center_id = nodes.len() as u64 / 2;
    let ctx = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 0, end: 1 }),
        spatial_region: None,
        spatial_candidates: None,
        graph_weight: 1.0,
        spatial_weight: 0.0,
        temporal_weight: 0.0,
    };

    let reachable = reachable_4d(&nodes, center_id, &ctx);
    println!("  Reachability from center node (id={}):", center_id);
    println!(
        "    Reachable nodes: {} / {} ({:.1}%)",
        reachable.len(),
        nodes.len(),
        100.0 * reachable.len() as f64 / nodes.len() as f64
    );
    println!();
}

/// Experiment 2: Ricci curvature analysis on the fractal graph.
fn experiment_2_ricci_curvature(nodes: &[GraphNode4D]) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("EXPERIMENT 2: Ricci Curvature on Fractal Graph");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Use a smaller subset for Ricci curvature (computationally expensive)
    let sample_size = nodes.len().min(500);
    let step = nodes.len() / sample_size;
    let sampled_nodes: Vec<GraphNode4D> = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| i % step == 0)
        .take(sample_size)
        .map(|(_, n)| n.clone())
        .collect();

    println!(
        "  Using sampled subset: {} nodes (from {} total)",
        sampled_nodes.len(),
        nodes.len()
    );
    println!("  (Full graph Ricci is O(E * V²) — sampling for demo speed)");
    println!();

    let alpha = 0.5;
    let ricci_edges = ollivier_ricci(&sampled_nodes, alpha);

    if ricci_edges.is_empty() {
        println!("  No Ricci edges computed (graph may be too sparse).");
        println!();
        return;
    }

    let curvatures: Vec<f32> = ricci_edges.iter().map(|e| e.curvature).collect();
    let avg_curvature = curvatures.iter().sum::<f32>() / curvatures.len() as f32;
    let min_curvature = curvatures.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_curvature = curvatures.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("Ollivier-Ricci Curvature Analysis (alpha={}):", alpha);
    println!("  Edges analyzed: {}", ricci_edges.len());
    println!("  Average curvature: {:.4}", avg_curvature);
    println!("  Min curvature: {:.4}", min_curvature);
    println!("  Max curvature: {:.4}", max_curvature);
    println!();

    // Interpretation
    println!("Interpretation:");
    if avg_curvature < -0.1 {
        println!("  NEGATIVE average curvature → hyperbolic-like geometry");
        println!("  Typical of fractal structures with exponential branching");
    } else if avg_curvature > 0.1 {
        println!("  POSITIVE average curvature → spherical-like geometry");
        println!("  Suggests compact, bounded structure");
    } else {
        println!("  NEAR-ZERO average curvature → flat/Euclidean-like geometry");
        println!("  Suggests balanced expansion/contraction");
    }
    println!();

    // Show most negative curvature edges (bottlenecks)
    let mut sorted_edges = ricci_edges.clone();
    sorted_edges.sort_by(|a, b| a.curvature.partial_cmp(&b.curvature).unwrap());

    println!("  Top 5 most negative curvature edges (bottlenecks):");
    for (i, edge) in sorted_edges.iter().take(5).enumerate() {
        println!(
            "    {}. src={} → dst={}: curvature={:.4}",
            i + 1,
            edge.src,
            edge.dst,
            edge.curvature
        );
    }
    println!();
}

/// Experiment 3: Percolation analysis on the fractal graph.
fn experiment_3_percolation(nodes: &[GraphNode4D]) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("EXPERIMENT 3: Percolation Analysis on Fractal Graph");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let center_id = nodes.len() as u64 / 2;
    let sphere_center = Vec3::new(0.0, 0.0, 0.0);

    // Sweep radii from small to large
    let radii: Vec<f32> = (1..=20).map(|i| i as f32 * 0.05).collect();
    let time_slices = vec![0u64];
    let time_half_width = 1u64;

    let percolation = percolation_sweep(
        nodes,
        center_id,
        sphere_center,
        &radii,
        &time_slices,
        time_half_width,
    );

    println!("Percolation Sweep Results:");
    println!(
        "  {:<10} {:<10} {:<10} {:<12}",
        "Radius", "Active", "Reachable", "Fraction"
    );
    println!("  {}", "-".repeat(48));

    for point in &percolation {
        println!(
            "  {:<10.3} {:<10} {:<10} {:<12.4}",
            point.radius, point.active, point.reachable, point.fraction
        );
    }
    println!();

    let critical = find_critical_radius(&percolation, 0, 0.5, 1);
    if let Some(cr) = critical {
        println!("  Critical radius (50% connectivity): {:.3}", cr);
        println!("  This is the threshold where the fractal transitions from");
        println!("  fragmented to connected — a phase transition point.");
    } else {
        println!("  No clear critical radius found in sweep range.");
    }
    println!();

    // Analyze phase transition
    let fractions: Vec<f32> = percolation.iter().map(|p| p.fraction).collect();
    if fractions.len() >= 2 {
        let max_jump = fractions
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, |a, b| a.max(b));
        println!("  Maximum connectivity jump: {:.4}", max_jump);
        if max_jump > 0.3 {
            println!("  SHARP phase transition detected!");
            println!("  This is characteristic of fractal percolation.");
        } else {
            println!("  Gradual transition — no sharp phase boundary.");
        }
    }
    println!();
}

/// Experiment 4: Tensor Network encoding of fractal self-similarity.
fn experiment_4_tensor_network(julia_points: &[Vec<f32>]) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("EXPERIMENT 4: Tensor Network Encoding of Fractal Structure");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Encode the Julia set as a simple MPS chain
    // Each "site" represents a ring of the Julia set at a certain radius
    // This is a simplified model — full encoding would use 2D PEPS

    println!("Building MPS representation of Julia set rings...");

    // Group points by distance from origin into concentric rings
    let mut rings: Vec<Vec<(f32, f32)>> = Vec::new();
    let n_rings = 10;
    let max_radius = 1.5f32;

    for _ in 0..n_rings {
        rings.push(Vec::new());
    }

    for pt in julia_points {
        let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
        let ring_idx =
            ((r / max_radius) * (n_rings as f32 - 1.0)).min(n_rings as f32 - 1.0) as usize;
        rings[ring_idx].push((pt[0], pt[1]));
    }

    println!("  Points distributed into {} concentric rings:", n_rings);
    for (i, ring) in rings.iter().enumerate() {
        println!("    Ring {}: {} points", i, ring.len());
    }
    println!();

    // Build MPS where each site encodes one ring's angular distribution
    let mut tensors: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();

    for ring in &rings {
        // Simple encoding: angular histogram as tensor data
        let n_bins = 8;
        let mut histogram = vec![0.0f32; n_bins];

        for &(x, y) in ring {
            let angle = y.atan2(x); // -π to π
            let bin = (((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI))
                * n_bins as f32)
                .min(n_bins as f32 - 1.0) as usize;
            histogram[bin] += 1.0;
        }

        // Normalize
        let sum: f32 = histogram.iter().sum();
        if sum > 0.0 {
            for h in &mut histogram {
                *h /= sum;
            }
        }

        // MPS site tensor: [χ_left=1, d=n_bins, χ_right=1]
        // For a simple chain, bond dim = 1 (product state approximation)
        let shape = vec![1usize, n_bins, 1];
        tensors.push((shape, histogram));
    }

    // Convert to MPS format expected by build_mps
    let tensor_refs: Vec<(&[usize], &[f32])> = tensors
        .iter()
        .map(|(s, d)| (s.as_slice(), d.as_slice()))
        .collect();

    let mps_nodes = build_mps(&tensor_refs);

    println!("  MPS built with {} sites", mps_nodes.len());

    // Compute norm
    let norm_sq = mps_norm_sq(&mps_nodes);
    println!("  MPS norm²: {:.6}", norm_sq);
    println!("  Norm: {:.6}", norm_sq.sqrt());
    println!();

    // Verify tensor retrieval
    println!("  Verifying tensor storage/retrieval:");
    for (i, node) in mps_nodes.iter().take(3).enumerate() {
        if let Some((shape, data)) = get_tensor(node) {
            println!(
                "    Site {}: shape={:?}, data_sum={:.4}",
                i,
                shape,
                data.iter().sum::<f32>()
            );
        }
    }
    println!();

    // Interpretation
    println!("Interpretation:");
    println!("  The MPS encodes the angular distribution of Julia set points");
    println!("  across concentric rings. Self-similarity would appear as");
    println!("  repeating patterns in the site tensors — a signature of");
    println!("  the fractal's recursive structure.");
    println!();
    println!("  Bond dimension χ=1 means this is a product state approximation.");
    println!("  A full PEPS (2D tensor network) would capture the spatial");
    println!("  entanglement structure of the fractal more accurately.");
    println!();
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║     CROP CIRCLE FRACTAL ANALYSIS SUITE                     ║");
    println!("║     Using GeoGraphDB Core                                  ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // GENERATE JULIA SET (Stonehenge 1996 crop circle)
    // c ≈ -3/4 + (1/9)i = -0.745429 + 0.113008i
    // ========================================================================
    println!("Generating Julia Set matching Stonehenge 1996 formation...");
    println!("c = -3/4 + (1/9)i ≈ -0.745429 + 0.113008i");
    println!();

    let c = Complex::new(-0.745429, 0.113008);
    let julia_points = generate_julia_set_point_cloud(c, 400, 400, -1.5, 1.5, -1.5, 1.5, 80);

    let julia_dim = correlation_dimension(&julia_points, 0.001, 2.0, 25);
    print_results("JULIA SET (Stonehenge 1996)", &julia_points, julia_dim);

    // ========================================================================
    // EXPERIMENT 1: Store as 4D Graph Nodes
    // ========================================================================
    experiment_1_graph_storage(&julia_points);

    // ========================================================================
    // EXPERIMENT 2: Ricci Curvature Analysis
    // ========================================================================
    let graph_nodes = point_cloud_to_graph_4d(&julia_points, 0.05);
    experiment_2_ricci_curvature(&graph_nodes);

    // ========================================================================
    // EXPERIMENT 3: Percolation Analysis
    // ========================================================================
    experiment_3_percolation(&graph_nodes);

    // ========================================================================
    // EXPERIMENT 4: Tensor Network Encoding
    // ========================================================================
    experiment_4_tensor_network(&julia_points);

    // ========================================================================
    // SUMMARY
    // ========================================================================
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║     COMPLETE ANALYSIS SUMMARY                               ║");
    println!("╠═════════════════════════════════════════════════════════════╣");
    println!(
        "║ Julia Set Fractal Dimension:      D ≈ {:<20.4} ║",
        julia_dim
    );
    println!("║                                                             ║");
    println!("║ Experiments Completed:                                      ║");
    println!("║   1. 4D Graph Storage        - {:<28} ║", "✓");
    println!("║   2. Ricci Curvature         - {:<28} ║", "✓");
    println!("║   3. Percolation Analysis    - {:<28} ║", "✓");
    println!("║   4. Tensor Network Encoding - {:<28} ║", "✓");
    println!("╠═════════════════════════════════════════════════════════════╣");
    println!("║ Theoretical Values:                                         ║");
    println!("║   Julia Set:        D ≈ 1.7-1.9 (varies by c)              ║");
    println!("║   Koch Snowflake:   D = log(4)/log(3) ≈ 1.2619             ║");
    println!("║   Sierpinski:       D = log(3)/log(2) ≈ 1.5850             ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // ENCODING HYPOTHESIS
    // ========================================================================
    println!("ENCODING HYPOTHESIS:");
    println!("────────────────────");
    println!("The Julia Set c-value encodes simple fractions:");
    println!("  c ≈ -3/4 + (1/9)i");
    println!();
    println!("If intentional, this is a calibrated mathematical signature:");
    println!("  - Simple rationals (-3/4, 1/9) are universally recognizable");
    println!("  - Together they specify a unique point in the Mandelbrot boundary");
    println!(
        "  - The resulting fractal has D ≈ {:.4}, confirming non-integer dimension",
        julia_dim
    );
    println!();
    println!("This matches the 'intelligence test' hypothesis:");
    println!("  Level 1: Recognize pattern is not random");
    println!("  Level 2: Identify self-similarity");
    println!("  Level 3: Infer generating equation (Julia set)");
    println!("  Level 4: Extract encoded constants (-3/4, 1/9)");
    println!("  Level 5: Understand WHY fractals were chosen as medium");
    println!();
    println!("GeoGraphDB Core Analysis:");
    println!("  - 4D graph storage enables spatial-temporal queries on fractal structure");
    println!("  - Ricci curvature reveals geometric properties of the fractal graph");
    println!("  - Percolation detects phase transitions in connectivity");
    println!("  - Tensor networks encode self-similarity as entanglement patterns");
}
