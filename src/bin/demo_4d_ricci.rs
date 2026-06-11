//! Ollivier-Ricci curvature experiment on 4D tensor-network graphs.
//!
//! Computes κ(u,v) = 1 − W₁(μ_u^α, μ_v^α) / d(u,v) for every spatial and
//! temporal edge in several tnet4d grids.  W₁ is solved exactly via the
//! transportation simplex (North-West corner + MODI + stepping-stone).
//!
//! Geometry interpretation:
//!   κ > 0  — locally positively curved (like a sphere)
//!   κ = 0  — locally flat (like Euclidean space)
//!   κ < 0  — locally negatively curved (like a hyperbolic plane)
//!
//! Usage:
//!   cargo run --example demo_4d_ricci

use geographdb_core::{build_tnet4d, ollivier_ricci, GraphNode4D};

fn spatial_edge_count(nodes: &[GraphNode4D]) -> usize {
    nodes
        .iter()
        .flat_map(|n| n.successors.iter())
        .filter(|e| e.begin_ts == e.end_ts)
        .count()
        / 2 // undirected
}

fn temporal_edge_count(nodes: &[GraphNode4D]) -> usize {
    nodes
        .iter()
        .flat_map(|n| n.successors.iter())
        .filter(|e| e.end_ts > e.begin_ts)
        .count()
}

fn report(label: &str, nodes: &[GraphNode4D], alpha: f32) {
    let edges = ollivier_ricci(nodes, alpha);
    let n_edges = edges.len();
    if n_edges == 0 {
        println!("{label},0,0,0,0,0,0");
        return;
    }

    let curvatures: Vec<f32> = edges.iter().map(|e| e.curvature).collect();
    let mean = curvatures.iter().sum::<f32>() / n_edges as f32;
    let min = curvatures.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = curvatures.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let n_positive = curvatures.iter().filter(|&&k| k > 1e-4).count();
    let n_zero = curvatures.iter().filter(|&&k| k.abs() <= 1e-4).count();
    let n_negative = curvatures.iter().filter(|&&k| k < -1e-4).count();

    println!(
        "{label},{alpha},{n_edges},{mean:.4},{min:.4},{max:.4},{n_positive},{n_zero},{n_negative}"
    );
    eprintln!(
        "  {label} α={alpha}: edges={n_edges} mean_κ={mean:.4} \
         min={min:.4} max={max:.4} (+{n_positive} 0{n_zero} -{n_negative})"
    );
}

fn main() {
    println!(
        "# Ollivier-Ricci curvature — geographdb-core v{}",
        geographdb_core::VERSION
    );
    println!("experiment,alpha,n_edges,mean_kappa,min_kappa,max_kappa,n_pos,n_zero,n_neg");

    eprintln!("Ollivier-Ricci experiments (α=0.5):");

    // ── 2×2×1 grid, depth 1 (C4 cycle) ──────────────────────────────────────
    // Every edge of a 4-cycle has κ = 0.5 with α = 0.5 (positive curvature).
    let g1 = build_tnet4d(2, 2, 1, 1);
    report("2x2x1_d1", &g1, 0.5);

    // ── 3×3×1 grid, depth 1 (torus-like patch) ───────────────────────────────
    let g2 = build_tnet4d(3, 3, 1, 1);
    report("3x3x1_d1", &g2, 0.5);
    report("3x3x1_d1_alpha0", &g2, 0.0);

    // ── 2×2×1 grid, depth 3 (temporal edges add shortcuts) ───────────────────
    let g3 = build_tnet4d(2, 2, 1, 3);
    report("2x2x1_d3", &g3, 0.5);

    // ── 2×2×2 grid, depth 2 (3D spatial) ─────────────────────────────────────
    let g4 = build_tnet4d(2, 2, 2, 2);
    report("2x2x2_d2", &g4, 0.5);

    // ── 4×4×1 grid, depth 1 (larger 2D lattice) ──────────────────────────────
    let g5 = build_tnet4d(4, 4, 1, 1);
    report("4x4x1_d1", &g5, 0.5);

    // ── Detailed per-edge dump for the 2×2×1 d=1 graph ───────────────────────
    eprintln!("\nPer-edge Ricci curvature for 2×2×1 depth=1:");
    let edges1 = ollivier_ricci(&g1, 0.5);
    for e in &edges1 {
        let kind = if g1
            .iter()
            .find(|n| n.id == e.src)
            .and_then(|n| n.successors.iter().find(|ed| ed.dst == e.dst))
            .map(|ed| ed.begin_ts == ed.end_ts)
            .unwrap_or(false)
        {
            "spatial"
        } else {
            "temporal"
        };
        eprintln!(
            "  ({} → {}) {}  W₁={:.4}  κ={:.4}",
            e.src, e.dst, kind, e.w1, e.curvature
        );
    }

    // ── Summary statistics ────────────────────────────────────────────────────
    eprintln!("\nSummary (spatial edge counts):");
    for (label, nodes) in [
        ("2x2x1 d=1", &g1),
        ("3x3x1 d=1", &g2),
        ("2x2x1 d=3", &g3),
        ("2x2x2 d=2", &g4),
        ("4x4x1 d=1", &g5),
    ] {
        eprintln!(
            "  {label}: spatial_edges={} temporal_edges={}",
            spatial_edge_count(nodes),
            temporal_edge_count(nodes)
        );
    }
}
