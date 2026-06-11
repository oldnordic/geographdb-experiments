//! Information geometry experiment on 4D tensor-network graphs.
//!
//! For each edge (u,v) in several tnet4d grids, computes:
//!   - Shannon entropy H(μ_u^α) at each node
//!   - KL divergence D_KL(μ_u ∥ μ_v) and D_KL(μ_v ∥ μ_u)
//!   - Jensen-Shannon divergence JSD(μ_u, μ_v) ∈ [0, ln 2]
//!   - Fisher-Rao geodesic distance d_FR = 2·arccos(Σ√(p·q)) ∈ [0, π]
//!
//! Also compares Fisher-Rao distance (information geometry) vs W₁ (optimal
//! transport / Ricci curvature) to show how the two views of local curvature
//! relate across different graph structures.
//!
//! Usage:
//!   cargo run --example demo_4d_infogeo

use geographdb_core::{build_tnet4d, info_geometry, ollivier_ricci};

fn report_summary(label: &str, nodes: &[geographdb_core::GraphNode4D], alpha: f32) {
    let (info_nodes, info_edges) = info_geometry(nodes, alpha);
    let ricci_edges = ollivier_ricci(nodes, alpha);

    if info_edges.is_empty() {
        return;
    }

    let mean_entropy = info_nodes.iter().map(|n| n.entropy).sum::<f32>() / info_nodes.len() as f32;
    let mean_fr = info_edges.iter().map(|e| e.fisher_rao).sum::<f32>() / info_edges.len() as f32;
    let mean_js = info_edges.iter().map(|e| e.js_div).sum::<f32>() / info_edges.len() as f32;
    let mean_kl_sym = info_edges
        .iter()
        .map(|e| (e.kl_uv + e.kl_vu) * 0.5)
        .sum::<f32>()
        / info_edges.len() as f32;

    // Mean Ricci curvature for comparison
    let mean_kappa = if ricci_edges.is_empty() {
        f32::NAN
    } else {
        ricci_edges.iter().map(|e| e.curvature).sum::<f32>() / ricci_edges.len() as f32
    };

    println!(
        "{label},{alpha},{},{mean_entropy:.4},{mean_fr:.4},{mean_js:.4},{mean_kl_sym:.4},{mean_kappa:.4}",
        info_edges.len()
    );
    eprintln!(
        "  {label} α={alpha}: H̄={mean_entropy:.4}  d̄_FR={mean_fr:.4}  \
         JSD̄={mean_js:.4}  KL̄_sym={mean_kl_sym:.4}  κ̄={mean_kappa:.4}"
    );
}

fn main() {
    println!(
        "# Information geometry — geographdb-core v{}",
        geographdb_core::VERSION
    );
    println!("experiment,alpha,n_edges,mean_H,mean_FR,mean_JSD,mean_KL_sym,mean_kappa");

    eprintln!("Information geometry experiments:");

    // ── 2×2×1 grid, depth 1 (C4) ─────────────────────────────────────────────
    let g1 = build_tnet4d(2, 2, 1, 1);
    report_summary("2x2x1_d1", &g1, 0.5);
    report_summary("2x2x1_d1_a0", &g1, 0.0);

    // ── 3×3×1 grid, depth 1 ──────────────────────────────────────────────────
    let g2 = build_tnet4d(3, 3, 1, 1);
    report_summary("3x3x1_d1", &g2, 0.5);

    // ── 2×2×1 depth 3 (temporal) ─────────────────────────────────────────────
    let g3 = build_tnet4d(2, 2, 1, 3);
    report_summary("2x2x1_d3", &g3, 0.5);

    // ── 4×4×1 grid ───────────────────────────────────────────────────────────
    let g4 = build_tnet4d(4, 4, 1, 1);
    report_summary("4x4x1_d1", &g4, 0.5);

    // ── Per-edge detail for C4 ───────────────────────────────────────────────
    eprintln!("\nPer-edge detail for 2×2×1 depth=1 (C4), α=0.5:");
    let (_, edges) = info_geometry(&g1, 0.5);
    let ricci = ollivier_ricci(&g1, 0.5);
    eprintln!(
        "  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "src", "dst", "H_src", "H_dst", "d_FR", "JSD", "κ"
    );
    for e in &edges {
        let kappa = ricci
            .iter()
            .find(|r| r.src == e.src && r.dst == e.dst)
            .map(|r| r.curvature)
            .unwrap_or(f32::NAN);
        eprintln!(
            "  {:>6}  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            e.src, e.dst, e.entropy_src, e.entropy_dst, e.fisher_rao, e.js_div, kappa
        );
    }

    // ── Fisher-Rao vs Ricci correlation note ─────────────────────────────────
    eprintln!("\nFisher-Rao vs Ricci curvature (C4):");
    eprintln!("  d_FR measures geodesic distance on the probability simplex.");
    eprintln!("  κ     measures optimal-transport curvature (W₁-based).");
    eprintln!("  For the C4: both detect the same uniform positive curvature,");
    eprintln!("  but d_FR is a pure information-geometric quantity while κ");
    eprintln!("  encodes metric structure of the graph (distances, not just measures).");

    // ── Entropy field across the 3×3 grid ────────────────────────────────────
    eprintln!("\nEntropy field for 3×3×1 depth=1, α=0.5:");
    let (nodes3, _) = info_geometry(&g2, 0.5);
    for n in &nodes3 {
        eprintln!(
            "  node {}  H={:.4}  (deg={})",
            n.id,
            n.entropy,
            g2.iter()
                .find(|nd| nd.id == n.id)
                .map(|nd| nd.successors.len())
                .unwrap_or(0)
        );
    }
    eprintln!("  (interior nodes deg=4 → higher entropy; corner nodes deg=2 → lower entropy)");
}
