//! 4D Tensor Network experiment.
//!
//! Builds a (nx × ny × nz) × depth spatiotemporal tensor network where:
//!   - Spatial edges connect neighbouring qubits within the same time layer
//!   - Temporal edges connect the same qubit across adjacent time layers
//!
//! This is the structure used to simulate quantum circuits on a 3D lattice —
//! space in (x,y,z), circuit depth along the time axis.
//!
//! Experiments:
//!   1. 2×2×1 grid, 1 layer  — product state |0000⟩
//!   2. 2×2×1 grid, 3 layers — circuit depth 3, single-site gates each layer
//!   3. 2×2×2 grid, 2 layers — 3D spatial layout with temporal evolution
//!   4. 3×3×1 grid, 2 layers — larger 2D circuit (Google Sycamore-style layout)
//!
//! Output: CSV + graph structure summary to stderr.
//!
//! Usage:
//!   cargo run --example demo_4d_tnet

use geographdb_core::{
    build_tnet4d, get_tensor, mps_norm_sq, site_id, tnet4d_apply_gate_1site,
    tnet4d_apply_gate_2site, tnet4d_find, tnet4d_norm_sq, GraphNode4D, GridDims, SiteCoord,
};

fn hadamard() -> Vec<f32> {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    vec![s, s, s, -s]
}

fn pauli_x() -> Vec<f32> {
    vec![0.0, 1.0, 1.0, 0.0]
}

fn report(label: &str, nodes: &[GraphNode4D], nx: usize, ny: usize, nz: usize, depth: usize) {
    let norm = tnet4d_norm_sq(nodes, nx, ny, nz, depth);
    let n_spatial = nx * ny * nz;
    let n_nodes = nodes.len();
    let n_spatial_edges: usize = nodes
        .iter()
        .flat_map(|n| n.successors.iter())
        .filter(|e| e.begin_ts == e.end_ts)
        .count();
    let n_temporal_edges: usize = nodes
        .iter()
        .flat_map(|n| n.successors.iter())
        .filter(|e| e.end_ts > e.begin_ts)
        .count();
    println!(
        "{label},{nx}x{ny}x{nz},{depth},{n_nodes},{n_spatial_edges},{n_temporal_edges},{norm:.8}"
    );
    eprintln!(
        "  {label}: sites={n_spatial} depth={depth} nodes={n_nodes} \
         spatial_edges={n_spatial_edges} temporal_edges={n_temporal_edges} norm={norm:.6}"
    );
}

fn main() {
    println!(
        "# 4D Tensor Network — geographdb-core v{}",
        geographdb_core::VERSION
    );
    println!("experiment,grid,depth,n_nodes,spatial_edges,temporal_edges,norm_sq");

    eprintln!("4D Tensor Network experiments:");

    // ── Experiment 1: 2×2×1, depth 1 ────────────────────────────────────────
    let nodes1 = build_tnet4d(2, 2, 1, 1);
    report("2x2x1_d1_product", &nodes1, 2, 2, 1, 1);

    // ── Experiment 2: 2×2×1, depth 3 — Hadamard circuit ────────────────────
    let mut nodes2 = build_tnet4d(2, 2, 1, 3);
    let h = hadamard();
    let dims221 = GridDims {
        nx: 2,
        ny: 2,
        nz: 1,
    };
    // Layer 0: Hadamard on all sites
    for sx in 0..2 {
        for sy in 0..2 {
            tnet4d_apply_gate_1site(&mut nodes2, sx, sy, 0, 0, dims221, &h);
        }
    }
    // Layer 1: Pauli-X on all sites
    let x = pauli_x();
    for sx in 0..2 {
        for sy in 0..2 {
            tnet4d_apply_gate_1site(&mut nodes2, sx, sy, 0, 1, dims221, &x);
        }
    }
    // Layer 2: Hadamard again
    for sx in 0..2 {
        for sy in 0..2 {
            tnet4d_apply_gate_1site(&mut nodes2, sx, sy, 0, 2, dims221, &h);
        }
    }
    report("2x2x1_d3_HXH_circuit", &nodes2, 2, 2, 1, 3);

    // ── Experiment 3: 2×2×2, depth 2 — 3D spatial + time ───────────────────
    let mut nodes3 = build_tnet4d(2, 2, 2, 2);
    let dims222 = GridDims {
        nx: 2,
        ny: 2,
        nz: 2,
    };
    // Apply Hadamard to every site in layer 0
    for sx in 0..2 {
        for sy in 0..2 {
            for sz in 0..2 {
                tnet4d_apply_gate_1site(&mut nodes3, sx, sy, sz, 0, dims222, &h);
            }
        }
    }
    report("2x2x2_d2_H_layer0", &nodes3, 2, 2, 2, 2);

    // ── Experiment 4: 3×3×1, depth 2 — Sycamore-style 2D grid ──────────────
    let mut nodes4 = build_tnet4d(3, 3, 1, 2);
    let dims331 = GridDims {
        nx: 3,
        ny: 3,
        nz: 1,
    };
    // Hadamard on entire layer 0
    for sx in 0..3 {
        for sy in 0..3 {
            tnet4d_apply_gate_1site(&mut nodes4, sx, sy, 0, 0, dims331, &h);
        }
    }
    report("3x3x1_d2_sycamore_style", &nodes4, 3, 3, 1, 2);

    // ── Graph structure dump for experiment 1 ────────────────────────────────
    eprintln!("\n2×2×1 depth=1 graph structure:");
    for node in &nodes1 {
        let (shape, _) = get_tensor(node).unwrap();
        let spatial: Vec<String> = node
            .successors
            .iter()
            .filter(|e| e.begin_ts == e.end_ts)
            .map(|e| format!("→{}", e.dst))
            .collect();
        let temporal: Vec<String> = node
            .successors
            .iter()
            .filter(|e| e.end_ts > e.begin_ts)
            .map(|e| format!("→{}", e.dst))
            .collect();
        eprintln!(
            "  node {} ({:.0},{:.0},{:.0}) t=[{},{}]  tensor[{},{},{}]  \
             spatial:{:?}  temporal:{:?}",
            node.id,
            node.x,
            node.y,
            node.z,
            node.begin_ts,
            node.end_ts,
            shape[0],
            shape[1],
            shape[2],
            spatial,
            temporal
        );
    }

    // ── Show the 2×2×1 depth=2 temporal connections ──────────────────────────
    eprintln!("\n2×2×1 depth=2 temporal backbone:");
    let nodes_d2 = build_tnet4d(2, 2, 1, 2);
    for sx in 0..2usize {
        for sy in 0..2usize {
            let id0 = site_id(sx, sy, 0, 0, 2, 2, 1);
            let id1 = site_id(sx, sy, 0, 1, 2, 2, 1);
            let idx0 = tnet4d_find(&nodes_d2, sx, sy, 0, 0, dims221).unwrap();
            let has_edge = nodes_d2[idx0].successors.iter().any(|e| e.dst == id1);
            eprintln!(
                "  ({sx},{sy},0) layer0[id={id0}] ──temporal──▶ layer1[id={id1}]  edge_exists={has_edge}"
            );
        }
    }

    // ── Experiment 5: 2-site CNOT — Bell pair between spatial neighbours ──────
    eprintln!("\n── Experiment 5: two-site CNOT gate (Bell pair creation) ──");
    let cnot: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ];

    // Step A: product state |+,0⟩ on (0,0,0) and (1,0,0)
    let mut nodes5 = build_tnet4d(2, 2, 1, 1);
    tnet4d_apply_gate_1site(&mut nodes5, 0, 0, 0, 0, dims221, &h);
    let idx_a = tnet4d_find(&nodes5, 0, 0, 0, 0, dims221).unwrap();
    let idx_b = tnet4d_find(&nodes5, 1, 0, 0, 0, dims221).unwrap();
    let (sa_before, _) = get_tensor(&nodes5[idx_a]).unwrap();
    let (sb_before, _) = get_tensor(&nodes5[idx_b]).unwrap();
    eprintln!(
        "  Before CNOT: site(0,0,0) bond_dims=[{},{},{}]  site(1,0,0) bond_dims=[{},{},{}]",
        sa_before[0], sa_before[1], sa_before[2], sb_before[0], sb_before[1], sb_before[2],
    );

    // Step B: apply CNOT — (0,0,0) is control, (1,0,0) is target
    let chi_new = tnet4d_apply_gate_2site(
        &mut nodes5,
        SiteCoord {
            sx: 0,
            sy: 0,
            sz: 0,
        },
        SiteCoord {
            sx: 1,
            sy: 0,
            sz: 0,
        },
        0,
        dims221,
        &cnot,
        2,
    );
    let (sa_after, _) = get_tensor(&nodes5[idx_a]).unwrap();
    let (sb_after, _) = get_tensor(&nodes5[idx_b]).unwrap();
    eprintln!(
        "  After CNOT:  site(0,0,0) bond_dims=[{},{},{}]  site(1,0,0) bond_dims=[{},{},{}]  chi_new={chi_new}",
        sa_after[0], sa_after[1], sa_after[2],
        sb_after[0], sb_after[1], sb_after[2],
    );

    // Step C: norm of the Bell pair (column-by-column is not valid for entangled
    // states, but we can compute it for the two-site slice directly via mps_norm_sq).
    let pair = vec![nodes5[idx_a].clone(), nodes5[idx_b].clone()];
    let bell_norm = mps_norm_sq(&pair);
    eprintln!("  Bell pair norm = {bell_norm:.6}  (expected 1.0)");
    println!("2x2x1_d1_cnot_bell,2x2x1,1,4,{chi_new},{bell_norm:.8}");
}
