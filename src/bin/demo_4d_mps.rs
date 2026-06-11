//! Matrix Product State experiment on the 4D graph.
//!
//! Demonstrates that quantum states can be represented and manipulated as
//! chains of GraphNode4D nodes, with tensor data stored in node properties
//! and bond dimensions encoded as edge weights.
//!
//! Experiments:
//!   1. Product states |0000⟩, |1111⟩ — norm = 1.0
//!   2. Hadamard circuit — apply H to every qubit → |++++⟩
//!   3. Bell state (|00⟩ + |11⟩)/√2 — entangled, bond dim 2, norm = 1.0
//!   4. GHZ state (|0000⟩ + |1111⟩)/√2 — max entanglement across 4 qubits
//!
//! Output: one line per experiment, CSV format:
//!   experiment,n_sites,bond_dim,norm_sq,description
//!
//! Usage:
//!   cargo run --example demo_4d_mps

use geographdb_core::{build_mps, get_tensor, mps_apply_gate, mps_norm_sq, GraphNode4D};

fn print_header() {
    println!(
        "# MPS experiment — geographdb-core v{}",
        geographdb_core::VERSION
    );
    println!("experiment,n_sites,max_bond_dim,norm_sq,description");
}

fn max_bond_dim(nodes: &[GraphNode4D]) -> usize {
    nodes
        .iter()
        .flat_map(|n| n.successors.iter().map(|e| e.weight as usize))
        .max()
        .unwrap_or(1)
}

fn report(label: &str, nodes: &[GraphNode4D], desc: &str) {
    let n = nodes.len();
    let bond = max_bond_dim(nodes);
    let norm = mps_norm_sq(nodes);
    println!("{label},{n},{bond},{norm:.8},{desc}");
    eprintln!("  {label}: n={n} bond={bond} norm_sq={norm:.6}");
}

// ── Tensor constructors ───────────────────────────────────────────────────────

fn ket0() -> ([usize; 3], Vec<f32>) {
    ([1, 2, 1], vec![1.0, 0.0])
}

fn ket1() -> ([usize; 3], Vec<f32>) {
    ([1, 2, 1], vec![0.0, 1.0])
}

fn hadamard() -> Vec<f32> {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    vec![s, s, s, -s]
}

// ── Experiments ───────────────────────────────────────────────────────────────

fn exp_product_state_0000() -> Vec<GraphNode4D> {
    let (s, d) = ket0();
    build_mps(&[(&s, &d), (&s, &d), (&s, &d), (&s, &d)])
}

fn exp_product_state_1111() -> Vec<GraphNode4D> {
    let (s, d) = ket1();
    build_mps(&[(&s, &d), (&s, &d), (&s, &d), (&s, &d)])
}

fn exp_hadamard_circuit() -> Vec<GraphNode4D> {
    let (s, d) = ket0();
    let mut nodes = build_mps(&[(&s, &d), (&s, &d), (&s, &d), (&s, &d)]);
    let h = hadamard();
    for i in 0..4 {
        mps_apply_gate(&mut nodes, i, &h);
    }
    nodes
}

fn exp_bell_state() -> Vec<GraphNode4D> {
    // |Φ+⟩ = (|00⟩ + |11⟩)/√2
    // A0[1,2,2]: A0[0,0,0]=1/√2, A0[0,1,1]=1/√2
    // A1[2,2,1]: A1[0,0,0]=1, A1[1,1,0]=1
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let a0_shape = [1usize, 2, 2];
    let a0_data = vec![s, 0.0, 0.0, s];
    let a1_shape = [2usize, 2, 1];
    let a1_data = vec![1.0, 0.0, 0.0, 1.0];
    build_mps(&[(&a0_shape, &a0_data), (&a1_shape, &a1_data)])
}

fn exp_ghz_state() -> Vec<GraphNode4D> {
    // GHZ = (|0000⟩ + |1111⟩)/√2
    // Bond dim 2 throughout: each tensor threads one "branch" for 0s and one for 1s.
    // A0[1,2,2]:  A0[0,0,0]=1/√2, A0[0,1,1]=1/√2
    // A1[2,2,2]:  A1[0,0,0]=1, A1[1,1,1]=1  (pass-through)
    // A2[2,2,2]:  same as A1
    // A3[2,2,1]:  A3[0,0,0]=1, A3[1,1,0]=1  (merge back to bond 1)
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let a0 = ([1usize, 2, 2], vec![s, 0.0, 0.0, s]);
    let a_mid = ([2usize, 2, 2], vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let a_last = ([2usize, 2, 1], vec![1.0, 0.0, 0.0, 1.0]);
    build_mps(&[
        (&a0.0, &a0.1),
        (&a_mid.0, &a_mid.1),
        (&a_mid.0, &a_mid.1),
        (&a_last.0, &a_last.1),
    ])
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    print_header();

    eprintln!("MPS experiments:");
    report("product_0000", &exp_product_state_0000(), "|0000>");
    report("product_1111", &exp_product_state_1111(), "|1111>");
    report(
        "hadamard_circuit",
        &exp_hadamard_circuit(),
        "H⊗H⊗H⊗H |0000> = |++++>",
    );
    report("bell_state", &exp_bell_state(), "(|00>+|11>)/sqrt(2)");
    report("ghz_state", &exp_ghz_state(), "(|0000>+|1111>)/sqrt(2)");

    // Show graph structure for the Bell state
    eprintln!("\nBell state graph structure:");
    let bell = exp_bell_state();
    for node in &bell {
        let (shape, _) = get_tensor(node).unwrap();
        eprintln!(
            "  node {} at x={:.0}  tensor shape [{},{},{}]  edges: {:?}",
            node.id,
            node.x,
            shape[0],
            shape[1],
            shape[2],
            node.successors
                .iter()
                .map(|e| format!("→{} (bond={})", e.dst, e.weight))
                .collect::<Vec<_>>()
        );
    }
}
