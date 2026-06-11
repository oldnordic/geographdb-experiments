//! Sparse Attention via Octree + Causal Masking
//! -----------------------------------------------
//! Demonstrates embedding tokens in 3D concept space and computing
//! O(n log n) sparse causal attention using the octree.
//!
//! Run with:
//!   cargo run --example demo_sparse_attn

use geographdb_core::algorithms::sparse_attn::{attention_stats, sparse_attention, Token};
use glam::Vec3;

fn main() {
    println!("=======================================================================");
    println!("    SPARSE ATTENTION: OCTREE NEAREST-NEIGHBOR + CAUSAL MASKING        ");
    println!("=======================================================================");
    println!("-> Tokens embedded in 3D concept space. Octree gives O(n log n) KNN.");
    println!("-> Causal mask excludes keys with time_step > query.time_step.");

    // ── Scenario: 16 tokens arranged on two semantic axes ────────────────────
    // Tokens 0-7  : "subject" cluster near (0, 0, 0)
    // Tokens 8-15 : "predicate" cluster near (1, 0, 0)
    // Time increases linearly so earlier tokens can only attend to prior ones.
    let n = 16usize;
    let tokens: Vec<Token> = (0..n)
        .map(|i| {
            let cluster_offset = if i < 8 { 0.0f32 } else { 1.0f32 };
            let jitter = (i % 8) as f32 * 0.1;
            Token {
                id: i as u64,
                position: Vec3::new(cluster_offset + jitter * 0.05, jitter, 0.0),
                time_step: i as u64,
            }
        })
        .collect();

    // ── Experiment A: Dense baseline (k = n, no masking except causal) ───────
    println!("\n-----------------------------------------------------------------------");
    println!(" EXPERIMENT A: Dense Causal Attention (k = n = 16)");
    println!("-----------------------------------------------------------------------");
    let dense = sparse_attention(&tokens, &tokens, n, 1.0);
    let dense_stats = attention_stats(&dense, n);
    println!("  Total query-key pairs (dense): {}", n * n);
    println!(
        "  Attended pairs after causal mask: {}",
        dense_stats.attended_pairs
    );
    println!(
        "  Sparsity from causal mask alone: {:.1}%",
        dense_stats.sparsity * 100.0
    );
    println!("  -> Lower triangle only: causal mask removes the upper triangle.");

    // ── Experiment B: Sparse attention with k = 4 neighbors ──────────────────
    println!("\n-----------------------------------------------------------------------");
    println!(" EXPERIMENT B: Sparse Causal Attention (k = 4 nearest neighbors)");
    println!("-----------------------------------------------------------------------");
    let k = 4;
    let sparse = sparse_attention(&tokens, &tokens, k, 1.0);
    let sparse_stats = attention_stats(&sparse, n);
    println!(
        "  Attended pairs (octree k={}): {} / {} dense",
        k,
        sparse_stats.attended_pairs,
        n * n
    );
    println!(
        "  Sparsity: {:.1}%  (mean {:.1} keys per query, max {})",
        sparse_stats.sparsity * 100.0,
        sparse_stats.mean_attended,
        sparse_stats.max_attended
    );
    println!(
        "  -> Octree + causal = {:.1}% fewer pairs than dense.",
        sparse_stats.sparsity * 100.0
    );

    // ── Experiment C: Temperature effect on weight distribution ──────────────
    println!("\n-----------------------------------------------------------------------");
    println!(" EXPERIMENT C: Temperature Effect on Attention Sharpness");
    println!("-----------------------------------------------------------------------");
    let query = &tokens[15..16]; // last token — can attend to all prior
    for &temp in &[0.01f32, 0.1, 1.0, 10.0] {
        let result = sparse_attention(query, &tokens, 4, temp);
        if result[0].attended.is_empty() {
            println!("  T={temp:.2}: no causal neighbors");
            continue;
        }
        let max_w = result[0]
            .attended
            .iter()
            .map(|(_, w)| *w)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_w = result[0]
            .attended
            .iter()
            .map(|(_, w)| *w)
            .fold(f32::INFINITY, f32::min);
        println!(
            "  T={temp:.2}: max_weight={max_w:.4}  min_weight={min_w:.4}  \
             ratio={:.1}x  (higher T → flatter distribution)",
            max_w / min_w.max(1e-9)
        );
    }

    println!("\n=======================================================================");
    println!(" SUMMARY");
    println!("=======================================================================");
    println!(
        "  Dense  causal:  {} pairs  (O(n²) = {})",
        dense_stats.attended_pairs,
        n * n
    );
    println!(
        "  Sparse causal:  {} pairs  (k={}, {:.1}x reduction)",
        sparse_stats.attended_pairs,
        k,
        dense_stats.attended_pairs as f32 / sparse_stats.attended_pairs.max(1) as f32
    );
    println!("  -> Octree nearest-neighbor + causal time masking reduces attention");
    println!("     cost from O(n²) toward O(n log n) with minimal fidelity loss.");
    println!("=======================================================================");
}
