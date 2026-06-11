//! Causal sets / interval algebra on 4D spatiotemporal graphs.
//!
//! Treats directed temporal edges (end_ts > begin_ts) as the causal order.
//! For each causally-related pair (u,v), computes:
//!   - Alexandrov interval I[u,v] = {w : u →* w →* v} (inclusive)
//!   - Volume |I[u,v]|
//!   - Proper time τ(u,v) = longest directed path length from u to v
//!
//! Also estimates the Myrheim-Meyer spacetime dimension from the log-log
//! regression slope of interval volume vs proper time:
//!   d̂ ≈ slope of ln|I| vs ln(τ)
//!
//! Structures tested:
//!   1. Pure 1D chain → d̂ ≈ 0.8 (finite-size corrected 1D)
//!   2. tnet4d(1,1,1,8): single temporal chain → d̂ ≈ 1D
//!   3. tnet4d(2,2,1,4): 4 parallel chains → d̂ ≈ 1D (no cross-site causal links)
//!   4. 2D causal lattice (light cone spread=1) → d̂ ≈ 1.8 (approaches 2D)
//!
//! Usage:
//!   cargo run --example demo_4d_causal

use geographdb_core::{
    build_tnet4d, causal_intervals, causal_stats, CausalStats, GraphNode4D, GraphProperties,
    TemporalEdge,
};

fn print_stats(label: &str, stats: &CausalStats) {
    println!(
        "{label},{},{},{:.4},{:.4},{:.4}",
        stats.n_nodes,
        stats.n_related_pairs,
        stats.mean_proper_time,
        stats.mean_volume,
        stats.mm_dimension
    );
    eprintln!(
        "  {label}: N={} pairs={} τ̄={:.3} V̄={:.3} d̂={:.3}",
        stats.n_nodes,
        stats.n_related_pairs,
        stats.mean_proper_time,
        stats.mean_volume,
        stats.mm_dimension
    );
}

/// Build a 2D causal lattice: node (x,t) → (x,t+1), (x±1,t+1) (light cone spread=1).
/// Node id = t * nx + x. Temporal edges only (begin_ts=t, end_ts=t+1).
fn build_causal_grid_2d(nx: usize, nt: usize) -> Vec<GraphNode4D> {
    let mut nodes = Vec::with_capacity(nx * nt);
    for t in 0..nt {
        for x in 0..nx {
            let id = (t * nx + x) as u64;
            let mut succs = Vec::new();
            if t + 1 < nt {
                for dx in [-1i32, 0, 1] {
                    let nx2 = x as i32 + dx;
                    if nx2 >= 0 && nx2 < nx as i32 {
                        succs.push(TemporalEdge {
                            dst: ((t + 1) * nx + nx2 as usize) as u64,
                            weight: 1.0,
                            begin_ts: t as u64,
                            end_ts: (t + 1) as u64,
                        });
                    }
                }
            }
            nodes.push(GraphNode4D {
                id,
                x: x as f32,
                y: t as f32,
                z: 0.0,
                begin_ts: t as u64,
                end_ts: (t + 1) as u64,
                properties: GraphProperties::new(),
                successors: succs,
            });
        }
    }
    nodes
}

fn main() {
    println!(
        "# Causal sets / interval algebra — geographdb-core v{}",
        geographdb_core::VERSION
    );
    println!("experiment,n_nodes,n_pairs,mean_tau,mean_volume,mm_dim");
    eprintln!("Causal set experiments:");

    // ── 1D chain (manual) ────────────────────────────────────────────────────
    let chain10: Vec<GraphNode4D> = (0u64..10)
        .map(|i| GraphNode4D {
            id: i,
            x: i as f32,
            y: 0.0,
            z: 0.0,
            begin_ts: i,
            end_ts: i + 1,
            properties: GraphProperties::new(),
            successors: if i + 1 < 10 {
                vec![TemporalEdge {
                    dst: i + 1,
                    weight: 1.0,
                    begin_ts: i,
                    end_ts: i + 1,
                }]
            } else {
                vec![]
            },
        })
        .collect();
    let s1 = causal_stats(&chain10);
    print_stats("1d_chain_10", &s1);

    // ── tnet4d single site ───────────────────────────────────────────────────
    let g_single = build_tnet4d(1, 1, 1, 8);
    let s2 = causal_stats(&g_single);
    print_stats("tnet4d_1x1x1_d8", &s2);

    // ── tnet4d 4 parallel chains ─────────────────────────────────────────────
    let g_parallel = build_tnet4d(2, 2, 1, 5);
    let s3 = causal_stats(&g_parallel);
    print_stats("tnet4d_2x2x1_d5", &s3);

    // ── 2D causal lattice ────────────────────────────────────────────────────
    let g_2d = build_causal_grid_2d(7, 7);
    let s4 = causal_stats(&g_2d);
    print_stats("causal_grid_2d_7x7", &s4);

    // ── Per-interval detail for small 2D lattice ─────────────────────────────
    eprintln!("\nInterval detail for 2D causal lattice 5×5 (src=(2,0)→dst=(2,τ)):");
    let g_small = build_causal_grid_2d(5, 6);
    let intervals = causal_intervals(&g_small);
    eprintln!(
        "  {:>6}  {:>6}  {:>8}  {:>8}",
        "src", "dst", "volume", "tau"
    );
    let mut src2: Vec<_> = intervals.iter().filter(|e| e.src == 2).collect();
    src2.sort_by_key(|e| (e.proper_time, e.dst));
    for e in src2.iter().take(10) {
        eprintln!(
            "  {:>6}  {:>6}  {:>8}  {:>8}",
            e.src, e.dst, e.volume, e.proper_time
        );
    }

    // ── Dimension interpretation ─────────────────────────────────────────────
    eprintln!("\nMyrheim-Meyer dimension interpretation:");
    eprintln!("  d̂ = log-log regression slope of |I[u,v]| vs τ(u,v) for τ ≥ 2.");
    eprintln!("  1D chain: V = τ+1 → d̂ ≈ 0.8 (finite lattice; → 1 as N→∞).");
    eprintln!("  2D causal lattice: V ≈ τ²/2 → d̂ → 2 as N→∞; ~1.4 at 7×7 (boundary effects).");
    eprintln!("  tnet4d parallel chains: no cross-site causal links → d̂ ≈ 1D.");
}
