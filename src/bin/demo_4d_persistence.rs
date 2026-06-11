//! Temporal persistence barcode experiment.
//!
//! Generates the same random 4D graph as demo_4d_percolation (same seed,
//! same parameters) and sweeps the temporal window across [0, T_MAX] with a
//! fixed sphere radius large enough to include all nodes. The SCC count at
//! each time slice forms the Betti-0 curve; birth/death events are extracted
//! as a persistence barcode.
//!
//! Output (stdout): two CSV blocks separated by a blank line.
//!
//! Block 1 — Betti-0 sweep:
//!   time,active_nodes,n_components,largest_size,fraction_largest
//!
//! Block 2 — persistence barcode:
//!   birth,death,lifetime,peak_size
//!
//! Usage:
//!   cargo run --example demo_4d_persistence
//!   cargo run --example demo_4d_persistence > results/persistence.csv

use geographdb_core::{
    compute_temporal_barcode, temporal_persistence_sweep, GraphNode4D, GraphProperties,
    TemporalEdge,
};
use glam::Vec3;

// ── Experiment parameters (match demo_4d_percolation for comparability) ──────

const N_NODES: u64 = 300;
const SIDE: f32 = 10.0;
const NEIGHBOURHOOD: f32 = 2.5;
const T_MAX: u64 = 100;
const LIFETIME_FRACTION: f64 = 0.4;

/// Large enough to include all nodes regardless of position.
const SPHERE_RADIUS: f32 = SIDE * 2.0;
const CENTER: Vec3 = Vec3::new(SIDE / 2.0, SIDE / 2.0, SIDE / 2.0);

/// Step size for the time sweep.
const T_STEP: u64 = 2;

/// Half-width of the temporal window around each time slice.
const T_HALF: u64 = 8;

// ── Deterministic PRNG (Knuth LCG — matches demo_4d_percolation) ─────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    fn next_bounded(&mut self, max: u64) -> u64 {
        self.next_u64() % max
    }
}

// ── Graph generation (identical to demo_4d_percolation) ──────────────────────

fn generate_nodes(rng: &mut Lcg) -> Vec<GraphNode4D> {
    let lifetime_avg = (T_MAX as f64 * LIFETIME_FRACTION) as u64;

    let mut nodes: Vec<GraphNode4D> = (0..N_NODES)
        .map(|id| {
            let x = rng.next_f32() * SIDE;
            let y = rng.next_f32() * SIDE;
            let z = rng.next_f32() * SIDE;
            let begin_ts = rng.next_bounded(T_MAX.saturating_sub(lifetime_avg).max(1));
            let lifetime = lifetime_avg / 2 + rng.next_bounded(lifetime_avg);
            let end_ts = (begin_ts + lifetime).min(T_MAX);
            GraphNode4D {
                id,
                x,
                y,
                z,
                begin_ts,
                end_ts,
                properties: GraphProperties::new(),
                successors: Vec::new(),
            }
        })
        .collect();

    for i in 0..nodes.len() {
        let pi = Vec3::new(nodes[i].x, nodes[i].y, nodes[i].z);
        let i_begin = nodes[i].begin_ts;
        let i_end = nodes[i].end_ts;
        let succs: Vec<TemporalEdge> = nodes
            .iter()
            .enumerate()
            .filter_map(|(j, nb)| {
                if j == i {
                    return None;
                }
                let pj = Vec3::new(nb.x, nb.y, nb.z);
                let dist = pi.distance(pj);
                if dist > NEIGHBOURHOOD {
                    return None;
                }
                let e_begin = i_begin.max(nb.begin_ts);
                let e_end = i_end.min(nb.end_ts);
                if e_begin >= e_end {
                    return None;
                }
                Some(TemporalEdge {
                    dst: nb.id,
                    weight: dist,
                    begin_ts: e_begin,
                    end_ts: e_end,
                })
            })
            .collect();
        nodes[i].successors = succs;
    }

    nodes
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let mut rng = Lcg::new(0xdeadbeef_cafebabe);
    let nodes = generate_nodes(&mut rng);

    let time_slices: Vec<u64> = (0..=T_MAX).step_by(T_STEP as usize).collect();

    let points = temporal_persistence_sweep(&nodes, CENTER, SPHERE_RADIUS, &time_slices, T_HALF);

    // ── CSV block 1: Betti-0 curve ───────────────────────────────────────────
    println!(
        "# Temporal persistence sweep — geographdb-core v{}",
        geographdb_core::VERSION
    );
    println!(
        "# nodes={N_NODES} neighbourhood={NEIGHBOURHOOD} T_MAX={T_MAX} T_HALF={T_HALF} seed=0xdeadbeef_cafebabe"
    );
    println!("time,active_nodes,n_components,largest_size,fraction_largest");
    for p in &points {
        println!(
            "{},{},{},{},{:.6}",
            p.t, p.active, p.n_components, p.largest_size, p.fraction_largest
        );
    }

    // ── CSV block 2: persistence barcode ────────────────────────────────────
    let bars = compute_temporal_barcode(&points);

    println!();
    println!("# Persistence barcode — birth/death of connected components");
    println!("birth,death,lifetime,peak_size");
    for b in &bars {
        let (death_str, lifetime_str) = match b.death {
            Some(d) => (d.to_string(), (d - b.birth).to_string()),
            None => ("NA".to_string(), "NA".to_string()),
        };
        println!("{},{},{},{}", b.birth, death_str, lifetime_str, b.peak_size);
    }

    // ── Summary to stderr ────────────────────────────────────────────────────
    let total_edges: usize = nodes.iter().map(|n| n.successors.len()).sum();
    eprintln!("nodes={N_NODES}  edges={total_edges}");

    let max_components = points.iter().map(|p| p.n_components).max().unwrap_or(0);
    let peak_active = points.iter().map(|p| p.active).max().unwrap_or(0);
    eprintln!("peak_active={peak_active}  max_components={max_components}");

    let active_range: Vec<u64> = points
        .iter()
        .filter(|p| p.active > 0)
        .map(|p| p.t)
        .collect();
    if let (Some(&t_first), Some(&t_last)) = (active_range.first(), active_range.last()) {
        eprintln!(
            "Active window: t=[{t_first}, {t_last}]  width={}",
            t_last - t_first
        );
    }

    eprintln!("\nPersistence barcode ({} bars):", bars.len());
    for b in &bars {
        match b.death {
            Some(d) => eprintln!(
                "  birth={:>3}  death={:>3}  lifetime={:>3}  peak_size={}",
                b.birth,
                d,
                d - b.birth,
                b.peak_size
            ),
            None => eprintln!(
                "  birth={:>3}  death= NA  lifetime= NA  peak_size={}",
                b.birth, b.peak_size
            ),
        }
    }

    // Topological summary: longest-lived bar.
    if let Some(longest) = bars
        .iter()
        .filter(|b| b.death.is_some())
        .max_by_key(|b| b.death.unwrap() - b.birth)
    {
        eprintln!(
            "\nLongest-lived component: birth={} death={:?} lifetime={}",
            longest.birth,
            longest.death,
            longest.death.unwrap() - longest.birth
        );
    }
    if let Some(immortal) = bars.iter().find(|b| b.death.is_none()) {
        eprintln!(
            "Surviving component (no death): birth={}  peak_size={}",
            immortal.birth, immortal.peak_size
        );
    }
}
