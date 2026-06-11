//! Spatiotemporal percolation phase diagram experiment.
//!
//! Generates a random 4D graph and sweeps spatial radius × time window to find
//! the connectivity phase boundary r*(t) — the critical radius at each time
//! slice where the graph transitions from fragmented to connected.
//!
//! Output (stdout): two CSV blocks separated by a blank line.
//!
//! Block 1 — full sweep:
//!   radius,time,active_nodes,reachable_nodes,fraction
//!
//! Block 2 — phase boundary:
//!   time,critical_radius
//!
//! Usage:
//!   cargo run --example demo_4d_percolation
//!   cargo run --example demo_4d_percolation > results/percolation.csv

use geographdb_core::{
    find_critical_radius, percolation_sweep, GraphNode4D, GraphProperties, TemporalEdge,
};
use glam::Vec3;

// ── Experiment parameters ────────────────────────────────────────────────────

/// Number of nodes scattered in the bounding cube.
const N_NODES: u64 = 300;

/// Bounding cube side length (nodes placed in [0, SIDE]³).
const SIDE: f32 = 10.0;

/// A node connects to every other node within this Euclidean distance.
/// This is the fixed neighbourhood radius that wires up the graph;
/// it is independent of the query sphere radius swept during measurement.
const NEIGHBOURHOOD: f32 = 2.5;

/// Maximum timestamp. Node validity intervals are drawn from [0, T_MAX].
const T_MAX: u64 = 100;

/// Average node lifetime as a fraction of T_MAX.
const LIFETIME_FRACTION: f64 = 0.4;

/// Query sphere centre (middle of the bounding cube).
const CENTER: Vec3 = Vec3::new(SIDE / 2.0, SIDE / 2.0, SIDE / 2.0);

/// Radius sweep: `N_RADII` steps from `R_MIN` to `R_MAX`.
const N_RADII: usize = 50;
const R_MIN: f32 = 0.2;
const R_MAX: f32 = SIDE * 1.1; // slightly beyond the bounding cube diagonal

/// Time slices to evaluate (evenly spaced over [0, T_MAX]).
const N_TIME_SLICES: usize = 8;

/// Half-width of the temporal window around each time slice.
const T_HALF: u64 = 8;

/// Percolation threshold for locating r* (fraction of active nodes reachable).
const THRESHOLD: f32 = 0.5;

/// Minimum active nodes required before r* detection fires.
/// Suppresses the trivial single-node detection (1 active node → fraction=1.0).
const MIN_ACTIVE_FOR_CRITICAL: usize = 10;

// ── Deterministic PRNG (Knuth LCG — no rand dependency) ─────────────────────

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

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Uniform u64 in [0, max).
    fn next_bounded(&mut self, max: u64) -> u64 {
        self.next_u64() % max
    }
}

// ── Graph generation ─────────────────────────────────────────────────────────

fn generate_nodes(rng: &mut Lcg) -> Vec<GraphNode4D> {
    // Phase 1: positions and temporal intervals.
    let lifetime_avg = (T_MAX as f64 * LIFETIME_FRACTION) as u64;

    let mut nodes: Vec<GraphNode4D> = (0..N_NODES)
        .map(|id| {
            let x = rng.next_f32() * SIDE;
            let y = rng.next_f32() * SIDE;
            let z = rng.next_f32() * SIDE;

            // begin_ts in [0, T_MAX - lifetime_avg], end_ts = begin_ts + lifetime
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

    // Phase 2: wire edges — connect every node to all neighbours within NEIGHBOURHOOD.
    // Edges are valid during the intersection of both endpoint validity intervals.
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

/// Find the node closest to `CENTER` — used as the BFS origin.
fn find_center_node(nodes: &[GraphNode4D]) -> u64 {
    nodes
        .iter()
        .min_by(|a, b| {
            let da = Vec3::new(a.x, a.y, a.z).distance(CENTER);
            let db = Vec3::new(b.x, b.y, b.z).distance(CENTER);
            da.partial_cmp(&db).unwrap()
        })
        .map(|n| n.id)
        .expect("node list must not be empty")
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let mut rng = Lcg::new(0xdeadbeef_cafebabe);
    let nodes = generate_nodes(&mut rng);

    let center_id = find_center_node(&nodes);

    // Radius sweep.
    let radii: Vec<f32> = (0..N_RADII)
        .map(|i| R_MIN + (R_MAX - R_MIN) * i as f32 / (N_RADII - 1) as f32)
        .collect();

    // Time slices evenly spaced over [T_HALF, T_MAX - T_HALF].
    let t_inner = T_MAX.saturating_sub(2 * T_HALF);
    let time_slices: Vec<u64> = (0..N_TIME_SLICES)
        .map(|i| T_HALF + t_inner * i as u64 / (N_TIME_SLICES - 1).max(1) as u64)
        .collect();

    let points = percolation_sweep(&nodes, center_id, CENTER, &radii, &time_slices, T_HALF);

    // ── CSV block 1: full sweep ──────────────────────────────────────────────
    println!(
        "# Spatiotemporal percolation sweep — geographdb-core v{}",
        geographdb_core::VERSION
    );
    println!(
        "# nodes={N_NODES} neighbourhood={NEIGHBOURHOOD} T_MAX={T_MAX} seed=0xdeadbeef_cafebabe"
    );
    println!("radius,time,active_nodes,reachable_nodes,fraction");
    for p in &points {
        println!(
            "{:.4},{},{},{},{:.6}",
            p.radius, p.time, p.active, p.reachable, p.fraction
        );
    }

    // ── CSV block 2: phase boundary ──────────────────────────────────────────
    println!();
    println!("# Phase boundary r*(t) — first radius where fraction >= {THRESHOLD} with active >= {MIN_ACTIVE_FOR_CRITICAL}");
    println!("time,critical_radius");
    for &t in &time_slices {
        match find_critical_radius(&points, t, THRESHOLD, MIN_ACTIVE_FOR_CRITICAL) {
            Some(r) => println!("{t},{r:.4}"),
            None => println!("{t},NA"),
        }
    }

    // ── CSV block 3: temporal connectivity curve ─────────────────────────────
    // Fix sphere at maximum radius (whole graph) and sweep time.
    // This reveals the temporal basin of connectivity — how reachability
    // from the centre node evolves as the time window moves across [0, T_MAX].
    let max_radius = *radii.last().expect("radii must not be empty");
    let t_sweep: Vec<u64> = (0..=T_MAX).step_by(2).collect();
    let temporal_points =
        percolation_sweep(&nodes, center_id, CENTER, &[max_radius], &t_sweep, T_HALF);

    println!();
    println!("# Temporal connectivity curve — sphere radius={max_radius:.1} (whole graph), time swept over [0,{T_MAX}]");
    println!("time,active_nodes,reachable_nodes,fraction");
    for p in &temporal_points {
        println!("{},{},{},{:.6}", p.time, p.active, p.reachable, p.fraction);
    }

    // ── Summary to stderr ────────────────────────────────────────────────────
    let total_edges: usize = nodes.iter().map(|n| n.successors.len()).sum();
    eprintln!("nodes={N_NODES}  edges={total_edges}  centre_node={center_id}");
    eprintln!("avg_degree={:.1}", total_edges as f64 / N_NODES as f64);

    eprintln!("\nPhase boundary r*(t):");
    for &t in &time_slices {
        match find_critical_radius(&points, t, THRESHOLD, MIN_ACTIVE_FOR_CRITICAL) {
            Some(r) => eprintln!("  t={t:>3}  r* = {r:.3}"),
            None => eprintln!("  t={t:>3}  r* = NA"),
        }
    }

    // Temporal connectivity basin: first and last t where fraction > 0.
    let active_times: Vec<u64> = temporal_points
        .iter()
        .filter(|p| p.reachable > 0)
        .map(|p| p.time)
        .collect();
    if let (Some(&t_first), Some(&t_last)) = (active_times.first(), active_times.last()) {
        eprintln!(
            "\nTemporal connectivity basin: t=[{t_first}, {t_last}]  width={}",
            t_last - t_first
        );
        let max_frac = temporal_points
            .iter()
            .map(|p| p.fraction)
            .fold(0.0f32, f32::max);
        eprintln!("Peak fraction: {max_frac:.4}");
        let isolated: Vec<_> = temporal_points
            .iter()
            .filter(|p| p.active > 0 && p.fraction < 1.0)
            .collect();
        eprintln!(
            "Time slices with isolated nodes (fraction < 1.0): {}",
            isolated.len()
        );
    } else {
        eprintln!("\nTemporal connectivity basin: centre node unreachable at all time slices");
    }
}
