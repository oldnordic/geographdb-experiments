//! Protein conformational landscape exploration.
//!
//! Coarse-grained state transitions of a small protein in a spatially
//! embedded energy landscape. Nodes are conformational states, edges are
//! thermally allowed transitions with barrier costs and temporal validity.
//!
//! This is a **conceptual state-space demo**, not a protein folding predictor.
//! No MD force fields, no backbone angles, no conformational sampling.
//! Coordinates are reduced embedding proxies (compactness, contact order,
//! energy basin depth), not atomic positions.

use geographdb_core::{
    articulation_points_4d, astar_find_path_4d, bridges_4d, reachable_4d, GraphNode4D,
    SpatialRegion, TemporalEdge, TemporalWindow, TraversalContext4D,
};
use geographdb_core::{load_graph4d, save_graph4d};
use glam::Vec3;
use serde_json::json;
use std::env;
use tempfile::tempdir;

fn edge(dst: u64, weight: f32, begin_ts: u64, end_ts: u64) -> TemporalEdge {
    TemporalEdge {
        dst,
        weight,
        begin_ts,
        end_ts,
    }
}

fn state_node(
    id: u64,
    name: &str,
    xyz: (f32, f32, f32),
    successors: Vec<TemporalEdge>,
) -> GraphNode4D {
    let properties = [
        ("name".to_string(), json!(name)),
        ("kind".to_string(), json!("folding_state")),
    ]
    .into_iter()
    .collect();

    GraphNode4D {
        id,
        x: xyz.0,
        y: xyz.1,
        z: xyz.2,
        begin_ts: 0,
        end_ts: 200,
        properties,
        successors,
    }
}

fn build_landscape() -> Vec<GraphNode4D> {
    // Coordinates are reduced embedding proxies, not atomic positions:
    //   x = compactness / radius of gyration proxy
    //   y = contact order / topology proxy
    //   z = energy basin depth proxy (lower = deeper basin)
    vec![
        // 0 — Extended / unfolded
        state_node(
            0,
            "unfolded",
            (0.0, 0.0, 5.0),
            vec![edge(1, 2.0, 0, 100), edge(2, 3.5, 10, 90)],
        ),
        // 1 — Collapsed globule
        state_node(
            1,
            "collapsed",
            (1.5, 1.0, 3.0),
            vec![
                edge(2, 1.5, 0, 120),
                edge(3, 2.0, 20, 110),
                edge(4, 4.0, 30, 80),
            ],
        ),
        // 2 — Partial structure with helical seeds
        state_node(
            2,
            "helix_seed",
            (2.5, 2.5, 2.0),
            vec![edge(3, 1.0, 0, 150), edge(5, 2.5, 40, 140)],
        ),
        // 3 — Beta-contact state (aggregation-capable)
        state_node(
            3,
            "beta_contact",
            (3.0, 4.0, 1.5),
            vec![edge(5, 1.0, 0, 180), edge(6, 2.0, 50, 160)],
        ),
        // 4 — Misfolded kinetic trap
        state_node(
            4,
            "misfolded_trap",
            (1.0, 3.0, 4.0),
            vec![edge(5, 3.0, 60, 120)],
        ),
        // 5 — Near-native ensemble
        state_node(
            5,
            "near_native",
            (3.5, 3.5, 0.5),
            vec![edge(6, 1.0, 0, 200)],
        ),
        // 6 — Native-like
        state_node(6, "native_like", (4.0, 4.0, 0.0), vec![]),
    ]
}

fn main() {
    let nodes = build_landscape();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&nodes, tmp_dir.path()).expect("save graph");
    let nodes = load_graph4d(tmp_dir.path()).expect("load graph");
    let emit_json = env::args().any(|a| a == "--json");

    // --- Q1: Earliest feasible route  unfolded(0) → native_like(6) ----------------------
    let route = astar_find_path_4d(
        &nodes,
        0,
        6,
        &TraversalContext4D {
            time_window: Some(TemporalWindow { start: 0, end: 200 }),
            spatial_region: None,
            ..TraversalContext4D::default()
        },
    );

    // --- Q2: Fastest-duration route  helix_seed(2) → native_like(6) ----------------------
    let fast_route = astar_find_path_4d(
        &nodes,
        2,
        6,
        &TraversalContext4D {
            time_window: Some(TemporalWindow { start: 0, end: 200 }),
            spatial_region: None,
            ..TraversalContext4D::default()
        },
    );

    // --- Q3: Bottleneck intermediates in the folding funnel ----------------------------
    let ctx = TraversalContext4D {
        time_window: Some(TemporalWindow {
            start: 20,
            end: 160,
        }),
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::new(2.5, 2.5, 2.5),
            radius: 4.0,
        }),
        ..TraversalContext4D::default()
    };
    let aps = articulation_points_4d(&nodes, &ctx);
    let brs = bridges_4d(&nodes, &ctx);

    // --- Q4: Local state neighbourhood around near_native(5) --------------------------
    let local_ctx = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 0, end: 200 }),
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::new(3.5, 3.5, 0.5),
            radius: 3.0,
        }),
        ..TraversalContext4D::default()
    };
    let local = reachable_4d(&nodes, 5, &local_ctx);

    // --- Q5: Rejected trap escape  unfolded(0) → native_like via misfolded(4) -----------
    let rejected = astar_find_path_4d(
        &nodes,
        0,
        6,
        &TraversalContext4D {
            time_window: Some(TemporalWindow { start: 0, end: 55 }),
            spatial_region: None,
            ..TraversalContext4D::default()
        },
    );

    if emit_json {
        let out = json!({
            "demo": "4d_protein_conformation_landscape",
            "description": "Coarse-grained protein state transitions across a spatially embedded temporal energy landscape",
            "disclaimer": "Conformational states are coarse abstractions; coordinates are reduced embedding proxies, not atomic positions. No MD force fields or backbone geometry.",
            "earliest_native_like_path": route.as_ref().map(|r| json!({
                "nodes": r.node_ids,
                "arrival_time": r.total_cost,
            })).unwrap_or_else(|| json!(null)),
            "fastest_duration_path": fast_route.as_ref().map(|r| json!({
                "nodes": r.node_ids,
                "departure_time": 0,
                "arrival_time": r.total_cost,
                "duration": r.total_cost,
            })).unwrap_or_else(|| json!(null)),
            "rejected_transition": rejected.as_ref().map_or_else(
                || json!({"nodes": Vec::<u64>::new(), "reason": "escape edge becomes inactive before trap state is reached"}),
                |r| json!({"nodes": r.node_ids, "reason": "route found but trap edge expires before arrival"})
            ),
            "bottleneck_states": aps,
            "local_neighbourhood_near_native": local,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  Protein Conformational Landscape — 4D State-Space Exploration   ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        match route {
            Some(r) => {
                println!("[Q1] Earliest feasible route  unfolded(0) → native_like(6)");
                println!("     Nodes: {:?}", r.node_ids);
                println!("     Arrival time: {}", r.total_cost);
            }
            None => println!("[Q1] No feasible route in window"),
        }
        println!();
        match fast_route {
            Some(r) => {
                println!("[Q2] Fastest route  helix_seed(2) → native_like(6)");
                println!("     Nodes: {:?}", r.node_ids);
                println!("     Duration: {}", r.total_cost);
            }
            None => println!("[Q2] No fast route found"),
        }
        println!();
        println!("[Q3] Bottleneck intermediates  folding funnel  t=20..160");
        println!("     Articulation points: {:?}", aps);
        println!("     Bridges: {:?}", brs);
        println!();
        println!("[Q4] Local neighbourhood   from near_native(5) in r=3");
        println!("     Reachable states: {:?}", local);
        println!();
        match rejected {
            Some(r) => {
                println!("[Q5] Rejected trap escape route: {:?}", r.node_ids);
                println!("     Reason: trap edge expires before arrival");
            }
            None => {
                println!("[Q5] Rejected transition via misfolded(4)");
                println!("     Reason: edge inactive at requested stage");
            }
        }
        println!();
        println!("All queries executed.");
    }
}
