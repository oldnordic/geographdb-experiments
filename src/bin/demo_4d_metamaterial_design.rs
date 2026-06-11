//! Metamaterial lattice design exploration.
//!
//! Navigate a graph of candidate lattice designs under geometric and stage
//! constraints to identify feasible transitions toward a target behaviour.
//!
//! This is a **conceptual design-space demo**, not a physical simulator.
//! Coordinates are property proxies (stiffness, auxetic response, damping),
//! not atomic positions or engineering quantities.

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

fn design_node(
    id: u64,
    name: &str,
    xyz: (f32, f32, f32),
    successors: Vec<TemporalEdge>,
) -> GraphNode4D {
    let properties = [
        ("name".to_string(), json!(name)),
        ("kind".to_string(), json!("lattice")),
    ]
    .into_iter()
    .collect();

    GraphNode4D {
        id,
        x: xyz.0,
        y: xyz.1,
        z: xyz.2,
        begin_ts: 0,
        end_ts: 100,
        properties,
        successors,
    }
}

fn build_design_graph() -> Vec<GraphNode4D> {
    // Coordinates are design-space proxies, not physical positions:
    //   x = stiffness proxy
    //   y = auxetic / Poisson response proxy
    //   z = damping / energy-absorption proxy
    vec![
        // 0 — Baseline cubic truss
        design_node(
            0,
            "baseline_truss",
            (0.0, 0.0, 0.0),
            vec![edge(1, 1.0, 0, 60), edge(2, 2.0, 5, 50)],
        ),
        // 1 — Re-entrant auxetic cell
        design_node(
            1,
            "reentrant_auxetic",
            (1.0, 2.0, 0.5),
            vec![
                edge(2, 1.0, 0, 70),
                edge(3, 2.5, 10, 65),
                edge(4, 3.0, 15, 55),
            ],
        ),
        // 2 — Octet lattice (stiff, low auxetic)
        design_node(
            2,
            "octet_lattice",
            (3.0, 0.5, 1.0),
            vec![edge(3, 1.0, 0, 80), edge(5, 2.0, 20, 75)],
        ),
        // 3 — Graded auxetic panel
        design_node(
            3,
            "graded_auxetic",
            (2.0, 3.0, 1.5),
            vec![edge(5, 1.0, 0, 90), edge(6, 2.0, 25, 85)],
        ),
        // 4 — Pentamode-like (intermediate, exotic)
        design_node(
            4,
            "pentamode_like",
            (1.5, 1.0, 2.5),
            vec![edge(3, 1.5, 30, 70)],
        ),
        // 5 — Damped panel candidate
        design_node(
            5,
            "damped_panel",
            (3.5, 2.5, 3.0),
            vec![edge(6, 1.0, 0, 100)],
        ),
        // 6 — Target dual-property design (auxetic + damped)
        design_node(6, "target_dual", (4.0, 4.0, 4.0), vec![]),
    ]
}

fn main() {
    let nodes = build_design_graph();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&nodes, tmp_dir.path()).expect("save graph");
    let nodes = load_graph4d(tmp_dir.path()).expect("load graph");
    let emit_json = env::args().any(|a| a == "--json");

    // --- Q1: Earliest feasible route to target_dual (6) @ stage 0 ------
    let route = astar_find_path_4d(
        &nodes,
        0,
        6,
        &TraversalContext4D {
            time_window: Some(TemporalWindow { start: 0, end: 60 }),
            spatial_region: None,
            ..TraversalContext4D::default()
        },
    );

    // --- Q2: Best-quality route (longer but more direct to target) -------
    let best_route = astar_find_path_4d(
        &nodes,
        2,
        6,
        &TraversalContext4D {
            time_window: Some(TemporalWindow { start: 0, end: 80 }),
            spatial_region: None,
            ..TraversalContext4D::default()
        },
    );

    // --- Q3: Bottleneck motifs in high-performance region --------------
    let ctx = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 30, end: 80 }),
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::new(2.0, 2.0, 2.0),
            radius: 5.0,
        }),
        ..TraversalContext4D::default()
    };
    let aps = articulation_points_4d(&nodes, &ctx);
    let brs = bridges_4d(&nodes, &ctx);

    // --- Q4: Local candidate radius from graded_auxetic (3) -------------
    let local_ctx = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 0, end: 100 }),
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::new(2.0, 3.0, 1.5),
            radius: 3.0,
        }),
        ..TraversalContext4D::default()
    };
    let local = reachable_4d(&nodes, 3, &local_ctx);

    // --- Q5: Rejected shortcut via pentamode ---------------------------
    let rejected = astar_find_path_4d(
        &nodes,
        0,
        6,
        &TraversalContext4D {
            time_window: Some(TemporalWindow { start: 0, end: 25 }),
            spatial_region: None,
            ..TraversalContext4D::default()
        },
    );

    if emit_json {
        let out = json!({
            "demo": "4d_metamaterial_design_exploration",
            "description": "Traversal of a temporal graph of candidate lattice designs under property and manufacturability constraints",
            "disclaimer": "Coordinates are design-space property proxies, not physical positions. Edges represent allowable conceptual mutations, not manufacturing operations.",
            "target": {
                "property": "high_damping_auxetic",
                "constraints": ["printable", "stress_proxy_below_threshold"]
            },
            "earliest_feasible_design_path": route.as_ref().map(|r| json!({
                "nodes": r.node_ids,
                "arrival_stage": r.node_ids.len().saturating_sub(1),
            })).unwrap_or_else(|| json!(null)),
            "best_quality_alternative": best_route.as_ref().map(|r| json!({
                "nodes": r.node_ids,
                "cost": r.total_cost,
            })).unwrap_or_else(|| json!(null)),
            "rejected_shortcut": rejected.as_ref().map_or_else(
                || json!({"nodes": Vec::<u64>::new(), "reason": "intermediate edge inactive at requested stage"}),
                |r| json!({"nodes": r.node_ids, "reason": "route found but later than earliest"})
            ),
            "design_bottlenecks": aps,
            "local_candidate_radius": local,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  Metamaterial Lattice Design — 4D Graph Exploration             ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        match route {
            Some(r) => {
                println!("[Q1] Earliest feasible route  baseline_truss(0) → target_dual(6)");
                println!("     Nodes: {:?}", r.node_ids);
                println!("     Arrival stage: {}", r.node_ids.len().saturating_sub(1));
            }
            None => println!("[Q1] No feasible route in window"),
        }
        println!();
        match best_route {
            Some(r) => {
                println!("[Q2] Best-quality alternative  octet_lattice(2) → target_dual(6)");
                println!("     Nodes: {:?}", r.node_ids);
                println!("     Cost: {}", r.total_cost);
            }
            None => println!("[Q2] No quality route found"),
        }
        println!();
        println!("[Q3] Bottleneck motifs  design-space sphere@t=30..80");
        println!("     Articulation points: {:?}", aps);
        println!("     Bridges: {:?}", brs);
        println!();
        println!("[Q4] Local candidates   from graded_auxetic(3) in r=3");
        println!("     Reachable: {:?}", local);
        println!();
        match rejected {
            Some(_) => println!("[Q5] Shortcut rejected: route exists but not earliest"),
            None => {
                println!("[Q5] Shortcut rejected: intermediate edge inactive at requested stage")
            }
        }
        println!();
        println!("All queries executed.");
    }
}
