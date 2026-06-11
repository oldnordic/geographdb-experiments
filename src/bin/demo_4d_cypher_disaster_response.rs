//! Disaster-response drone network — Cypher 4D query showcase.
//!
//! A single demo that exercises three Cypher procedures on a shared graph:
//!   1. Signal propagation  — which relays are reachable from the command centre at t=10.
//!   2. Temporal routing    — fastest route from a reachable relay to the medkit depot.
//!   3. Resilience analysis — articulation points and bridges in the forward operating area.
//!
//! All positions, travel times, and time windows are synthetic teaching data.

use geographdb_core::{
    load_graph4d, query_4d, save_graph4d, GeoCypherResult, GraphNode4D, TemporalEdge,
};
use serde_json::{json, Value};
use std::env;
use tempfile::tempdir;

// ---- Synthetic disaster-response graph ----------------------------------------------------------
//
// Node layout (units are abstract “model kilometres”):
//   0  Command centre      (0, 0, 0)
//   1  Relay Alpha          (2, 1, 0)
//   2  Relay Bravo          (4, 2, 0)
//   3  Relay Charlie        (3, 4, 0)
//   4  Relay Delta          (6, 3, 0)   — medkit launch point
//   5  Medkit depot         (8, 5, 0)
//   6  Supply drop zone     (7, 1, 0)
//
// Edges are directional radio links with travel-time weights and
// temporal validity (e.g. relay batteries, weather windows).

fn edge(dst: u64, travel_time: f32, begin_ts: u64, end_ts: u64) -> TemporalEdge {
    TemporalEdge {
        dst,
        weight: travel_time,
        begin_ts,
        end_ts,
    }
}

fn node(id: u64, xyz: (f32, f32, f32), successors: Vec<TemporalEdge>) -> GraphNode4D {
    let name = match id {
        0 => "command-centre",
        1 => "relay-alpha",
        2 => "relay-bravo",
        3 => "relay-charlie",
        4 => "relay-delta",
        5 => "medkit-depot",
        6 => "drop-zone",
        _ => "unknown",
    };
    let properties = [
        ("name".to_string(), json!(name)),
        (
            "kind".to_string(),
            json!(if id == 0 {
                "base"
            } else if id == 5 {
                "depot"
            } else {
                "relay"
            }),
        ),
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

fn build_network() -> Vec<GraphNode4D> {
    vec![
        // 0 — Command centre
        node(
            0,
            (0.0, 0.0, 0.0),
            vec![edge(1, 1.0, 0, 50), edge(3, 2.0, 0, 40)],
        ),
        // 1 — Relay Alpha
        node(
            1,
            (2.0, 1.0, 0.0),
            vec![edge(2, 1.0, 0, 60), edge(3, 1.5, 5, 55)],
        ),
        // 2 — Relay Bravo
        node(
            2,
            (4.0, 2.0, 0.0),
            vec![edge(4, 1.0, 0, 70), edge(6, 2.0, 0, 30)],
        ),
        // 3 — Relay Charlie
        node(
            3,
            (3.0, 4.0, 0.0),
            vec![edge(4, 1.0, 10, 50), edge(5, 2.5, 15, 45)],
        ),
        // 4 — Relay Delta (medkit launch point)
        node(
            4,
            (6.0, 3.0, 0.0),
            vec![edge(5, 1.0, 0, 80), edge(6, 1.5, 20, 80)],
        ),
        // 5 — Medkit depot
        node(5, (8.0, 5.0, 0.0), vec![]),
        // 6 — Supply drop zone
        node(6, (7.0, 1.0, 0.0), vec![edge(4, 1.5, 0, 50)]),
    ]
}

// ---- JSON helpers -------------------------------------------------------------------------------

fn json_journey(j: &geographdb_core::TemporalJourney4D) -> Value {
    json!({
        "arrival_time": j.arrival_time,
        "duration": j.duration,
        "departure_time": j.departure_time,
    })
}

fn json_dijkstra(d: &geographdb_core::TemporalDijkstraResult4D) -> Value {
    let arrivals: Vec<Value> = d
        .reachable
        .iter()
        .map(|a| json!({"node": a.node_id, "time": a.arrival_time, "cost": a.cost, "path": a.path}))
        .collect();
    json!({
        "start_node": d.start_node,
        "departure_time": d.departure_time,
        "reachable_count": d.reachable.len(),
        "reachable_nodes": d.reachable.iter().map(|a| a.node_id).collect::<Vec<_>>(),
        "arrivals": arrivals,
    })
}

// ---- Main ---------------------------------------------------------------------------------------

fn main() {
    let nodes = build_network();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&nodes, tmp_dir.path()).expect("save graph");
    let nodes = load_graph4d(tmp_dir.path()).expect("load graph");
    let emit_json = env::args().any(|a| a == "--json");

    // --- Query 1: Signal propagation from command centre at t=10 -------------------------------
    let q1 =
        r#"CALL db.signal.propagate(0, departure: 10, spatial_sphere([0.0, 0.0, 0.0], 100.0))"#;
    let r1 = query_4d(&nodes, q1).expect("signal propagation query must parse");

    let (reachable_10, dijkstra_10) = match r1 {
        GeoCypherResult::Dijkstra(d) => (d.reachable.len(), json_dijkstra(&d)),
        _ => panic!("expected Dijkstra result for signal propagation"),
    };

    // --- Query 2: Temporal route from Relay Delta (node 4) to medkit depot (5) at t=12 --------
    // Delta was reachable from command at t=10, so we route a medkit from there at t=12.
    let q2 = r#"CALL db.route.temporal(4, 5, departure: 12)"#;
    let r2 = query_4d(&nodes, q2).expect("temporal route query must parse");

    let journey = match r2 {
        GeoCypherResult::Journey(j) => json_journey(&j),
        _ => panic!("expected Journey result for temporal route"),
    };

    // --- Query 3: Bottleneck analysis in the forward area (t=25..35) ----------------------------
    let q3 = r#"CALL db.resilience.bottlenecks(spatial_sphere([3.0, 2.0, 0.0], 10.0), time_window(25, 35))"#;
    let r3 = query_4d(&nodes, q3).expect("bottleneck query must parse");

    let (aps, brs) = match r3 {
        GeoCypherResult::Bottlenecks {
            articulation_points,
            bridges,
        } => (articulation_points, bridges),
        _ => panic!("expected Bottlenecks result for resilience analysis"),
    };

    if emit_json {
        let out = json!({
            "demo": "4d_cypher_disaster_response",
            "description": "Disaster-response drone network — signal propagation, temporal routing, and bottleneck analysis via Cypher 4D queries.",
            "disclaimer": "Synthetic positions, travel times, and time windows for teaching. Not operational flight planning data.",
            "queries": {
                "signal_propagation": {
                    "query": q1,
                    "result": dijkstra_10,
                },
                "temporal_route_medkit": {
                    "query": q2,
                    "result": journey,
                },
                "resilience_bottlenecks": {
                    "query": q3,
                    "result": {
                        "articulation_points": aps,
                        "bridges": brs.iter().map(|(a,b)| vec![*a,*b]).collect::<Vec<_>>(),
                    },
                },
            },
            "narrative": format!(
                "At t=10, {} nodes are reachable from the command centre. \
                 Relay Delta (node 4) is one of them, so a medkit is dispatched \
                 at t=12 and arrives at the depot at t={}. \
                 In the t=25..35 window, the forward area has {} \
                 articulation point(s) and {} bridge(s).",
                reachable_10,
                journey.get("arrival_time").and_then(Value::as_u64).unwrap_or(0),
                aps.len(),
                brs.len()
            ),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║  Disaster-Response Drone Network — Cypher 4D Query Showcase     ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("[Q1] Signal propagation from command centre at t=10");
        println!("     Reachable nodes: {}", reachable_10);
        println!();
        println!("[Q2] Temporal route  Relay Delta (4) → Medkit depot (5)  @ t=12");
        println!(
            "     Arrival time: {}",
            journey
                .get("arrival_time")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        );
        println!(
            "     Duration: {}",
            journey.get("duration").and_then(Value::as_u64).unwrap_or(0)
        );
        println!();
        println!("[Q3] Bottleneck analysis  forward area  @ t=25..35");
        println!("     Articulation points: {:?}", aps);
        println!("     Bridges: {:?}", brs);
        println!();
        println!("All queries executed successfully.");
    }
}
