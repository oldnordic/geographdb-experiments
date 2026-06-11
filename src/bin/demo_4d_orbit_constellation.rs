//! Satellite mesh routing in a Walker constellation.
//!
//! This is a **simplified illustrative model**, not production orbital
//! mechanics.  Circular orbits, fixed line-of-sight thresholds, and discrete
//! time steps are used to show how 4D graph queries apply to space
//! communication.  Real satellite routing would use J2-perturbed Keplerian
//! propagation, atmospheric drag, and actual antenna patterns.

use geographdb_core::{
    articulation_points_4d, astar_find_path_4d, bridges_4d, load_graph4d, reachable_4d,
    save_graph4d, time_dependent_dijkstra_4d, GraphNode4D, SpatialRegion, TemporalEdge,
    TemporalWindow, TraversalContext4D,
};
use glam::Vec3;
use serde_json::json;
use std::env;
use tempfile::tempdir;

const ORBIT_RADIUS: f32 = 3000.0; // km — model shell radius, not literal geocentric altitude
const LINK_RANGE: f32 = 4000.0; // km, inter-satellite visibility threshold
const PERIOD: u64 = 100; // discrete timesteps per orbit

fn edge(dst: u64, begin_ts: u64, end_ts: u64) -> TemporalEdge {
    TemporalEdge {
        dst,
        weight: 1.0,
        begin_ts,
        end_ts,
    }
}

/// Place satellites on a spherical shell using a Walker-delta-like layout.
/// Returns nodes with successors initially empty.
fn constellation() -> Vec<GraphNode4D> {
    let mut nodes = Vec::new();
    let mut id = 1u64;
    let inclination = 60f32.to_radians();
    let num_planes = 3usize;
    let sats_per_plane = 6usize;

    for plane in 0..num_planes {
        let raan = (plane as f32) * 2.0 * std::f32::consts::PI / (num_planes as f32);
        for sat in 0..sats_per_plane {
            let mean_anomaly = (sat as f32) * 2.0 * std::f32::consts::PI / (sats_per_plane as f32);

            // Position in orbital plane (circular orbit)
            let x_orb = ORBIT_RADIUS * mean_anomaly.cos();
            let y_orb = ORBIT_RADIUS * mean_anomaly.sin();
            let z_orb = 0.0f32;

            // Rotate by inclination around x-axis
            let y_inc = y_orb * inclination.cos() - z_orb * inclination.sin();
            let z_inc = y_orb * inclination.sin() + z_orb * inclination.cos();

            // Rotate by RAAN around z-axis
            let x = x_orb * raan.cos() - y_inc * raan.sin();
            let y = x_orb * raan.sin() + y_inc * raan.cos();
            let z = z_inc;

            nodes.push(GraphNode4D {
                id,
                x,
                y,
                z,
                begin_ts: 0,
                end_ts: PERIOD,
                properties: [
                    ("name".to_string(), json!(format!("sat-{plane}-{sat}"))),
                    ("plane".to_string(), json!(plane)),
                ]
                .into_iter()
                .collect(),
                successors: Vec::new(),
            });
            id += 1;
        }
    }
    nodes
}

/// Build inter-satellite links for a given time step using simple
/// Euclidean distance.  A link exists when two satellites are within
/// LINK_RANGE km.  The edge is valid for a short window around the
/// sampled timestep so the 4D temporal filter can include/exclude it.
fn build_links(nodes: &mut [GraphNode4D], t: u64, window_half: u64) {
    let n = nodes.len();
    for i in 0..n {
        let a = Vec3::new(nodes[i].x, nodes[i].y, nodes[i].z);
        for j in (i + 1)..n {
            let b = Vec3::new(nodes[j].x, nodes[j].y, nodes[j].z);
            if a.distance(b) <= LINK_RANGE {
                let begin = t.saturating_sub(window_half);
                let end = (t + window_half).min(PERIOD);
                nodes[i].successors.push(edge(nodes[j].id, begin, end));
                nodes[j].successors.push(edge(nodes[i].id, begin, end));
            }
        }
    }
}

fn main() {
    let mut nodes = constellation();
    build_links(&mut nodes, 25, 10);
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&nodes, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");

    // 1. Reachability from sat-0-0 at timestep 25 within the whole shell
    let topology_only = TraversalContext4D::default();
    let reachable_all = reachable_4d(&graph, 1, &topology_only);

    // 2. Reachability constrained to a spatial sphere around a ground station
    let ground_station = TraversalContext4D {
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::new(ORBIT_RADIUS, 0.0, 0.0),
            radius: 2500.0,
        }),
        ..TraversalContext4D::default()
    };
    let reachable_near_ground = reachable_4d(&graph, 1, &ground_station);

    // 3. Temporal route from sat-0-0 to a satellite on the opposite side
    let route_ctx = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 20, end: 30 }),
        ..TraversalContext4D::default()
    };
    let route = astar_find_path_4d(&graph, 1, 10, &route_ctx);

    // 4. Bottleneck analysis for the shell at this timestep
    let bottleneck_ctx = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 20, end: 30 }),
        ..TraversalContext4D::default()
    };
    let articulation = articulation_points_4d(&graph, &bottleneck_ctx);
    let bridge_edges = bridges_4d(&graph, &bottleneck_ctx);

    // 5. Signal propagation (time-dependent Dijkstra) from sat-0-0
    let signal =
        time_dependent_dijkstra_4d(&graph, 1, 20, None).expect("start node should be usable");

    let output = json!({
        "demo": "4d_orbit_constellation",
        "description": "Satellite mesh routing in a simplified Walker constellation (simplified orbital mechanics)",
        "disclaimer": "Circular orbits, fixed line-of-sight threshold, discrete time steps. Not production Keplerian propagation.",
        "parameters": {
            "shell_radius_km": ORBIT_RADIUS,
            "link_range_km": LINK_RANGE,
            "period_timesteps": PERIOD,
            "sampled_timestep": 25
        },
        "reachable_whole_shell": {
            "count": reachable_all.len(),
            "nodes": reachable_all
        },
        "reachable_near_ground_station": {
            "count": reachable_near_ground.len(),
            "nodes": reachable_near_ground
        },
        "route_across_shell": route.clone().map(|p| json!({
            "nodes": p.node_ids,
            "cost": p.total_cost
        })),
        "bottlenecks": {
            "articulation_points": articulation,
            "bridges": bridge_edges.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>()
        },
        "signal_propagation": {
            "start_node": signal.start_node,
            "departure_time": signal.departure_time,
            "reachable_count": signal.reachable.len(),
            "earliest_arrivals": signal.reachable.iter().map(|a| json!({
                "node": a.node_id,
                "arrival_time": a.arrival_time,
                "path": a.path
            })).collect::<Vec<_>>()
        }
    });

    if env::args().any(|a| a == "--json") {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("Orbit Constellation Demo");
        println!("========================");
        println!();
        println!("Disclaimer: {}", output["disclaimer"]);
        println!();
        println!("Satellites in shell: {}", 18);
        println!(
            "Reachable from sat-0-0 (whole shell): {} nodes",
            reachable_all.len()
        );
        println!(
            "Reachable near ground station: {} nodes",
            reachable_near_ground.len()
        );
        if let Some(ref path) = route {
            println!(
                "Route across shell: {:?} (cost {:.1})",
                path.node_ids, path.total_cost
            );
        } else {
            println!("No route found across shell at sampled timestep");
        }
        println!("Articulation points: {:?}", articulation);
        println!("Bridges: {:?}", bridge_edges);
        println!(
            "Signal propagation reached {} satellites",
            signal.reachable.len()
        );
    }
}
