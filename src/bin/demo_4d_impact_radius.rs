use geographdb_core::{load_graph4d, save_graph4d};
use geographdb_core::{
    reachable_4d, GraphNode4D, SpatialRegion, TemporalEdge, TemporalWindow, TraversalContext4D,
};
use glam::Vec3;
use serde_json::json;
use std::env;
use tempfile::tempdir;

fn edge(dst: u64, begin_ts: u64, end_ts: u64) -> TemporalEdge {
    TemporalEdge {
        dst,
        weight: 1.0,
        begin_ts,
        end_ts,
    }
}

fn node(id: u64, xyz: (f32, f32, f32), successors: Vec<TemporalEdge>) -> GraphNode4D {
    GraphNode4D {
        id,
        x: xyz.0,
        y: xyz.1,
        z: xyz.2,
        begin_ts: 0,
        end_ts: 100,
        properties: Default::default(),
        successors,
    }
}

fn impact_graph() -> Vec<GraphNode4D> {
    vec![
        node(
            10,
            (0.0, 0.0, 0.0),
            vec![edge(11, 0, 100), edge(14, 0, 100)],
        ),
        node(11, (1.0, 0.0, 0.0), vec![edge(12, 0, 100)]),
        node(12, (2.0, 0.0, 0.0), vec![edge(13, 0, 35)]),
        node(13, (3.0, 0.0, 0.0), vec![]),
        node(14, (40.0, 0.0, 0.0), vec![edge(15, 0, 100)]),
        node(15, (41.0, 0.0, 0.0), vec![]),
    ]
}

fn main() {
    let graph = impact_graph();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&graph, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");

    let topology_only = TraversalContext4D::default();
    let local_region = TraversalContext4D {
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        }),
        ..TraversalContext4D::default()
    };
    let local_after_refactor = TraversalContext4D {
        time_window: Some(TemporalWindow { start: 50, end: 60 }),
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::ZERO,
            radius: 5.0,
        }),
        ..TraversalContext4D::default()
    };

    let all_impact = reachable_4d(&graph, 10, &topology_only);
    let local_impact = reachable_4d(&graph, 10, &local_region);
    let current_local_impact = reachable_4d(&graph, 10, &local_after_refactor);

    assert_eq!(all_impact, vec![10, 11, 14, 12, 15, 13]);
    assert_eq!(local_impact, vec![10, 11, 12, 13]);
    assert_eq!(current_local_impact, vec![10, 11, 12]);

    let result = json!({
        "demo": "4d_impact_radius",
        "description": "Topology-only impact analysis compared with spatial and temporal filtering",
        "start_node": 10,
        "queries": {
            "topology_only": {
                "reachable": all_impact
            },
            "spatial_radius": {
                "region": {
                    "kind": "sphere",
                    "center": [0.0, 0.0, 0.0],
                    "radius": 5.0
                },
                "reachable": local_impact
            },
            "spatial_radius_time_window": {
                "time_window": {"start": 50, "end": 60},
                "region": {
                    "kind": "sphere",
                    "center": [0.0, 0.0, 0.0],
                    "radius": 5.0
                },
                "reachable": current_local_impact
            }
        }
    });

    if env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return;
    }

    println!("4D impact radius");
    println!(
        "topology-only impact from gateway: {:?}",
        result["queries"]["topology_only"]["reachable"]
    );
    println!(
        "spatial radius impact from gateway: {:?}",
        result["queries"]["spatial_radius"]["reachable"]
    );
    println!(
        "spatial + time 50..60 impact from gateway: {:?}",
        result["queries"]["spatial_radius_time_window"]["reachable"]
    );
}
