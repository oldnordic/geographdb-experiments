use geographdb_core::{
    articulation_points_4d, bridges_4d, reachable_4d, GraphNode4D, SpatialRegion, TemporalEdge,
    TemporalWindow, TraversalContext4D,
};
use geographdb_core::{load_graph4d, save_graph4d};
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

fn relay_mesh() -> Vec<GraphNode4D> {
    vec![
        node(1, (0.0, 0.0, 0.0), vec![edge(2, 0, 100), edge(4, 40, 100)]),
        node(2, (1.0, 0.0, 0.0), vec![edge(3, 0, 100)]),
        node(3, (2.0, 0.0, 0.0), vec![edge(6, 0, 100)]),
        node(4, (0.0, 2.0, 0.0), vec![edge(5, 0, 100)]),
        node(5, (1.0, 2.0, 0.0), vec![edge(3, 40, 100), edge(6, 40, 100)]),
        node(6, (3.0, 0.0, 0.0), vec![]),
    ]
}

fn context(start: u64, end: u64) -> TraversalContext4D {
    TraversalContext4D {
        time_window: Some(TemporalWindow { start, end }),
        spatial_region: Some(SpatialRegion::Sphere {
            center: Vec3::ZERO,
            radius: 10.0,
        }),
        ..TraversalContext4D::default()
    }
}

fn reachable_after_removing_node(
    graph: &[GraphNode4D],
    removed_node: u64,
    from: u64,
    ctx: &TraversalContext4D,
) -> Vec<u64> {
    let filtered: Vec<GraphNode4D> = graph
        .iter()
        .filter(|node| node.id != removed_node)
        .map(|node| {
            let mut node = node.clone();
            node.successors.retain(|edge| edge.dst != removed_node);
            node
        })
        .collect();

    reachable_4d(&filtered, from, ctx)
}

fn main() {
    let graph = relay_mesh();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&graph, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");
    let early = context(10, 20);
    let late = context(50, 60);

    let early_articulation = articulation_points_4d(&graph, &early);
    let late_articulation = articulation_points_4d(&graph, &late);
    let early_bridges = bridges_4d(&graph, &early);
    let late_bridges = bridges_4d(&graph, &late);
    let early_after_failure = reachable_after_removing_node(&graph, 2, 1, &early);
    let late_after_failure = reachable_after_removing_node(&graph, 2, 1, &late);

    assert_eq!(early_articulation, vec![2, 3]);
    assert_eq!(early_bridges, vec![(1, 2), (2, 3), (3, 6), (4, 5)]);
    assert!(late_articulation.is_empty());
    assert!(late_bridges.is_empty());
    assert_eq!(early_after_failure, vec![1]);
    assert_eq!(late_after_failure, vec![1, 4, 5, 3, 6]);

    let early_articulation_json = early_articulation.clone();
    let late_articulation_json = late_articulation.clone();
    let early_bridges_json = early_bridges.clone();
    let late_bridges_json = late_bridges.clone();
    let early_after_failure_json = early_after_failure.clone();
    let late_after_failure_json = late_after_failure.clone();
    let result = json!({
        "demo": "4d_temporal_bottlenecks",
        "description": "Critical nodes and edges change across temporal graph windows",
        "region": {
            "kind": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": 10.0
        },
        "windows": {
            "early": {
                "time_window": {"start": 10, "end": 20},
                "articulation_points": early_articulation_json,
                "bridges": early_bridges_json,
                "reachable_after_failure": {
                    "removed_node": 2,
                    "from": 1,
                    "reachable": early_after_failure_json
                }
            },
            "late": {
                "time_window": {"start": 50, "end": 60},
                "articulation_points": late_articulation_json,
                "bridges": late_bridges_json,
                "reachable_after_failure": {
                    "removed_node": 2,
                    "from": 1,
                    "reachable": late_after_failure_json
                }
            }
        }
    });

    if env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return;
    }

    println!("4D temporal bottlenecks");
    println!("early articulation points: {:?}", early_articulation);
    println!("early bridges: {:?}", early_bridges);
    println!(
        "early reachable after node 2 failure: {:?}",
        early_after_failure
    );
    println!("late articulation points: {:?}", late_articulation);
    println!("late bridges: {:?}", late_bridges);
    println!(
        "late reachable after node 2 failure: {:?}",
        late_after_failure
    );
}
