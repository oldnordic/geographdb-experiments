use geographdb_core::{
    astar_find_path_4d, GraphNode4D, TemporalEdge, TemporalWindow, TraversalContext4D,
};
use geographdb_core::{load_graph4d, save_graph4d};
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

fn route_graph() -> Vec<GraphNode4D> {
    vec![
        node(
            1,
            (0.0, 0.0, 0.0),
            vec![edge(2, 1.0, 0, 100), edge(5, 2.0, 0, 100)],
        ),
        node(2, (1.0, 0.0, 0.0), vec![edge(3, 1.0, 0, 40)]),
        node(3, (2.0, 0.0, 0.0), vec![edge(4, 1.0, 0, 40)]),
        node(4, (3.0, 0.0, 0.0), vec![]),
        node(5, (0.0, 5.0, 0.0), vec![edge(6, 2.0, 0, 100)]),
        node(6, (3.0, 5.0, 0.0), vec![edge(4, 2.0, 0, 100)]),
    ]
}

fn context(start: u64, end: u64) -> TraversalContext4D {
    TraversalContext4D {
        time_window: Some(TemporalWindow { start, end }),
        graph_weight: 1.0,
        spatial_weight: 0.2,
        temporal_weight: 0.0,
        ..TraversalContext4D::default()
    }
}

fn main() {
    let graph = route_graph();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&graph, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");
    let early = context(10, 20);
    let late = context(50, 60);

    let early_path = astar_find_path_4d(&graph, 1, 4, &early).expect("early path should exist");
    let late_path = astar_find_path_4d(&graph, 1, 4, &late).expect("late path should exist");

    assert_eq!(early_path.node_ids, vec![1, 2, 3, 4]);
    assert_eq!(late_path.node_ids, vec![1, 5, 6, 4]);

    let result = json!({
        "demo": "4d_route_planning",
        "description": "A* route selection across time windows with graph and spatial costs",
        "start_node": 1,
        "goal_node": 4,
        "routes": {
            "early": {
                "time_window": {"start": 10, "end": 20},
                "nodes": early_path.node_ids,
                "total_cost": early_path.total_cost,
                "actual_cost": early_path.actual_cost,
                "heuristic_cost": early_path.heuristic_cost
            },
            "late": {
                "time_window": {"start": 50, "end": 60},
                "nodes": late_path.node_ids,
                "total_cost": late_path.total_cost,
                "actual_cost": late_path.actual_cost,
                "heuristic_cost": late_path.heuristic_cost
            }
        }
    });

    if env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return;
    }

    println!("4D route planning");
    println!(
        "time 10..20 best route: {:?}, cost {:.2}",
        result["routes"]["early"]["nodes"],
        result["routes"]["early"]["actual_cost"]
            .as_f64()
            .unwrap_or(0.0)
    );
    println!(
        "time 50..60 best route: {:?}, cost {:.2}",
        result["routes"]["late"]["nodes"],
        result["routes"]["late"]["actual_cost"]
            .as_f64()
            .unwrap_or(0.0)
    );
}
