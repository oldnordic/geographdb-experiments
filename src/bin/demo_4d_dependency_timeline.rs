use geographdb_core::{
    astar_find_path_4d, reachable_4d, strongly_connected_components_4d, GraphNode4D, TemporalEdge,
    TemporalWindow, TraversalContext4D,
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

fn node(
    id: u64,
    xyz: (f32, f32, f32),
    begin_ts: u64,
    end_ts: u64,
    successors: Vec<TemporalEdge>,
) -> GraphNode4D {
    GraphNode4D {
        id,
        x: xyz.0,
        y: xyz.1,
        z: xyz.2,
        begin_ts,
        end_ts,
        properties: Default::default(),
        successors,
    }
}

fn dependency_graph() -> Vec<GraphNode4D> {
    vec![
        node(1, (0.0, 0.0, 0.0), 0, 100, vec![edge(2, 1.0, 0, 100)]),
        node(
            2,
            (1.0, 0.0, 0.0),
            0,
            100,
            vec![edge(3, 1.0, 0, 40), edge(4, 1.0, 35, 100)],
        ),
        node(3, (2.0, 0.0, 0.0), 0, 100, vec![edge(1, 1.0, 0, 30)]),
        node(
            4,
            (2.0, 1.0, 0.0),
            35,
            100,
            vec![edge(3, 1.0, 35, 100), edge(2, 1.0, 35, 60)],
        ),
    ]
}

fn context(start: u64, end: u64) -> TraversalContext4D {
    TraversalContext4D {
        time_window: Some(TemporalWindow { start, end }),
        graph_weight: 1.0,
        spatial_weight: 0.25,
        temporal_weight: 0.0,
        ..TraversalContext4D::default()
    }
}

fn main() {
    let graph = dependency_graph();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&graph, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");
    let early = context(10, 20);
    let late = context(50, 60);

    let early_reachable = reachable_4d(&graph, 1, &early);
    let late_reachable = reachable_4d(&graph, 1, &late);
    let early_path = astar_find_path_4d(&graph, 1, 3, &early).expect("early path should exist");
    let late_path = astar_find_path_4d(&graph, 1, 3, &late).expect("late path should exist");
    let early_scc = strongly_connected_components_4d(&graph, &early);
    let late_scc = strongly_connected_components_4d(&graph, &late);

    assert_eq!(early_reachable, vec![1, 2, 3]);
    assert_eq!(late_reachable, vec![1, 2, 4, 3]);
    assert_eq!(early_path.node_ids, vec![1, 2, 3]);
    assert_eq!(late_path.node_ids, vec![1, 2, 4, 3]);
    assert!(early_scc
        .iter()
        .any(|component| component == &vec![1, 2, 3]));
    assert!(late_scc.iter().any(|component| component == &vec![2, 4]));

    let result = json!({
        "demo": "4d_dependency_timeline",
        "description": "Reachability, path search, and SCCs across dependency graph time windows",
        "windows": {
            "early": {
                "time_window": {"start": 10, "end": 20},
                "reachable_from_parser": early_reachable,
                "path_parser_to_storage": {
                    "nodes": early_path.node_ids,
                    "total_cost": early_path.total_cost,
                    "actual_cost": early_path.actual_cost,
                    "heuristic_cost": early_path.heuristic_cost
                },
                "strongly_connected_components": early_scc
            },
            "late": {
                "time_window": {"start": 50, "end": 60},
                "reachable_from_parser": late_reachable,
                "path_parser_to_storage": {
                    "nodes": late_path.node_ids,
                    "total_cost": late_path.total_cost,
                    "actual_cost": late_path.actual_cost,
                    "heuristic_cost": late_path.heuristic_cost
                },
                "strongly_connected_components": late_scc
            }
        }
    });

    if env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return;
    }

    println!("4D dependency timeline");
    println!(
        "time 10..20 reachable from parser: {:?}",
        result["windows"]["early"]["reachable_from_parser"]
    );
    println!(
        "time 10..20 path parser -> storage: {:?}",
        result["windows"]["early"]["path_parser_to_storage"]["nodes"]
    );
    println!(
        "time 10..20 SCCs: {:?}",
        result["windows"]["early"]["strongly_connected_components"]
    );
    println!(
        "time 50..60 reachable from parser: {:?}",
        result["windows"]["late"]["reachable_from_parser"]
    );
    println!(
        "time 50..60 path parser -> storage: {:?}",
        result["windows"]["late"]["path_parser_to_storage"]["nodes"]
    );
    println!(
        "time 50..60 SCCs: {:?}",
        result["windows"]["late"]["strongly_connected_components"]
    );
}
