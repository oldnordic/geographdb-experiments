use geographdb_core::{
    earliest_arrival_path_4d, fastest_temporal_path_4d, GraphNode4D, TemporalEdge,
};
use geographdb_core::{load_graph4d, save_graph4d};
use serde_json::json;
use std::env;
use tempfile::tempdir;

fn edge(dst: u64, travel_time: f32, begin_ts: u64, end_ts: u64) -> TemporalEdge {
    TemporalEdge {
        dst,
        weight: travel_time,
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

fn delivery_graph() -> Vec<GraphNode4D> {
    vec![
        node(
            1,
            (0.0, 0.0, 0.0),
            vec![edge(2, 4.0, 10, 30), edge(3, 4.0, 10, 18)],
        ),
        node(
            2,
            (1.0, 0.0, 0.0),
            vec![edge(6, 4.0, 10, 14), edge(5, 4.0, 22, 26)],
        ),
        node(3, (0.0, 2.0, 0.0), vec![edge(4, 5.0, 14, 25)]),
        node(4, (0.0, 4.0, 0.0), vec![edge(6, 4.0, 23, 40)]),
        node(5, (2.0, 0.0, 0.0), vec![edge(6, 4.0, 26, 40)]),
        node(6, (3.0, 0.0, 0.0), vec![]),
    ]
}

fn main() {
    let graph = delivery_graph();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&graph, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");
    let earliest = earliest_arrival_path_4d(&graph, 1, 6, 10, None)
        .expect("earliest-arrival temporal journey should exist");
    let fastest = fastest_temporal_path_4d(&graph, 1, 6, 10, 20, None)
        .expect("fastest temporal journey should exist");

    assert_eq!(earliest.node_ids, vec![1, 3, 4, 6]);
    assert_eq!(earliest.departure_time, 10);
    assert_eq!(earliest.arrival_time, 27);
    assert_eq!(earliest.duration, 17);
    assert_eq!(fastest.node_ids, vec![1, 2, 5, 6]);
    assert_eq!(fastest.departure_time, 18);
    assert_eq!(fastest.arrival_time, 30);
    assert_eq!(fastest.duration, 12);

    let earliest_nodes = earliest.node_ids.clone();
    let fastest_nodes = fastest.node_ids.clone();
    let result = json!({
        "demo": "4d_temporal_delivery",
        "description": "Temporal delivery where edge schedules make path feasibility causal",
        "start_node": 1,
        "goal_node": 6,
        "departure_search": {
            "earliest_departure": 10,
            "latest_departure": 20
        },
        "earliest_arrival": {
            "nodes": earliest.node_ids,
            "departure_time": earliest.departure_time,
            "arrival_time": earliest.arrival_time,
            "journey_duration": earliest.duration
        },
        "fastest_duration": {
            "nodes": fastest.node_ids,
            "departure_time": fastest.departure_time,
            "arrival_time": fastest.arrival_time,
            "journey_duration": fastest.duration
        },
        "rejected_routes": [
            {
                "nodes": [1, 2, 6],
                "reason": "edge 2->6 expires before the traveler can arrive at node 2 from node 1"
            }
        ]
    });

    if env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return;
    }

    println!("4D temporal delivery");
    println!(
        "earliest arrival: {:?}, depart {}, arrive {}, duration {}",
        earliest_nodes, earliest.departure_time, earliest.arrival_time, earliest.duration
    );
    println!(
        "fastest duration: {:?}, depart {}, arrive {}, duration {}",
        fastest_nodes, fastest.departure_time, fastest.arrival_time, fastest.duration
    );
    println!("rejected route: edge 2->6 expires before arrival at node 2");
}
