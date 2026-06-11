use geographdb_core::{
    load_graph4d, save_graph4d, time_dependent_dijkstra_4d, GraphNode4D, TemporalEdge,
};
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

fn signal_graph() -> Vec<GraphNode4D> {
    vec![
        node(
            1,
            (0.0, 0.0, 0.0),
            vec![edge(2, 4.0, 10, 30), edge(4, 6.0, 10, 30)],
        ),
        node(2, (1.0, 0.0, 0.0), vec![edge(3, 5.0, 14, 30)]),
        node(3, (2.0, 0.0, 0.0), vec![edge(6, 4.0, 10, 18)]),
        node(4, (0.0, 2.0, 0.0), vec![edge(5, 8.0, 16, 40)]),
        node(5, (1.0, 2.0, 0.0), vec![]),
        node(6, (3.0, 0.0, 0.0), vec![]),
    ]
}

fn frontier(arrivals: &[(u64, u64)], threshold: u64) -> Vec<u64> {
    arrivals
        .iter()
        .filter_map(|(node, arrival_time)| (*arrival_time <= threshold).then_some(*node))
        .collect()
}

fn main() {
    let start_node = 1;
    let departure_time = 10;
    let graph = signal_graph();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&graph, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");
    let result = time_dependent_dijkstra_4d(&graph, start_node, departure_time, None)
        .expect("start node should be usable");

    let arrivals: Vec<(u64, u64)> = result
        .reachable
        .iter()
        .map(|arrival| (arrival.node_id, arrival.arrival_time))
        .collect();
    let earliest_arrivals: Vec<_> = result
        .reachable
        .iter()
        .map(|arrival| {
            json!({
                "node": arrival.node_id,
                "arrival_time": arrival.arrival_time,
                "cost": arrival.cost,
                "path": arrival.path
            })
        })
        .collect();
    let unreachable: Vec<_> = result
        .unreachable
        .iter()
        .map(|node| {
            json!({
                "node": node,
                "reason": "all incoming edges expire before arrival"
            })
        })
        .collect();

    assert_eq!(frontier(&arrivals, 15), vec![1, 2]);
    assert_eq!(frontier(&arrivals, 20), vec![1, 2, 4, 3]);
    assert_eq!(frontier(&arrivals, 30), vec![1, 2, 4, 3, 5]);
    assert_eq!(result.unreachable, vec![6]);

    let json_result = json!({
        "demo": "4d_signal_propagation",
        "description": "Time-dependent Dijkstra computes earliest arrival to all reachable nodes under temporal edge schedules",
        "start_node": start_node,
        "departure_time": departure_time,
        "earliest_arrivals": earliest_arrivals,
        "unreachable": unreachable,
        "arrival_frontiers": {
            "t<=15": frontier(&arrivals, 15),
            "t<=20": frontier(&arrivals, 20),
            "t<=30": frontier(&arrivals, 30)
        }
    });

    if env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&json_result).unwrap());
        return;
    }

    println!("4D signal propagation");
    for arrival in &result.reachable {
        println!(
            "node {} reached at t={} via {:?}",
            arrival.node_id, arrival.arrival_time, arrival.path
        );
    }
    println!("unreachable: {:?}", result.unreachable);
    println!("frontier t<=15: {:?}", frontier(&arrivals, 15));
    println!("frontier t<=20: {:?}", frontier(&arrivals, 20));
    println!("frontier t<=30: {:?}", frontier(&arrivals, 30));
}
