use geographdb_core::{
    load_graph4d, query_4d, save_graph4d, GeoCypherResult, GraphNode4D, TemporalEdge,
};
use serde_json::{json, Value};
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
    let properties = [
        ("name".to_string(), json!(format!("relay-{id}"))),
        ("kind".to_string(), json!("drone-relay")),
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

pub fn demo_graph() -> Vec<GraphNode4D> {
    vec![
        node(
            1,
            (0.0, 0.0, 0.0),
            vec![edge(2, 1.0, 0, 100), edge(4, 1.0, 40, 100)],
        ),
        node(2, (1.0, 0.0, 0.0), vec![edge(3, 1.0, 0, 100)]),
        node(3, (2.0, 0.0, 0.0), vec![edge(6, 1.0, 0, 30)]),
        node(4, (0.0, 2.0, 0.0), vec![edge(5, 1.0, 0, 100)]),
        node(
            5,
            (1.0, 2.0, 0.0),
            vec![edge(3, 1.0, 40, 100), edge(6, 1.0, 40, 100)],
        ),
        node(6, (3.0, 0.0, 0.0), vec![]),
    ]
}

pub fn result_to_json(result: GeoCypherResult) -> Value {
    match result {
        GeoCypherResult::NodeIds(nodes) => json!({ "type": "node_ids", "nodes": nodes }),
        GeoCypherResult::Edges(edges) => json!({ "type": "edges", "edges": edges }),
        GeoCypherResult::Rows(rows) => json!({ "type": "rows", "rows": rows }),
        GeoCypherResult::Path(path) => json!({
            "type": "path",
            "nodes": path.node_ids,
            "total_cost": path.total_cost,
            "actual_cost": path.actual_cost,
            "heuristic_cost": path.heuristic_cost
        }),
        GeoCypherResult::Journey(journey) => json!({
            "type": "journey",
            "nodes": journey.node_ids,
            "departure_time": journey.departure_time,
            "arrival_time": journey.arrival_time,
            "duration": journey.duration
        }),
        GeoCypherResult::Dijkstra(result) => json!({
            "type": "dijkstra",
            "start_node": result.start_node,
            "departure_time": result.departure_time,
            "reachable": result.reachable.iter().map(|arrival| json!({
                "node": arrival.node_id,
                "arrival_time": arrival.arrival_time,
                "cost": arrival.cost,
                "path": arrival.path
            })).collect::<Vec<_>>(),
            "unreachable": result.unreachable
        }),
        GeoCypherResult::Bottlenecks {
            articulation_points,
            bridges,
        } => json!({
            "type": "bottlenecks",
            "articulation_points": articulation_points,
            "bridges": bridges
        }),
        GeoCypherResult::Components(components) => {
            json!({ "type": "components", "components": components })
        }
    }
}

pub fn run_queries(demo: &str, description: &str, queries: &[&str]) -> Value {
    let graph = demo_graph();
    let tmp_dir = tempdir().unwrap();
    save_graph4d(&graph, tmp_dir.path()).expect("save graph");
    let graph = load_graph4d(tmp_dir.path()).expect("load graph");
    let results: Vec<_> = queries
        .iter()
        .map(|query| {
            let result = query_4d(&graph, query).expect("query should execute");
            json!({
                "query": query,
                "result": result_to_json(result)
            })
        })
        .collect();

    json!({
        "demo": demo,
        "description": description,
        "queries": results
    })
}

pub fn print_or_json(output: Value) {
    if std::env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return;
    }

    println!(
        "{}",
        output["description"].as_str().unwrap_or("4D Cypher demo")
    );
    for item in output["queries"].as_array().unwrap() {
        println!("{}", item["query"].as_str().unwrap());
        println!("{}", serde_json::to_string_pretty(&item["result"]).unwrap());
    }
}
