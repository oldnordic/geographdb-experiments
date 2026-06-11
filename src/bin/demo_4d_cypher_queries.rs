use geographdb_core::{query_4d, GeoCypherResult, GraphNode4D, TemporalEdge};
use serde_json::{json, Value};
use std::env;

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

fn graph() -> Vec<GraphNode4D> {
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

fn result_to_json(result: GeoCypherResult) -> Value {
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

fn main() {
    let graph = graph();
    let queries = [
        "MATCH (a)-[:CONNECTS]->(b) RETURN a, b",
        "MATCH (a)-[:CONNECTS]->(b) WHERE time_window(10, 60) RETURN a, b",
        "MATCH (a)-[:CONNECTS]->(b) WHERE spatial_sphere([0.0, 0.0, 0.0], 2.1) RETURN a, b",
        "CALL db.route.temporal(1, 6, departure: 10) YIELD path, arrival_time, duration RETURN path, arrival_time",
        "CALL db.impact.radius(1, spatial_sphere([0,0,0], 2.1), time_window(50, 60)) YIELD reachable RETURN reachable",
        "CALL db.resilience.bottlenecks(spatial_sphere([0,0,0], 10.0), time_window(10, 20)) YIELD articulation_points, bridges RETURN articulation_points, bridges",
        "CALL db.signal.propagate(1, departure: 10, spatial_sphere([0,0,0], 10.0)) YIELD reachable, unreachable RETURN reachable, unreachable",
    ];

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

    let output = json!({
        "demo": "4d_cypher_queries",
        "description": "Small Cypher-like 4D query layer dispatching to geographdb-core algorithms",
        "queries": results
    });

    if env::args().any(|arg| arg == "--json") {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return;
    }

    println!("4D Cypher-like queries");
    for item in output["queries"].as_array().unwrap() {
        println!("{}", item["query"].as_str().unwrap());
        println!("{}", serde_json::to_string_pretty(&item["result"]).unwrap());
    }
}
