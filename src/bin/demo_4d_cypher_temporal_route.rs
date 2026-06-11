use geographdb_experiments::cypher_demo_common;

fn main() {
    let output = cypher_demo_common::run_queries(
        "4d_cypher_temporal_route",
        "Cypher CALL query for time-respecting temporal routing",
        &[
            "CALL db.route.temporal(1, 6, departure: 10) YIELD path, arrival_time, duration RETURN path, arrival_time",
        ],
    );
    cypher_demo_common::print_or_json(output);
}
