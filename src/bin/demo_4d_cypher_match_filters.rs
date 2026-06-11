use geographdb_experiments::cypher_demo_common;

fn main() {
    let output = cypher_demo_common::run_queries(
        "4d_cypher_match_filters",
        "Cypher MATCH queries with temporal and spatial filters",
        &[
            "MATCH (a)-[:CONNECTS]->(b) RETURN a.name, b.name",
            "MATCH (a)-[:CONNECTS]->(b) WHERE time_window(10, 60) RETURN a, b",
            "MATCH (a)-[:CONNECTS]->(b) WHERE spatial_sphere([0.0, 0.0, 0.0], 2.1) RETURN a, b",
        ],
    );
    cypher_demo_common::print_or_json(output);
}
