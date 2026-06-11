use geographdb_experiments::cypher_demo_common;

fn main() {
    let output = cypher_demo_common::run_queries(
        "4d_cypher_bottlenecks",
        "Cypher CALL query for temporal bottleneck resilience",
        &[
            "CALL db.resilience.bottlenecks(spatial_sphere([0,0,0], 10.0), time_window(10, 20)) YIELD articulation_points, bridges RETURN articulation_points, bridges",
            "CALL db.resilience.bottlenecks(spatial_sphere([0,0,0], 10.0), time_window(50, 60)) YIELD articulation_points, bridges RETURN articulation_points, bridges",
        ],
    );
    cypher_demo_common::print_or_json(output);
}
