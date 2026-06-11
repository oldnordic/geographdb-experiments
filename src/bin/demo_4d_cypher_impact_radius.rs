use geographdb_experiments::cypher_demo_common;

fn main() {
    let output = cypher_demo_common::run_queries(
        "4d_cypher_impact_radius",
        "Cypher CALL query for 4D impact radius under space and time filters",
        &[
            "CALL db.impact.radius(1, spatial_sphere([0,0,0], 2.1), time_window(50, 60)) YIELD reachable RETURN reachable",
        ],
    );
    cypher_demo_common::print_or_json(output);
}
