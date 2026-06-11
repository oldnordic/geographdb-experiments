use geographdb_experiments::cypher_demo_common;

fn main() {
    let output = cypher_demo_common::run_queries(
        "4d_cypher_signal_propagation",
        "Cypher CALL query for time-dependent signal propagation",
        &[
            "CALL db.signal.propagate(1, departure: 10, spatial_sphere([0,0,0], 10.0)) YIELD reachable, unreachable RETURN reachable, unreachable",
        ],
    );
    cypher_demo_common::print_or_json(output);
}
