//! Compare distance-based walker selection with Rodrigues parallel transport.
//!
//! Graph layout:
//!
//!     A -- B -- C   (straight line)
//!           |
//!           D       (90° turn in y)
//!           |
//!           E       (another 90° turn in z)
//!
//! From B, geometric transport should prefer C (continuation) over D or E
//! because the rotation angle is smaller. Distance-based scoring has no such
//! preference if all candidates are at the same Euclidean distance.

use geographdb_core::algorithms::four_d::{GraphProperties, TemporalEdge};
use geographdb_core::corpus::{
    build_edge_weights, build_node_index, build_octree, GeometricWalker, TransitionMode,
    WalkerConfig,
};
use geographdb_core::GraphNode4D;

fn make_node(id: u64, x: f32, y: f32, z: f32) -> GraphNode4D {
    GraphNode4D {
        id,
        x,
        y,
        z,
        begin_ts: 0,
        end_ts: u64::MAX,
        properties: GraphProperties::default(),
        successors: Vec::new(),
    }
}

fn main() {
    let mut a = make_node(1000, 0.0, 0.0, 0.0);
    let mut b = make_node(2000, 1.0, 0.0, 0.0);
    let c = make_node(3000, 2.0, 0.0, 0.0);
    let mut d = make_node(4000, 1.0, 1.0, 0.0);
    let e = make_node(5000, 1.0, 1.0, 1.0);

    a.successors.push(TemporalEdge {
        dst: 2000,
        weight: 1.0,
        begin_ts: 0,
        end_ts: u64::MAX,
    });
    b.successors.push(TemporalEdge {
        dst: 3000,
        weight: 1.0,
        begin_ts: 0,
        end_ts: u64::MAX,
    });
    b.successors.push(TemporalEdge {
        dst: 4000,
        weight: 1.0,
        begin_ts: 0,
        end_ts: u64::MAX,
    });
    d.successors.push(TemporalEdge {
        dst: 5000,
        weight: 1.0,
        begin_ts: 0,
        end_ts: u64::MAX,
    });

    let graph = vec![a, b, c, d, e];
    let idx = build_node_index(&graph);
    let octree = build_octree(&graph);
    let weights = build_edge_weights(&graph);

    let mut distance_config = WalkerConfig::default();
    distance_config.knn = 0;
    distance_config.plan_interval = 0;
    distance_config.temperature = 0.01;

    let mut rodrigues_config = distance_config;
    rodrigues_config.transition_mode = TransitionMode::RodriguesTransport { lambda: 2.0 };

    let distance_traj = GeometricWalker::walk_beam(
        &graph,
        &idx,
        &octree,
        &weights,
        None,
        &graph[0],
        2,
        1,
        &distance_config,
    );

    let rodrigues_traj = GeometricWalker::walk_beam(
        &graph,
        &idx,
        &octree,
        &weights,
        None,
        &graph[0],
        2,
        1,
        &rodrigues_config,
    );

    println!("Distance-KNN trajectory: {:?}", distance_traj);
    println!("Rodrigues transport trajectory: {:?}", rodrigues_traj);

    assert_eq!(
        rodrigues_traj,
        vec![1000, 2000, 3000],
        "Rodrigues transport should continue straight"
    );
}
