use geographdb_core::algorithms::oscar_rotation::{
    compress_rotation, covariance_to_rotation, empirical_query_covariance,
    identity_quantization_error, mpo_compression_ratio_rotation, rotation_fidelity,
    rotation_quantization_error,
};

fn main() {
    let d = 16;
    let n_sites = 2;

    // Simulate calibration queries: attention query vectors from two "heads"
    let queries: Vec<Vec<f32>> = (0..32)
        .map(|i| {
            let mut q = vec![0.0f32; d];
            // Head 0: concentrated on dims 0-7 with outlier at dim 0
            if i < 16 {
                q[0] = 5.0 * (i as f32 / 16.0);
                for (k, qk) in q[1..8].iter_mut().enumerate().map(|(idx, v)| (idx + 1, v)) {
                    *qk = 0.5 * (k as f32) * (i as f32 * 0.3 + k as f32).sin();
                }
            } else {
                // Head 1: concentrated on dims 8-15
                q[8] = 5.0 * ((i - 16) as f32 / 16.0);
                for (k, qk) in q[9..16].iter_mut().enumerate().map(|(idx, v)| (idx + 9, v)) {
                    *qk = 0.5 * (k as f32) * ((i - 16) as f32 * 0.3 + k as f32).cos();
                }
            }
            q
        })
        .collect();

    let cov = empirical_query_covariance(&queries, d);
    let v = covariance_to_rotation(&cov, d);

    // Build test keys: random-ish vectors representing KV cache entries
    let test_keys: Vec<Vec<f32>> = (0..20)
        .map(|i| {
            (0..d)
                .map(|j| (i as f32 * 1.7 + j as f32 * 0.9).sin())
                .collect()
        })
        .collect();

    println!("OSCAR Rotation — MPO Compression Experiment");
    println!(
        "d={d}, n_sites={n_sites}, calibration queries={}",
        queries.len()
    );
    println!();

    // Fidelity vs chi_max
    println!(
        "{:<10} {:<16} {:<20}",
        "chi_max", "fidelity", "compression_ratio"
    );
    println!("{}", "-".repeat(46));
    for chi in [1, 2, 4, 8, 16] {
        let rot = compress_rotation(&v, d, n_sites, chi);
        let fid = rotation_fidelity(&v, &rot, &test_keys);
        let ratio = mpo_compression_ratio_rotation(&rot);
        println!("{:<10} {:<16.4} {:<20.4}", chi, fid, ratio);
    }

    println!();

    // Quantization error: rotation vs identity (no rotation)
    println!("Quantization error (L2) after INT2 on rotated vs raw key:");
    println!(
        "{:<8} {:<22} {:<22} {:<10}",
        "key", "with_rotation", "without_rotation", "improvement"
    );
    println!("{}", "-".repeat(62));
    for (i, k) in test_keys.iter().take(8).enumerate() {
        let err_rot = rotation_quantization_error(&v, k, d);
        let err_id = identity_quantization_error(k);
        let improvement = err_id / err_rot.max(1e-9);
        println!(
            "{:<8} {:<22.5} {:<22.5} {:.2}x",
            i, err_rot, err_id, improvement
        );
    }

    println!();
    println!("Rotation aligns KV cache dimensions with principal attention axes,");
    println!("reducing outlier impact and improving INT2 quantization fidelity.");
    println!(
        "MPO compression with chi_max=4 gives ~{}x fewer parameters than dense.",
        (1.0 / mpo_compression_ratio_rotation(&compress_rotation(&v, d, n_sites, 4))) as usize
    );
}
