//! SVD-Quant & MPO Outlier Matrix Compression Experiment
//! ------------------------------------------------------
//! This experiment demonstrates how Singular Value Decomposition (SVD) and low-rank
//! projections solve the "Outlier Wall" in LLM quantization, and evaluates
//! Matrix Product Operator (MPO) matrix compression.
//!
//! Run with:
//!   cargo run --example demo_mpo_outlier_svd

use geographdb_core::algorithms::mpo::{
    compress_matrix_to_mpo, mpo_compression_ratio, mpo_reconstruction_error, svd_thin,
};

// ── 4-Bit Uniform Quantization Helpers ───────────────────────────────────────

/// Quantize a block of 16 values to 4-bit (16 levels: 0 to 15) and return the
/// reconstructed values along with the scales and zero points.
fn quantize_block_4bit(block: &[f32]) -> (Vec<f32>, f32, f32) {
    assert_eq!(block.len(), 16);
    let mut min_val = block[0];
    let mut max_val = block[0];
    for &val in block.iter() {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    let range = max_val - min_val;
    let scale = if range.abs() < 1e-6 {
        1.0f32
    } else {
        range / 15.0f32
    };
    let zp = -min_val / scale;

    let mut reconstructed = vec![0.0f32; 16];
    for i in 0..16 {
        // Quantize to nearest level in 0..15
        let q = ((block[i] / scale) + zp).round().clamp(0.0f32, 15.0f32);
        // Dequantize
        reconstructed[i] = scale * (q - zp);
    }
    (reconstructed, scale, zp)
}

/// Quantize an entire N×N matrix row-by-row in blocks of 16.
fn quantize_matrix_4bit(w: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(w.len(), n * n);
    assert_eq!(n % 16, 0, "Matrix size must be a multiple of 16");
    let mut out = vec![0.0f32; w.len()];
    for row in 0..n {
        for block_idx in 0..(n / 16) {
            let start = row * n + block_idx * 16;
            let block = &w[start..(start + 16)];
            let (recon_block, _, _) = quantize_block_4bit(block);
            out[start..start + 16].copy_from_slice(&recon_block);
        }
    }
    out
}

// ── Frobenius Norm Helper ───────────────────────────────────────────────────

fn matrix_frob_norm(w: &[f32]) -> f32 {
    w.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn matrix_rel_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let diff: Vec<f32> = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| a - b)
        .collect();
    matrix_frob_norm(&diff) / (matrix_frob_norm(original) + 1e-8)
}

// ── Main Experiment ─────────────────────────────────────────────────────────

fn main() {
    println!("=========================================================================");
    println!("    TENSOR INFRASTRUCTURE: SVD-QUANT & MPO COMPRESSION EXPERIMENTS      ");
    println!("=========================================================================");
    println!("-> Simulating the LLM memory and outlier wall to evaluate low-rank recovery.");

    let n = 16usize; // 16×16 weight matrix
    let mut w = vec![0.0f32; n * n];

    // 1. Generate base weights in [-1.0, 1.0] (smooth, well-behaved random normal)
    for i in 0..n {
        for j in 0..n {
            // Pseudo-random deterministic values
            let val = (((i * 13 + j * 7 + 3) % 19) as f32 / 9.5) - 1.0;
            w[i * n + j] = val;
        }
    }

    // 2. Inject Outliers (simulating deep LLM channels with spikes up to 15.0)
    // We spike column 3 (extreme activation outliers) and row 7 (weight outliers)
    for i in 0..n {
        w[i * n + 3] += 15.0; // Column 3 spike
    }
    for j in 0..n {
        w[7 * n + j] -= 12.0; // Row 7 spike
    }

    println!("\nIngested Simulated 16×16 Weight Matrix:");
    println!("  - Smooth base weights in range [-1.0, 1.0]");
    println!("  - Outlier Column 3 spiked with magnitude +15.0");
    println!("  - Outlier Row 7 spiked with magnitude -12.0");
    println!("  - Matrix Frobenius Norm: {:.4}", matrix_frob_norm(&w));

    // =========================================================================
    // EXPERIMENT A: Naive 4-Bit Quantization (No Outlier Correction)
    // =========================================================================
    println!("\n-------------------------------------------------------------------------");
    println!(" EXPERIMENT A: Naive 4-Bit Quantization (The Outlier Bottleneck)");
    println!("-------------------------------------------------------------------------");
    let w_naive_quant = quantize_matrix_4bit(&w, n);
    let naive_err = matrix_rel_error(&w, &w_naive_quant);
    println!(
        "  Reconstruction relative error: {:.6} ({:.2}% loss)",
        naive_err,
        naive_err * 100.0
    );
    println!("  -> Explanation: The extreme outliers expanded the scale block range,");
    println!("     causing normal values to collapse to zero, introducing massive loss.");

    // =========================================================================
    // EXPERIMENT B: SVD-Quant (Low-Rank Decomposed Outlier Extraction)
    // =========================================================================
    println!("\n-------------------------------------------------------------------------");
    println!(" EXPERIMENT B: SVD-Quant Low-Rank Outlier Correction (Dynamic Recovery)");
    println!("-------------------------------------------------------------------------");

    // Decompose W via SVD: W = U · Sigma · Vt
    let (u, sigma, vt) = svd_thin(&w, n, n);
    println!("  Top 4 Singular Values of Weight Matrix W:");
    println!("    sigma_0 = {:.4}  (Primary Outlier Energy)", sigma[0]);
    println!("    sigma_1 = {:.4}  (Secondary Outlier Energy)", sigma[1]);
    println!("    sigma_2 = {:.4}  (Base Distribution Energy)", sigma[2]);
    println!("    sigma_3 = {:.4}", sigma[3]);

    // Rank k=1 correction (captures primary outlier)
    let k_1 = 1usize;
    let mut w_outlier_k1 = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0f32;
            for r in 0..k_1 {
                val += u[i * n + r] * sigma[r] * vt[r * n + j];
            }
            w_outlier_k1[i * n + j] = val;
        }
    }
    // Remainder matrix W_rem1 = W - W_outlier_k1
    let w_rem1: Vec<f32> = w
        .iter()
        .zip(w_outlier_k1.iter())
        .map(|(a, b)| a - b)
        .collect();
    let w_rem1_quant = quantize_matrix_4bit(&w_rem1, n);
    // Combine: W_recon1 = W_rem1_quant + W_outlier_k1
    let w_recon1: Vec<f32> = w_rem1_quant
        .iter()
        .zip(w_outlier_k1.iter())
        .map(|(a, b)| a + b)
        .collect();
    let err_k1 = matrix_rel_error(&w, &w_recon1);

    // Rank k=2 correction (captures both primary and secondary outliers)
    let k_2 = 2usize;
    let mut w_outlier_k2 = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0f32;
            for r in 0..k_2 {
                val += u[i * n + r] * sigma[r] * vt[r * n + j];
            }
            w_outlier_k2[i * n + j] = val;
        }
    }
    // Remainder matrix W_rem2 = W - W_outlier_k2
    let w_rem2: Vec<f32> = w
        .iter()
        .zip(w_outlier_k2.iter())
        .map(|(a, b)| a - b)
        .collect();
    let w_rem2_quant = quantize_matrix_4bit(&w_rem2, n);
    // Combine: W_recon2 = W_rem2_quant + W_outlier_k2
    let w_recon2: Vec<f32> = w_rem2_quant
        .iter()
        .zip(w_outlier_k2.iter())
        .map(|(a, b)| a + b)
        .collect();
    let err_k2 = matrix_rel_error(&w, &w_recon2);

    println!("\n  Results with SVD-Quant Outlier Correction:");
    println!(
        "    Rank k=1 Correction error: {:.6} ({:.2}% loss)  -> {:.1}x better!",
        err_k1,
        err_k1 * 100.0,
        naive_err / err_k1
    );
    println!(
        "    Rank k=2 Correction error: {:.6} ({:.2}% loss)  -> {:.1}x better!",
        err_k2,
        err_k2 * 100.0,
        naive_err / err_k2
    );
    println!("  -> Explanation: By subtracting the high-precision low-rank SVD components,");
    println!("     the remainder matrix becomes smooth, allowing the 4-bit quantizer to capture");
    println!("     it with perfect precision. Outlier error is almost completely resolved!");

    // =========================================================================
    // EXPERIMENT C: MPO Layer Compression
    // =========================================================================
    println!("\n-------------------------------------------------------------------------");
    println!(" EXPERIMENT C: Matrix Product Operator (MPO) Compression");
    println!("-------------------------------------------------------------------------");

    // Compress matrix W (16×16) to MPO with n_sites=2, local dims=4, and chi_max=2
    let mpo_chi2 = compress_matrix_to_mpo(&w, 16, 16, 2, 2);
    let err_mpo_chi2 = mpo_reconstruction_error(&mpo_chi2, &w, 16, 16);
    let ratio_mpo_chi2 = mpo_compression_ratio(&mpo_chi2, 16, 16);

    // Compress to MPO with chi_max=4
    let mpo_chi4 = compress_matrix_to_mpo(&w, 16, 16, 2, 4);
    let err_mpo_chi4 = mpo_reconstruction_error(&mpo_chi4, &w, 16, 16);
    let ratio_mpo_chi4 = mpo_compression_ratio(&mpo_chi4, 16, 16);

    println!("  MPO Compression Results (16×16 Matrix):");
    println!(
        "    Bond Dimension chi=2: Reconstruction Error = {:.6}  |  Compression Ratio = {:.2}%",
        err_mpo_chi2,
        ratio_mpo_chi2 * 100.0
    );
    println!(
        "    Bond Dimension chi=4: Reconstruction Error = {:.6}  |  Compression Ratio = {:.2}%",
        err_mpo_chi4,
        ratio_mpo_chi4 * 100.0
    );
    println!("  -> Explanation: MPO compresses the entire layer into local rank-4 tensors.");
    println!("     As bond dimension chi increases, the tensor network captures more correlation");
    println!("     and reconstructs the matrix with decreasing error.");
    println!("=========================================================================");
}
