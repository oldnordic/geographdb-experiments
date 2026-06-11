//! Real Attention Outlier Test — OSCAR-Style Rotation for KV-Cache Quantization
//! -----------------------------------------------------------------------------
//! Simulates realistic LLM KV-cache vectors where a few channels have 30-100×
//! larger magnitude than the rest ("massive activation" outliers).
//!
//! Tests:
//!   - Naive INT2 quantization of K and V (no rotation)
//!   - Hadamard-rotation + INT2 (OSCAR-style: rotate → quantize → un-rotate)
//!   - Naive INT8 and Hadamard + INT8 for comparison
//!
//! Key metric: attention output cosine similarity vs float32 baseline.
//!
//! Run with:
//!   cargo run --example demo_real_outlier_attn

use geographdb_core::algorithms::oscar_rotation::{dequantize_int2, quantize_int2};

// ─── Hadamard rotation ────────────────────────────────────────────────────────

/// Normalised Walsh–Hadamard matrix H_d / √d  (d must be a power of two).
/// Property: W·W^T = I, and W·e_j = (1/√d)·(±1, ±1, …) — any standard-basis
/// spike is spread uniformly over all d dimensions.
fn hadamard_rotation(d: usize) -> Vec<f32> {
    assert!(d.is_power_of_two(), "d must be a power of two");
    // Build via iterative Kronecker product with [[1,1],[1,-1]]
    let mut h = vec![1.0f32]; // H_1
    let mut size = 1usize;
    while size < d {
        let new = size * 2;
        let mut h2 = vec![0.0f32; new * new];
        for i in 0..size {
            for j in 0..size {
                let v = h[i * size + j];
                h2[i * new + j] = v;
                h2[i * new + j + size] = v;
                h2[(i + size) * new + j] = v;
                h2[(i + size) * new + j + size] = -v;
            }
        }
        h = h2;
        size = new;
    }
    let scale = (d as f32).sqrt();
    h.iter_mut().for_each(|v| *v /= scale);
    h
}

// ─── Local math helpers ───────────────────────────────────────────────────────

/// R · x  (d×d row-major matrix times d-dim column vector)
fn mat_vec(r: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    (0..d)
        .map(|i| (0..d).map(|j| r[i * d + j] * x[j]).sum())
        .collect()
}

/// R^T · x
fn mat_t_vec(r: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    (0..d)
        .map(|i| (0..d).map(|j| r[j * d + i] * x[j]).sum())
        .collect()
}

// ─── Quantisation helpers ─────────────────────────────────────────────────────

/// INT8 per-token quantisation (256 levels, symmetric around midpoint).
fn quantize_int8(x: &[f32]) -> Vec<f32> {
    let min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scale = if (max - min).abs() < 1e-9 {
        1.0
    } else {
        (max - min) / 255.0
    };
    x.iter()
        .map(|&v| {
            let q = ((v - min) / scale).round().clamp(0.0, 255.0);
            q * scale + min
        })
        .collect()
}

fn quantize_int2_roundtrip(v: &[f32]) -> Vec<f32> {
    let (q, scale, zero) = quantize_int2(v);
    dequantize_int2(&q, scale, zero)
}

fn quant_tokens_int2(vecs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    vecs.iter().map(|v| quantize_int2_roundtrip(v)).collect()
}

fn quant_tokens_int8(vecs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    vecs.iter().map(|v| quantize_int8(v)).collect()
}

// ─── Attention ────────────────────────────────────────────────────────────────

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

/// Full causal-agnostic attention: O = softmax(Q·K^T / √d) · V
/// Q, K, V: T×D (each is a Vec<Vec<f32>> of length T, inner len D).
/// Returns T×D output.
fn attention(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>], d: usize) -> Vec<Vec<f32>> {
    let t = q.len();
    let scale = (d as f32).sqrt();
    (0..t)
        .map(|s| {
            // Scores for query s
            let scores: Vec<f32> = (0..t)
                .map(|tt| {
                    q[s].iter()
                        .zip(k[tt].iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / scale
                })
                .collect();
            let weights = softmax(&scores);
            // Weighted sum of values
            let mut out = vec![0.0f32; d];
            for (tt, &w) in weights.iter().enumerate() {
                for dim in 0..d {
                    out[dim] += w * v[tt][dim];
                }
            }
            out
        })
        .collect()
}

// ─── Data generation ─────────────────────────────────────────────────────────

/// Xorshift64 pseudo-random number generator (deterministic, no std).
fn xorshift(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32) - 0.5 // in [-0.5, 0.5]
}

/// T token vectors of dimension D.
/// Normal channels ∈ [-0.3, 0.3].
/// Outlier channels ∈ [-outlier_scale, +outlier_scale] — **bidirectional**:
/// sign varies per token, exactly as seen in LLM "massive activation" channels
/// (e.g. LLaMA2 channels 1, 32, 64 alternate sign across positions).
/// It is the *bidirectional* nature that matters: a symmetric range [-M, +M]
/// forces INT2/INT4 to spend its few levels on the full ±M span, leaving
/// normal channels with ≈ 1/M resolution. Rotation fixes this by spreading
/// the outlier energy uniformly, halving the per-channel max magnitude.
fn make_tokens(
    t: usize,
    d: usize,
    outlier_channels: &[usize],
    outlier_scale: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = seed;
    (0..t)
        .map(|_| {
            (0..d)
                .map(|ch| {
                    if outlier_channels.contains(&ch) {
                        // Bidirectional: full ±outlier_scale range, per-token sign varies
                        xorshift(&mut rng) * 2.0 * outlier_scale
                    } else {
                        xorshift(&mut rng) * 0.6 // ∈ [-0.3, 0.3]
                    }
                })
                .collect()
        })
        .collect()
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn mean_cosine(baseline: &[Vec<f32>], other: &[Vec<f32>]) -> f32 {
    baseline
        .iter()
        .zip(other.iter())
        .map(|(b, o)| cosine_sim(b, o))
        .sum::<f32>()
        / baseline.len() as f32
}

fn mean_l2(baseline: &[Vec<f32>], other: &[Vec<f32>]) -> f32 {
    baseline
        .iter()
        .zip(other.iter())
        .map(|(b, o)| {
            b.iter()
                .zip(o.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .sum::<f32>()
        / baseline.len() as f32
}

/// Per-channel mean absolute quantisation error across all tokens.
fn per_channel_quant_error(original: &[Vec<f32>], quantised: &[Vec<f32>], d: usize) -> Vec<f32> {
    let t = original.len();
    (0..d)
        .map(|c| {
            original
                .iter()
                .zip(quantised.iter())
                .map(|(o, q)| (o[c] - q[c]).abs())
                .sum::<f32>()
                / t as f32
        })
        .collect()
}

// ─── Rotation pipeline helpers ────────────────────────────────────────────────

fn rotate_all(r: &[f32], vecs: &[Vec<f32>], d: usize) -> Vec<Vec<f32>> {
    vecs.iter().map(|v| mat_vec(r, v, d)).collect()
}

fn unrotate_all(r: &[f32], vecs: &[Vec<f32>], d: usize) -> Vec<Vec<f32>> {
    vecs.iter().map(|v| mat_t_vec(r, v, d)).collect()
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let d: usize = 16; // head dimension (must be power of 2 for Hadamard)
    let t: usize = 32; // sequence length
    let outlier_channels = [3usize, 11]; // fixed outlier channels (simulating LLM massive activations)
    let outlier_scale = 30.0f32; // 100× normal channel magnitude (realistic: LLaMA2 outlier ratio)

    println!("══════════════════════════════════════════════════════════════════");
    println!("    Real Attention Outlier Test — OSCAR-Style KV-Cache Rotation  ");
    println!("══════════════════════════════════════════════════════════════════");
    println!(
        "  D={d}, T={t}, outlier channels: {outlier_channels:?}, outlier scale: {outlier_scale}×normal"
    );
    println!(
        "  Normal channels: ±0.3  |  Outlier channels: ±{:.0} (bidirectional, sign varies per token)",
        outlier_scale
    );
    println!(
        "  Outlier-to-normal ratio: {:.0}×  → naive INT2 scale = ±{:.0}, normal channels get ~1/{:.0} resolution",
        outlier_scale / 0.3,
        outlier_scale,
        (outlier_scale / 0.3) as u32
    );
    println!();

    // ── Data ────────────────────────────────────────────────────────────────
    // Q: no outliers (queries tend to be more uniform in LLMs)
    // K, V: outliers at fixed channels (cached activations accumulate them)
    let q = make_tokens(t, d, &[], 0.0, 0x1234_5678);
    let k = make_tokens(t, d, &outlier_channels, outlier_scale, 0xDEAD_BEEF);
    let v = make_tokens(t, d, &outlier_channels, outlier_scale, 0xCAFE_BABE);

    // ── Float32 baseline ────────────────────────────────────────────────────
    let o_float = attention(&q, &k, &v, d);

    // ── Rotation matrix ─────────────────────────────────────────────────────
    let r = hadamard_rotation(d); // W = H_16 / 4 — spreads any spike uniformly

    // ── Rotation-only sanity check (no quantisation → should be near-perfect) ─
    let k_rot_exact = rotate_all(&r, &k, d);
    let v_rot_exact = rotate_all(&r, &v, d);
    let q_rot = rotate_all(&r, &q, d); // Q must also rotate so Q·K^T is preserved
    let o_rot_only = unrotate_all(&r, &attention(&q_rot, &k_rot_exact, &v_rot_exact, d), d);
    let rot_only_cos = mean_cosine(&o_float, &o_rot_only);
    assert!(
        rot_only_cos > 0.9999,
        "Rotation-only (no quant) should preserve attention: cosine={rot_only_cos:.6}"
    );

    // ════════════════════════════════════════════════════════════════════════
    // EXPERIMENT 1: INT2 (4 levels) — the worst case
    // ════════════════════════════════════════════════════════════════════════
    println!("────────────────────────────────────────────────────────────────");
    println!("  EXPERIMENT 1: INT2 Quantisation (4 levels per token vector)   ");
    println!("────────────────────────────────────────────────────────────────");

    // Naive INT2
    let k_naive2 = quant_tokens_int2(&k);
    let v_naive2 = quant_tokens_int2(&v);
    let o_naive2 = attention(&q, &k_naive2, &v_naive2, d);

    // Hadamard + INT2
    let k_rot2 = quant_tokens_int2(&k_rot_exact);
    let v_rot2 = quant_tokens_int2(&v_rot_exact);
    let o_oscar2 = unrotate_all(&r, &attention(&q_rot, &k_rot2, &v_rot2, d), d);

    let naive2_cos = mean_cosine(&o_float, &o_naive2);
    let oscar2_cos = mean_cosine(&o_float, &o_oscar2);
    let naive2_l2 = mean_l2(&o_float, &o_naive2);
    let oscar2_l2 = mean_l2(&o_float, &o_oscar2);

    // Channel error breakdown
    let k_naive2_err = per_channel_quant_error(&k, &k_naive2, d);
    let k_rot2_err = per_channel_quant_error(&k_rot_exact, &k_rot2, d);
    let avg_naive2_k_err: f32 = k_naive2_err.iter().sum::<f32>() / d as f32;
    let avg_oscar2_k_err: f32 = k_rot2_err.iter().sum::<f32>() / d as f32;

    println!("  K quantisation error (mean |quant - original| per channel):");
    println!(
        "    Naive INT2   — normal channels: {:.4}  outlier channels: {:.4}  (avg: {:.4})",
        k_naive2_err
            .iter()
            .enumerate()
            .filter(|(c, _)| !outlier_channels.contains(c))
            .map(|(_, e)| e)
            .sum::<f32>()
            / (d - outlier_channels.len()) as f32,
        outlier_channels
            .iter()
            .map(|&c| k_naive2_err[c])
            .sum::<f32>()
            / outlier_channels.len() as f32,
        avg_naive2_k_err
    );
    println!(
        "    Hadamard+INT2 — all channels homogenised: avg = {:.4}",
        avg_oscar2_k_err
    );
    println!();
    println!("  Attention output vs float32 baseline:");
    println!(
        "    {:<22} cosine={:.4}  L2={:.4}",
        "Naive INT2:", naive2_cos, naive2_l2
    );
    println!(
        "    {:<22} cosine={:.4}  L2={:.4}  (+{:.4} Δcos, {:.2}× L2 reduction)",
        "Hadamard+INT2:",
        oscar2_cos,
        oscar2_l2,
        oscar2_cos - naive2_cos,
        naive2_l2 / oscar2_l2.max(1e-9)
    );

    // ════════════════════════════════════════════════════════════════════════
    // EXPERIMENT 2: INT8 (256 levels) — realistic LLM case
    // ════════════════════════════════════════════════════════════════════════
    println!();
    println!("────────────────────────────────────────────────────────────────");
    println!("  EXPERIMENT 2: INT8 Quantisation (256 levels per token vector) ");
    println!("────────────────────────────────────────────────────────────────");

    // Naive INT8
    let k_naive8 = quant_tokens_int8(&k);
    let v_naive8 = quant_tokens_int8(&v);
    let o_naive8 = attention(&q, &k_naive8, &v_naive8, d);

    // Hadamard + INT8
    let k_rot8 = quant_tokens_int8(&k_rot_exact);
    let v_rot8 = quant_tokens_int8(&v_rot_exact);
    let o_oscar8 = unrotate_all(&r, &attention(&q_rot, &k_rot8, &v_rot8, d), d);

    let naive8_cos = mean_cosine(&o_float, &o_naive8);
    let oscar8_cos = mean_cosine(&o_float, &o_oscar8);
    let naive8_l2 = mean_l2(&o_float, &o_naive8);
    let oscar8_l2 = mean_l2(&o_float, &o_oscar8);

    let k_naive8_err = per_channel_quant_error(&k, &k_naive8, d);
    let k_rot8_err = per_channel_quant_error(&k_rot_exact, &k_rot8, d);
    let avg_naive8_k_err: f32 = k_naive8_err.iter().sum::<f32>() / d as f32;
    let avg_oscar8_k_err: f32 = k_rot8_err.iter().sum::<f32>() / d as f32;

    println!("  K quantisation error per channel:");
    println!(
        "    Naive INT8   — normal channels: {:.4}  outlier channels: {:.4}  (avg: {:.4})",
        k_naive8_err
            .iter()
            .enumerate()
            .filter(|(c, _)| !outlier_channels.contains(c))
            .map(|(_, e)| e)
            .sum::<f32>()
            / (d - outlier_channels.len()) as f32,
        outlier_channels
            .iter()
            .map(|&c| k_naive8_err[c])
            .sum::<f32>()
            / outlier_channels.len() as f32,
        avg_naive8_k_err
    );
    println!(
        "    Hadamard+INT8 — all channels homogenised: avg = {:.4}",
        avg_oscar8_k_err
    );
    println!();
    println!("  Attention output vs float32 baseline:");
    println!(
        "    {:<22} cosine={:.4}  L2={:.4}",
        "Naive INT8:", naive8_cos, naive8_l2
    );
    println!(
        "    {:<22} cosine={:.4}  L2={:.4}  (+{:.4} Δcos, {:.2}× L2 reduction)",
        "Hadamard+INT8:",
        oscar8_cos,
        oscar8_l2,
        oscar8_cos - naive8_cos,
        naive8_l2 / oscar8_l2.max(1e-9)
    );

    // ════════════════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════════════════
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  Summary: rotation improvement at {outlier_scale}× outlier ratio");
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  {:>18}  {:>10}  {:>10}  {:>10}",
        "Method", "Cosine↑", "L2 Err↓", "Δ Cosine"
    );
    println!("  {}", "─".repeat(54));
    println!(
        "  {:>18}  {:>10.4}  {:>10.4}  {:>10}",
        "Float32 baseline", 1.0, 0.0, "—"
    );
    println!(
        "  {:>18}  {:>10.4}  {:>10.4}  {:>10}",
        "Naive INT2", naive2_cos, naive2_l2, "—"
    );
    println!(
        "  {:>18}  {:>10.4}  {:>10.4}  {:>+10.4}",
        "Hadamard+INT2",
        oscar2_cos,
        oscar2_l2,
        oscar2_cos - naive2_cos
    );
    println!(
        "  {:>18}  {:>10.4}  {:>10.4}  {:>10}",
        "Naive INT8", naive8_cos, naive8_l2, "—"
    );
    println!(
        "  {:>18}  {:>10.4}  {:>10.4}  {:>+10.4}",
        "Hadamard+INT8",
        oscar8_cos,
        oscar8_l2,
        oscar8_cos - naive8_cos
    );
    println!();
    println!(
        "  INT2 rotation improvement: {:.2}× L2 reduction  ({:+.1}% cosine gain)",
        naive2_l2 / oscar2_l2.max(1e-9),
        (oscar2_cos - naive2_cos) * 100.0
    );
    println!(
        "  INT8 rotation improvement: {:.2}× L2 reduction  ({:+.1}% cosine gain)",
        naive8_l2 / oscar8_l2.max(1e-9),
        (oscar8_cos - naive8_cos) * 100.0
    );
    println!("══════════════════════════════════════════════════════════════════");

    // ─── Assertions ──────────────────────────────────────────────────────────
    assert!(
        oscar2_cos > naive2_cos,
        "Hadamard rotation must improve INT2 attention cosine: oscar={oscar2_cos:.4} naive={naive2_cos:.4}"
    );
    assert!(
        oscar8_cos > naive8_cos,
        "Hadamard rotation must improve INT8 attention cosine: oscar={oscar8_cos:.4} naive={naive8_cos:.4}"
    );
    assert!(
        oscar2_l2 < naive2_l2,
        "Hadamard rotation must reduce INT2 attention L2 error: oscar={oscar2_l2:.4} naive={naive2_l2:.4}"
    );
    assert!(
        oscar8_l2 < naive8_l2,
        "Hadamard rotation must reduce INT8 attention L2 error: oscar={oscar8_l2:.4} naive={naive8_l2:.4}"
    );
}
