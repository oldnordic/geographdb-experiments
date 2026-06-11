//! KV Cache MPS Compression — Real-Sequence Validation
//! ------------------------------------------------------
//! Loads a KV cache dump produced by rocmforge (`--kv-dump <path>`) and
//! measures how well MPS (online low-rank basis factorization) reconstructs
//! attention outputs at various bond dimensions χ.
//!
//! Falls back to synthetic sinusoidal KV sequences when no dump file is given,
//! which is enough to validate the algorithm without requiring a GPU run.
//!
//! Run with:
//!   cargo run --example demo_kv_compression
//!   cargo run --example demo_kv_compression -- --kv-dump /tmp/kv.bin
//!   cargo run --example demo_kv_compression -- --kv-dump /tmp/kv.bin --layer 0

use geographdb_core::algorithms::kv_cache_mps::KvCacheMps;
use std::path::Path;

// ── KvDump reader (mirrors rocmforge's format) ────────────────────────────────

const KV_DUMP_MAGIC: &[u8; 8] = b"KVCACHE1";

struct KvDump {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_tokens: usize,
    /// k[layer] is a flat Vec<f32> of length num_tokens × num_kv_heads × head_dim
    k: Vec<Vec<f32>>,
    /// v[layer] same shape
    v: Vec<Vec<f32>>,
}

impl KvDump {
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 32 {
            return Err("file too short to contain header".into());
        }
        if &bytes[0..8] != KV_DUMP_MAGIC {
            return Err(format!(
                "bad magic: expected {:?}, got {:?}",
                KV_DUMP_MAGIC,
                &bytes[0..8]
            )
            .into());
        }
        let read_u32 = |off: usize| -> usize {
            u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize
        };
        let num_layers = read_u32(8);
        let num_kv_heads = read_u32(12);
        let head_dim = read_u32(16);
        let num_tokens = read_u32(20);

        let floats_per_layer = num_tokens * num_kv_heads * head_dim;
        let expected = 32 + 2 * num_layers * floats_per_layer * 4;
        if bytes.len() < expected {
            return Err(format!("truncated: need {} bytes, have {}", expected, bytes.len()).into());
        }

        let mut cursor = 32usize;
        let read_layer = |c: &mut usize| -> Vec<f32> {
            let n = floats_per_layer;
            let slice = &bytes[*c..*c + n * 4];
            *c += n * 4;
            slice
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };

        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(read_layer(&mut cursor));
        }
        for _ in 0..num_layers {
            v.push(read_layer(&mut cursor));
        }

        Ok(Self {
            num_layers,
            num_kv_heads,
            head_dim,
            num_tokens,
            k,
            v,
        })
    }

    /// Key vector for a single head at a given token.
    fn key_head(&self, layer: usize, head: usize, token: usize) -> &[f32] {
        let stride = self.num_kv_heads * self.head_dim;
        let off = token * stride + head * self.head_dim;
        &self.k[layer][off..off + self.head_dim]
    }

    /// Value vector for a single head at a given token.
    fn val_head(&self, layer: usize, head: usize, token: usize) -> &[f32] {
        let stride = self.num_kv_heads * self.head_dim;
        let off = token * stride + head * self.head_dim;
        &self.v[layer][off..off + self.head_dim]
    }
}

// ── Synthetic fallback ────────────────────────────────────────────────────────

/// Sinusoidal KV sequence — dim-dimensional, `n_tokens` steps.
///
/// Keys: unit-norm vectors whose components rotate in a low-dimensional subspace.
/// Values: same basis plus a positive constant term so attention outputs are
///         well above zero (avoids division-by-near-zero in relative error).
fn synthetic_kv(n_tokens: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // Use the first min(8, dim) sinusoidal modes as a low-rank basis.
    let n_modes = dim.min(8);
    let keys = (0..n_tokens)
        .map(|t| {
            let mut row = vec![0.0f32; dim];
            for (m, r) in row.iter_mut().enumerate().take(n_modes) {
                let freq = 1.0 + m as f32 * 0.3;
                *r = (freq * t as f32 * 0.2).sin();
            }
            // Normalize to unit sphere so scales are consistent
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            row.iter_mut().for_each(|x| *x /= norm);
            row
        })
        .collect();
    let vals = (0..n_tokens)
        .map(|t| {
            let mut row = vec![0.5f32; dim]; // positive DC offset so ‖attn_out‖ ≫ 0
            for (m, r) in row.iter_mut().enumerate().take(n_modes) {
                let freq = 1.0 + m as f32 * 0.3;
                *r += 0.3 * (freq * t as f32 * 0.2 + 1.0).sin();
            }
            row
        })
        .collect();
    (keys, vals)
}

// ── Evaluation ────────────────────────────────────────────────────────────────

/// Compute mean relative error between two equal-length slices.
fn relative_error(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2) as f64)
        .sum::<f64>()
        .sqrt();
    let den: f64 = a.iter().map(|x| x.powi(2) as f64).sum::<f64>().sqrt();
    if den < 1e-12 {
        0.0
    } else {
        num / den
    }
}

/// Build a full KV cache and compute attention for one query, return the output.
fn full_attend(keys: &[Vec<f32>], vals: &[Vec<f32>], query: &[f32], scale: f32) -> Vec<f32> {
    let d_v = vals[0].len();
    // Softmax(Q·K^T / sqrt(d)) · V
    let scores: Vec<f32> = keys
        .iter()
        .map(|k| {
            query
                .iter()
                .zip(k.iter())
                .map(|(q, ki)| q * ki)
                .sum::<f32>()
                * scale
        })
        .collect();
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
    let sum_exp: f32 = exp.iter().sum();
    let weights: Vec<f32> = exp.iter().map(|e| e / sum_exp).collect();
    let mut out = vec![0.0f32; d_v];
    for (w, v) in weights.iter().zip(vals.iter()) {
        for (o, vi) in out.iter_mut().zip(v.iter()) {
            *o += w * vi;
        }
    }
    out
}

/// Evaluate MPS compression of a KV sequence at bond dimension chi.
/// Returns (mean_relative_error, compression_ratio).
fn evaluate_chi(keys: &[Vec<f32>], vals: &[Vec<f32>], chi: usize, n_queries: usize) -> (f64, f64) {
    let d_k = keys[0].len();
    let d_v = vals[0].len();
    let scale = 1.0 / (d_k as f32).sqrt();

    // Build MPS cache
    let mut cache = KvCacheMps::new(d_k, d_v, chi);
    for (k, v) in keys.iter().zip(vals.iter()) {
        cache.append(k, v);
    }

    // Sample n_queries random queries (deterministic: use key[t % T] as query)
    let mut total_err = 0.0f64;
    let n = keys.len();
    for qi in 0..n_queries {
        let query = &keys[qi % n];
        let ref_out = full_attend(keys, vals, query, scale);
        let mps_out = cache.attend(query, scale);
        total_err += relative_error(&ref_out, &mps_out);
    }

    let mean_err = total_err / n_queries as f64;
    let ratio = cache.compression_ratio();
    (mean_err, ratio)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dump_path: Option<String> = None;
    let mut target_layer: Option<usize> = None;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--kv-dump" => dump_path = args.next(),
            "--layer" => target_layer = args.next().and_then(|s| s.parse().ok()),
            other => {
                eprintln!("Unknown flag: {other}");
                eprintln!("Usage: demo_kv_compression [--kv-dump <path>] [--layer N]");
                std::process::exit(1);
            }
        }
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        KV CACHE MPS COMPRESSION — REAL-SEQUENCE VALIDATION      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Data source ───────────────────────────────────────────────────────────
    enum Source {
        Dump(KvDump),
        Synthetic {
            keys: Vec<Vec<f32>>,
            vals: Vec<Vec<f32>>,
            n_tokens: usize,
            head_dim: usize,
        },
    }

    let source = if let Some(ref path) = dump_path {
        eprint!("Loading KV dump from {}... ", path);
        match KvDump::load(Path::new(path)) {
            Ok(dump) => {
                eprintln!(
                    "ok ({} layers, {} kv-heads, {} head-dim, {} tokens)",
                    dump.num_layers, dump.num_kv_heads, dump.head_dim, dump.num_tokens
                );
                Source::Dump(dump)
            }
            Err(e) => {
                eprintln!("failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        let n_tokens = 64;
        let head_dim = 32;
        eprintln!("No --kv-dump provided; using synthetic sinusoidal sequence");
        eprintln!("  ({n_tokens} tokens × {head_dim} head-dim per head)");
        let (keys, vals) = synthetic_kv(n_tokens, head_dim);
        Source::Synthetic {
            keys,
            vals,
            n_tokens,
            head_dim,
        }
    };

    // ── Sweep chi for each requested layer ────────────────────────────────────
    let chi_values = [1, 2, 4, 8, 16, 32];
    let n_queries = 8;

    match source {
        Source::Dump(ref dump) => {
            let layers: Vec<usize> = if let Some(l) = target_layer {
                vec![l]
            } else {
                // Evenly sample up to 4 layers
                let step = (dump.num_layers.max(1) - 1) / 3;
                (0..4)
                    .map(|i| (i * step).min(dump.num_layers - 1))
                    .collect()
            };

            for layer in layers {
                println!("── Layer {layer} ──────────────────────────────────────────────────────");
                println!(
                    "  Sequence: {} tokens, {} kv-heads, {} head-dim",
                    dump.num_tokens, dump.num_kv_heads, dump.head_dim
                );

                // Evaluate per-head and average
                println!(
                    "  {:>4}  {:>12}  {:>16}  status",
                    "χ", "rel. error", "compression"
                );
                for &chi in &chi_values {
                    let mut total_err = 0.0f64;
                    for head in 0..dump.num_kv_heads {
                        let keys: Vec<Vec<f32>> = (0..dump.num_tokens)
                            .map(|t| dump.key_head(layer, head, t).to_vec())
                            .collect();
                        let vals: Vec<Vec<f32>> = (0..dump.num_tokens)
                            .map(|t| dump.val_head(layer, head, t).to_vec())
                            .collect();
                        let (err, _) = evaluate_chi(&keys, &vals, chi, n_queries);
                        total_err += err;
                    }
                    let mean_err = total_err / dump.num_kv_heads as f64;

                    // Compression ratio for one head (same for all at same chi)
                    let keys: Vec<Vec<f32>> = (0..dump.num_tokens)
                        .map(|t| dump.key_head(layer, 0, t).to_vec())
                        .collect();
                    let vals: Vec<Vec<f32>> = (0..dump.num_tokens)
                        .map(|t| dump.val_head(layer, 0, t).to_vec())
                        .collect();
                    let (_, ratio) = evaluate_chi(&keys, &vals, chi, 1);

                    let status = if mean_err < 0.01 {
                        "✓ <1%"
                    } else if mean_err < 0.05 {
                        "~ <5%"
                    } else {
                        "✗ >5%"
                    };
                    println!(
                        "  {:>4}  {:>11.4}%  {:>15.2}x  {}",
                        chi,
                        mean_err * 100.0,
                        ratio,
                        status
                    );
                }
                println!();
            }
        }

        Source::Synthetic {
            ref keys,
            ref vals,
            n_tokens,
            head_dim,
        } => {
            println!("── Synthetic sequence ──────────────────────────────────────────────────");
            println!("  {n_tokens} tokens × {head_dim} head-dim  (sinusoidal low-rank basis)");
            println!();
            println!(
                "  {:>4}  {:>12}  {:>16}  status",
                "χ", "rel. error", "compression"
            );
            for &chi in &chi_values {
                let (err, ratio) = evaluate_chi(keys, vals, chi, n_queries);
                let status = if err < 0.01 {
                    "✓ <1%"
                } else if err < 0.05 {
                    "~ <5%"
                } else {
                    "✗ >5%"
                };
                println!(
                    "  {:>4}  {:>11.4}%  {:>15.2}x  {}",
                    chi,
                    err * 100.0,
                    ratio,
                    status
                );
            }
            println!();
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Interpretation                                                  ║");
    println!("║  χ = bond dimension = number of basis vectors in the MPS cache   ║");
    println!("║  compression = (full cache floats) / (basis + coeff floats)      ║");
    println!("║  Error < 1%  →  lossless for most practical attention use cases  ║");
    println!("║  Error < 5%  →  acceptable quality loss, significant compression ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Next step: run rocmforge with --kv-dump /tmp/kv.bin --gpu \\");
    println!("           then re-run this demo with --kv-dump /tmp/kv.bin");
}
