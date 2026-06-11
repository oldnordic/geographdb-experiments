//! Natural Gradient via Fisher-Rao Information Metric
//! ----------------------------------------------------
//! Demonstrates how preconditioning gradients with the inverse Fisher
//! information matrix (natural gradient) produces more principled parameter
//! updates than vanilla gradient descent.
//!
//! Run with:
//!   cargo run --example demo_natural_grad

use geographdb_core::algorithms::natural_grad::{
    compare_steps, diagonal_fisher, fisher_rao_dist, kl_divergence, natural_gradient, softmax,
};

fn main() {
    println!("=======================================================================");
    println!("    NATURAL GRADIENT VIA FISHER-RAO INFORMATION METRIC               ");
    println!("=======================================================================");
    println!("-> Vanilla GD moves in Euclidean logit space (ignores curvature).");
    println!("-> Natural GD moves in Fisher-Rao manifold (invariant to reparameterisation).");

    // ── Setup ─────────────────────────────────────────────────────────────────
    // 4-class softmax: class 0 is dominant (simulating a peaked distribution,
    // like a confident LLM token prediction).
    let logits_peaked = vec![4.0f32, 0.5, 0.5, 0.5];
    // 4-class softmax: near-uniform (low-confidence prediction).
    let logits_flat = vec![0.1f32, 0.05, 0.12, 0.08];

    // Gradient that pushes class 0 up and class 1 down.
    let grad = vec![1.0f32, -0.5, 0.2, 0.1];

    println!("\n-----------------------------------------------------------------------");
    println!(" EXPERIMENT A: Peaked Distribution (confident prediction)");
    println!("-----------------------------------------------------------------------");
    let p_peaked = softmax(&logits_peaked);
    let f_peaked = diagonal_fisher(&p_peaked);
    println!("  Logits:       {:?}", logits_peaked);
    println!(
        "  Probabilities: [{:.3}, {:.3}, {:.3}, {:.3}]",
        p_peaked[0], p_peaked[1], p_peaked[2], p_peaked[3]
    );
    println!(
        "  Fisher diag:   [{:.4}, {:.4}, {:.4}, {:.4}]",
        f_peaked[0], f_peaked[1], f_peaked[2], f_peaked[3]
    );
    println!(
        "  -> Class 0 dominates (p={:.3}); F_00={:.4} is small (near boundary of simplex).",
        p_peaked[0], f_peaked[0]
    );
    println!("     Natural gradient amplifies the step there — overcoming flat Fisher landscape.");

    let nat_peaked = natural_gradient(&grad, &p_peaked, 1e-4);
    println!("\n  Gradient:          {:?}", grad);
    println!(
        "  Natural gradient:  [{:.2}, {:.2}, {:.2}, {:.2}]",
        nat_peaked[0], nat_peaked[1], nat_peaked[2], nat_peaked[3]
    );
    println!(
        "  Ratio nat/vanilla: [{:.1}x, {:.1}x, {:.1}x, {:.1}x]",
        nat_peaked[0] / grad[0],
        nat_peaked[1] / grad[1],
        nat_peaked[2] / grad[2],
        nat_peaked[3] / grad[3]
    );

    println!("\n-----------------------------------------------------------------------");
    println!(" EXPERIMENT B: Flat Distribution (uncertain prediction)");
    println!("-----------------------------------------------------------------------");
    let p_flat = softmax(&logits_flat);
    let f_flat = diagonal_fisher(&p_flat);
    println!(
        "  Probabilities: [{:.3}, {:.3}, {:.3}, {:.3}]",
        p_flat[0], p_flat[1], p_flat[2], p_flat[3]
    );
    println!(
        "  Fisher diag:   [{:.4}, {:.4}, {:.4}, {:.4}]",
        f_flat[0], f_flat[1], f_flat[2], f_flat[3]
    );
    println!(
        "  -> Near-uniform distribution; F_ii ≈ 0.25 (maximum curvature at center of simplex)."
    );
    let nat_flat = natural_gradient(&grad, &p_flat, 1e-4);
    println!(
        "  Ratio nat/vanilla: [{:.1}x, {:.1}x, {:.1}x, {:.1}x]",
        nat_flat[0] / grad[0],
        nat_flat[1] / grad[1],
        nat_flat[2] / grad[2],
        nat_flat[3] / grad[3]
    );
    println!("  -> Uniform Fisher → ratio ≈ 4x for all components (1/0.25).");

    println!("\n-----------------------------------------------------------------------");
    println!(" EXPERIMENT C: Step Size in Fisher-Rao Space (Peaked Distribution)");
    println!("-----------------------------------------------------------------------");
    println!(
        "  Starting from peaked logits. Each row shows one LR value.\n  \
         Natural gradient should take more consistent Fisher-Rao steps\n  \
         as LR varies (invariance to scale)."
    );
    println!(
        "  {:<6}  {:>12}  {:>12}  {:>14}  {:>14}",
        "lr", "euclid_van", "euclid_nat", "FR_vanilla", "FR_natural"
    );
    for &lr in &[0.01f32, 0.05, 0.1, 0.2, 0.5] {
        let cmp = compare_steps(&logits_peaked, &grad, lr, 1e-4);
        println!(
            "  {:<6.3}  {:>12.5}  {:>12.5}  {:>14.6}  {:>14.6}",
            lr,
            cmp.euclid_vanilla,
            cmp.euclid_natural,
            cmp.fisher_rao_vanilla,
            cmp.fisher_rao_natural,
        );
    }

    println!("\n-----------------------------------------------------------------------");
    println!(" EXPERIMENT D: Multi-Step Optimisation — KL Distance Per Step");
    println!("-----------------------------------------------------------------------");
    println!("  Minimising cross-entropy loss on a 4-class problem.");
    println!("  Target: class 1 is correct (one-hot [0,1,0,0]).");
    let target = vec![0.0f32, 1.0, 0.0, 0.0];

    // Cross-entropy gradient: g_i = p_i - y_i
    let mut logits_v = vec![2.0f32, 0.5, 1.0, 0.8]; // vanilla start
    let mut logits_n = logits_v.clone(); // natural start

    let lr = 0.1;
    println!(
        "\n  {:>5}  {:>10}  {:>10}  {:>12}  {:>12}",
        "step", "CE_vanilla", "CE_natural", "KL_step_van", "KL_step_nat"
    );

    for step in 1..=8 {
        let pv = softmax(&logits_v);
        let pn = softmax(&logits_n);

        // Cross-entropy loss = -log(p_correct)
        let ce_v = -pv[1].max(1e-30).ln();
        let ce_n = -pn[1].max(1e-30).ln();

        // CE gradient: g_i = p_i - y_i
        let grad_v: Vec<f32> = pv.iter().zip(&target).map(|(&p, &y)| p - y).collect();
        let grad_n: Vec<f32> = pn.iter().zip(&target).map(|(&p, &y)| p - y).collect();

        let nat_n = natural_gradient(&grad_n, &pn, 1e-4);

        // Measure KL moved this step
        let next_logits_v: Vec<f32> = logits_v
            .iter()
            .zip(&grad_v)
            .map(|(&t, &g)| t - lr * g)
            .collect();
        let next_logits_n: Vec<f32> = logits_n
            .iter()
            .zip(&nat_n)
            .map(|(&t, &g)| t - lr * g)
            .collect();

        let pv_next = softmax(&next_logits_v);
        let pn_next = softmax(&next_logits_n);
        let kl_v = kl_divergence(&pv, &pv_next);
        let kl_n = kl_divergence(&pn, &pn_next);

        println!(
            "  {:>5}  {:>10.4}  {:>10.4}  {:>12.6}  {:>12.6}",
            step, ce_v, ce_n, kl_v, kl_n
        );

        logits_v = next_logits_v;
        logits_n = next_logits_n;
    }

    // ── Final summary: distances in distribution space ────────────────────────
    println!("\n-----------------------------------------------------------------------");
    println!(" SUMMARY");
    println!("-----------------------------------------------------------------------");
    let pv_final = softmax(&logits_v);
    let pn_final = softmax(&logits_n);
    println!("  After 8 steps, final cross-entropy:");
    println!("    Vanilla GD:  {:.4}", -pv_final[1].max(1e-30).ln());
    println!("    Natural GD:  {:.4}", -pn_final[1].max(1e-30).ln());
    let fr = fisher_rao_dist(&pv_final, &pn_final);
    println!("  Fisher-Rao distance between final distributions: {fr:.4}");
    println!(
        "  -> Natural gradient reaches lower CE via steps proportional to\n     \
         information content rather than Euclidean parameter distance."
    );
    println!("  -> Fisher-Rao metric makes this geometry explicit.");
    println!("=======================================================================");
}
