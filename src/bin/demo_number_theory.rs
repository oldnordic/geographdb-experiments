//! Demo of number-theoretic algorithms in GeoGraphDB Core
//!
//! Tests the crop circle completion hints using the new number_theory module.

use geographdb_core::algorithms::number_theory::*;

fn main() {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  GEoGraphDB Core Number Theory Demo                        ║");
    println!("║  Testing Crop Circle Completion Hints                      ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // TEST 1: Arecibo Reply Semiprime Factoring
    // ========================================================================
    println!("TEST 1: Arecibo Reply Semiprime Factoring");
    println!("───────────────────────────────────────────────────────────────");

    let test_cases = vec![
        (1679, "Arecibo Reply: 23×73"),
        (154, "Test: 7×22"),
        (385, "Test: 11×35"),
        (533, "Test: 13×41"),
    ];

    for (n, desc) in test_cases {
        println!("  {} (n={}):", desc, n);

        // Standard trial division
        let factors = factor_trial(n);
        print!("    Trial division: {:?}", factors);

        // Semiprime check
        if let Some((p, q)) = semiprime_factors(n) {
            let ratio = q as f64 / p as f64;
            println!(", factors = {}×{}", p, q);
            println!(
                "    q/p = {:.6}, π = {:.6}, |diff| = {:.6}",
                ratio,
                std::f64::consts::PI,
                (ratio - std::f64::consts::PI).abs()
            );

            // Fast factoring using π-family algorithm
            if let Some((p_fast, q_fast)) = factor_semiprime_pi_family(n, 10) {
                println!("    π-family factoring: {}×{} ✓", p_fast, q_fast);
            }
        } else {
            println!(" (not semiprime)");
        }
        println!();
    }

    // ========================================================================
    // TEST 2: Milk Hill Prime Analysis
    // ========================================================================
    println!("TEST 2: Milk Hill - 409 Circles and Primes");
    println!("───────────────────────────────────────────────────────────────");

    println!("  409 is prime: {}", is_prime(409));

    let primes = sieve_primes(500);
    let primes_up_to_409: Vec<&u64> = primes.iter().filter(|&&p| p <= 409).collect();
    println!("  Primes ≤ 409: {}", primes_up_to_409.len());
    println!("  π(409) ≈ {}", logarithmic_integral(409.0));
    println!();

    // The 409th prime
    let p_409 = nth_prime(409);
    println!("  409th prime: {}", p_409);
    println!();

    // ========================================================================
    // TEST 3: Riemann Zeta Function
    // ========================================================================
    println!("TEST 3: Riemann Zeta Function");
    println!("───────────────────────────────────────────────────────────────");

    println!(
        "  ζ(2) = {:.6} (expected π²/6 = {:.6})",
        zeta_real(2.0, 100000),
        std::f64::consts::PI * std::f64::consts::PI / 6.0
    );

    println!(
        "  ζ(3) = {:.6} (Apéry's constant, irrational)",
        zeta_real(3.0, 100000)
    );

    println!(
        "  ζ(4) = {:.6} (expected π⁴/90 = {:.6})",
        zeta_real(4.0, 100000),
        std::f64::consts::PI.powi(4) / 90.0
    );
    println!();

    // Zero count
    let n_zeros = 409;
    let t_approx = 700.0; // Approximate height
    println!(
        "  Approximate zeros up to T={}: {:.0}",
        t_approx,
        zeta_zero_count_approx(t_approx)
    );
    println!(
        "  Approximate {}th zero: {:.2}",
        n_zeros,
        zeta_zero_approx(n_zeros as u64)
    );
    println!();

    // ========================================================================
    // TEST 4: Continued Fractions
    // ========================================================================
    println!("TEST 4: Continued Fractions");
    println!("───────────────────────────────────────────────────────────────");

    let pi = std::f64::consts::PI;
    let conv = continued_fraction(pi, 10);
    println!("  π continued fraction convergents:");
    for (i, (p, q)) in conv.iter().take(6).enumerate() {
        let approx = *p as f64 / *q as f64;
        let error = (approx - pi).abs();
        println!(
            "    {}: {}/{} = {:.8} (error: {:.2e})",
            i, p, q, approx, error
        );
    }
    println!();

    // Best rational approximation to π with small denominator
    let (p_best, q_best) = best_rational_approx(pi, 1000);
    println!(
        "  Best rational approx to π with den ≤ 1000: {}/{} = {:.10}",
        p_best,
        q_best,
        p_best as f64 / q_best as f64
    );
    println!();

    // ========================================================================
    // TEST 5: Modular Arithmetic
    // ========================================================================
    println!("TEST 5: Modular Arithmetic");
    println!("───────────────────────────────────────────────────────────────");

    println!("  2^10 mod 1000 = {}", mod_pow(2, 10, 1000));
    println!("  3^5 mod 7 = {}", mod_pow(3, 5, 7));
    println!("  3⁻¹ mod 11 = {:?}", mod_inverse(3, 11));
    println!(
        "  Legendre(2, 7) = {} (expected -1, non-residue)",
        legendre_symbol(2, 7)
    );
    println!(
        "  Legendre(2, 17) = {} (expected 1, residue)",
        legendre_symbol(2, 17)
    );
    println!();

    // ========================================================================
    // SUMMARY
    // ========================================================================
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                    ║");
    println!("╠═════════════════════════════════════════════════════════════╣");
    println!("║  Number theory module integrated into GeoGraphDB Core      ║");
    println!("║                                                             ║");
    println!("║  Capabilities:                                              ║");
    println!("║    • Prime testing (trial division)                        ║");
    println!("║    • Sieve of Eratosthenes                                 ║");
    println!("║    • Integer factorization                                 ║");
    println!("║    • Semiprime detection                                   ║");
    println!("║    • π-family fast factoring (crop circle validated)       ║");
    println!("║    • Riemann zeta approximation                            ║");
    println!("║    • Continued fractions                                   ║");
    println!("║    • Modular arithmetic                                    ║");
    println!("║                                                             ║");
    println!("║  Next steps for PARI/GP-level functionality:               ║");
    println!("║    • Miller-Rabin primality test                           ║");
    println!("║    • Quadratic sieve factorization                         ║");
    println!("║    • Elliptic curve factorization (ECM)                    ║");
    println!("║    • Exact zeta zero computation                           ║");
    println!("║    • Algebraic number theory                               ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
}
