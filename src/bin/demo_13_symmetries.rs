//! Demo of 13-Fold Symmetries in GeoGraphDB Core
//!
//! Demonstrates geometric, algebraic, and graph-theoretic 13-fold
//! symmetry constructions, with applications to crop circle analysis.

use geographdb_core::algorithms::symmetry_13::*;

fn main() {
    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  GeoGraphDB Core — 13-Fold Symmetries Demo                 ║");
    println!("║  Geometric | Algebraic | Graph-Theoretic                   ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    //  GEOMETRIC 13-FOLD SYMMETRIES
    // ═══════════════════════════════════════════════════════════════════════
    println!("━ GEOMETRIC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Regular 13-gon
    let gon = regular_13_gon();
    println!("Regular 13-gon vertices (first 5):");
    for (i, (x, y)) in gon.iter().take(5).enumerate() {
        println!("  V{}: ({:.6}, {:.6})", i, x, y);
    }
    println!("  ... (13 vertices total, on unit circle)");
    println!();

    // Dihedral group D13
    let perms = dihedral_d13_permutations();
    println!("Dihedral group D13: {} permutations", perms.len());
    println!("  Rotations: 13");
    println!("  Reflections: 13");
    println!("  Order: |D13| = 26 = 2 × 13");
    println!();

    // Verify reflection is involution
    let id: Vec<usize> = (0..13).collect();
    let refl = &perms[13]; // first reflection
    let double: Vec<usize> = refl.iter().map(|&i| refl[i]).collect();
    println!(
        "Reflection involution check: refl² = identity → {}",
        double == id
    );
    println!();

    // Star polygons
    println!("Star polygons {{13/k}}:");
    for k in [1, 2, 3, 4, 5, 6] {
        if let Some(star) = star_polygon_13(k) {
            println!("  {{13/{:2}}}: {:?}", k, &star[..5.min(star.len())]);
        }
    }
    println!();

    // 13-fold symmetry detection on regular 13-gon
    let (is_sym, chi, peak) = detect_13_fold_symmetry(&gon, 26.0);
    println!("13-fold symmetry detection on regular 13-gon:");
    println!("  Is 13-symmetric: {}", is_sym);
    println!("  χ² uniformity:   {:.4}", chi);
    println!("  Radial peak:     {:.2}%", peak * 100.0);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    //  ALGEBRAIC 13-FOLD SYMMETRIES
    // ═══════════════════════════════════════════════════════════════════════
    println!("━ ALGEBRAIC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // 13th roots of unity
    let roots = roots_of_unity_13();
    println!("13th roots of unity ζ₁₃^k = e^(2πik/13):");
    println!("  ζ^0 = 1 + 0i");
    println!("  ζ^1 ≈ {:.6} + {:.6}i", roots[1].0, roots[1].1);
    println!("  ζ^2 ≈ {:.6} + {:.6}i", roots[2].0, roots[2].1);
    println!();

    // Sum of all roots = 0
    let sum_r: f64 = roots.iter().map(|(r, _)| r).sum();
    let sum_i: f64 = roots.iter().map(|(_, i)| i).sum();
    println!(
        "Sum of all 13th roots: {:.2e} + {:.2e}i (should be 0)",
        sum_r, sum_i
    );
    println!();

    // Cyclotomic polynomial
    println!("13th cyclotomic polynomial Φ₁₃(x):");
    println!("  Φ₁₃(0) = {} (expected 1)", cyclotomic_13(0.0));
    println!("  Φ₁₃(1) = {} (expected 13)", cyclotomic_13(1.0));
    let zeta1 = roots[1];
    println!("  Φ₁₃(ζ) ≈ {:.2e} (expected ~0)", cyclotomic_13(zeta1.0));
    println!();

    // Real subfield basis
    let basis = real_subfield_basis_13();
    println!("Real subfield Q(ζ₁₃ + ζ₁₃⁻¹) basis (degree 6):");
    for (k, eta) in basis.iter().enumerate() {
        println!("  η_{} = 2·cos(2π·{}/13) ≈ {:.6}", k + 1, k + 1, eta);
    }
    let basis_sum: f64 = basis.iter().sum();
    println!("  Sum = {:.6} (expected -1)", basis_sum);
    println!();

    // Gaussian periods (cubic subfield)
    let (eta0, eta1, eta2) = gaussian_periods_cubic_13();
    println!("Gaussian periods (cubic subfield of Q(ζ₁₃)):");
    println!("  η₀ ≈ {:.6}", eta0);
    println!("  η₁ ≈ {:.6}", eta1);
    println!("  η₂ ≈ {:.6}", eta2);
    println!("  η₀ + η₁ + η₂ ≈ {:.6} (expected -1)", eta0 + eta1 + eta2);
    println!("  These are roots of x³ + x² − 4x + 1 = 0");
    println!();

    // Modular forms for Γ₀(13)
    let f = cusp_form_gamma0_13(0.5, 30);
    println!("Cusp form for Γ₀(13) at τ = 0.5i:");
    println!("  f(τ) = η(τ)² · η(13τ)² ≈ {:.6e}", f);
    println!();

    let t = hauptmodul_x0_13(0.01, 8);
    println!("Hauptmodul for X₀(13) at q = 0.01:");
    println!("  t₁₃(q) ≈ {:.2}", t);
    println!();

    // Power residue symbol
    println!("13th-power residue symbols:");
    println!(
        "  (2/53)_13 = {} (53 ≡ 1 mod 13)",
        power_residue_symbol_13(2, 53)
    );
    println!(
        "  (2/17)_13 = {} (17 ≢ 1 mod 13)",
        power_residue_symbol_13(2, 17)
    );
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    //  GRAPH-THEORETIC 13-FOLD SYMMETRIES
    // ═══════════════════════════════════════════════════════════════════════
    println!("━ GRAPH-THEORETIC ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Circulant graph C_13(1)
    let cycle = circulant_graph_13(&[1]);
    println!("Circulant graph C_13(1) — the 13-cycle:");
    println!("  Vertices: {}", cycle.len());
    println!("  Degree: {} (each vertex)", cycle[0].len());
    println!();

    // Paley graph of order 13
    let paley = paley_graph_13();
    println!("Paley graph of order 13:");
    println!("  Vertices: {}", paley.len());
    let deg = {
        let mut u = paley[0].clone();
        u.sort_unstable();
        u.dedup();
        u.len()
    };
    println!("  Degree: {} (strongly regular srg(13, 6, 2, 3))", deg);
    println!("  |Aut(Paley(13))| = 156 = 13 × 12 (affine group AGL(1,13))");
    println!();

    // C13 automorphism detection
    let (has_c13, orbits) = detect_c13_automorphism(&cycle);
    println!("C13 automorphism detection on C_13(1):");
    println!("  Has C13 symmetry: {}", has_c13);
    println!("  Number of orbits: {}", orbits.len());
    if !orbits.is_empty() {
        println!(
            "  Orbit sizes: {:?}",
            orbits.iter().map(|o| o.len()).collect::<Vec<_>>()
        );
    }
    println!();

    // 13-regular graph: K_14
    let mut k14: Vec<Vec<usize>> = vec![Vec::new(); 14];
    for i in 0..14 {
        for j in 0..14 {
            if i != j {
                k14[i].push(j);
            }
        }
    }
    println!("Complete graph K_14 (13-regular):");
    println!("  is_13_regular: {}", is_13_regular(&k14));
    if let Some((girth, desc)) = check_13_cage(&k14) {
        println!("  Cage status: ({}, {}) — {}", girth, girth + 1, desc);
    }
    println!();

    // Graph cover
    let base = vec![vec![1], vec![0]];
    let mut voltages = std::collections::HashMap::new();
    voltages.insert((0, 1), 1usize);
    voltages.insert((1, 0), 12usize);
    let cover = graph_cover_13(&base, &voltages);
    println!("13-fold graph cover (voltage construction):");
    println!("  Base vertices: {}", base.len());
    println!("  Cover vertices: {} ( = 2 × 13)", cover.len());
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    //  COMPREHENSIVE ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════
    println!("━ COMPREHENSIVE CROP CIRCLE ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Metatron's Cube: 13 circles
    let metatron: Vec<(f64, f64)> = {
        let mut pts = Vec::new();
        // Center
        pts.push((0.0, 0.0));
        // 12 surrounding in ring
        for k in 0..12 {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) / 12.0;
            pts.push((theta.cos(), theta.sin()));
        }
        pts
    };

    let (geo, alg, graph, overall) = analyze_13_symmetry_comprehensive(&metatron);
    println!("Metatron's Cube pattern (1 center + 12 ring):");
    println!("  Geometric score:  {:.3}", geo);
    println!("  Algebraic score:  {:.3}", alg);
    println!("  Graph score:      {:.3}", graph);
    println!("  ─────────────────────────");
    println!("  Overall score:    {:.3}", overall);
    println!();

    // Regular 13-gon for comparison
    let (geo2, alg2, graph2, overall2) = analyze_13_symmetry_comprehensive(&gon);
    println!("Regular 13-gon (pure 13-fold symmetry):");
    println!("  Geometric score:  {:.3}", geo2);
    println!("  Algebraic score:  {:.3}", alg2);
    println!("  Graph score:      {:.3}", graph2);
    println!("  ─────────────────────────");
    println!("  Overall score:    {:.3}", overall2);
    println!();

    // Random points (control)
    let random: Vec<(f64, f64)> = (0..50)
        .map(|i| ((i * 17) as f64 * 0.1, (i * 31) as f64 * 0.1))
        .collect();
    let (geo3, alg3, graph3, overall3) = analyze_13_symmetry_comprehensive(&random);
    println!("Random points (control, no symmetry):");
    println!("  Geometric score:  {:.3}", geo3);
    println!("  Algebraic score:  {:.3}", alg3);
    println!("  Graph score:      {:.3}", graph3);
    println!("  ─────────────────────────");
    println!("  Overall score:    {:.3}", overall3);
    println!();

    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                    ║");
    println!("╠═════════════════════════════════════════════════════════════╣");
    println!("║  Geometric:                                                 ║");
    println!("║    • Regular 13-gon construction                            ║");
    println!("║    • Dihedral group D13 (26 elements)                       ║");
    println!("║    • Star polygons {{13/k}}                                   ║");
    println!("║    • 13-fold rotational symmetry detection                  ║");
    println!("║                                                             ║");
    println!("║  Algebraic:                                                 ║");
    println!("║    • 13th roots of unity                                    ║");
    println!("║    • Cyclotomic polynomial Φ₁₃(x)                           ║");
    println!("║    • Real subfield Q(ζ₁₃ + ζ₁₃⁻¹), degree 6                 ║");
    println!("║    • Gaussian periods (cubic subfield)                      ║");
    println!("║    • Cusp forms for Γ₀(13)                                  ║");
    println!("║    • Hauptmodul for X₀(13)                                  ║");
    println!("║    • 13th-power residue symbol                              ║");
    println!("║                                                             ║");
    println!("║  Graph-Theoretic:                                           ║");
    println!("║    • Circulant graphs C₁₃(S)                                ║");
    println!("║    • Paley graph of order 13 (srg(13,6,2,3))                ║");
    println!("║    • 13-regular graph detection                             ║");
    println!("║    • 13-fold graph covers (voltage construction)            ║");
    println!("║    • C13 automorphism detection                             ║");
    println!("║    • (13, g)-cage identification                            ║");
    println!("║                                                             ║");
    println!("║  Integration:                                               ║");
    println!("║    • Comprehensive 13-symmetry score for point clouds       ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
}
