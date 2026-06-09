//! # Tropical Geometry Tutorial
//!
//! Progressive lessons covering the tropical semiring, tropical polynomials,
//! ReLU networks as tropical rational functions, and tropical attention.
//!
//! Run with: `cargo run --example tutorial`

use tropical_geometry::{
    TropicalAttention, TropicalMonomial, TropicalNetwork, TropicalPolynomial,
};

// ── Lesson 1: The Tropical Semiring ──────────────────────────────────────────

fn lesson_1_tropical_semiring() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 1: The Tropical Semiring (ℝ ∪ {{-∞}}, ⊕, ⊗)");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("In the tropical semiring, addition becomes MAX and multiplication becomes +:");
    println!("  a ⊕ b = max(a, b)    (tropical addition)");
    println!("  a ⊗ b = a + b        (tropical multiplication)");
    println!();

    let a = 3.0_f64;
    let b = 7.0_f64;

    // Tropical arithmetic is simple: ⊕ = max, ⊗ = +
    let trop_add = a.max(b);       // tropical addition
    let trop_mul = a + b;           // tropical multiplication
    let trop_div = a - b;           // tropical division
    let trop_pow = a * 4.0_f64;     // tropical exponentiation: a^⊗n = n×a

    println!("  {} ⊕ {} = max({}, {}) = {}", a, b, a, b, trop_add);
    println!("  {} ⊗ {} = {} + {} = {}", a, b, a, b, trop_mul);
    println!();

    // Identities
    println!("  Tropical zero (additive identity):  -∞  (max(x, -∞) = x)");
    println!("  Tropical one  (multiplicative identity): 0  (x + 0 = x)");
    println!();

    // Exponentiation
    println!("  3^⊗4 = 4 × 3 = {}", trop_pow);
    println!("  5^⊗0 = 0 × 5 = {} (tropical: any element to the 0th power = 0)", 5.0_f64 * 0.0);
    println!();

    // Division
    println!("  10 ⊘ 3 = 10 − 3 = {}", trop_div);
    println!();
}

// ── Lesson 2: Tropical Monomials ─────────────────────────────────────────────

fn lesson_2_monomials() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 2: Tropical Monomials");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A tropical monomial c ⊗ x₁^a₁ ⊗ x₂^a₂ ⊗ ... evaluates to c + a₁x₁ + a₂x₂ + ...");
    println!("This is an affine function in normal arithmetic.");
    println!();

    // 3 ⊗ x₁² ⊗ x₂ = 3 + 2x₁ + x₂
    let m = TropicalMonomial::new(3.0, vec![2, 1]);
    println!("  Monomial: 3 + 2x₁ + x₂ (coefficient=3, exponents=[2,1])");
    println!("  Number of variables: {}", m.num_vars());
    println!();

    let x = vec![1.0, 4.0];
    println!("  At x = ({}, {}):  3 + 2·{} + {} = {}", x[0], x[1], x[0], x[1], m.evaluate(&x));
    let x2 = vec![0.0, 0.0];
    println!("  At x = ({}, {}):  3 + 2·{} + {} = {}", x2[0], x2[1], x2[0], x2[1], m.evaluate(&x2));
    println!();
}

// ── Lesson 3: Tropical Polynomials ───────────────────────────────────────────

fn lesson_3_polynomials() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 3: Tropical Polynomials");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A tropical polynomial f(x) = max_i(cᵢ + aᵢ·x) is a convex piecewise-linear function.");
    println!();

    // f(x) = max(2x, 3, x + 1) in 1D
    let mut p = TropicalPolynomial::new(1);
    p.add_monomial(0.0, vec![2]); // 2x
    p.add_monomial(3.0, vec![0]); // 3
    p.add_monomial(1.0, vec![1]); // 1 + x
    println!("  f(x) = max(2x, 3, 1+x)  [{} pieces]", p.num_pieces());
    println!();

    println!("  Evaluation:");
    for x in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let val = p.evaluate(&[x]);
        let active = p.active_monomial(&[x]).unwrap();
        let labels = ["2x", "3", "1+x"];
        println!("    f({:.1}) = {:5.1}   (active piece: {})", x, val, labels[active]);
    }
    println!();

    // Breakpoints
    let bp = p.breakpoints_1d();
    println!("  Breakpoints (where the active piece changes): {:?}", bp);
    println!();
}

// ── Lesson 4: Newton Polytope & Tropical Hypersurface ────────────────────────

fn lesson_4_newton_polytope() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 4: Newton Polytope & Tropical Hypersurface");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("The Newton polytope is the convex hull of the exponent vectors.");
    println!("The tropical hypersurface is the dual subdivision of this polytope.");
    println!();

    // 2-variable polynomial: max(0, x₁, x₂, x₁+x₂)
    let mut p = TropicalPolynomial::new(2);
    p.add_monomial(0.0, vec![0, 0]); // constant 0
    p.add_monomial(0.0, vec![1, 0]); // x₁
    p.add_monomial(0.0, vec![0, 1]); // x₂
    p.add_monomial(0.0, vec![1, 1]); // x₁ + x₂
    println!("  f(x₁, x₂) = max(0, x₁, x₂, x₁+x₂)");
    println!();

    let vertices = p.newton_polytope_vertices();
    println!("  Newton polytope vertices (exponent vectors):");
    for (i, v) in vertices.iter().enumerate() {
        println!("    monomial {}: {:?}", i, v);
    }
    println!();

    // Evaluate at several points
    println!("  Sample evaluations:");
    for (x1, x2) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0), (-1.0, 3.0)] {
        let val = p.evaluate(&[x1, x2]);
        let active = p.active_monomial(&[x1, x2]).unwrap();
        let labels = ["0", "x₁", "x₂", "x₁+x₂"];
        println!("    f({:.0}, {:.0}) = {:.1}  (active: {})", x1, x2, val, labels[active]);
    }
    println!();
}

// ── Lesson 5: Tropical Arithmetic on Polynomials ─────────────────────────────

fn lesson_5_poly_arithmetic() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 5: Tropical Polynomial Arithmetic");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("Tropical addition (⊕) of polynomials = max of two piecewise-linear functions.");
    println!("Tropical multiplication (⊗) of polynomials = pointwise sum.");
    println!();

    // f = max(x, 1)
    let mut f = TropicalPolynomial::new(1);
    f.add_monomial(0.0, vec![1]); // x
    f.add_monomial(1.0, vec![0]); // 1

    // g = max(2x, 3)
    let mut g = TropicalPolynomial::new(1);
    g.add_monomial(0.0, vec![2]); // 0 + 2*x = 2x
    g.add_monomial(3.0, vec![0]); // 3

    println!("  f(x) = max(x, 1)");
    println!("  g(x) = max(2x, 3)");
    println!();

    // Tropical addition
    let sum = f.tropical_add(&g);
    println!("  f ⊕ g = max(x, 1, 2x, 3)  [{} pieces]", sum.num_pieces());
    for x in [0.0, 1.0, 2.0, 3.0] {
        println!("    (f ⊕ g)({:.0}) = {:.1}", x, sum.evaluate(&[x]));
    }
    println!();

    // Tropical multiplication
    let prod = f.tropical_multiply(&g);
    println!("  f ⊗ g = f(x) + g(x)  [{} monomials from pairwise combinations]", prod.num_pieces());
    for x in [0.0, 1.0, 2.0] {
        let fv = f.evaluate(&[x]);
        let gv = g.evaluate(&[x]);
        let pv = prod.evaluate(&[x]);
        println!("    (f⊗g)({:.0}) = f({:.0}) + g({:.0}) = {:.1} + {:.1} = {:.1}", x, x, x, fv, gv, pv);
    }
    println!();
}

// ── Lesson 6: ReLU Networks as Tropical Rational Functions ───────────────────

fn lesson_6_relu_networks() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 6: ReLU Networks as Tropical Rational Functions");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A ReLU network computes a tropical rational function: f(x) − g(x)");
    println!("where f and g are tropical polynomials (convex piecewise-linear).");
    println!();

    // A single ReLU neuron: max(0, 2x − 1)
    let layer = tropical_geometry::ReLULayer::new(
        vec![vec![2.0]],  // weight
        vec![-1.0],        // bias
    );
    println!("  Single ReLU neuron: y = max(0, 2x − 1)");
    println!("  (Input dim: {}, Output dim: {})", layer.input_dim(), layer.output_dim());
    println!();
    for x in [-1.0, 0.0, 0.5, 1.0, 2.0] {
        let y = layer.forward(&[x]);
        let lin = layer.linear(&[x]);
        println!("    x={:5.1}  →  linear={:6.1}  →  ReLU(linear)={:5.1}", x, lin[0], y[0]);
    }
    println!();

    // Multi-layer network using simple_1d
    let net = TropicalNetwork::simple_1d(
        &[1.0, -1.0, 2.0],  // w1: 3 hidden neurons
        &[0.0, 0.0, -1.0],   // b1
        &[1.0, 1.0, 1.0],    // w2: sum
        0.0,
    );
    println!("  2-layer ReLU network (3 hidden neurons):");
    println!("    h = ReLU([1, -1, 2]·x + [0, 0, -1])");
    println!("    y = [1, 1, 1]·h + 0");
    println!("    Max linear regions (upper bound): {}", net.max_linear_regions());
    println!();

    // Active path analysis
    println!("  Active neuron analysis at x = 2.0:");
    let path = net.active_path(&[2.0]);
    for (layer_idx, active) in path.iter().enumerate() {
        let active_str: Vec<&str> = active.iter().map(|&a| if a { "ON" } else { "OFF" }).collect();
        println!("    Layer {}: neurons = [{}]", layer_idx, active_str.join(", "));
    }
    let y = net.forward(&[2.0]);
    println!("    Output: {:.1}", y[0]);
    println!();
}

// ── Lesson 7: Tropical Attention ─────────────────────────────────────────────

fn lesson_7_tropical_attention() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 7: Tropical Attention");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("Tropical attention replaces softmax with max-plus algebra.");
    println!("Instead of soft attention weights, it produces hard (sparse) attention.");
    println!();

    let ta = TropicalAttention::new(1, 1.0);
    println!("  Attention heads: {}, Temperature: {}", ta.num_heads, ta.temperature);
    println!();

    let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    println!("  Queries: {:?}", queries);
    println!("  Keys:    {:?}", keys);
    println!();

    let weights = ta.attention_weights(&queries, &keys);
    println!("  Attention weights (each row sums to 1.0):");
    for (i, row) in weights.iter().enumerate() {
        let s: String = row.iter().map(|w| format!("{:.2}", w)).collect::<Vec<_>>().join(", ");
        println!("    query {} → [{}]", i, s);
    }
    println!();

    // Self-attention
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let sa = ta.self_attention(&vectors);
    println!("  Tropical self-attention output:");
    for (i, v) in sa.iter().enumerate() {
        let s: String = v.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>().join(", ");
        println!("    output {} → [{}]", i, s);
    }
    println!();

    // Max-plus matrix multiplication
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
    let c = TropicalAttention::max_plus_matmul(&a, &b);
    println!("  Max-plus matrix multiplication (A ⊕⊗ B):");
    println!("    A = {:?}, B = {:?}", a, b);
    println!("    (A ⊕⊗ B)[i][j] = max_k(A[i][k] + B[k][j])");
    for row in &c {
        println!("    → {:?}", row);
    }
    println!();
}

// ── Lesson 8: Tropical Rational Functions ────────────────────────────────────

fn lesson_8_tropical_rational() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 8: Tropical Rational Functions & Full Pipeline");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A tropical rational function f(x) − g(x) can represent any ReLU network.");
    println!("Let's build one and analyze it.");
    println!();

    // Numerator: max(3x, 5)
    let mut num = TropicalPolynomial::new(1);
    num.add_monomial(0.0, vec![3]); // 3x
    num.add_monomial(5.0, vec![0]); // 5

    // Denominator: max(x, 2)
    let mut den = TropicalPolynomial::new(1);
    den.add_monomial(0.0, vec![1]); // x
    den.add_monomial(2.0, vec![0]); // 2

    println!("  Tropical rational function: r(x) = max(3x, 5) − max(x, 2)");
    println!();

    println!("  Newton polytope vertices (numerator): {:?}", num.newton_polytope_vertices());
    println!("  Newton polytope vertices (denominator): {:?}", den.newton_polytope_vertices());
    println!();

    println!("  Evaluation:");
    println!("    {:>5}  {:>8}  {:>8}  {:>8}  {}", "x", "num(x)", "den(x)", "r(x)", "Active (num)");
    for x in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let n = num.evaluate(&[x]);
        let d = den.evaluate(&[x]);
        let r = num.tropical_rational_eval(&den, &[x]);
        let active = num.active_monomial(&[x]).unwrap();
        let labels = ["3x", "5"];
        println!("    {:5.1}  {:8.2}  {:8.2}  {:8.2}  {}", x, n, d, r, labels[active]);
    }
    println!();

    // Compose with a ReLU network
    let net = TropicalNetwork::simple_1d(
        &[1.0, 2.0, -1.0],
        &[0.0, -1.0, 0.5],
        &[1.0, -1.0, 0.5],
        0.0,
    );
    println!("  ReLU network output at selected points:");
    for x in [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let y = net.forward(&[x]);
        println!("    N({:.1}) = {:.3}", x, y[0]);
    }
    println!();
    println!("  This network is a tropical rational function with up to {} linear regions.",
             net.max_linear_regions());
    println!();
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Tropical Geometry Tutorial                     ║");
    println!("║   Max-Plus Algebra for Neural Network Analysis   ║");
    println!("╚═══════════════════════════════════════════════════╝");
    println!();

    lesson_1_tropical_semiring();
    lesson_2_monomials();
    lesson_3_polynomials();
    lesson_4_newton_polytope();
    lesson_5_poly_arithmetic();
    lesson_6_relu_networks();
    lesson_7_tropical_attention();
    lesson_8_tropical_rational();

    println!("═══════════════════════════════════════════════════");
    println!("  Tutorial complete! Key takeaways:");
    println!("    • Tropical algebra: ⊕ = max, ⊗ = +");
    println!("    • Tropical polynomials are convex piecewise-linear functions");
    println!("    • ReLU networks compute tropical rational functions");
    println!("    • Tropical attention = hard (sparse) attention via max-plus");
    println!("═══════════════════════════════════════════════════");
}
