# tropical-geometry

Tropical geometry replaces + with max and × with +. ReLU networks ARE tropical. This crate gives you the tools to see it.

## The Core Idea

Standard arithmetic:
```
3 + 5 = 8
3 × 5 = 15
```

Tropical arithmetic (max-plus semiring):
```
3 ⊕ 5 = max(3, 5) = 5
3 ⊗ 5 = 3 + 5 = 8
```

Why care? Because `max(0, x)` is ReLU. And ReLU is the activation function in every neural network you've ever used. A neural network IS a tropical polynomial.

```rust
use tropical_geometry::*;
```

---

## 1. Tropical Arithmetic

```rust
use tropical_geometry::polynomial::tropical_arith;

let a = 3.0;
let b = 5.0;

// Tropical addition: take the max
let sum = tropical_arith::tropical_add(a, b);
println!("{} ⊕ {} = {}", a, b, sum); // 3 ⊕ 5 = 5

// Tropical multiplication: regular addition
let prod = tropical_arith::tropical_mul(a, b);
println!("{} ⊗ {} = {}", a, b, prod); // 3 ⊗ 5 = 8

// Tropical division: regular subtraction
let quot = tropical_arith::tropical_div(a, b);
println!("{} ⊘ {} = {}", a, b, quot); // 3 ⊘ 5 = -2

// Tropical exponentiation: repeated multiplication = scalar multiplication
let pow = tropical_arith::tropical_pow(a, 3);
println!("{}^⊗3 = {}", a, pow); // 3^⊗3 = 3×3 = 9

// Identity elements
println!("Additive identity (tropical zero): -∞");
println!("Multiplicative identity (tropical one): 0");
// Because max(x, -∞) = x and x + 0 = x
```

---

## 2. A Tropical Polynomial IS a Piecewise-Linear Function

A tropical polynomial `f(x) = max(c₁ + a₁x, c₂ + a₂x, ...)` is a convex, piecewise-linear function. Each monomial is one flat segment.

```rust
use tropical_geometry::TropicalPolynomial;

// f(x) = max(0 + 2x, 3 + 0x, -1 + 1x)
//      = max(2x, 3, x - 1)
let mut f = TropicalPolynomial::new(1);
f.add_monomial(0.0, vec![2]);  // 2x
f.add_monomial(3.0, vec![0]);  // 3 (constant)
f.add_monomial(-1.0, vec![1]); // x - 1

// Evaluate at different points
for x in [-1.0, 0.0, 1.0, 2.0, 5.0] {
    let val = f.evaluate(&[x]);
    let active = f.active_monomial(&[x]).unwrap();
    println!("f({:4.1}) = {:5.2}  (piece {})", x, val, active);
}
// f(-1.0) = -2.00  (piece 0: 2x = -2)
// f( 0.0) =  3.00  (piece 1: 3)
// f( 1.0) =  3.00  (piece 1: 3, or piece 2: 0, they tie)
// f( 2.0) =  4.00  (piece 0: 2x = 4)
// f( 5.0) = 10.00  (piece 0: 2x = 10)

// Find the breakpoints — where the active piece changes
let bps = f.breakpoints_1d();
println!("Breakpoints: {:?}", bps);
// 2x = 3 at x = 1.5, x-1 = 3 at x = 4, 2x = x-1 at x = 1
```

The **Newton polytope** is the convex hull of the exponent vectors. It governs the shape of the tropical hypersurface (where the function bends):

```rust
let vertices = f.newton_polytope_vertices();
println!("Newton polytope vertices: {:?}", vertices);
// [[2], [0], [1]] — the exponents of each monomial
// In 1D this is just points on a line
// In higher dimensions, it's a convex polytope whose faces
// correspond to the pieces of the piecewise-linear function
```

---

## 3. Two-Variable Tropical Polynomial: A Surface

```rust
use tropical_geometry::TropicalPolynomial;

// f(x,y) = max(x + y, 2x, 2y, 0)
let mut f = TropicalPolynomial::new(2);
f.add_monomial(0.0, vec![1, 1]); // x + y
f.add_monomial(0.0, vec![2, 0]); // 2x
f.add_monomial(0.0, vec![0, 2]); // 2y
f.add_monomial(0.0, vec![0, 0]); // 0

// The active monomial partitions the plane into regions
for &(x, y) in &[(1.0, 1.0), (5.0, 0.0), (0.0, 5.0), (-1.0, -1.0)] {
    let val = f.evaluate(&[x, y]);
    let active = f.active_monomial(&[x, y]).unwrap();
    let exponents = &f.monomials[active].exponents;
    println!("f({},{}) = {:.1}  (active: {}x + {}y)",
        x, y, val, exponents[0], exponents[1]);
}
// f(1,1) = 2.0  (active: 1x + 1y)
// f(5,0) = 10.0  (active: 2x + 0y)
// f(0,5) = 10.0  (active: 0x + 2y)
// f(-1,-1) = 0.0  (active: 0x + 0y)

let vertices = f.newton_polytope_vertices();
println!("Newton polytope (4 vertices in 2D): {:?}", vertices);
// The polytope is a tetrahedron in the space of exponents
// Each face = one linear region of the function
```

---

## 4. Tropical Multiplication: Convolution of Pieces

```rust
use tropical_geometry::TropicalPolynomial;

// f(x) = max(x, 0) — this IS ReLU!
let mut f = TropicalPolynomial::new(1);
f.add_monomial(0.0, vec![1]); // x
f.add_monomial(0.0, vec![0]); // 0

// g(x) = max(x - 1, 2) — shifted ReLU with bias
let mut g = TropicalPolynomial::new(1);
g.add_monomial(-1.0, vec![1]); // x - 1
g.add_monomial(2.0, vec![0]);  // 2

// Tropical multiplication: f ⊗ g = f + g (pointwise)
let product = f.tropical_multiply(&g);
println!("f ⊗ g has {} pieces:", product.num_pieces());
for (i, m) in product.monomials.iter().enumerate() {
    println!("  piece {}: {} + {}x", i, m.coefficient, m.exponents[0]);
}

// Tropical addition: f ⊕ g = max(f, g)
let sum = f.tropical_add(&g);
println!("f ⊕ g has {} pieces", sum.num_pieces());
```

---

## 5. ReLU Networks ARE Tropical

```rust
use tropical_geometry::{ReLULayer, TropicalNetwork};

// A single ReLU neuron: y = max(0, 2x - 1)
// This is EXACTLY the tropical polynomial max(2x - 1, 0)
let layer = ReLULayer::new(vec![vec![2.0]], vec![-1.0]);

println!("Single ReLU neuron: max(0, 2x - 1)");
for x in [-1.0, 0.0, 0.5, 1.0, 2.0] {
    let y = layer.forward(&[x]);
    println!("  x={:4.1} → y={:5.2}", x, y[0]);
}
// x=-1.0 → y= 0.00  (below threshold, ReLU kills it)
// x= 0.0 → y= 0.00
// x= 0.5 → y= 0.00  (2×0.5-1 = 0, right at the breakpoint)
// x= 1.0 → y= 1.00
// x= 2.0 → y= 3.00

// The ReLU IS tropical addition with 0.
// max(0, f(x)) = 0 ⊕ f(x) in the tropical semiring.
```

Now a 2-layer network — this is a **tropical rational function** (difference of two tropical polynomials):

```rust
use tropical_geometry::{TropicalNetwork, ReLULayer};

// Build a network:
// Hidden layer: 3 ReLU neurons
//   h₁ = max(0,  x + 0)
//   h₂ = max(0, -x + 0)
//   h₃ = max(0,  2x - 1)
// Output layer: 1 linear neuron (no ReLU)
//   y = 1·h₁ + 1·h₂ + 1·h₃ + 0

let net = TropicalNetwork::simple_1d(
    &[1.0, -1.0, 2.0],  // input weights
    &[0.0,  0.0, -1.0],  // biases
    &[1.0,  1.0,  1.0],  // output weights
    0.0,                  // output bias
);

println!("2-layer ReLU network:");
for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0] {
    let y = net.forward(&[x]);
    println!("  f({:4.1}) = {:5.2}", x, y[0]);
}
// This network is piecewise linear with breakpoints at 0 and 0.5
// f(x) = max(0,x) + max(0,-x) + max(0,2x-1)
//       = |x| + max(0, 2x-1)

// Which neurons are active at each point?
for x in [-1.0, 0.0, 1.0] {
    let path = net.active_path(&[x]);
    let active: Vec<String> = path[0].iter().enumerate()
        .filter(|(_, &a)| a)
        .map(|(i, _)| format!("h{}", i+1))
        .collect();
    println!("At x={}: active neurons = [{}]", x, active.join(", "));
}
// At x=-1: h2 is active (detects negative input)
// At x=0: h1 is active (zero-crossing detector)
// At x=1: h1 and h3 are active (positive input, steep slope)
```

---

## 6. The Correspondence: Every ReLU Network = Tropical Rational Function

```rust
use tropical_geometry::{TropicalNetwork, ReLULayer, TropicalPolynomial};

// The numerator tropical polynomial of a network
let mut numerator = TropicalPolynomial::new(1);
numerator.add_monomial(0.0, vec![1]);  // x
numerator.add_monomial(0.0, vec![0]);  // 0

// The denominator tropical polynomial
let mut denominator = TropicalPolynomial::new(1);
denominator.add_monomial(0.0, vec![0]); // 0

// The tropical rational function f/g evaluated at a point
for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
    let val = numerator.tropical_rational_eval(&denominator, &[x]);
    println!("f/g at x={:.1} = {:.2}", x, val);
}
// A ReLU network with d layers computes a tropical rational function
// where the numerator and denominator are tropical polynomials
// whose monomials correspond to the neurons in the network

// Upper bound on number of linear regions
let net = TropicalNetwork::simple_1d(
    &[1.0, -1.0, 1.0, -1.0, 1.0],
    &[0.0, 0.0, 0.0, 0.0, 0.0],
    &[1.0, 1.0, 1.0, 1.0, 1.0],
    0.0,
);
println!("Max linear regions: {}", net.max_linear_regions());
// 5 neurons = up to 5 pieces in the piecewise-linear function
```

---

## 7. Tropical Attention: Max-Plus for Transformers

```rust
use tropical_geometry::TropicalAttention;

let ta = TropicalAttention::new(1, 1.0);

// 3 tokens with 2D embeddings
let queries = vec![
    vec![1.0, 0.0],  // token 0: x-aligned
    vec![0.0, 1.0],  // token 1: y-aligned
    vec![1.0, 1.0],  // token 2: diagonal
];
let keys = vec![
    vec![1.0, 0.0],  // key 0: matches token 0
    vec![0.0, 1.0],  // key 1: matches token 1
    vec![1.0, 1.0],  // key 2: matches token 2
];

let weights = ta.attention_weights(&queries, &keys);
println!("Tropical attention weights:");
for (i, row) in weights.iter().enumerate() {
    println!("  token {} → [{:.1}, {:.1}, {:.1}]",
        i, row[0], row[1], row[2]);
}
// Standard attention uses softmax: every token gets some weight
// Tropical attention uses hard max: only the best match gets weight
// This is sparser, faster, and more interpretable

// Max-plus matrix multiplication: the tropical analogue of matmul
let a = vec![vec![1.0, 3.0], vec![2.0, 0.0]];
let b = vec![vec![0.0, 1.0], vec![4.0, 2.0]];
let c = TropicalAttention::max_plus_matmul(&a, &b);
println!("Max-plus A × B:");
for row in &c {
    println!("  [{:.1}, {:.1}]", row[0], row[1]);
}
// c[0][0] = max(1+0, 3+4) = max(1, 7) = 7
// c[0][1] = max(1+1, 3+2) = max(2, 5) = 5
// c[1][0] = max(2+0, 0+4) = max(2, 4) = 4
// c[1][1] = max(2+1, 0+2) = max(3, 2) = 3
```

---

## The Map

```
Standard math           Tropical math            Neural networks
─────────────           ─────────────            ───────────────
a + b                   max(a, b)                max(0, x) = ReLU
a × b                   a + b                    w·x + b
x²                      2x                       quadratic ReLU
polynomial              piecewise-linear         1-layer ReLU net
rational function       difference of PWL        multi-layer ReLU net
softmax                 hard max                 tropical attention
```

**ReLU networks ARE tropical geometry.** The activation pattern of each neuron IS a monomial. The network IS a piecewise-linear function. The breakpoints ARE the decision boundaries. The number of linear regions IS the expressiveness. You can analyze all of it with tropical tools.

---

## API Reference

| Type | What it does |
|------|-------------|
| `TropicalPolynomial` | Piecewise-linear convex function via max of affine pieces. Breakpoints, Newton polytope. |
| `TropicalMonomial` | One affine piece: coefficient + weighted sum of inputs. |
| `ReLULayer` | One layer of a ReLU network. Forward pass, active neurons. |
| `TropicalNetwork` | Multi-layer ReLU network. Linear regions, active paths. |
| `TropicalAttention` | Max-plus attention mechanism. Max-plus matrix multiplication. |
| `tropical_arith::*` | The primitives: `tropical_add`, `tropical_mul`, `tropical_div`, `tropical_pow`. |
