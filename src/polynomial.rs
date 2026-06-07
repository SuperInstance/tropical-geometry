//! Tropical polynomials and the tropical semiring.

/// A monomial in tropical notation: c ⊗ x₁^a₁ ⊗ x₂^a₂ ⊗ ... = c + a₁x₁ + a₂x₂ + ...
#[derive(Debug, Clone)]
pub struct TropicalMonomial {
    /// Coefficient (the "tropical" constant term, added in normal arithmetic).
    pub coefficient: f64,
    /// Exponents for each variable.
    pub exponents: Vec<usize>,
}

impl TropicalMonomial {
    pub fn new(coefficient: f64, exponents: Vec<usize>) -> Self {
        Self { coefficient, exponents }
    }

    /// Evaluate this monomial at a point (tropical: c + Σ aᵢxᵢ).
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        self.coefficient + self.exponents.iter().zip(x).map(|(&e, &v)| e as f64 * v).sum::<f64>()
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.exponents.len()
    }
}

/// A tropical polynomial: max of affine functions.
/// f(x) = max_i (cᵢ + aᵢ₁x₁ + aᵢ₂x₂ + ...)
///
/// This is a convex piecewise-linear function.
#[derive(Debug, Clone)]
pub struct TropicalPolynomial {
    /// The monomials (affine pieces).
    pub monomials: Vec<TropicalMonomial>,
    /// Number of variables.
    pub num_vars: usize,
}

impl TropicalPolynomial {
    /// Create an empty polynomial in n variables.
    pub fn new(num_vars: usize) -> Self {
        Self { monomials: vec![], num_vars }
    }

    /// Add a monomial (tropical addition = adding another piece to the max).
    pub fn add_monomial(&mut self, coefficient: f64, exponents: Vec<usize>) {
        self.monomials.push(TropicalMonomial::new(coefficient, exponents));
    }

    /// Evaluate the tropical polynomial at a point.
    /// Tropical evaluation: max over all monomials of their evaluations.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        self.monomials.iter().map(|m| m.evaluate(x)).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Find which monomial is active (achieves the max) at point x.
    pub fn active_monomial(&self, x: &[f64]) -> Option<usize> {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0;
        for (i, m) in self.monomials.iter().enumerate() {
            let val = m.evaluate(x);
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }
        if best_val == f64::NEG_INFINITY { None } else { Some(best_idx) }
    }

    /// The Newton polytope vertices (the exponent vectors of the monomials).
    /// In tropical geometry, the tropical hypersurface is the dual of this polytope.
    pub fn newton_polytope_vertices(&self) -> Vec<Vec<usize>> {
        self.monomials.iter().map(|m| m.exponents.clone()).collect()
    }

    /// Tropical addition (max) of two tropical polynomials.
    pub fn tropical_add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.monomials.extend(other.monomials.iter().cloned());
        result
    }

    /// Tropical multiplication (normal addition) of two tropical polynomials.
    /// (f ⊗ g)(x) = f(x) + g(x) — this produces all pairwise combinations.
    pub fn tropical_multiply(&self, other: &Self) -> Self {
        let mut result = Self::new(self.num_vars.max(other.num_vars));
        for m1 in &self.monomials {
            for m2 in &other.monomials {
                let n = result.num_vars;
                let exps: Vec<usize> = (0..n)
                    .map(|i| {
                        let e1 = m1.exponents.get(i).copied().unwrap_or(0);
                        let e2 = m2.exponents.get(i).copied().unwrap_or(0);
                        e1 + e2
                    })
                    .collect();
                result.add_monomial(m1.coefficient + m2.coefficient, exps);
            }
        }
        result
    }

    /// Subtract two tropical polynomials to form a tropical rational function.
    /// f(x) - g(x) = max(f) - max(g). This is a tropical rational function.
    /// A ReLU network computes a tropical rational function.
    /// Evaluate the tropical rational function f - g at a point.
    pub fn tropical_rational_eval(&self, other: &Self, x: &[f64]) -> f64 {
        self.evaluate(x) - other.evaluate(x)
    }

    /// Number of monomials (pieces in the piecewise-linear function).
    pub fn num_pieces(&self) -> usize {
        self.monomials.len()
    }

    /// Find breakpoints in 1D: values of x where the active monomial changes.
    /// Only works for single-variable polynomials.
    pub fn breakpoints_1d(&self) -> Vec<f64> {
        if self.num_vars != 1 { return vec![]; }
        let mut breakpoints = vec![];
        // Two monomials c1+a1*x and c2+a2*x meet when c1+a1*x = c2+a2*x
        for i in 0..self.monomials.len() {
            for j in (i+1)..self.monomials.len() {
                let m1 = &self.monomials[i];
                let m2 = &self.monomials[j];
                let a1 = m1.exponents[0] as f64;
                let a2 = m2.exponents[0] as f64;
                if (a1 - a2).abs() > 1e-10 {
                    let bp = (m2.coefficient - m1.coefficient) / (a1 - a2);
                    breakpoints.push(bp);
                }
            }
        }
        breakpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
        breakpoints
    }
}

/// Tropical arithmetic operations.
pub mod tropical_arith {
    /// Tropical addition: a ⊕ b = max(a, b).
    pub fn tropical_add(a: f64, b: f64) -> f64 {
        a.max(b)
    }

    /// Tropical multiplication: a ⊗ b = a + b.
    pub fn tropical_mul(a: f64, b: f64) -> f64 {
        a + b
    }

    /// Tropical division: a ⊘ b = a - b.
    pub fn tropical_div(a: f64, b: f64) -> f64 {
        a - b
    }

    /// Tropical exponentiation: a^⊗n = n × a.
    pub fn tropical_pow(a: f64, n: usize) -> f64 {
        a * n as f64
    }

    /// Tropical identity for addition: -∞ (zero element).
    pub const TROPICAL_ZERO: f64 = f64::NEG_INFINITY;

    /// Tropical identity for multiplication: 0 (unit element).
    pub const TROPICAL_ONE: f64 = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_addition() {
        assert_eq!(tropical_arith::tropical_add(3.0, 5.0), 5.0);
        assert_eq!(tropical_arith::tropical_add(-1.0, 2.0), 2.0);
    }

    #[test]
    fn test_tropical_multiplication() {
        assert_eq!(tropical_arith::tropical_mul(3.0, 5.0), 8.0);
    }

    #[test]
    fn test_tropical_polynomial_eval() {
        let mut p = TropicalPolynomial::new(1);
        p.add_monomial(1.0, vec![1]); // 1 + x
        p.add_monomial(3.0, vec![0]); // 3
        // f(2) = max(1+2, 3) = max(3, 3) = 3
        assert_eq!(p.evaluate(&[2.0]), 3.0);
        // f(0) = max(1+0, 3) = max(1, 3) = 3
        assert_eq!(p.evaluate(&[0.0]), 3.0);
        // f(5) = max(1+5, 3) = max(6, 3) = 6
        assert_eq!(p.evaluate(&[5.0]), 6.0);
    }

    #[test]
    fn test_breakpoints() {
        let mut p = TropicalPolynomial::new(1);
        p.add_monomial(0.0, vec![2]); // 2x
        p.add_monomial(3.0, vec![0]); // 3
        let bp = p.breakpoints_1d();
        // 2x = 3 → x = 1.5
        assert_eq!(bp.len(), 1);
        assert!((bp[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_tropical_multiply_poly() {
        let mut p1 = TropicalPolynomial::new(1);
        p1.add_monomial(0.0, vec![1]); // x
        let mut p2 = TropicalPolynomial::new(1);
        p2.add_monomial(1.0, vec![1]); // 1 + x
        let prod = p1.tropical_multiply(&p2);
        // x ⊗ (1+x) = x + (2x) = should have monomials [1+x, 0+2x]
        assert_eq!(prod.num_pieces(), 1);
        assert_eq!(prod.monomials[0].coefficient, 1.0);
        assert_eq!(prod.monomials[0].exponents[0], 2);
    }

    #[test]
    fn test_active_monomial() {
        let mut p = TropicalPolynomial::new(1);
        p.add_monomial(0.0, vec![2]); // 2x
        p.add_monomial(10.0, vec![0]); // 10
        // At x=0: max(0, 10) = 10, monomial 1 is active
        assert_eq!(p.active_monomial(&[0.0]), Some(1));
        // At x=100: max(200, 10) = 200, monomial 0 is active
        assert_eq!(p.active_monomial(&[100.0]), Some(0));
    }
}
