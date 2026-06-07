#![allow(clippy::needless_range_loop, clippy::new_without_default, dead_code)]
//! # Tropical Geometry
//!
//! Tropical geometry replaces addition with max and multiplication with addition.
//! This makes neural networks with ReLU activations into piecewise-linear functions
//! that can be analyzed exactly using tropical tools.
//!
//! # Key Concepts
//!
//! - **Tropical semiring**: (ℝ ∪ {-∞}, ⊕, ⊗) where a ⊕ b = max(a,b), a ⊗ b = a + b
//! - **Tropical polynomials**: Piecewise-linear convex functions
//! - **Tropical rational functions**: Difference of two tropical polynomials = ReLU network
//! - **Newton polytope**: The combinatorial object governing tropical hypersurface structure

mod polynomial;
mod attention;
mod network;

pub use polynomial::{TropicalPolynomial, TropicalMonomial};
pub use attention::TropicalAttention;
pub use network::{ReLULayer, TropicalNetwork};
