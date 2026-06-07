//! Tropical attention: max-plus algebra for attention mechanisms.

/// Tropical attention replaces the softmax with tropical operations.
/// Instead of exp(score_i) / Σ exp(score_j), we use max(score_i - score_j).
#[derive(Debug, Clone)]
pub struct TropicalAttention {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Temperature parameter (scales the tropical max).
    pub temperature: f64,
}

impl TropicalAttention {
    pub fn new(num_heads: usize, temperature: f64) -> Self {
        Self { num_heads, temperature }
    }

    /// Compute tropical attention weights.
    /// Returns a sparse attention pattern: for each query, which key gets attention.
    ///
    /// Standard attention: α_ij = softmax(q_i · k_j)
    /// Tropical attention: α_ij = max_j(q_i · k_j / τ) — hard attention
    pub fn attention_weights(&self, queries: &[Vec<f64>], keys: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = queries.len();
        let m = keys.len();
        let mut scores = vec![vec![f64::NEG_INFINITY; m]; n];

        // Compute scores
        for i in 0..n {
            for j in 0..m {
                let dot: f64 = queries[i].iter().zip(&keys[j]).map(|(a, b)| a * b).sum();
                scores[i][j] = dot / self.temperature;
            }
        }

        // Tropical softmax: exp(score - max_score) normalized
        // But in pure tropical: just the max gets weight 1.0
        // We use a "soft tropical" blend for practical use
        let mut weights = vec![vec![0.0; m]; n];
        for i in 0..n {
            let max_score = scores[i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for j in 0..m {
                // Tropical-sigmoid blend: give weight proportional to max-gap
                let gap = scores[i][j] - max_score;
                weights[i][j] = if gap.abs() < 1e-10 { 1.0 } else { 0.0 };
                total += weights[i][j];
            }
            if total > 0.0 {
                for j in 0..m {
                    weights[i][j] /= total;
                }
            }
        }
        weights
    }

    /// Tropical self-attention: queries = keys = values.
    pub fn self_attention(&self, vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let weights = self.attention_weights(vectors, vectors);
        let n = vectors.len();
        let d = vectors.first().map(|v| v.len()).unwrap_or(0);

        (0..n)
            .map(|i| {
                (0..d)
                    .map(|k| {
                        (0..n).map(|j| weights[i][j] * vectors[j][k]).sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Max-plus matrix multiplication: (A ⊕⊗ B)[i][j] = max_k(A[i][k] + B[k][j])
    /// This is the tropical analogue of matrix multiplication.
    pub fn max_plus_matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b.first().map(|r| r.len()).unwrap_or(0);
        let p = b.len();
        let mut result = vec![vec![f64::NEG_INFINITY; m]; n];

        for i in 0..n {
            for j in 0..m {
                for k in 0..p {
                    result[i][j] = result[i][j].max(a[i][k] + b[k][j]);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_attention() {
        let ta = TropicalAttention::new(1, 1.0);
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let weights = ta.attention_weights(&queries, &keys);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].len(), 3);
        // Row sums should be 1.0
        for row in &weights {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_max_plus_matmul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        let c = TropicalAttention::max_plus_matmul(&a, &b);
        // c[0][0] = max(1+0, 2+2) = max(1, 4) = 4
        assert_eq!(c[0][0], 4.0);
        // c[0][1] = max(1+1, 2+3) = max(2, 5) = 5
        assert_eq!(c[0][1], 5.0);
        // c[1][0] = max(3+0, 4+2) = max(3, 6) = 6
        assert_eq!(c[1][0], 6.0);
    }
}
