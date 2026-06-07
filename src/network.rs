//! ReLU networks as tropical rational functions.

/// A ReLU layer: y = max(0, Wx + b).
/// In tropical terms, max(0, f(x)) = max(f(x), 0) is tropical addition
/// of f(x) with the zero function.
#[derive(Debug, Clone)]
pub struct ReLULayer {
    /// Weight matrix (output_dim × input_dim).
    pub weights: Vec<Vec<f64>>,
    /// Bias vector (output_dim).
    pub bias: Vec<f64>,
}

impl ReLULayer {
    pub fn new(weights: Vec<Vec<f64>>, bias: Vec<f64>) -> Self {
        Self { weights, bias }
    }

    /// Forward pass: y = max(0, Wx + b).
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(row, &b)| {
                let linear: f64 = row.iter().zip(x).map(|(w, xi)| w * xi).sum::<f64>() + b;
                linear.max(0.0) // This IS tropical addition: max(linear, -∞ shifted to 0)
            })
            .collect()
    }

    /// The linear part without ReLU: Wx + b (tropical polynomial evaluation).
    pub fn linear(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(row, &b)| row.iter().zip(x).map(|(w, xi)| w * xi).sum::<f64>() + b)
            .collect()
    }

    /// Number of neurons (output dimension).
    pub fn output_dim(&self) -> usize {
        self.weights.len()
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// A tropical network: a sequence of ReLU layers.
/// Each layer adds tropical max operations, building up a piecewise-linear function.
#[derive(Debug, Clone)]
pub struct TropicalNetwork {
    pub layers: Vec<ReLULayer>,
}

impl TropicalNetwork {
    pub fn new(layers: Vec<ReLULayer>) -> Self {
        Self { layers }
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut activation = x.to_vec();
        for layer in &self.layers {
            activation = layer.forward(&activation);
        }
        activation
    }

    /// Count the total number of linear regions (upper bound).
    /// A ReLU network with layers of width n₁, n₂, ... can have at most
    /// Π nᵢ linear regions.
    pub fn max_linear_regions(&self) -> usize {
        self.layers.iter().map(|l| l.output_dim()).product()
    }

    /// Extract the tropical polynomial at a specific point.
    /// Returns which neurons are active (output > 0) at each layer.
    pub fn active_path(&self, x: &[f64]) -> Vec<Vec<bool>> {
        let mut activation = x.to_vec();
        let mut path = vec![];
        for layer in &self.layers {
            let linear = layer.linear(&activation);
            let active: Vec<bool> = linear.iter().map(|&v| v > 0.0).collect();
            path.push(active);
            activation = layer.forward(&activation);
        }
        path
    }

    /// Create a simple 2-layer ReLU network for 1D input.
    /// h = ReLU(w1*x + b1)  (hidden layer)
    /// y = w2*h + b2        (output layer, no ReLU)
    pub fn simple_1d(
        w1: &[f64], b1: &[f64],
        w2: &[f64], b2: f64,
    ) -> Self {
        let hidden = w1.len();
        let layer1 = ReLULayer::new(
            w1.iter().map(|&w| vec![w]).collect(),
            b1.to_vec(),
        );
        let layer2 = ReLULayer::new(
            vec![w2.to_vec()],
            vec![b2],
        );
        Self { layers: vec![layer1, layer2] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_layer() {
        let layer = ReLULayer::new(
            vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
            vec![0.0, 0.0],
        );
        let y = layer.forward(&[2.0, 1.0]);
        assert_eq!(y[0], 1.0); // max(0, 2-1) = 1
        assert_eq!(y[1], 0.0); // max(0, -2+1) = max(0,-1) = 0 (ReLU kills it)
    }

    #[test]
    fn test_relu_is_tropical() {
        let layer = ReLULayer::new(vec![vec![2.0]], vec![-1.0]);
        // At x=1: max(0, 2*1-1) = max(0, 1) = 1
        assert_eq!(layer.forward(&[1.0])[0], 1.0);
        // At x=0: max(0, 2*0-1) = max(0, -1) = 0
        assert_eq!(layer.forward(&[0.0])[0], 0.0);
        // The ReLU IS the tropical max with 0
    }

    #[test]
    fn test_network_forward() {
        let net = TropicalNetwork::simple_1d(
            &[1.0, -1.0, 2.0], // w1: 3 hidden neurons
            &[0.0, 0.0, -1.0],  // b1
            &[1.0, 1.0, 1.0],   // w2: sum
            0.0,
        );
        let y = net.forward(&[2.0]);
        // h = [ReLU(2), ReLU(-2), ReLU(3)] = [2, 0, 3]
        // y = 2 + 0 + 3 = 5
        assert_eq!(y[0], 5.0);
    }

    #[test]
    fn test_active_path() {
        let net = TropicalNetwork::simple_1d(
            &[1.0, -1.0], &[0.0, 0.0],
            &[1.0, 1.0], 0.0,
        );
        let path = net.active_path(&[2.0]);
        // Layer 0: [ReLU(2)=active, ReLU(-2)=inactive]
        assert!(path[0][0]);
        assert!(!path[0][1]);
    }

    #[test]
    fn test_max_linear_regions() {
        let net = TropicalNetwork::simple_1d(
            &[1.0, -1.0, 1.0], &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0], 0.0,
        );
        assert_eq!(net.max_linear_regions(), 3);
    }
}
