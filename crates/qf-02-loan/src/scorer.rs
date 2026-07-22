//! Linear scoring model with sigmoid probability.

use crate::traits::{BinaryClassifier, Scorer};

/// Linear scoring model: score = weights · features + bias
pub struct LinearScorer {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearScorer {
    /// Create a new linear scorer with given weights and bias.
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }
    
    /// Predict class using threshold on probability.
    pub fn predict(&self, features: &[f64], threshold: f64) -> bool {
        self.predict_proba(features) > threshold
    }
}

impl Scorer for LinearScorer {
    /// Compute weighted sum: weights · features + bias
    fn score(&self, features: &[f64]) -> f64 {
        let dot_product: f64 = self.weights.iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum();
        dot_product + self.bias
    }
}

impl BinaryClassifier for LinearScorer {
    /// Predict class (default threshold 0.5).
    fn predict(&self, features: &[f64]) -> bool {
        self.predict_proba(features) > 0.5
    }
    
    /// Predict probability using sigmoid function.
    fn predict_proba(&self, features: &[f64]) -> f64 {
        sigmoid(self.score(features))
    }
}

/// Hand-rolled sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sigmoid_properties() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }
}
