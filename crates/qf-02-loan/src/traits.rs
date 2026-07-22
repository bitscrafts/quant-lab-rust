//! Core traits for classification and scoring.
//!
//! These traits enable composable, reusable implementations across
//! the quant-finance library.

/// Binary classifier trait for prediction tasks.
pub trait BinaryClassifier {
    /// Predict class (true/false) for given features.
    fn predict(&self, features: &[f64]) -> bool;
    
    /// Predict probability of positive class (0.0 to 1.0).
    fn predict_proba(&self, features: &[f64]) -> f64;
}

/// Scoring model that produces continuous scores.
pub trait Scorer {
    /// Compute score for given features.
    fn score(&self, features: &[f64]) -> f64;
}
