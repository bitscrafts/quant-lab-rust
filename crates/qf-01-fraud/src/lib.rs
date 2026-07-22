//! Credit card fraud detection using z-score anomaly detection.
//!
//! This crate implements a simple statistical fraud detector that flags
//! transactions with unusually high feature values (z-score > threshold).

use qf_common::Transaction;

/// Z-score based anomaly detector for fraud detection.
///
/// Fits on training data to learn feature means and standard deviations,
/// then predicts fraud by checking if any feature exceeds the z-score threshold.
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    /// Z-score threshold for anomaly detection (typically 3.0).
    threshold: f64,
    
    /// Mean value for each feature.
    feature_means: Vec<f64>,
    
    /// Standard deviation for each feature.
    feature_stds: Vec<f64>,
}

impl ZScoreDetector {
    /// Create a new detector with the given z-score threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Z-score threshold (e.g., 3.0 means 3 standard deviations)
    ///
    /// # Examples
    ///
    /// ```
    /// use qf_01_fraud::ZScoreDetector;
    ///
    /// let detector = ZScoreDetector::new(3.0);
    /// ```
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }
    
    /// Fit the detector on a set of transactions.
    ///
    /// Computes mean and standard deviation for each feature across all transactions.
    ///
    /// # Arguments
    ///
    /// * `transactions` - Training data
    ///
    /// # Panics
    ///
    /// Panics if transactions have inconsistent feature counts.
    pub fn fit(&mut self, transactions: &[Transaction]) {
        if transactions.is_empty() {
            return;
        }
        
        let num_features = transactions[0].features.len();
        
        // Initialize means and stds
        self.feature_means = vec![0.0; num_features];
        self.feature_stds = vec![0.0; num_features];
        
        // Compute mean for each feature
        for tx in transactions {
            assert_eq!(
                tx.features.len(),
                num_features,
                "All transactions must have the same number of features"
            );
            
            for (i, &value) in tx.features.iter().enumerate() {
                self.feature_means[i] += value;
            }
        }
        
        for mean in &mut self.feature_means {
            *mean /= transactions.len() as f64;
        }
        
        // Compute standard deviation for each feature
        for tx in transactions {
            for (i, &value) in tx.features.iter().enumerate() {
                let diff = value - self.feature_means[i];
                self.feature_stds[i] += diff * diff;
            }
        }
        
        for std in &mut self.feature_stds {
            *std = (*std / transactions.len() as f64).sqrt();
        }
    }
    
    /// Predict whether a transaction is fraudulent.
    ///
    /// Returns true if any feature's z-score exceeds the threshold.
    ///
    /// # Arguments
    ///
    /// * `transaction` - Transaction to classify
    ///
    /// # Returns
    ///
    /// `true` if fraudulent, `false` if normal.
    ///
    /// # Panics
    ///
    /// Panics if detector hasn't been fitted yet or if transaction has wrong feature count.
    pub fn predict(&self, transaction: &Transaction) -> bool {
        assert!(
            !self.feature_means.is_empty(),
            "Detector must be fitted before prediction"
        );
        assert_eq!(
            transaction.features.len(),
            self.feature_means.len(),
            "Transaction feature count must match training data"
        );
        
        for (i, &value) in transaction.features.iter().enumerate() {
            let mean = self.feature_means[i];
            let std = self.feature_stds[i];
            
            // Avoid division by zero
            if std == 0.0 {
                continue;
            }
            
            let z_score = (value - mean).abs() / std;
            
            if z_score > self.threshold {
                return true;
            }
        }
        
        false
    }
}

/// Confusion matrix for binary classification.
///
/// Tracks true positives, false positives, true negatives, and false negatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfusionMatrix {
    /// True positives: correctly predicted fraud.
    pub tp: usize,
    
    /// False positives: normal transactions predicted as fraud.
    pub fp: usize,
    
    /// True negatives: correctly predicted normal.
    pub tn: usize,
    
    /// False negatives: fraud transactions predicted as normal.
    pub fn_: usize,
}

impl ConfusionMatrix {
    /// Compute precision (positive predictive value).
    ///
    /// Precision = TP / (TP + FP)
    ///
    /// Returns 0.0 if TP + FP = 0 (no positive predictions).
    pub fn precision(&self) -> f64 {
        let denominator = self.tp + self.fp;
        if denominator == 0 {
            return 0.0;
        }
        self.tp as f64 / denominator as f64
    }
    
    /// Compute recall (true positive rate, sensitivity).
    ///
    /// Recall = TP / (TP + FN)
    ///
    /// Returns 0.0 if TP + FN = 0 (no actual positives).
    pub fn recall(&self) -> f64 {
        let denominator = self.tp + self.fn_;
        if denominator == 0 {
            return 0.0;
        }
        self.tp as f64 / denominator as f64
    }
    
    /// Compute F1 score (harmonic mean of precision and recall).
    ///
    /// F1 = 2 * (precision * recall) / (precision + recall)
    ///
    /// Returns 0.0 if precision + recall = 0.
    pub fn f1(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();
        let denominator = precision + recall;
        
        if denominator == 0.0 {
            return 0.0;
        }
        
        2.0 * (precision * recall) / denominator
    }
}

/// Evaluate a detector on a set of transactions.
///
/// Runs the detector on each transaction and compares predictions to actual labels.
///
/// # Arguments
///
/// * `detector` - Trained detector
/// * `transactions` - Test data
///
/// # Returns
///
/// Confusion matrix with TP, FP, TN, FN counts.
pub fn evaluate(detector: &ZScoreDetector, transactions: &[Transaction]) -> ConfusionMatrix {
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_ = 0;
    
    for tx in transactions {
        let predicted = detector.predict(tx);
        let actual = tx.class == 1;
        
        match (predicted, actual) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    
    ConfusionMatrix { tp, fp, tn, fn_ }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detector_creation() {
        let detector = ZScoreDetector::new(3.0);
        assert_eq!(detector.threshold, 3.0);
    }
    
    #[test]
    fn test_confusion_matrix_creation() {
        let cm = ConfusionMatrix {
            tp: 10,
            fp: 5,
            tn: 80,
            fn_: 5,
        };
        assert_eq!(cm.tp, 10);
    }
}
