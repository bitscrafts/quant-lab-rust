//! Feature engineering for loan default prediction.

use crate::data::HomeOwnership;

/// Utility functions for deriving features from loan data.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Calculate debt-to-income ratio.
    pub fn debt_to_income(income: f64, debt: f64) -> f64 {
        if income == 0.0 {
            0.0
        } else {
            debt / income
        }
    }
    
    /// Calculate loan-to-income ratio.
    pub fn loan_to_income(loan_amount: f64, income: f64) -> f64 {
        if income == 0.0 {
            0.0
        } else {
            loan_amount / income
        }
    }
    
    /// Convert credit grade to risk score.
    ///
    /// Maps A (1) → 0.0 (lowest risk) to G (7) → 1.0 (highest risk).
    pub fn grade_to_score(grade: u8) -> f64 {
        if grade < 1 {
            0.0
        } else if grade > 7 {
            1.0
        } else {
            (grade - 1) as f64 / 6.0
        }
    }
}

/// One-hot encoder for categorical features.
pub struct OneHotEncoder;

impl OneHotEncoder {
    /// Encode home ownership status as one-hot vector.
    ///
    /// Returns [RENT, OWN, MORTGAGE] as [f64; 3].
    pub fn encode(home_ownership: &HomeOwnership) -> Vec<f64> {
        match home_ownership {
            HomeOwnership::Rent => vec![1.0, 0.0, 0.0],
            HomeOwnership::Own => vec![0.0, 1.0, 0.0],
            HomeOwnership::Mortgage => vec![0.0, 0.0, 1.0],
        }
    }
}

/// Min-max normalizer for feature scaling to [0, 1].
pub struct Normalizer {
    min: f64,
    max: f64,
}

impl Normalizer {
    /// Fit normalizer to data by computing min and max.
    pub fn fit(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self { min: 0.0, max: 0.0 };
        }
        
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        
        Self { min, max }
    }
    
    /// Transform value to [0, 1] range using fitted min-max.
    ///
    /// Returns 0.0 if min == max (constant data).
    pub fn transform(&self, value: f64) -> f64 {
        let range = self.max - self.min;
        if range == 0.0 {
            0.0
        } else {
            (value - self.min) / range
        }
    }
}
