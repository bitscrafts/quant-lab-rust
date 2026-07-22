//! # qf-02-loan — Loan Default Prediction
//!
//! Phase 2 of the quant-finance curriculum: feature engineering, scoring models,
//! and ROC-AUC evaluation for credit risk assessment.
//!
//! This crate implements:
//! - Feature engineering (debt-to-income, loan-to-income, grade scoring)
//! - Categorical encoding (one-hot)
//! - Feature normalization (min-max scaling)
//! - Linear scoring model with sigmoid probability
//! - ROC-AUC evaluation metrics

pub mod data;
pub mod eval;
pub mod features;
pub mod scorer;
pub mod traits;

pub use data::{HomeOwnership, LoanApplication, load_loans};
pub use eval::{RocResult, auc, roc_curve};
pub use features::{FeatureExtractor, Normalizer, OneHotEncoder};
pub use scorer::LinearScorer;
pub use traits::{BinaryClassifier, Scorer};
