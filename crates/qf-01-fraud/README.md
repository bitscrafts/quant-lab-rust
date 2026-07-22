# qf-01-fraud — Credit Card Fraud Detection

Statistical fraud detection using z-score anomaly detection on the Kaggle Credit Card Fraud dataset.

## Overview

This crate implements a simple but effective fraud detector:
- Fits on transaction data to learn feature statistics
- Flags transactions with unusually high feature values (z-score > threshold)
- Evaluates performance with confusion matrix and metrics (precision, recall, F1)

## Algorithm: Z-Score Anomaly Detection

For each feature:
1. Compute mean μ and standard deviation σ across all training transactions
2. For a new transaction, compute z-score: z = |x - μ| / σ
3. Flag as fraud if ANY feature's z-score exceeds threshold (default: 3.0)

This is a simple baseline that works well for datasets where fraud has unusual feature values.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)

- 284,807 transactions
- 492 frauds (0.172% of total)
- Features: Time, V1-V28 (PCA-transformed), Amount, Class
- Class: 0 = normal, 1 = fraud

Download and place in `crates/quant-lab/data/creditcard.csv`.

## Usage

### Library

```rust
use qf_01_fraud::{ZScoreDetector, evaluate};
use qf_common::load_transactions;

// Load data
let transactions = load_transactions("data/creditcard.csv")?;

// Train detector
let mut detector = ZScoreDetector::new(3.0);
detector.fit(&transactions);

// Evaluate
let cm = evaluate(&detector, &transactions);
println!("Precision: {:.4}", cm.precision());
println!("Recall: {:.4}", cm.recall());
println!("F1: {:.4}", cm.f1());
```

### Example Binary

```bash
# Download creditcard.csv first (see Dataset section)
cargo run -p qf-01-fraud --example fraud_analysis
```

Output includes:
- Transaction counts (normal vs fraud)
- Confusion matrix (TP, FP, TN, FN)
- Metrics (precision, recall, F1)

## API

### ZScoreDetector

```rust
pub struct ZScoreDetector {
    threshold: f64,
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
}

impl ZScoreDetector {
    pub fn new(threshold: f64) -> Self
    pub fn fit(&mut self, transactions: &[Transaction])
    pub fn predict(&self, transaction: &Transaction) -> bool
}
```

### ConfusionMatrix

```rust
pub struct ConfusionMatrix {
    pub tp: usize,  // true positives
    pub fp: usize,  // false positives
    pub tn: usize,  // true negatives
    pub fn_: usize, // false negatives
}

impl ConfusionMatrix {
    pub fn precision(&self) -> f64
    pub fn recall(&self) -> f64
    pub fn f1(&self) -> f64
}
```

### evaluate

```rust
pub fn evaluate(
    detector: &ZScoreDetector,
    transactions: &[Transaction]
) -> ConfusionMatrix
```

## Performance Notes

This is a baseline approach for learning purposes. Limitations:
- Uses all features equally (no feature weighting)
- Assumes Gaussian feature distributions
- No threshold tuning or cross-validation
- Evaluates on training set (overfitting risk)

Future improvements (Phase 2+):
- Feature engineering and selection
- Threshold optimization via ROC curve
- Train/test split or cross-validation
- More sophisticated models (logistic regression, etc.)

## Testing

```bash
cargo test -p qf-01-fraud
```

Tests cover:
- Detector fitting and prediction
- Confusion matrix metrics
- Edge cases (zero division, perfect classification)

## Related Crates

- `qf-common` — Shared utilities (Transaction, load_transactions, compute_stats)
- `qf-02-loan` (future) — Loan default prediction
- `qf-03-stocks` (future) — Stock price analysis

## References

- Original dataset: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Book chapter: `docs/quant-finance/book/chapters/ch01.tex`
