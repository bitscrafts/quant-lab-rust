# qf-02-loan — Loan Default Prediction

Phase 2 of the quant-finance curriculum: Risk basics through credit scoring and ROC-AUC evaluation.

## Overview

This crate implements fundamental credit risk concepts:
- **Feature engineering**: Debt-to-income ratio, loan-to-income ratio, credit grade scoring
- **Categorical encoding**: One-hot encoding for home ownership status
- **Feature normalization**: Min-max scaling to [0, 1]
- **Linear scoring model**: Weighted sum with sigmoid probability
- **ROC-AUC evaluation**: Hand-rolled ROC curve computation and area under curve calculation

## Key Concepts

### Feature Engineering

Credit risk models require derived features that capture borrower risk:

```rust
use qf_02_loan::FeatureExtractor;

// Debt-to-income ratio: higher = more leveraged
let dti = FeatureExtractor::debt_to_income(income, debt);

// Loan-to-income ratio: higher = larger loan relative to income
let lti = FeatureExtractor::loan_to_income(loan_amount, income);

// Credit grade to risk score: A (1) → 0.0, G (7) → 1.0
let risk_score = FeatureExtractor::grade_to_score(grade);
```

### Categorical Encoding

Convert categorical features to numeric representations:

```rust
use qf_02_loan::{HomeOwnership, OneHotEncoder};

let encoded = OneHotEncoder::encode(&HomeOwnership::Rent);
// Returns [1.0, 0.0, 0.0] for [RENT, OWN, MORTGAGE]
```

### Feature Normalization

Scale features to [0, 1] using min-max normalization:

```rust
use qf_02_loan::Normalizer;

let data = vec![10.0, 20.0, 30.0];
let normalizer = Normalizer::fit(&data);

let scaled = normalizer.transform(20.0);  // Returns 0.5
```

### Scoring Model

Linear model with sigmoid probability:

```rust
use qf_02_loan::{LinearScorer, BinaryClassifier, Scorer};

let scorer = LinearScorer::new(
    vec![1.5, -0.8, 0.3],  // weights
    -2.0,                   // bias
);

let score = scorer.score(&features);           // Linear score
let prob = scorer.predict_proba(&features);    // Sigmoid probability
let prediction = scorer.predict(&features);    // Binary prediction (threshold 0.5)
```

### ROC-AUC Evaluation

Evaluate model performance using ROC curve and AUC:

```rust
use qf_02_loan::{roc_curve, auc};

let scores = vec![0.9, 0.8, 0.3, 0.2];
let labels = vec![true, true, false, false];

let points = roc_curve(&scores, &labels);  // (FPR, TPR) pairs
let auc_val = auc(&points);                // Area under curve (0.0 to 1.0)

println!("AUC: {:.3}", auc_val);  // Higher = better discrimination
```

## Architecture

This crate follows the trait-based modular design principle:

```rust
pub trait BinaryClassifier {
    fn predict(&self, features: &[f64]) -> bool;
    fn predict_proba(&self, features: &[f64]) -> f64;
}

pub trait Scorer {
    fn score(&self, features: &[f64]) -> f64;
}
```

This enables composable, reusable implementations across quant projects.

## Modules

- **`data`**: LoanApplication struct and CSV loading
- **`features`**: FeatureExtractor, OneHotEncoder, Normalizer
- **`scorer`**: LinearScorer with sigmoid
- **`eval`**: ROC curve and AUC computation
- **`traits`**: BinaryClassifier and Scorer traits

## Learning Objectives

Phase 2 teaches:
1. Feature engineering for credit risk
2. Categorical variable encoding
3. Feature scaling and normalization
4. Probability calibration via sigmoid
5. Model evaluation with ROC-AUC
6. Hand-rolling ML primitives (no external ML libraries)

## Testing

All 17 tests from the TDD contract pass:

```bash
cargo test -p qf-02-loan
```

Tests cover:
- Feature extraction formulas
- Encoding correctness
- Normalization edge cases (constant values)
- Sigmoid properties
- ROC-AUC computation (perfect, random, trapezoid)

## Next Steps

Phase 3 will cover market data (OHLCV), simple moving averages, and stock price analysis.

## References

- Credit scoring fundamentals
- ROC-AUC methodology
- Feature engineering for financial data

For the complete curriculum, see `docs/quant-finance/spec.md`.
