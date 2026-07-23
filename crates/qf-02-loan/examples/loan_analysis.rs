//! Example: Loan default prediction with feature engineering and ROC-AUC.
//!
//! Demonstrates the complete loan scoring pipeline:
//! 1. Create synthetic loan applications
//! 2. Extract and normalize features
//! 3. Score with a linear model
//! 4. Evaluate with ROC-AUC
//!
//! Run from quant-lab directory:
//!   cargo run -p qf-02-loan --example loan_analysis

use qf_02_loan::{
    auc, roc_curve, FeatureExtractor, HomeOwnership, LinearScorer, LoanApplication, Normalizer,
    OneHotEncoder, BinaryClassifier,
};

fn main() {
    println!("Loan Default Prediction");
    println!("=======================\n");

    // Create synthetic loan applications (mix of defaults and non-defaults)
    let applications = create_synthetic_data();
    println!("Loaded {} loan applications", applications.len());

    let defaults = applications.iter().filter(|a| a.defaulted).count();
    println!(
        "Default rate: {:.1}% ({} defaults)\n",
        100.0 * defaults as f64 / applications.len() as f64,
        defaults
    );

    // Extract features for all applications
    println!("=== Feature Engineering ===");
    let features: Vec<Vec<f64>> = applications
        .iter()
        .map(extract_features)
        .collect();

    println!("Features per application: {}", features[0].len());
    println!("  - Loan-to-income ratio");
    println!("  - Interest rate");
    println!("  - DTI ratio");
    println!("  - Grade score (0-1)");
    println!("  - Home ownership (one-hot: 3 features)");

    // Normalize numerical features (first 4 columns)
    let normalized = normalize_features(&features);

    // Create scorer with hand-tuned weights
    // Higher weight = more predictive of default
    let weights = vec![
        0.8,  // loan-to-income: high loan relative to income = risky
        0.5,  // interest rate: higher rate = higher risk
        0.7,  // DTI: more debt = riskier
        1.0,  // grade score: strongest predictor
        0.2,  // RENT: slightly riskier
        -0.3, // OWN: less risky
        0.0,  // MORTGAGE: neutral
    ];
    let scorer = LinearScorer::new(weights, -1.5); // bias shifts threshold

    println!("\n=== Scoring Model ===");
    println!("Linear scorer with {} weights + bias", 7);

    // Score all applications
    let scores: Vec<f64> = normalized
        .iter()
        .map(|f| scorer.predict_proba(f))
        .collect();

    let labels: Vec<bool> = applications.iter().map(|a| a.defaulted).collect();

    // Compute ROC curve and AUC
    let roc_points = roc_curve(&scores, &labels);
    let auc_score = auc(&roc_points);

    println!("\n=== Evaluation Results ===");
    println!("AUC: {:.3}", auc_score);

    // Find optimal threshold (maximize TPR - FPR)
    let optimal = find_optimal_threshold(&roc_points);
    println!("Optimal threshold: {:.2} (Youden's J)", optimal.threshold);
    println!("  at FPR={:.2}, TPR={:.2}", optimal.fpr, optimal.tpr);

    // Evaluate at threshold 0.5
    let threshold = 0.5;
    let predictions: Vec<bool> = scores.iter().map(|&s| s > threshold).collect();
    let (tp, fp, _tn, fn_) = confusion_matrix(&predictions, &labels);

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("\nAt threshold {:.1}:", threshold);
    println!("  Precision: {:.2}", precision);
    println!("  Recall: {:.2}", recall);
    println!("  F1 Score: {:.2}", f1);

    // Show sample predictions
    println!("\n=== Sample Predictions ===");
    println!("{:<6} {:<8} {:<8} {:<8}", "App#", "Score", "Pred", "Actual");
    println!("{}", "-".repeat(32));
    for i in 0..5.min(applications.len()) {
        println!(
            "{:<6} {:<8.3} {:<8} {:<8}",
            i + 1,
            scores[i],
            if predictions[i] { "Default" } else { "OK" },
            if labels[i] { "Default" } else { "OK" }
        );
    }
}

/// Create synthetic loan applications for demonstration.
fn create_synthetic_data() -> Vec<LoanApplication> {
    vec![
        // Low-risk applications (grade A-B, low DTI, homeowners)
        LoanApplication::new(80000.0, 15000.0, 6.5, 0.15, 1, HomeOwnership::Own, false),
        LoanApplication::new(95000.0, 20000.0, 7.0, 0.18, 2, HomeOwnership::Mortgage, false),
        LoanApplication::new(120000.0, 25000.0, 6.8, 0.12, 1, HomeOwnership::Own, false),
        LoanApplication::new(75000.0, 10000.0, 7.2, 0.20, 2, HomeOwnership::Mortgage, false),
        LoanApplication::new(85000.0, 18000.0, 6.9, 0.16, 1, HomeOwnership::Own, false),
        // Medium-risk applications (grade C-D, moderate DTI)
        LoanApplication::new(55000.0, 20000.0, 12.5, 0.28, 3, HomeOwnership::Rent, false),
        LoanApplication::new(48000.0, 15000.0, 14.0, 0.32, 4, HomeOwnership::Rent, false),
        LoanApplication::new(62000.0, 22000.0, 11.8, 0.25, 3, HomeOwnership::Mortgage, false),
        LoanApplication::new(52000.0, 18000.0, 13.5, 0.30, 4, HomeOwnership::Rent, true),
        LoanApplication::new(45000.0, 16000.0, 15.0, 0.35, 4, HomeOwnership::Rent, true),
        // High-risk applications (grade E-G, high DTI, renters)
        LoanApplication::new(35000.0, 20000.0, 18.5, 0.42, 5, HomeOwnership::Rent, true),
        LoanApplication::new(32000.0, 18000.0, 20.0, 0.48, 6, HomeOwnership::Rent, true),
        LoanApplication::new(28000.0, 15000.0, 22.5, 0.55, 7, HomeOwnership::Rent, true),
        LoanApplication::new(38000.0, 22000.0, 19.0, 0.45, 6, HomeOwnership::Rent, true),
        LoanApplication::new(30000.0, 12000.0, 21.0, 0.50, 7, HomeOwnership::Rent, true),
        // Edge cases
        LoanApplication::new(65000.0, 25000.0, 10.5, 0.22, 3, HomeOwnership::Mortgage, false),
        LoanApplication::new(42000.0, 14000.0, 16.0, 0.38, 5, HomeOwnership::Rent, true),
        LoanApplication::new(58000.0, 19000.0, 13.0, 0.28, 4, HomeOwnership::Rent, false),
        LoanApplication::new(72000.0, 30000.0, 9.5, 0.20, 2, HomeOwnership::Mortgage, false),
        LoanApplication::new(40000.0, 20000.0, 17.5, 0.40, 5, HomeOwnership::Rent, true),
    ]
}

/// Extract features from a loan application.
fn extract_features(app: &LoanApplication) -> Vec<f64> {
    let mut features = vec![
        FeatureExtractor::loan_to_income(app.loan_amount, app.income),
        app.interest_rate,
        app.dti,
        FeatureExtractor::grade_to_score(app.grade),
    ];
    features.extend(OneHotEncoder::encode(&app.home_ownership));
    features
}

/// Normalize numerical features (first 4) to [0, 1].
fn normalize_features(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return vec![];
    }

    let num_numerical = 4;
    let mut normalizers = Vec::with_capacity(num_numerical);

    // Fit normalizers on each numerical column
    for col in 0..num_numerical {
        let values: Vec<f64> = features.iter().map(|f| f[col]).collect();
        normalizers.push(Normalizer::fit(&values));
    }

    // Transform all features
    features
        .iter()
        .map(|f| {
            let mut normalized = Vec::with_capacity(f.len());
            for (i, &val) in f.iter().enumerate() {
                if i < num_numerical {
                    normalized.push(normalizers[i].transform(val));
                } else {
                    normalized.push(val); // one-hot already 0/1
                }
            }
            normalized
        })
        .collect()
}

struct OptimalPoint {
    threshold: f64,
    fpr: f64,
    tpr: f64,
}

/// Find optimal threshold using Youden's J statistic (TPR - FPR).
fn find_optimal_threshold(roc_points: &[(f64, f64)]) -> OptimalPoint {
    let mut best_j = f64::NEG_INFINITY;
    let mut best_point = OptimalPoint {
        threshold: 0.5,
        fpr: 0.0,
        tpr: 0.0,
    };

    for (i, &(fpr, tpr)) in roc_points.iter().enumerate() {
        let j = tpr - fpr;
        if j > best_j {
            best_j = j;
            // Estimate threshold as position in sorted scores
            best_point = OptimalPoint {
                threshold: 1.0 - (i as f64 / roc_points.len() as f64),
                fpr,
                tpr,
            };
        }
    }

    best_point
}

/// Compute confusion matrix from predictions and labels.
fn confusion_matrix(predictions: &[bool], labels: &[bool]) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_ = 0;

    for (&pred, &actual) in predictions.iter().zip(labels.iter()) {
        match (pred, actual) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }

    (tp, fp, tn, fn_)
}
