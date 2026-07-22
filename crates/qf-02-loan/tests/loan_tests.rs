//! Integration tests for qf-02-loan.
//!
//! TDD contract: 17 tests covering all Phase 2 requirements.

use approx::assert_relative_eq;
use qf_02_loan::{
    HomeOwnership, LoanApplication, FeatureExtractor, OneHotEncoder, Normalizer,
    LinearScorer, roc_curve, auc, BinaryClassifier, Scorer,
};

// Test 1: LoanApplication creation
#[test]
fn test_loan_application_creation() {
    let loan = LoanApplication::new(
        50000.0,  // income
        20000.0,  // loan_amount
        5.5,      // interest_rate
        0.3,      // dti
        2,        // grade (B)
        HomeOwnership::Rent,
        false,    // defaulted
    );
    
    assert_eq!(loan.income, 50000.0);
    assert_eq!(loan.loan_amount, 20000.0);
    assert_eq!(loan.interest_rate, 5.5);
    assert_eq!(loan.dti, 0.3);
    assert_eq!(loan.grade, 2);
    assert_eq!(loan.home_ownership, HomeOwnership::Rent);
    assert!(!loan.defaulted);
}

// Test 2: debt-to-income ratio
#[test]
fn test_debt_to_income_ratio() {
    let dti = FeatureExtractor::debt_to_income(50000.0, 15000.0);
    assert_relative_eq!(dti, 0.3, epsilon = 1e-10);
}

// Test 3: loan-to-income ratio
#[test]
fn test_loan_to_income_ratio() {
    let lti = FeatureExtractor::loan_to_income(20000.0, 50000.0);
    assert_relative_eq!(lti, 0.4, epsilon = 1e-10);
}

// Test 4: grade A to score
#[test]
fn test_grade_to_score() {
    let score = FeatureExtractor::grade_to_score(1);  // A
    assert_relative_eq!(score, 0.0, epsilon = 1e-10);
}

// Test 5: grade G to score
#[test]
fn test_grade_to_score_g() {
    let score = FeatureExtractor::grade_to_score(7);  // G
    assert_relative_eq!(score, 1.0, epsilon = 1e-10);
}

// Test 6: one-hot RENT
#[test]
fn test_one_hot_encoder() {
    let encoded = OneHotEncoder::encode(&HomeOwnership::Rent);
    assert_eq!(encoded, vec![1.0, 0.0, 0.0]);
}

// Test 7: one-hot OWN
#[test]
fn test_one_hot_encoder_own() {
    let encoded = OneHotEncoder::encode(&HomeOwnership::Own);
    assert_eq!(encoded, vec![0.0, 1.0, 0.0]);
}

// Test 8: normalizer min-max
#[test]
fn test_normalizer_minmax() {
    let data = vec![10.0, 20.0, 30.0];
    let normalizer = Normalizer::fit(&data);
    
    assert_relative_eq!(normalizer.transform(10.0), 0.0, epsilon = 1e-10);
    assert_relative_eq!(normalizer.transform(20.0), 0.5, epsilon = 1e-10);
    assert_relative_eq!(normalizer.transform(30.0), 1.0, epsilon = 1e-10);
}

// Test 9: normalizer constant values
#[test]
fn test_normalizer_constant() {
    let data = vec![5.0, 5.0, 5.0];
    let normalizer = Normalizer::fit(&data);
    
    // All should map to 0.0 (avoid division by zero)
    assert_relative_eq!(normalizer.transform(5.0), 0.0, epsilon = 1e-10);
}

// Test 10: linear scorer basic
#[test]
fn test_linear_scorer_basic() {
    let scorer = LinearScorer::new(vec![1.0, 2.0], 0.0);
    let features = vec![1.0, 1.0];
    let score = scorer.score(&features);
    assert_relative_eq!(score, 3.0, epsilon = 1e-10);
}

// Test 11: sigmoid at zero
#[test]
fn test_sigmoid_zero() {
    let scorer = LinearScorer::new(vec![1.0], 0.0);
    let proba = scorer.predict_proba(&[0.0]);
    assert_relative_eq!(proba, 0.5, epsilon = 1e-10);
}

// Test 12: sigmoid large positive
#[test]
fn test_sigmoid_large_positive() {
    let scorer = LinearScorer::new(vec![1.0], 0.0);
    let proba = scorer.predict_proba(&[10.0]);
    assert!(proba > 0.999);
}

// Test 13: sigmoid large negative
#[test]
fn test_sigmoid_large_negative() {
    let scorer = LinearScorer::new(vec![1.0], 0.0);
    let proba = scorer.predict_proba(&[-10.0]);
    assert!(proba < 0.001);
}

// Test 14: predict_proba with score=0
#[test]
fn test_predict_proba() {
    let scorer = LinearScorer::new(vec![0.0], 0.0);
    let proba = scorer.predict_proba(&[100.0]);  // score will be 0
    assert_relative_eq!(proba, 0.5, epsilon = 1e-10);
}

// Test 15: ROC curve perfect classifier
#[test]
fn test_roc_curve_perfect() {
    // Perfect: all positives scored higher than negatives
    let scores = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
    let labels = vec![true, true, true, false, false, false];
    
    let points = roc_curve(&scores, &labels);
    let auc_val = auc(&points);
    
    assert_relative_eq!(auc_val, 1.0, epsilon = 1e-10);
}

// Test 16: ROC curve random classifier
#[test]
fn test_roc_curve_random() {
    // Random: interleaved positives and negatives
    let scores = vec![0.9, 0.7, 0.5, 0.3, 0.1];
    let labels = vec![true, false, true, false, true];
    
    let points = roc_curve(&scores, &labels);
    let auc_val = auc(&points);
    
    // Should be approximately 0.5 for random
    // With 3 pos and 2 neg, exact AUC depends on ordering
    assert!(auc_val > 0.3 && auc_val < 0.7);
}

// Test 17: AUC trapezoid integration
#[test]
fn test_auc_trapezoid() {
    // Triangle: (0,0) -> (0,1) -> (1,1)
    // Area = 0.5 * base * height + rectangle = 0.5 * 0 * 1 + 1 * 1 = 1.0
    // Actually: vertical line from (0,0) to (0,1) contributes 0,
    // then horizontal from (0,1) to (1,1) contributes 1*1 = 1.0
    let points = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    let auc_val = auc(&points);
    assert_relative_eq!(auc_val, 1.0, epsilon = 1e-10);
}

// Bonus test: AUC diagonal (already covered in eval.rs tests, but included for completeness)
#[test]
fn test_auc_diagonal() {
    let points = vec![(0.0, 0.0), (1.0, 1.0)];
    let auc_val = auc(&points);
    assert_relative_eq!(auc_val, 0.5, epsilon = 1e-10);
}
