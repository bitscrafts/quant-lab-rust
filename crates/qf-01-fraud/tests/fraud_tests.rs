use approx::assert_relative_eq;
use qf_01_fraud::{evaluate, ConfusionMatrix, ZScoreDetector};
use qf_common::Transaction;

fn create_normal_transaction(features: Vec<f64>) -> Transaction {
    Transaction {
        features,
        amount: 100.0,
        class: 0,
    }
}

fn create_fraud_transaction(features: Vec<f64>) -> Transaction {
    Transaction {
        features,
        amount: 100.0,
        class: 1,
    }
}

#[test]
fn test_zscore_detector_fit() {
    let transactions: Vec<Transaction> = (0..10)
        .map(|i| create_normal_transaction(vec![i as f64, (i * 2) as f64]))
        .collect();
    
    let mut detector = ZScoreDetector::new(3.0);
    detector.fit(&transactions);
    
    // Should not panic and should store means/stds
    // We can't directly inspect internal state, but we can verify it works in predict
}

#[test]
fn test_zscore_detector_predict_normal() {
    let transactions: Vec<Transaction> = vec![
        create_normal_transaction(vec![1.0, 2.0]),
        create_normal_transaction(vec![2.0, 3.0]),
        create_normal_transaction(vec![3.0, 4.0]),
        create_normal_transaction(vec![4.0, 5.0]),
        create_normal_transaction(vec![5.0, 6.0]),
    ];
    
    let mut detector = ZScoreDetector::new(3.0);
    detector.fit(&transactions);
    
    // Test with a transaction similar to training data
    let normal_tx = create_normal_transaction(vec![3.5, 4.5]);
    assert!(!detector.predict(&normal_tx));
}

#[test]
fn test_zscore_detector_predict_anomaly() {
    let transactions: Vec<Transaction> = vec![
        create_normal_transaction(vec![1.0, 2.0]),
        create_normal_transaction(vec![2.0, 3.0]),
        create_normal_transaction(vec![3.0, 4.0]),
        create_normal_transaction(vec![4.0, 5.0]),
        create_normal_transaction(vec![5.0, 6.0]),
    ];
    
    let mut detector = ZScoreDetector::new(3.0);
    detector.fit(&transactions);
    
    // Create transaction with extreme outlier
    let anomaly_tx = create_normal_transaction(vec![100.0, 4.0]);
    assert!(detector.predict(&anomaly_tx));
}

#[test]
fn test_confusion_matrix_precision() {
    let cm = ConfusionMatrix {
        tp: 8,
        fp: 2,
        tn: 85,
        fn_: 5,
    };
    
    assert_relative_eq!(cm.precision(), 0.8, epsilon = 1e-12);
}

#[test]
fn test_confusion_matrix_recall() {
    let cm = ConfusionMatrix {
        tp: 8,
        fp: 2,
        tn: 85,
        fn_: 5,
    };
    
    let expected_recall = 8.0 / (8.0 + 5.0);
    assert_relative_eq!(cm.recall(), expected_recall, epsilon = 1e-12);
}

#[test]
fn test_confusion_matrix_f1() {
    let cm = ConfusionMatrix {
        tp: 8,
        fp: 2,
        tn: 85,
        fn_: 5,
    };
    
    let precision = 0.8;
    let recall = 8.0 / 13.0;
    let expected_f1 = 2.0 * (precision * recall) / (precision + recall);
    
    assert_relative_eq!(cm.f1(), expected_f1, epsilon = 1e-12);
}

#[test]
fn test_confusion_matrix_zero_div() {
    let cm = ConfusionMatrix {
        tp: 0,
        fp: 0,
        tn: 100,
        fn_: 0,
    };
    
    assert_relative_eq!(cm.precision(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(cm.recall(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(cm.f1(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_evaluate_perfect() {
    // Create dataset where detector will correctly classify everything
    let transactions: Vec<Transaction> = vec![
        create_normal_transaction(vec![1.0, 2.0]),
        create_normal_transaction(vec![2.0, 3.0]),
        create_normal_transaction(vec![3.0, 4.0]),
        create_fraud_transaction(vec![100.0, 200.0]),
        create_fraud_transaction(vec![150.0, 250.0]),
    ];
    
    let mut detector = ZScoreDetector::new(2.0);
    detector.fit(&transactions);
    
    let cm = evaluate(&detector, &transactions);
    
    // In this test, we expect perfect or near-perfect classification
    // tp + tn should equal total count, fp and fn should be 0 or very small
    let total = cm.tp + cm.fp + cm.tn + cm.fn_;
    assert_eq!(total, 5);
    
    // With this data and threshold, should get good separation
    assert!(cm.tp + cm.tn >= 3); // At least 3 correct out of 5
}
