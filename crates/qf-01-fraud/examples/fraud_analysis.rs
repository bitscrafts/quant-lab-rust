//! Credit card fraud detection example.
//!
//! Demonstrates z-score anomaly detection on credit card transaction data.
//! Uses a sample dataset bundled with the repository.
//!
//! Run from quant-lab directory:
//!   cargo run -p qf-01-fraud --example fraud_analysis

use qf_01_fraud::{evaluate, ZScoreDetector};
use qf_common::load_transactions;

fn main() {
    println!("Credit Card Fraud Detection");
    println!("============================\n");

    // Sample dataset bundled with the repository
    let data_path = "data/creditcard_sample.csv";

    println!("Loading transactions from: {}", data_path);

    match load_transactions(data_path) {
        Ok(transactions) => {
            println!("Loaded {} transactions", transactions.len());

            // Count actual fraud cases
            let num_fraud = transactions.iter().filter(|tx| tx.class == 1).count();
            println!("  Normal: {}", transactions.len() - num_fraud);
            println!("  Fraud: {}\n", num_fraud);

            // Train detector
            println!("Training z-score detector (threshold=3.0)...");
            let mut detector = ZScoreDetector::new(3.0);
            detector.fit(&transactions);

            // Evaluate
            println!("Evaluating on training set...\n");
            let cm = evaluate(&detector, &transactions);

            println!("Confusion Matrix:");
            println!("  TP: {}, FP: {}", cm.tp, cm.fp);
            println!("  TN: {}, FN: {}\n", cm.tn, cm.fn_);

            println!("Metrics:");
            println!("  Precision: {:.4}", cm.precision());
            println!("  Recall: {:.4}", cm.recall());
            println!("  F1: {:.4}", cm.f1());

            println!("\nNote: This uses a small sample dataset for demonstration.");
            println!("For full analysis, download the complete dataset from:");
            println!("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud");
        }
        Err(e) => {
            eprintln!("Error loading data: {}", e);
            eprintln!("\nMake sure you run from the quant-lab directory:");
            eprintln!("  cd crates/quant-lab");
            eprintln!("  cargo run -p qf-01-fraud --example fraud_analysis");
            std::process::exit(1);
        }
    }
}
