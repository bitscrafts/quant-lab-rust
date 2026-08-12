//! Exercise solutions for Chapter 1: Credit Card Fraud
//!
//! Run: `cargo run -p quant-lib --example solutions-ch01_fraud_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch01_fraud_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::prelude::*;
use std::path::PathBuf;

/// Resolve path to the bundled `creditcard_sample.csv` from the quant-lib crate dir.
fn creditcard_csv_path(crate_dir: &str) -> PathBuf {
    let mut p = PathBuf::from(crate_dir);
    p.pop(); // quant-lib
    p.pop(); // crates
    p.pop(); // quant-lab
    p.push("quant-lab");
    p.push("data");
    p.push("creditcard_sample.csv");
    p
}

/// A single transaction: 28 PCA features (V1..V28) + amount + class label.
struct Transaction {
    features: Vec<f64>,
    amount: f64,
    is_fraud: bool,
}

/// Load the bundled `creditcard_sample.csv`.
/// Columns: Time, V1..V28, Amount, Class (31 total).
fn load_creditcard(path: &PathBuf) -> Vec<Transaction> {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for (i, line) in contents.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 31 {
            continue;
        }
        let feats: Vec<f64> = (1..29).map(|j| cols[j].parse().unwrap_or(0.0)).collect();
        let amount: f64 = cols[29].parse().unwrap_or(0.0);
        let class: u8 = cols[30].parse().unwrap_or(0);
        rows.push(Transaction {
            features: feats,
            amount,
            is_fraud: class == 1,
        });
    }
    rows
}

/// Per-feature mean and std from a training slice.
fn fit_zscore(rows: &[Transaction]) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len();
    let k = if n > 0 { rows[0].features.len() } else { 0 };
    let mut mean = vec![0.0_f64; k];
    let mut std = vec![0.0_f64; k];
    for j in 0..k {
        let col: Vec<f64> = rows.iter().map(|r| r.features[j]).collect();
        mean[j] = quant_lib::core::mean(&col);
        std[j] = std_dev(&col).unwrap_or(1.0).max(1e-12);
    }
    (mean, std)
}

/// Per-feature z-scores for one row given fitted mean/std.
fn zscores(row: &Transaction, mean: &[f64], std: &[f64]) -> Vec<f64> {
    row.features
        .iter()
        .zip(mean.iter().zip(std.iter()))
        .map(|(&x, (&m, &s))| (x - m) / s)
        .collect()
}

/// Flag a row as anomaly if any |z| > threshold. Returns max |z|.
fn predict(row: &Transaction, mean: &[f64], std: &[f64], threshold: f64) -> (bool, f64) {
    let zs = zscores(row, mean, std);
    let max_z = zs.iter().map(|z| z.abs()).fold(0.0_f64, f64::max);
    (max_z > threshold, max_z)
}

/// Precision / recall / F1 over a labelled slice given a threshold.
fn precision_recall_f1(
    rows: &[Transaction],
    mean: &[f64],
    std: &[f64],
    threshold: f64,
) -> (f64, f64, f64) {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    for r in rows {
        let (flag, _) = predict(r, mean, std, threshold);
        if flag && r.is_fraud {
            tp += 1;
        } else if flag && !r.is_fraud {
            fp += 1;
        } else if !flag && r.is_fraud {
            fn_ += 1;
        }
    }
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
    (precision, recall, f1)
}

fn main() {
    println!("=== Chapter 1: Credit Card Fraud - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 1 exercises complete.");
}

fn exercise_1() {
    println!("1. Threshold Sensitivity:");
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let (m, s) = fit_zscore(&rows);
    println!(
        "   loaded {} rows ({} fraud)",
        rows.len(),
        rows.iter().filter(|r| r.is_fraud).count()
    );
    for &thr in &[2.0, 2.5, 3.0, 3.5, 4.0] {
        let (p, r, f1) = precision_recall_f1(&rows, &m, &s, thr);
        println!("   thr={thr:.1}: precision={p:.3}, recall={r:.3}, F1={f1:.3}");
    }
}

fn exercise_2() {
    println!("\n2. Feature Analysis:");
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let (m, s) = fit_zscore(&rows);
    let fraud: Vec<&Transaction> = rows.iter().filter(|r| r.is_fraud).collect();
    if fraud.is_empty() {
        println!("   no fraud rows in sample; cannot rank features");
        return;
    }
    let k = m.len();
    let mut mean_abs_z = vec![0.0_f64; k];
    for r in &fraud {
        let zs = zscores(r, &m, &s);
        for j in 0..k {
            mean_abs_z[j] += zs[j].abs();
        }
    }
    for v in &mut mean_abs_z {
        *v /= fraud.len() as f64;
    }
    let best = mean_abs_z
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(j, v)| (j, *v))
        .unwrap_or((0, 0.0));
    println!(
        "   most discriminative feature: V{} (mean |z| on fraud = {:.3})",
        best.0 + 1,
        best.1
    );
}

fn exercise_3() {
    println!("\n3. Train/Test Split (chronological 80/20):");
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let split = (rows.len() * 4) / 5;
    let train = &rows[..split];
    let test = &rows[split..];
    let (m, s) = fit_zscore(train);
    for &thr in &[3.0, 3.5] {
        let (pt, rt, _) = precision_recall_f1(train, &m, &s, thr);
        let (pe, re, _) = precision_recall_f1(test, &m, &s, thr);
        println!("   thr={thr}: train P/R={pt:.3}/{rt:.3}, test P/R={pe:.3}/{re:.3}");
    }
}

fn exercise_4() {
    println!("\n4. Amount Feature (F1 comparison):");
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    // Baseline: 28 PCA features only.
    let (m, s) = fit_zscore(&rows);
    let (_, _, f1_base) = precision_recall_f1(&rows, &m, &s, 3.0);
    // Augmented: append z-scored amount as a 29th feature.
    let amounts: Vec<f64> = rows.iter().map(|r| r.amount).collect();
    let am = mean(&amounts);
    let asd = std_dev(&amounts).unwrap_or(1.0).max(1e-12);
    let mut aug: Vec<Transaction> = rows
        .iter()
        .map(|r| {
            let mut f = r.features.clone();
            f.push((r.amount - am) / asd);
            Transaction {
                features: f,
                amount: r.amount,
                is_fraud: r.is_fraud,
            }
        })
        .collect();
    let (m2, s2) = fit_zscore(&aug);
    let (_, _, f1_aug) = precision_recall_f1(&aug, &m2, &s2, 3.0);
    println!("   F1 baseline (28 feats) = {f1_base:.3}, F1 with amount = {f1_aug:.3}");
    // suppress unused warning
    let _ = &mut aug;
}

fn exercise_5() {
    println!("\n5. Trait Extraction (AnomalyDetector):");
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let mut det = ZScoreDetector::new(3.0);
    det.fit(&rows.iter().map(|r| r.features.clone()).collect::<Vec<_>>());
    let n_flag = rows
        .iter()
        .filter(|r| det.predict(&r.features).is_anomaly)
        .count();
    let n_fraud = rows.iter().filter(|r| r.is_fraud).count();
    println!("   ZScoreDetector(thr=3.0) flagged {n_flag} rows ({n_fraud} actual fraud)");
}

// ---- Chapter 1 Exercise 5: AnomalyDetector trait ----

/// Result of an anomaly prediction.
pub struct AnomalyResult {
    pub is_anomaly: bool,
    pub score: f64,
    pub triggering_features: Vec<usize>,
}

/// Trait for multivariate anomaly detectors.
pub trait AnomalyDetector {
    fn fit(&mut self, x: &[Vec<f64>]);
    fn predict(&self, x: &[f64]) -> AnomalyResult;
}

/// Z-score-based anomaly detector: flags a point if any feature's |z| exceeds threshold.
pub struct ZScoreDetector {
    threshold: f64,
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl ZScoreDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            mean: Vec::new(),
            std: Vec::new(),
        }
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn fit(&mut self, x: &[Vec<f64>]) {
        if x.is_empty() {
            return;
        }
        let k = x[0].len();
        self.mean = vec![0.0; k];
        self.std = vec![0.0; k];
        for j in 0..k {
            let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
            self.mean[j] = mean(&col);
            self.std[j] = std_dev(&col).unwrap_or(1.0).max(1e-12);
        }
    }

    fn predict(&self, x: &[f64]) -> AnomalyResult {
        let mut max_z = 0.0_f64;
        let mut triggers = Vec::new();
        for (j, (&xi, (&m, &s))) in x
            .iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .enumerate()
        {
            let z = (xi - m) / s;
            if z.abs() > self.threshold {
                triggers.push(j);
            }
            max_z = max_z.max(z.abs());
        }
        AnomalyResult {
            is_anomaly: max_z > self.threshold,
            score: max_z,
            triggering_features: triggers,
        }
    }
}

#[test]
fn test_ex1_threshold_sweep_finite() {
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    assert!(!rows.is_empty(), "data file must load");
    let (m, s) = fit_zscore(&rows);
    for &thr in &[2.0, 2.5, 3.0, 3.5, 4.0] {
        let (p, r, f1) = precision_recall_f1(&rows, &m, &s, thr);
        assert!(p.is_finite() && r.is_finite() && f1.is_finite());
        assert!((0.0..=1.0).contains(&p) && (0.0..=1.0).contains(&r));
    }
}

#[test]
fn test_ex2_most_discriminative_feature() {
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let (m, s) = fit_zscore(&rows);
    let fraud: Vec<&Transaction> = rows.iter().filter(|r| r.is_fraud).collect();
    if fraud.is_empty() {
        return;
    }
    let k = m.len();
    let mut mean_abs_z = vec![0.0_f64; k];
    for r in &fraud {
        let zs = zscores(r, &m, &s);
        for j in 0..k {
            mean_abs_z[j] += zs[j].abs();
        }
    }
    for v in &mut mean_abs_z {
        *v /= fraud.len() as f64;
    }
    let best_idx = mean_abs_z
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(j, _)| j)
        .unwrap_or(0);
    assert!(
        mean_abs_z[best_idx] > 0.0,
        "best feature should have positive mean |z|"
    );
}

#[test]
fn test_ex3_chronological_split() {
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let split = (rows.len() * 4) / 5;
    assert!(split > 0 && split < rows.len());
    let train = &rows[..split];
    let test = &rows[split..];
    let (m, s) = fit_zscore(train);
    let (p, r, _) = precision_recall_f1(test, &m, &s, 3.0);
    assert!(p.is_finite() && r.is_finite());
}

#[test]
fn test_ex4_amount_feature_runs() {
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let (m, s) = fit_zscore(&rows);
    let (_, _, f1_base) = precision_recall_f1(&rows, &m, &s, 3.0);
    assert!(f1_base.is_finite());
    let amounts: Vec<f64> = rows.iter().map(|r| r.amount).collect();
    let am = mean(&amounts);
    let asd = std_dev(&amounts).unwrap_or(1.0).max(1e-12);
    let aug: Vec<Transaction> = rows
        .iter()
        .map(|r| {
            let mut f = r.features.clone();
            f.push((r.amount - am) / asd);
            Transaction {
                features: f,
                amount: r.amount,
                is_fraud: r.is_fraud,
            }
        })
        .collect();
    assert_eq!(aug[0].features.len(), rows[0].features.len() + 1);
}

#[test]
fn test_ex5_trait_implementation() {
    let path = creditcard_csv_path(env!("CARGO_MANIFEST_DIR"));
    let rows = load_creditcard(&path);
    let x: Vec<Vec<f64>> = rows.iter().map(|r| r.features.clone()).collect();
    let mut det = ZScoreDetector::new(3.0);
    det.fit(&x);
    let res = det.predict(&rows[0].features);
    assert!(res.score.is_finite());
    assert!(res.triggering_features.len() <= rows[0].features.len());
}
