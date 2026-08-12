//! Exercise solutions for Chapter 2: Loan Default
//!
//! Run: `cargo run -p quant-lib --example solutions-ch02_loan_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch02_loan_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::prelude::*;

/// A loan applicant record: features + binary default label.
struct Applicant {
    features: Vec<f64>,
    defaulted: bool,
}

/// Sigmoid function.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Linear scorer: sigma(w . x + b). Returns probability of default.
fn score(x: &[f64], w: &[f64], b: f64) -> f64 {
    let z: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + b;
    sigmoid(z)
}

/// AUC via the trapezoidal rule on the ROC curve (Mann-Whitney U / (n+ * n-)).
fn auc(applicants: &[Applicant], w: &[f64], b: f64) -> f64 {
    let mut preds: Vec<(f64, bool)> = applicants
        .iter()
        .map(|a| (score(&a.features, w, b), a.defaulted))
        .collect();
    preds.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos = applicants.iter().filter(|a| a.defaulted).count() as f64;
    let n_neg = applicants.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    // Rank-sum: for each positive, count negatives with lower predicted score.
    let mut rank_sum = 0.0_f64;
    for (i, &(_, is_pos)) in preds.iter().enumerate() {
        if is_pos {
            let n_below = preds.iter().take(i).filter(|(_, neg)| !*neg).count() as f64;
            rank_sum += n_below;
        }
    }
    rank_sum / (n_pos * n_neg)
}

/// F1 at a given threshold.
fn f1_at(applicants: &[Applicant], w: &[f64], b: f64, thr: f64) -> f64 {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_ = 0u64;
    for a in applicants {
        let p = score(&a.features, w, b);
        let flag = p >= thr;
        if flag && a.defaulted {
            tp += 1;
        } else if flag {
            fp += 1;
        } else if a.defaulted {
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
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Generate synthetic loan applicants: 3 features (credit_score, debt_to_income, income),
/// default correlated with high debt-to-income and low credit score.
fn make_applicants(n: usize, seed: u64) -> Vec<Applicant> {
    let mut rng = XorShift64::new(seed);
    let normal = Normal::standard();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let credit = 300.0 + 500.0 * rng.next_f64(); // 300..800
        let dti = 0.05 + 0.45 * rng.next_f64(); // 5%..50%
        let income = 20_000.0 + 80_000.0 * rng.next_f64();
        // Higher default risk: low credit, high dti.
        let logit =
            -3.0 + 0.004 * (600.0 - credit) + 6.0 * (dti - 0.2) + 0.1 * normal.sample(&mut rng);
        let p = sigmoid(logit);
        let defaulted = rng.next_f64() < p;
        out.push(Applicant {
            features: vec![
                (credit - 600.0) / 100.0,
                (dti - 0.25) / 0.15,
                (income - 50_000.0) / 30_000.0,
            ],
            defaulted,
        });
    }
    out
}

fn main() {
    println!("=== Chapter 2: Loan Default - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 2 exercises complete.");
}

fn exercise_1() {
    println!("1. Weight Sensitivity (AUC under +/-50% weight perturbation):");
    let data = make_applicants(500, 42);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let base_auc = auc(&data, &w, b);
    println!("   base AUC = {base_auc:.3}");
    for j in 0..w.len() {
        let mut w_up = w.clone();
        let mut w_dn = w.clone();
        w_up[j] *= 1.5;
        w_dn[j] *= 0.5;
        let auc_up = auc(&data, &w_up, b);
        let auc_dn = auc(&data, &w_dn, b);
        println!("   w[{j}] +50%: AUC={auc_up:.3}, -50%: AUC={auc_dn:.3}");
    }
}

fn exercise_2() {
    println!("\n2. Threshold Selection (F1-maximising threshold):");
    let data = make_applicants(500, 42);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let base_rate = data.iter().filter(|a| a.defaulted).count() as f64 / data.len() as f64;
    let mut best_thr = 0.5;
    let mut best_f1 = 0.0;
    let mut results = Vec::new();
    let mut thr = 0.01;
    while thr <= 0.99 {
        let f1 = f1_at(&data, &w, b, thr);
        results.push((thr, f1));
        if f1 > best_f1 {
            best_f1 = f1;
            best_thr = thr;
        }
        thr += 0.02;
    }
    println!("   base rate (default fraction) = {base_rate:.3}");
    println!("   best F1 = {best_f1:.3} at threshold = {best_thr:.3}");
}

fn exercise_3() {
    println!("\n3. Feature Importance (single-feature AUC):");
    let data = make_applicants(500, 42);
    let b = -0.5;
    let mut aucs = Vec::new();
    for j in 0..3 {
        let mut w = vec![0.0, 0.0, 0.0];
        w[j] = 1.0;
        let a = auc(&data, &w, b);
        aucs.push((j, a));
    }
    aucs.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (j, a) in &aucs {
        println!("   feature {j}: AUC = {a:.3}");
    }
}

fn exercise_4() {
    println!("\n4. Cross-Validation (5 chronological folds, AUC mean/std):");
    let data = make_applicants(500, 42);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let fold = data.len() / 5;
    let mut aucs = Vec::new();
    for i in 0..5 {
        let start = i * fold;
        let end = if i == 4 { data.len() } else { (i + 1) * fold };
        let a = auc(&data[start..end], &w, b);
        aucs.push(a);
        println!("   fold {i}: AUC = {a:.3}");
    }
    let mean_auc = mean(&aucs);
    let std_auc = std_dev(&aucs).unwrap_or(0.0);
    println!("   AUC mean = {mean_auc:.3}, std = {std_auc:.3}");
}

fn exercise_5() {
    println!("\n5. Calibration Plot (deciles: predicted prob vs empirical default rate):");
    let data = make_applicants(500, 42);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let mut preds: Vec<(f64, bool)> = data
        .iter()
        .map(|a| (score(&a.features, &w, b), a.defaulted))
        .collect();
    preds.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = preds.len();
    let decile = n / 10;
    for d in 0..10 {
        let start = d * decile;
        let end = if d == 9 { n } else { (d + 1) * decile };
        let bin = &preds[start..end];
        if bin.is_empty() {
            continue;
        }
        let mean_pred = mean(&bin.iter().map(|(p, _)| *p).collect::<Vec<_>>());
        let emp = bin.iter().filter(|(_, def)| *def).count() as f64 / bin.len() as f64;
        println!("   decile {d}: mean_pred={mean_pred:.3}, empirical_default={emp:.3}");
    }
}

#[test]
fn test_ex1_weight_sensitivity_auc_finite() {
    let data = make_applicants(200, 7);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let base = auc(&data, &w, b);
    assert!(base.is_finite() && (0.0..=1.0).contains(&base));
    let mut w_up = w.clone();
    w_up[1] *= 1.5;
    let up = auc(&data, &w_up, b);
    assert!(up.is_finite());
}

#[test]
fn test_ex2_threshold_selection_f1_max() {
    let data = make_applicants(300, 11);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let mut best = 0.0;
    let mut thr = 0.01;
    while thr <= 0.99 {
        let f1 = f1_at(&data, &w, b, thr);
        if f1 > best {
            best = f1;
        }
        thr += 0.05;
    }
    assert!(best >= 0.0 && best <= 1.0);
}

#[test]
fn test_ex3_feature_importance_ranks() {
    let data = make_applicants(300, 5);
    let b = -0.5;
    let aucs: Vec<f64> = (0..3)
        .map(|j| {
            let mut w = vec![0.0, 0.0, 0.0];
            w[j] = 1.0;
            auc(&data, &w, b)
        })
        .collect();
    assert!(
        aucs.iter()
            .all(|a| a.is_finite() && (0.0..=1.0).contains(a))
    );
}

#[test]
fn test_ex4_cross_validation_std() {
    let data = make_applicants(250, 9);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let fold = data.len() / 5;
    let aucs: Vec<f64> = (0..5)
        .map(|i| {
            let start = i * fold;
            let end = if i == 4 { data.len() } else { (i + 1) * fold };
            auc(&data[start..end], &w, b)
        })
        .collect();
    let s = std_dev(&aucs).unwrap_or(0.0);
    assert!(s.is_finite() && s >= 0.0);
}

#[test]
fn test_ex5_calibration_deciles() {
    let data = make_applicants(300, 13);
    let w = vec![0.8, 1.2, 0.3];
    let b = -0.5;
    let mut preds: Vec<(f64, bool)> = data
        .iter()
        .map(|a| (score(&a.features, &w, b), a.defaulted))
        .collect();
    preds.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(preds.len(), data.len());
    // Highest decile should have higher empirical default rate than lowest decile.
    let n = preds.len();
    let low = &preds[..n / 10];
    let high = &preds[9 * (n / 10)..];
    let low_rate = low.iter().filter(|(_, d)| *d).count() as f64 / low.len() as f64;
    let high_rate = high.iter().filter(|(_, d)| *d).count() as f64 / high.len() as f64;
    assert!(
        high_rate >= low_rate,
        "high decile default rate should be >= low decile"
    );
}
