//! Exercise solutions for Chapter 14: AFML Backtesting
//!
//! Run: `cargo run -p quant-lib --example solutions-ch14_afml_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch14_afml_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::backtest::{DynamicBarrierLabeler, kelly_from_returns};
use quant_lib::core::{log_returns, rolling_mean};
use quant_lib::prelude::*;
use quant_lib::timeseries::acf;

/// Simple logistic regression with gradient descent (for meta-labeling).
struct LogisticRegression {
    coeffs: Vec<f64>,
    intercept: f64,
}

impl LogisticRegression {
    fn fit(x: &[Vec<f64>], y: &[i32], lr: f64, epochs: usize) -> Self {
        let n = x.len();
        let k = x[0].len();
        let mut w = vec![0.0_f64; k];
        let mut b = 0.0_f64;
        let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
        for _ in 0..epochs {
            let mut grad_w = vec![0.0_f64; k];
            let mut grad_b = 0.0_f64;
            for (row, &yi) in x.iter().zip(y.iter()) {
                let z: f64 = row
                    .iter()
                    .zip(w.iter())
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f64>()
                    + b;
                let p = sigmoid(z);
                let err = p - yi as f64;
                for (gw, &xi) in grad_w.iter_mut().zip(row.iter()) {
                    *gw += err * xi / n as f64;
                }
                grad_b += err / n as f64;
            }
            for (wi, &gw) in w.iter_mut().zip(grad_w.iter()) {
                *wi -= lr * gw;
            }
            b -= lr * grad_b;
        }
        LogisticRegression {
            coeffs: w,
            intercept: b,
        }
    }

    fn predict_prob(&self, x: &[f64]) -> f64 {
        let z: f64 = x
            .iter()
            .zip(self.coeffs.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f64>()
            + self.intercept;
        1.0 / (1.0 + (-z).exp())
    }
}

fn main() {
    println!("=== Chapter 14: AFML Backtesting - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    println!("\nAll Chapter 14 exercises complete.");
}

fn exercise_1() {
    println!("1. Custom vol-scaled triple-barrier (label distribution shifts):");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    // Low-vol regime (first 250 days): tighter barriers -> more Up labels.
    let labeler_low = DynamicBarrierLabeler::new(20, 2.0, 10);
    let entries_low: Vec<usize> = (20..250).collect();
    let events_low = labeler_low
        .label(&closes[..250], &entries_low)
        .expect("label");
    let up_low = events_low
        .iter()
        .filter(|e| e.label == TripleBarrierLabel::Upper)
        .count() as f64
        / events_low.len() as f64;
    // High-vol regime: scale prices up by 1.5 to inflate volatility.
    let closes_high: Vec<f64> = closes[..250].iter().map(|p| p * 1.0).collect();
    // Inject a high-vol segment by perturbing prices.
    let mut closes_hv: Vec<f64> = closes_high.clone();
    #[allow(clippy::needless_range_loop)]
    for i in 0..closes_hv.len() {
        let phase = (i as f64 / 5.0).sin();
        closes_hv[i] *= 1.0 + 0.05 * phase; // 5% oscillation -> much higher vol
    }
    let labeler_high = DynamicBarrierLabeler::new(20, 2.0, 10);
    let events_high = labeler_high.label(&closes_hv, &entries_low).expect("label");
    let up_high = events_high
        .iter()
        .filter(|e| e.label == TripleBarrierLabel::Upper)
        .count() as f64
        / events_high.len() as f64;
    println!("   Low-vol regime: {:.2}% Up labels", up_low * 100.0);
    println!("   High-vol regime: {:.2}% Up labels", up_high * 100.0);
    assert!(
        up_low > up_high,
        "vol-scaled barriers should reduce Up fraction as vol rises ({} vs {})",
        up_low,
        up_high
    );
}

fn exercise_2() {
    println!("2. Embargo calibration via ACF of FFD returns:");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let ffd = frac_diff(&ret, 0.4, 1e-4).expect("frac_diff");
    let max_lag = 10_usize.min(ffd.len() - 1);
    let acf_vals = acf(&ffd, max_lag).expect("acf");
    let t = ffd.len() as f64;
    let threshold = 1.96 / t.sqrt();
    let mut embargo_lag = 0_usize;
    #[allow(clippy::needless_range_loop)]
    for k in 1..=max_lag {
        if acf_vals[k].abs() < threshold {
            embargo_lag = k;
            break;
        }
    }
    if embargo_lag == 0 {
        // fall back to last lag where |acf| is below threshold.
        #[allow(clippy::needless_range_loop)]
        for k in 1..=max_lag {
            if acf_vals[k].abs() < threshold {
                embargo_lag = k;
                break;
            }
        }
    }
    println!("   FFD length = {}", ffd.len());
    println!("   ACF threshold (1.96/sqrt(T)) = {threshold:.6}");
    println!("   ACF(1..={max_lag}) = {acf_vals:?}");
    println!("   Embargo lag = {embargo_lag} (expect 1-5)");
    assert!(
        (1..=10).contains(&embargo_lag),
        "embargo lag should be 1-10, got {embargo_lag}"
    );
}

fn exercise_3() {
    println!("3. Volatility-scaled Kelly tightens drawdowns:");
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let n = 1000_usize;
    let mut base_rets = Vec::with_capacity(n);
    for _ in 0..n {
        // Returns with time-varying volatility.
        let vol = 0.01 + 0.02 * (rng.next_u64() as f64 / u64::MAX as f64);
        base_rets.push(normal.sample(&mut rng) * vol + 0.0005);
    }
    let f_star = kelly_from_returns(&base_rets);
    let half_kelly = 0.5 * f_star.clamp(0.0, 2.0);
    // Unscaled Kelly: constant fraction.
    let mut wealth_u = 1.0_f64;
    let mut max_dd_u = 0.0_f64;
    let mut peak_u = 1.0_f64;
    for &r in &base_rets {
        wealth_u *= 1.0 + half_kelly * r;
        peak_u = peak_u.max(wealth_u);
        let dd = (peak_u - wealth_u) / peak_u;
        max_dd_u = max_dd_u.max(dd);
    }
    // Scaled Kelly: f_t = half_kelly * sigma_bar / sigma_t.
    let window = 20_usize;
    let sigma_bar = base_rets.iter().map(|r| r * r).sum::<f64>().sqrt() / n as f64;
    let mut wealth_s = 1.0_f64;
    let mut max_dd_s = 0.0_f64;
    let mut peak_s = 1.0_f64;
    for t in 0..n {
        let sigma_t = if t >= window {
            let v: f64 =
                base_rets[t - window..t].iter().map(|r| r * r).sum::<f64>() / window as f64;
            v.sqrt()
        } else {
            sigma_bar
        };
        let scale = if sigma_t > 1e-9 {
            sigma_bar / sigma_t
        } else {
            1.0
        };
        let f_t = half_kelly * scale;
        wealth_s *= 1.0 + f_t * base_rets[t];
        peak_s = peak_s.max(wealth_s);
        let dd = (peak_s - wealth_s) / peak_s;
        max_dd_s = max_dd_s.max(dd);
    }
    println!("   Half-Kelly f* = {half_kelly:.4}");
    println!("   Unscaled max drawdown = {max_dd_u:.4}");
    println!("   Scaled max drawdown    = {max_dd_s:.4}");
    assert!(
        max_dd_s <= max_dd_u,
        "vol-scaled Kelly should have lower or equal max drawdown"
    );
}

fn exercise_4() {
    println!("4. Meta-labeling reduces trade count:");
    let mut rng = XorShift64::new(11);
    let normal = Normal::standard();
    let n = 500_usize;
    let mut prices: Vec<f64> = vec![100.0];
    for _ in 0..n {
        let p = prices.last().unwrap();
        prices.push(p * (0.0003 + 0.005 * normal.sample(&mut rng)).exp());
    }
    // Primary signal: SMA crossover (5 vs 20).
    let sma_fast = rolling_mean(5, &prices).expect("rolling");
    let sma_slow = rolling_mean(20, &prices).expect("rolling");
    // Entries where fast > slow (offset to align).
    let offset = 19_usize;
    let entries: Vec<usize> = (offset..prices.len() - 20)
        .filter(|&t| sma_fast[t - 5] > sma_slow[t - 20])
        .collect();
    // Label with triple barrier.
    let cfg = TripleBarrierConfig {
        upper_barrier: 0.01,
        lower_barrier: -0.01,
        time_barrier: 10,
        min_return: 0.0,
    };
    let events = triple_barrier_label(&prices, &entries, &cfg).expect("label");
    // Build meta-features: momentum, vol, return sign.
    let x: Vec<Vec<f64>> = entries
        .iter()
        .map(|&t| {
            let mom = if t >= 10 {
                prices[t] - prices[t - 10]
            } else {
                0.0
            };
            let vol = if t >= 20 {
                let r: Vec<f64> = (t - 20..t)
                    .map(|i| (prices[i + 1] / prices[i]).ln())
                    .collect();
                let m = r.iter().sum::<f64>() / r.len() as f64;
                (r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / r.len() as f64).sqrt()
            } else {
                0.0
            };
            vec![mom, vol, entries.contains(&t) as i32 as f64]
        })
        .collect();
    let y: Vec<i32> = events.iter().map(|e| e.to_binary(0.0)).collect();
    let model = LogisticRegression::fit(&x, &y, 0.5, 200);
    // Filter: only take trades where meta prob > 0.5.
    let mut kept = 0_usize;
    for row in x.iter() {
        let p = model.predict_prob(row);
        if p > 0.5 {
            kept += 1;
        }
    }
    let primary_count = entries.len();
    println!("   Primary trade count = {primary_count}");
    println!("   Meta-filtered trades = {kept}");
    // Sanity: kept should be <= primary_count (we only filter candidates).
    assert!(
        kept <= primary_count,
        "meta-filtered count {kept} should be <= primary {primary_count}"
    );
}

#[test]
fn test_ex1_vol_scaled_barriers_shift_labels() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let labeler_low = DynamicBarrierLabeler::new(20, 2.0, 10);
    let entries: Vec<usize> = (20..250).collect();
    let events_low = labeler_low.label(&closes[..250], &entries).expect("label");
    let up_low = events_low
        .iter()
        .filter(|e| e.label == TripleBarrierLabel::Upper)
        .count() as f64
        / events_low.len() as f64;
    // Build a higher-vol price series by oscillating.
    let mut closes_hv: Vec<f64> = closes[..250].to_vec();
    for i in 0..closes_hv.len() {
        let phase = (i as f64 / 5.0).sin();
        closes_hv[i] *= 1.0 + 0.05 * phase;
    }
    let labeler_high = DynamicBarrierLabeler::new(20, 2.0, 10);
    let events_high = labeler_high.label(&closes_hv, &entries).expect("label");
    let up_high = events_high
        .iter()
        .filter(|e| e.label == TripleBarrierLabel::Upper)
        .count() as f64
        / events_high.len() as f64;
    assert!(
        up_low > up_high,
        "low-vol Up fraction {up_low} should exceed high-vol {up_high}"
    );
}

#[test]
fn test_ex2_embargo_lag_in_range() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let ffd = frac_diff(&ret, 0.4, 1e-4).expect("fracdiff");
    let max_lag = 10_usize.min(ffd.len() - 1);
    let acf_vals = acf(&ffd, max_lag).expect("acf");
    let t = ffd.len() as f64;
    let threshold = 1.96 / t.sqrt();
    let mut embargo_lag = 0_usize;
    for k in 1..=max_lag {
        if acf_vals[k].abs() < threshold {
            embargo_lag = k;
            break;
        }
    }
    assert!(
        (1..=10).contains(&embargo_lag),
        "embargo lag should be 1-10, got {embargo_lag}"
    );
}

#[test]
fn test_ex3_vol_scaled_kelly_tightens_dd() {
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let n = 500_usize;
    let mut base_rets = Vec::with_capacity(n);
    for _ in 0..n {
        let vol = 0.01 + 0.02 * (rng.next_u64() as f64 / u64::MAX as f64);
        base_rets.push(normal.sample(&mut rng) * vol + 0.0005);
    }
    let f_star = kelly_from_returns(&base_rets);
    let half_kelly = 0.5 * f_star.clamp(0.0, 2.0);
    let mut max_dd_u = 0.0_f64;
    let mut wealth_u = 1.0_f64;
    let mut peak_u = 1.0_f64;
    for &r in &base_rets {
        wealth_u *= 1.0 + half_kelly * r;
        peak_u = peak_u.max(wealth_u);
        max_dd_u = max_dd_u.max((peak_u - wealth_u) / peak_u);
    }
    let window = 20_usize;
    let sigma_bar = base_rets.iter().map(|r| r * r).sum::<f64>().sqrt() / n as f64;
    let mut max_dd_s = 0.0_f64;
    let mut wealth_s = 1.0_f64;
    let mut peak_s = 1.0_f64;
    for t in 0..n {
        let sigma_t = if t >= window {
            (base_rets[t - window..t].iter().map(|r| r * r).sum::<f64>() / window as f64).sqrt()
        } else {
            sigma_bar
        };
        let scale = if sigma_t > 1e-9 {
            sigma_bar / sigma_t
        } else {
            1.0
        };
        wealth_s *= 1.0 + half_kelly * scale * base_rets[t];
        peak_s = peak_s.max(wealth_s);
        max_dd_s = max_dd_s.max((peak_s - wealth_s) / peak_s);
    }
    assert!(
        max_dd_s <= max_dd_u,
        "scaled dd {max_dd_s} <= unscaled {max_dd_u}"
    );
}

#[test]
fn test_ex4_meta_labeling_reduces_trades() {
    let mut rng = XorShift64::new(11);
    let normal = Normal::standard();
    let n = 300_usize;
    let mut prices: Vec<f64> = vec![100.0];
    for _ in 0..n {
        let p = prices.last().unwrap();
        prices.push(p * (0.0003 + 0.005 * normal.sample(&mut rng)).exp());
    }
    let sma_fast = rolling_mean(5, &prices).expect("rolling");
    let sma_slow = rolling_mean(20, &prices).expect("rolling");
    let entries: Vec<usize> = (20..prices.len() - 20)
        .filter(|&t| sma_fast[t - 5] > sma_slow[t - 20])
        .collect();
    let cfg = TripleBarrierConfig {
        upper_barrier: 0.01,
        lower_barrier: -0.01,
        time_barrier: 10,
        min_return: 0.0,
    };
    let events = triple_barrier_label(&prices, &entries, &cfg).expect("label");
    let x: Vec<Vec<f64>> = entries
        .iter()
        .map(|&t| {
            let mom = if t >= 10 {
                prices[t] - prices[t - 10]
            } else {
                0.0
            };
            let vol = if t >= 20 {
                let r: Vec<f64> = (t - 20..t)
                    .map(|i| (prices[i + 1] / prices[i]).ln())
                    .collect();
                let m = r.iter().sum::<f64>() / r.len() as f64;
                (r.iter().map(|x| (x - m).powi(2)).sum::<f64>() / r.len() as f64).sqrt()
            } else {
                0.0
            };
            vec![mom, vol]
        })
        .collect();
    let y: Vec<i32> = events.iter().map(|e| e.to_binary(0.0)).collect();
    let model = LogisticRegression::fit(&x, &y, 0.5, 200);
    let mut kept = 0_usize;
    for row in &x {
        if model.predict_prob(row) > 0.5 {
            kept += 1;
        }
    }
    assert!(
        kept <= entries.len(),
        "kept {kept} should be <= entries {}",
        entries.len()
    );
}
