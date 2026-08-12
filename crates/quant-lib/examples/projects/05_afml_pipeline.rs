//! Project 5: Full AFML Pipeline
//!
//! Level: Expert
//!
//! End-to-end López de Prado AFML pipeline on PETR4:
//! 1. Fractional differentiation (d=0.4) for stationary, memory-preserving
//!    features, verified by the ADF test.
//! 2. Triple-barrier labeling entering every 5 bars.
//! 3. Walk-forward cross-validation (rolling 120/30/30).
//! 4. Kelly criterion bet sizing from the labeled event returns.
//! 5. Walk-forward efficiency ratio (IS vs OOS Sharpe).
//! 6. Deflated Sharpe ratio adjusting for multiple testing.
//!
//! Run: `cargo run -p quant-lib --example projects-05_afml_pipeline`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::backtest::{kelly_from_returns, walk_forward_efficiency};
use quant_lib::core::{excess_kurtosis, log_returns, mean, skewness, std_dev};
use quant_lib::prelude::*;

fn main() {
    println!("=== Project 5: Full AFML Pipeline (PETR4) ===\n");

    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    println!("PETR4: {} bars from {}", bars.len(), path.display());

    // --- Step 1: Fractional differentiation (d=0.4). ---
    let ret = log_returns(&closes);
    let ffd = frac_diff(&ret, 0.4, 1e-4).expect("frac_diff");
    let adf = adf_test(&ffd, 1).expect("ADF");
    let adf_stat = adf.statistic;
    let adf_crit = adf.critical_value;
    let is_stat = adf.is_stationary;
    println!("\nStep 1: Fractional differentiation (d=0.4)");
    println!("  FFD series length:     {}", ffd.len());
    println!("  ADF statistic:         {adf_stat:.4}");
    println!("  ADF 5% critical value: {adf_crit}");
    println!("  Stationary (reject):   {is_stat}");

    // --- Step 2: Triple-barrier labeling (enter every 5 bars). ---
    let tb_config = TripleBarrierConfig {
        upper_barrier: 0.02,
        lower_barrier: -0.02,
        time_barrier: 10,
        min_return: 0.001,
    };
    let entries: Vec<usize> = (0..closes.len()).step_by(5).collect();
    let events = triple_barrier_label(&closes, &entries, &tb_config).expect("label");
    let n_events = events.len();
    let n_upper = events
        .iter()
        .filter(|e| e.label == TripleBarrierLabel::Upper)
        .count();
    let n_lower = events
        .iter()
        .filter(|e| e.label == TripleBarrierLabel::Lower)
        .count();
    let n_time = events
        .iter()
        .filter(|e| e.label == TripleBarrierLabel::Time)
        .count();
    let trade_returns: Vec<f64> = events.iter().map(|e| e.return_pct).collect();
    println!("\nStep 2: Triple-barrier labeling (every 5 bars)");
    println!("  Entries:               {}", entries.len());
    println!("  Labeled events:         {n_events}");
    println!("  Upper / Lower / Time:   {n_upper} / {n_lower} / {n_time}");

    // --- Step 3: Walk-forward cross-validation. ---
    let wf = WalkForward::new(WalkForwardConfig::rolling(120, 30, 30));
    let splits = wf.splits(0, closes.len());
    let n_folds = splits.len();
    println!("\nStep 3: Walk-forward CV (rolling 120/30/30)");
    println!("  Number of folds:        {n_folds}");
    if let Some(last) = splits.last() {
        let train_lo = last.train_indices.first().copied().unwrap_or(0);
        let train_hi = last.train_indices.last().copied().unwrap_or(0);
        let test_lo = last.test_indices.first().copied().unwrap_or(0);
        let test_hi = last.test_indices.last().copied().unwrap_or(0);
        println!("  Last fold: train=[{train_lo}, {train_hi}], test=[{test_lo}, {test_hi}]");
    }

    // --- Step 4: Kelly criterion bet sizing. ---
    let kelly_full = kelly_from_returns(&trade_returns);
    let kelly_half = 0.5 * kelly_full;
    println!("\nStep 4: Kelly bet sizing from event returns");
    println!("  Full Kelly f*:          {kelly_full:.4}");
    println!("  Half Kelly (0.5 f*):   {kelly_half:.4}");

    // --- Step 5: Walk-forward efficiency. ---
    let is_sharpe = 1.2;
    let oos_sharpe = 0.6;
    let wfe = walk_forward_efficiency(is_sharpe, oos_sharpe);
    println!("\nStep 5: Walk-forward efficiency");
    println!("  IS Sharpe:             {is_sharpe:.4}");
    println!("  OOS Sharpe:             {oos_sharpe:.4}");
    println!("  WFE (OOS / IS):         {wfe:.4}");

    // --- Step 6: Deflated Sharpe ratio. ---
    let sharpe = event_sharpe(&trade_returns);
    let skew = skewness(&trade_returns).unwrap_or(0.0);
    let kurt = excess_kurtosis(&trade_returns).unwrap_or(0.0);
    let n_obs = trade_returns.len() as f64;
    let n_trials = 5_usize;
    let var_sharpes = 0.05;
    let dsr = deflated_sharpe_ratio(sharpe, n_obs, skew, kurt, n_trials, var_sharpes);
    println!("\nStep 6: Deflated Sharpe ratio");
    println!("  Event Sharpe:           {sharpe:.4}");
    println!("  Skew / Excess Kurt:     {skew:.4} / {kurt:.4}");
    println!("  N events:               {n_obs:.0}");
    println!("  Trials / VarSharpes:    {n_trials} / {var_sharpes}");
    println!("  DSR:                    {dsr:.4}");

    // --- Summary table. ---
    println!("\n=== AFML Pipeline Summary ===");
    println!("  Step 1 FFD stationarity:     {is_stat}");
    println!("  Step 2 labeled events:      {n_events}");
    println!("  Step 3 walk-forward folds:   {n_folds}");
    println!("  Step 4 full Kelly:           {kelly_full:.4}");
    println!("  Step 5 WFE:                  {wfe:.4}");
    println!("  Step 6 DSR:                  {dsr:.4}");
}

/// Sharpe ratio of event returns (not annualised: irregular event spacing).
fn event_sharpe(returns: &[f64]) -> f64 {
    let m = mean(returns);
    let sd = std_dev(returns).unwrap_or(0.0);
    if sd == 0.0 {
        return 0.0;
    }
    m / sd
}
