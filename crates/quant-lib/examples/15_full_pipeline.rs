//! Example 15: Full Pipeline - GBM -> BS -> Greeks -> Kelly -> WalkForward
//!
//! Level: Advanced
//!
//! End-to-end pipeline that ties together five crates:
//!
//! 1. `quant-stochastic`: simulate a GBM price path
//! 2. `quant-options`: price an ATM call with Black-Scholes (trait form)
//! 3. `quant-options`: compute analytical Greeks
//! 4. `quant-backtest`: derive a Kelly fraction from trade returns
//! 5. `quant-backtest`: run walk-forward cross-validation on the path
//!
//! This is the "hello world" of the composable quant-lib facade: every
//! step uses a different sub-crate, all reached through `quant_lib`.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 15_full_pipeline
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::backtest::{compute_position_size, kelly_from_returns};
use quant_lib::core::simple_returns;
use quant_lib::prelude::*;

fn main() {
    println!("=== Example 15: Full Pipeline (GBM -> BS -> Greeks -> Kelly -> WalkForward) ===");

    // --- Step 1: Simulate a GBM price path (252 trading days). ---
    let s0 = 100.0;
    let mu = 0.08;
    let sigma = 0.20;
    let t = 1.0;
    let n = 252;
    let mut rng = XorShift64::new(123);
    let prices = gbm(s0, mu, sigma, t, n, &mut rng);
    assert_eq!(prices.len(), n + 1);
    println!(
        "1. GBM: S0={s0}, mu={mu}, sigma={sigma}, S_T={:.4}",
        prices[n]
    );

    // --- Step 2: Price an ATM call with the trait-based BlackScholes pricer. ---
    let r = 0.05;
    let pricer = quant_options::BlackScholes::new(r, sigma);
    let call = pricer.price(s0, s0, t, OptionType::Call).unwrap();
    let put = pricer.price(s0, s0, t, OptionType::Put).unwrap();
    println!("2. BS call (ATM) = {call:.4}, BS put = {put:.4}");

    // --- Step 3: Analytical Greeks via the Greeks trait. ---
    let d = pricer.delta(s0, s0, t, OptionType::Call).unwrap();
    let g = pricer.gamma(s0, s0, t).unwrap();
    let v = pricer.vega(s0, s0, t).unwrap();
    let th = pricer.theta(s0, s0, t, OptionType::Call).unwrap();
    println!("3. Greeks: delta={d:.4}, gamma={g:.6}, vega={v:.4}, theta={th:.4}");

    // --- Step 4: Kelly fraction from synthetic trade returns. ---
    // Build simple returns from the simulated path.
    let rets = simple_returns(&prices);
    // Split into "win" / "loss" trades: positive vs negative daily returns.
    let f_kelly = kelly_from_returns(&rets);
    let ps = compute_position_size(&rets);
    println!(
        "4. Kelly from path returns: f*={f_kelly:.4}, win_prob={:.4}, b={:.4}",
        ps.win_probability, ps.win_loss_ratio
    );

    // --- Step 5: Walk-forward cross-validation on the simulated path. ---
    let wf_config = WalkForwardConfig::rolling(120, 30, 30);
    let wf = WalkForward::new(wf_config);
    let splits = wf.splits(0, prices.len());
    println!(
        "5. Walk-forward (rolling 120/30/30): {} splits",
        splits.len()
    );
    for (i, split) in splits.iter().enumerate() {
        println!(
            "   fold {i}: train=[{}, {}], test=[{}, {}]",
            split.train_indices.first().unwrap_or(&0),
            split.train_indices.last().unwrap_or(&0),
            split.test_indices.first().unwrap_or(&0),
            split.test_indices.last().unwrap_or(&0)
        );
    }

    // --- Step 6: Walk-forward efficiency (WFE) on in-sample vs OOS Sharpe. ---
    let is_sharpe = 1.2;
    let oos_sharpe = 0.6;
    let wfe = walk_forward_efficiency(is_sharpe, oos_sharpe);
    println!("6. WFE: IS Sharpe={is_sharpe}, OOS Sharpe={oos_sharpe} -> WFE={wfe:.4}");

    // --- Step 7: Run the generic composable backtest on the simulated path. ---
    let labeler = FixedHorizonLabeler::new(5, 0.01);
    let cv = WalkForward::new(WalkForwardConfig::rolling(100, 25, 25));
    let sizer = KellyBetSizer::new(0.5); // half Kelly
    let bt = GenericBacktest::new(labeler, cv, sizer, 5);
    // Build returns aligned with prices (returns[0] = 0).
    let mut bt_returns = vec![0.0_f64; prices.len()];
    for i in 1..prices.len() {
        bt_returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
    }
    let results = bt.run(&prices, &bt_returns).expect("backtest");
    println!("7. GenericBacktest: {} fold results", results.len());
    for (i, r) in results.iter().enumerate() {
        println!(
            "   fold {i}: total_return={:+.4}, sharpe={:+.4}",
            r.total_return, r.sharpe
        );
    }

    // All numbers finite sanity check.
    assert!(call.is_finite() && put.is_finite());
    assert!(d.is_finite() && g.is_finite() && v.is_finite() && th.is_finite());
    assert!(f_kelly.is_finite() && wfe.is_finite());
    for r in &results {
        assert!(r.total_return.is_finite() && r.sharpe.is_finite());
    }
    println!("\nAll pipeline outputs finite. Pipeline complete.");
}
