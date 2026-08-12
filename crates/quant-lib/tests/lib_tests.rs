//! Integration tests for quant-lib (Phase 15).
//!
//! 20-test TDD contract verifying the non-destructive facade
//! principle: every re-export matches its source crate, all Phase
//! 14.5 traits are usable through the facade, the generic backtest
//! composes across crates, the full pipeline produces finite
//! results, feature flags gate modules, and source crates remain
//! independently usable.

#![allow(clippy::useless_vec)]

// Source crates are accessible because quant-lib depends on them
// (they are regular path dependencies, not optional). The facade
// re-exports them under unified namespaces. These tests verify
// identity between the facade path and the source-crate path.

use quant_lib::prelude::*;

// =========================================================================
// t01-t08: Module re-exports match source crates
// =========================================================================

#[test]
fn t01_prelude_imports() {
    // If the prelude compiles, all the named items are available.
    let r: Vec<f64> = vec![0.0, 0.01, 0.0098, -0.0049, 0.0148];
    let m = mean(&r);
    let s = std_dev(&r).unwrap();
    assert!(m.is_finite());
    assert!(s.is_finite());
}

#[test]
fn t02_core_reexport() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let m_facade = quant_lib::core::mean(&data);
    let m_source = quant_core::mean(&data);
    assert!((m_facade - m_source).abs() < 1e-12);
    // XorShift64 is reachable through the facade.
    let mut rng = quant_lib::core::XorShift64::new(42);
    let u = rng.next_f64();
    assert!((0.0..1.0).contains(&u));
}

#[test]
fn t03_stochastic_reexport() {
    let bs = quant_lib::stochastic::bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    let bs_src = quant_stochastic::bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    assert!((bs - bs_src).abs() < 1e-12);
    assert!(bs > 10.0);
}

#[test]
fn t04_options_reexport() {
    let d = quant_lib::options::delta(100.0, 100.0, 0.05, 0.2, 1.0, true);
    let d_src = quant_options::delta(100.0, 100.0, 0.05, 0.2, 1.0, true);
    assert!((d - d_src).abs() < 1e-12);
    assert!(d > 0.5 && d < 0.7);
}

#[test]
fn t05_portfolio_reexport() {
    let mu = vec![0.08, 0.12];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.04]];
    let w_facade = quant_lib::portfolio::min_variance_portfolio(&mu, &cov).unwrap();
    let w_source = quant_portfolio::min_variance_portfolio(&mu, &cov).unwrap();
    assert_eq!(w_facade.len(), w_source.len());
    for (a, b) in w_facade.iter().zip(w_source.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    // Weights sum to 1.
    let sum: f64 = w_facade.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

#[test]
fn t06_factors_reexport() {
    let returns = vec![
        vec![0.01, 0.02],
        vec![0.02, 0.03],
        vec![-0.01, 0.0],
        vec![0.03, 0.04],
        vec![0.0, 0.01],
    ];
    let res_facade = quant_lib::factors::pca(&returns, 2).unwrap();
    let res_source = quant_factors::pca(&returns, 2).unwrap();
    assert_eq!(res_facade.eigenvalues.len(), res_source.eigenvalues.len());
    for (a, b) in res_facade
        .eigenvalues
        .iter()
        .zip(res_source.eigenvalues.iter())
    {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn t07_microstructure_reexport() {
    use quant_lib::microstructure::{Order, OrderBook, Side};
    let mut book_facade = OrderBook::new(1);
    book_facade
        .add_order(Order {
            id: 1,
            side: Side::Bid,
            price: 100,
            quantity: 10,
            timestamp: 1,
        })
        .unwrap();
    let best = book_facade.best_bid().unwrap();
    assert_eq!(best.price, 100);
    assert_eq!(best.quantity, 10);
    // Same types come from the source crate.
    let _book_src = quant_microstructure::OrderBook::new(1);
}

#[test]
fn t08_backtest_reexport() {
    let f_facade = quant_lib::backtest::kelly_fraction(0.6, 1.0);
    let f_source = quant_backtest::kelly_fraction(0.6, 1.0);
    assert!((f_facade - f_source).abs() < 1e-12);
    assert!((f_facade - 0.2).abs() < 1e-9);
}

// =========================================================================
// t09-t13: Phase 14.5 traits are usable through the facade
// =========================================================================

#[test]
fn t09_traits_cross_validator() {
    use quant_lib::traits::CrossValidator;
    let wf = WalkForward::new(WalkForwardConfig::rolling(60, 20, 20));
    let splits = wf.splits(100, 200);
    assert!(!splits.is_empty());
    for split in &splits {
        assert!(!split.train_indices.is_empty());
        assert!(!split.test_indices.is_empty());
    }
}

#[test]
fn t10_traits_labeler() {
    use quant_lib::traits::Labeler;
    let labeler = FixedHorizonLabeler::new(5, 0.01);
    let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.01)).collect();
    let entries: Vec<usize> = (0..50).step_by(10).collect();
    let events = labeler.label(&prices, &entries).unwrap();
    assert_eq!(events.len(), 5);
}

#[test]
fn t11_traits_bet_sizer() {
    use quant_lib::traits::BetSizer;
    let sizer = KellyBetSizer::new(0.5);
    let returns = vec![
        0.02_f64, -0.01, 0.03, -0.02, 0.04, -0.01, 0.02, 0.03, -0.02, 0.05,
    ];
    let size = sizer.size(&returns);
    assert!((0.0..=1.0).contains(&size));
}

#[test]
fn t12_traits_stochastic_process() {
    use quant_lib::traits::StochasticProcess;
    let mut rng = quant_lib::core::XorShift64::new(7);
    let mut gbm = quant_stochastic::Gbm::new(0.05, 0.20, &mut rng);
    let path = gbm.simulate(100.0, 1.0, 252).unwrap();
    assert_eq!(path.len(), 253);
    assert!((path[0] - 100.0).abs() < 1e-12);
}

#[test]
fn t13_traits_option_pricer() {
    use quant_lib::traits::{Greeks, OptionPricer, OptionType};
    let pricer = quant_options::BlackScholes::new(0.05, 0.20);
    let call = pricer.price(100.0, 100.0, 1.0, OptionType::Call).unwrap();
    assert!(call > 10.0);
    let d = pricer.delta(100.0, 100.0, 1.0, OptionType::Call).unwrap();
    assert!(d > 0.5 && d < 0.7);
}

// =========================================================================
// t14-t16: Risk metrics and composable backtest
// =========================================================================

#[test]
fn t14_risk_metrics_reexport() {
    // deflated_sharpe_ratio signature: (sharpe, n, skew, kurtosis, n_trials, var_sharpes)
    let dsr = quant_lib::risk::deflated_sharpe_ratio(1.5, 252.0, 0.0, 3.0, 10, 0.25);
    let dsr_src = quant_core::deflated_sharpe_ratio(1.5, 252.0, 0.0, 3.0, 10, 0.25);
    assert!((dsr - dsr_src).abs() < 1e-12);
    assert!(dsr < 1.5);
}

#[test]
fn t15_generic_backtest() {
    let labeler = FixedHorizonLabeler::new(5, 0.01);
    let cv = WalkForward::new(WalkForwardConfig::rolling(40, 10, 10));
    let sizer = KellyBetSizer::new(0.5);
    let bt = GenericBacktest::new(labeler, cv, sizer, 5);
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01)).collect();
    // Build returns of equal length to prices (returns[0] = 0.0).
    let mut returns = vec![0.0_f64; prices.len()];
    for i in 1..prices.len() {
        returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
    }
    let results = bt.run(&prices, &returns);
    assert!(results.is_ok(), "backtest failed: {:?}", results.err());
    let results = results.unwrap();
    assert!(!results.is_empty());
    for r in &results {
        assert!(r.total_return.is_finite());
        assert!(r.sharpe.is_finite());
    }
}

#[test]
fn t16_walk_forward() {
    let wf_rolling = WalkForward::new(WalkForwardConfig::rolling(60, 20, 20));
    let splits_rolling = wf_rolling.splits(0, 200);
    assert!(!splits_rolling.is_empty());

    let wf_anchored = WalkForward::new(WalkForwardConfig::anchored(60, 20, 20));
    let splits_anchored = wf_anchored.splits(0, 200);
    assert!(!splits_anchored.is_empty());

    // WFE: 1.2 in-sample, 0.6 out-of-sample -> 0.5
    let wfe = walk_forward_efficiency(1.2, 0.6);
    assert!((wfe - 0.5).abs() < 1e-9);
}

// =========================================================================
// t17: Full pipeline (GBM -> BS -> Greeks -> Kelly -> WalkForward)
// =========================================================================

#[test]
fn t17_full_pipeline() {
    use quant_lib::traits::{Greeks, OptionPricer, OptionType};
    // 1. Simulate GBM path.
    let mut rng = quant_lib::core::XorShift64::new(123);
    let prices = quant_lib::stochastic::gbm(100.0, 0.05, 0.20, 1.0, 252, &mut rng);
    assert_eq!(prices.len(), 253);
    // 2. Price an ATM call with Black-Scholes.
    let bs = quant_options::BlackScholes::new(0.05, 0.20);
    let call = bs.price(100.0, 100.0, 1.0, OptionType::Call).unwrap();
    assert!(call > 10.0);
    // 3. Compute delta via the Greeks trait.
    let d = bs.delta(100.0, 100.0, 1.0, OptionType::Call).unwrap();
    assert!(d.is_finite());
    // 4. Kelly fraction from synthetic win/loss stats.
    let f = kelly_fraction(0.55, 1.5);
    assert!(f > 0.0);
    // 5. Walk-forward on the simulated prices.
    let wf = WalkForward::new(WalkForwardConfig::rolling(120, 30, 30));
    let splits = wf.splits(0, 252);
    assert!(!splits.is_empty());
    // All results finite.
    assert!(call.is_finite());
    assert!(d.is_finite());
    assert!(f.is_finite());
}

// =========================================================================
// t18: Feature flags
// =========================================================================

#[test]
#[allow(clippy::assertions_on_constants)]
fn t18_feature_flags() {
    // With default features, "all" is enabled and all modules are
    // visible. Verify the feature is reported correctly.
    assert!(cfg!(feature = "all"));
    assert!(cfg!(feature = "core"));
    assert!(cfg!(feature = "timeseries"));
    assert!(cfg!(feature = "vol"));
    assert!(cfg!(feature = "stochastic"));
    assert!(cfg!(feature = "options"));
    assert!(cfg!(feature = "portfolio"));
    assert!(cfg!(feature = "factors"));
    assert!(cfg!(feature = "microstructure"));
    assert!(cfg!(feature = "backtest"));
}

// =========================================================================
// t19: Source crates work independently (non-destructive)
// =========================================================================

#[test]
fn t19_non_destructive_source_crates() {
    // The source crates are directly importable from this test file,
    // proving that quant-lib is a pure facade and does not modify
    // them. Each call below exercises a public API of a source crate
    // without going through the facade.
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    assert!(quant_core::mean(&data).is_finite());

    let x = vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, 2.0]];
    let y = vec![1.0, 3.0, 5.0];
    let fit = quant_timeseries::ols(&x, &y).unwrap();
    assert!((fit.r_squared - 1.0).abs() < 1e-9);

    let returns = vec![0.01_f64, -0.02, 0.015, -0.01, 0.02, -0.005, 0.01, -0.03];
    assert!(quant_vol::ewma_vol(&returns, 0.94).is_ok());

    let bs = quant_stochastic::bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    assert!(bs > 10.0);

    let d = quant_options::delta(100.0, 100.0, 0.05, 0.2, 1.0, true);
    assert!(d > 0.5);

    let mu = vec![0.08, 0.12];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.04]];
    assert!(quant_portfolio::min_variance_portfolio(&mu, &cov).is_ok());

    let rets = vec![vec![0.01, 0.02], vec![0.02, 0.03], vec![-0.01, 0.0]];
    assert!(quant_factors::pca(&rets, 2).is_ok());

    let mut book = quant_microstructure::OrderBook::new(1);
    book.add_order(quant_microstructure::Order {
        id: 1,
        side: quant_microstructure::Side::Bid,
        price: 100,
        quantity: 10,
        timestamp: 1,
    })
    .unwrap();
    assert_eq!(book.best_bid().unwrap().price, 100);

    let f = quant_backtest::kelly_fraction(0.6, 1.0);
    assert!((f - 0.2).abs() < 1e-9);
}

// =========================================================================
// t20: Identity re-export (same function pointer)
// =========================================================================

#[test]
fn t20_identity_reexport() {
    // The facade re-export must be the *same* function, not a wrapper.
    // We compare function pointers to verify identity.
    let facade_fn: fn(f64, f64) -> f64 = quant_lib::backtest::kelly_fraction;
    let source_fn: fn(f64, f64) -> f64 = quant_backtest::kelly_fraction;
    assert_eq!(
        facade_fn as usize, source_fn as usize,
        "kelly_fraction re-export must be identical to source (same function pointer)"
    );

    // Also check mean from quant-core via the core module.
    let facade_mean: fn(&[f64]) -> f64 = quant_lib::core::mean;
    let source_mean: fn(&[f64]) -> f64 = quant_core::mean;
    assert_eq!(
        facade_mean as usize, source_mean as usize,
        "mean re-export must be identical to source"
    );

    // And bs_call from quant-stochastic.
    let facade_bs: fn(f64, f64, f64, f64, f64) -> f64 = quant_lib::stochastic::bs_call;
    let source_bs: fn(f64, f64, f64, f64, f64) -> f64 = quant_stochastic::bs_call;
    assert_eq!(
        facade_bs as usize, source_bs as usize,
        "bs_call re-export must be identical to source"
    );
}
