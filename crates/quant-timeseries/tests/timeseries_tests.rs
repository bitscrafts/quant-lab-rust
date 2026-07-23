//! Integration tests for the quant-timeseries crate (TDD contract: 17 tests).

use approx::assert_relative_eq;
use quant_core::{gbm_paths, Distribution, Normal, XorShift64};
use quant_timeseries::{
    acf, adf_test, find_min_d, frac_diff, ffd_weights, ols, MACKINNON_5PCT, TimeSeriesError,
};

// R7.2: OLS regression --------------------------------------------------------

#[test]
fn test_ols_simple_regression() {
    // y = 3 + 2x for x = 0, 1, 2, 3, 4 -> y = 3, 5, 7, 9, 11 (perfect).
    // Add tiny noise to test approximate recovery; here we keep it perfect to
    // also validate exact recovery, and the contract says "coeffs ≈ [3, 2]".
    let x: Vec<Vec<f64>> = (0..5)
        .map(|t| vec![1.0, t as f64])
        .collect();
    let y: Vec<f64> = (0..5).map(|t| 3.0 + 2.0 * t as f64).collect();
    let fit = ols(&x, &y).unwrap();
    assert_eq!(fit.coeffs.len(), 2);
    assert_relative_eq!(fit.coeffs[0], 3.0, epsilon = 1e-9);
    assert_relative_eq!(fit.coeffs[1], 2.0, epsilon = 1e-9);
}

#[test]
fn test_ols_perfect_fit() {
    // y = 2x + 1 exactly -> R^2 = 1.0.
    let x: Vec<Vec<f64>> = (0..6).map(|t| vec![1.0, t as f64]).collect();
    let y: Vec<f64> = (0..6).map(|t| 1.0 + 2.0 * t as f64).collect();
    let fit = ols(&x, &y).unwrap();
    assert_relative_eq!(fit.r_squared, 1.0, epsilon = 1e-12);
    assert_relative_eq!(fit.coeffs[0], 1.0, epsilon = 1e-9);
    assert_relative_eq!(fit.coeffs[1], 2.0, epsilon = 1e-9);
}

#[test]
fn test_ols_multiple_regression() {
    // y = 1 + 2*x1 + 3*x2 exactly -> 3 coefficients recovered.
    let x: Vec<Vec<f64>> = (0..6)
        .map(|t| vec![1.0, t as f64, (t * t) as f64])
        .collect();
    let y: Vec<f64> = (0..6)
        .map(|t| 1.0 + 2.0 * t as f64 + 3.0 * (t * t) as f64)
        .collect();
    let fit = ols(&x, &y).unwrap();
    assert_eq!(fit.coeffs.len(), 3);
    assert_relative_eq!(fit.coeffs[0], 1.0, epsilon = 1e-9);
    assert_relative_eq!(fit.coeffs[1], 2.0, epsilon = 1e-9);
    assert_relative_eq!(fit.coeffs[2], 3.0, epsilon = 1e-9);
}

#[test]
fn test_ols_singular_matrix() {
    // Two identical columns -> X'X is singular.
    let x: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 2.0], vec![1.0, 3.0, 3.0], vec![1.0, 4.0, 4.0]];
    let y: Vec<f64> = vec![1.0, 2.0, 3.0];
    let err = ols(&x, &y).unwrap_err();
    assert!(matches!(err, TimeSeriesError::Singular), "got {err:?}");
}

#[test]
fn test_ols_residuals() {
    // For a fit with an intercept, the residuals sum to ~0.
    let x: Vec<Vec<f64>> = (0..10).map(|t| vec![1.0, t as f64]).collect();
    let y: Vec<f64> = (0..10)
        .map(|t| 1.0 + 0.5 * t as f64 + ((t as f64).sin() * 0.1))
        .collect();
    let fit = ols(&x, &y).unwrap();
    let sum_res: f64 = fit.residuals.iter().sum();
    assert!(sum_res.abs() < 1e-9, "residuals sum {sum_res} should be ~0");
}

// R7.3: ACF -------------------------------------------------------------------

#[test]
fn test_acf_lag_zero() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = acf(&data, 3).unwrap();
    assert_relative_eq!(r[0], 1.0, epsilon = 1e-12);
    assert_eq!(r.len(), 4);
}

#[test]
fn test_acf_white_noise() {
    // 1000 standard normal draws: ACF(k>0) near 0.
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let data: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
    let r = acf(&data, 5).unwrap();
    for (k, val) in r.iter().enumerate().take(6).skip(1) {
        assert!(
            val.abs() < 0.1,
            "ACF({k}) = {val} should be within 0.1 of 0 for white noise"
        );
    }
}

#[test]
fn test_acf_ar1() {
    // AR(1) with phi = 0.8: ACF(1) should be near 0.8.
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let n = 2000;
    let mut series = Vec::with_capacity(n);
    series.push(0.0_f64);
    for _ in 1..n {
        let prev = *series.last().unwrap();
        let z = normal.sample(&mut rng);
        series.push(0.8 * prev + z);
    }
    let r = acf(&series, 3).unwrap();
    assert!(
        (r[1] - 0.8).abs() < 0.1,
        "ACF(1) = {} should be near 0.8 for AR(1) phi=0.8",
        r[1]
    );
}

// R7.4: ADF -------------------------------------------------------------------

#[test]
fn test_adf_stationary() {
    // White noise is stationary: ADF rejects unit root.
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let data: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();
    let r = adf_test(&data, 1).unwrap();
    assert!(r.is_stationary, "white noise should be stationary");
    assert_relative_eq!(r.critical_value, MACKINNON_5PCT, epsilon = 1e-12);
}

#[test]
fn test_adf_random_walk() {
    // Cumsum of white noise is a random walk: ADF fails to reject unit root.
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let mut data = Vec::with_capacity(500);
    let mut s = 100.0_f64;
    data.push(s);
    for _ in 0..500 {
        s += normal.sample(&mut rng);
        data.push(s);
    }
    let r = adf_test(&data, 1).unwrap();
    assert!(
        !r.is_stationary,
        "random walk should NOT be stationary (stat={}, crit={})",
        r.statistic,
        r.critical_value
    );
}

#[test]
fn test_adf_critical_value() {
    // Any data that produces a valid fit: the critical value is a constant.
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let data: Vec<f64> = (0..200).map(|_| normal.sample(&mut rng)).collect();
    let r = adf_test(&data, 1).unwrap();
    assert_relative_eq!(r.critical_value, -2.86, epsilon = 1e-12);
}

// R7.5: Fractional differentiation --------------------------------------------

#[test]
fn test_ffd_weights_d_zero() {
    // d = 0: the only weight is [1.0] (identity).
    let w = ffd_weights(0.0, 1e-4).unwrap();
    assert_eq!(w.len(), 1);
    assert_relative_eq!(w[0], 1.0, epsilon = 1e-12);
}

#[test]
fn test_ffd_weights_d_one() {
    // d = 1: weights are [1.0, -1.0, ...] with a tiny tail below threshold.
    let w = ffd_weights(1.0, 1e-4).unwrap();
    assert_relative_eq!(w[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(w[1], -1.0, epsilon = 1e-12);
    // For d = 1 the recurrence gives w_k = 0 for k >= 2 (binomial of (1-B)^1),
    // so the threshold cutoff should land at k = 2 with w_2 = 0 < threshold.
    assert!(w.len() >= 2);
}

#[test]
fn test_ffd_weights_decay() {
    // For d = 0.5 the weight magnitudes strictly decrease.
    let w = ffd_weights(0.5, 1e-5).unwrap();
    for k in 1..w.len() {
        assert!(
            w[k].abs() <= w[k - 1].abs() + 1e-12,
            "weight magnitude should not increase: |w[{k}]|={} > |w[{}]|={}",
            w[k].abs(),
            k - 1,
            w[k - 1].abs()
        );
    }
}

#[test]
fn test_frac_diff_length() {
    // 500 prices, d = 0.4: output shorter by window - 1.
    let data: Vec<f64> = (0..500).map(|t| 100.0 + t as f64).collect();
    let weights = ffd_weights(0.4, 1e-4).unwrap();
    let diff = frac_diff(&data, 0.4, 1e-4).unwrap();
    assert_eq!(diff.len(), data.len() - weights.len() + 1);
}

#[test]
fn test_frac_diff_d_zero() {
    // d = 0: weights = [1.0], output equals input.
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let diff = frac_diff(&data, 0.0, 1e-4).unwrap();
    assert_eq!(diff.len(), data.len());
    for (a, b) in diff.iter().zip(data.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-12);
    }
}

#[test]
fn test_find_min_d() {
    // Random walk generated from cumsum of normals: minimum d for stationarity
    // should be in (0, 1].
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let mut data = Vec::with_capacity(500);
    let mut s = 100.0_f64;
    data.push(s);
    for _ in 0..500 {
        s += normal.sample(&mut rng);
        data.push(s);
    }
    let d = find_min_d(&data, 1e-4, 0.01).unwrap();
    assert!(d > 0.0 && d <= 1.0, "min d = {d} should be in (0, 1]");
}

// Extra: GBM-generated prices are also non-stationary and need differencing.

#[test]
fn test_gbm_prices_need_differencing() {
    // GBM terminal prices along one path are non-stationary; find_min_d returns
    // a positive d. This ties Phase 7 back to Phase 6's simulator. Use a long
    // enough path so the fixed-width window fits even for small d candidates.
    let mut rng = XorShift64::new(7);
    let paths = gbm_paths(100.0, 0.05, 0.2, 1.0, 1000, 1, &mut rng);
    let prices = &paths[0];
    let d = find_min_d(prices, 1e-4, 0.01).unwrap();
    assert!(d > 0.0, "GBM path should need d > 0 to become stationary");
}