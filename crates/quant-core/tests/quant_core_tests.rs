//! Integration tests for the quant-core crate (TDD contract: 16 tests).

use approx::assert_relative_eq;
use quant_core::{
    box_muller_normal, excess_kurtosis, gbm_paths, log_returns, mean, rolling, rolling_mean,
    skewness, simple_returns, std_dev, variance, CoreError, Distribution, Moments, Normal,
    PriceSeries, Rng, RollingWindow, XorShift64,
};

// R6.2: PriceSeries newtype ----------------------------------------------------

#[test]
fn test_price_series_valid() {
    let ps = PriceSeries::new(vec![100.0, 101.0, 99.5]).unwrap();
    assert_eq!(ps.len(), 3);
    assert!(!ps.is_empty());
    assert_eq!(ps.as_slice(), &[100.0, 101.0, 99.5]);
}

#[test]
fn test_price_series_rejects_nonpositive() {
    let err = PriceSeries::new(vec![100.0, 0.0]).unwrap_err();
    assert_eq!(err, CoreError::InvalidPrice);
    // Negative price.
    let err = PriceSeries::new(vec![100.0, -5.0]).unwrap_err();
    assert_eq!(err, CoreError::InvalidPrice);
}

#[test]
fn test_price_series_rejects_nan() {
    let err = PriceSeries::new(vec![100.0, f64::NAN]).unwrap_err();
    assert_eq!(err, CoreError::InvalidPrice);
    // Infinity.
    let err = PriceSeries::new(vec![100.0, f64::INFINITY]).unwrap_err();
    assert_eq!(err, CoreError::InvalidPrice);
}

// R6.3: Returns ----------------------------------------------------------------

#[test]
fn test_simple_returns_basic() {
    let r = simple_returns(&[100.0, 110.0, 99.0]);
    assert_eq!(r.len(), 2);
    assert_relative_eq!(r[0], 0.10, epsilon = 1e-12);
    assert_relative_eq!(r[1], -0.10, epsilon = 1e-12);
}

#[test]
fn test_log_returns_basic() {
    let r = log_returns(&[100.0, 110.0]);
    assert_eq!(r.len(), 1);
    assert_relative_eq!(r[0], (1.1_f64).ln(), epsilon = 1e-12);
}

#[test]
fn test_returns_single_element() {
    assert!(simple_returns(&[100.0]).is_empty());
    assert!(log_returns(&[100.0]).is_empty());
}

#[test]
fn test_log_simple_consistency() {
    // For small returns, log ≈ ln(1 + simple). Use a GBM with small steps.
    let mut rng = XorShift64::new(42);
    let paths = gbm_paths(100.0, 0.05, 0.2, 1.0, 500, 1, &mut rng);
    let prices = &paths[0];
    let simple = simple_returns(prices);
    let logr = log_returns(prices);
    // Identity: log_t = ln(1 + simple_t)
    for (s, l) in simple.iter().zip(logr.iter()) {
        assert_relative_eq!(*l, (1.0 + s).ln(), epsilon = 1e-9);
    }
}

// R6.4: Moments ----------------------------------------------------------------

#[test]
fn test_mean_variance_known() {
    let data = [2.0, 4.0, 6.0];
    assert_relative_eq!(mean(&data), 4.0, epsilon = 1e-12);
    // Sample variance (n-1 denominator): ((2-4)^2 + 0 + (6-4)^2)/2 = 8/2 = 4.
    assert_relative_eq!(variance(&data).unwrap(), 4.0, epsilon = 1e-12);
    assert_relative_eq!(std_dev(&data).unwrap(), 2.0, epsilon = 1e-12);
}

#[test]
fn test_moments_insufficient_data() {
    // Variance of a single point is undefined (n-1 = 0).
    let err = variance(&[1.0]).unwrap_err();
    assert!(matches!(err, CoreError::InsufficientData { .. }));
    // Skewness needs at least 3.
    let err = skewness(&[1.0, 2.0]).unwrap_err();
    assert!(matches!(err, CoreError::InsufficientData { .. }));
    // Kurtosis needs at least 4.
    let err = excess_kurtosis(&[1.0, 2.0, 3.0]).unwrap_err();
    assert!(matches!(err, CoreError::InsufficientData { .. }));
}

#[test]
fn test_gaussian_moments() {
    // 10k standard normal draws: skewness near 0, excess kurtosis near 0.
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let draws: Vec<f64> = (0..10_000).map(|_| normal.sample(&mut rng)).collect();
    let sk = skewness(&draws).unwrap();
    let ku = excess_kurtosis(&draws).unwrap();
    assert!(
        sk.abs() < 0.1,
        "skewness {sk} should be within 0.1 of 0"
    );
    assert!(
        ku.abs() < 0.2,
        "excess kurtosis {ku} should be within 0.2 of 0"
    );
}

#[test]
fn test_fat_tails_detected() {
    // Every 100th draw multiplied by 10 → heavy tails → excess kurtosis > 1.
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let mut draws: Vec<f64> = (0..10_000).map(|_| normal.sample(&mut rng)).collect();
    for i in (0..draws.len()).step_by(100) {
        draws[i] *= 10.0;
    }
    let ku = excess_kurtosis(&draws).unwrap();
    assert!(
        ku > 1.0,
        "excess kurtosis {ku} should be > 1.0 for fat tails"
    );
}

// R6.5: Rolling windows --------------------------------------------------------

#[test]
fn test_rolling_mean() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let r = rolling_mean(2, &data).unwrap();
    assert_eq!(r.len(), 4);
    assert_relative_eq!(r[0], 1.5, epsilon = 1e-12);
    assert_relative_eq!(r[1], 2.5, epsilon = 1e-12);
    assert_relative_eq!(r[2], 3.5, epsilon = 1e-12);
    assert_relative_eq!(r[3], 4.5, epsilon = 1e-12);
}

#[test]
fn test_rolling_invalid_window() {
    let data = [1.0, 2.0, 3.0];
    // window = 0
    let err = rolling(0, &data, |w| w.len()).unwrap_err();
    assert!(matches!(err, CoreError::InvalidWindow { .. }));
    // window > len
    let err = rolling(4, &data, |w| w.len()).unwrap_err();
    assert!(matches!(err, CoreError::InvalidWindow { .. }));
    // rolling_mean also rejects invalid windows.
    let err = rolling_mean(0, &data).unwrap_err();
    assert!(matches!(err, CoreError::InvalidWindow { .. }));
}

// R6.6: Simulation -------------------------------------------------------------

#[test]
fn test_rng_deterministic() {
    let mut a = XorShift64::new(123);
    let mut b = XorShift64::new(123);
    let sa: Vec<u64> = (0..100).map(|_| a.next_u64()).collect();
    let sb: Vec<u64> = (0..100).map(|_| b.next_u64()).collect();
    assert_eq!(sa, sb, "same seed must produce identical sequences");
}

#[test]
fn test_gbm_reproducible() {
    let mut a = XorShift64::new(42);
    let mut b = XorShift64::new(42);
    let pa = gbm_paths(100.0, 0.05, 0.2, 1.0, 252, 1, &mut a);
    let pb = gbm_paths(100.0, 0.05, 0.2, 1.0, 252, 1, &mut b);
    assert_eq!(pa, pb, "GBM with same seed must produce identical paths");
}

#[test]
fn test_gbm_zero_vol() {
    // With sigma = 0, the path is deterministic: S_T = s0 * exp(mu * T).
    let mut rng = XorShift64::new(42);
    let paths = gbm_paths(100.0, 0.05, 0.0, 1.0, 252, 1, &mut rng);
    let final_price = paths[0][paths[0].len() - 1];
    let expected = 100.0 * (0.05_f64).exp();
    assert_relative_eq!(final_price, expected, epsilon = 1e-6);
}

// Extra: trait usage sanity (not in contract but covers trait impls) -----------

#[test]
fn test_traits_smoke() {
    // Moments on a PriceSeries.
    let ps = PriceSeries::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert_relative_eq!(ps.mean(), 3.0, epsilon = 1e-12);
    assert_relative_eq!(ps.variance().unwrap(), 2.5, epsilon = 1e-12);

    // RollingWindow on a slice.
    let s: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];
    let rm = s.rolling_mean(3).unwrap();
    assert_eq!(rm.len(), 3);
    assert_relative_eq!(rm[0], 2.0, epsilon = 1e-12);

    // Distribution and free function produce a finite value.
    let mut rng = XorShift64::new(1);
    let n = Normal::new(5.0, 2.0);
    let v = n.sample(&mut rng);
    assert!(v.is_finite());
    let v2 = box_muller_normal(&mut rng, 0.0, 1.0);
    assert!(v2.is_finite());
}