//! Integration tests for the quant-vol crate (TDD contract: 15 tests).

use approx::assert_relative_eq;
use quant_core::{Distribution, Normal, XorShift64};
use quant_vol::{ewma_vol, ArchModel, GarchModel, VolError};

// Helper: generate GARCH(1,1) returns from a known model.
fn garch11_returns(
    omega: f64,
    alpha: f64,
    beta: f64,
    n: usize,
    rng: &mut XorShift64,
) -> Vec<f64> {
    let normal = Normal::standard();
    let mut sigma2 = omega / (1.0 - alpha - beta); // start at long-run variance
    let mut returns = Vec::with_capacity(n);
    for _ in 0..n {
        let z = normal.sample(rng);
        let r = sigma2.sqrt() * z;
        returns.push(r);
        sigma2 = omega + alpha * r * r + beta * sigma2;
    }
    returns
}

// R8.2: EWMA ------------------------------------------------------------------

#[test]
fn test_ewma_lambda_zero() {
    // lambda = 0: sigma_t^2 = r_{t-1}^2 (tracks lagged squared return).
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
    let sigma2 = ewma_vol(&returns, 0.0).unwrap();
    assert_eq!(sigma2.len(), returns.len());
    // sigma_0 = r_0^2
    assert_relative_eq!(sigma2[0], 0.01_f64 * 0.01, epsilon = 1e-15);
    // sigma_t = r_{t-1}^2 for t >= 1
    assert_relative_eq!(sigma2[1], 0.01_f64 * 0.01, epsilon = 1e-15);
    assert_relative_eq!(sigma2[2], (-0.02_f64) * (-0.02), epsilon = 1e-15);
    assert_relative_eq!(sigma2[3], 0.03_f64 * 0.03, epsilon = 1e-15);
    assert_relative_eq!(sigma2[4], (-0.01_f64) * (-0.01), epsilon = 1e-15);
}

#[test]
fn test_ewma_lambda_one() {
    // lambda = 1: sigma_t^2 = sigma_{t-1}^2 (constant at initial value).
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
    let sigma2 = ewma_vol(&returns, 1.0).unwrap();
    let init = 0.01_f64 * 0.01;
    for &s in &sigma2 {
        assert_relative_eq!(s, init, epsilon = 1e-15);
    }
}

#[test]
fn test_ewma_decay() {
    // lambda = 0.94: weights decay geometrically. The influence of a past
    // shock decays by factor lambda each period.
    let returns = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let sigma2 = ewma_vol(&returns, 0.94).unwrap();
    // sigma_0 = 1^2 = 1
    assert_relative_eq!(sigma2[0], 1.0, epsilon = 1e-15);
    // sigma_1 = 0.94 * 1 + 0.06 * 1^2 = 1.0 (still 1 because r_0 = 1)
    // Wait: sigma_1 = lambda * sigma_0 + (1-lambda) * r_0^2 = 0.94*1 + 0.06*1 = 1.0
    // sigma_2 = 0.94 * 1 + 0.06 * 0 = 0.94
    // sigma_3 = 0.94 * 0.94 + 0.06 * 0 = 0.94^2
    // sigma_t = 0.94^(t-1) for t >= 2
    assert_relative_eq!(sigma2[1], 1.0, epsilon = 1e-15);
    for (t, &val) in sigma2.iter().enumerate().skip(2) {
        let expected = 0.94_f64.powi((t - 1) as i32);
        assert_relative_eq!(val, expected, epsilon = 1e-12);
    }
}

#[test]
fn test_ewma_variance() {
    // White noise: mean of EWMA variance should approximate sample variance.
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let returns: Vec<f64> = (0..5000).map(|_| normal.sample(&mut rng)).collect();
    let sigma2 = ewma_vol(&returns, 0.94).unwrap();
    // Discard the first 500 (burn-in), average the rest.
    let mean_ewma: f64 = sigma2[500..].iter().sum::<f64>() / (sigma2.len() - 500) as f64;
    let sample_var: f64 = {
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        returns.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64
    };
    assert!(
        (mean_ewma - sample_var).abs() / sample_var < 0.15,
        "EWMA mean var {mean_ewma:.6} should be close to sample var {sample_var:.6}"
    );
}

// R8.3: ARCH ------------------------------------------------------------------

#[test]
fn test_arch_zero() {
    // All alphas = 0: constant sigma^2 = omega.
    let model = ArchModel {
        omega: 0.04,
        alphas: vec![0.0],
    };
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
    let sigma2 = model.conditional_variances(&returns);
    for &s in &sigma2 {
        assert_relative_eq!(s, 0.04, epsilon = 1e-15);
    }
}

#[test]
fn test_arch_forecast() {
    // ARCH(1) forecast reverts to long-run variance omega / (1 - alpha).
    let model = ArchModel {
        omega: 0.01,
        alphas: vec![0.3],
    };
    let long_run = model.long_run_variance();
    let forecast = model.forecast(50);
    // The forecast should converge to the long-run variance.
    let last = *forecast.last().unwrap();
    assert!(
        (last - long_run).abs() / long_run < 0.05,
        "ARCH forecast {last:.6} should converge to long-run {long_run:.6}"
    );
}

#[test]
fn test_arch_log_likelihood() {
    // Any valid fit should produce a finite, non-positive log-likelihood.
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let returns: Vec<f64> = (0..200).map(|_| normal.sample(&mut rng) * 0.01).collect();
    let model = ArchModel::fit(&returns, 1).unwrap();
    let ll = model.log_likelihood(&returns);
    assert!(ll.is_finite(), "log-likelihood must be finite, got {ll}");
}

// R8.4: GARCH -----------------------------------------------------------------

#[test]
fn test_garch11_stationarity() {
    // A fitted GARCH(1,1) on stationary returns should have persistence < 1.
    let mut rng = XorShift64::new(42);
    let returns = garch11_returns(0.01, 0.05, 0.90, 1000, &mut rng);
    let model = GarchModel::fit(&returns, 1, 1).unwrap();
    let pers = model.persistence();
    assert!(
        pers < 1.0,
        "persistence {pers:.4} must be < 1 for stationarity"
    );
    assert!(pers >= 0.0, "persistence must be non-negative");
}

#[test]
fn test_garch11_persistence() {
    // Fitted GARCH(1,1) persistence should be in (0, 1).
    let mut rng = XorShift64::new(99);
    let returns = garch11_returns(0.02, 0.08, 0.88, 2000, &mut rng);
    let model = GarchModel::fit(&returns, 1, 1).unwrap();
    let pers = model.persistence();
    assert!(
        pers > 0.0 && pers < 1.0,
        "persistence {pers:.4} should be in (0, 1)"
    );
}

#[test]
fn test_garch11_long_run() {
    // Long-run variance should be positive for a stationary model.
    let mut rng = XorShift64::new(42);
    let returns = garch11_returns(0.01, 0.05, 0.90, 1000, &mut rng);
    let model = GarchModel::fit(&returns, 1, 1).unwrap();
    let lr = model.long_run_variance();
    assert!(lr.is_finite(), "long-run variance must be finite");
    assert!(lr > 0.0, "long-run variance must be positive, got {lr}");
}

#[test]
fn test_garch_forecast_decay() {
    // Forecast should revert toward the long-run variance.
    let model = GarchModel {
        omega: 0.01,
        alphas: vec![0.05],
        betas: vec![0.90],
    };
    let long_run = model.long_run_variance();
    let returns = vec![0.05, -0.04, 0.03, -0.02, 0.01]; // recent high vol
    let forecast = model.forecast_from(&returns, 100);
    let last = *forecast.last().unwrap();
    assert!(
        (last - long_run).abs() / long_run < 0.10,
        "forecast {last:.6} should converge to long-run {long_run:.6}"
    );
}

#[test]
fn test_garch_log_likelihood() {
    // Valid fit should produce a finite, non-positive log-likelihood.
    let mut rng = XorShift64::new(7);
    let returns = garch11_returns(0.01, 0.05, 0.90, 500, &mut rng);
    let model = GarchModel::fit(&returns, 1, 1).unwrap();
    let ll = model.log_likelihood(&returns);
    assert!(ll.is_finite(), "log-likelihood must be finite, got {ll}");
}

#[test]
fn test_garch_vol_clustering() {
    // GARCH fit on clustered-volatility data should beat constant-volatility
    // (sample variance) in log-likelihood.
    let mut rng = XorShift64::new(42);
    let returns = garch11_returns(0.01, 0.10, 0.85, 2000, &mut rng);
    let model = GarchModel::fit(&returns, 1, 1).unwrap();
    let ll_garch = model.log_likelihood(&returns);

    // Constant-volatility model: sigma^2 = sample variance for all t.
    let sample_var: f64 = {
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        returns.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64
    };
    let ll_constant: f64 = returns
        .iter()
        .map(|&r| -0.5 * (1.8378770664093453 + sample_var.ln() + r * r / sample_var))
        .sum();

    assert!(
        ll_garch > ll_constant,
        "GARCH LL {ll_garch:.2} should exceed constant-vol LL {ll_constant:.2}"
    );
}

#[test]
fn test_fit_convergence() {
    // GARCH(1,1) MLE should recover parameters close to the true values.
    let mut rng = XorShift64::new(123);
    let true_omega = 0.02;
    let true_alpha = 0.08;
    let true_beta = 0.88;
    let returns = garch11_returns(true_omega, true_alpha, true_beta, 5000, &mut rng);
    let model = GarchModel::fit(&returns, 1, 1).unwrap();
    let pers = model.persistence();
    let true_pers = true_alpha + true_beta;
    assert!(
        (pers - true_pers).abs() < 0.05,
        "fitted persistence {pers:.4} should be close to true {true_pers:.4}"
    );
    let lr = model.long_run_variance();
    let true_lr = true_omega / (1.0 - true_pers);
    assert!(
        (lr - true_lr).abs() / true_lr < 0.20,
        "fitted long-run var {lr:.6} should be close to true {true_lr:.6}"
    );
}

#[test]
fn test_vol_smoke() {
    // Smoke test: all models produce finite output on real-ish returns.
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let returns: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng) * 0.01).collect();

    let ewma = ewma_vol(&returns, 0.94).unwrap();
    assert!(ewma.iter().all(|&v| v.is_finite()));

    let arch = ArchModel::fit(&returns, 1).unwrap();
    let arch_sigma2 = arch.conditional_variances(&returns);
    assert!(arch_sigma2.iter().all(|&v| v.is_finite() && v > 0.0));

    let garch = GarchModel::fit(&returns, 1, 1).unwrap();
    let garch_sigma2 = garch.conditional_variances(&returns);
    assert!(garch_sigma2.iter().all(|&v| v.is_finite() && v > 0.0));
}

// Extra: error handling --------------------------------------------------------

#[test]
fn test_ewma_invalid_lambda() {
    let returns = vec![0.01, 0.02];
    assert!(matches!(
        ewma_vol(&returns, 1.5).unwrap_err(),
        VolError::InvalidParam(_)
    ));
    assert!(matches!(
        ewma_vol(&returns, -0.1).unwrap_err(),
        VolError::InvalidParam(_)
    ));
}

#[test]
fn test_ewma_empty() {
    let err = ewma_vol(&[], 0.94).unwrap_err();
    assert!(matches!(err, VolError::InsufficientData { .. }));
}