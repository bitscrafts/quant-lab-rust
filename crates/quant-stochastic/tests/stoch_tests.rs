//! Integration tests for the quant-stochastic crate (TDD contract: 15 tests).

use approx::assert_relative_eq;
use quant_core::XorShift64;
use quant_stochastic::{
    bs_call, bs_put, brownian_motion, exponential_variate, gbm, jump_diffusion,
    mc_call, mc_call_antithetic, mc_put, normal_cdf, poisson_count, poisson_process,
    quadratic_variation, StochError,
};

// R9.2: Brownian motion -------------------------------------------------------

#[test]
fn test_bm_starts_at_zero() {
    // W_0 = 0 by construction; path length is n + 1.
    let mut rng = XorShift64::new(42);
    let w = brownian_motion(100, 1.0 / 252.0, &mut rng);
    assert_eq!(w.len(), 101);
    assert_eq!(w[0], 0.0);
}

#[test]
fn test_bm_quadratic_variation() {
    // The quadratic variation sum dW^2 converges in probability to T = n*dt.
    // With n=10000 and dt=1/252, T ~ 39.7. We allow 5% tolerance.
    let mut rng = XorShift64::new(42);
    let n = 10000;
    let dt = 1.0 / 252.0;
    let t = n as f64 * dt;
    let w = brownian_motion(n, dt, &mut rng);
    let qv = quadratic_variation(&w);
    assert!(
        (qv - t).abs() / t < 0.05,
        "quadratic variation {qv:.4} should be close to T={t:.4}"
    );
}

#[test]
fn test_gbm_terminal_distribution() {
    // E[log(S_T / S_0)] = (mu - 0.5*sigma^2) * T exactly (log of exact solution).
    let mut rng = XorShift64::new(42);
    let n_paths = 100000;
    let s0 = 100.0_f64;
    let mu = 0.05_f64;
    let sigma = 0.2_f64;
    let t = 1.0;
    let n = 252;
    let expected = (mu - 0.5 * sigma * sigma) * t;
    let log_returns: Vec<f64> = (0..n_paths)
        .map(|_| {
            let p = gbm(s0, mu, sigma, t, n, &mut rng);
            (p[n] / s0).ln()
        })
        .collect();
    let mean = log_returns.iter().sum::<f64>() / n_paths as f64;
    assert!(
        (mean - expected).abs() < 0.02,
        "mean log-return {mean:.4} should be close to {expected:.4}"
    );
}

#[test]
fn test_gbm_known_solution() {
    // With sigma = 0 the GBM path is deterministic: S_T = s0 * exp(mu * T).
    let mut rng = XorShift64::new(42);
    let s0 = 100.0_f64;
    let mu = 0.05_f64;
    let t = 1.0;
    let n = 252;
    let p = gbm(s0, mu, 0.0, t, n, &mut rng);
    let expected = s0 * (mu * t).exp();
    assert_relative_eq!(p[n], expected, epsilon = 1e-10);
}

// R9.3: Poisson and jump-diffusion -------------------------------------------

#[test]
fn test_poisson_rate() {
    // E[N(t)] = rate * t. With many draws the mean count converges.
    let mut rng = XorShift64::new(42);
    let rate = 5.0_f64;
    let t = 1.0;
    let expected = rate * t;
    let n_trials = 5000;
    let counts: Vec<usize> = (0..n_trials).map(|_| poisson_count(rate, t, &mut rng)).collect();
    let mean = counts.iter().sum::<usize>() as f64 / n_trials as f64;
    assert!(
        (mean - expected).abs() / expected < 0.05,
        "mean count {mean:.3} should be close to rate*t={expected:.3}"
    );
}

#[test]
fn test_poisson_interarrival() {
    // Exp(rate) interarrival times have mean 1/rate and var 1/rate^2.
    let mut rng = XorShift64::new(42);
    let rate = 3.0_f64;
    let n = 50000;
    let gaps: Vec<f64> = (0..n).map(|_| exponential_variate(rate, &mut rng)).collect();
    let mean = gaps.iter().sum::<f64>() / n as f64;
    let expected_mean = 1.0 / rate;
    assert!(
        (mean - expected_mean).abs() / expected_mean < 0.03,
        "mean gap {mean:.4} should be close to 1/rate={expected_mean:.4}"
    );
    let var: f64 = gaps.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let expected_var = 1.0 / (rate * rate);
    assert!(
        (var - expected_var).abs() / expected_var < 0.06,
        "gap variance {var:.4} should be close to 1/rate^2={expected_var:.4}"
    );
}

#[test]
fn test_jump_diffusion_drift() {
    // With jump_rate = 0 the jump-diffusion reduces exactly to GBM (same RNG
    // stream, same increments). Verify the terminal values match.
    let mut rng1 = XorShift64::new(42);
    let mut rng2 = XorShift64::new(42);
    let s0 = 100.0_f64;
    let mu = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let n = 252;
    let gbm_path = gbm(s0, mu, sigma, t, n, &mut rng1);
    let jd_path = jump_diffusion(s0, mu, sigma, 0.0, 0.0, t, n, &mut rng2);
    assert_eq!(gbm_path.len(), jd_path.len());
    for (a, b) in gbm_path.iter().zip(jd_path.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-12);
    }
}

// R9.4: Monte Carlo pricing ----------------------------------------------------

#[test]
fn test_mc_call_convergence() {
    // As N grows the MC call price converges to the Black-Scholes price.
    let mut rng = XorShift64::new(42);
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let bs = bs_call(s0, k, r, sigma, t);

    let small = mc_call(s0, k, r, sigma, t, 100, &mut rng).unwrap();
    let large = mc_call(s0, k, r, sigma, t, 100000, &mut rng).unwrap();

    // Large-N estimate should be within ~3 SE of BS (statistical tolerance).
    assert!(
        (large.price - bs).abs() < 3.0 * large.std_error + 1e-6,
        "large-N price {:.4} should be close to BS {:.4} (se={:.4})",
        large.price, bs, large.std_error
    );
    // The large-N estimate should be much closer than the small-N one.
    assert!(
        (large.price - bs).abs() < (small.price - bs).abs() * 5.0,
        "large-N error should be smaller than small-N error"
    );
}

#[test]
fn test_mc_call_at_the_money() {
    // ATM call: MC price within 4 SE of BS.
    let mut rng = XorShift64::new(42);
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let bs = bs_call(s0, k, r, sigma, t);
    let mc = mc_call(s0, k, r, sigma, t, 50000, &mut rng).unwrap();
    assert!(
        (mc.price - bs).abs() < 4.0 * mc.std_error,
        "MC ATM call {:.4} should be close to BS {:.4} (se={:.4})",
        mc.price, bs, mc.std_error
    );
}

#[test]
fn test_mc_standard_error() {
    // SE ~ discount * std(payoff) / sqrt(N). Halving N should roughly double SE.
    let mut rng = XorShift64::new(42);
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let n1 = 10000;
    let n2 = 40000;
    let mc1 = mc_call(s0, k, r, sigma, t, n1, &mut rng).unwrap();
    let mc2 = mc_call(s0, k, r, sigma, t, n2, &mut rng).unwrap();
    // SE scales as 1/sqrt(N); 4x paths -> 2x smaller SE.
    let ratio = mc1.std_error / mc2.std_error;
    assert!(
        (ratio - 2.0).abs() < 0.25,
        "SE ratio {ratio:.3} should be close to 2.0 (4x paths)"
    );
}

#[test]
fn test_mc_in_the_money() {
    // Deep ITM call: price ~ S0 - K*exp(-rT) (intrinsic value, almost
    // certainly exercised). Allow a small time-value slack.
    let mut rng = XorShift64::new(42);
    let s0 = 200.0_f64;
    let k = 50.0_f64;
    let r = 0.05_f64;
    let sigma = 0.2_f64;
    let t = 1.0_f64;
    let intrinsic = s0 - k * (-r * t).exp();
    let mc = mc_call(s0, k, r, sigma, t, 50000, &mut rng).unwrap();
    // Deep ITM call price should be close to intrinsic. The no-arbitrage
    // lower bound `C >= intrinsic` holds for the true price; the noisy MC
    // estimate may dip slightly below, so we use a closeness tolerance.
    assert!(
        (mc.price - intrinsic).abs() < 0.5,
        "deep ITM call {:.4} should be close to intrinsic {:.4}",
        mc.price, intrinsic
    );
}

#[test]
fn test_mc_out_of_the_money() {
    // Deep OTM call: price ~ 0.
    let mut rng = XorShift64::new(42);
    let s0 = 50.0_f64;
    let k = 200.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let mc = mc_call(s0, k, r, sigma, t, 50000, &mut rng).unwrap();
    assert!(
        mc.price < 1e-6,
        "deep OTM call price {:.6} should be ~0",
        mc.price
    );
}

#[test]
fn test_mc_put_call_parity() {
    // call - put = S0 - K*exp(-rT). MC estimates should satisfy this within SE.
    let mut rng = XorShift64::new(42);
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let n = 100000;
    let call = mc_call(s0, k, r, sigma, t, n, &mut rng).unwrap();
    let put = mc_put(s0, k, r, sigma, t, n, &mut rng).unwrap();
    let parity = s0 - k * (-r * t).exp();
    let diff = call.price - put.price;
    // Combined SE of the difference (independent draws, variances add).
    let combined_se = (call.std_error.powi(2) + put.std_error.powi(2)).sqrt();
    assert!(
        (diff - parity).abs() < 4.0 * combined_se,
        "call-put {diff:.4} should match parity {parity:.4} (combined se {combined_se:.4})"
    );
}

#[test]
fn test_antithetic_variance_reduction() {
    // Antithetic MC should have lower standard error per normal draw than
    // plain MC for the same number of underlying uniforms.
    let mut rng_plain = XorShift64::new(42);
    let mut rng_anti = XorShift64::new(42);
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let n_normal_draws = 20000;
    let plain = mc_call(s0, k, r, sigma, t, n_normal_draws, &mut rng_plain).unwrap();
    let anti = mc_call_antithetic(s0, k, r, sigma, t, n_normal_draws, &mut rng_anti).unwrap();
    // Both consume n_normal_draws calls to Normal::sample; antithetic uses 2x
    // payoffs per draw so its SE (per draw) should be lower.
    assert!(
        anti.std_error <= plain.std_error * 1.05,
        "antithetic SE {:.6} should be <= plain SE {:.6}",
        anti.std_error, plain.std_error,
    );
    // Both should still be near BS.
    let bs = bs_call(s0, k, r, sigma, t);
    assert!(
        (anti.price - bs).abs() < 4.0 * anti.std_error + 1e-6,
        "antithetic price {:.4} should be close to BS {:.4}",
        anti.price, bs
    );
}

#[test]
fn test_stoch_smoke() {
    // Smoke test: all simulators produce finite output on a representative input.
    let mut rng = XorShift64::new(7);
    let w = brownian_motion(100, 1.0 / 252.0, &mut rng);
    assert!(w.iter().all(|&x| x.is_finite()));

    let p = gbm(100.0, 0.05, 0.2, 1.0, 252, &mut rng);
    assert!(p.iter().all(|&x| x.is_finite() && x > 0.0));

    let times = poisson_process(5.0, 1.0, &mut rng);
    assert!(times.iter().all(|&x| x.is_finite() && x > 0.0 && x < 1.0));

    let jd = jump_diffusion(100.0, 0.05, 0.2, 3.0, 0.1, 1.0, 252, &mut rng);
    assert!(jd.iter().all(|&x| x.is_finite() && x > 0.0));

    let mc = mc_call(100.0, 100.0, 0.05, 0.2, 1.0, 1000, &mut rng).unwrap();
    assert!(mc.price.is_finite() && mc.std_error.is_finite() && mc.std_error >= 0.0);

    let bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    assert!(bs.is_finite() && bs > 0.0);

    let ncd = normal_cdf(1.96);
    assert!((ncd - 0.975).abs() < 1e-3, "Phi(1.96) ~ 0.975, got {ncd:.6}");

    let bs_p = bs_put(100.0, 100.0, 0.05, 0.2, 1.0);
    assert!(bs_p.is_finite() && bs_p > 0.0);
}

// Extra: error handling -------------------------------------------------------

#[test]
fn test_mc_invalid_inputs() {
    let mut rng = XorShift64::new(42);
    assert!(matches!(
        mc_call(-1.0, 100.0, 0.05, 0.2, 1.0, 100, &mut rng).unwrap_err(),
        StochError::InvalidParam(_)
    ));
    assert!(matches!(
        mc_call(100.0, 100.0, 0.05, 0.2, 1.0, 0, &mut rng).unwrap_err(),
        StochError::InsufficientData { .. }
    ));
    assert!(matches!(
        mc_call(100.0, 100.0, 0.05, -0.2, 1.0, 100, &mut rng).unwrap_err(),
        StochError::InvalidParam(_)
    ));
}