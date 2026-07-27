//! Tests for the quant-options crate (Phase 10 TDD contract, 15 tests).

use approx::assert_abs_diff_eq;
use quant_options::{
    bs_call, bs_put, delta, delta_fd, gamma, gamma_fd, implied_vol, normal_pdf, rho, theta,
    vega, vega_fd,
};

const S0: f64 = 100.0;
const K: f64 = 100.0;
const R: f64 = 0.05;
const SIGMA: f64 = 0.2;
const T: f64 = 1.0;

#[test]
fn test_bs_call_put_parity() {
    // Call - Put = S0 - K * exp(-rT)
    let c = bs_call(S0, K, R, SIGMA, T);
    let p = bs_put(S0, K, R, SIGMA, T);
    let parity = S0 - K * (-R * T).exp();
    assert_abs_diff_eq!(c - p, parity, epsilon = 1e-10);
}

#[test]
fn test_normal_pdf_unit_interval() {
    // phi(0) = 1 / sqrt(2 pi) ~ 0.39894228
    assert_abs_diff_eq!(normal_pdf(0.0), 0.3989422804014327, epsilon = 1e-12);
    // phi is symmetric: phi(x) == phi(-x)
    assert_abs_diff_eq!(normal_pdf(1.5), normal_pdf(-1.5), epsilon = 1e-12);
}

#[test]
fn test_delta_call_itm() {
    // K < S0 -> ITM call -> delta in (0.5, 1)
    let d = delta(S0, 80.0, R, SIGMA, T, true);
    assert!(d > 0.0 && d < 1.0, "delta out of (0,1): {d}");
    assert!(d > 0.5, "ITM call delta should exceed 0.5: {d}");
}

#[test]
fn test_delta_put_otm() {
    // K > S0 -> OTM put -> delta in (-1, 0)
    let d = delta(S0, 120.0, R, SIGMA, T, false);
    assert!(d > -1.0 && d < 0.0, "delta out of (-1,0): {d}");
}

#[test]
fn test_gamma_positive() {
    let g = gamma(S0, K, R, SIGMA, T);
    assert!(g > 0.0, "gamma must be positive: {g}");
}

#[test]
fn test_gamma_atm_max() {
    // Gamma peaks at ATM and shrinks in the wings.
    let g_atm = gamma(S0, S0, R, SIGMA, T);
    let g_itm = gamma(S0, 80.0, R, SIGMA, T);
    let g_otm = gamma(S0, 120.0, R, SIGMA, T);
    assert!(
        g_atm >= g_itm && g_atm >= g_otm,
        "ATM gamma ({g_atm}) should exceed ITM ({g_itm}) and OTM ({g_otm})"
    );
}

#[test]
fn test_vega_positive() {
    let v = vega(S0, K, R, SIGMA, T);
    assert!(v > 0.0, "vega must be positive: {v}");
}

#[test]
fn test_theta_call_negative() {
    // Short-dated ATM call: theta is negative (long call loses time value).
    let th = theta(S0, K, R, SIGMA, T, true);
    assert!(th < 0.0, "call theta should be negative: {th}");
}

#[test]
fn test_rho_call_positive() {
    let rho_call = rho(S0, K, R, SIGMA, T, true);
    assert!(rho_call > 0.0, "call rho should be positive: {rho_call}");
    let rho_put = rho(S0, K, R, SIGMA, T, false);
    assert!(rho_put < 0.0, "put rho should be negative: {rho_put}");
}

#[test]
fn test_delta_fd_matches_analytical() {
    let h = 1e-4;
    let d_ana = delta(S0, K, R, SIGMA, T, true);
    let d_fd = delta_fd(S0, K, R, SIGMA, T, true, h);
    assert!((d_ana - d_fd).abs() < 1e-4, "|delta - delta_fd| = {}", (d_ana - d_fd).abs());
}

#[test]
fn test_gamma_fd_matches_analytical() {
    let h = 1e-3;
    let g_ana = gamma(S0, K, R, SIGMA, T);
    let g_fd = gamma_fd(S0, K, R, SIGMA, T, h);
    assert!((g_ana - g_fd).abs() < 1e-3, "|gamma - gamma_fd| = {}", (g_ana - g_fd).abs());
}

#[test]
fn test_vega_fd_matches_analytical() {
    let h = 1e-4;
    let v_ana = vega(S0, K, R, SIGMA, T);
    let v_fd = vega_fd(S0, K, R, SIGMA, T, h);
    assert!((v_ana - v_fd).abs() < 1e-3, "|vega - vega_fd| = {}", (v_ana - v_fd).abs());
}

#[test]
fn test_implied_vol_recovers() {
    let price = bs_call(S0, K, R, SIGMA, T);
    let iv = implied_vol(price, S0, K, R, T, true).expect("IV should converge");
    assert_abs_diff_eq!(iv, SIGMA, epsilon = 1e-8);
}

#[test]
fn test_implied_vol_zero_vega() {
    // Deep ITM call: vega collapses. Newton's derivative becomes unreliable
    // and the bisection fallback must take over. We use a moderately deep
    // ITM strike where the time value is still above float precision, so
    // sigma is recoverable, but vega is small enough to exercise the
    // fallback path.
    let s0_deep = 150.0;
    let k_deep = 100.0;
    let price = bs_call(s0_deep, k_deep, R, SIGMA, T);
    let iv = implied_vol(price, s0_deep, k_deep, R, T, true).expect("bisection fallback");
    assert_abs_diff_eq!(iv, SIGMA, epsilon = 1e-4);
}

#[test]
fn test_put_call_parity_iv() {
    // The implied vol from a call price equals the implied vol from the
    // corresponding put price (put-call parity).
    let c = bs_call(S0, K, R, SIGMA, T);
    let p = bs_put(S0, K, R, SIGMA, T);
    let iv_call = implied_vol(c, S0, K, R, T, true).unwrap();
    let iv_put = implied_vol(p, S0, K, R, T, false).unwrap();
    assert_abs_diff_eq!(iv_call, iv_put, epsilon = 1e-6);
}

#[test]
fn test_options_smoke() {
    // All greeks produce finite output for a vanilla ATM call and put.
    for &is_call in &[true, false] {
        let d = delta(S0, K, R, SIGMA, T, is_call);
        let g = gamma(S0, K, R, SIGMA, T);
        let v = vega(S0, K, R, SIGMA, T);
        let th = theta(S0, K, R, SIGMA, T, is_call);
        let rh = rho(S0, K, R, SIGMA, T, is_call);
        for &x in &[d, g, v, th, rh] {
            assert!(x.is_finite(), "non-finite greek: {x}");
        }
    }
}