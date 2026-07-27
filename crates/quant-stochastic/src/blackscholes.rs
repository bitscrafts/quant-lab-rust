//! Black-Scholes closed-form pricing — the analytical benchmark for
//! validating the Monte Carlo estimator.
//!
//! See `README.md` in this directory for the module overview.

/// Standard normal CDF `Phi(x)` via the Abramowitz & Stegun (1964) formula 7.1.26
/// approximation of the error function, with `Phi(x) = 0.5 * (1 + erf(x / sqrt 2))`.
///
/// Maximum absolute error `< 7.5e-8` versus the exact integral.
pub fn normal_cdf(x: f64) -> f64 {
    let sqrt2 = std::f64::consts::SQRT_2;
    0.5 * (1.0 + erf(x / sqrt2))
}

/// Abramowitz & Stegun (1964) formula 7.1.26 approximation of `erf(x)`.
fn erf(x: f64) -> f64 {
    // Sign handling: erf is odd.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// Black-Scholes `d1`.
pub fn d1(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

/// Black-Scholes `d2 = d1 - sigma * sqrt(t)`.
pub fn d2(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    d1(s0, k, r, sigma, t) - sigma * t.sqrt()
}

/// Black-Scholes European call price:
/// `C = S0 * Phi(d1) - K * exp(-rT) * Phi(d2)`.
pub fn bs_call(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let n_d1 = normal_cdf(d1(s0, k, r, sigma, t));
    let n_d2 = normal_cdf(d2(s0, k, r, sigma, t));
    s0 * n_d1 - k * (-r * t).exp() * n_d2
}

/// Black-Scholes European put price:
/// `P = K * exp(-rT) * Phi(-d2) - S0 * Phi(-d1)`.
pub fn bs_put(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let n_d1 = normal_cdf(-d1(s0, k, r, sigma, t));
    let n_d2 = normal_cdf(-d2(s0, k, r, sigma, t));
    k * (-r * t).exp() * n_d2 - s0 * n_d1
}
