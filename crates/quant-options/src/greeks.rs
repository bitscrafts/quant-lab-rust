//! Analytical Black-Scholes Greeks.
//!
//! All five first- and second-order sensitivities are expressed in closed
//! form. `normal_cdf` (`Phi`) is reused from `quant-stochastic`; the standard
//! normal density `phi` (the derivative of `Phi`) is implemented inline as
//! `phi(x) = (1 / sqrt(2 pi)) exp(-x^2 / 2)`.
//!
//! Conventions:
//! - `s0` spot, `k` strike, `r` continuous risk-free rate, `sigma`
//!   volatility, `t` time to maturity (years).
//! - `is_call = true` for calls, `false` for puts. Gamma and Vega are
//!   identical for calls and puts (same `d1`), so they take no `is_call` flag.
//! - Theta is quoted per year (NOT per calendar day); divide by 365 if a
//!   per-day figure is needed.
//!
//! Formulas (call):
//! ```text
//! Delta = Phi(d1)
//! Gamma = phi(d1) / (S0 sigma sqrt(T))
//! Vega  = S0 phi(d1) sqrt(T)
//! Theta = -(S0 phi(d1) sigma) / (2 sqrt(T)) - r K exp(-rT) Phi(d2)
//! Rho   = K T exp(-rT) Phi(d2)
//! ```
//! Put Greeks follow from put-call parity:
//! ```text
//! Delta_p = Phi(d1) - 1
//! Theta_p = Theta_c + r K exp(-rT)
//! Rho_p   = -K T exp(-rT) Phi(-d2)
//! ```

use quant_stochastic::{d1, d2, normal_cdf};

/// Standard normal PDF `phi(x) = (1/sqrt(2 pi)) exp(-x^2 / 2)`.
///
/// This is the derivative of `normal_cdf`. The constant `1/sqrt(2 pi)` is
/// evaluated once; the rest is a single `exp`.
pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Delta: the sensitivity of the option price to a unit change in the spot.
///
/// - Call: `Delta = Phi(d1)` (in `(0, 1)`)
/// - Put:  `Delta = Phi(d1) - 1` (in `(-1, 0)`)
pub fn delta(s0: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    let d1_val = d1(s0, k, r, sigma, t);
    let n_d1 = normal_cdf(d1_val);
    if is_call {
        n_d1
    } else {
        n_d1 - 1.0
    }
}

/// Gamma: the second derivative of the option price with respect to spot.
///
/// Identical for calls and puts. Peaks at-the-money and shrinks in the tails.
/// `Gamma = phi(d1) / (S0 sigma sqrt(T))`.
pub fn gamma(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let d1_val = d1(s0, k, r, sigma, t);
    normal_pdf(d1_val) / (s0 * sigma * t.sqrt())
}

/// Vega: the sensitivity of the option price to a 1% (i.e. 0.01) change in
/// volatility. Returned here in raw units (per unit vol); multiply by 0.01 for
/// the per-percent convention used on trading desks.
///
/// Identical for calls and puts. `Vega = S0 phi(d1) sqrt(T)`.
pub fn vega(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let d1_val = d1(s0, k, r, sigma, t);
    s0 * normal_pdf(d1_val) * t.sqrt()
}

/// Theta: the sensitivity of the option price to the passage of time.
///
/// Quoted per year (negative for long calls and puts, since time value decays).
/// To get per-calendar-day theta, divide by 365.
///
/// - Call: `Theta = -(S0 phi(d1) sigma) / (2 sqrt(T)) - r K exp(-rT) Phi(d2)`
/// - Put:  `Theta = -(S0 phi(d1) sigma) / (2 sqrt(T)) + r K exp(-rT) Phi(-d2)`
pub fn theta(s0: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    let d1_val = d1(s0, k, r, sigma, t);
    let d2_val = d2(s0, k, r, sigma, t);
    let common = -(s0 * normal_pdf(d1_val) * sigma) / (2.0 * t.sqrt());
    let disc = k * (-r * t).exp();
    if is_call {
        common - r * disc * normal_cdf(d2_val)
    } else {
        common + r * disc * normal_cdf(-d2_val)
    }
}

/// Rho: the sensitivity of the option price to a 1% change in the risk-free
/// rate. Returned here in raw units (per unit rate); multiply by 0.01 for the
/// per-percent convention.
///
/// - Call: `Rho = K T exp(-rT) Phi(d2)` (positive)
/// - Put:  `Rho = -K T exp(-rT) Phi(-d2)` (negative)
pub fn rho(s0: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    let d2_val = d2(s0, k, r, sigma, t);
    let disc = k * t * (-r * t).exp();
    if is_call {
        disc * normal_cdf(d2_val)
    } else {
        -disc * normal_cdf(-d2_val)
    }
}