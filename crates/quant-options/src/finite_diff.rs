//! Finite-difference Greeks.
//!
//! Analytical Greeks require the closed form; numerical Greeks only require a
//! pricing function and work for any model (Black-Scholes, local vol, Monte
//! Carlo). We use central differences where the symmetry cancels the
//! first-order truncation error, giving `O(h^2)` accuracy.
//!
//! Bump sizes:
//! - Spot/price bumps: `h ~ 1e-3` to `1e-4` is typical. Too small and
//!   floating-point round-off dominates; too large and truncation dominates.
//! - Volatility bump: `h ~ 1e-4` (vega is smooth in `sigma`).
//! - Time bump: use a *forward* difference because we cannot go below `t = 0`
//!   for short-dated options (`t - h` would be negative).

use quant_stochastic::{bs_call, bs_put};

/// Price helper dispatching on `is_call`.
fn price(s0: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    if is_call {
        bs_call(s0, k, r, sigma, t)
    } else {
        bs_put(s0, k, r, sigma, t)
    }
}

/// Numerical Delta via central difference: `(C(S+h) - C(S-h)) / (2h)`.
pub fn delta_fd(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    is_call: bool,
    h: f64,
) -> f64 {
    let up = price(s0 + h, k, r, sigma, t, is_call);
    let down = price(s0 - h, k, r, sigma, t, is_call);
    (up - down) / (2.0 * h)
}

/// Numerical Gamma via central second difference:
/// `(C(S+h) - 2 C(S) + C(S-h)) / h^2`.
///
/// Same for calls and puts (Gamma is the derivative of Delta, and put-call
/// parity shifts Delta by a constant).
pub fn gamma_fd(s0: f64, k: f64, r: f64, sigma: f64, t: f64, h: f64) -> f64 {
    let up = price(s0 + h, k, r, sigma, t, true);
    let mid = price(s0, k, r, sigma, t, true);
    let down = price(s0 - h, k, r, sigma, t, true);
    (up - 2.0 * mid + down) / (h * h)
}

/// Numerical Vega via central difference on `sigma`:
/// `(C(sigma+h) - C(sigma-h)) / (2h)`.
///
/// We bump the call price; since Vega is identical for calls and puts, the
/// choice does not matter.
pub fn vega_fd(s0: f64, k: f64, r: f64, sigma: f64, t: f64, h: f64) -> f64 {
    let up = bs_call(s0, k, r, sigma + h, t);
    let down = bs_call(s0, k, r, sigma - h, t);
    (up - down) / (2.0 * h)
}

/// Numerical Theta via forward difference in `t`:
/// `(C(t) - C(t-h)) / h`.
///
/// Forward (not central) because `t - h` may be negative for short-dated
/// options. Theta is the *negative* of time decay, so the sign is `(price at
/// later t) - (price at earlier t)`, i.e. `-(C(t) - C(t-h)) / h = (C(t-h) -
/// C(t)) / h`. We return the per-year figure; divide by 365 for per-day.
pub fn theta_fd(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    is_call: bool,
    h: f64,
) -> f64 {
    let now = price(s0, k, r, sigma, t, is_call);
    let earlier = price(s0, k, r, sigma, t - h, is_call);
    // Long option loses value as time passes: earlier price >= now price.
    // Theta = dC/dt, but since t decreases as we hold, desk convention is
    // Theta = -(dC/d(remaining t)) = (C(t-h) - C(t)) / h.
    (earlier - now) / h
}

