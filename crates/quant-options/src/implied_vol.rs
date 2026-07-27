//! Implied volatility: invert the Black-Scholes formula to recover the
//! volatility the market is pricing in.
//!
//! Given a market price `C_mkt`, solve `bs_call(S, K, r, sigma, T) = C_mkt`
//! for `sigma`. Newton's method is the textbook choice because Vega is the
//! exact derivative of the BS price with respect to `sigma`:
//!
//! ```text
//! sigma_{n+1} = sigma_n - (C(sigma_n) - C_mkt) / Vega(sigma_n)
//! ```
//!
//! Newton converges quadratically when the initial guess is reasonable and the
//! option is not deep in/out of the money (where Vega -> 0 and the iteration
//! becomes ill-conditioned). When Vega collapses or the step leaves the valid
//! bracket, we fall back to bisection on `[sigma_min, sigma_max]`, which is
//! robust but linear.
//!
//! The implementation here uses a **hybrid** strategy:
//! 1. Try Newton from a Brenner-Subrahmanyam (1988) initial guess.
//! 2. If Newton leaves the bracket or Vega is too small, switch to bisection
//!    for the remainder.
//! 3. The first method to drive `|C(sigma) - C_mkt|` below `tol` wins.

use crate::error::OptionsError;
use crate::greeks::vega;
use quant_stochastic::bs_call;

/// Default convergence tolerance on the price residual (1e-8 is far below
/// typical bid-ask spread).
const DEFAULT_TOL: f64 = 1e-8;

/// Default maximum iterations (Newton and bisection budgets combined).
const DEFAULT_MAX_ITERS: usize = 200;

/// Lower / upper bisection bounds on volatility. The upper bound is generous
/// (500% annualised) to accommodate distressed markets.
const SIGMA_MIN: f64 = 1e-6;
const SIGMA_MAX: f64 = 5.0;

/// Vega threshold below which the Newton step is unstable. When `|Vega|` falls
/// below this, we hand off to bisection.
const VEGA_FLOOR: f64 = 1e-6;

/// Solve `bs_call(S, K, r, sigma, T) = market_price` for `sigma`.
///
/// Newton's method with a bisection fallback. Works for both calls and puts:
/// by put-call parity the implied vol that reproduces a put price equals the
/// implied vol that reproduces the corresponding call price, so we always
/// invert the call formula and let the caller pass the call price.
/// (For a put, pass `bs_put + S0 - K*exp(-rT)` as the market price, or simply
/// use `implied_vol_put` which does the conversion.)
pub fn implied_vol(
    market_price: f64,
    s0: f64,
    k: f64,
    r: f64,
    t: f64,
    is_call: bool,
) -> Result<f64, OptionsError> {
    validate_inputs(market_price, s0, k, r, t)?;

    // Translate a put price into the equivalent call price via put-call
    // parity: C = P + S0 - K exp(-rT). The implied vol is invariant.
    let call_price = if is_call {
        market_price
    } else {
        market_price + s0 - k * (-r * t).exp()
    };

    // No-arbitrage bounds on the call price. The lower bound is intrinsic
    // (max(S0 - K*exp(-rT), 0)); the upper bound is S0.
    let lower = (s0 - k * (-r * t).exp()).max(0.0);
    let upper = s0;
    if call_price < lower - 1e-9 || call_price > upper + 1e-9 {
        return Err(OptionsError::ArbitrageViolation {
            market_price,
            lower,
            upper,
        });
    }

    // Edge case: market price equals intrinsic. Implied vol is zero (or
    // SIGMA_MIN). Return the floor.
    if (call_price - lower).abs() < DEFAULT_TOL {
        return Ok(SIGMA_MIN);
    }
    if (call_price - upper).abs() < DEFAULT_TOL {
        return Ok(SIGMA_MAX);
    }

    // Brenner-Subrahmanyam (1988) initial guess: sigma ~= sqrt(2 pi / T) * C/S0.
    // Robust for ATM options; bracketed by bisection if it diverges.
    let mut sigma = (2.0 * std::f64::consts::PI / t).sqrt() * call_price / s0;
    if !sigma.is_finite() || sigma <= SIGMA_MIN || sigma >= SIGMA_MAX {
        sigma = 0.2; // 20% fall-back initial guess
    }

    let mut lo = SIGMA_MIN;
    let mut hi = SIGMA_MAX;

    for iteration in 0..DEFAULT_MAX_ITERS {
        let price = bs_call(s0, k, r, sigma, t);
        let diff = price - call_price;

        if diff.abs() < DEFAULT_TOL {
            return Ok(sigma);
        }

        // Keep the bisection bracket consistent: we know C is monotone
        // increasing in sigma, so sign the bounds by the residual.
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }

        let v = vega(s0, k, r, sigma, t);

        // Newton step. If vega is too small OR the step leaves the bracket,
        // use bisection instead.
        let newton_step = if v.abs() > VEGA_FLOOR {
            sigma - diff / v
        } else {
            f64::NAN
        };

        let next = if newton_step.is_finite()
            && newton_step > lo
            && newton_step < hi
        {
            newton_step
        } else {
            0.5 * (lo + hi)
        };

        // If bisection bracket has collapsed, we are done.
        if (hi - lo).abs() < DEFAULT_TOL {
            return Ok(0.5 * (lo + hi));
        }

        sigma = next;
        // If we have fallen back to pure bisection for many iterations, keep
        // going; otherwise continue Newton.
        let _ = iteration;
    }

    Err(OptionsError::NoConvergence {
        iterations: DEFAULT_MAX_ITERS,
    })
}

fn validate_inputs(
    market_price: f64,
    s0: f64,
    k: f64,
    r: f64,
    t: f64,
) -> Result<(), OptionsError> {
    if s0 <= 0.0 {
        return Err(OptionsError::InvalidParam(format!("s0 must be positive, got {s0}")));
    }
    if k <= 0.0 {
        return Err(OptionsError::InvalidParam(format!("k must be positive, got {k}")));
    }
    if t <= 0.0 {
        return Err(OptionsError::InvalidParam(format!("t must be positive, got {t}")));
    }
    if !market_price.is_finite() || market_price < 0.0 {
        return Err(OptionsError::InvalidParam(format!(
            "market_price must be non-negative finite, got {market_price}"
        )));
    }
    if !r.is_finite() {
        return Err(OptionsError::InvalidParam(format!("r must be finite, got {r}")));
    }
    Ok(())
}