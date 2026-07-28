//! Black-Scholes options toolkit: closed-form pricing, the Greeks (analytical
//! and finite-difference), and implied volatility via Newton / bisection.
//!
//! `quant-options` is the Phase 10 crate of the quant-finance curriculum. It
//! reuses `normal_cdf`, `bs_call`, `bs_put`, `d1`, and `d2` from
//! `quant-stochastic` and layers the Greeks and an implied-volatility solver
//! on top. No `argmin`, `optimization`, `rand`, `nalgebra`, or `statrs` —
//! Newton's method is implemented inline with a bisection fallback.
//!
//! # Modules
//!
//! - [`greeks`][]: analytical Delta, Gamma, Vega, Theta, Rho
//! - [`finite_diff`][]: finite-difference Greeks (central / forward differences)
//! - [`mod@implied_vol`][]: implied-volatility solver (Newton + bisection fallback)
//! - [`error`][]: `OptionsError`
//!
//! # Example
//!
//! ```
//! use quant_options::{delta, gamma, vega, theta, rho, bs_call};
//!
//! let (s0, k, r, sigma, t) = (100.0, 100.0, 0.05, 0.2, 1.0);
//! println!("call price = {:.4}", bs_call(s0, k, r, sigma, t));
//! println!("delta     = {:.4}", delta(s0, k, r, sigma, t, true));
//! println!("gamma     = {:.4}", gamma(s0, k, r, sigma, t));
//! println!("vega      = {:.4}", vega(s0, k, r, sigma, t));
//! println!("theta     = {:.4}", theta(s0, k, r, sigma, t, true));
//! println!("rho       = {:.4}", rho(s0, k, r, sigma, t, true));
//! ```

pub mod error;
pub mod finite_diff;
pub mod greeks;
pub mod implied_vol;
pub mod pricer;

pub use error::OptionsError;
pub use finite_diff::{delta_fd, gamma_fd, theta_fd, vega_fd};
pub use greeks::{delta, gamma, normal_pdf, rho, theta, vega};
pub use implied_vol::implied_vol;
pub use pricer::BlackScholes;

// Re-export the Black-Scholes pricing primitives from `quant-stochastic` so
// that `quant-options` is a single import surface for the options toolkit.
pub use quant_stochastic::{bs_call, bs_put, d1, d2, normal_cdf};
