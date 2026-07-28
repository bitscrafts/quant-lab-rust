//! Stochastic processes and Monte Carlo pricing for the quant-finance
//! curriculum (Phase 9).
//!
//! Hand-rolled Brownian motion, geometric Brownian motion, Poisson
//! jump-diffusion, and Monte Carlo European option pricing. A closed-form
//! Black-Scholes module provides the analytical benchmark for validating the
//! Monte Carlo estimator. No `rand`, `nalgebra`, or `statrs` — randomness
//! comes from `quant-core`'s `XorShift64` and `Normal`.
//!
//! # Modules
//!
//! - [`brownian`]: standard Brownian motion and GBM simulators
//! - [`poisson`]: Poisson process and Merton jump-diffusion
//! - [`blackscholes`]: closed-form BS call/put (analytical benchmark)
//! - [`montecarlo`]: Monte Carlo European option pricing with antithetic variates
//!
//! # Example
//!
//! ```
//! use quant_core::{XorShift64, Rng};
//! use quant_stochastic::{brownian_motion, bs_call, mc_call};
//!
//! let mut rng = XorShift64::new(42);
//! let w = brownian_motion(100, 1.0 / 252.0, &mut rng);
//! assert_eq!(w.len(), 101);
//! assert_eq!(w[0], 0.0);
//!
//! let mc = mc_call(100.0, 100.0, 0.05, 0.2, 1.0, 50_000, &mut rng).unwrap();
//! let bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
//! assert!((mc.price - bs).abs() < 0.5);
//! ```

pub mod blackscholes;
pub mod brownian;
pub mod error;
pub mod montecarlo;
pub mod poisson;
pub mod processes;

pub use blackscholes::{bs_call, bs_put, d1, d2, normal_cdf};
pub use brownian::{brownian_motion, gbm, quadratic_variation};
pub use error::StochError;
pub use montecarlo::{McResult, ci_half_width, mc_call, mc_call_antithetic, mc_put};
pub use poisson::{exponential_variate, jump_diffusion, poisson_count, poisson_process};
pub use processes::{Gbm, JumpDiffusion, Poisson};
