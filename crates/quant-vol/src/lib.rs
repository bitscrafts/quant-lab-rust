//! Volatility models: EWMA, ARCH, GARCH.
//!
//! `quant-vol` is the Phase 8 crate of the quant-finance curriculum. All
//! math is hand-rolled — no `argmin`, `optimization`, `nalgebra`, or
//! `statrs`. Maximum-likelihood fitting uses a hand-rolled Nelder-Mead
//! simplex search.
//!
//! # Modules
//!
//! - [`ewma`]: exponentially weighted moving average (RiskMetrics)
//! - [`arch`]: `ArchModel`, ARCH(q)
//! - [`garch`]: `GarchModel`, GARCH(p, q)
//! - [`nelder_mead`]: derivative-free optimiser
//!
//! # Example
//!
//! ```
//! use quant_vol::{ewma_vol, GarchModel};
//!
//! let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.01, -0.03];
//! let sigma2 = ewma_vol(&returns, 0.94).unwrap();
//! let model = GarchModel::fit(&returns, 1, 1).unwrap();
//! println!("persistence = {:.4}", model.persistence());
//! ```

pub mod arch;
pub mod error;
pub mod ewma;
pub mod garch;
pub mod nelder_mead;

pub use arch::ArchModel;
pub use error::VolError;
pub use ewma::ewma_vol;
pub use garch::GarchModel;
