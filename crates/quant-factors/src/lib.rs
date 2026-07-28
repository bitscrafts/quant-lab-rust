//! Factor attribution: PCA on returns, the Fama-French 3-factor model,
//! and risk decomposition.
//!
//! `quant-factors` is the Phase 12 crate of the quant-finance curriculum.
//! All math is hand-rolled — no `argmin`, `optimization`, `nalgebra`, or
//! `statrs`. Eigenvalues are computed via the power method with
//! deflation (see [`eigen`]); PCA uses the covariance eigendecomposition;
//! the Fama-French regression reuses the OLS machinery from
//! `quant-timeseries`.
//!
//! # Modules
//!
//! - [`eigen`]: `power_method`, `deflate`, `top_k_eigen`
//! - [`mod@pca`]: `PcaResult`, `pca`, `pca_transform`, `pca_reconstruct`
//! - [`fama_french`]: `FF3Exposure`, `ff3_regression`
//! - [`risk`]: `RiskAttribution`, `risk_attribution`
//! - `error`: `FactorError`
//!
//! # Example
//!
//! ```
//! use quant_factors::pca;
//!
//! let returns = vec![
//!     vec![0.01, 0.02],
//!     vec![0.02, 0.03],
//!     vec![-0.01, 0.0],
//!     vec![0.03, 0.04],
//!     vec![0.0, 0.01],
//! ];
//! let res = pca(&returns, 2).unwrap();
//! println!("top eigenvalue = {:.6}", res.eigenvalues[0]);
//! ```

pub mod eigen;
pub mod error;
pub mod fama_french;
pub mod models;
pub mod pca;
pub mod risk;

pub use eigen::{deflate, power_method, top_k_eigen};
pub use error::FactorError;
pub use fama_french::{FF3Exposure, ff3_regression};
pub use models::{FamaFrench3, Pca};
pub use pca::{PcaResult, pca, pca_reconstruct, pca_transform};
pub use risk::{RiskAttribution, risk_attribution};
