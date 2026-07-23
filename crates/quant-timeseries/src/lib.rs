//! Time-series econometrics: OLS regression, autocorrelation, ADF
//! stationarity testing, and fractional differentiation.
//!
//! `quant-timeseries` is the Phase 7 crate of the quant-finance curriculum.
//! All math is hand-rolled — no `rand`, `nalgebra`, or `statrs`. OLS is
//! solved via Gaussian elimination with partial pivoting on the normal
//! equations; the ADF test uses the MacKinnon (1996) 5% critical value
//! (-2.86); fractional differentiation follows López de Prado's
//! fixed-width approach (AFML, Ch.5).
//!
//! # Modules
//!
//! - [`ols`]: `OlsFit`, `ols`, Gaussian elimination
//! - [`acf`]: autocorrelation function
//! - [`adf`]: `AdfResult`, `adf_test`, `MACKINNON_5PCT`
//! - [`fracdiff`]: `ffd_weights`, `frac_diff`, `find_min_d`
//!
//! # Example
//!
//! ```
//! use quant_timeseries::{ols, OlsFit};
//!
//! // y = 1 + 2x (perfect fit).
//! let x = vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, 2.0]];
//! let y = vec![1.0, 3.0, 5.0];
//! let fit = ols(&x, &y).unwrap();
//! assert!((fit.r_squared - 1.0).abs() < 1e-9);
//! ```

pub mod acf;
pub mod adf;
pub mod error;
pub mod fracdiff;
pub mod ols;

pub use acf::acf;
pub use adf::{adf_test, AdfResult, MACKINNON_5PCT};
pub use error::TimeSeriesError;
pub use fracdiff::{find_min_d, frac_diff, ffd_weights};
pub use ols::{ols, OlsFit};