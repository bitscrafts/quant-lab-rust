//! Markowitz mean-variance portfolio theory: the efficient frontier, the
//! tangency portfolio, the capital market line, two-fund separation, and CAPM.
//!
//! `quant-portfolio` is the Phase 11 crate of the quant-finance curriculum.
//! It builds on the returns and covariance machinery of `quant-core` and
//! `quant-timeseries`. All math is hand-rolled — no `argmin`, `optimization`,
//! `nalgebra`, or `statrs`. Linear systems are solved with a small Gaussian
//! elimination routine in [`linalg`][]; the N-asset minimum-variance and
//! tangency portfolios are closed-form linear-algebra problems once the
//! covariance matrix inverse is available.
//!
//! # Modules
//!
//! - [`linalg`][]: small dense linear-algebra helpers (solve, inverse, matvec)
//! - [`portfolio`][]: `Portfolio` struct, return/variance/Sharpe, `Allocator` trait
//! - [`frontier`][]: two-asset closed-form frontier + N-asset min-variance and
//!   target-return portfolios
//! - [`tangency`][]: maximum-Sharpe tangency portfolio, capital market line,
//!   two-fund separation
//! - [`capm`][]: beta, alpha, security market line
//! - [`risk`][]: historical VaR / CVaR, `RiskModel` trait
//! - [`error`][]: `PortfolioError`
//!
//! # Example
//!
//! ```
//! use quant_portfolio::{Portfolio, tangency_portfolio, efficient_frontier_point};
//!
//! // Two uncorrelated assets with equal variances.
//! let mu = vec![0.08, 0.12];
//! let cov = vec![vec![0.04, 0.0], vec![0.0, 0.04]];
//! let rf = 0.02;
//! let w_tan = tangency_portfolio(&mu, &cov, rf).unwrap();
//! println!("tangency weights = {:?}", w_tan);
//! ```
//!
//! See the [`frontier`][] module for the analytical two-asset formulas and
//! the N-asset Lagrangian derivations.

pub mod capm;
pub mod error;
pub mod frontier;
pub mod linalg;
pub mod portfolio;
pub mod risk;
pub mod tangency;

pub use capm::{alpha, beta, sml};
pub use error::PortfolioError;
pub use frontier::{
    FrontierPoint, efficient_frontier_point, min_variance_portfolio, two_asset_frontier_point,
    two_asset_min_variance_weight,
};
pub use portfolio::{
    Allocator, Portfolio, PortfolioStats, marginal_risk, portfolio_return, portfolio_variance,
    portfolio_volatility, sharpe_ratio,
};
pub use risk::{RiskModel, historical_cvar, historical_var};
pub use tangency::{TangencyResult, capital_market_line, tangency_portfolio, two_fund_separation};
