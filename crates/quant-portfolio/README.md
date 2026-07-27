# quant-portfolio

Markowitz mean-variance portfolio theory, the efficient frontier, the
tangency portfolio, the capital market line, two-fund separation, CAPM,
and historical tail risk metrics. Phase 11 of the quant-finance curriculum.

## Overview

`quant-portfolio` builds the classical portfolio-theory toolkit on top of
the returns and covariance machinery from `quant-core` and
`quant-timeseries`. All math is hand-rolled — no `argmin`, `optimization`,
`nalgebra`, or `statrs`. Linear systems are solved with a small Gaussian
elimination routine in [`linalg`](src/linalg.rs); the N-asset
minimum-variance and tangency portfolios are closed-form linear-algebra
problems once `Sigma^{-1}` is available.

## Modules

| Module | Purpose |
|---|---|
| [`linalg`](src/linalg.rs) | Dense linear algebra: `solve`, `inverse`, `matvec`, `matmul`, `quadratic_form` |
| [`portfolio`](src/portfolio.rs) | `Portfolio` struct, return/variance/Sharpe, `Allocator` trait |
| [`frontier`](src/frontier.rs) | Two-asset closed-form frontier + N-asset min-variance and target-return Lagrangian |
| [`tangency`](src/tangency.rs) | Maximum-Sharpe tangency portfolio, CML, two-fund separation |
| [`capm`](src/capm.rs) | Beta, Jensen's alpha, security market line |
| [`risk`](src/risk.rs) | Historical VaR, CVaR, `RiskModel` trait |

## API Surface

```rust
use quant_portfolio::{
    Portfolio, PortfolioStats, Allocator,
    portfolio_return, portfolio_variance, portfolio_volatility, sharpe_ratio,
    min_variance_portfolio, efficient_frontier_point,
    two_asset_frontier_point, two_asset_min_variance_weight, FrontierPoint,
    tangency_portfolio, capital_market_line, two_fund_separation, TangencyResult,
    beta, alpha, sml,
    historical_var, historical_cvar, RiskModel,
    PortfolioError,
};
```

## Design Principles

- **Closed-form over numerical**: the N-asset minimum-variance, target-return,
  and tangency portfolios are all linear-algebra problems with explicit
  solutions in terms of `Sigma^{-1}`. No gradient descent, no QP solver.
- **Two paths for the frontier**: the two-asset case is closed-form
  (single parameter `w`); the N-asset case uses the Lagrangian system
  with the standard scalars `a, b, c, d`.
- **Trait-based design**: `Allocator` (portfolio-construction rules) and
  `RiskModel` (tail-risk estimators) are exposed as traits so new rules
  can be added without touching the call sites.
- **Honest errors**: `PortfolioError` distinguishes invalid parameters,
  singular covariance, insufficient data, infeasible target return, and
  dimension mismatch.

## Dependencies

| Crate | Role |
|---|---|
| `quant-core` | `Moments`, `mean`, `variance` (re-exported for convenience) |
| `quant-timeseries` | `ols`, `acf`, `adf` (available for future factor-regression work) |
| `thiserror` | `Error` derive |

## Test Contract

| Suite | Count | Coverage |
|---|---|---|
| `lib` unit tests | 30 | linalg, portfolio, frontier, tangency, capm, risk internals |
| `tests/portfolio_tests.rs` | 16 | Public-API TDD contract (Phase 11) |
| Doc test | 1 | `lib.rs` overview example |

Run the full gate:

```bash
cargo test -p quant-portfolio
cargo clippy -p quant-portfolio --all-targets -- -D warnings
```

## Examples

```bash
# Efficient frontier, GMV, tangency, CML, N-asset Lagrangian table
cargo run -p quant-portfolio --example frontier

# CAPM regression: synthetic noisy linear asset, beta/alpha/SML
cargo run -p quant-portfolio --example capm
```

## Verified Results

Two-asset universe: `mu = [0.10, 0.05]`, `cov = diag(0.04, 0.09)`,
`rf = 0.02`:

| Portfolio | `w_A` | `w_B` | `mu_p` | `sigma_p` | Sharpe |
|---|---|---|---|---|---|
| Global min variance | 0.6923 | 0.3077 | 0.0846 | 0.1664 | 0.388 |
| Tangency (max Sharpe) | 0.8571 | 0.1429 | 0.0929 | 0.1767 | 0.4123 |

CAPM synthetic regression (60 observations, true `beta = 1.2`, `rf = 0.02`):

| Estimate | Value |
|---|---|
| `beta_hat` | 1.1752 |
| `alpha_hat` | -0.0002 |
| SML-predicted return | 0.0919 |
| Realised mean return | 0.0917 |

Historical VaR/CVaR on the arithmetic series `[-0.045, -0.035, ..., 0.045]`
at 95% confidence:

| Metric | Value |
|---|---|
| VaR (5% quantile, linear interp) | 0.0405 |
| CVaR (mean of tail `<=` quantile) | 0.0450 |

## Error Model

`PortfolioError` variants:

- `InvalidParam(String)` — empty universe, `rf` outside `[-1, 1]`, etc.
- `SingularCovariance(String)` — `Sigma` not invertible (zero pivot, etc.)
- `InsufficientData { required, actual }` — too few observations
- `InfeasibleTarget { target, lo, hi }` — target return outside reachable
  range (only fires when the Lagrangian denominator `d = ac - b^2 ~ 0`)
- `DimensionMismatch(String)` — weights, `mu`, `Sigma` disagree on `n`

## References

- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*.
- Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market
  Equilibrium under Conditions of Risk". *Journal of Finance*.
- Lintner, J. (1965). "The Valuation of Risk Assets and the Selection of
  Risky Investments in Stock Portfolios and Capital Budgets".
- Roy, A. D. (1952). "Safety First and the Holding of Assets".
- Rockafellar, R. T. and Uryasev, S. (2000). "Optimization of Conditional
  Value-at-Risk". *Journal of Risk*.

## Related Research

- See [book/chapters/ch11.tex](../../book/chapters/ch11.tex) for the
  textbook treatment with derivations.
- Memory insight: `quant-finance/experiment/phase11-markowitz-portfolio` in
  agent-memory.