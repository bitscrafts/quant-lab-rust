# quant-portfolio/src

Source layout for the `quant-portfolio` crate (Phase 11 of the
quant-finance curriculum). All math is hand-rolled — no `argmin`,
`optimization`, `nalgebra`, or `statrs`.

## Module map

| File | Purpose |
|---|---|
| [`lib.rs`](lib.rs) | Crate root, re-exports, doc example |
| [`error.rs`](error.rs) | `PortfolioError` enum (5 variants) |
| [`linalg.rs`](linalg.rs) | Dense linear algebra: `solve`, `inverse`, `matvec`, `matmul`, `quadratic_form` (Gaussian elimination with partial pivoting) |
| [`portfolio.rs`](portfolio.rs) | `Portfolio` struct, return/variance/volatility/Sharpe, `Allocator` trait, free-function helpers |
| [`frontier.rs`](frontier.rs) | Two-asset closed-form frontier (`two_asset_frontier_point`, `two_asset_min_variance_weight`); N-asset Lagrangian (`min_variance_portfolio`, `efficient_frontier_point`); `FrontierPoint` |
| [`tangency.rs`](tangency.rs) | `tangency_portfolio`, `capital_market_line`, `two_fund_separation`, `TangencyResult` |
| [`capm.rs`](capm.rs) | `beta`, `alpha`, `sml` |
| [`risk.rs`](risk.rs) | `historical_var`, `historical_cvar`, `RiskModel` trait, empirical quantile with linear interpolation |

## API at a glance

```rust
// Portfolio statistics
let p = Portfolio::new(weights, mu, cov)?;
p.expected_return(); p.variance(); p.volatility(); p.sharpe(rf);

// Two-asset frontier
two_asset_frontier_point(w, mu_a, mu_b, var_a, var_b, cov_ab);
two_asset_min_variance_weight(var_a, var_b, cov_ab);

// N-asset Lagrangian
min_variance_portfolio(&mu, &cov)?;
efficient_frontier_point(&mu, &cov, mu_target)?;

// Tangency and CML
let tan = tangency_portfolio(&mu, &cov, rf)?;
capital_market_line(rf, &tan, sigma);
two_fund_separation(&tan, target_vol);

// CAPM
beta(&asset, &market)?;
alpha(&asset, &market, rf)?;
sml(beta, market_mean, rf);

// Risk
historical_var(&returns, 0.95)?;
historical_cvar(&returns, 0.95)?;
```

## Design notes

### Why closed-form over numerical?

The Markowitz problem with only equality constraints (budget and target
return) is a quadratic program with a linear KKT system. Once $\Sigma^{-1}$
is available, every portfolio of interest (global min variance, target
return, tangency) is a closed-form linear combination of $\Sigma^{-1} 1$
and $\Sigma^{-1} \mu$. There is no need for gradient descent, active-set
methods, or a QP solver.

### Linear algebra backend

[`linalg.rs`](linalg.rs) implements:
- `solve(a, b)` — Gaussian elimination with partial pivoting (forward
  elimination + back substitution)
- `inverse(a)` — column-by-column solve of $A X = I$
- `matvec`, `matmul`, `quadratic_form` — convenience helpers

These are O(n^3) and sized for the small covariance matrices that show up
in textbook portfolio theory (n is typically 2..20 assets).

### Two-asset closed form vs. N-asset Lagrangian

The two-asset case is a single-parameter curve indexed by the weight $w$
on asset A; the frontier point and the minimum-variance weight both have
explicit formulas. The N-asset case uses the Lagrangian system with the
standard scalars $a, b, c, d$ — see the derivations in
[`book/chapters/ch11.tex`](../../book/chapters/ch11.tex).

### CAPM without OLS

CAPM beta is a regression coefficient, but for the closed-form
definition $\beta = \text{Cov}(R_i, R_m) / \text{Var}(R_m)$ we do not need
to invoke the full OLS machinery from `quant-timeseries`. A direct
ratio of sums of products is sufficient and avoids the design-matrix
plumbing. The $(n-1)$ denominators in sample covariance and sample
variance cancel, so the estimator is invariant to the
population-vs-sample convention.

### Historical VaR / CVaR

Both are non-parametric: the empirical quantile uses linear interpolation
between adjacent order statistics (NumPy default convention). The
$10^{-12}$ slack in the CVaR tail filter guards against floating-point
ties at the boundary. CVaR is coherent (sub-additive, convex); VaR is
not.

## Error model

`PortfolioError` has five variants — see [`error.rs`](error.rs) for the
exact messages:

- `InvalidParam(String)`
- `SingularCovariance(String)`
- `InsufficientData { required, actual }`
- `InfeasibleTarget { target, lo, hi }`
- `DimensionMismatch(String)`

## Test contract

| Suite | File | Count |
|---|---|---|
| Unit (per module) | `src/*.rs` `mod tests` | 30 |
| Integration (public API) | `tests/portfolio_tests.rs` | 16 |
| Doc test | `lib.rs` | 1 |

Total: 47 tests, all passing. Clippy clean with `-D warnings`.