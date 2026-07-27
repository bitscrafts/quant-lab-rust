# quant-factors

Factor attribution for the quant-finance curriculum (Phase 12).

PCA via the power method with deflation, the Fama-French 3-factor model,
and risk decomposition into systematic and idiosyncratic components.
All math is hand-rolled — no `argmin`, `optimization`, `nalgebra`, or
`statrs`. Eigenvalues come from `power_method` + `deflate`; the FF3
regression reuses the OLS machinery from `quant-timeseries`.

## Modules

| Module | Purpose |
|---|---|
| [`eigen`](src/eigen.rs) | Power method, deflation, top-k eigenpairs |
| [`pca`](src/pca.rs) | PcaResult, pca, pca_transform, pca_reconstruct |
| [`fama_french`](src/fama_french.rs) | FF3Exposure, ff3_regression |
| [`risk`](src/risk.rs) | RiskAttribution, risk_attribution |
| [`error`](src/error.rs) | FactorError |

## API surface

```rust
// Eigenvalues
let (lambda, v) = power_method(&matrix, 0, 0.0)?;
let a_def = deflate(&matrix, lambda, &v);
let (eigs, vecs) = top_k_eigen(&matrix, k)?;

// PCA
let res = pca(&returns, n_components)?;
// res.eigenvalues, res.eigenvectors, res.explained_variance_ratio,
// res.cumulative_variance, res.mean
let scores = pca_transform(&returns, &res.eigenvectors, &res.mean);
let recon = pca_reconstruct(&scores, &res.eigenvectors, &res.mean);

// Fama-French 3-factor
let ff = ff3_regression(&asset_excess, &factors)?;
// ff.alpha, ff.beta_mkt, ff.beta_smb, ff.beta_hml,
// ff.r_squared, ff.residual_var

// Risk attribution
let ra = risk_attribution(&weights, &loadings, &factor_cov, &resid)?;
// ra.total_variance, ra.systematic_variance,
// ra.idiosyncratic_variance, ra.factor_contributions
```

## Dependencies

- `quant-core` — Moments, RollingWindow
- `quant-timeseries` — OLS (reused for FF3 regression)
- `quant-portfolio` — CAPM beta/alpha for comparison
- `thiserror` — Error derive

## Test contract

| Suite | File | Count |
|---|---|---|
| Unit (per module) | `src/*.rs` `mod tests` | 12 |
| Integration (public API) | `tests/factor_tests.rs` | 15 |
| Doc test | `lib.rs` | 1 |

Total: 28 tests, all passing. Clippy clean with `-D warnings`.

## Verified results

### PCA (12 obs x 3 synthetic assets)

| PC | Eigenvalue | EVR | Cumulative |
|---|---|---|---|
| 1 | 1.96e-4 | 0.9608 | 0.9608 |
| 2 | 7.94e-6 | 0.0389 | 0.9997 |
| 3 | 7.00e-8 | 0.0003 | 1.0000 |

Reconstruction SSE: k=1 -> 8.81e-5, k=2 -> 7.17e-7, k=3 -> 4.08e-23.

### Fama-French 3-factor (100 synthetic observations)

DGP: alpha=0.001, beta_mkt=1.2, beta_smb=0.4, beta_hml=-0.3 + noise.

| Parameter | True | Estimated |
|---|---|---|
| alpha | 0.001 | 0.001003 |
| beta_mkt | 1.2 | 1.2008 |
| beta_smb | 0.4 | 0.4016 |
| beta_hml | -0.3 | -0.2977 |
| R^2 (FF3) | — | 0.9893 |
| R^2 (CAPM) | — | 0.9391 |

Improvement: 5.34% over single-factor CAPM.

### Risk attribution

| Component | Variance | % |
|---|---|---|
| Total | 8.94e-5 | 100% |
| Systematic | 8.84e-5 | 98.9% |
| Idiosyncratic | 9.85e-7 | 1.1% |

## Design notes

### Power method with sign alignment

The power method oscillates (delta = 2.0) when the dominant eigenvalue
is negative or when the iterate flips sign between steps. The fix is
to align each new iterate with its predecessor before the convergence
test: if `dot(v_new, v_old) < 0`, negate `v_new`. This makes the
residual `||v_new - v_old||` monotone non-increasing.

### Deflation error accumulation

Deflation removes the found eigenpair by subtracting `lambda * v * v'`
from the matrix. Each deflation introduces floating-point error, so
the smaller eigenvalues are less accurate. For a near-singular
covariance, the last eigenvalue may carry relative error of 1e-3 or
worse. For the top 2-3 eigenpairs of a well-conditioned matrix, the
accuracy is 1e-6 or better.

### EVR via the covariance trace

The explained variance ratio uses `lambda_i / trace(Sigma)` where the
trace is the sum of the diagonal of the full covariance matrix. This
is the sum of ALL eigenvalues (not just the retained ones), so when
`n_components < n` the ratios correctly sum to less than 1.0 — they
reflect the fraction of TOTAL variance captured, not just the fraction
among the retained components.

### FF3 reuses OLS

The Fama-French regression builds the design matrix `[1, Mkt-Rf, SMB,
HML]` and calls the `ols()` function from `quant-timeseries`. This
reuses the Gaussian-elimination solver, standard errors, t-stats, and
R^2 — no new regression code is needed.

## References

- Fama, E. F. and French, K. R. (1992). "The Cross-Section of Expected
  Stock Returns." *Journal of Finance* 47(2), 427-465.
- Golub, G. H. and Van Loan, C. F. (2013). *Matrix Computations*,
  4th ed. Johns Hopkins University Press. (Power method, deflation.)
- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed.
  Springer.