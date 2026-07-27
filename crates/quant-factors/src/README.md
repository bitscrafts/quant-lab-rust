# quant-factors/src

Source layout for the `quant-factors` crate (Phase 12 of the
quant-finance curriculum). All math is hand-rolled — no `argmin`,
`optimization`, `nalgebra`, or `statrs`.

## Module map

| File | Purpose |
|---|---|
| [`lib.rs`](lib.rs) | Crate root, re-exports, doc example |
| [`error.rs`](error.rs) | `FactorError` enum (6 variants) |
| [`eigen.rs`](eigen.rs) | `power_method`, `deflate`, `top_k_eigen` (power method with sign alignment + deflation) |
| [`pca.rs`](pca.rs) | `PcaResult`, `pca`, `pca_transform`, `pca_reconstruct` (covariance eigendecomposition) |
| [`fama_french.rs`](fama_french.rs) | `FF3Exposure`, `ff3_regression` (reuses OLS from quant-timeseries) |
| [`risk.rs`](risk.rs) | `RiskAttribution`, `risk_attribution` (systematic + idiosyncratic decomposition) |

## API at a glance

```rust
// Eigenvalues
let (lambda, v) = power_method(&matrix, 0, 0.0)?;
let a_def = deflate(&matrix, lambda, &v);
let (eigs, vecs) = top_k_eigen(&matrix, k)?;

// PCA
let res = pca(&returns, n_components)?;
let scores = pca_transform(&returns, &res.eigenvectors, &res.mean);
let recon = pca_reconstruct(&scores, &res.eigenvectors, &res.mean);

// Fama-French 3-factor
let ff = ff3_regression(&asset_excess, &factors)?;

// Risk attribution
let ra = risk_attribution(&weights, &loadings, &factor_cov, &resid)?;
```

## Design notes

### Power method sign alignment

The power method can oscillate (delta = 2.0) when the iterate flips
sign between steps. The fix is to align each new iterate with its
predecessor: if `dot(v_new, v_old) < 0`, negate `v_new` before the
convergence test.

### Deflation error

Each deflation `A' = A - lambda * v * v'` introduces floating-point
error. The top 2-3 eigenpairs are accurate to 1e-6; smaller
eigenvalues may carry relative error of 1e-3 or worse.

### EVR via covariance trace

`EVR_i = lambda_i / trace(Sigma)` where the trace is the sum of the
diagonal of the full covariance matrix. When `n_components < n`, the
ratios correctly sum to less than 1.0 (fraction of TOTAL variance).

### FF3 reuses OLS

The FF3 regression builds the design matrix `[1, Mkt-Rf, SMB, HML]`
and calls `quant_timeseries::ols()`. No new regression code needed.

### Risk decomposition

`sigma_p^2 = w' B Sigma_F B' w + w' D w` (systematic + idiosyncratic).
The portfolio factor exposure is `f_p = B' w` (K-vector); the
systematic variance is `f_p' Sigma_F f_p`; the per-factor contribution
is the diagonal approximation `f_p[k]^2 * Sigma_F[k][k]`.

## Error model

`FactorError` has six variants — see [`error.rs`](error.rs):

- `InvalidParam(String)`
- `NonConverged(usize, f64)` — power method did not converge
- `Singular(String)`
- `InsufficientData { required, actual }`
- `DimensionMismatch(String)`
- `Infeasible(String)`

## Test contract

| Suite | File | Count |
|---|---|---|
| Unit (per module) | `src/*.rs` `mod tests` | 12 |
| Integration (public API) | `tests/factor_tests.rs` | 15 |
| Doc test | `lib.rs` | 1 |

Total: 28 tests, all passing. Clippy clean with `-D warnings`.