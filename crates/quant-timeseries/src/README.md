# quant-timeseries source modules

Phase 7 of the quant-finance curriculum: hand-rolled time-series econometrics
— OLS regression via Gaussian elimination, the autocorrelation function, the
Augmented Dickey-Fuller test for stationarity, and López de Prado's fixed-width
fractional differentiation. All math is hand-rolled (no `nalgebra`, `statrs`).

## Module map

| Module | File | Responsibility |
|---|---|---|
| `error` | `error.rs` | `TimeSeriesError` enum (Singular, DimensionMismatch, InvalidLag, InvalidParam, InsufficientData) |
| `ols` | `ols.rs` | `OlsFit`, `ols`, `gauss_solve` (pub crate), `inverse_diagonal_sqrt` |
| `acf` | `acf.rs` | `acf` — autocorrelation for lags 0..=max_lag |
| `adf` | `adf.rs` | `AdfResult`, `adf_test`, `MACKINNON_5PCT = -2.86` |
| `fracdiff` | `fracdiff.rs` | `ffd_weights`, `frac_diff`, `find_min_d` |

`lib.rs` re-exports the public surface so callers can write
`use quant_timeseries::{ols, acf, adf_test, frac_diff, find_min_d};` without
navigating the module tree. `gauss_solve` is intentionally kept `pub(crate)`
because it is an internal numerical helper.

## Design principles

- **Hand-rolled linear algebra.** No `nalgebra`, `ndarray`, or `statrs`. OLS
  forms the normal equations `X'X β = X'y` and solves them with Gaussian
  elimination and partial pivoting. The diagonal of `(X'X)^{-1}` (needed for
  standard errors) is recovered by re-solving `X'X e_j = unit_j` per column —
  `O(k^4)` overall, fine for the small `k` (2..5) we fit here.
- **No panics in library paths.** All fallible functions return
  `Result<_, TimeSeriesError>`. Dimension mismatches, collinearity
  (`TimeSeriesError::Singular`), too-short series, and invalid parameters are
  reported as typed errors.
- **Trait-free pedagogy.** Unlike `quant-core`, this crate favors concrete
  functions over traits because the operations are standalone rather than
  polymorphic over types. `Rng` and `Distribution` from `quant-core` are
  reused for test/example data generation.
- **Fixed-width FFD.** Weights follow the recurrence
  `w_0 = 1`, `w_k = -w_{k-1} (d - k + 1) / k`. The loop breaks *before* pushing
  a weight whose absolute value falls below `threshold`, so `d = 0` yields
  `[1.0]` and `d = 1` yields `[1.0, -1.0]` exactly — matching the binomial
  interpretation of `(1 - B)^d`.
- **MacKinnon critical value.** Hardcoded `-2.86` for the constant-no-trend
  ADF specification at 5% significance, per MacKinnon (1996). The test rejects
  the unit-root null when `statistic < critical_value`.
- **Binary-search `find_min_d`.** Endpoints `d = 0` and `d = 1` are tested
  first; if `d = 0` already passes, return `0.0`; if `d = 1` fails, return
  `1.0` conservatively. Otherwise binary search narrows `hi - lo` below the
  tolerance and returns `hi`.

## Error model

`TimeSeriesError` (via `thiserror`) has five variants:

| Variant | When |
|---|---|
| `Singular` | Zero pivot encountered in Gaussian elimination (collinear X) |
| `DimensionMismatch { x_rows, y_len }` | `x.len() != y.len()` or ragged rows |
| `InvalidLag { lag, len }` | `max_lag >= data.len()` in `acf` |
| `InvalidParam(String)` | `d` outside `[0, 2]`, non-positive threshold/tolerance |
| `InsufficientData { required, actual }` | Too few observations for the regression |

## Dependencies

- `quant-core` — `XorShift64`, `Normal`, `Distribution`, `gbm_paths` (used by
  tests and examples to generate stationary and non-stationary series)
- `thiserror` (derive error types)
- Dev: `approx` (float comparisons), `quant-core`

No `nalgebra`, `statrs`, `ndarray`. The crate is offline and synthetic.

## Run

```bash
cargo test -p quant-timeseries
cargo clippy -p quant-timeseries --all-targets -- -D warnings
cargo run -p quant-timeseries --example stationarity
cargo run -p quant-timeseries --example ffd_demo
```