# quant-vol source modules

Phase 8 of the quant-finance curriculum: hand-rolled volatility models —
exponentially weighted moving average (EWMA/RiskMetrics), ARCH(q) (Engle 1982),
and GARCH(p, q) (Bollerslev 1986). Maximum-likelihood fitting uses a hand-rolled
Nelder-Mead simplex search and a coordinate-ascent multi-start optimiser. No
`argmin`, `optimization`, `nalgebra`, or `statrs`.

## Module map

| Module | File | Responsibility |
|---|---|---|
| `error` | `error.rs` | `VolError` enum (InvalidParam, InsufficientData, ConvergenceFailure, NonStationary) |
| `ewma` | `ewma.rs` | `ewma_vol` — exponentially weighted moving average variance |
| `arch` | `arch.rs` | `ArchModel` — ARCH(q) with MLE fit, forecast, log-likelihood |
| `garch` | `garch.rs` | `GarchModel` — GARCH(p,q) with MLE fit, forecast, persistence, half-life |
| `nelder_mead` | `nelder_mead.rs` | `maximize` (Nelder-Mead), `coordinate_ascent`, `multistart` |

`lib.rs` re-exports the public surface so callers can write
`use quant_vol::{ewma_vol, ArchModel, GarchModel, VolError};` without navigating
the module tree. The optimisers in `nelder_mead` are `pub` because `arch` and
`garch` import them directly.

## Design principles

- **Hand-rolled optimisation.** No `argmin`, `optimization`, or `nalgebra`.
  `nelder_mead::maximize` implements the classic Nelder-Mead simplex reflection/
  expansion/contraction/shrink loop. `coordinate_ascent` is a coordinate-wise
  hill climber with adaptive step halving — more robust than Nelder-Mead for
  the low-dimensional (2..4 parameter) MLE problems here. `multistart` runs
  coordinate ascent from `x0` plus five perturbations and keeps the best.
- **Maximise, do not minimise.** All optimisers *maximise* the objective. The
  MLE closures pass `model.log_likelihood(returns)` directly — there is no
  negation. (Negating the LL and feeding it to a maximiser would find the
  *worst* model, a subtle bug that bit an earlier iteration of this crate.)
- **Constrained MLE via parameter transformation.** GARCH parameters must
  satisfy `omega > 0`, `alpha_i >= 0`, `beta_j >= 0`, and
  `sum(alpha) + sum(beta) < 1` (covariance stationarity). We optimise an
  unconstrained vector `t` and map it to the constrained space:
  - `omega = exp(t0)` (positivity)
  - `alpha_i = sigmoid(t_i) * 0.49 / q` (each in `(0, 0.49/q)`)
  - `beta_j = sigmoid(t_j) * (0.998 - sum(alpha)) / p` (persistence < 0.998)
  Transformed parameters are clamped to `[-30, 30]` before `exp`/`sigmoid` to
  prevent overflow. ARCH uses a softmax transform: `alpha_i = exp(t_i) / (1 + sum(exp))`.
- **No panics in library paths.** All fallible functions return
  `Result<_, VolError>`. Invalid parameters, too-short series, and convergence
  failures are reported as typed errors.
- **Warmup skipping in GARCH LL.** The first `max(p, q)` observations are
  initialised to the sample variance and excluded from the log-likelihood sum
  to avoid contamination from the arbitrary initialisation. ARCH does not skip
  (its recursion handles short prefixes by zero-padding).
- **Variance clamping.** `conditional_variances` clamps each `sigma_t^2` to
  `[1e-300, 1e15]` to prevent overflow in long recursions. `log_likelihood`
  returns `-infinity` on any non-finite or non-positive variance.

## Error model

`VolError` (via `thiserror`) has four variants:

| Variant | When |
|---|---|
| `InvalidParam(String)` | `p == 0` or `q == 0` in GARCH, `q == 0` in ARCH, lambda outside `[0, 1]` in EWMA |
| `InsufficientData { required, actual }` | Series shorter than `p + q + 3` (GARCH) or `q + 3` (ARCH) or empty (EWMA) |
| `ConvergenceFailure { iterations }` | Optimiser fails to converge within `max_iter` |
| `NonStationary { persistence }` | Fitted model has `persistence >= 1` |

## Optimisation notes

The Nelder-Mead simplex search (`maximize`) is the textbook algorithm with
coefficients `alpha = 1` (reflection), `gamma = 2` (expansion),
`rho = 0.5` (contraction), `sigma = 0.5` (shrink). Convergence is declared when
both the simplex size and the function spread fall below `tol`.

For GARCH(1,1) MLE, coordinate ascent with multi-start proved more reliable
than Nelder-Mead: the simplex can degenerate near the constraint boundary,
while coordinate ascent with step halving tracks the ridge between `alpha` and
`beta` cleanly. `multistart` perturbs the initial guess by
`[0.1, -0.1, 0.3, -0.3, 0.5]` and keeps the best result, which is enough to
escape the occasional local plateau.

## Dependencies

- `quant-core` — `XorShift64`, `Normal`, `Distribution` (used by tests and
  examples to generate GARCH(1,1) return series)
- `thiserror` (derive error types)
- Dev: `approx` (float comparisons), `quant-core`

No `argmin`, `optimization`, `nalgebra`, `statrs`, or `rand`. The crate is
offline and synthetic.

## Run

```bash
cargo test -p quant-vol
cargo clippy -p quant-vol --all-targets -- -D warnings
cargo run -p quant-vol --example vol_demo
cargo run -p quant-vol --example vol_clustering
```