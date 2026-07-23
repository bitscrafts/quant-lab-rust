# quant-core source modules

Phase 6 of the quant-finance curriculum: the foundations — series types,
moments, rolling windows, and deterministic simulation. All math is
hand-rolled (no `rand`, `nalgebra`, `statrs`).

## Module map

| Module | File | Responsibility |
|---|---|---|
| `error` | `error.rs` | `CoreError` enum (InvalidPrice, InsufficientData, InvalidWindow) |
| `series` | `series.rs` | `PriceSeries` newtype, `simple_returns`, `log_returns` |
| `moments` | `moments.rs` | `mean`, sample `variance`/`std_dev`, `skewness`, `excess_kurtosis`, `Moments` trait |
| `rolling` | `rolling.rs` | generic `rolling(window, data, f)`, `rolling_mean`, `rolling_std_dev`, `RollingWindow` trait |
| `sim` | `sim.rs` | `Rng` trait, `XorShift64`, `Distribution` trait, `Normal` (Box-Muller), `gbm_paths` |

`lib.rs` re-exports the public surface so callers can write
`use quant_core::{XorShift64, Normal, Distribution, gbm_paths};` without
navigating the module tree.

## Design principles

- **Hand-rolled math.** No `rand`, `nalgebra`, `statrs`, or statistics crates.
  The pedagogy is in the implementation.
- **No panics on degenerate input.** Functions return `Result` with explicit
  `CoreError` variants; `mean` returns `0.0` for empty input rather than
  dividing by zero.
- **Trait-first architecture.** `Moments`, `RollingWindow`, `Rng`, and
  `Distribution` are defined before concrete types, per the workspace-wide
  trait-based design rule. `Moments` is implemented for `&[f64]`, `Vec<f64>`,
  and `PriceSeries`; `Rng` is implemented for `XorShift64`; `Distribution` is
  implemented for `Normal`.
- **Population central moments for higher moments.** `variance` uses the
  sample (n - 1) estimator, consistent with `qf_common` and `qf-04-returns`.
  `skewness` and `excess_kurtosis` use population central moments (denominator
  n), the textbook definition; the difference vanishes for large n.
- **Deterministic simulation.** `XorShift64` is seedable and reproducible. A
  seed of 0 is replaced with a fixed default to avoid the degenerate zero
  stream. GBM uses the exact solution `S_t = S_0 * exp((mu - sigma^2/2) dt +
  sigma sqrt(dt) Z)`.

## Dependencies

- `thiserror` (derive error types)
- Dev: `approx` (float comparisons in tests)

No `rand`, `nalgebra`, `statrs`. The entire crate is offline and synthetic.

## Run

```bash
cargo test -p quant-core
cargo clippy -p quant-core --all-targets -- -D warnings
cargo run -p quant-core --example fat_tails
```