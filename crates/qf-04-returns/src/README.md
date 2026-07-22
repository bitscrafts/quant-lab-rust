# qf-04-returns source modules

Phase 4 of the quant-finance curriculum: returns, volatility, and
risk-adjusted metrics, implemented from first principles (no external
statistics crates).

## Module map

| Module | File | Responsibility |
|---|---|---|
| `error` | `error.rs` | `ReturnsError` enum for structurally invalid inputs |
| `returns` | `returns.rs` | Simple, log, and cumulative returns; `Returns` trait |
| `volatility` | `volatility.rs` | Sample std, annualised, and rolling volatility |
| `risk` | `risk.rs` | Sharpe, annualised Sharpe, and Sortino ratios |
| `drawdown` | `drawdown.rs` | Drawdown series, max drawdown, `DrawdownStats` |

`lib.rs` re-exports the public surface so callers can write
`use qf_04_returns::sharpe_ratio;` without navigating the module tree.

## Design principles

- **Hand-rolled math.** No `rand`, `nalgebra`, `statrs`, or statistics crates.
  The pedagogy is in the implementation.
- **No panics on degenerate input.** Empty slices, single elements, and
  division-by-zero cases return `0.0` rather than panicking.
- **Trait-first architecture.** The `Returns` trait abstracts over
  `&[f64]`, `Vec<f64>`, and `Vec<Ohlcv>`, following the workspace-wide rule
  that every struct should implement a defined capability.
- **Consistency with `qf_common`.** `volatility` uses the same sample
  standard deviation (`n - 1` denominator) as `qf_common::compute_stats`.

## Dependencies

- `qf_common` (error types)
- `qf_03_stocks` (`Ohlcv` for the `Returns` impl on OHLCV data)
- `thiserror` (derive error types)