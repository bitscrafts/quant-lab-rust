# quant-backtest

AFML backtesting framework: triple-barrier labeling, sample weights
from event uniqueness, purged k-fold cross-validation with embargo,
Kelly criterion bet sizing, and an end-to-end backtest pipeline.
Phase 14 crate of the quant-finance curriculum.

[← back to quant-lab](../README.md)

## Overview

`quant-backtest` implements the López de Prado (2018) AFML framework
that bridges academic portfolio theory to production strategy
research. Traditional backtests leak information in three ways: fixed
horizon labels ignore the path, overlapping events share information,
and standard k-fold CV trains on test data. The AFML framework fixes
all three:

- **Triple-barrier labeling** produces path-dependent labels based on
  the first barrier touched (profit-taking, stop-loss, or time).
- **Sample weights** discount overlapping events by the inverse of
  their average concurrency.
- **Purged k-fold CV** removes training samples overlapping the test
  fold; an optional embargo period additionally removes samples
  whose entry follows the test window.
- **Kelly criterion** sizes bets by the edge divided by the odds.

All math is hand-rolled --- no external ML or optimization crates.

## Modules

| Module | Public API |
|---|---|
| `triple_barrier` | `TripleBarrierConfig`, `TripleBarrierLabel`, `LabeledEvent`, `triple_barrier_label`, `to_binary_label_helper` |
| `weights` | `concurrent_events`, `average_uniqueness`, `sample_weights` |
| `purged_kfold` | `PurgedKFoldConfig`, `PurgedSplit`, `purged_kfold_splits`, `event_overlaps` |
| `kelly` | `kelly_fraction`, `fractional_kelly`, `kelly_from_returns`, `compute_position_size`, `PositionSize` |
| `backtest` | `AfmlBacktestConfig`, `AfmlBacktestResult`, `BetSizing`, `afml_backtest` |
| `error` | `BacktestError` |

## Dependencies

- `quant-core` --- shared error and numeric traits
- `quant-timeseries` --- fractional differentiation
- `quant-portfolio` --- risk metrics
- `thiserror` --- derive `Error`

Dev dependencies: `approx`, `quant-core`.

## Usage

```rust
use quant_backtest::{AfmlBacktestConfig, BetSizing, PurgedKFoldConfig,
    TripleBarrierConfig, afml_backtest};

let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01)).collect();
let config = AfmlBacktestConfig {
    barrier_config: TripleBarrierConfig {
        upper_barrier: 0.02, lower_barrier: -0.02,
        time_barrier: 5, min_return: 0.0,
    },
    cv_config: PurgedKFoldConfig { n_folds: 3, embargo: 2 },
    use_weights: true,
    bet_sizing: BetSizing::Equal,
    entry_step: 3,
};
let result = afml_backtest(&prices, &config).unwrap();
assert!(!result.events.is_empty());
```

## Examples

- `triple_barrier` --- triple-barrier labeling on a 250-bar LCG random
  walk, prints label distribution and per-event returns
- `purged_cv` --- 200 events over 1000 bars, 5 folds with embargo=10,
  prints per-fold train/test/purged/embargoed counts and verifies no
  leakage
- `kelly_sizing` --- full and half Kelly on three synthetic streams
  (60%/1:1, 55%/2:1, 50%/1:1), compares closed form to
  `kelly_from_returns`

## Tests

- 19 unit tests across `triple_barrier`, `weights`, `purged_kfold`,
  `kelly`, `backtest`, `error`
- 15 integration tests in `tests/backtest_tests.rs` (TDD contract)
- Total: 34 tests, all passing; clippy clean

## Design Notes

- **Path-dependent labels**: `label_one` walks the price path from the
  entry index and returns on the first barrier hit; the time barrier
  is the catch-all when neither horizontal barrier is touched.
- **Inverse concurrency weights**: `sample_weights` discounts each
  event by the inverse of the average number of concurrent events
  during its lifetime, then normalises to sum to one.
- **Purge + embargo**: purging removes events whose `[entry, exit]`
  interval intersects the test fold; embargo additionally removes
  events whose entry falls in `[t_end, t_end + embargo)` to handle
  autocorrelation.
- **Kelly closed form**: `f* = p - q/b` with `q = 1 - p`,
  `b = mean_win / mean_loss`. Half Kelly is the production default.
- **No look-ahead in features**: triple-barrier labels intentionally
  use future prices (they are the target). Features must be
  computed strictly from information available at the entry index.

## Related Crates

- [`quant-portfolio`](../quant-portfolio/): Markowitz frontier, CAPM,
  VaR/CVaR (risk metrics and position sizing inputs)
- [`quant-timeseries`](../quant-timeseries/): fractional
  differentiation (AFML-recommended stationarity transform for
  features)
- [`quant-factors`](../quant-factors/): PCA and Fama-French (signal
  generators that feed into the backtest pipeline)

## References

- López de Prado, M. (2018). _Advances in Financial Machine
  Learning_. Wiley. Chapters 3 (triple-barrier), 4 (sample weights),
  7 (cross-validation), 10 (bet sizing).
- Kelly, J. L. (1956). _A New Interpretation of Information Rate_.
  Bell System Technical Journal.