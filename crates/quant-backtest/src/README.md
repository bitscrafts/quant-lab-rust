# src/ — quant-backtest source

[← back to crate README](../README.md)

## Module Map

| File | Purpose | Key types / functions |
|---|---|---|
| `lib.rs` | Crate root, re-exports | `pub use` of all public APIs |
| `error.rs` | Error enum | `BacktestError` (InvalidConfig, InsufficientData, DimensionMismatch, InvalidEvent) |
| `triple_barrier.rs` | Triple-barrier labeling | `TripleBarrierConfig`, `TripleBarrierLabel` (Upper/Lower/Time), `LabeledEvent`, `triple_barrier_label`, `to_binary_label_helper` |
| `weights.rs` | Sample weights from uniqueness | `concurrent_events`, `average_uniqueness`, `sample_weights` |
| `purged_kfold.rs` | Purged k-fold CV with embargo | `PurgedKFoldConfig`, `PurgedSplit`, `purged_kfold_splits`, `event_overlaps` |
| `kelly.rs` | Kelly criterion bet sizing | `kelly_fraction`, `fractional_kelly`, `kelly_from_returns`, `compute_position_size`, `PositionSize` |
| `backtest.rs` | End-to-end AFML pipeline | `AfmlBacktestConfig`, `AfmlBacktestResult`, `BetSizing`, `afml_backtest`; private `sharpe`, `max_drawdown` helpers |

## Architecture

```
afml_backtest(prices, config)
  1. validate config
  2. generate entry indices at every `entry_step` bars
  3. triple_barrier_label  -> Vec<LabeledEvent>
  4. sample_weights        -> Vec<f64>           (if use_weights)
  5. purged_kfold_splits   -> Vec<PurgedSplit>
  6. compute_position_size -> PositionSize       (from trade returns)
  7. size_per_trade        -> Vec<f64>            (BetSizing rule)
  8. equity curve          -> compounded returns  -> total_return, max_drawdown
  9. in_sample_sharpe      -> sharpe(trade_returns)
 10. out_of_sample_sharpe  -> mean of per-fold test Sharpes
```

## Testing

- Unit tests in each module file (`#[cfg(test)] mod tests`)
- Integration contract in `tests/backtest_tests.rs` (15 tests)
- 34 tests total, all passing; `cargo clippy -D warnings` clean

## Design Decisions

### Why triple-barrier rather than fixed-horizon labels?

A fixed-horizon label depends only on the price at $t + h$; two
trades with very different paths (one volatile, one smooth) can
share a label. The triple-barrier label depends on the whole path:
which barrier was touched first, and when. The holding period
becomes a random variable, not a constant.

### Why inverse-concurrency weights?

If three events overlap on a bar, each one contributes a third of
the unique information there. Summing $1/c_t$ over the event
duration and taking the mean averages that discount across the
event's life. Normalising to sum to one makes the weights a
probability distribution over the sample.

### Why purge + embargo?

Purging removes training events whose `[entry, exit]` interval
intersects the test fold --- their labels depend on test-window
prices. Embargo additionally removes events whose entry falls in
`[t_end, t_end + embargo)` to handle autocorrelated features that
would otherwise carry test-window information into training. Set
the embargo to the lag at which the ACF of the features drops below
significance.

### Why hand-rolled Kelly?

The Kelly criterion is one line: $f^* = p - q/b$. Estimating $p$
and $b$ from a trade-return stream is a few more lines of
arithmetic. Pulling in an optimisation library for this would
obscure the math and add a dependency. Half Kelly is the
production default: same growth rate to first order, half the
variance of log-wealth.

### Why no look-ahead in features?

Triple-barrier labels use future prices by design --- they are the
target. Features must be computed strictly from information
available at the entry index. The discipline is: train on labels
that depend on the future, predict them from features that do not.