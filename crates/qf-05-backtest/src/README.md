# qf-05-backtest source modules

Phase 5 of the quant-finance curriculum: backtesting basics — signal
generation, position management, P&L accounting with transaction costs, and
performance evaluation reusing `qf-04-returns`.

## Module map

| Module | File | Responsibility |
|---|---|---|
| `error` | `error.rs` | `BacktestError` enum for invalid params, data, or config |
| `signal` | `signal.rs` | `Signal` (Buy/Sell/Hold), `Position` (Long/Short/Flat) |
| `strategy` | `strategy.rs` | `Strategy` trait, `SmaCrossover`, `BuyAndHold` benchmark |
| `backtest` | `backtest.rs` | `BacktestConfig`, `Trade`, `BacktestResult`, `run_backtest` |

`lib.rs` re-exports the public surface so callers can write
`use qf_05_backtest::run_backtest;` without navigating the module tree.

## Design principles

- **No look-ahead.** `Strategy::signal(data, index)` is only ever called with
  `data[..=index]`. Implementations must never read past the current bar.
- **Long-only for Phase 5.** `Short` is declared in `Position` so future phases
  can extend the engine without breaking the public API, but the engine
  rejects `allow_short = true` in `BacktestConfig::validate`.
- **Transaction costs on every state change.** Entry and exit each pay
  `transaction_cost * notional`. The engine never pyramids into an existing
  position (a `Buy` while long is a no-op).
- **Mark-to-market equity curve.** Equity is `shares * close` when long and
  cash when flat, so the per-bar `daily_returns` (computed by
  `qf_04_returns::simple_returns`) reflect real performance.
- **Reuse, don't reinvent.** Sharpe, Sortino, drawdown, and volatility are
  delegated to `qf-04-returns` via `BacktestResult` helper methods, per the
  workspace anti-NIH rule.
- **Trait-first.** `Strategy` abstracts over any signal generator; the engine
  is generic over `S: Strategy`, so new strategies drop in without engine
  changes.

## Dependencies

- `qf_common` (error types)
- `qf_03_stocks` (`Ohlcv` bars)
- `qf_04_returns` (Sharpe, Sortino, drawdown, volatility, `simple_returns`)
- `thiserror` (derive error types)

## Run

```bash
cargo test -p qf-05-backtest
cargo clippy -p qf-05-backtest --all-targets -- -D warnings
cargo run -p qf-05-backtest --example backtest_demo
```