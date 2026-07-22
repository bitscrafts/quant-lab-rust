# qf-04-returns — Returns & Volatility

Phase 4 of the quant-finance curriculum: the foundational vocabulary of
quantitative finance — returns, volatility, risk-adjusted metrics, and
drawdowns, implemented from first principles in Rust.

## Overview

This crate introduces:

- **Returns**: simple (arithmetic) and log (continuously compounded), plus
  running cumulative returns
- **Volatility**: sample standard deviation, annualised, and rolling
- **Risk-adjusted metrics**: Sharpe and Sortino ratios (annualised variants
  included)
- **Drawdown analysis**: drawdown series, max drawdown, and `DrawdownStats`
- **`Returns` trait**: abstracts over `&[f64]`, `Vec<f64>`, and `Vec<Ohlcv>`
  (uses closing prices)

All math is hand-rolled (no external statistics crates), following the
pedagogy-first policy of the `quant-lab` workspace. Division-by-zero and
degenerate-input cases return `0.0` rather than panicking.

## Quick start

```bash
cargo test -p qf-04-returns
cargo run -p qf-04-returns --example returns_analysis
```

## API

### Returns

```rust
use qf_04_returns::{simple_returns, log_returns, cumulative_returns, Returns};

let prices = vec![100.0, 102.0, 101.0, 105.0];
let r = simple_returns(&prices);       // [0.02, -0.0098, 0.0396]
let lr = log_returns(&prices);          // [ln(1.02), ln(101/102), ln(105/101)]
let cum = cumulative_returns(&r);       // running compounded return

// Trait form works on slices, Vec<f64>, and Vec<Ohlcv>:
let r2 = prices.simple_returns();
```

### Volatility

```rust
use qf_04_returns::{volatility, annualized_volatility, rolling_volatility};

let vol = volatility(&r);                       // sample std (n-1)
let ann = annualized_volatility(&r, 252.0);       // vol * sqrt(252)
let rv = rolling_volatility(&r, 20);             // 20-period rolling std
```

### Risk-adjusted metrics

```rust
use qf_04_returns::{sharpe_ratio, annualized_sharpe, sortino_ratio};

let sharpe = sharpe_ratio(&r, 0.0);               // (mean - rf) / vol
let ann_sharpe = annualized_sharpe(&r, 0.0, 252.0); // sharpe * sqrt(252)
let sortino = sortino_ratio(&r, 0.0);              // downside-only denominator
```

### Drawdown

```rust
use qf_04_returns::{drawdown, max_drawdown, drawdown_stats};

let dd = drawdown(&prices);            // always <= 0
let md = max_drawdown(&prices);         // most negative value
let stats = drawdown_stats(&prices);    // max_drawdown, duration, recovery
```

## Design principles

- **Trait-first architecture.** Before adding a struct or function, define the
  capability it provides. The `Returns` trait abstracts return computation
  across price sources.
- **Consistency with `qf_common`.** `volatility` uses the sample standard
  deviation (`n - 1` denominator) matching `qf_common::compute_stats`.
- **No panics.** Empty input, single elements, and zero denominators return
  `0.0`.
- **Hand-rolled math.** No `rand`, `nalgebra`, `statrs`, or statistics crates.

## Module map

See [`src/README.md`](src/README.md) for the per-module breakdown.

| Module | Responsibility |
|---|---|
| `returns` | Simple, log, cumulative returns; `Returns` trait |
| `volatility` | Sample std, annualised, rolling volatility |
| `risk` | Sharpe, annualised Sharpe, Sortino |
| `drawdown` | Drawdown series, max drawdown, `DrawdownStats` |
| `error` | `ReturnsError` for structurally invalid inputs |

## Testing

The test suite (25 tests + 1 doctest) covers:

- Returns: basic, empty, single, log/simple relationship, cumulative compounding
- Volatility: sample std, constant, empty, annualisation, rolling (valid + invalid)
- Risk: positive/negative/zero Sharpe, annualised Sharpe, Sortino > Sharpe
- Drawdown: basic, always-non-positive, max drawdown, no-decline, recovery
- Trait: `&[f64]`, `Vec<Ohlcv>` (closing prices)

```bash
cargo test -p qf-04-returns
cargo clippy -p qf-04-returns --all-targets -- -D warnings
```

## Dependencies

- `qf_common` (error types)
- `qf_03_stocks` (`Ohlcv` for the `Returns` impl)
- `thiserror` (derive error types)

Dev: `approx` (float assertions).

## Integration

Part of the `quant-lab` workspace:

```
crates/quant-lab/
├── crates/
│   ├── qf-common/        # Shared utilities
│   ├── qf-01-fraud/      # Phase 1: Fraud detection
│   ├── qf-02-loan/       # Phase 2: Loan risk
│   ├── qf-03-stocks/     # Phase 3: Stock analysis
│   └── qf-04-returns/    # Phase 4: Returns (this crate)
└── data/                 # Sample datasets
```

## Next steps

Phase 5 builds on this foundation with the first backtest: SMA crossover
strategy, P&L accounting, and drawdown-based risk evaluation.

## License

MIT