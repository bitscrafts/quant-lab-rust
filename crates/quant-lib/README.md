# quant-lib

Unified quantitative finance library - a non-destructive facade consolidating all Phase 6-14.5 crates.

## Overview

`quant-lib` provides a single dependency for accessing the entire quant-finance curriculum. It re-exports the public APIs of:

| Module | Source Crate | Content |
|--------|--------------|---------|
| `core` | quant-core | Moments, rolling windows, RNG, GBM |
| `traits` | quant-core | 11 pluggable component traits (Phase 14.5) |
| `risk` | quant-core | PSR, DSR, Calmar, Omega, IR, Ulcer Index |
| `timeseries` | quant-timeseries | OLS, ACF, ADF, frac-diff, CUSUM |
| `vol` | quant-vol | EWMA, ARCH, GARCH |
| `stochastic` | quant-stochastic | BM, GBM, Poisson, Monte Carlo |
| `options` | quant-options | Black-Scholes, Greeks, implied vol |
| `portfolio` | quant-portfolio | Markowitz, frontier, tangency, CAPM, VaR |
| `factors` | quant-factors | PCA, Fama-French, risk attribution |
| `microstructure` | quant-microstructure | LOB, OFI, market impact |
| `backtest` | quant-backtest | Triple-barrier, purged CV, Kelly, walk-forward |

## Non-Destructive Facade Principle

`quant-lib` adds **zero new logic**. Every line in `lib.rs` is `pub mod` or `pub use`. All math, types, and traits live in the source crates. The facade exists only to provide:

- Single dependency for downstream projects
- Unified namespace
- Feature flags for selective compilation
- Prelude for common imports

Source crates remain independently usable - you can depend on `quant-backtest` directly without `quant-lib`.

## Quick Start

```rust
use quant_lib::prelude::*;

let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
let m = quant_lib::core::mean(&data);
assert!(m.is_finite());
```

## Feature Flags

All modules are enabled by the `all` feature (on by default). Disable for smaller binaries:

```toml
[dependencies]
quant-lib = { version = "0.1", default-features = false, features = ["core", "backtest"] }
```

| Feature | Depends On | Enables |
|---------|------------|---------|
| `core` | - | core, traits, risk |
| `timeseries` | core | timeseries |
| `vol` | core, timeseries | vol |
| `stochastic` | core | stochastic |
| `options` | stochastic | options |
| `portfolio` | core, timeseries | portfolio |
| `factors` | core, timeseries, portfolio | factors |
| `microstructure` | core | microstructure |
| `backtest` | core, timeseries, portfolio | backtest |
| `all` | all of the above | everything |

## Traits (Phase 14.5)

The `traits` module provides 11 pluggable component traits:

```rust
use quant_lib::traits::{
    CrossValidator,           // Purged k-fold, walk-forward
    Labeler,                  // Triple-barrier, fixed-horizon, trend-scanning
    BetSizer,                 // Kelly, fixed, equal
    SampleWeighter,           // Sample weight computation
    StochasticProcess,        // GBM, Heston, Vasicek
    OptionPricer,             // Black-Scholes
    Greeks,                   // Delta, gamma, vega, theta, rho
    FactorModel,              // CAPM, multi-factor
    ImpactModel,              // Kyle model
    OrderBookOps,             // Order book operations
    StructuralBreakDetector,  // CUSUM
};
```

## Composable Backtesting

```rust
use quant_lib::backtest::{
    BacktestBuilder, WalkForward, WalkForwardConfig,
    FixedHorizonLabeler, KellyBetSizer,
};

let backtest = BacktestBuilder::new()
    .labeler(FixedHorizonLabeler::new(5, 0.01))
    .cross_validator(WalkForward::rolling(
        WalkForwardConfig { train_size: 100, test_size: 20, step: 10 }
    ))
    .bet_sizer(KellyBetSizer::new(0.5))  // Half-Kelly
    .build();
```

## Examples

16 examples demonstrating the library tour:

```bash
cargo run -p quant-lib --example 01_mean_variance
cargo run -p quant-lib --example 15_full_pipeline
cargo run -p quant-lib --example 16_trait_compose
```

| # | Example | Demonstrates |
|---|---------|--------------|
| 01 | mean_variance | Basic statistics |
| 02 | returns_basics | Return computation |
| 03 | rolling_window | Rolling statistics |
| 04 | random_walk | RNG and simulation |
| 05 | ols_regression | Time series regression |
| 06 | stationarity | ADF tests |
| 07 | volatility | GARCH models |
| 08 | options_greeks | Black-Scholes Greeks |
| 09 | portfolio_frontier | Efficient frontier |
| 10 | pca_factors | Factor models |
| 11 | microstructure_lob | Order book |
| 12 | triple_barrier | Event labeling |
| 13 | purged_cv | Cross-validation |
| 14 | kelly_sizing | Bet sizing |
| 15 | full_pipeline | End-to-end workflow |
| 16 | trait_compose | Pluggable components |

## Testing

```bash
# All 20 tests + 2 doc tests
cargo test -p quant-lib

# Verify non-destructive (source crates work independently)
cargo test -p quant-backtest
cargo test -p quant-core
```

## Related

- **Book chapter**: ch15.tex (quant-lib facade)
- **Phase 14.5 spec**: `specs/phase-14.5-trait-architecture.md`
- **Repository**: https://github.com/bitscrafts/quant-lab-rust
