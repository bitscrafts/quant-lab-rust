# quant-lib/src

Source code for the quant-lib facade crate.

## Module Structure

```
src/
├── lib.rs      # Main facade: pub mod + pub use re-exports only
└── prelude.rs  # Curated common imports
```

## Non-Destructive Design

**CRITICAL**: This crate contains ZERO new logic.

- `lib.rs` contains only `pub mod` and `pub use` statements
- `prelude.rs` contains only `pub use` re-exports
- No `fn`, `struct`, `enum`, `trait`, or `impl` definitions

All math, types, and traits live in the source crates:
- quant-core
- quant-timeseries
- quant-vol
- quant-stochastic
- quant-options
- quant-portfolio
- quant-factors
- quant-microstructure
- quant-backtest

## Feature-Gated Modules

Each module is gated behind a feature flag:

```rust
#[cfg(feature = "core")]
pub mod core { pub use quant_core::*; }

#[cfg(feature = "backtest")]
pub mod backtest { pub use quant_backtest::*; }
```

## Prelude Contents

The prelude exports commonly used items:

- Core: `mean`, `variance`, `std_dev`, `XorShift64`, `Normal`
- Traits: `CrossValidator`, `Labeler`, `BetSizer`, `StochasticProcess`, etc.
- Risk: `sharpe_ratio`, `deflated_sharpe_ratio`, `calmar_ratio`
- Backtest: `triple_barrier_label`, `kelly_fraction`, `GenericBacktest`
