# quant-lib Examples

Runnable Rust examples demonstrating the `quant-lib` unified quantitative
finance library. Examples progress from basic concepts to full trading
pipelines, covering all modules in the library.

## Directory Structure

```
examples/
├── 01_mean_variance.rs      # Basic statistics
├── 02_returns_basics.rs     # Return calculations
├── ...                      # Core concept examples (01-16)
├── common/                  # Shared utilities
├── projects/                # Real-world trading strategies
│   └── README.md
└── solutions/               # Chapter exercise solutions
    └── README.md
```

## Core Examples (01-16)

Sequential examples covering fundamental quant concepts:

| # | Example | Topic | Run Command |
|---|---------|-------|-------------|
| 01 | `01_mean_variance.rs` | Basic mean/variance statistics | `cargo run -p quant-lib --example 01_mean_variance` |
| 02 | `02_returns_basics.rs` | Simple and log returns | `cargo run -p quant-lib --example 02_returns_basics` |
| 03 | `03_rolling_window.rs` | Rolling mean, std, correlations | `cargo run -p quant-lib --example 03_rolling_window` |
| 04 | `04_random_walk.rs` | Random walk simulation | `cargo run -p quant-lib --example 04_random_walk` |
| 05 | `05_ols_regression.rs` | OLS regression basics | `cargo run -p quant-lib --example 05_ols_regression` |
| 06 | `06_stationarity.rs` | ADF test, unit roots | `cargo run -p quant-lib --example 06_stationarity` |
| 07 | `07_volatility.rs` | EWMA, GARCH volatility | `cargo run -p quant-lib --example 07_volatility` |
| 08 | `08_options_greeks.rs` | Black-Scholes, Greeks | `cargo run -p quant-lib --example 08_options_greeks` |
| 09 | `09_portfolio_frontier.rs` | Efficient frontier | `cargo run -p quant-lib --example 09_portfolio_frontier` |
| 10 | `10_pca_factors.rs` | PCA factor decomposition | `cargo run -p quant-lib --example 10_pca_factors` |
| 11 | `11_microstructure_lob.rs` | Limit order book | `cargo run -p quant-lib --example 11_microstructure_lob` |
| 12 | `12_triple_barrier.rs` | Triple-barrier labeling | `cargo run -p quant-lib --example 12_triple_barrier` |
| 13 | `13_purged_cv.rs` | Purged cross-validation | `cargo run -p quant-lib --example 13_purged_cv` |
| 14 | `14_kelly_sizing.rs` | Kelly criterion sizing | `cargo run -p quant-lib --example 14_kelly_sizing` |
| 15 | `15_full_pipeline.rs` | End-to-end AFML pipeline | `cargo run -p quant-lib --example 15_full_pipeline` |
| 16 | `16_trait_compose.rs` | Trait composition patterns | `cargo run -p quant-lib --example 16_trait_compose` |

## Real-World Projects

Five production-style trading strategies in `projects/`:

| Project | Description | Level |
|---------|-------------|-------|
| `01_momentum` | Cross-sectional momentum on B3 stocks | Beginner |
| `02_pairs_trading` | Cointegration pairs trade (PETR4/VALE3) | Intermediate |
| `03_risk_parity` | Risk-parity portfolio construction | Advanced |
| `04_options_mm` | Options market making with delta hedging | Expert |
| `05_afml_pipeline` | Full AFML pipeline (FFD, barriers, Kelly) | Expert |

See [projects/README.md](projects/README.md) for detailed documentation.

## Chapter Solutions

Exercise solutions for all 14 book chapters in `solutions/`:

| Chapter | Topic | Exercises |
|---------|-------|-----------|
| ch01 | Credit Card Fraud | Anomaly detection, ROC curves |
| ch02 | Loan Default | Classification metrics |
| ch03 | Stock Analysis | Price series analysis |
| ch04 | Returns | Return distributions, moments |
| ch05 | Backtesting | Walk-forward, overfitting |
| ch06 | Moments | Higher moments, normality tests |
| ch07 | Time Series | ARIMA, stationarity |
| ch08 | Volatility | GARCH, realized vol |
| ch09 | Stochastic | GBM, jump diffusion |
| ch10 | Options | Greeks, implied vol |
| ch11 | Portfolio | Optimization, risk parity |
| ch12 | Factors | PCA, Fama-French |
| ch13 | Microstructure | LOB, market impact |
| ch14 | AFML | Triple barrier, purged CV |

See [solutions/README.md](solutions/README.md) for detailed documentation.

## Running Examples

```bash
# From the quant-lab workspace root

# Run a specific example
cargo run -p quant-lib --example 01_mean_variance

# Run a project
cargo run -p quant-lib --example projects-01_momentum

# Run chapter solutions
cargo run -p quant-lib --example solutions-ch01_fraud_solutions

# Run tests in an example
cargo test -p quant-lib --example solutions-ch01_fraud_solutions

# Check all examples compile
cargo check -p quant-lib --examples

# Lint all examples
cargo clippy -p quant-lib --examples -- -D warnings
```

## Data

Examples use bundled data in `quant-lab/data/`:

| File | Description |
|------|-------------|
| `creditcard_sample.csv` | Credit card fraud detection dataset |
| `stock_prices.csv` | Historical stock prices |
| `PETR4.json`, `VALE3.json`, ... | B3 stock OHLCV data (8 tickers) |

## Dependencies

All examples use the `quant-lib` facade which re-exports:

- `quant-core` - Core primitives (returns, risk metrics)
- `quant-timeseries` - Time series analysis
- `quant-vol` - Volatility models
- `quant-stochastic` - Stochastic processes
- `quant-options` - Option pricing
- `quant-portfolio` - Portfolio optimization
- `quant-factors` - Factor models
- `quant-microstructure` - Market microstructure
- `quant-backtest` - Backtesting framework
