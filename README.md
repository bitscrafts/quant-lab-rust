# Learning Quantitative Finance in Rust

**[Download the Book (PDF)](book/build/quant-finance-book.pdf)**

A progressive learning journey from basic Kaggle projects to advanced quant research, implemented from first principles in Rust with a companion LaTeX book.

**Author:** Marcelo Correa

## Overview

This repository contains:
- **Rust crates** implementing quantitative finance concepts (155 tests passing)
- **LaTeX book** documenting the learning journey with mathematical foundations
- **Sample datasets** for running examples immediately

## Quick Start

```bash
# Clone the repository
git clone https://github.com/bitscrafts/quant-lab-rust.git
cd quant-lab-rust

# Run all tests (155 tests)
cargo test

# Run examples
cargo run -p qf-01-fraud --example fraud_analysis
cargo run -p qf-02-loan --example loan_analysis
cargo run -p qf-03-stocks --example stock_analysis
cargo run -p qf-04-returns --example returns_analysis
cargo run -p qf-05-backtest --example backtest_demo
cargo run -p quant-core --example fat_tails
cargo run -p quant-timeseries --example stationarity
cargo run -p quant-timeseries --example ffd_demo
```

## Project Structure

```
quant-lab-rust/
├── README.md              # This file
├── Cargo.toml             # Rust workspace
├── crates/                # Rust implementations
│   ├── qf-common/         # Shared utilities, CSV loading
│   ├── qf-01-fraud/       # Ch01: Credit card fraud detection
│   ├── qf-02-loan/        # Ch02: Loan default prediction
│   ├── qf-03-stocks/      # Ch03: Stock price analysis
│   ├── qf-04-returns/     # Ch04: Returns and risk metrics
│   ├── qf-05-backtest/    # Ch05: Backtesting strategies
│   ├── quant-core/        # Ch06: Moments, RNG, GBM
│   └── quant-timeseries/  # Ch07: OLS, ACF, ADF, FFD
├── data/                  # Sample datasets (bundled)
└── book/                  # LaTeX book source
    ├── main.tex           # Main document
    ├── chapters/          # Chapter files (ch00-ch07)
    ├── references.bib     # Bibliography
    ├── kaobook/           # LaTeX template
    └── build/             # Compiled PDF
        └── quant-finance-book.pdf
```

## Book: Learning Quantitative Finance in Rust

The companion book uses the kaobook LaTeX template with margin notes for formulas and TikZ figures.

### Download PDF

Pre-compiled PDF: [`book/build/quant-finance-book.pdf`](book/build/quant-finance-book.pdf)

### Compile Locally

Requires a full LaTeX distribution (TeX Live or MacTeX):

```bash
cd book
pdflatex -output-directory=build main.tex
biber --output-directory build main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

## Curriculum

### Part I: Foundations (Beginner) - COMPLETE

| Ch | Title | Crate | Tests | Key Concepts |
|----|-------|-------|-------|--------------|
| 0 | Foundations | - | - | Math, statistics, Rust basics |
| 1 | Hello Finance | `qf-01-fraud` | 10 | Z-score detection, confusion matrix, F1 |
| 2 | Risk Basics | `qf-02-loan` | 21 | Feature engineering, ROC-AUC |
| 3 | Market Data | `qf-03-stocks` | 20 | OHLCV, SMA, EMA, candlesticks |

### Part II: Core Concepts (Intermediate) - COMPLETE

| Ch | Title | Crate | Tests | Key Concepts |
|----|-------|-------|-------|--------------|
| 4 | Returns | `qf-04-returns` | 25 | Simple/log returns, volatility, Sharpe, Sortino |
| 5 | Backtesting | `qf-05-backtest` | 20 | SMA crossover, transaction costs, equity curve |

### Part III: Quantitative Methods (Advanced) - IN PROGRESS

| Ch | Title | Crate | Tests | Key Concepts |
|----|-------|-------|-------|--------------|
| 6 | Foundations | `quant-core` | 17 | Moments, skewness, kurtosis, XorShift64, GBM |
| 7 | Time Series | `quant-timeseries` | 18 | OLS, ACF, ADF test, fractional differentiation |
| 8 | Volatility | `quant-vol` | - | EWMA, ARCH, GARCH (coming next) |
| 9 | Stochastic | `quant-stochastic` | - | Brownian motion, Monte Carlo |

### Part IV: Derivatives & Portfolio (Expert)

| Ch | Title | Crate | Key Concepts |
|----|-------|-------|--------------|
| 10 | Options | `quant-options` | Black-Scholes, Greeks, IV |
| 11 | Portfolio | `quant-portfolio` | Markowitz, efficient frontier |
| 12 | Factors | `quant-factors` | PCA, Fama-French |

### Part V: Advanced (Expert)

| Ch | Title | Crate | Key Concepts |
|----|-------|-------|--------------|
| 13 | Microstructure | `quant-microstructure` | Limit order book, order flow |
| 14 | AFML Bridge | `quant-backtest` | Triple-barrier, purged CV |

## Design Philosophy

### Library-First

All code is designed to evolve into a reusable `quant-lib` library. Before writing any struct, we ask: what trait should this implement?

**Core traits:**
- `AnomalyDetector` — Detect outliers in financial data
- `BinaryClassifier` — Classification with probability outputs
- `Scorer` — Produce continuous scores
- `Evaluator<M>` — Evaluate predictions against actuals
- `TimeSeries` — Generic time series operations
- `Strategy` — Trading signal generation
- `Moments` — Statistical moments computation
- `Distribution` — Random sampling

### Hand-Rolled Math (Phase 6+)

Advanced phases implement algorithms from first principles:
- No `rand` crate — hand-rolled XorShift64*
- No `nalgebra` — Gaussian elimination for OLS
- No `statrs` — implement distributions ourselves

This ensures deep understanding of the mathematics.

### Parallel Book Writing

Each chapter is written alongside the code implementation, not as an afterthought. The book documents:
- Mathematical foundations with margin notes
- Rust implementation details
- Key insights and pitfalls
- Exercises for practice

## Example Outputs

### Fat Tails Detection (quant-core)

```
Fat Tails Detection
====================
1. Pure Gaussian (10k N(0,1) draws)
  excess kurt  = +0.0323

2. Fat tails (every 100th draw x10)
  excess kurt  = +74.1272

Gaussian risk models underestimate tail risk!
```

### Stationarity Analysis (quant-timeseries)

```
Random Walk (500 steps):
  ADF statistic: -1.1064
  Stationary:     NO

Fractional Diff (d=0.4):
  ADF statistic: -4.8991
  Stationary:     YES
  ACF(1):         0.6878 (memory preserved)
```

## Development

### Testing

```bash
# Run all tests (155 total)
cargo test

# Run tests for specific crate
cargo test -p quant-core
cargo test -p quant-timeseries
```

### Linting

```bash
cargo clippy --all-targets -- -D warnings
```

### Documentation

```bash
cargo doc --open
```

## References

- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) by Marcos Lopez de Prado
- [Options, Futures, and Other Derivatives](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) by John C. Hull
- [kaobook LaTeX template](https://github.com/fmarotta/kaobook)

## License

MIT

## Author

Marcelo Correa
