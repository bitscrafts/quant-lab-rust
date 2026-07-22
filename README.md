# Learning Quantitative Finance in Rust

**[Download the Book (PDF)](book/build/quant-finance-book.pdf)**

A progressive learning journey from basic Kaggle projects to advanced quant research, implemented from first principles in Rust with a companion LaTeX book.

**Author:** Marcelo Correa

## Overview

This repository contains:
- **Rust crates** implementing quantitative finance concepts
- **LaTeX book** documenting the learning journey with mathematical foundations
- **Sample datasets** for running examples immediately

## Quick Start

```bash
# Clone the repository
git clone https://github.com/bitscrafts/quant-lab-rust.git
cd quant-lab-rust

# Run examples
cargo run -p qf-01-fraud --example fraud_analysis
cargo run -p qf-03-stocks --example stock_analysis

# Run tests
cargo test
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
│   └── qf-03-stocks/      # Ch03: Stock price analysis
├── data/                  # Sample datasets (bundled)
├── book/                  # LaTeX book source
│   ├── main.tex           # Main document
│   ├── chapters/          # Chapter files
│   ├── references.bib     # Bibliography
│   └── kaobook.cls        # LaTeX template
└── build/                 # Compiled PDF
    └── quant-finance-book.pdf
```

## Book: Learning Quantitative Finance in Rust

The companion book uses the kaobook LaTeX template with margin notes for formulas and TikZ figures.

### Download PDF

Pre-compiled PDF: [`build/quant-finance-book.pdf`](build/quant-finance-book.pdf)

### Compile Locally

Requires a full LaTeX distribution (TeX Live or MacTeX):

```bash
cd book
pdflatex -output-directory=../build main.tex
biber --output-directory ../build main
pdflatex -output-directory=../build main.tex
pdflatex -output-directory=../build main.tex
```

## Curriculum

### Part I: Foundations (Beginner)

| Ch | Title | Crate | Key Concepts |
|----|-------|-------|--------------|
| 0 | Foundations | - | Math, statistics, Rust basics |
| 1 | Hello Finance | `qf-01-fraud` | Z-score detection, confusion matrix, F1 |
| 2 | Risk Basics | `qf-02-loan` | Feature engineering, ROC-AUC |
| 3 | Market Data | `qf-03-stocks` | OHLCV, SMA, EMA, candlesticks |

### Part II: Core Concepts (Intermediate)

| Ch | Title | Crate | Key Concepts |
|----|-------|-------|--------------|
| 4 | Returns | `qf-04-returns` | Simple/log returns, volatility, Sharpe |
| 5 | Backtesting | `qf-05-backtest` | SMA crossover, P&L, drawdown |

### Part III: Quantitative Methods (Advanced)

| Ch | Title | Crate | Key Concepts |
|----|-------|-------|--------------|
| 6 | Foundations | `quant-core` | Moments, fat tails, GBM |
| 7 | Time Series | `quant-timeseries` | OLS, ACF, ADF, FFD |
| 8 | Volatility | `quant-vol` | EWMA, ARCH, GARCH |
| 9 | Stochastic | `quant-stochastic` | Brownian motion, Monte Carlo |

### Part IV: Derivatives & Portfolio (Expert)

| Ch | Title | Crate | Key Concepts |
|----|-------|-------|--------------|
| 10 | Options | `quant-options` | Black-Scholes, Greeks, IV |
| 11 | Portfolio | `quant-portfolio` | Markowitz, efficient frontier |
| 12 | Factors | `quant-factors` | PCA, Fama-French |

### Part V: Advanced (Expert)

| Ch | Title | Crate | Key Concepts |
|----|-------|-------|--------------|
| 13 | Microstructure | `quant-microstructure` | LOB, order flow |
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

### Hand-Rolled Math (Phase 6+)

Advanced phases implement algorithms from first principles:
- No `rand` crate — hand-rolled xorshift64*
- No `nalgebra` — manual matrix operations
- No `statrs` — implement distributions ourselves

This ensures deep understanding of the mathematics.

### Parallel Book Writing

Each chapter is written alongside the code implementation, not as an afterthought. The book documents:
- Mathematical foundations
- Rust implementation details
- Key insights and pitfalls
- Exercises for practice

## Running Examples

Examples use bundled sample data and work immediately after clone:

```bash
# Credit card fraud detection
cargo run -p qf-01-fraud --example fraud_analysis

# Stock price analysis with moving averages
cargo run -p qf-03-stocks --example stock_analysis
```

## Development

### Testing

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p qf-01-fraud
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
