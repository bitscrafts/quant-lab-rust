# HANDOFF: Quant-Finance Implementation

**Project**: `quant-lab-rust`
**Repository**: https://github.com/bitscrafts/quant-lab-rust
**Last Updated**: 2026-07-22
**Status**: Part I Complete (Phases 1-3), Part II Pending (Phase 4 next)

---

## Quick Start

```bash
# Clone and verify
git clone https://github.com/bitscrafts/quant-lab-rust.git
cd quant-lab-rust

# Build all crates
cargo build

# Run all tests (52 tests total)
cargo test

# Run examples
cargo run -p qf-01-fraud --example fraud_analysis
cargo run -p qf-02-loan --example loan_analysis
cargo run -p qf-03-stocks --example stock_analysis
```

---

## Project Overview

This is a progressive learning curriculum for quantitative finance, implemented
from first principles in Rust with a companion LaTeX book (kaobook template).

**Philosophy**:
- Learn by doing, document in parallel
- Library-first design (code will evolve into `quant-lib`)
- Hand-rolled math (no black-box libraries)
- TDD: tests written BEFORE production code

**Two deliverables per phase**:
1. Working, tested Rust crate in `crates/`
2. Corresponding LaTeX chapter in `book/chapters/`

---

## Repository Structure

```
quant-lab-rust/
├── Cargo.toml              # Workspace root
├── README.md               # Project overview with book link
├── HANDOFF.md              # This file
├── .gitignore              # macOS, Rust, LaTeX exclusions
│
├── crates/                 # Rust implementations
│   ├── qf-common/          # Shared utilities (CSV loading, stats)
│   ├── qf-01-fraud/        # Ch01: Credit card fraud detection
│   ├── qf-02-loan/         # Ch02: Loan default prediction
│   └── qf-03-stocks/       # Ch03: Stock price analysis (OHLCV, SMA, EMA)
│
├── data/                   # Sample datasets (included)
│   ├── spy.csv             # SPY stock data for examples
│   └── ...
│
└── book/                   # LaTeX book source
    ├── main.tex            # Main document
    ├── references.bib      # Bibliography
    ├── chapters/           # Chapter files (ch00.tex - ch03.tex)
    ├── figures/            # Externalized TikZ figures
    ├── kaobook/            # kaobook template files
    └── build/              # Compiled PDF output
        └── quant-finance-book.pdf
```

---

## Completed Work (Part I: Foundations)

### Phase 1: Hello Finance — Credit Card Fraud Detection

**Crate**: `qf-01-fraud`
**Book**: `book/chapters/ch01.tex`
**Tests**: 10 tests passing

Implements:
- `ZScoreDetector` — anomaly detection via z-score threshold
- `ConfusionMatrix` — precision, recall, F1 metrics
- `evaluate()` — runs detector on dataset

Key patterns:
```rust
pub trait AnomalyDetector {
    fn fit(&mut self, data: &[Transaction]);
    fn predict(&self, item: &Transaction) -> bool;
}
```

### Phase 2: Risk Basics — Loan Default Prediction

**Crate**: `qf-02-loan`
**Book**: `book/chapters/ch02.tex`
**Tests**: 21 tests passing

Implements:
- `FeatureExtractor` — debt-to-income, loan-to-income, grade score
- `OneHotEncoder` — categorical encoding (RENT/OWN/MORTGAGE)
- `Normalizer` — min-max scaling
- `LinearScorer` — weighted sum + sigmoid for probability
- `roc_curve()`, `auc()` — ROC-AUC evaluation

Key patterns:
```rust
pub trait Scorer {
    fn score(&self, features: &[f64]) -> f64;
}

pub trait BinaryClassifier {
    fn predict(&self, features: &[f64]) -> bool;
    fn predict_proba(&self, features: &[f64]) -> f64;
}
```

### Phase 3: Market Data — Stock Price Analysis

**Crate**: `qf-03-stocks`
**Book**: `book/chapters/ch03.tex`
**Tests**: 20 tests passing

Implements:
- `Ohlcv` struct with candlestick methods (body, shadows, typical price)
- `sma()`, `ema()` — simple and exponential moving averages
- Statistics: `average_volume()`, `price_change()`, `highest_high()`, `lowest_low()`
- `TimeSeries` trait for generic operations

Key patterns:
```rust
pub trait TimeSeries {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn closes(&self) -> Vec<f64>;
    fn volumes(&self) -> Vec<u64>;
}
```

---

## Next Task: Phase 4 — Returns & Volatility

**Crate to create**: `qf-04-returns`
**Book chapter**: `book/chapters/ch04.tex`
**Dependencies**: `qf-common`, `qf-03-stocks`

### Overview

This phase introduces fundamental quant concepts:
- Simple and logarithmic returns
- Volatility measures (sample std dev, annualized, rolling)
- Risk-adjusted metrics (Sharpe ratio, Sortino ratio)
- Drawdown analysis

**Key insight**: Log returns are additive across time (easier math), simple
returns are easier to interpret ("I made 10%").

### Requirements

#### R4.1: Create qf-04-returns Crate

```bash
cd crates/quant-lab/crates
cargo new qf-04-returns --lib
```

Add to workspace `Cargo.toml`:
```toml
[workspace]
members = [
    "crates/qf-common",
    "crates/qf-01-fraud",
    "crates/qf-02-loan",
    "crates/qf-03-stocks",
    "crates/qf-04-returns",  # ADD THIS
]
```

Dependencies in `qf-04-returns/Cargo.toml`:
```toml
[dependencies]
qf-common = { path = "../qf-common" }
qf-03-stocks = { path = "../qf-03-stocks" }
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
```

#### R4.2: Returns Calculation

```rust
/// Simple returns: (P_t - P_{t-1}) / P_{t-1}
pub fn simple_returns(prices: &[f64]) -> Vec<f64>;

/// Log returns: ln(P_t / P_{t-1})
pub fn log_returns(prices: &[f64]) -> Vec<f64>;

/// Cumulative returns from a series of period returns
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64>;
```

- Output length is `prices.len() - 1`
- Empty or single-element input returns empty Vec

#### R4.3: Volatility Measures

```rust
/// Sample standard deviation of returns
pub fn volatility(returns: &[f64]) -> f64;

/// Annualized volatility: vol × sqrt(periods_per_year)
/// Use 252 for daily data
pub fn annualized_volatility(returns: &[f64], periods_per_year: f64) -> f64;

/// Rolling standard deviation
pub fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64>;
```

#### R4.4: Risk-Adjusted Metrics

```rust
/// Sharpe ratio: (mean_return - risk_free_rate) / volatility
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64;

/// Annualized Sharpe: sharpe × sqrt(periods_per_year)
pub fn annualized_sharpe(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64;

/// Sortino ratio: uses downside deviation instead of volatility
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64) -> f64;
```

#### R4.5: Drawdown Analysis

```rust
/// Percentage decline from running maximum at each point
/// Always non-positive values
pub fn drawdown(prices: &[f64]) -> Vec<f64>;

/// Minimum value of drawdown series (most negative = worst)
pub fn max_drawdown(prices: &[f64]) -> f64;

pub struct DrawdownStats {
    pub max_drawdown: f64,
    pub max_drawdown_duration: usize,
    pub recovery_time: Option<usize>,
}
```

#### R4.6: Returns Trait

```rust
pub trait Returns {
    fn simple_returns(&self) -> Vec<f64>;
    fn log_returns(&self) -> Vec<f64>;
}

// Implement for:
impl Returns for &[f64] { ... }
impl Returns for Vec<f64> { ... }
impl Returns for Vec<Ohlcv> { ... }  // Uses closing prices
```

#### R4.7: Example Binary

Create `qf-04-returns/examples/returns_analysis.rs`:
- Load stock data using `qf-03-stocks`
- Compute simple and log returns
- Calculate volatility (daily and annualized)
- Compute Sharpe ratio
- Analyze drawdowns

Output format:
```
Returns Analysis
================
Simple Returns: mean=0.0012, std=0.015
Log Returns: mean=0.0011, std=0.015
Volatility (daily): 0.0150
Volatility (annualized): 0.238
Sharpe Ratio (annualized): 0.79
Max Drawdown: -12.5%
```

#### R4.8: Book Chapter

Create `book/chapters/ch04.tex` with:
- Learning objectives
- Simple vs log returns (formulas in margin notes)
- Volatility concepts and annualization
- Sharpe ratio derivation
- Sortino ratio (downside risk)
- Drawdown visualization
- Rust code samples
- Exercises

### TDD Contract (24 tests)

**File**: `crates/qf-04-returns/tests/returns_tests.rs`

| Test name | Given | Expects |
|---|---|---|
| `test_simple_returns_basic` | `[100.0, 110.0, 99.0]` | `[0.10, -0.10]` ±1e-9 |
| `test_simple_returns_empty` | `[]` | empty Vec |
| `test_simple_returns_single` | `[100.0]` | empty Vec |
| `test_log_returns_basic` | `[100.0, 110.0]` | `[ln(1.1)]` ±1e-9 |
| `test_log_returns_matches_formula` | `[100.0, 105.0, 110.0]` | each = ln(P_t/P_{t-1}) |
| `test_log_simple_relationship` | `[100, 101, 102, 103]` | `log ≈ ln(1 + simple)` ±1e-9 |
| `test_cumulative_returns` | simple `[0.10, -0.05, 0.08]` | ending ≈0.1286 |
| `test_volatility_basic` | `[0.01, -0.02, 0.015, -0.01, 0.02]` | std_dev ≈ 0.0158 |
| `test_volatility_constant` | `[0.01, 0.01, 0.01]` | 0.0 |
| `test_volatility_empty` | `[]` | 0.0 |
| `test_annualized_volatility` | daily vol 0.01, 252 periods | ≈ 0.159 |
| `test_rolling_volatility_basic` | `[1,2,3,4,5]` as returns, w=3 | len = 3 |
| `test_rolling_volatility_invalid` | w=0 or w>len | empty Vec |
| `test_sharpe_ratio_positive` | returns with mean>rf | positive Sharpe |
| `test_sharpe_ratio_negative` | returns with mean<rf | negative Sharpe |
| `test_sharpe_ratio_zero_vol` | constant returns | 0.0 |
| `test_annualized_sharpe` | daily Sharpe 0.05, 252 | ≈ 0.79 |
| `test_sortino_basic` | mixed returns | Sortino > Sharpe |
| `test_drawdown_basic` | `[100, 110, 105, 115, 100]` | drawdowns at each point |
| `test_drawdown_always_negative` | any price series | all values ≤ 0 |
| `test_max_drawdown_basic` | `[100, 110, 90, 95]` | ≈ -0.182 |
| `test_max_drawdown_no_decline` | `[100, 110, 120]` | 0.0 |
| `test_returns_trait_slice` | `[100.0, 105.0, 110.0]` as slice | trait methods work |
| `test_returns_trait_ohlcv` | Vec<Ohlcv> from Phase 3 | uses closes() |

### Exit Criteria (Phase 4)

Run from repository root:

```bash
# Structure exists
test -f crates/qf-04-returns/Cargo.toml
grep -q "qf-04-returns" Cargo.toml

# Tests pass
cargo test -p qf-04-returns 2>&1 | grep -E "test result.*0 failed"

# No clippy warnings
cargo clippy -p qf-04-returns --all-targets 2>&1 | grep -qv "^warning:"

# README exists
test -f crates/qf-04-returns/README.md

# Example runs
cargo run -p qf-04-returns --example returns_analysis 2>&1 | grep -i "sharpe"

# Book chapter exists
test -f book/chapters/ch04.tex
```

### Guardrails

- **Approved dependencies**: `csv`, `thiserror`, `serde`. Dev: `approx`
- **Depend on**: `qf-common`, `qf-03-stocks` for `Ohlcv` and `TimeSeries`
- **Package-scoped builds only**: `-p qf-04-returns`
- **Tests use inline data** or system temp dir (see `qf-common` pattern)
- **All math hand-rolled**: no external statistics libraries
- **Handle division by zero**: return 0.0, never panic

---

## Code Conventions

### Error Handling

Use `thiserror` for custom error types:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReturnsError {
    #[error("Insufficient data: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}
```

### Testing Pattern

Tests go in `tests/` directory with `approx` for float comparisons:

```rust
use approx::assert_relative_eq;

#[test]
fn test_volatility_basic() {
    let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
    let vol = volatility(&returns);
    assert_relative_eq!(vol, 0.0158, epsilon = 1e-3);
}
```

For tests needing temp files, use system temp dir:

```rust
fn create_test_file(filename: &str, content: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push("quant-lab-tests");
    fs::create_dir_all(&path).unwrap();
    path.push(filename);
    fs::write(&path, content).unwrap();
    path
}
```

### Documentation

Every public item needs doc comments:

```rust
/// Computes the simple return between consecutive prices.
///
/// # Formula
/// r_t = (P_t - P_{t-1}) / P_{t-1}
///
/// # Arguments
/// * `prices` - Slice of prices ordered chronologically
///
/// # Returns
/// Vector of returns with length `prices.len() - 1`
///
/// # Example
/// ```
/// use qf_04_returns::simple_returns;
/// let prices = vec![100.0, 110.0, 99.0];
/// let returns = simple_returns(&prices);
/// assert_eq!(returns.len(), 2);
/// ```
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    // ...
}
```

---

## Building the Book

The book uses the kaobook LaTeX template and compiles remotely via Podman.

### Remote Compilation (Preferred)

```bash
# From quant-lab-rust directory
# 1. Sync files to remote server
rsync -avz --delete book/ mvcorrea@lnx:~/podman/latex-remote/quant-lab/src/

# 2. Compile on remote
ssh mvcorrea@lnx "podman exec latex-compiler-shared sh -c '
    cd /workspace/quant-lab/src && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    biber --output-directory ../build main && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex
'"

# 3. Download PDF
scp mvcorrea@lnx:~/podman/latex-remote/quant-lab/build/main.pdf book/build/quant-finance-book.pdf
```

### Local Compilation (If TeX Live installed)

```bash
cd book
pdflatex -output-directory=build main.tex
biber --output-directory build main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

### Book Structure

```
book/
├── main.tex              # Main document (includes chapters)
├── references.bib        # Bibliography
├── kaobook/              # Template files (DO NOT MODIFY)
│   ├── kaobook.cls
│   ├── kao.sty
│   └── ...
├── chapters/
│   ├── ch00.tex          # Chapter 0: Foundations
│   ├── ch01.tex          # Chapter 1: Fraud Detection
│   ├── ch02.tex          # Chapter 2: Loan Default
│   ├── ch03.tex          # Chapter 3: Stock Analysis
│   └── ch04.tex          # Chapter 4: Returns (TO CREATE)
└── figures/
    ├── confusion-matrix.tex
    ├── roc-curve.tex
    └── candlesticks.tex
```

### Adding a New Chapter

1. Create `book/chapters/ch04.tex`
2. Add to `main.tex` after ch03:
   ```latex
   \input{chapters/ch04}
   ```
3. Follow existing chapter structure:
   - `\chapter{Title}` and `\label{ch:label}`
   - `\begin{companioncode}{crate-name}` block
   - `\section{Learning Objectives}`
   - `\section{Introduction}`
   - Content with `\marginnote{}` for formulas
   - `\begin{keyinsight}` boxes
   - `\begin{lstlisting}[language=Rust]` for code
   - `\section{Exercises}`
   - `\section{Key Takeaways}`

---

## Future Phases

After Phase 4, the curriculum continues:

| Phase | Crate | Key Concepts |
|-------|-------|--------------|
| 5 | `qf-05-backtest` | SMA crossover strategy, P&L, drawdown |
| 6 | `quant-core` | Moments, skewness, kurtosis, GBM simulation |
| 7 | `quant-timeseries` | OLS, ACF, ADF test, fractional differentiation |
| 8 | `quant-vol` | EWMA, ARCH, GARCH(1,1) |
| 9 | `quant-stochastic` | Brownian motion, Monte Carlo |
| 10 | `quant-options` | Black-Scholes, Greeks, implied volatility |
| 11 | `quant-portfolio` | Markowitz mean-variance optimization |
| 12 | `quant-factors` | PCA, Fama-French factors |
| 13 | `quant-microstructure` | Limit order book, order flow |
| 14 | `quant-backtest` | Triple-barrier, purged CV (AFML) |

Full spec: `on-research/docs/quant-finance/spec.md`

---

## Git Workflow

```bash
# Standard workflow
git add <files>
git commit -m "feat(qf-04-returns): implement simple and log returns"

# Commit message prefixes
# feat: new feature
# fix: bug fix
# docs: documentation
# test: tests
# refactor: code refactoring

# Push to GitHub
git push origin master
```

---

## Troubleshooting

### Tests fail with permission denied

Tests write to system temp dir. If issues persist:
```bash
rm -rf /tmp/quant-lab-tests
```

### LaTeX compilation fails

1. Check container is running:
   ```bash
   ssh mvcorrea@lnx "podman ps | grep latex-compiler"
   ```

2. If not running:
   ```bash
   ssh mvcorrea@lnx "cd ~/podman/latex-remote && podman-compose up -d"
   ```

3. Check logs:
   ```bash
   ssh mvcorrea@lnx "cat ~/podman/latex-remote/quant-lab/build/main.log | tail -50"
   ```

### Clippy warnings

Fix all warnings before committing:
```bash
cargo clippy -p qf-04-returns --all-targets -- -D warnings
```

---

## Contact

Repository: https://github.com/bitscrafts/quant-lab-rust
Author: Marcelo Correa
