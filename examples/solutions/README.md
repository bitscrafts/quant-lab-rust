# Exercise Solutions

Complete, runnable Rust code for all exercises in Appendix D of the book.

## Structure

```
solutions/
├── ch01_fraud_solutions.rs      # Credit Card Fraud (5 exercises)
├── ch02_loan_solutions.rs       # Loan Default (5 exercises)
├── ch03_stocks_solutions.rs     # Market Data (5 exercises)
├── ch04_returns_solutions.rs    # Returns & Volatility (5 exercises)
├── ch05_backtest_solutions.rs   # Backtesting (5 exercises)
├── ch06_core_solutions.rs       # Statistical Foundations (5 exercises)
├── ch07_timeseries_solutions.rs # Time Series (5 exercises)
├── ch08_vol_solutions.rs        # Volatility Models (5 exercises)
├── ch09_stochastic_solutions.rs # Stochastic Processes (4 exercises)
├── ch10_options_solutions.rs    # Options Pricing (6 exercises)
├── ch11_portfolio_solutions.rs  # Portfolio Optimization (6 exercises)
├── ch12_factors_solutions.rs    # Factor Models (5 exercises)
├── ch13_micro_solutions.rs      # Market Microstructure (5 exercises)
└── ch14_afml_solutions.rs       # AFML Backtesting (4 exercises)
```

**Total: 70 exercises with full code solutions**

## Running Solutions

```bash
# Run all solutions for a chapter
cargo run -p quant-lib --example ch01_fraud_solutions
cargo run -p quant-lib --example ch02_loan_solutions
# ... etc

# Or run individual exercise (modify main() to call only one)
```

## Solution Format

Each solution file follows this pattern:

```rust
//! Chapter N Exercise Solutions
//!
//! Companion code for Appendix D of "Learning Quantitative Finance in Rust"
//! Run: cargo run -p quant-lib --example chN_<topic>_solutions

use quant_lib::prelude::*;

/// Exercise 1: <Title from appD.tex>
///
/// Problem: <Brief statement>
/// Approach: <Solution strategy>
fn exercise_1() {
    println!("\n=== Exercise 1: <Title> ===\n");

    // Solution code
    let result = /* ... */;

    // Output
    println!("Result: {:?}", result);
    println!("Expected: <from appD.tex>");
}

fn main() {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
}
```

## Cross-Reference with Book

Each exercise in the code corresponds to an exercise in `book/chapters/appD.tex`.
The LaTeX file contains:
- Problem statement (prose)
- Solution approach (explanation)
- Solution code (lstlisting)
- Expected output (verbatim)

## Verification

All solutions are deterministic (seeded RNG) and produce expected outputs documented in appD.tex.

```bash
# Run all solutions and verify output
for ch in ch{01..14}*_solutions; do
    cargo run -p quant-lib --example "$ch" 2>&1 | grep "Result:"
done
```

## Adding New Solutions

When adding a new chapter's exercises:

1. Create `chNN_topic_solutions.rs` following the template above
2. Add entry to `quant-lib/Cargo.toml`:
   ```toml
   [[example]]
   name = "chNN_topic_solutions"
   path = "examples/solutions/chNN_topic_solutions.rs"
   ```
3. Update `book/chapters/appD.tex` with code listings
4. Add test in `tests/solutions_tests.rs`
