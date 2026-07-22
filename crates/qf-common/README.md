# qf-common — Common Utilities for Quantitative Finance

Shared data structures and functions used across quantitative finance projects in the quant-lab workspace.

## Overview

This crate provides foundational utilities needed by multiple quant-finance crates:
- CSV loading for financial datasets
- Basic statistics computation
- Common data types (Transaction, Stats)
- Shared error types

## Features

### Data Loading

```rust
use qf_common::load_transactions;

let transactions = load_transactions("data/creditcard.csv")?;
println!("Loaded {} transactions", transactions.len());
```

Supports CSV files with format:
- Header row (automatically skipped)
- Columns: Time, V1, V2, ..., VN, Amount, Class
- Features extracted from middle columns
- Amount and Class from last two columns

### Statistics

```rust
use qf_common::compute_stats;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let stats = compute_stats(&data);

println!("Mean: {:.2}", stats.mean);
println!("Std: {:.2}", stats.std);
println!("Min: {:.2}", stats.min);
println!("Max: {:.2}", stats.max);
```

Computes:
- Arithmetic mean
- Sample standard deviation (n-1 denominator)
- Minimum and maximum values

### Data Types

**Transaction**:
```rust
pub struct Transaction {
    pub features: Vec<f64>,  // Feature vector
    pub amount: f64,          // Transaction amount
    pub class: u8,            // 0=normal, 1=fraud
}
```

**Stats**:
```rust
pub struct Stats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}
```

### Error Handling

```rust
use qf_common::CommonError;

match load_transactions(path) {
    Ok(transactions) => { /* ... */ },
    Err(CommonError::FileNotFound(path)) => {
        eprintln!("File not found: {}", path);
    },
    Err(CommonError::ParseError(msg)) => {
        eprintln!("Parse error: {}", msg);
    },
    Err(e) => eprintln!("Error: {}", e),
}
```

## Dependencies

- `csv` — CSV parsing
- `thiserror` — Error handling

## Usage in Other Crates

```toml
[dependencies]
qf-common = { path = "../qf-common" }
```

## Testing

All tests use `_tmp/` directory for test fixtures:

```bash
cargo test -p qf-common
```

## Examples

This crate is used by:
- `qf-01-fraud` — Credit card fraud detection
- `qf-02-loan` (future) — Loan default prediction
- `qf-03-stocks` (future) — Stock price analysis

See individual crates for usage examples.
