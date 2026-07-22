# qf-03-stocks — Stock Price Analysis

Phase 3 of the quant-finance curriculum: working with OHLCV market data, candlestick patterns, basic statistics, and moving averages.

## Overview

This crate provides tools for analyzing stock market data:
- **OHLCV data structures**: Open, High, Low, Close, Volume with optional adjusted close
- **Candlestick analysis**: body, shadows, bullish/bearish patterns
- **Basic statistics**: volume analysis, price changes, ranges
- **Moving averages**: hand-rolled SMA and EMA implementations
- **TimeSeries trait**: common operations on time series data

## Features

### OHLCV Operations

The `Ohlcv` struct represents a single trading period:

```rust
use qf_03_stocks::Ohlcv;

let candle = Ohlcv {
    date: "2024-01-15".to_string(),
    open: 100.0,
    high: 105.0,
    low: 95.0,
    close: 102.0,
    volume: 1000000,
    adj_close: Some(101.5),
};

// Candlestick measurements
assert_eq!(candle.daily_range(), 10.0);      // high - low
assert_eq!(candle.body(), 2.0);              // close - open
assert_eq!(candle.upper_shadow(), 3.0);      // high - max(open, close)
assert_eq!(candle.lower_shadow(), 5.0);      // min(open, close) - low
assert!(candle.is_bullish());                // close > open
assert_eq!(candle.typical_price(), 100.67);  // (H+L+C)/3
```

### Statistics

Compute aggregate statistics across multiple periods:

```rust
use qf_03_stocks::*;

let data: Vec<Ohlcv> = load_ohlcv("prices.csv")?;

let avg_vol = average_volume(&data);           // Mean volume
let change = price_change(&data);              // (last - first) / first
let high = highest_high(&data);                // Max of all highs
let low = lowest_low(&data);                   // Min of all lows
let avg_range = average_daily_range(&data);    // Mean daily range
```

### Moving Averages

Hand-rolled implementations (no external time-series libraries):

```rust
use qf_03_stocks::{sma, ema, TimeSeries};

let data: Vec<Ohlcv> = load_ohlcv("prices.csv")?;
let closes = data.closes();  // TimeSeries trait

// Simple Moving Average
let sma_20 = sma(&closes, 20);  // 20-period SMA
let sma_50 = sma(&closes, 50);  // 50-period SMA

// Exponential Moving Average (alpha = 2 / (period + 1))
let ema_12 = ema(&closes, 12);  // 12-period EMA
let ema_26 = ema(&closes, 26);  // 26-period EMA
```

### TimeSeries Trait

The `TimeSeries` trait provides common operations on OHLCV data:

```rust
use qf_03_stocks::{TimeSeries, Ohlcv};

let data: Vec<Ohlcv> = load_ohlcv("prices.csv")?;

assert_eq!(data.len(), 252);         // Number of trading days
assert!(!data.is_empty());

let closes = data.closes();          // Extract all closing prices
let volumes = data.volumes();        // Extract all volumes
```

## Design Principles

### Trait-Based Architecture

Following the quant-finance modular design philosophy, this crate implements:
- `TimeSeries` trait for common time-series operations
- Functions accepting generic slices for composability
- Reusable components across quant projects

### Hand-Rolled Math

SMA and EMA are implemented from first principles:
- **SMA**: Sliding window average
- **EMA**: Recursive exponential smoothing with alpha = 2/(period+1)

No external time-series libraries used (pedagogy-first approach).

### Validation

OHLCV data is validated on load:
- `high >= low`
- `low <= open <= high`
- `low <= close <= high`

Invalid data triggers `CommonError::ParseError`.

## Examples

### Stock Analysis

Run the example to analyze stock data:

```bash
cargo run -p qf-03-stocks --example stock_analysis
```

Expected output (if data file exists):
```
Loaded 252 trading days

=== Basic Statistics ===
Average Volume: 1250000
Price Change: 12.50%
Highest High: 155.00
Lowest Low: 90.00
Average Daily Range: 3.50

=== Moving Averages ===
SMA(20): 142.50
SMA(50): 138.75

=== Crossovers ===
  Golden Cross at index 180
Total Crossovers: 1
```

### Dataset

Download stock price data from Kaggle:
- [S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500)
- [Yahoo Finance Historical Data](https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price)

Place CSV at: `crates/quant-lab/data/stock_prices.csv`

Expected format:
```csv
Date,Open,High,Low,Close,Volume,Adj Close
2024-01-02,100.0,105.0,99.0,103.0,1000000,102.5
2024-01-03,103.0,107.0,102.0,106.0,1200000,105.5
...
```

## Testing

Run all tests:

```bash
cargo test -p qf-03-stocks
```

The test suite includes 19 tests covering:
- OHLCV creation and methods (9 tests)
- Statistics functions (4 tests)
- Moving averages (6 tests)

All tests use inline data (no external files required).

## Dependencies

- **qf-common**: Shared error types and utilities
- **csv**: CSV file parsing
- **serde**: Serialization/deserialization
- **thiserror**: Error handling

Dev dependencies:
- **approx**: Floating-point assertions

## Integration

This crate is part of the `quant-lab` workspace:

```
crates/quant-lab/
├── crates/
│   ├── qf-common/        # Shared utilities
│   ├── qf-01-fraud/      # Phase 1: Fraud detection
│   ├── qf-02-loan/       # Phase 2: Loan risk
│   ├── qf-03-stocks/     # Phase 3: Stock analysis (this crate)
│   └── ...               # Future phases
└── data/                 # Kaggle datasets (not committed)
```

## Next Steps

Phase 4 will build on this foundation:
- **Returns & Volatility**: Simple and log returns, rolling volatility
- **Sharpe Ratio**: Risk-adjusted performance metrics
- Uses the time-series infrastructure built here

## License

MIT
