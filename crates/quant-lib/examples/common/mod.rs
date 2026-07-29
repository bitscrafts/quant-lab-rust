#![allow(dead_code)]
//! Shared utilities for quant-lib examples.
//!
//! Provides a minimal CSV reader for the bundled `stock_prices.csv`
//! dataset and a JSON OHLCV parser for the bundled Brazilian stock
//! JSON files. No external dependencies --- just `std::fs` and string
//! splitting.
//!
//! This module lives in `examples/common/mod.rs` (not
//! `examples/common.rs`) so that Cargo does not attempt to compile it
//! as a standalone example target. Each example includes it via
//! `#[path = "common/mod.rs"] mod common;`.
//!
//! # Bundled datasets
//!
//! - `crates/quant-lab/data/stock_prices.csv` --- synthetic OHLCV
//!   for a single stock (Jan 2024 - Jun 2024, 124 rows).
//! - `data/*.json` --- real Brazilian stock OHLCV (PETR4, VALE3,
//!   ITSA4, BBDC4, B3SA3, ABEV3, GGBR4, WEGE3) from B3 exchange,
//!   Jul 2021 - Jul 2024.
//!
//! # Public data sources for extending the examples
//!
//! For more real data, download from these free public sources:
//!
//! | Source | URL | Format |
//! |---|---|---|
//! | Stooq | https://stooq.com/q/d/?s={symbol}&i=d | CSV OHLCV |
//! | Yahoo Finance | https://finance.yahoo.com/quote/{TICKER}/history | CSV OHLCV |
//! | Ken French Data Library | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | ZIP/CSV factors |
//! | FRED (St. Louis Fed) | https://fred.stlouisfed.org/series/{ID} | CSV macro |
//! | Binance API | https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d | JSON OHLCV |

use std::fs;
use std::path::Path;

/// A single OHLCV bar.
#[derive(Debug, Clone)]
pub struct OhlcvBar {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Load OHLCV bars from the bundled `stock_prices.csv` file.
///
/// The file lives at `crates/quant-lab/data/stock_prices.csv`. The
/// caller supplies the path so the same loader works from any
/// crate's examples directory.
pub fn load_stock_csv(path: &Path) -> Vec<OhlcvBar> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut bars = Vec::new();
    for (i, line) in contents.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 7 {
            continue;
        }
        bars.push(OhlcvBar {
            date: cols[0].to_string(),
            open: cols[1].parse().unwrap_or(0.0),
            high: cols[2].parse().unwrap_or(0.0),
            low: cols[3].parse().unwrap_or(0.0),
            close: cols[4].parse().unwrap_or(0.0),
            volume: cols[5].parse().unwrap_or(0.0),
        });
    }
    bars
}

/// Resolve the path to the bundled `stock_prices.csv` from a crate
/// root. Call as `common::stock_csv_path(env!("CARGO_MANIFEST_DIR"))`.
pub fn stock_csv_path(crate_dir: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(crate_dir);
    // crates/quant-lib/examples -> crates/quant-lab/data/stock_prices.csv
    p.pop(); // examples
    p.pop(); // quant-lib
    p.pop(); // crates
    p.push("quant-lab");
    p.push("data");
    p.push("stock_prices.csv");
    p
}

/// Extract the close prices as a `Vec<f64>`.
pub fn closes(bars: &[OhlcvBar]) -> Vec<f64> {
    bars.iter().map(|b| b.close).collect()
}

/// Resolve the path to a bundled B3 stock JSON file from a crate root.
/// Call as `common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4")`.
///
/// The JSON files live alongside `stock_prices.csv` in the project's
/// bundled data directory at `crates/quant-lab/data/{symbol}.json`.
/// From `crates/quant-lab/crates/quant-lib` we pop three components
/// (quant-lib, crates, quant-lab) to reach `crates/`, then push
/// `quant-lab/data/{symbol}.json`.
pub fn b3_json_path(crate_dir: &str, symbol: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(crate_dir);
    p.pop(); // quant-lib
    p.pop(); // crates
    p.pop(); // quant-lab
    p.push("quant-lab");
    p.push("data");
    p.push(format!("{symbol}.json"));
    p
}

/// Simple JSON OHLCV parser for the bundled B3 stock files.
///
/// Each file is a JSON array of objects with keys
/// `date, open, high, low, close, volume`. This is a minimal parser
/// that relies on the fixed key order; it is not a general JSON
/// parser.
pub fn load_json_ohlcv(path: &Path) -> Vec<OhlcvBar> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut bars = Vec::new();
    // Each record looks like: {"date":"2021-07-22","open":27.0,"high":...,"low":...,"close":...,"volume":...}
    for record in contents.split("},{") {
        let record = record
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_start_matches('{')
            .trim_end_matches('}');
        let fields: Vec<&str> = record.split(',').collect();
        if fields.len() < 6 {
            continue;
        }
        let date = fields[0]
            .split(':')
            .nth(1)
            .unwrap_or("\"\"")
            .trim_matches('"')
            .to_string();
        let open = parse_json_num(fields[1]);
        let high = parse_json_num(fields[2]);
        let low = parse_json_num(fields[3]);
        let close = parse_json_num(fields[4]);
        let volume = parse_json_num(fields[5]);
        bars.push(OhlcvBar {
            date,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    bars
}

fn parse_json_num(field: &str) -> f64 {
    field
        .split(':')
        .nth(1)
        .unwrap_or("0")
        .trim_matches('"')
        .parse()
        .unwrap_or(0.0)
}
