//! Stock market data analysis tools.
//!
//! This crate provides data structures and functions for working with OHLCV
//! (Open, High, Low, Close, Volume) market data, including candlestick analysis,
//! basic statistics, and moving averages.

use qf_common::CommonError;
use serde::Deserialize;
use std::path::Path;

/// OHLCV (Open, High, Low, Close, Volume) market data for a single day.
#[derive(Debug, Clone, Deserialize)]
pub struct Ohlcv {
    /// Trading date (as string, e.g., "2024-01-15").
    pub date: String,
    
    /// Opening price.
    pub open: f64,
    
    /// Highest price during the period.
    pub high: f64,
    
    /// Lowest price during the period.
    pub low: f64,
    
    /// Closing price.
    pub close: f64,
    
    /// Trading volume.
    pub volume: u64,
    
    /// Adjusted closing price (optional, for stock splits/dividends).
    pub adj_close: Option<f64>,
}

impl Ohlcv {
    /// Returns the daily price range (high - low).
    pub fn daily_range(&self) -> f64 {
        self.high - self.low
    }
    
    /// Returns the candle body (close - open).
    ///
    /// Positive value indicates a bullish (green) candle.
    /// Negative value indicates a bearish (red) candle.
    pub fn body(&self) -> f64 {
        self.close - self.open
    }
    
    /// Returns the upper shadow length.
    ///
    /// Distance from the top of the body to the high.
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }
    
    /// Returns the lower shadow length.
    ///
    /// Distance from the bottom of the body to the low.
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }
    
    /// Returns true if this is a bullish candle (close > open).
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
    
    /// Returns the typical price: (high + low + close) / 3.
    ///
    /// Often used as a representative price for the period.
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Load OHLCV data from a CSV file.
///
/// Expected CSV format:
/// - Header row (skipped)
/// - Columns: Date, Open, High, Low, Close, Volume, [Adj Close]
///
/// # Arguments
///
/// * `path` - Path to the CSV file
///
/// # Returns
///
/// Vector of OHLCV records or an error if the file cannot be read or parsed.
///
/// # Errors
///
/// - `CommonError::FileNotFound` if the file doesn't exist
/// - `CommonError::ParseError` if CSV parsing fails or OHLCV validation fails
/// - `CommonError::IoError` for other I/O errors
pub fn load_ohlcv(path: impl AsRef<Path>) -> Result<Vec<Ohlcv>, CommonError> {
    let path = path.as_ref();
    
    // Check if file exists
    if !path.exists() {
        return Err(CommonError::FileNotFound(path.display().to_string()));
    }
    
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| CommonError::IoError(std::io::Error::other(e.to_string())))?;
    
    let mut data = Vec::new();
    
    for (idx, result) in reader.deserialize().enumerate() {
        let record: Ohlcv = result.map_err(|e| {
            CommonError::ParseError(format!("Row {}: {}", idx + 2, e))
        })?;
        
        // Validate OHLCV relationships
        if record.high < record.low {
            return Err(CommonError::ParseError(format!(
                "Row {}: high ({}) < low ({})",
                idx + 2,
                record.high,
                record.low
            )));
        }
        
        if record.open > record.high || record.open < record.low {
            return Err(CommonError::ParseError(format!(
                "Row {}: open ({}) outside [low, high] range",
                idx + 2,
                record.open
            )));
        }
        
        if record.close > record.high || record.close < record.low {
            return Err(CommonError::ParseError(format!(
                "Row {}: close ({}) outside [low, high] range",
                idx + 2,
                record.close
            )));
        }
        
        data.push(record);
    }
    
    Ok(data)
}

/// Returns the average trading volume across all records.
pub fn average_volume(data: &[Ohlcv]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let sum: u64 = data.iter().map(|ohlcv| ohlcv.volume).sum();
    sum as f64 / data.len() as f64
}

/// Returns the price change as a fraction: (last_close - first_close) / first_close.
///
/// Returns 0.0 if data is empty or has only one element.
pub fn price_change(data: &[Ohlcv]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    
    let first_close = data[0].close;
    let last_close = data[data.len() - 1].close;
    
    (last_close - first_close) / first_close
}

/// Returns the highest high price across all records.
///
/// Returns f64::NEG_INFINITY if data is empty.
pub fn highest_high(data: &[Ohlcv]) -> f64 {
    data.iter()
        .map(|ohlcv| ohlcv.high)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Returns the lowest low price across all records.
///
/// Returns f64::INFINITY if data is empty.
pub fn lowest_low(data: &[Ohlcv]) -> f64 {
    data.iter()
        .map(|ohlcv| ohlcv.low)
        .fold(f64::INFINITY, f64::min)
}

/// Returns the average daily range across all records.
pub fn average_daily_range(data: &[Ohlcv]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let sum: f64 = data.iter().map(|ohlcv| ohlcv.daily_range()).sum();
    sum / data.len() as f64
}

/// Computes the Simple Moving Average (SMA) for the given prices.
///
/// Returns a vector of length `prices.len() - period + 1`.
/// The first SMA value (at index 0) corresponds to the price at index `period - 1`.
///
/// # Arguments
///
/// * `prices` - Slice of prices
/// * `period` - Window size for the moving average
///
/// # Returns
///
/// Vector of SMA values. Returns empty vector if period > prices.len() or period == 0.
pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || period > prices.len() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(prices.len() - period + 1);
    
    for i in 0..=(prices.len() - period) {
        let window = &prices[i..i + period];
        let sum: f64 = window.iter().sum();
        let avg = sum / period as f64;
        result.push(avg);
    }
    
    result
}

/// Computes the Exponential Moving Average (EMA) for the given prices.
///
/// Uses smoothing factor alpha = 2 / (period + 1).
/// The first EMA value is initialized as the first price.
///
/// # Arguments
///
/// * `prices` - Slice of prices
/// * `period` - Period for calculating the smoothing factor
///
/// # Returns
///
/// Vector of EMA values with same length as input prices.
/// Returns empty vector if period == 0 or prices is empty.
pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
    if period == 0 || prices.is_empty() {
        return Vec::new();
    }
    
    let alpha = 2.0 / (period + 1) as f64;
    let mut result = Vec::with_capacity(prices.len());
    
    // Initialize with first price
    result.push(prices[0]);
    
    // Compute EMA recursively: EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
    for i in 1..prices.len() {
        let prev_ema = result[i - 1];
        let current_ema = alpha * prices[i] + (1.0 - alpha) * prev_ema;
        result.push(current_ema);
    }
    
    result
}

/// Trait for time series operations on OHLCV data.
pub trait TimeSeries {
    /// Returns the number of records in the series.
    fn len(&self) -> usize;
    
    /// Returns true if the series is empty.
    fn is_empty(&self) -> bool;
    
    /// Extracts all closing prices as a vector.
    fn closes(&self) -> Vec<f64>;
    
    /// Extracts all volumes as a vector.
    fn volumes(&self) -> Vec<u64>;
}

impl TimeSeries for Vec<Ohlcv> {
    fn len(&self) -> usize {
        self.len()
    }
    
    fn is_empty(&self) -> bool {
        self.is_empty()
    }
    
    fn closes(&self) -> Vec<f64> {
        self.iter().map(|ohlcv| ohlcv.close).collect()
    }
    
    fn volumes(&self) -> Vec<u64> {
        self.iter().map(|ohlcv| ohlcv.volume).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ohlcv_methods_basic() {
        let ohlcv = Ohlcv {
            date: "2024-01-01".to_string(),
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000,
            adj_close: None,
        };
        
        assert_eq!(ohlcv.daily_range(), 10.0);
        assert_eq!(ohlcv.body(), 2.0);
        assert!(ohlcv.is_bullish());
    }
}
