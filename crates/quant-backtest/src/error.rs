//! Errors for the AFML backtesting framework.

use thiserror::Error;

/// Backtest error.
#[derive(Debug, Error)]
pub enum BacktestError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("insufficient data: {0}")]
    InsufficientData(String),
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("invalid event: {0}")]
    InvalidEvent(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Result of a single backtest fold.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
    pub train_start: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = BacktestError::InvalidConfig("bad".to_string());
        assert_eq!(e.to_string(), "invalid config: bad");
    }
}
