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