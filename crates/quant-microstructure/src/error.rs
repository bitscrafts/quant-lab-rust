//! Error type for the `quant-microstructure` crate.

use thiserror::Error;

/// Errors returned by order-book and microstructure routines.
#[derive(Debug, Error)]
pub enum MicroError {
    #[error("order not found: id {0}")]
    OrderNotFound(u64),

    #[error("order book error: {0}")]
    InvalidOrder(String),

    #[error("empty order book: {0}")]
    EmptyBook(String),

    #[error("insufficient liquidity: requested {requested}, available {available}")]
    InsufficientLiquidity { requested: u64, available: u64 },

    #[error("invalid tick size: {0}")]
    InvalidTickSize(String),

    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("insufficient data: required {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },
}
