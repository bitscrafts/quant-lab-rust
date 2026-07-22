//! Common utilities for quantitative finance projects.
//!
//! This crate provides shared data structures and functions used across
//! multiple quant-finance projects, including CSV loading, statistics
//! computation, and common types.

use std::path::Path;
use thiserror::Error;

/// Errors that can occur in qf-common operations.
#[derive(Error, Debug)]
pub enum CommonError {
    /// File not found at the specified path.
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    /// Error parsing CSV data or converting values.
    #[error("Parse error: {0}")]
    ParseError(String),
    
    /// I/O error occurred.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Represents a single financial transaction.
///
/// Used primarily for fraud detection tasks where transactions have
/// multiple features (anonymized via PCA), an amount, and a class label.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Feature vector (e.g., PCA-transformed features V1, V2, ..., V28).
    pub features: Vec<f64>,
    
    /// Transaction amount in dollars.
    pub amount: f64,
    
    /// Class label: 0 = normal, 1 = fraud.
    pub class: u8,
}

/// Load transactions from a CSV file.
///
/// Expected CSV format:
/// - Header row (skipped)
/// - Columns: Time, V1, V2, ..., VN, Amount, Class
/// - Features are all columns except Time, Amount, and Class
///
/// # Arguments
///
/// * `path` - Path to the CSV file
///
/// # Returns
///
/// Vector of transactions or an error if the file cannot be read or parsed.
///
/// # Errors
///
/// - `CommonError::FileNotFound` if the file doesn't exist
/// - `CommonError::ParseError` if CSV parsing fails
/// - `CommonError::IoError` for other I/O errors
pub fn load_transactions(path: impl AsRef<Path>) -> Result<Vec<Transaction>, CommonError> {
    let path = path.as_ref();
    
    // Check if file exists
    if !path.exists() {
        return Err(CommonError::FileNotFound(path.display().to_string()));
    }
    
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| CommonError::IoError(std::io::Error::other(
            
            e.to_string()
        )))?;
    
    let mut transactions = Vec::new();
    
    for (idx, result) in reader.records().enumerate() {
        let record = result.map_err(|e| CommonError::ParseError(
            format!("Row {}: {}", idx + 2, e)
        ))?;
        
        // Parse all fields except Time (first) and last two (Amount, Class)
        let num_fields = record.len();
        if num_fields < 3 {
            return Err(CommonError::ParseError(
                format!("Row {}: insufficient columns (need at least 3)", idx + 2)
            ));
        }
        
        // Features are columns 1 to (n-2)
        let mut features = Vec::new();
        for i in 1..(num_fields - 2) {
            let value = record.get(i)
                .ok_or_else(|| CommonError::ParseError(
                    format!("Row {}: missing field {}", idx + 2, i)
                ))?
                .parse::<f64>()
                .map_err(|e| CommonError::ParseError(
                    format!("Row {}, field {}: {}", idx + 2, i, e)
                ))?;
            features.push(value);
        }
        
        // Amount is second-to-last column
        let amount = record.get(num_fields - 2)
            .ok_or_else(|| CommonError::ParseError(
                format!("Row {}: missing Amount", idx + 2)
            ))?
            .parse::<f64>()
            .map_err(|e| CommonError::ParseError(
                format!("Row {}, Amount: {}", idx + 2, e)
            ))?;
        
        // Class is last column
        let class = record.get(num_fields - 1)
            .ok_or_else(|| CommonError::ParseError(
                format!("Row {}: missing Class", idx + 2)
            ))?
            .parse::<u8>()
            .map_err(|e| CommonError::ParseError(
                format!("Row {}, Class: {}", idx + 2, e)
            ))?;
        
        transactions.push(Transaction {
            features,
            amount,
            class,
        });
    }
    
    Ok(transactions)
}

/// Basic statistics for a dataset.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Arithmetic mean.
    pub mean: f64,
    
    /// Sample standard deviation.
    pub std: f64,
    
    /// Minimum value.
    pub min: f64,
    
    /// Maximum value.
    pub max: f64,
}

/// Compute basic statistics for a slice of f64 values.
///
/// Uses sample standard deviation (divides by n-1).
///
/// # Arguments
///
/// * `data` - Slice of floating point values
///
/// # Returns
///
/// `Stats` struct with mean, std, min, max.
///
/// # Special Cases
///
/// - Empty slice: mean=NaN, std=0.0, min=NaN, max=NaN
/// - Single element: mean=value, std=0.0, min=value, max=value
pub fn compute_stats(data: &[f64]) -> Stats {
    if data.is_empty() {
        return Stats {
            mean: f64::NAN,
            std: 0.0,
            min: f64::NAN,
            max: f64::NAN,
        };
    }
    
    if data.len() == 1 {
        return Stats {
            mean: data[0],
            std: 0.0,
            min: data[0],
            max: data[0],
        };
    }
    
    // Compute mean
    let sum: f64 = data.iter().sum();
    let mean = sum / data.len() as f64;
    
    // Compute variance (sample: divide by n-1)
    let variance: f64 = data.iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>() / (data.len() - 1) as f64;
    
    let std = variance.sqrt();
    
    // Find min and max
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    Stats {
        mean,
        std,
        min,
        max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transaction_creation() {
        let t = Transaction {
            features: vec![1.0, 2.0, 3.0],
            amount: 100.0,
            class: 0,
        };
        
        assert_eq!(t.features.len(), 3);
        assert_eq!(t.amount, 100.0);
        assert_eq!(t.class, 0);
    }
}
