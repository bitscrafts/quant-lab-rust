//! Loan application data structures and loading.

use qf_common::CommonError;
use serde::Deserialize;
use std::path::Path;

/// Home ownership status.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum HomeOwnership {
    #[serde(rename = "RENT")]
    Rent,
    #[serde(rename = "OWN")]
    Own,
    #[serde(rename = "MORTGAGE")]
    Mortgage,
}

/// Loan application with features and default status.
#[derive(Debug, Clone, Deserialize)]
pub struct LoanApplication {
    pub income: f64,
    pub loan_amount: f64,
    pub interest_rate: f64,
    pub dti: f64,  // debt-to-income ratio
    pub grade: u8,  // A=1, B=2, ..., G=7
    pub home_ownership: HomeOwnership,
    pub defaulted: bool,
}

impl LoanApplication {
    /// Create a new loan application.
    pub fn new(
        income: f64,
        loan_amount: f64,
        interest_rate: f64,
        dti: f64,
        grade: u8,
        home_ownership: HomeOwnership,
        defaulted: bool,
    ) -> Self {
        Self {
            income,
            loan_amount,
            interest_rate,
            dti,
            grade,
            home_ownership,
            defaulted,
        }
    }
}

/// Load loan applications from CSV file.
///
/// Handles missing values gracefully by skipping rows with parse errors.
pub fn load_loans(path: &Path) -> Result<Vec<LoanApplication>, CommonError> {
    // Check if file exists first
    if !path.exists() {
        return Err(CommonError::FileNotFound(path.display().to_string()));
    }
    
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| CommonError::IoError(std::io::Error::other(e.to_string())))?;
    
    // Skip rows that fail to parse (missing values, etc.)
    let loans = reader.deserialize().flatten().collect();
    
    Ok(loans)
}
