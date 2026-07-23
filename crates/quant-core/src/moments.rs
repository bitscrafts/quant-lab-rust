//! Moment functions and the `Moments` trait.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::CoreError;

/// Arithmetic mean of a non-empty slice. Returns `0.0` for empty input.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Sample variance (unbiased estimator, denominator `n - 1`).
///
/// # Errors
/// Returns [`CoreError::InsufficientData`] when `data.len() < 2`.
pub fn variance(data: &[f64]) -> Result<f64, CoreError> {
    let n = data.len();
    if n < 2 {
        return Err(CoreError::InsufficientData {
            required: 2,
            actual: n,
        });
    }
    let m = mean(data);
    let ssd: f64 = data.iter().map(|&x| (x - m).powi(2)).sum();
    Ok(ssd / (n - 1) as f64)
}

/// Sample standard deviation: `sqrt(variance)`.
///
/// # Errors
/// Returns [`CoreError::InsufficientData`] when `data.len() < 2`.
pub fn std_dev(data: &[f64]) -> Result<f64, CoreError> {
    variance(data).map(f64::sqrt)
}

/// Population second central moment: `m2 = (1/n) * sum (x_i - mean)^2`.
fn central_moment_2(data: &[f64], mean: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
}

/// Population third central moment: `m3 = (1/n) * sum (x_i - mean)^3`.
fn central_moment_3(data: &[f64], mean: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n
}

/// Population fourth central moment: `m4 = (1/n) * sum (x_i - mean)^4`.
fn central_moment_4(data: &[f64], mean: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n
}

/// Skewness (Fisher-Pearson standardized moment coefficient `g1`).
///
/// Uses population central moments: `g1 = m3 / m2^{3/2}`.
///
/// # Errors
/// Returns [`CoreError::InsufficientData`] when `data.len() < 3` (not enough
/// data for a meaningful third moment), or when the second central moment is
/// zero (constant data).
pub fn skewness(data: &[f64]) -> Result<f64, CoreError> {
    if data.len() < 3 {
        return Err(CoreError::InsufficientData {
            required: 3,
            actual: data.len(),
        });
    }
    let m = mean(data);
    let m2 = central_moment_2(data, m);
    if m2 == 0.0 {
        return Err(CoreError::InsufficientData {
            required: 3,
            actual: data.len(),
        });
    }
    let m3 = central_moment_3(data, m);
    Ok(m3 / m2.powf(1.5))
}

/// Excess kurtosis: `g2 = m4 / m2^2 - 3`.
///
/// Uses population central moments. The Gaussian distribution has excess
/// kurtosis 0; a distribution with heavier tails than Gaussian has positive
/// excess kurtosis.
///
/// # Errors
/// Returns [`CoreError::InsufficientData`] when `data.len() < 4`, or when the
/// second central moment is zero (constant data).
pub fn excess_kurtosis(data: &[f64]) -> Result<f64, CoreError> {
    if data.len() < 4 {
        return Err(CoreError::InsufficientData {
            required: 4,
            actual: data.len(),
        });
    }
    let m = mean(data);
    let m2 = central_moment_2(data, m);
    if m2 == 0.0 {
        return Err(CoreError::InsufficientData {
            required: 4,
            actual: data.len(),
        });
    }
    let m4 = central_moment_4(data, m);
    Ok(m4 / (m2 * m2) - 3.0)
}

/// Trait for types that expose sample moments.
///
/// Implemented for `&[f64]`, `Vec<f64>`, and `PriceSeries`. Higher-moment
/// methods return `Result` because they require a minimum number of
/// observations and non-constant data.
pub trait Moments {
    /// Arithmetic mean.
    fn mean(&self) -> f64;
    /// Sample variance (n - 1 denominator).
    fn variance(&self) -> Result<f64, CoreError>;
    /// Sample standard deviation.
    fn std_dev(&self) -> Result<f64, CoreError>;
    /// Skewness `g1`.
    fn skewness(&self) -> Result<f64, CoreError>;
    /// Excess kurtosis `g2`.
    fn excess_kurtosis(&self) -> Result<f64, CoreError>;
}

impl Moments for &[f64] {
    fn mean(&self) -> f64 {
        mean(self)
    }
    fn variance(&self) -> Result<f64, CoreError> {
        variance(self)
    }
    fn std_dev(&self) -> Result<f64, CoreError> {
        std_dev(self)
    }
    fn skewness(&self) -> Result<f64, CoreError> {
        skewness(self)
    }
    fn excess_kurtosis(&self) -> Result<f64, CoreError> {
        excess_kurtosis(self)
    }
}

impl Moments for Vec<f64> {
    fn mean(&self) -> f64 {
        mean(self.as_slice())
    }
    fn variance(&self) -> Result<f64, CoreError> {
        variance(self.as_slice())
    }
    fn std_dev(&self) -> Result<f64, CoreError> {
        std_dev(self.as_slice())
    }
    fn skewness(&self) -> Result<f64, CoreError> {
        skewness(self.as_slice())
    }
    fn excess_kurtosis(&self) -> Result<f64, CoreError> {
        excess_kurtosis(self.as_slice())
    }
}

impl Moments for crate::series::PriceSeries {
    fn mean(&self) -> f64 {
        mean(self.as_slice())
    }
    fn variance(&self) -> Result<f64, CoreError> {
        variance(self.as_slice())
    }
    fn std_dev(&self) -> Result<f64, CoreError> {
        std_dev(self.as_slice())
    }
    fn skewness(&self) -> Result<f64, CoreError> {
        skewness(self.as_slice())
    }
    fn excess_kurtosis(&self) -> Result<f64, CoreError> {
        excess_kurtosis(self.as_slice())
    }
}