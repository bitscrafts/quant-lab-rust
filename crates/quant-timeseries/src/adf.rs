//! Augmented Dickey-Fuller test for stationarity.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::TimeSeriesError;
use crate::ols::ols;

/// MacKinnon (1996) critical value at 5% significance for the ADF test with a
/// constant and no trend. Reject the unit-root null when the test statistic
/// is below this value.
pub const MACKINNON_5PCT: f64 = -2.86;

/// Result of an ADF test.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// ADF test statistic (t-ratio on the lagged-level coefficient).
    pub statistic: f64,
    /// Critical value at 5% significance (MacKinnon: -2.86).
    pub critical_value: f64,
    /// Number of lagged differences used in the test regression.
    pub lags: usize,
    /// Whether the null hypothesis (unit root) is rejected: `statistic <
    /// critical_value`.
    pub is_stationary: bool,
}

/// Perform the Augmented Dickey-Fuller test for a unit root.
///
/// Tests `H_0: unit root present (non-stationary)` against `H_1: stationary`.
/// The test regression is
///
/// ```text
/// Delta y_t = alpha + gamma * y_{t-1} + sum_{i=1..p} delta_i * Delta y_{t-i} + eps_t
/// ```
///
/// The test statistic is the t-ratio on `gamma`. If `statistic < -2.86` we
/// reject `H_0` and conclude the series is stationary.
///
/// # Arguments
/// * `data` - Time series to test.
/// * `lags` - Number of lagged differences to include (controls for serial
///   correlation). `lags = 0` fits the simple Dickey-Fuller regression.
///
/// # Errors
/// - [`TimeSeriesError::InsufficientData`] when the series is too short for
///   the requested lag count (need at least `lags + 3` observations to form a
///   non-trivial regression).
pub fn adf_test(data: &[f64], lags: usize) -> Result<AdfResult, TimeSeriesError> {
    let n = data.len();
    // Need at least enough points to form the regression: y_{t-1}, the
    // difference, lags lagged differences, and an intercept. In practice
    // lags + 3 is the safe minimum.
    let required = lags + 3;
    if n < required {
        return Err(TimeSeriesError::InsufficientData {
            required,
            actual: n,
        });
    }

    // First differences Delta y_t = y_t - y_{t-1}, length n - 1.
    let dy: Vec<f64> = (1..n).map(|t| data[t] - data[t - 1]).collect();

    // Build the regression rows. For each t in [lags+1 .. n-1] (index into dy),
    // the response is dy[t] and the regressors are:
    //   [1.0 (intercept), y_{t} (lagged level), dy[t-1], dy[t-2], ..., dy[t-lags]]
    // Note: dy[t] corresponds to y_{t+1} - y_t, so the lagged level is y_t =
    // data[t]. We index data directly to avoid confusion.
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    // t indexes into the original data; dy index is t - 1.
    // We need t >= lags + 1 so dy[t-1], dy[t-2], ..., dy[t-lags] exist
    // (dy index >= 0), and t < n so dy[t-1] is the response.
    for t in (lags + 1)..n {
        let mut row = Vec::with_capacity(lags + 2);
        row.push(1.0); // intercept
        row.push(data[t - 1]); // lagged level y_{t-1}
        for i in 1..=lags {
            // Delta y_{t-i} = data[t-i] - data[t-i-1] = dy[(t-i) - 1] = dy[t-i-1]
            row.push(dy[t - 1 - i]);
        }
        y.push(dy[t - 1]); // Delta y_t = data[t] - data[t-1] = dy[t-1]
        x.push(row);
    }

    if x.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            required,
            actual: n,
        });
    }

    let fit = ols(&x, &y)?;

    // The t-statistic on gamma is the coefficient on the lagged level, which
    // is index 1 in the design (intercept=0, level=1, lagged diffs=2..).
    let stat = fit.t_stats.get(1).copied().unwrap_or(0.0);
    let is_stationary = stat < MACKINNON_5PCT;

    Ok(AdfResult {
        statistic: stat,
        critical_value: MACKINNON_5PCT,
        lags,
        is_stationary,
    })
}