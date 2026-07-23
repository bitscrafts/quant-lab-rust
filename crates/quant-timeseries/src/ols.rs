//! Ordinary least squares regression via Gaussian elimination.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::TimeSeriesError;

/// Result of an OLS regression fit.
#[derive(Debug, Clone)]
pub struct OlsFit {
    /// Regression coefficients (including intercept if a constant column was
    /// included in the design matrix).
    pub coeffs: Vec<f64>,
    /// Residuals `y - X * coeffs`.
    pub residuals: Vec<f64>,
    /// Standard errors of the coefficients.
    pub std_errors: Vec<f64>,
    /// t-statistics (`coeff / std_error`).
    pub t_stats: Vec<f64>,
    /// R-squared (coefficient of determination, in `[0, 1]`).
    pub r_squared: f64,
}

/// Fit an OLS regression `y = X * beta + eps` via the normal equations.
///
/// Forms `A = X'X`, `b = X'y`, then solves `A * beta = b` with Gaussian
/// elimination and partial pivoting. Residual variance is
/// `s^2 = sum(e^2) / (n - k)` where `n` is the number of observations and `k`
/// the number of coefficients. Standard errors are
/// `sqrt(diag((X'X)^{-1}) * s^2)`, t-statistics are `coeff / std_error`, and
/// `R^2 = 1 - SSE / SST`.
///
/// # Arguments
/// * `x` - Design matrix as row-major `&[Vec<f64>]`. Each inner `Vec` is one
///   observation; include a column of `1.0` to fit an intercept.
/// * `y` - Response vector, length must equal `x.len()`.
///
/// # Errors
/// - [`TimeSeriesError::DimensionMismatch`] when `x.len() != y.len()`.
/// - [`TimeSeriesError::InsufficientData`] when there are fewer observations
///   than coefficients (under-determined system).
/// - [`TimeSeriesError::Singular`] when `X'X` is singular (collinear columns).
pub fn ols(x: &[Vec<f64>], y: &[f64]) -> Result<OlsFit, TimeSeriesError> {
    let n = x.len();
    if n != y.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            x_rows: n,
            y_len: y.len(),
        });
    }
    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    let k = x[0].len();
    if k == 0 {
        return Err(TimeSeriesError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    if n < k {
        return Err(TimeSeriesError::InsufficientData {
            required: k,
            actual: n,
        });
    }
    // Validate row widths are consistent.
    for (i, row) in x.iter().enumerate() {
        if row.len() != k {
            return Err(TimeSeriesError::DimensionMismatch {
                x_rows: i,
                y_len: row.len(),
            });
        }
    }

    // Form normal equations: A = X'X (k x k), b = X'y (k).
    let mut a = vec![vec![0.0_f64; k]; k];
    let mut b = vec![0.0_f64; k];
    for (xi, yi) in x.iter().zip(y.iter()) {
        for i in 0..k {
            b[i] += xi[i] * yi;
            for j in 0..k {
                a[i][j] += xi[i] * xi[j];
            }
        }
    }

    // Solve A * beta = b via Gaussian elimination (mutates a and b).
    let coeffs = gauss_solve(&mut a, &mut b)?;

    // Residuals e = y - X * beta, and SSE.
    let mut residuals = vec![0.0_f64; n];
    let mut sse = 0.0_f64;
    for (r, (xi, yi)) in x.iter().zip(y.iter()).enumerate() {
        let pred: f64 = xi.iter().zip(coeffs.iter()).map(|(xv, c)| xv * c).sum();
        let e = yi - pred;
        residuals[r] = e;
        sse += e * e;
    }

    // Total sum of squares SST = sum (y - ybar)^2.
    let ybar: f64 = y.iter().sum::<f64>() / n as f64;
    let sst: f64 = y.iter().map(|&v| (v - ybar).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };

    // Residual variance s^2 = SSE / (n - k).
    let dof = (n - k) as f64;
    let s2 = if dof > 0.0 { sse / dof } else { 0.0 };

    // Standard errors: sqrt(diag((X'X)^{-1}) * s^2).
    let std_errors = inverse_diagonal_sqrt(k, x, s2)?;
    let t_stats: Vec<f64> = coeffs
        .iter()
        .zip(std_errors.iter())
        .map(|(&c, &se)| if se > 0.0 { c / se } else { 0.0 })
        .collect();

    Ok(OlsFit {
        coeffs,
        residuals,
        std_errors,
        t_stats,
        r_squared,
    })
}

/// Solve `A * x = b` via Gaussian elimination with partial pivoting.
///
/// Mutates `a` (the matrix) and `b` (the right-hand side). Returns the
/// solution vector or [`TimeSeriesError::Singular`] when a zero pivot is
/// encountered.
pub(crate) fn gauss_solve(
    a: &mut [Vec<f64>],
    b: &mut [f64],
) -> Result<Vec<f64>, TimeSeriesError> {
    let n = a.len();
    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find the row with the largest absolute value in this column at or
        // below the diagonal.
        let mut pivot_row = col;
        let mut max_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().take(n).skip(col + 1) {
            let v = row[col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = r;
            }
        }
        if max_val < 1e-12 {
            return Err(TimeSeriesError::Singular);
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        // Eliminate below.
        let pivot = a[col][col];
        let (a_top, a_bot) = a.split_at_mut(col + 1);
        let pivot_row_a = &a_top[col];
        let pivot_b = b[col];
        for (ar, br) in a_bot.iter_mut().zip(b[col + 1..].iter_mut()) {
            let factor = ar[col] / pivot;
            if factor != 0.0 {
                for c in col..n {
                    ar[c] -= factor * pivot_row_a[c];
                }
                *br -= factor * pivot_b;
            }
        }
    }
    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let ai = &a[i];
        let tail_sum: f64 = ai[i + 1..n].iter().zip(x[i + 1..n].iter()).map(|(a, x)| a * x).sum();
        let diag = ai[i];
        if diag.abs() < 1e-12 {
            return Err(TimeSeriesError::Singular);
        }
        x[i] = (b[i] - tail_sum) / diag;
    }
    Ok(x)
}

/// Compute `sqrt(diag((X'X)^{-1}) * s2)` by solving `X'X * e_j = unit_j` for
/// each column `j` and reading the j-th component of the solution.
///
/// Rebuilds `X'X` from the design matrix, then uses a fresh Gaussian
/// elimination per column. This is O(k^3) per column and O(k^4) overall, fine
/// for the small systems we fit here (k is typically 2..5).
fn inverse_diagonal_sqrt(k: usize, x: &[Vec<f64>], s2: f64) -> Result<Vec<f64>, TimeSeriesError> {
    let n = x.len();
    let mut out = Vec::with_capacity(k);
    for j in 0..k {
        // Build A = X'X again (fresh copy).
        let mut a = vec![vec![0.0_f64; k]; k];
        for xi in x.iter().take(n) {
            for i in 0..k {
                for c in 0..k {
                    a[i][c] += xi[i] * xi[c];
                }
            }
        }
        let mut b = vec![0.0_f64; k];
        b[j] = 1.0;
        let e = gauss_solve(&mut a, &mut b)?;
        let inv_jj = e[j].abs();
        out.push((inv_jj * s2).sqrt());
    }
    Ok(out)
}