//! Small dense linear-algebra helpers for the portfolio crate.
//!
//! See `README.md` in this directory for the module overview.
//!
//! Everything here is O(n^3) Gaussian elimination with partial pivoting, sized
//! for the small covariance matrices that show up in portfolio theory (n is
//! typically 2..20 assets). No `nalgebra` dependency — the pedagogy stays in
//! the implementation.

use crate::error::PortfolioError;

/// Solve the linear system `A * x = b` via Gaussian elimination with partial
/// pivoting.
///
/// `a` is row-major: `a[i][j]` is row `i`, column `j`. Both `a` and `b` are
/// consumed (mutated) by the elimination. Returns the solution vector `x`.
///
/// # Errors
/// - [`PortfolioError::DimensionMismatch`] when `a.len() != b.len()` or `a`
///   is not square.
/// - [`PortfolioError::SingularCovariance`] when a zero pivot is encountered.
pub fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, PortfolioError> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n != b.len() {
        return Err(PortfolioError::DimensionMismatch(format!(
            "A rows ({n}) != b len ({})",
            b.len()
        )));
    }
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "row {i} has length {} != {n}",
                row.len()
            )));
        }
    }
    // Forward elimination with partial pivoting.
    for col in 0..n {
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
            return Err(PortfolioError::SingularCovariance(format!(
                "zero pivot at column {col}"
            )));
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
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
        let tail_sum: f64 = ai[i + 1..n]
            .iter()
            .zip(x[i + 1..n].iter())
            .map(|(a, x)| a * x)
            .sum();
        let diag = ai[i];
        if diag.abs() < 1e-12 {
            return Err(PortfolioError::SingularCovariance(format!(
                "zero diagonal at row {i}"
            )));
        }
        x[i] = (b[i] - tail_sum) / diag;
    }
    Ok(x)
}

/// Invert a square matrix by solving `A * X = I` column by column.
///
/// Returns the inverse as a row-major `Vec<Vec<f64>>` or
/// [`PortfolioError::SingularCovariance`] when the matrix is singular.
pub fn inverse(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PortfolioError> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "row {i} has length {} != {n}",
                row.len()
            )));
        }
    }
    let mut inv = vec![vec![0.0_f64; n]; n];
    for j in 0..n {
        let mut a_copy: Vec<Vec<f64>> = a.to_vec();
        let mut e = vec![0.0_f64; n];
        e[j] = 1.0;
        let col = solve(&mut a_copy, &mut e)?;
        for (i, row) in inv.iter_mut().enumerate() {
            row[j] = col[i];
        }
    }
    Ok(inv)
}

/// Matrix-vector product `y = A * x`.
///
/// `a` is row-major; length of each row must equal `x.len()`.
pub fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, x)| a * x).sum())
        .collect()
}

/// Matrix-matrix product `C = A * B` (both row-major).
pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    let k = b.len();
    let m = if k > 0 { b[0].len() } else { 0 };
    // Transpose b for column access.
    let mut bt = vec![vec![0.0_f64; k]; m];
    for (i, b_row) in b.iter().enumerate().take(k) {
        for (j, val) in b_row.iter().enumerate().take(m) {
            bt[j][i] = *val;
        }
    }
    let mut c = vec![vec![0.0_f64; m]; n];
    for (i, c_row) in c.iter_mut().enumerate().take(n) {
        let a_row = &a[i];
        for (j, cij) in c_row.iter_mut().enumerate().take(m) {
            *cij = a_row.iter().zip(bt[j].iter()).map(|(a, b)| a * b).sum();
        }
    }
    c
}

/// Quadratic form `w' * A * w` for a symmetric `A`.
pub fn quadratic_form(w: &[f64], a: &[Vec<f64>]) -> f64 {
    let aw = matvec(a, w);
    w.iter().zip(aw.iter()).map(|(w, aw)| w * aw).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn solve_identity() {
        let mut a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut b = vec![3.0, 5.0];
        let x = solve(&mut a, &mut b).unwrap();
        assert_abs_diff_eq!(x[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn solve_2x2_system() {
        // 2x +  y = 5
        //  x + 3y = 10  -> x=1, y=3
        let mut a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let mut b = vec![5.0, 10.0];
        let x = solve(&mut a, &mut b).unwrap();
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn inverse_diagonal() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let inv = inverse(&a).unwrap();
        assert_abs_diff_eq!(inv[0][0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(inv[1][1], 0.25, epsilon = 1e-12);
    }

    #[test]
    fn inverse_times_original_is_identity() {
        let a = vec![vec![4.0, 2.0], vec![1.0, 3.0]];
        let inv = inverse(&a).unwrap();
        let prod = matmul(&a, &inv);
        assert_abs_diff_eq!(prod[0][0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(prod[1][1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(prod[0][1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(prod[1][0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn quadratic_form_2d() {
        // w = (1, 2), A = [[1, 0],[0, 1]] -> w'Aw = 1 + 4 = 5
        let w = vec![1.0, 2.0];
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_abs_diff_eq!(quadratic_form(&w, &a), 5.0, epsilon = 1e-12);
    }
}
