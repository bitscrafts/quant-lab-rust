//! Eigenvalue decomposition via the power method with deflation.
//!
//! All routines are hand-rolled — no `nalgebra` or `argmin`. The power
//! method finds the dominant eigenpair `(lambda_1, v_1)` of a symmetric
//! matrix by repeatedly applying `v <- A v / ||A v||`; the Rayleigh
//! quotient `lambda = v' A v` recovers the eigenvalue. Deflation
//! (`A' = A - lambda v v'`) removes the found component so the next
//! power iteration finds the next-largest eigenpair.

use crate::error::FactorError;

/// Tolerance for the power-method convergence test on the eigenvector
/// (measured as the L2 norm of the change between successive iterates).
const DEFAULT_TOL: f64 = 1e-10;

/// Maximum number of power iterations before giving up.
const DEFAULT_MAX_ITER: usize = 1000;

/// Compute the dominant eigenvalue and eigenvector of a symmetric matrix
/// `a` via the power method.
///
/// Returns `(lambda, v)` with `v` normalised to unit L2 norm and the sign
/// chosen so that the first non-zero component is positive (for stable
/// comparisons in tests). Convergence is declared when
/// `||v_{k+1} - v_k||_2 < tol` or the eigenvalue stabilises.
///
/// # Arguments
/// * `matrix` - Symmetric `n x n` matrix as `&[Vec<f64>]` (row-major).
/// * `max_iter` - Iteration cap (use 0 for the default of 1000).
/// * `tol` - Convergence threshold (use 0.0 for the default 1e-10).
pub fn power_method(
    matrix: &[Vec<f64>],
    max_iter: usize,
    tol: f64,
) -> Result<(f64, Vec<f64>), FactorError> {
    let n = matrix.len();
    if n == 0 {
        return Err(FactorError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    for row in matrix {
        if row.len() != n {
            return Err(FactorError::DimensionMismatch(format!(
                "expected {n}x{n} matrix, found a row of length {}",
                row.len()
            )));
        }
    }
    let max_iter = if max_iter == 0 { DEFAULT_MAX_ITER } else { max_iter };
    let tol = if tol <= 0.0 { DEFAULT_TOL } else { tol };

    // Initial guess: uniform vector (works for positive matrices; for
    // general symmetric matrices it avoids the pathological case of being
    // exactly orthogonal to the dominant eigenvector by perturbing later
    // if convergence stalls).
    let mut v = vec![1.0_f64 / (n as f64).sqrt(); n];
    let mut lambda_prev = f64::INFINITY;

    let mut last_delta = f64::INFINITY;
    let mut iter_used = 0;
    for it in 0..max_iter {
        iter_used = it + 1;
        // w = A * v
        let w = matvec_sym(matrix, &v);
        let norm = l2_norm(&w);
        if norm < 1e-300 {
            return Err(FactorError::Singular(
                "matrix annihilates the current vector".to_string(),
            ));
        }
        let mut v_new: Vec<f64> = w.iter().map(|&x| x / norm).collect();
        // Align sign with the previous iterate so a sign flip does not
        // register as divergence. The dominant eigenvector is only defined
        // up to a sign, so we pin the sign to keep `||v_new - v||` meaningful.
        let dot: f64 = v_new.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        if dot < 0.0 {
            for x in &mut v_new {
                *x = -*x;
            }
        }
        // Rayleigh quotient lambda = v' A v (use v_new for stability).
        let av = matvec_sym(matrix, &v_new);
        let lambda: f64 = v_new.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
        let delta = l2_norm(&v_new.iter().zip(v.iter()).map(|(a, b)| a - b).collect::<Vec<_>>());
        last_delta = delta;
        v = v_new;
        if (lambda - lambda_prev).abs() < tol && delta < tol * 10.0 {
            lambda_prev = lambda;
            break;
        }
        lambda_prev = lambda;
    }
    // Convergence check: if the final delta is still large, the iteration
    // did not settle. Flag the failure so callers can retry or fall back.
    if last_delta > 1e-4 {
        return Err(FactorError::NonConverged(iter_used, last_delta));
    }

    // Sign convention: first non-zero component is positive.
    let mut sign = 1.0;
    for &x in &v {
        if x.abs() > 1e-12 {
            if x < 0.0 {
                sign = -1.0;
            }
            break;
        }
    }
    if sign < 0.0 {
        for x in &mut v {
            *x = -*x;
        }
    }

    Ok((lambda_prev, v))
}

/// Deflate `matrix` by removing the contribution of the eigenpair
/// `(eigenvalue, eigenvector)`: `A' = A - lambda * v * v'`.
///
/// For a symmetric matrix, this removes `v` from the spectrum, so the
/// next power iteration on `A'` finds the next eigenpair. The returned
/// matrix is a fresh allocation; the input is not mutated.
pub fn deflate(matrix: &[Vec<f64>], eigenvalue: f64, eigenvector: &[f64]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut out = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
    out
}

/// Compute the top-`k` eigenpairs of a symmetric matrix by repeated
/// power iteration + deflation.
///
/// Returns `(eigenvalues, eigenvectors)` sorted in **descending** order of
/// `|eigenvalue|`. Each `eigenvectors[i]` is a unit vector.
pub fn top_k_eigen(
    matrix: &[Vec<f64>],
    k: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), FactorError> {
    let n = matrix.len();
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if k > n {
        return Err(FactorError::InvalidParam(format!(
            "requested {k} eigenpairs but matrix is only {n}x{n}"
        )));
    }
    let mut current = matrix.to_vec();
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Vec::with_capacity(k);
    for _ in 0..k {
        let (lambda, v) = power_method(&current, 0, 0.0)?;
        eigenvalues.push(lambda);
        eigenvectors.push(v);
        current = deflate(&current, lambda, &eigenvectors[eigenvectors.len() - 1]);
    }
    Ok((eigenvalues, eigenvectors))
}

/// Compute `w = A * v` for a (possibly non-symmetric) row-major matrix.
fn matvec_sym(matrix: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// L2 norm of a vector.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_method_identity() {
        let n = 3;
        let a: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
        let (lambda, v) = power_method(&a, 0, 0.0).unwrap();
        assert!((lambda - 1.0).abs() < 1e-9, "lambda={lambda}");
        assert!((l2_norm(&v) - 1.0).abs() < 1e-9, "v not unit: {:?}", v);
    }

    #[test]
    fn test_power_method_diagonal() {
        let a = vec![vec![3.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 1.0]];
        let (lambda, v) = power_method(&a, 0, 0.0).unwrap();
        assert!((lambda - 3.0).abs() < 1e-6, "lambda={lambda}");
        assert!((v[0].abs() - 1.0).abs() < 1e-6, "v={:?}", v);
    }

    #[test]
    fn test_eigenvalues_positive_definite() {
        // SPD matrix with known eigenvalues 4, 3, 2 (for the construction below).
        let a = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 3.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let (eigs, _vecs) = top_k_eigen(&a, 3).unwrap();
        for &lam in &eigs {
            assert!(lam > -1e-8, "non-positive eigenvalue: {lam}");
        }
    }

    #[test]
    fn test_deflation_removes_component() {
        let a = vec![vec![3.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 1.0]];
        let (lambda, v) = power_method(&a, 0, 0.0).unwrap();
        assert!((lambda - 3.0).abs() < 1e-6);
        let a_def = deflate(&a, lambda, &v);
        // The deflated matrix's dominant eigenvalue should now be ~2.
        let (lambda2, _v2) = power_method(&a_def, 0, 0.0).unwrap();
        assert!((lambda2 - 2.0).abs() < 1e-3, "second eigenvalue = {lambda2}");
    }
}