//! Principal Component Analysis on a returns matrix.
//!
//! PCA centres the data, forms the sample covariance matrix, and
//! decomposes it via the power method + deflation (see [`crate::eigen`]).
//! The principal components are the eigenvectors of the covariance
//! matrix; the explained-variance ratios are `lambda_i / sum(lambda_i)`.

use crate::eigen::top_k_eigen;
use crate::error::FactorError;

/// Result of a PCA fit.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Eigenvalues in descending order (one per retained component).
    pub eigenvalues: Vec<f64>,
    /// Principal components (each row is a unit eigenvector of the
    /// covariance matrix).
    pub eigenvectors: Vec<Vec<f64>>,
    /// `lambda_i / sum(lambda_i)` — fraction of total variance captured
    /// by component `i`.
    pub explained_variance_ratio: Vec<f64>,
    /// Cumulative sum of `explained_variance_ratio`.
    pub cumulative_variance: Vec<f64>,
    /// Per-column mean of the input (used by [`pca_transform`] and
    /// [`pca_reconstruct`] to keep the centring consistent).
    pub mean: Vec<f64>,
}

/// Fit PCA on a returns matrix `returns` (`T` observations x `N` assets),
/// keeping `n_components` principal components.
///
/// The data is centred by subtracting the per-column mean before the
/// covariance is formed. The covariance estimate uses the `(T-1)`
/// denominator.
pub fn pca(returns: &[Vec<f64>], n_components: usize) -> Result<PcaResult, FactorError> {
    let t = returns.len();
    if t < 2 {
        return Err(FactorError::InsufficientData {
            required: 2,
            actual: t,
        });
    }
    let n = returns[0].len();
    if n == 0 {
        return Err(FactorError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    for (i, row) in returns.iter().enumerate() {
        if row.len() != n {
            return Err(FactorError::DimensionMismatch(format!(
                "row {i} has length {} but the first row has length {n}",
                row.len()
            )));
        }
    }
    if n_components == 0 || n_components > n {
        return Err(FactorError::InvalidParam(format!(
            "n_components={n_components} must satisfy 1 <= n_components <= {n}"
        )));
    }

    // Per-column mean.
    let mean: Vec<f64> = (0..n)
        .map(|j| returns.iter().map(|r| r[j]).sum::<f64>() / t as f64)
        .collect();

    // Sample covariance with (T-1) denominator.
    let mut cov = vec![vec![0.0_f64; n]; n];
    for row in returns {
        for i in 0..n {
            let di = row[i] - mean[i];
            for j in 0..n {
                let dj = row[j] - mean[j];
                cov[i][j] += di * dj;
            }
        }
    }
    let denom = (t - 1) as f64;
    for row in &mut cov {
        for cell in row.iter_mut() {
            *cell /= denom;
        }
    }

    let (eigenvalues, eigenvectors) = top_k_eigen(&cov, n_components)?;

    // Explained variance ratios. The denominator is the trace of the
    // full covariance matrix (sum of all n eigenvalues), which equals
    // the sum of the retained eigenvalues only when n_components == n.
    // For n_components < n we use the trace of the covariance directly,
    // so the ratios correctly reflect the fraction of TOTAL variance
    // captured (they sum to < 1 when components are dropped).
    let trace: f64 = (0..n).map(|i| cov[i][i]).sum();
    if trace <= 0.0 {
        return Err(FactorError::Singular(
            "covariance trace is non-positive (degenerate data)".to_string(),
        ));
    }
    let explained_variance_ratio: Vec<f64> = eigenvalues.iter().map(|&l| l / trace).collect();
    let mut cumulative_variance = Vec::with_capacity(explained_variance_ratio.len());
    let mut acc = 0.0;
    for &r in &explained_variance_ratio {
        acc += r;
        cumulative_variance.push(acc);
    }

    Ok(PcaResult {
        eigenvalues,
        eigenvectors,
        explained_variance_ratio,
        cumulative_variance,
        mean,
    })
}

/// Project `returns` onto the principal components (factor scores).
///
/// Each row of the output is `x_t - mean` dotted with each eigenvector.
/// The result has `returns.len()` rows and `eigenvectors.len()` columns.
pub fn pca_transform(returns: &[Vec<f64>], eigenvectors: &[Vec<f64>], mean: &[f64]) -> Vec<Vec<f64>> {
    returns
        .iter()
        .map(|row| {
            let centred: Vec<f64> = row.iter().zip(mean.iter()).map(|(x, m)| x - m).collect();
            eigenvectors
                .iter()
                .map(|pc| pc.iter().zip(centred.iter()).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect()
}

/// Reconstruct `returns` from `scores` using the principal components.
///
/// Returns `mean + scores * eigenvectors'`. With `k = n` components the
/// reconstruction is exact (up to floating-point error).
pub fn pca_reconstruct(
    scores: &[Vec<f64>],
    eigenvectors: &[Vec<f64>],
    mean: &[f64],
) -> Vec<Vec<f64>> {
    scores
        .iter()
        .map(|row| {
            let mut out = mean.to_vec();
            for (k, score) in row.iter().enumerate() {
                for (out_j, ev) in out.iter_mut().zip(eigenvectors[k].iter()) {
                    *out_j += score * ev;
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_variance_explained_sums_to_one_full() {
        // With all components retained, EVR sums to 1.
        let returns = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        let res = pca(&returns, 2).unwrap();
        let total: f64 = res.explained_variance_ratio.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "EVR sum = {total}");
        // Last cumulative entry must be 1.
        assert!((res.cumulative_variance.last().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pca_reconstruction_full_is_exact() {
        let returns = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 1.0],
            vec![3.0, 5.0, 2.0],
            vec![4.0, 1.0, 4.0],
            vec![5.0, 2.0, 1.0],
        ];
        let res = pca(&returns, 3).unwrap();
        let scores = pca_transform(&returns, &res.eigenvectors, &res.mean);
        let recon = pca_reconstruct(&scores, &res.eigenvectors, &res.mean);
        for (orig, rec) in returns.iter().zip(recon.iter()) {
            for (a, b) in orig.iter().zip(rec.iter()) {
                assert!((a - b).abs() < 1e-6, "reconstruction mismatch: {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_pca_reconstruction_partial_has_error() {
        let returns = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 1.0],
            vec![3.0, 5.0, 2.0],
            vec![4.0, 1.0, 4.0],
            vec![5.0, 2.0, 1.0],
        ];
        let res1 = pca(&returns, 1).unwrap();
        let scores1 = pca_transform(&returns, &res1.eigenvectors, &res1.mean);
        let recon1 = pca_reconstruct(&scores1, &res1.eigenvectors, &res1.mean);
        let err1: f64 = returns
            .iter()
            .zip(recon1.iter())
            .flat_map(|(o, r)| o.iter().zip(r.iter()).map(|(a, b)| (a - b).powi(2)))
            .sum();
        let res2 = pca(&returns, 2).unwrap();
        let scores2 = pca_transform(&returns, &res2.eigenvectors, &res2.mean);
        let recon2 = pca_reconstruct(&scores2, &res2.eigenvectors, &res2.mean);
        let err2: f64 = returns
            .iter()
            .zip(recon2.iter())
            .flat_map(|(o, r)| o.iter().zip(r.iter()).map(|(a, b)| (a - b).powi(2)))
            .sum();
        assert!(err1 > err2, "1-component error {err1} should exceed 2-component error {err2}");
        assert!(err2 > 1e-9, "2-component error should be nonzero");
    }

    #[test]
    fn test_pca_mean_centered() {
        let returns = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let res = pca(&returns, 2).unwrap();
        assert!((res.mean[0] - 3.0).abs() < 1e-12);
        assert!((res.mean[1] - 4.0).abs() < 1e-12);
    }
}