//! GARCH(p, q) volatility model (Bollerslev 1986).
//!
//! See `README.md` in this directory for the module overview.

use crate::error::VolError;
use crate::nelder_mead::multistart;

/// `ln(2 * pi)` precomputed for the Gaussian log-likelihood.
const LN_2PI: f64 = 1.8378770664093453;

/// GARCH(p, q) model:
/// `sigma_t^2 = omega + sum_{i=1..q} alpha_i * r_{t-i}^2 + sum_{j=1..p} beta_j * sigma_{t-j}^2`.
#[derive(Debug, Clone)]
pub struct GarchModel {
    /// Constant term (`omega > 0`).
    pub omega: f64,
    /// ARCH coefficients (`alpha_i >= 0`), length `q`.
    pub alphas: Vec<f64>,
    /// GARCH coefficients (`beta_j >= 0`), length `p`.
    pub betas: Vec<f64>,
}

impl GarchModel {
    /// Fit a GARCH(p, q) model to a return series via Gaussian MLE.
    ///
    /// The optimiser is a hand-rolled Nelder-Mead simplex search over the
    /// `1 + p + q` parameters `(omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p)`.
    /// Positivity is enforced by a penalty barrier; stationarity
    /// (`persistence < 1`) is encouraged by the initialisation but not
    /// hard-enforced during the search.
    ///
    /// # Arguments
    /// * `returns` - Return series.
    /// * `p` - GARCH order (lagged conditional variances).
    /// * `q` - ARCH order (lagged squared returns).
    ///
    /// # Errors
    /// - [`VolError::InvalidParam`] when `p == 0` or `q == 0`.
    /// - [`VolError::InsufficientData`] when the series is too short.
    /// - [`VolError::ConvergenceFailure`] when the optimiser returns invalid
    ///   parameters.
    pub fn fit(returns: &[f64], p: usize, q: usize) -> Result<Self, VolError> {
        if p == 0 || q == 0 {
            return Err(VolError::InvalidParam(
                "p and q must be >= 1 for GARCH".to_string(),
            ));
        }
        let n = returns.len();
        let required = p + q + 3;
        if n < required {
            return Err(VolError::InsufficientData {
                required,
                actual: n,
            });
        }

        let sample_var = sample_variance(returns).max(1e-10);

        // Parameterisation (unconstrained → constrained):
        //   t0           → omega = exp(t0)
        //   t1..tq       → alphas via sigmoid, each in (0, 0.49/q)
        //   t_{q+1}..    → betas via sigmoid, each in (0, (0.998 - sum(alpha))/p)
        // This guarantees omega > 0, all alphas/betas >= 0, and
        // persistence < 0.998 (hard cap, prevents IGARCH boundary).
        //
        // Initial guess for GARCH(1,1): omega ≈ 0.05 * sample_var,
        // alpha ≈ 0.05, beta ≈ 0.90.
        //   alpha = sigmoid(t1) * 0.49 → t1 = ln(0.05/0.44) ≈ -2.18
        //   beta = sigmoid(t2) * (0.998 - 0.05) → t2 = ln(0.90/0.048) ≈ 2.93
        let mut t0 = vec![(0.05 * sample_var).ln()];
        for _ in 0..q {
            t0.push(-2.18); // alpha ≈ 0.05
        }
        t0.extend(std::iter::repeat_n(2.93, p)); // beta ≈ 0.90

        let ll = |params: &[f64]| -> f64 {
            let model = GarchModel::from_transformed(params, p, q);
            model.log_likelihood(returns)
        };

        let result = multistart(ll, &t0, 0.3, 5000, 1e-10);
        Ok(GarchModel::from_transformed(&result.x, p, q))
    }

    /// Construct a `GarchModel` from unconstrained transformed parameters.
    /// Uses sigmoid-based transforms with a hard persistence cap at 0.998.
    fn from_transformed(params: &[f64], p: usize, q: usize) -> Self {
        let omega = params[0].clamp(-30.0, 30.0).exp();
        let alpha_max = 0.49;
        let alphas: Vec<f64> = (0..q)
            .map(|i| sigmoid(params[1 + i].clamp(-30.0, 30.0)) * alpha_max / q as f64)
            .collect();
        let alpha_sum: f64 = alphas.iter().sum();
        let beta_max = (0.998 - alpha_sum).max(1e-6);
        let betas: Vec<f64> = (0..p)
            .map(|j| sigmoid(params[1 + q + j].clamp(-30.0, 30.0)) * beta_max / p as f64)
            .collect();
        GarchModel {
            omega,
            alphas,
            betas,
        }
    }

    /// Compute the conditional variance path `sigma_t^2` for the given returns.
    ///
    /// Initial variances before `max(p, q)` are set to the unconditional
    /// sample variance of `returns`.
    pub fn conditional_variances(&self, returns: &[f64]) -> Vec<f64> {
        let n = returns.len();
        let q = self.alphas.len();
        let p = self.betas.len();
        let warmup = p.max(q);
        let init_var = if n > 0 {
            sample_variance(returns)
        } else {
            self.omega
        };
        let mut sigma2 = vec![init_var; n];
        for t in warmup..n {
            let mut s = self.omega;
            for i in 0..q {
                let r = returns[t - 1 - i];
                s += self.alphas[i] * r * r;
            }
            for j in 0..p {
                s += self.betas[j] * sigma2[t - 1 - j];
            }
            sigma2[t] = s.clamp(1e-300, 1e15);
        }
        sigma2
    }

    /// Gaussian log-likelihood of the returns under this model.
    ///
    /// `L = -0.5 * sum_t [log(2*pi) + log(sigma_t^2) + r_t^2 / sigma_t^2]`.
    pub fn log_likelihood(&self, returns: &[f64]) -> f64 {
        let sigma2 = self.conditional_variances(returns);
        let warmup = self.alphas.len().max(self.betas.len());
        let mut ll = 0.0_f64;
        for (t, &r) in returns.iter().enumerate() {
            if t < warmup {
                continue; // skip warmup period (initialisation noise)
            }
            let s2 = sigma2[t];
            if s2 <= 0.0 || !s2.is_finite() {
                return f64::NEG_INFINITY;
            }
            ll -= 0.5 * (LN_2PI + s2.ln() + r * r / s2);
        }
        ll
    }

    /// Persistence: `sum(alpha) + sum(beta)`. Must be `< 1` for covariance
    /// stationarity.
    pub fn persistence(&self) -> f64 {
        self.alphas.iter().sum::<f64>() + self.betas.iter().sum::<f64>()
    }

    /// Long-run (unconditional) variance: `omega / (1 - persistence)`.
    /// Returns `infinity` when `persistence >= 1` (non-stationary).
    pub fn long_run_variance(&self) -> f64 {
        let pers = self.persistence();
        if pers < 1.0 {
            self.omega / (1.0 - pers)
        } else {
            f64::INFINITY
        }
    }

    /// Half-life of volatility shocks (in periods): `log(0.5) / log(persistence)`.
    /// Returns `infinity` when `persistence >= 1`.
    pub fn half_life(&self) -> f64 {
        let pers = self.persistence();
        if pers > 0.0 && pers < 1.0 {
            (0.5_f64).ln() / pers.ln()
        } else {
            f64::INFINITY
        }
    }

    /// Forecast conditional variance `horizon` steps ahead from the end of
    /// `returns`.
    ///
    /// The multi-step forecast follows:
    /// `E[sigma_{t+h}^2] = omega + persistence * E[sigma_{t+h-1}^2]`,
    /// which reverts geometrically to the long-run variance.
    pub fn forecast_from(&self, returns: &[f64], horizon: usize) -> Vec<f64> {
        let sigma2 = self.conditional_variances(returns);
        let last = *sigma2.last().unwrap_or(&self.omega);
        let pers = self.persistence();
        let mut forecast = Vec::with_capacity(horizon);
        let mut prev = last;
        for _ in 0..horizon {
            let next = self.omega + pers * prev;
            forecast.push(next);
            prev = next;
        }
        forecast
    }

    /// Forecast from the long-run variance (no data conditioning).
    pub fn forecast(&self, horizon: usize) -> Vec<f64> {
        let long_run = self.long_run_variance();
        let pers = self.persistence();
        let mut forecast = Vec::with_capacity(horizon);
        let mut prev = long_run;
        for _ in 0..horizon {
            let next = self.omega + pers * prev;
            forecast.push(next);
            prev = next;
        }
        forecast
    }
}

fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
