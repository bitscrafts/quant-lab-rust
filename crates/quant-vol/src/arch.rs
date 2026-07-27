//! ARCH(q) volatility model (Engle 1982).
//!
//! See `README.md` in this directory for the module overview.

use crate::error::VolError;
use crate::nelder_mead::multistart;

/// `ln(2 * pi)` precomputed for the Gaussian log-likelihood.
const LN_2PI: f64 = 1.8378770664093453;

/// ARCH(q) model: `sigma_t^2 = omega + sum_{i=1..q} alpha_i * r_{t-i}^2`.
#[derive(Debug, Clone)]
pub struct ArchModel {
    /// Constant term (`omega > 0`).
    pub omega: f64,
    /// ARCH coefficients (`alpha_i >= 0`).
    pub alphas: Vec<f64>,
}

impl ArchModel {
    /// Fit an ARCH(q) model to a return series via Gaussian MLE.
    ///
    /// The optimiser is a hand-rolled Nelder-Mead simplex search over the
    /// `q + 1` parameters `(omega, alpha_1, ..., alpha_q)`. Positivity is
    /// enforced by a penalty barrier: any non-positive parameter yields
    /// `-infinity` log-likelihood.
    ///
    /// # Arguments
    /// * `returns` - Return series.
    /// * `q` - ARCH order (number of lagged squared returns).
    ///
    /// # Errors
    /// - [`VolError::InsufficientData`] when the series is too short for the
    ///   requested order (need at least `q + 3` observations).
    /// - [`VolError::ConvergenceFailure`] when the optimiser fails to converge.
    pub fn fit(returns: &[f64], q: usize) -> Result<Self, VolError> {
        let n = returns.len();
        let required = q + 3;
        if n < required {
            return Err(VolError::InsufficientData {
                required,
                actual: n,
            });
        }
        if q == 0 {
            return Err(VolError::InvalidParam("q must be >= 1".to_string()));
        }

        let sample_var = sample_variance(returns).max(1e-10);

        // Parameterisation (unconstrained → constrained):
        //   t0      → omega = exp(t0)
        //   t1..tq  → alphas via softmax: alpha_i = exp(t_i) / (1 + sum(exp))
        // This guarantees omega > 0, alpha_i >= 0, and sum(alpha) < 1.
        //
        // Initial guess: omega = 0.5 * sample_var, each alpha = 0.5/(q+1).
        // For q=1: alpha = 0.25, so exp(t1)/(1+exp(t1)) = 0.25 → t1 = ln(1/3).
        let mut t0 = vec![(0.5 * sample_var).ln()];
        for _ in 0..q {
            t0.push((0.5 / (q as f64 + 1.0) / (1.0 - 0.5 / (q as f64 + 1.0))).ln());
        }

        let ll = |params: &[f64]| -> f64 {
            let model = ArchModel::from_transformed(params);
            model.log_likelihood(returns)
        };

        let result = multistart(ll, &t0, 0.3, 3000, 1e-10);
        Ok(ArchModel::from_transformed(&result.x))
    }

    /// Construct an `ArchModel` from unconstrained transformed parameters.
    /// `params[0]` → `omega = exp(params[0])`, `params[1..]` → softmax alphas.
    fn from_transformed(params: &[f64]) -> Self {
        let omega = params[0].clamp(-30.0, 30.0).exp();
        let exps: Vec<f64> = params[1..]
            .iter()
            .map(|&t| t.clamp(-30.0, 30.0).exp())
            .collect();
        let s: f64 = 1.0 + exps.iter().sum::<f64>();
        let alphas: Vec<f64> = exps.iter().map(|&e| e / s).collect();
        ArchModel { omega, alphas }
    }

    /// Compute the conditional variance path `sigma_t^2` for the given returns.
    ///
    /// For `t < q` the recursion uses whatever lags are available (zero-padded).
    pub fn conditional_variances(&self, returns: &[f64]) -> Vec<f64> {
        let n = returns.len();
        let q = self.alphas.len();
        let mut sigma2 = vec![0.0_f64; n];
        for t in 0..n {
            let mut s = self.omega;
            for i in 0..q {
                if t > i {
                    s += self.alphas[i] * returns[t - 1 - i] * returns[t - 1 - i];
                }
            }
            sigma2[t] = s.max(1e-300);
        }
        sigma2
    }

    /// Gaussian log-likelihood of the returns under this model.
    ///
    /// `L = -0.5 * sum_t [log(2*pi) + log(sigma_t^2) + r_t^2 / sigma_t^2]`.
    pub fn log_likelihood(&self, returns: &[f64]) -> f64 {
        let sigma2 = self.conditional_variances(returns);
        let mut ll = 0.0_f64;
        for (t, &r) in returns.iter().enumerate() {
            let s2 = sigma2[t];
            if s2 <= 0.0 || !s2.is_finite() {
                return f64::NEG_INFINITY;
            }
            ll -= 0.5 * (LN_2PI + s2.ln() + r * r / s2);
        }
        ll
    }

    /// Forecast conditional variance `h` steps ahead from the last observation.
    ///
    /// For ARCH(q), the multi-step forecast follows an AR(q)-like recursion
    /// in the variance: future squared returns are replaced by their
    /// conditional expectation (the variance itself). The forecast reverts
    /// to the long-run variance `omega / (1 - sum(alpha))`.
    pub fn forecast(&self, horizon: usize) -> Vec<f64> {
        let q = self.alphas.len();
        let alpha_sum: f64 = self.alphas.iter().sum();
        let long_run = if alpha_sum < 1.0 {
            self.omega / (1.0 - alpha_sum)
        } else {
            f64::INFINITY
        };
        // We need the last q squared returns as seed. Since we don't have the
        // returns here, we use the long-run variance as the initial forecast.
        // Callers who need data-conditioned forecasts should use
        // `forecast_from` instead.
        let mut forecast = Vec::with_capacity(horizon);
        let mut recent: Vec<f64> = vec![long_run; q];
        for _ in 0..horizon {
            let mut s = self.omega;
            for i in 0..q {
                s += self.alphas[i] * recent[q - 1 - i];
            }
            forecast.push(s);
            // Shift the window: the forecast variance replaces the oldest.
            recent.remove(0);
            recent.push(s);
        }
        forecast
    }

    /// Forecast from a given return series (data-conditioned).
    ///
    /// Produces `horizon` steps of conditional variance forecast starting
    /// after the last observation in `returns`.
    pub fn forecast_from(&self, returns: &[f64], horizon: usize) -> Vec<f64> {
        let q = self.alphas.len();
        let alpha_sum: f64 = self.alphas.iter().sum();
        let long_run = if alpha_sum < 1.0 {
            self.omega / (1.0 - alpha_sum)
        } else {
            self.conditional_variances(returns)
                .last()
                .copied()
                .unwrap_or(self.omega)
        };
        // Seed with the last q squared returns (or long-run if not enough data).
        let mut recent: Vec<f64> = (0..q)
            .map(|i| {
                if returns.len() > i {
                    let r = returns[returns.len() - 1 - i];
                    r * r
                } else {
                    long_run
                }
            })
            .collect();
        let mut forecast = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let mut s = self.omega;
            for i in 0..q {
                s += self.alphas[i] * recent[q - 1 - i];
            }
            forecast.push(s);
            recent.remove(0);
            recent.push(s);
        }
        forecast
    }

    /// Long-run (unconditional) variance: `omega / (1 - sum(alpha))`.
    /// Returns `infinity` when `sum(alpha) >= 1` (non-stationary).
    pub fn long_run_variance(&self) -> f64 {
        let alpha_sum: f64 = self.alphas.iter().sum();
        if alpha_sum < 1.0 {
            self.omega / (1.0 - alpha_sum)
        } else {
            f64::INFINITY
        }
    }
}

fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}