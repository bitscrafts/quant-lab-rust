//! Nelder-Mead simplex optimiser (derivative-free).
//!
//! See `README.md` in this directory for the module overview.

/// Result of a Nelder-Mead optimisation run.
#[derive(Debug, Clone)]
pub struct OptResult {
    /// Best parameter vector found.
    pub x: Vec<f64>,
    /// Objective value at `x` (the optimiser maximises).
    pub f: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the optimiser converged within `max_iter`.
    pub converged: bool,
}

/// Maximise `f` over unconstrained `n`-dimensional parameters using
/// Nelder-Mead.
///
/// The caller supplies a penalty wrapper for constraint violations
/// (return `f64::NEG_INFINITY` when constraints are broken). The initial
/// simplex is built from `x0` with each vertex perturbed by `step` in one
/// coordinate.
///
/// # Arguments
/// * `f` - Objective function to maximise.
/// * `x0` - Initial guess (also the first simplex vertex).
/// * `step` - Initial simplex edge length.
/// * `max_iter` - Maximum iterations.
/// * `tol` - Convergence tolerance on both simplex size and function spread.
pub fn maximize<F>(f: F, x0: &[f64], step: f64, max_iter: usize, tol: f64) -> OptResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return OptResult { x: Vec::new(), f: f(&[]), iterations: 0, converged: true };
    }

    // Build the initial simplex: n+1 vertices.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += step;
        simplex.push(v);
    }
    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    // Nelder-Mead coefficients.
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    let mut iter = 0;
    let mut converged = false;
    for _ in 0..max_iter {
        iter += 1;

        // Sort vertices by descending function value (best first).
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        values = idx.iter().map(|&i| values[i]).collect();

        let best = values[0];
        let worst = values[n];

        // Convergence: simplex is small and function spread is small.
        let size = simplex_size(&simplex);
        let spread = (best - worst).abs();
        if size < tol && spread < tol {
            converged = true;
            break;
        }

        // Centroid of all vertices except the worst.
        let centroid = centroid(&simplex);

        // Reflection.
        let xr = reflect(&centroid, &simplex[n], alpha);
        let fr = f(&xr);
        if fr > values[0] {
            // Expansion.
            let xe = reflect(&centroid, &simplex[n], gamma);
            let fe = f(&xe);
            if fe > fr {
                simplex[n] = xe;
                values[n] = fe;
            } else {
                simplex[n] = xr;
                values[n] = fr;
            }
        } else if fr > values[n - 1] {
            // Better than second-worst: accept reflection.
            simplex[n] = xr;
            values[n] = fr;
        } else {
            // Contraction.
            let xc = reflect(&centroid, &simplex[n], rho);
            let fc = f(&xc);
            if fc > worst {
                simplex[n] = xc;
                values[n] = fc;
            } else {
                // Shrink toward best.
                let best_vertex = simplex[0].clone();
                for i in 1..=n {
                    simplex[i] = shrink(&best_vertex, &simplex[i], sigma);
                    values[i] = f(&simplex[i]);
                }
            }
        }
    }

    // Return the best vertex.
    let best_idx = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    OptResult {
        x: simplex[best_idx].clone(),
        f: values[best_idx],
        iterations: iter,
        converged,
    }
}

fn centroid(simplex: &[Vec<f64>]) -> Vec<f64> {
    let n = simplex.len() - 1;
    let dim = simplex[0].len();
    let mut c = vec![0.0_f64; dim];
    for v in simplex.iter().take(n) {
        for (ci, &vi) in c.iter_mut().zip(v.iter()) {
            *ci += vi;
        }
    }
    for ci in &mut c {
        *ci /= n as f64;
    }
    c
}

fn reflect(centroid: &[f64], worst: &[f64], coeff: f64) -> Vec<f64> {
    centroid
        .iter()
        .zip(worst.iter())
        .map(|(&c, &w)| c + coeff * (c - w))
        .collect()
}

fn shrink(best: &[f64], v: &[f64], coeff: f64) -> Vec<f64> {
    best.iter()
        .zip(v.iter())
        .map(|(&b, &vi)| b + coeff * (vi - b))
        .collect()
}

fn simplex_size(simplex: &[Vec<f64>]) -> f64 {
    let best = &simplex[0];
    simplex[1..]
        .iter()
        .map(|v| {
            v.iter()
                .zip(best.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |acc, d| acc + d * d)
                .sqrt()
        })
        .fold(0.0_f64, f64::max)
}

/// Coordinate-wise hill climbing with adaptive step size.
///
/// Cycles through each coordinate, trying a positive and negative step.
/// When no coordinate improves, halves the step. Converges when the step
/// falls below `tol`. More robust than Nelder-Mead for low-dimensional
/// problems where the simplex can degenerate.
pub fn coordinate_ascent<F>(f: F, x0: &[f64], step: f64, max_iter: usize, tol: f64) -> OptResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return OptResult {
            x: Vec::new(),
            f: f(&[]),
            iterations: 0,
            converged: true,
        };
    }
    let mut x = x0.to_vec();
    let mut best_f = f(&x);
    let mut step = step;
    let mut iter = 0;
    let mut converged = false;

    for _ in 0..max_iter {
        iter += 1;
        let mut improved = false;
        for i in 0..n {
            // Try positive step.
            let old = x[i];
            x[i] += step;
            let f_plus = f(&x);
            if f_plus > best_f {
                best_f = f_plus;
                improved = true;
                continue;
            }
            // Try negative step.
            x[i] = old - step;
            let f_minus = f(&x);
            if f_minus > best_f {
                best_f = f_minus;
                improved = true;
                continue;
            }
            // Restore.
            x[i] = old;
        }
        if !improved {
            step *= 0.5;
            if step < tol {
                converged = true;
                break;
            }
        }
    }
    OptResult {
        x,
        f: best_f,
        iterations: iter,
        converged,
    }
}

/// Multi-start optimiser: runs `coordinate_ascent` from several starting points
/// and returns the best result. The starting points are `x0` plus perturbations.
pub fn multistart<F>(f: F, x0: &[f64], step: f64, max_iter: usize, tol: f64) -> OptResult
where
    F: Fn(&[f64]) -> f64,
{
    let mut best = coordinate_ascent(&f, x0, step, max_iter, tol);

    // Perturbation magnitudes to try.
    let perturbs = [0.1, -0.1, 0.3, -0.3, 0.5];
    for &perturb in &perturbs {
        let x_alt: Vec<f64> = x0.iter().map(|&v| v + perturb).collect();
        let result = coordinate_ascent(&f, &x_alt, step, max_iter, tol);
        if result.f > best.f {
            best = result;
        }
    }
    best
}