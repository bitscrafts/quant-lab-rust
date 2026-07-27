# quant-stochastic source modules

Phase 9 of the quant-finance curriculum: hand-rolled stochastic processes and
Monte Carlo pricing. Standard Brownian motion, geometric Brownian motion, the
Poisson process, the Merton jump-diffusion, and Monte Carlo European option
pricing with antithetic variates. A closed-form Black-Scholes module provides
the analytical benchmark for validating the Monte Carlo estimator. No `rand`,
`nalgebra`, `statrs`, or `optimization` — randomness comes from `quant-core`.

## Module map

| Module | File | Responsibility |
|---|---|---|
| `error` | `error.rs` | `StochError` enum (InvalidParam, InsufficientData, ConvergenceFailure) |
| `brownian` | `brownian.rs` | `brownian_motion`, `gbm`, `quadratic_variation` |
| `poisson` | `poisson.rs` | `exponential_variate`, `poisson_process`, `poisson_count`, `jump_diffusion`, `validate_mc_inputs` |
| `blackscholes` | `blackscholes.rs` | `normal_cdf`, `erf`, `d1`, `d2`, `bs_call`, `bs_put` |
| `montecarlo` | `montecarlo.rs` | `McResult`, `mc_call`, `mc_put`, `mc_call_antithetic`, `ci_half_width`, `reduce` |

`lib.rs` re-exports the public surface so callers can write
`use quant_stochastic::{brownian_motion, gbm, mc_call, bs_call, ...};` without
navigating the module tree.

## Design principles

- **Exact solutions where possible.** GBM uses the closed-form
  `S_t = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)` rather than Euler
  discretisation, so the terminal distribution is exact for any step size.
  Monte Carlo European pricing uses a single normal draw per path (the
  terminal `Z`), which is the minimal-variance estimator — no time-stepping
  is needed for path-independent payoffs.
- **Brownian motion by increments.** `brownian_motion` builds the path from
  independent Gaussian increments `dW ~ N(0, dt)`, the defining property. The
  quadratic variation `sum(dW^2)` converges in probability to `T`, which the
  `quadratic_variation` helper exposes for tests and pedagogy.
- **Inverse-CDF sampling.** `exponential_variate` uses `X = -ln(1-U)/rate`
  with `U ~ Uniform(0,1)` — the exact inverse CDF of `Exp(rate)`. The Poisson
  process is then the cumulative sum of exponential interarrival times,
  stopped at `t`. `u` is clamped away from 0 and 1 to avoid `ln(0)`.
- **Merton jump-diffusion.** Between jumps the process follows GBM; at each
  Poisson event time the price is multiplied by `J = exp(jump_mean)`. With
  `jump_rate = 0` there are no jumps and the process reduces exactly to
  `gbm` (same RNG stream, identical increments), which the
  `test_jump_diffusion_drift` test verifies to `1e-12`. The jump size is
  deterministic here; a log-normal jump (`J = exp(mu_J + sigma_J * Z)`) is
  the natural Phase 10 extension.
- **Black-Scholes benchmark.** The normal CDF uses the Abramowitz-Stegun
  (1964) formula 7.1.26 rational approximation of `erf`, with
  `Phi(x) = 0.5 * (1 + erf(x / sqrt 2))`. Maximum absolute error `< 7.5e-8`
  versus the exact integral — sufficient for validating Monte Carlo (whose
  statistical error is orders of magnitude larger for practical `N`).
- **Monte Carlo estimator.** `mc_call` draws `S_T = S0 * exp((r - 0.5*sigma^2)*T
  + sigma*sqrt(T)*Z)`, computes `max(S_T - K, 0)`, and returns
  `exp(-rT) * mean(payoff)` with standard error
  `exp(-rT) * std(payoff) / sqrt(N)`. The sample standard deviation uses the
  `n-1` denominator. SE scales as `1/sqrt(N)`: quadrupling paths halves the SE.
- **Antithetic variates.** `mc_call_antithetic` pairs each `Z` with `-Z`. The
  two payoffs are perfectly negatively correlated (the call payoff is monotone
  increasing in `Z`), so the variance of the pair average is lower than the
  variance of two independent samples. The reported `n_paths` is `2 * n_draws`
  but only `n_draws` normal variates are consumed — the computational saving.
- **No panics in library paths.** All fallible functions return
  `Result<_, StochError>`. `validate_mc_inputs` checks `s0 > 0`, `k > 0`,
  `t > 0`, `sigma >= 0`, and `n_paths >= 1` before simulating.

## Error model

`StochError` (via `thiserror`) has three variants:

| Variant | When |
|---|---|
| `InvalidParam(String)` | `s0 <= 0`, `k <= 0`, `t <= 0`, `sigma < 0` in MC pricing |
| `InsufficientData { required, actual }` | `n_paths == 0` in MC pricing |
| `ConvergenceFailure { n_paths }` | Reserved for future use (MC did not converge) |

The Brownian, GBM, and Poisson simulators use `assert!` for their structural
inputs (`n >= 1`, `dt > 0`, `rate > 0`) because these are programming errors,
not data errors.

## Dependencies

- `quant-core` — `XorShift64` (xorshift64*), `Normal` (Box-Muller),
  `Distribution`, `Rng`
- `thiserror` (derive error types)
- Dev: `approx` (float comparisons), `quant-core`

No `rand`, `nalgebra`, `statrs`, or `optimization`. The crate is offline and
synthetic.

## Run

```bash
cargo test -p quant-stochastic
cargo clippy -p quant-stochastic --all-targets -- -D warnings
cargo run -p quant-stochastic --example mc_pricing
cargo run -p quant-stochastic --example brownian_paths
```