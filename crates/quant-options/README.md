# quant-options

Phase 10 of the quant-finance curriculum: the Black-Scholes options toolkit.
Closed-form call and put (re-exported from `quant-stochastic`), the five
Greeks (Delta, Gamma, Vega, Theta, Rho) analytically and by finite
difference, and an implied-volatility solver (Newton with a bisection
fallback).

All math is hand-rolled. No `argmin`, `optimization`, `rand`, `nalgebra`, or
`statrs`. `normal_cdf`, `bs_call`, `bs_put`, `d1`, and `d2` are reused from
`quant-stochastic`; the standard normal PDF (`normal_pdf`) and Newton's
method are implemented inline.

## Modules

| Module | File | Responsibility |
|---|---|---|
| `error` | `src/error.rs` | `OptionsError` (InvalidParam, NoConvergence, ArbitrageViolation) |
| `greeks` | `src/greeks.rs` | Analytical Delta, Gamma, Vega, Theta, Rho; `normal_pdf` |
| `finite_diff` | `src/finite_diff.rs` | Numerical Greeks by central / forward difference |
| `implied_vol` | `src/implied_vol.rs` | Newton + bisection implied-volatility solver |

## API

### Pricing (re-exported from `quant-stochastic`)

- `bs_call(s0, k, r, sigma, t) -> f64`
- `bs_put(s0, k, r, sigma, t) -> f64`
- `d1(s0, k, r, sigma, t) -> f64`
- `d2(s0, k, r, sigma, t) -> f64`
- `normal_cdf(x) -> f64`

### Analytical Greeks

- `delta(s0, k, r, sigma, t, is_call: bool) -> f64`
  - Call: `Phi(d1)`, Put: `Phi(d1) - 1`
- `gamma(s0, k, r, sigma, t) -> f64` (same for call/put)
  - `phi(d1) / (S0 * sigma * sqrt(T))`
- `vega(s0, k, r, sigma, t) -> f64` (same for call/put)
  - `S0 * phi(d1) * sqrt(T)`
- `theta(s0, k, r, sigma, t, is_call: bool) -> f64`
  - Call: `-(S0 * phi(d1) * sigma) / (2 sqrt(T)) - r K exp(-rT) Phi(d2)`
  - Put:  `-(S0 * phi(d1) * sigma) / (2 sqrt(T)) + r K exp(-rT) Phi(-d2)`
- `rho(s0, k, r, sigma, t, is_call: bool) -> f64`
  - Call: `K T exp(-rT) Phi(d2)`
  - Put:  `-K T exp(-rT) Phi(-d2)`

### Numerical Greeks (finite difference)

- `delta_fd(s0, k, r, sigma, t, is_call, h) -> f64` — central
  `(C(S+h) - C(S-h)) / (2h)`
- `gamma_fd(s0, k, r, sigma, t, h) -> f64` — central second
  `(C(S+h) - 2C(S) + C(S-h)) / h^2`
- `vega_fd(s0, k, r, sigma, t, h) -> f64` — central in sigma
- `theta_fd(s0, k, r, sigma, t, is_call, h) -> f64` — forward in t
  `(C(t-h) - C(t)) / h` (cannot step below t=0)

### Implied Volatility

- `implied_vol(market_price, s0, k, r, t, is_call) -> Result<f64, OptionsError>`
  - Newton's method with bisection fallback
  - Tolerance: 1e-8 on the price residual
  - Bracket: `[1e-6, 5.0]`
  - Vega floor: 1e-6 (below this, bisection takes over)
  - Initial guess: Brenner-Subrahmanyam (1988)
    `sigma_0 = sqrt(2 pi / T) * C / S0`
  - Put-call parity: put IV equals call IV

## Design

- **Reuse, not reinvent.** `normal_cdf`, `bs_call`, `bs_put`, `d1`, `d2`
  are re-exported from `quant-stochastic` and not reimplemented. The single
  new math primitive is `normal_pdf`, the derivative of `normal_cdf`, which
  is a one-liner.
- **Gamma and Vega are call/put symmetric.** They share `d1` and do not take
  an `is_call` flag. This is a direct reading of the formulas, not a
  simplification.
- **Theta is quoted per year.** Divide by 365 for the per-calendar-day
  convention used on trading desks. The forward difference is used because
  `t - h` may be negative for short-dated options.
- **Rho is quoted in raw units.** Multiply by 0.01 for the per-percent
  convention. Desk convention: Rho is usually "per 1% rate move".
- **Implied volatility is hybrid.** Newton is quadratic when vega is
  meaningful; bisection is linear but guaranteed. The solver keeps a
  bisection bracket `[lo, hi]` consistent at every step and uses it
  whenever the Newton step leaves the bracket or vega collapses. The
  initial guess is the Brenner-Subrahmanyam (1988) ATM approximation.
- **Put IV = Call IV.** The solver always inverts the call formula; for a
  put, we translate the put price to the equivalent call price via
  `C = P + S0 - K exp(-rT)` and proceed. The IV is invariant across
  calls and puts of the same strike and maturity (put-call parity).
- **No-arbitrage bounds.** The call price must lie in
  `[max(S0 - K exp(-rT), 0), S0]`. A price outside this range raises
  `ArbitrageViolation`. At the bounds, IV is `SIGMA_MIN` / `SIGMA_MAX`.

## Error model

`OptionsError` (via `thiserror`) has three variants:

| Variant | When |
|---|---|
| `InvalidParam(String)` | `s0 <= 0`, `k <= 0`, `t <= 0`, non-finite price/rate |
| `NoConvergence { iterations }` | Solver exhausted its iteration budget |
| `ArbitrageViolation { market_price, lower, upper }` | Price outside the no-arb bounds |

## Dependencies

- `quant-core` — workspace utilities
- `quant-stochastic` — `normal_cdf`, `bs_call`, `bs_put`, `d1`, `d2`
- `thiserror` (derive error types)
- Dev: `approx` (float comparisons), `quant-core`, `quant-stochastic`

No `argmin`, `optimization`, `rand`, `nalgebra`, or `statrs`. The crate is
offline and self-contained.

## Run

```bash
cargo test -p quant-options
cargo clippy -p quant-options --all-targets -- -D warnings
cargo run -p quant-options --example greeks
cargo run -p quant-options --example implied_vol
```

## Test contract (16 tests)

| Test | Verifies |
|---|---|
| `test_bs_call_put_parity` | C - P = S0 - K*exp(-rT) |
| `test_normal_pdf_unit_interval` | phi(0) = 1/sqrt(2 pi); phi is symmetric |
| `test_delta_call_itm` | ITM call delta in (0.5, 1) |
| `test_delta_put_otm` | OTM put delta in (-1, 0) |
| `test_gamma_positive` | gamma > 0 |
| `test_gamma_atm_max` | ATM gamma >= ITM/OTM gamma |
| `test_vega_positive` | vega > 0 |
| `test_theta_call_negative` | long call loses time value |
| `test_rho_call_positive` | call rho > 0, put rho < 0 |
| `test_delta_fd_matches_analytical` | |delta - delta_fd| < 1e-4 with h=1e-4 |
| `test_gamma_fd_matches_analytical` | |gamma - gamma_fd| < 1e-3 with h=1e-3 |
| `test_vega_fd_matches_analytical` | |vega - vega_fd| < 1e-3 with h=1e-4 |
| `test_implied_vol_recovers` | sigma recovered within 1e-8 |
| `test_implied_vol_zero_vega` | bisection fallback for deep ITM |
| `test_put_call_parity_iv` | call IV = put IV (put-call parity) |
| `test_options_smoke` | all greeks finite |