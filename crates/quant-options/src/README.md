# quant-options source modules

Phase 10 of the quant-finance curriculum: the Black-Scholes options toolkit.
Closed-form call and put (re-exported from `quant-stochastic`), the five
Greeks (Delta, Gamma, Vega, Theta, Rho) analytically and by finite
difference, and an implied-volatility solver (Newton + bisection fallback).
No `argmin`, `optimization`, `rand`, `nalgebra`, or `statrs`.

## Module map

| Module | File | Responsibility |
|---|---|---|
| `error` | `error.rs` | `OptionsError` (InvalidParam, NoConvergence, ArbitrageViolation) |
| `greeks` | `greeks.rs` | Analytical Greeks; `normal_pdf` (standard normal density) |
| `finite_diff` | `finite_diff.rs` | Numerical Greeks (delta, gamma, vega, theta) via finite difference |
| `implied_vol` | `implied_vol.rs` | Newton + bisection IV solver |

`lib.rs` re-exports the public surface so callers can write
`use quant_options::{bs_call, delta, gamma, vega, theta, rho, implied_vol, ...};`
without navigating the module tree. `bs_call`, `bs_put`, `d1`, `d2`, and
`normal_cdf` are re-exported from `quant-stochastic` for a single import
surface.

## Design principles

- **Reuse, not reinvent.** The Black-Scholes primitives (`normal_cdf`,
  `bs_call`, `bs_put`, `d1`, `d2`) are re-exported from
  `quant-stochastic`, not reimplemented. The only new math primitive is
  `normal_pdf` (`phi(x) = (1/sqrt(2 pi)) exp(-x^2 / 2)`), the derivative
  of `normal_cdf`, which is a one-liner.
- **Gamma and Vega are call/put symmetric.** They share `d1` and do not
  take an `is_call` flag. This is a direct reading of the formulas, not a
  simplification. Delta, Theta, and Rho are call/put asymmetric and take
  the flag.
- **Theta is per year, forward difference.** Quoted per year (negative for
  long options, since time value decays). The forward difference
  `(C(t-h) - C(t)) / h` is used because the central difference would step
  to `t - h < 0` for short-dated options.
- **Rho is in raw units.** Multiply by 0.01 for the per-percent convention
  used on trading desks. Vega likewise: multiply by 0.01 for per-percent
  vol moves.
- **Implied volatility is hybrid.** Newton is quadratic when vega is
  meaningful; bisection is linear but guaranteed. The solver maintains a
  bisection bracket `[lo, hi]` consistent at every step and uses it
  whenever the Newton step leaves the bracket or vega collapses below
  `VEGA_FLOOR = 1e-6`. The initial guess is the Brenner-Subrahmanyam
  (1988) ATM approximation `sigma_0 = sqrt(2 pi / T) * C / S0`.
- **Put IV = Call IV.** The solver always inverts the call formula; for a
  put, we translate the put price to the equivalent call price via
  `C = P + S0 - K exp(-rT)` (put-call parity) and proceed. The IV is
  invariant across calls and puts of the same strike and maturity.
- **No-arbitrage bounds.** The call price must lie in
  `[max(S0 - K exp(-rT), 0), S0]`. A price outside this range raises
  `ArbitrageViolation`. At the bounds, IV is `SIGMA_MIN` (1e-6) or
  `SIGMA_MAX` (5.0) — the option is intrinsic, sigma is unrecoverable.
- **No panics in library paths.** All fallible functions return
  `Result<_, OptionsError>`. Input validation in `implied_vol` checks
  `s0 > 0`, `k > 0`, `t > 0`, finite `r`, and non-negative finite
  `market_price`.

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