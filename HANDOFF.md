# HANDOFF: Quant-Finance Implementation

**Project**: `quant-lab-rust`
**Repository**: https://github.com/bitscrafts/quant-lab-rust
**Last Updated**: 2026-07-27
**Status**: Part V Started (Phases 1-13 Done, 312 tests passing, Phase 14 Next)

---

## Quick Start

```bash
# Clone and verify
git clone https://github.com/bitscrafts/quant-lab-rust.git
cd quant-lab-rust

# Build all crates
cargo build

# Run all tests (312 tests)
cargo test

# Run examples
cargo run -p qf-01-fraud --example fraud_analysis
cargo run -p qf-02-loan --example loan_analysis
cargo run -p qf-03-stocks --example stock_analysis
cargo run -p qf-04-returns --example returns_analysis
cargo run -p quant-portfolio --example frontier
cargo run -p quant-portfolio --example capm
cargo run -p quant-factors --example pca_demo
cargo run -p quant-factors --example fama_french
```

---

## Project Overview

This is a progressive learning curriculum for quantitative finance, implemented
from first principles in Rust with a companion LaTeX book (kaobook template).

**Philosophy**:
- Learn by doing, document in parallel
- Library-first design (code will evolve into `quant-lib`)
- Hand-rolled math (no black-box libraries)
- TDD: tests written BEFORE production code

**Two deliverables per phase**:
1. Working, tested Rust crate in `crates/`
2. Corresponding LaTeX chapter in `book/chapters/`

**Final Target: `quant-lib`**

After all 14 phases are complete, the individual crates will be consolidated into
a single unified library crate: `quant-lib`. This crate will:
- Re-export all public APIs from phase crates under a unified namespace
- Provide a single dependency for downstream projects
- Maintain the modular structure internally (feature flags for optional modules)
- Include comprehensive documentation and examples

```
quant-lib/
├── src/
│   ├── lib.rs           # Re-exports all modules
│   ├── core/            # From quant-core
│   ├── timeseries/      # From quant-timeseries
│   ├── vol/             # From quant-vol
│   ├── stochastic/      # From quant-stochastic
│   ├── options/         # From quant-options
│   ├── portfolio/       # From quant-portfolio
│   ├── factors/         # From quant-factors
│   ├── microstructure/  # From quant-microstructure
│   └── backtest/        # From quant-backtest
└── Cargo.toml           # Feature flags for optional modules
```

Phase 15 (final): Create `quant-lib` by consolidating Phases 6-14.

---

## Repository Structure

```
quant-lab-rust/
├── Cargo.toml              # Workspace root
├── README.md               # Project overview with book link
├── HANDOFF.md              # This file
├── .gitignore              # macOS, Rust, LaTeX exclusions
│
├── crates/                 # Rust implementations
│   ├── qf-common/          # Shared utilities (CSV loading, stats)
│   ├── qf-01-fraud/        # Ch01: Credit card fraud detection
│   ├── qf-02-loan/         # Ch02: Loan default prediction
│   ├── qf-03-stocks/       # Ch03: Stock price analysis (OHLCV, SMA, EMA)
│   └── qf-04-returns/      # Ch04: Returns, volatility, risk metrics
│
├── data/                   # Sample datasets (included)
│   ├── stock_prices.csv    # Synthetic stock data for examples
│   └── creditcard_sample.csv
│
└── book/                   # LaTeX book source
    ├── main.tex            # Main document
    ├── references.bib      # Bibliography
    ├── chapters/           # Chapter files (ch00.tex - ch04.tex)
    ├── figures/            # Externalized TikZ figures
    ├── kaobook/            # kaobook template files
    └── build/              # Compiled PDF output
        └── quant-finance-book.pdf
```

---

## Completed Work

### Phase 1: Hello Finance — Credit Card Fraud Detection

**Crate**: `qf-01-fraud`
**Book**: `book/chapters/ch01.tex`
**Tests**: 10 tests passing

Implements:
- `ZScoreDetector` — anomaly detection via z-score threshold
- `ConfusionMatrix` — precision, recall, F1 metrics
- `evaluate()` — runs detector on dataset

Key patterns:
```rust
// Note: AnomalyDetector trait mentioned in docs but ZScoreDetector
// is implemented as concrete type. Trait extraction deferred.
pub struct ZScoreDetector {
    threshold: f64,
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
}
```

### Phase 2: Risk Basics — Loan Default Prediction

**Crate**: `qf-02-loan`
**Book**: `book/chapters/ch02.tex`
**Tests**: 21 tests passing

Implements:
- `FeatureExtractor` — debt-to-income, loan-to-income, grade score
- `OneHotEncoder` — categorical encoding (RENT/OWN/MORTGAGE)
- `Normalizer` — min-max scaling
- `LinearScorer` — weighted sum + sigmoid for probability
- `roc_curve()`, `auc()` — ROC-AUC evaluation

Key patterns:
```rust
pub trait Scorer {
    fn score(&self, features: &[f64]) -> f64;
}

pub trait BinaryClassifier {
    fn predict(&self, features: &[f64]) -> bool;
    fn predict_proba(&self, features: &[f64]) -> f64;
}
```

Example: `cargo run -p qf-02-loan --example loan_analysis`

### Phase 3: Market Data — Stock Price Analysis

**Crate**: `qf-03-stocks`
**Book**: `book/chapters/ch03.tex`
**Tests**: 20 tests passing

Implements:
- `Ohlcv` struct with candlestick methods (body, shadows, typical price)
- `sma()`, `ema()` — simple and exponential moving averages
- Statistics: `average_volume()`, `price_change()`, `highest_high()`, `lowest_low()`
- `TimeSeries` trait for generic operations

Key patterns:
```rust
pub trait TimeSeries {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn closes(&self) -> Vec<f64>;
    fn volumes(&self) -> Vec<u64>;
}
```

### Phase 4: Returns & Volatility (COMPLETE)

**Crate**: `qf-04-returns`
**Book**: `book/chapters/ch04.tex`
**Tests**: 25 tests passing
**Dependencies**: `qf-common`, `qf-03-stocks`

Implements:
- `simple_returns()`, `log_returns()`, `cumulative_returns()` — return calculations
- `volatility()`, `annualized_volatility()`, `rolling_volatility()` — risk measures
- `sharpe_ratio()`, `annualized_sharpe()`, `sortino_ratio()` — risk-adjusted metrics
- `drawdown()`, `max_drawdown()`, `DrawdownStats` — drawdown analysis
- `Returns` trait — abstraction over price sources

Key patterns:
```rust
pub trait Returns {
    fn simple_returns(&self) -> Vec<f64>;
    fn log_returns(&self) -> Vec<f64>;
}

// Implemented for &[f64], Vec<f64>, Vec<Ohlcv>
```

Module structure:
```
qf-04-returns/
├── src/
│   ├── lib.rs          # Re-exports
│   ├── returns.rs      # simple_returns, log_returns, cumulative_returns, Returns trait
│   ├── volatility.rs   # volatility, annualized_volatility, rolling_volatility
│   ├── risk.rs         # sharpe_ratio, annualized_sharpe, sortino_ratio
│   ├── drawdown.rs     # drawdown, max_drawdown, DrawdownStats
│   └── error.rs        # ReturnsError
├── tests/
│   └── returns_tests.rs  # 25 integration tests
└── examples/
    └── returns_analysis.rs  # Working example
```

**Verified**: All 25 tests pass, clippy clean, example runs successfully.

---

## Completed: Phase 7 — Time Series: Stationarity and Fractional Differentiation

**Crate**: `quant-timeseries`
**Book chapter**: `book/chapters/ch07.tex`
**Dependencies**: `quant-core`
**Tests**: 17 contract + 1 extra GBM test = 18 passing, clippy clean

Implements hand-rolled time-series econometrics:
- `OlsFit` (`coeffs`, `residuals`, `std_errors`, `t_stats`, `r_squared`) and
  `ols()` via the normal equations solved with Gaussian elimination and
  partial pivoting
- `gauss_solve` (pub crate) — internal linear solver with pivoting
- `acf()` — autocorrelation function for lags `0..=max_lag`
- `adf_test()` — Augmented Dickey-Fuller with `MACKINNON_5PCT = -2.86`
  (constant, no trend, 5% significance)
- `ffd_weights`, `frac_diff`, `find_min_d` — López de Prado fixed-width
  fractional differentiation (AFML Ch.5) and binary search for the minimum
  stationary `d`

Key fixes during implementation:
- `ffd_weights` breaks *before* pushing a sub-threshold weight, giving
  `[1.0]` for `d = 0` and `[1.0, -1.0]` for `d = 1` exactly
- ADF lagged-difference indexing uses `dy[t - 1 - i]` (= $\Delta y_{t-i}$),
  not `dy[t - i]`
- `gauss_solve` is `pub(crate)` to avoid exposing the internal solver

Examples: `stationarity.rs` (ADF on random walk vs d=1 vs d=0.4),
`ffd_demo.rs` (find_min_d + memory comparison via ACF).

---

## Completed: Phase 8 — Volatility Models: EWMA, ARCH, GARCH

**Crate**: `quant-vol`
**Book chapter**: `book/chapters/ch08.tex`
**Dependencies**: `quant-core`

### Overview

This phase implements volatility models: exponentially weighted moving
average (EWMA/RiskMetrics), ARCH(q), and GARCH(p, q). Maximum-likelihood
fitting via hand-rolled Nelder-Mead simplex search and coordinate-ascent
multi-start. Builds on Phase 6 moments and Phase 7 OLS.

**Key insight**: Volatility clustering — large moves follow large moves —
is the empirical regularity that ARCH/GARCH capture. Gaussian returns
with constant volatility underestimate tail risk; GARCH corrects this.

**Key constraint**: Hand-rolled optimisation. No `argmin`, no `optimization`
crate. Nelder-Mead and coordinate ascent implemented inline.

### Requirements

#### R8.1: Create quant-vol Crate

```bash
cd crates/quant-lab/crates
cargo new quant-vol --lib
```

Add to workspace `Cargo.toml`:
```toml
members = [
    ...
    "crates/quant-timeseries",
    "crates/quant-vol",  # ADD THIS
]
```

Dependencies:
```toml
[dependencies]
quant-core = { path = "../quant-core" }
quant-timeseries = { path = "../quant-timeseries" }
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
quant-core = { path = "../quant-core" }
```

#### R8.2: EWMA Volatility (RiskMetrics)

```rust
/// Exponentially weighted moving average volatility.
/// sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
/// RiskMetrics lambda = 0.94 (daily).
pub fn ewma_vol(returns: &[f64], lambda: f64) -> Result<Vec<f64>, VolError>;
```

#### R8.3: ARCH(q)

```rust
/// ARCH(q) model: sigma_t^2 = omega + sum_{i=1..q} alpha_i r_{t-i}^2
pub struct ArchModel { omega: f64, alphas: Vec<f64> }

impl ArchModel {
    /// Fit ARCH(q) via MLE (assumes Gaussian innovations).
    pub fn fit(returns: &[f64], q: usize) -> Result<Self, VolError>;
    pub fn forecast(&self, horizon: usize) -> Vec<f64>;
    pub fn log_likelihood(&self, returns: &[f64]) -> f64;
}
```

#### R8.4: GARCH(p, q)

```rust
/// GARCH(p, q):
/// sigma_t^2 = omega + sum alpha_i r_{t-i}^2 + sum beta_j sigma_{t-j}^2
pub struct GarchModel {
    omega: f64,
    alphas: Vec<f64>,  // ARCH terms
    betas: Vec<f64>,   // GARCH terms
}

impl GarchModel {
    /// Fit GARCH(1,1) via MLE with hand-rolled gradient ascent.
    pub fn fit(returns: &[f64], p: usize, q: usize) -> Result<Self, VolError>;
    pub fn forecast(&self, horizon: usize) -> Vec<f64>;
    pub fn persistence(&self) -> f64;  // sum(alpha) + sum(beta)
    pub fn long_run_variance(&self) -> f64;  // omega / (1 - persistence)
}
```

#### R8.5: Example Binaries

- `vol_demo.rs`: Compare EWMA, ARCH(1), GARCH(1,1) on simulated returns.
- `vol_clustering.rs`: Show GARCH captures clustered volatility where
  constant-volatility models fail.

#### R8.6: Book Chapter

`book/chapters/ch08.tex` with:
1. Volatility stylized facts (clustering, fat tails, mean reversion)
2. EWMA derivation and RiskMetrics lambda
3. ARCH(q) specification and MLE
4. GARCH(p,q) specification, stationarity, persistence
5. Long-run variance and half-life of shocks
6. Rust implementation
7. Exercises: EGARCH, GJR-GARCH, forecast evaluation

### TDD Contract (15 tests)

**File**: `crates/quant-vol/tests/vol_tests.rs`

| Test | Given | Expects |
|---|---|---|
| `test_ewma_lambda_zero` | lambda=0 | sigma = 0 except first |
| `test_ewma_lambda_one` | lambda=1 | constant sigma |
| `test_ewma_decay` | lambda=0.94 | weights decay geometrically |
| `test_ewma_variance` | white noise | EWMA var ~ sample var |
| `test_arch_zero` | alphas all 0 | constant sigma = sqrt(omega) |
| `test_arch_forecast` | ARCH(1) | forecast reverts to long-run |
| `test_arch_log_likelihood` | any fit | non-negative LL |
| `test_garch11_stationarity` | alpha+beta < 1 | covariance stationary |
| `test_garch11_persistence` | fitted GARCH | persistence < 1 |
| `test_garch11_long_run` | fitted GARCH | long-run var positive |
| `test_garch_forecast_decay` | any GARCH | forecast reverts to LR var |
| `test_garch_log_likelihood` | any fit | non-negative LL |
| `test_garch_vol_clustering` | simulated clustering | GARCH fit LL > constant |
| `test_fit_convergence` | GARCH(1,1) | MLE converges to true params |
| `test_vol_smoke` | real returns | all models produce finite output |

### Exit Criteria (Phase 8)

```bash
test -f crates/quant-vol/Cargo.toml
grep -q "quant-vol" Cargo.toml
cargo test -p quant-vol 2>&1 | grep -E "test result.*0 failed"
cargo clippy -p quant-vol --all-targets -- -D warnings
cargo run -p quant-vol --example vol_demo
cargo run -p quant-vol --example vol_clustering
test -f book/chapters/ch08.tex
```

### Guardrails

- **Approved dependencies**: `quant-core`, `quant-timeseries`, `thiserror`.
  Dev: `approx`, `quant-core`
- **FORBIDDEN**: `argmin`, `optimization`, `rand`, `nalgebra`, `statrs`
- **Package-scoped builds only**: `-p quant-vol`
- **All optimisation hand-rolled**: gradient ascent or Nelder-Mead inline
- **All math hand-rolled**: no external MLE or GARCH libraries

### Mathematical Reference

**EWMA (RiskMetrics)**:
$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2, \quad \lambda = 0.94$$

**ARCH(q)** (Engle 1982):
$$\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i r_{t-i}^2$$

**GARCH(p, q)** (Bollerslev 1986):
$$\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i r_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2$$

**Persistence**: $\alpha + \beta < 1$ (covariance stationary).
**Long-run variance**: $\sigma^2 = \omega / (1 - \alpha - \beta)$.
**Half-life of shocks**: $\log(0.5) / \log(\alpha + \beta)$.

**MLE (Gaussian innovations)**:
$$\mathcal{L} = -\frac{1}{2} \sum_t \left[ \log(2\pi) + \log\sigma_t^2 + \frac{r_t^2}{\sigma_t^2} \right]$$

---

## Completed: Phase 9 — Stochastic Processes and Monte Carlo

**Crate**: `quant-stochastic`
**Book chapter**: `book/chapters/ch09.tex`
**Dependencies**: `quant-core`, `quant-vol`

### Overview

This phase implements stochastic processes for asset price modelling: standard
Brownian motion, geometric Brownian motion with closed-form solution, Poisson
jump processes, and Monte Carlo pricing. Builds on Phase 6 (GBM paths, Normal)
and Phase 8 (volatility models, for stochastic-vol extensions).

**Key insight**: Monte Carlo pricing converges to the analytical
Black-Scholes price as $N \to \infty$, with standard error decreasing as
$1/\sqrt{N}$. This is the law of large numbers in action and the foundation
of computational finance.

**Key constraint**: Hand-rolled math. No `argmin`, `optimization`, `rand`,
`nalgebra`, or `statrs`. Reuse `XorShift64` and `Normal` from `quant-core`.

### Requirements

#### R9.1: Create quant-stochastic Crate

```bash
cd crates/quant-lab/crates
cargo new quant-stochastic --lib
```

Add to workspace `Cargo.toml`:
```toml
members = [
    ...
    "crates/quant-vol",
    "crates/quant-stochastic",  # ADD THIS
]
```

Dependencies:
```toml
[dependencies]
quant-core = { path = "../quant-core" }
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
quant-core = { path = "../quant-core" }
```

#### R9.2: Brownian Motion

```rust
/// Standard Brownian motion path: W_0 = 0, W_t ~ N(0, t).
pub fn brownian_motion(n: usize, dt: f64, rng: &mut XorShift64) -> Vec<f64>;

/// Geometric Brownian motion: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t).
pub fn gbm(s0: f64, mu: f64, sigma: f64, t: f64, n: usize, rng: &mut XorShift64) -> Vec<f64>;
```

#### R9.3: Poisson Jump Process

```rust
/// Poisson process path with rate lambda.
pub fn poisson_process(rate: f64, t: f64, rng: &mut XorShift64) -> Vec<f64>;

/// Jump-diffusion: GBM with Poisson jumps (Merton model).
pub fn jump_diffusion(
    s0: f64, mu: f64, sigma: f64, jump_rate: f64, jump_mean: f64,
    t: f64, n: usize, rng: &mut XorShift64,
) -> Vec<f64>;
```

#### R9.4: Monte Carlo Pricing

```rust
/// Monte Carlo European call option price via GBM terminal distribution.
pub fn mc_call(s0: f64, k: f64, r: f64, sigma: f64, t: f64, n_paths: usize, rng: &mut XorShift64)
    -> McResult;

pub struct McResult {
    pub price: f64,
    pub std_error: f64,   // standard error of the mean
    pub n_paths: usize,
}
```

#### R9.5: Example Binaries

- `mc_pricing.rs`: Price a European call by Monte Carlo, compare to
  Black-Scholes, show convergence as N increases (100, 1000, 10000, 100000).
- `brownian_paths.rs`: Generate and print sample Brownian motion and GBM paths.

#### R9.6: Book Chapter

`book/chapters/ch09.tex` with:
1. Stochastic processes: Brownian motion, Markov property, quadratic variation
2. Ito's lemma and the GBM SDE
3. Poisson processes and jump-diffusion (Merton)
4. Monte Carlo method: sample, average, standard error
5. Convergence: $1/\sqrt{N}$ law, confidence intervals
6. Rust implementation
7. Exercises: variance reduction (antithetic, control variates), Asian options

### TDD Contract (15 tests)

**File**: `crates/quant-stochastic/tests/stoch_tests.rs`

| Test | Given | Expects |
|---|---|---|
| `test_bm_starts_at_zero` | brownian_motion | W_0 = 0 |
| `test_bm_quadratic_variation` | n=10000, dt=1/252 | sum dW^2 ~ T |
| `test_gbm_terminal_distribution` | n=100000 | mean log(S_T/S_0) ~ (mu - 0.5*sigma^2)*T |
| `test_gbm_known_solution` | s0=100, mu=0.05, sigma=0.2, t=1 | E[S_T] = s0*exp(mu*T) |
| `test_poisson_rate` | n=100000, rate=5 | mean count ~ rate*t |
| `test_poisson_interarrival` | exponential gaps with mean 1/rate |
| `test_jump_diffusion_drift` | zero jumps | reduces to GBM |
| `test_mc_call_convergence` | N=100..100000 | price -> BS price |
| `test_mc_call_at_the_money` | K=S0 | price ~ BS price |
| `test_mc_standard_error` | N=10000 | se ~ sigma*sqrt(t)/sqrt(N) |
| `test_mc_in_the_money` | K << S0 | price ~ S0 - K*exp(-rT) |
| `test_mc_out_of_the_money` | K >> S0 | price ~ 0 |
| `test_mc_put_call_parity` | call - put | = S0 - K*exp(-rT) |
| `test_antithetic_variance_reduction` | antithetic vs plain | lower se |
| `test_stoch_smoke` | all models | finite output |

### Exit Criteria (Phase 9)

```bash
test -f crates/quant-stochastic/Cargo.toml
grep -q "quant-stochastic" Cargo.toml
cargo test -p quant-stochastic 2>&1 | grep -E "test result.*0 failed"
cargo clippy -p quant-stochastic --all-targets -- -D warnings
cargo run -p quant-stochastic --example mc_pricing
cargo run -p quant-stochastic --example brownian_paths
test -f book/chapters/ch09.tex
```

### Guardrails

- **Approved dependencies**: `quant-core`, `thiserror`. Dev: `approx`, `quant-core`
- **FORBIDDEN**: `argmin`, `optimization`, `rand`, `nalgebra`, `statrs`
- **Package-scoped builds only**: `-p quant-stochastic`
- **All math hand-rolled**: no external SDE or Monte Carlo libraries
- **Reuse quant-core**: `XorShift64`, `Normal`, `Distribution` from Phase 6

### Mathematical Reference

**Brownian motion**:
$$W_t \sim \Normal(0, t), \quad W_{t+\Delta t} - W_t \sim \Normal(0, \Delta t)$$

**GBM SDE**:
$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$
**Closed-form solution**:
$$S_T = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right) T + \sigma W_T \right)$$

**Poisson process**: $N_t \sim \text{Poisson}(\lambda t)$, interarrival times
$\sim \text{Exp}(\lambda)$.

**Merton jump-diffusion**:
$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t + (J - 1) S_t \, dN_t$$

**Monte Carlo European call**:
$$\hat{C} = e^{-rT} \frac{1}{N} \sum_{i=1}^N \max(S_T^{(i)} - K, 0)$$
$$\text{SE}(\hat{C}) = e^{-rT} \frac{\text{std}(\max(S_T - K, 0))}{\sqrt{N}}$$

---

## Completed: Phase 10 — Options Pricing: Black-Scholes and Beyond

**Crate**: `quant-options` (implemented, 16 tests + 1 doc test, all green)
**Book chapter**: `book/chapters/ch09.tex` (92-page compiled PDF)
**Dependencies**: `quant-core`, `quant-stochastic`, `thiserror`

### Status

Phase 10 is **complete**. All exit criteria pass:

- `cargo test -p quant-options` — 16 tests + 1 doc test, 0 failed
- `cargo clippy -p quant-options --all-targets -- -D warnings` — clean
- `cargo run -p quant-options --example greeks` — prints all Greeks
  analytical vs finite-diff (agree to $10^{-4}$)
- `cargo run -p quant-options --example implied_vol` — recovers
  $\sigma = 0.2$ to $3.5 \times 10^{-14}$, demonstrates synthetic smile
- `book/chapters/ch10.tex` — 4 pages, compiled into the 92-page book

### What was built

| Module | Function | Notes |
|---|---|---|
| `greeks` | `delta`, `gamma`, `vega`, `theta`, `rho`, `normal_pdf` | Analytical Greeks; `normal_pdf` is the only new math primitive (one-liner) |
| `finite_diff` | `delta_fd`, `gamma_fd`, `vega_fd`, `theta_fd` | Central difference ($O(h^2)$); forward diff for Theta (cannot step to $t<0$) |
| `implied_vol` | `implied_vol` | Newton + bisection fallback; Brenner-Subrahmanyam initial guess; `VEGA_FLOOR = 1e-6` triggers bisection |

### Key design choices

- **Reuse, not reinvent.** `bs_call`, `bs_put`, `d1`, `d2`, `normal_cdf` are
  re-exported from `quant-stochastic`. The single new math primitive is
  `normal_pdf` ($\phi(x) = (2\pi)^{-1/2} e^{-x^2/2}$), the derivative of
  `normal_cdf`.
- **Gamma and Vega are call/put symmetric.** They share $d_1$ and take no
  `is_call` flag. Delta, Theta, Rho are call/put asymmetric and take it.
- **Implied volatility is hybrid.** Newton is quadratic when vega is
  meaningful; bisection is linear but guaranteed. The solver keeps a
  `[lo, hi]` bracket consistent at every step and falls back to bisection
  when vega collapses or Newton leaves the bracket.
- **Put IV = Call IV.** The solver always inverts the call formula; for a
  put, we translate via put-call parity $C = P + S_0 - K e^{-rT}$. The IV
  is invariant across calls and puts of the same strike/maturity.
- **No-arbitrage bounds.** The call price must lie in
  $[\max(S_0 - K e^{-rT}, 0), S_0]$. Out-of-bounds prices raise
  `ArbitrageViolation`. At the bounds, IV is the floor/ceiling.

### Verified numerical results

ATM call ($S_0 = K = 100$, $r = 0.05$, $\sigma = 0.2$, $T = 1$):
- Call price = 10.450575 (matches Phase 9 BS benchmark)
- Delta_c = 0.636831, Delta_p = -0.363169
- Gamma = 0.018762, Vega = 37.524035
- Theta_c = -6.414028, Theta_p = -1.657881
- Rho_c = 53.232483, Rho_p = -41.890459
- IV recovery: $|\hat{\sigma} - 0.2| = 3.54 \times 10^{-14}$

Synthetic smile ($\sigma(K) = 0.2 + 0.5 \cdot (\ln(K/S_0))^2$):
- K=70 IV=0.264, K=100 IV=0.200, K=130 IV=0.234 (symmetric in log-moneyness)

---

## Reference: Phase 10 Specification (kept for archival)

**Crate to create**: `quant-options`
**Book chapter**: `book/chapters/ch10.tex`
**Dependencies**: `quant-core`, `quant-stochastic` (reuse `normal_cdf`, `bs_call`)

### Overview

This phase builds the full Black-Scholes options toolkit: closed-form call
and put (already prototyped in `quant-stochastic::blackscholes`), the Greeks
(Delta, Gamma, Vega, Theta, Rho) analytically and by finite difference,
implied volatility via Newton / bisection, and the Black-76 / log-normal
extension. Builds on Phase 9 (GBM, `normal_cdf`, `bs_call`).

**Key insight**: The Greeks are the partial derivatives of the option price
with respect to its inputs. Delta-hedging (Delta = 0) eliminates first-order
price risk; Gamma controls the hedging error. Implied volatility inverts
the BS formula to recover the market's view of future variance --- the
"volatility smile" is the single most studied phenomenon in derivatives.

**Key constraint**: Hand-rolled math. No `argmin`, `optimization`, `rand`,
`nalgebra`, or `statrs`. Reuse `normal_cdf` from `quant-stochastic`. Newton's
method for implied vol is implemented inline.

### Requirements

#### R10.1: Create quant-options Crate

```bash
cd crates/quant-lab/crates
cargo new quant-options --lib
```

Add to workspace `Cargo.toml`:
```toml
members = [
    ...
    "crates/quant-stochastic",
    "crates/quant-options",  # ADD THIS
]
```

Dependencies:
```toml
[dependencies]
quant-core = { path = "../quant-core" }
quant-stochastic = { path = "../quant-stochastic" }  # normal_cdf, bs_call/put
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
quant-core = { path = "../quant-core" }
```

#### R10.2: Black-Scholes Call and Put

Reuse `bs_call`, `bs_put`, `d1`, `d2`, `normal_cdf` from `quant-stochastic`.
Re-export from `quant-options` for a single import surface.

#### R10.3: The Greeks (Analytical)

```rust
pub fn delta(s0: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64;
pub fn gamma(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64;      // same for call/put
pub fn vega(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64;        // same for call/put
pub fn theta(s0: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64;
pub fn rho(s0: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64;
```

- Delta: call $\Phi(d_1)$, put $\Phi(d_1) - 1$
- Gamma: $\phi(d_1) / (S_0 \sigma \sqrt{T})$ (normal pdf, not cdf)
- Vega: $S_0 \phi(d_1) \sqrt{T}$ (same for call/put)
- Theta: $-(S_0 \phi(d_1) \sigma)/(2\sqrt{T}) - r K e^{-rT} \Phi(d_2)$ (call)
- Rho: $K T e^{-rT} \Phi(d_2)$ (call), $-K T e^{-rT} \Phi(-d_2)$ (put)

#### R10.4: Numerical Greeks (Finite Difference)

```rust
pub fn delta_fd(s0, k, r, sigma, t, is_call, h: f64) -> f64;  // (C(S+h)-C(S-h))/(2h)
pub fn gamma_fd(s0, k, r, sigma, t, h: f64) -> f64;          // (C(S+h)-2C(S)+C(S-h))/h^2
pub fn vega_fd(s0, k, r, sigma, t, h: f64) -> f64;
pub fn theta_fd(s0, k, r, sigma, t, is_call, h: f64) -> f64;  // forward difference in t
```

#### R10.5: Implied Volatility

```rust
/// Solve C(S, K, r, sigma, T) = market_price for sigma.
/// Newton's method with vega as the derivative; bisection fallback.
pub fn implied_vol(market_price, s0, k, r, t, is_call) -> Result<f64, OptionsError>;
```

#### R10.6: Example Binaries

- `greeks.rs`: Print all Greeks for an ATM call; compare analytical vs
  finite-difference (should match to ~1e-4 with h=1e-4).
- `implied_vol.rs`: Recover sigma from a BS price; demonstrate the
  volatility smile on a synthetic quote table.

#### R10.7: Book Chapter

`book/chapters/ch10.tex` with:
1. Black-Scholes PDE and the risk-neutral derivation
2. The Greeks as partial derivatives; delta-hedging intuition
3. Analytical Greeks formulas
4. Numerical Greeks by finite difference
5. Implied volatility: Newton, bisection, the volatility smile
6. Rust implementation
7. Exercises: Black-Scholes for dividend yield, local volatility, Heston

### TDD Contract (15 tests)

**File**: `crates/quant-options/tests/options_tests.rs`

| Test | Given | Expects |
|---|---|---|
| `test_bs_call_put_parity` | ATM | call - put = S0 - K*exp(-rT) |
| `test_delta_call_itm` | K < S0 | delta in (0,1), delta > 0.5 |
| `test_delta_put_otm` | K > S0 | delta in (-1, 0) |
| `test_gamma_positive` | any | gamma > 0 |
| `test_gamma_atm_max` | ATM vs ITM/OTM | gamma ATM >= gamma ITM/OTM |
| `test_vega_positive` | any | vega > 0 |
| `test_theta_call_negative` | short-dated | theta < 0 |
| `test_rho_call_positive` | call | rho > 0 |
| `test_delta_fd_matches_analytical` | h=1e-4 | \|delta - delta_fd\| < 1e-4 |
| `test_gamma_fd_matches_analytical` | h=1e-3 | \|gamma - gamma_fd\| < 1e-3 |
| `test_vega_fd_matches_analytical` | h=1e-4 | \|vega - vega_fd\| < 1e-3 |
| `test_implied_vol_recovers` | BS price | sigma recovered within 1e-8 |
| `test_implied_vol_zero_vega` | deep ITM | bisection fallback works |
| `test_put_call_parity_iv` | call IV = put IV | same sigma |
| `test_options_smoke` | all greeks | finite output |

### Exit Criteria (Phase 10)

```bash
test -f crates/quant-options/Cargo.toml
grep -q "quant-options" Cargo.toml
cargo test -p quant-options 2>&1 | grep -E "test result.*0 failed"
cargo clippy -p quant-options --all-targets -- -D warnings
cargo run -p quant-options --example greeks
cargo run -p quant-options --example implied_vol
test -f book/chapters/ch10.tex
```

### Guardrails

- **Approved dependencies**: `quant-core`, `quant-stochastic`, `thiserror`. Dev: `approx`, `quant-core`
- **FORBIDDEN**: `argmin`, `optimization`, `rand`, `nalgebra`, `statrs`
- **Package-scoped builds only**: `-p quant-options`
- **Reuse quant-stochastic**: `normal_cdf`, `bs_call`, `bs_put`, `d1`, `d2`
- **No new RNG**: Greeks are analytical; IV uses Newton/bisection

### Mathematical Reference

**Black-Scholes call**:
$$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$
$$d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Greeks (call)**:
$$\Delta = \Phi(d_1), \quad \Gamma = \frac{\phi(d_1)}{S_0 \sigma \sqrt{T}}, \quad \mathcal{V} = S_0 \phi(d_1) \sqrt{T}$$
$$\Theta = -\frac{S_0 \phi(d_1) \sigma}{2\sqrt{T}} - r K e^{-rT} \Phi(d_2), \quad \rho = K T e^{-rT} \Phi(d_2)$$

**Implied volatility** (Newton):
$$\sigma_{n+1} = \sigma_n - \frac{C(\sigma_n) - C_{\text{mkt}}}{\mathcal{V}(\sigma_n)}$$

---

## Completed: Phase 6 — Foundations (`quant-core`)

**Crate to create**: `quant-core`
**Book chapter**: `book/chapters/ch06.tex`
**Dependencies**: `qf-common` (optional), none required

### Overview

This phase begins the advanced quant track: hand-rolled math, no external
statistics crates. The crate introduces population/sample moments, a
deterministic RNG (xorshift64*), Box-Muller normal samples, and a geometric
Brownian motion path generator.

**Key insight**: The pedagogy is in the implementation. By building these
primitives from scratch, you deeply understand what finance libraries do
internally. The resulting code is deterministic and fully reproducible.

**Key constraint**: NO `rand`, `nalgebra`, or `statrs`. All math hand-rolled.

### Requirements

#### R6.1: Create quant-core Crate

```bash
cd crates/quant-lab/crates
cargo new quant-core --lib
```

Add to workspace `Cargo.toml`:
```toml
[workspace]
members = [
    "crates/qf-common",
    "crates/qf-01-fraud",
    "crates/qf-02-loan",
    "crates/qf-03-stocks",
    "crates/qf-04-returns",
    "crates/qf-05-backtest",
    "crates/quant-core",  # ADD THIS
]
```

Dependencies in `quant-core/Cargo.toml`:
```toml
[dependencies]
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
```

**No qf-common dependency required** — this crate is self-contained.

#### R6.2: PriceSeries Newtype

```rust
/// Newtype wrapper around Vec<f64> for price time series.
/// Provides domain-specific methods and enforces valid price data.
#[derive(Debug, Clone, PartialEq)]
pub struct PriceSeries(Vec<f64>);

impl PriceSeries {
    /// Create a new PriceSeries from a Vec<f64>.
    /// Returns error if any price is NaN, infinite, or negative.
    pub fn new(prices: Vec<f64>) -> Result<Self, CoreError>;

    /// Number of prices in the series.
    pub fn len(&self) -> usize;

    /// Whether the series is empty.
    pub fn is_empty(&self) -> bool;

    /// Get the underlying slice.
    pub fn as_slice(&self) -> &[f64];

    /// First price (if any).
    pub fn first(&self) -> Option<f64>;

    /// Last price (if any).
    pub fn last(&self) -> Option<f64>;

    /// Get price at index.
    pub fn get(&self, index: usize) -> Option<f64>;
}

impl From<PriceSeries> for Vec<f64> {
    fn from(ps: PriceSeries) -> Self { ps.0 }
}
```

#### R6.3: Returns Functions

```rust
/// Compute simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
/// Returns Vec of length n-1 for n prices.
pub fn simple_returns(prices: &[f64]) -> Vec<f64>;

/// Compute log returns: r_t = ln(P_t / P_{t-1})
/// Returns Vec of length n-1 for n prices.
pub fn log_returns(prices: &[f64]) -> Vec<f64>;
```

These mirror the qf-04-returns implementations but are self-contained.

#### R6.4: Moments (Population)

```rust
/// Compute the arithmetic mean: μ = (1/n) Σ x_i
pub fn mean(data: &[f64]) -> Option<f64>;

/// Compute population variance: σ² = (1/n) Σ (x_i - μ)²
pub fn variance(data: &[f64]) -> Option<f64>;

/// Compute population standard deviation: σ = √variance
pub fn std_dev(data: &[f64]) -> Option<f64>;

/// Compute population skewness (Fisher's definition):
/// γ₁ = (1/n) Σ [(x_i - μ) / σ]³
/// Returns None for n < 3 or zero variance.
pub fn skewness(data: &[f64]) -> Option<f64>;

/// Compute population excess kurtosis (Fisher's definition):
/// γ₂ = (1/n) Σ [(x_i - μ) / σ]⁴ - 3
/// Normal distribution has excess kurtosis = 0.
/// Returns None for n < 4 or zero variance.
pub fn excess_kurtosis(data: &[f64]) -> Option<f64>;
```

**Note**: Population formulas (divide by n, not n-1). Finance literature
commonly uses population moments for historical analysis.

#### R6.5: Rolling Windows

```rust
/// Compute rolling mean with window size w.
/// Returns Vec of length n - w + 1.
/// Returns empty Vec if data.len() < window or window == 0.
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64>;

/// Compute rolling standard deviation with window size w.
/// Uses population formula within each window.
/// Returns Vec of length n - w + 1.
/// Returns empty Vec if data.len() < window or window < 2.
pub fn rolling_std_dev(data: &[f64], window: usize) -> Vec<f64>;
```

#### R6.6: XorShift64 RNG

```rust
/// A simple, fast, seedable PRNG implementing xorshift64*.
/// NOT cryptographically secure — use only for simulations.
///
/// Algorithm: Marsaglia's xorshift with multiplication (xorshift64*).
/// Period: 2^64 - 1.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new RNG with the given seed.
    /// Seed must be non-zero; zero seed is replaced with default.
    pub fn new(seed: u64) -> Self;

    /// Generate the next u64 value.
    pub fn next_u64(&mut self) -> u64;

    /// Generate a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64;
}
```

**Implementation**:
```rust
// xorshift64* algorithm
pub fn next_u64(&mut self) -> u64 {
    let mut x = self.state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    self.state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

pub fn next_f64(&mut self) -> f64 {
    // Upper 53 bits -> f64 in [0, 1)
    (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}
```

#### R6.7: Box-Muller Normal Sampling

```rust
/// Generate a pair of independent standard normal samples using Box-Muller.
///
/// Given U₁, U₂ ~ Uniform(0, 1):
/// Z₁ = √(-2 ln U₁) cos(2π U₂)
/// Z₂ = √(-2 ln U₁) sin(2π U₂)
///
/// Returns (Z₁, Z₂) where Z₁, Z₂ ~ N(0, 1).
pub fn box_muller(rng: &mut XorShift64) -> (f64, f64);

/// Generate n standard normal samples.
/// Uses Box-Muller internally, may generate n+1 samples and discard.
pub fn normal_samples(rng: &mut XorShift64, n: usize) -> Vec<f64>;
```

#### R6.8: Geometric Brownian Motion

```rust
/// Generate paths of Geometric Brownian Motion.
///
/// dS = μ S dt + σ S dW
///
/// Discretization (Euler-Maruyama):
/// S_{t+Δt} = S_t exp((μ - σ²/2)Δt + σ√Δt Z)
///
/// # Arguments
/// * `s0` - Initial price
/// * `mu` - Drift (annualized)
/// * `sigma` - Volatility (annualized)
/// * `t` - Time horizon in years
/// * `n_steps` - Number of time steps per path
/// * `n_paths` - Number of paths to generate
/// * `rng` - Random number generator (seeded for reproducibility)
///
/// # Returns
/// Vec of paths, each path is Vec<f64> of length n_steps + 1 (includes S0).
///
/// # Errors
/// Returns error if parameters are invalid (negative sigma, non-positive s0, etc.)
pub fn gbm_paths(
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    rng: &mut XorShift64,
) -> Result<Vec<Vec<f64>>, CoreError>;
```

#### R6.9: Example Binary

Create `quant-core/examples/moments_demo.rs`:

```rust
// Demonstrate all Phase 6 functionality:
// 1. Create PriceSeries
// 2. Compute returns
// 3. Compute moments (mean, variance, std_dev, skewness, kurtosis)
// 4. Compute rolling statistics
// 5. Generate GBM paths with seeded RNG
// 6. Compute moments of simulated returns

// Output format:
// ==============================
// quant-core: Moments & Simulation Demo
// ==============================
//
// PriceSeries: 100 synthetic prices
// Simple returns: 99 values
//
// Moments:
//   Mean return:        0.0012
//   Std dev:            0.0156
//   Skewness:           0.0234
//   Excess kurtosis:    0.1567
//
// Rolling statistics (window=20):
//   Rolling mean: 80 values
//   Rolling std:  80 values
//
// GBM Simulation (seed=42):
//   S0=100, μ=0.10, σ=0.20, T=1yr
//   Paths: 1000, Steps: 252
//   Final price range: [45.23, 312.67]
//   Mean final price:  110.52 (expected: ~110.52)
// ==============================
```

#### R6.10: Book Chapter

Create `book/chapters/ch06.tex` with:

1. **Learning objectives**:
   - Understand higher-order moments (skewness, kurtosis)
   - Build a seedable PRNG from scratch
   - Generate normal samples via Box-Muller
   - Simulate asset price paths with GBM

2. **Why hand-roll?** — The pedagogical value of implementation

3. **Population moments** (with formulas in margin notes):
   - Mean, variance, standard deviation
   - Skewness: asymmetry measure
   - Kurtosis: tail heaviness (leptokurtic vs platykurtic)

4. **Rolling statistics** — Sliding window analysis

5. **Random number generation**:
   - Why determinism matters for reproducibility
   - XorShift64* algorithm explained
   - Converting u64 → f64 in [0, 1)

6. **Box-Muller transform**:
   - Mathematical derivation
   - Why it produces two independent normals
   - Code walkthrough

7. **Geometric Brownian Motion**:
   - SDE formulation: dS = μS dt + σS dW
   - Euler-Maruyama discretization
   - Why log-normal prices
   - Relationship to Black-Scholes

8. **Rust code samples** — Full implementations

9. **Exercises**:
   - Implement sample moments (n-1 denominator)
   - Compare skewness of different distributions
   - Verify GBM expected value analytically
   - Implement antithetic variates for variance reduction

### TDD Contract (16 tests)

**File**: `crates/quant-core/tests/core_tests.rs`

| Test name | Given | Expects |
|---|---|---|
| `test_price_series_new_valid` | `[100.0, 101.0, 102.0]` | Ok(PriceSeries) |
| `test_price_series_new_invalid_nan` | `[100.0, NaN, 102.0]` | Err |
| `test_price_series_new_invalid_negative` | `[100.0, -1.0, 102.0]` | Err |
| `test_simple_returns` | `[100.0, 105.0, 110.25]` | `[0.05, 0.05]` |
| `test_log_returns` | `[100.0, 105.0, 110.25]` | `[ln(1.05), ln(1.05)]` |
| `test_mean` | `[1.0, 2.0, 3.0, 4.0, 5.0]` | `3.0` |
| `test_variance` | `[1.0, 2.0, 3.0, 4.0, 5.0]` | `2.0` (population) |
| `test_std_dev` | `[1.0, 2.0, 3.0, 4.0, 5.0]` | `√2 ≈ 1.4142` |
| `test_skewness_symmetric` | `[1.0, 2.0, 3.0, 4.0, 5.0]` | `0.0` |
| `test_excess_kurtosis_uniform` | 5 uniform values | negative (platykurtic) |
| `test_rolling_mean` | `[1,2,3,4,5]`, window=3 | `[2.0, 3.0, 4.0]` |
| `test_rolling_std_dev` | `[1,2,3,4,5]`, window=3 | 3 values, each `√(2/3) ≈ 0.8165` |
| `test_xorshift64_deterministic` | seed=42, 5 calls | same sequence always |
| `test_xorshift64_f64_range` | 1000 samples | all in `[0, 1)` |
| `test_box_muller_distribution` | 10000 samples | mean ≈ 0, std ≈ 1 |
| `test_gbm_paths_shape` | n_steps=100, n_paths=50 | 50 paths, each 101 points |

### Exit Criteria (Phase 6)

Run from repository root:

```bash
# Structure exists
test -f crates/quant-core/Cargo.toml
grep -q "quant-core" Cargo.toml

# Tests pass (16 tests)
cargo test -p quant-core 2>&1 | grep -E "test result.*0 failed"

# No clippy warnings
cargo clippy -p quant-core --all-targets 2>&1 | grep -qv "^warning:"

# README exists with module overview
test -f crates/quant-core/README.md

# Example runs
cargo run -p quant-core --example moments_demo 2>&1 | grep -i "GBM Simulation"

# Book chapter exists
test -f book/chapters/ch06.tex

# NO forbidden dependencies
! grep -E "^rand\s*=" crates/quant-core/Cargo.toml
! grep -E "^nalgebra\s*=" crates/quant-core/Cargo.toml
! grep -E "^statrs\s*=" crates/quant-core/Cargo.toml
```

### Guardrails

- **Approved dependencies**: `thiserror`. Dev: `approx`
- **FORBIDDEN dependencies**: `rand`, `nalgebra`, `statrs`, any external statistics crate
- **Package-scoped builds only**: `-p quant-core`
- **All math hand-rolled**: Formulas must be implemented from scratch
- **Deterministic RNG**: Same seed → same output, always
- **No external data files**: All examples use synthetic data
- **Population moments**: Use divide-by-n formulas (not n-1)

### Mathematical Reference

**Skewness** (Fisher's definition):
$$\gamma_1 = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \mu}{\sigma} \right)^3$$

**Excess Kurtosis** (Fisher's definition):
$$\gamma_2 = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \mu}{\sigma} \right)^4 - 3$$

**Box-Muller Transform**:
$$Z_1 = \sqrt{-2 \ln U_1} \cos(2\pi U_2)$$
$$Z_2 = \sqrt{-2 \ln U_1} \sin(2\pi U_2)$$

**GBM Discretization** (Euler-Maruyama):
$$S_{t+\Delta t} = S_t \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} Z\right)$$

where $Z \sim N(0,1)$.

---

## Completed: Phase 5 — Backtesting Basics
**Book chapter**: `book/chapters/ch05.tex`
**Dependencies**: `qf-common`, `qf-03-stocks`, `qf-04-returns`

### Overview

This phase introduces systematic trading strategy evaluation:
- Signal generation from indicators (SMA crossover)
- Position management (long/short/flat)
- P&L (Profit & Loss) accounting with transaction costs
- Performance evaluation using Phase 4 metrics
- Walk-forward validation concepts

**Key insight**: A backtest is a simulation, not a prediction. It tells you
how a strategy *would have* performed, not how it *will* perform. Overfitting
to historical data is the #1 failure mode.

### Requirements

#### R5.1: Create qf-05-backtest Crate

```bash
cd crates/quant-lab/crates
cargo new qf-05-backtest --lib
```

Add to workspace `Cargo.toml`:
```toml
[workspace]
members = [
    "crates/qf-common",
    "crates/qf-01-fraud",
    "crates/qf-02-loan",
    "crates/qf-03-stocks",
    "crates/qf-04-returns",
    "crates/qf-05-backtest",  # ADD THIS
]
```

Dependencies in `qf-05-backtest/Cargo.toml`:
```toml
[dependencies]
qf-common = { path = "../qf-common" }
qf-03-stocks = { path = "../qf-03-stocks" }
qf-04-returns = { path = "../qf-04-returns" }
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
```

#### R5.2: Signal Types

```rust
/// Trading signal generated by a strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Buy,      // Enter long position
    Sell,     // Exit long / enter short
    Hold,     // No action
}

/// Position state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Long,     // Holding the asset
    Short,    // Short selling (future phases)
    Flat,     // No position
}
```

#### R5.3: Strategy Trait

```rust
/// Trait for trading strategies that generate signals from price data
pub trait Strategy {
    /// Generate a signal for the current bar given historical data
    /// `data` contains all bars up to and including current
    /// `index` is the current bar index (0-based)
    fn signal(&self, data: &[Ohlcv], index: usize) -> Signal;

    /// Strategy name for reporting
    fn name(&self) -> &str;
}
```

#### R5.4: SMA Crossover Strategy

```rust
/// Simple Moving Average crossover strategy
/// Buy when short SMA crosses above long SMA (golden cross)
/// Sell when short SMA crosses below long SMA (death cross)
pub struct SmaCrossover {
    short_period: usize,  // e.g., 10
    long_period: usize,   // e.g., 30
}

impl SmaCrossover {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self, BacktestError>;
}

impl Strategy for SmaCrossover {
    fn signal(&self, data: &[Ohlcv], index: usize) -> Signal;
    fn name(&self) -> &str;
}
```

Logic:
- Requires `index >= long_period` to have enough data
- Compute SMA(short) and SMA(long) at current and previous bar
- Golden cross: short was below, now above → Buy
- Death cross: short was above, now below → Sell
- Otherwise → Hold

#### R5.5: Backtest Engine

```rust
/// Configuration for a backtest run
pub struct BacktestConfig {
    /// Initial capital in dollars
    pub initial_capital: f64,

    /// Transaction cost per trade (e.g., 0.001 = 0.1%)
    pub transaction_cost: f64,

    /// Allow short selling (Phase 5: always false)
    pub allow_short: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            transaction_cost: 0.001,
            allow_short: false,
        }
    }
}

/// Run a backtest and return results
pub fn run_backtest<S: Strategy>(
    strategy: &S,
    data: &[Ohlcv],
    config: &BacktestConfig,
) -> BacktestResult;
```

#### R5.6: Backtest Result

```rust
/// Complete results from a backtest run
pub struct BacktestResult {
    /// Strategy name
    pub strategy_name: String,

    /// Starting capital
    pub initial_capital: f64,

    /// Ending capital
    pub final_capital: f64,

    /// Total return as decimal (e.g., 0.15 = 15%)
    pub total_return: f64,

    /// Number of trades executed
    pub num_trades: usize,

    /// Winning trades / total trades
    pub win_rate: f64,

    /// Total transaction costs paid
    pub total_costs: f64,

    /// Daily returns for further analysis
    pub daily_returns: Vec<f64>,

    /// Equity curve (capital at each bar)
    pub equity_curve: Vec<f64>,

    /// Trade log
    pub trades: Vec<Trade>,
}

/// Record of a single trade
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_date: String,
    pub entry_price: f64,
    pub exit_date: String,
    pub exit_price: f64,
    pub shares: f64,
    pub pnl: f64,        // Profit/Loss in dollars
    pub return_pct: f64, // Return as percentage
}
```

#### R5.7: Performance Metrics Integration

Use Phase 4 functions on `BacktestResult.daily_returns`:

```rust
impl BacktestResult {
    /// Annualized Sharpe ratio (assumes daily data)
    pub fn sharpe(&self, risk_free_rate: f64) -> f64 {
        qf_04_returns::annualized_sharpe(&self.daily_returns, risk_free_rate, 252.0)
    }

    /// Sortino ratio
    pub fn sortino(&self, risk_free_rate: f64) -> f64 {
        qf_04_returns::sortino_ratio(&self.daily_returns, risk_free_rate)
    }

    /// Maximum drawdown from equity curve
    pub fn max_drawdown(&self) -> f64 {
        qf_04_returns::max_drawdown(&self.equity_curve)
    }

    /// Annualized volatility
    pub fn volatility(&self) -> f64 {
        qf_04_returns::annualized_volatility(&self.daily_returns, 252.0)
    }
}
```

#### R5.8: Buy-and-Hold Benchmark

```rust
/// Passive strategy: buy on first bar, hold forever
pub struct BuyAndHold;

impl Strategy for BuyAndHold {
    fn signal(&self, _data: &[Ohlcv], index: usize) -> Signal {
        if index == 0 { Signal::Buy } else { Signal::Hold }
    }

    fn name(&self) -> &str { "Buy and Hold" }
}
```

This provides a benchmark to compare active strategies against.

#### R5.9: Example Binary

Create `qf-05-backtest/examples/backtest_demo.rs`:

```rust
// Load stock data
// Run SMA crossover strategy
// Run buy-and-hold benchmark
// Compare results

// Output format:
// ==============================
// Backtest Results
// ==============================
// Strategy: SMA Crossover (10/30)
// Period: 2024-01-01 to 2024-12-31
//
// Performance:
//   Total Return: 12.5%
//   Buy & Hold:   8.2%
//   Outperformance: +4.3%
//
// Risk Metrics:
//   Sharpe Ratio: 1.24
//   Sortino Ratio: 1.85
//   Max Drawdown: -8.5%
//   Volatility: 15.2%
//
// Trade Statistics:
//   Trades: 12
//   Win Rate: 58.3%
//   Avg Win: +2.1%
//   Avg Loss: -1.2%
//   Total Costs: $45.20
// ==============================
```

#### R5.10: Book Chapter

Create `book/chapters/ch05.tex` with:
- Learning objectives
- What is backtesting? (simulation vs reality)
- Signal generation and position management
- SMA crossover strategy (with margin notes for formulas)
- P&L accounting with transaction costs
- Performance evaluation (connecting to Chapter 4)
- Common pitfalls: overfitting, look-ahead bias, survivorship bias
- Rust code samples
- Exercises

### TDD Contract (20 tests)

**File**: `crates/qf-05-backtest/tests/backtest_tests.rs`

| Test name | Given | Expects |
|---|---|---|
| `test_signal_types` | Signal enum variants | Buy, Sell, Hold work |
| `test_position_types` | Position enum variants | Long, Short, Flat work |
| `test_sma_crossover_new_valid` | (10, 30) | Ok(strategy) |
| `test_sma_crossover_new_invalid` | (30, 10) short > long | Err |
| `test_sma_crossover_new_zero` | (0, 30) | Err |
| `test_sma_crossover_golden_cross` | prices with upward crossover | Signal::Buy |
| `test_sma_crossover_death_cross` | prices with downward crossover | Signal::Sell |
| `test_sma_crossover_no_signal` | prices without crossover | Signal::Hold |
| `test_sma_crossover_insufficient_data` | index < long_period | Signal::Hold |
| `test_buy_and_hold_first_bar` | index = 0 | Signal::Buy |
| `test_buy_and_hold_subsequent` | index > 0 | Signal::Hold |
| `test_backtest_config_default` | BacktestConfig::default() | capital=10000, cost=0.001 |
| `test_backtest_buy_and_hold` | 10 bars rising 1% each | total_return ≈ 10.46% |
| `test_backtest_with_costs` | trades with 0.1% cost | final_capital < no-cost |
| `test_backtest_equity_curve_length` | 100 bars | equity_curve.len() == 100 |
| `test_backtest_no_trades` | strategy that never trades | num_trades == 0 |
| `test_trade_pnl_calculation` | buy at 100, sell at 110 | pnl = 10% |
| `test_win_rate_calculation` | 3 wins, 2 losses | win_rate = 0.6 |
| `test_sharpe_integration` | backtest result | sharpe() returns f64 |
| `test_max_drawdown_integration` | backtest result | max_drawdown() returns f64 ≤ 0 |

### Exit Criteria (Phase 5)

Run from repository root:

```bash
# Structure exists
test -f crates/qf-05-backtest/Cargo.toml
grep -q "qf-05-backtest" Cargo.toml

# Tests pass
cargo test -p qf-05-backtest 2>&1 | grep -E "test result.*0 failed"

# No clippy warnings
cargo clippy -p qf-05-backtest --all-targets 2>&1 | grep -qv "^warning:"

# README exists
test -f crates/qf-05-backtest/README.md

# Example runs
cargo run -p qf-05-backtest --example backtest_demo 2>&1 | grep -i "sharpe"

# Book chapter exists
test -f book/chapters/ch05.tex
```

### Guardrails

- **Approved dependencies**: `thiserror`. Dev: `approx`
- **Depend on**: `qf-common`, `qf-03-stocks`, `qf-04-returns`
- **Package-scoped builds only**: `-p qf-05-backtest`
- **Tests use synthetic data**: No external files required
- **All math hand-rolled**: No external statistics libraries
- **Long-only for Phase 5**: Short selling deferred to later phases
- **No look-ahead bias**: `signal()` only sees data up to current index

---

## Code Conventions

### Error Handling

Use `thiserror` for custom error types:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BacktestError {
    #[error("Invalid strategy parameters: {0}")]
    InvalidParams(String),

    #[error("Insufficient data: need at least {required} bars, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
```

### Testing Pattern

Tests go in `tests/` directory with `approx` for float comparisons:

```rust
use approx::assert_relative_eq;

#[test]
fn test_backtest_buy_and_hold() {
    let data = create_rising_prices(10, 0.01); // 1% daily rise
    let strategy = BuyAndHold;
    let config = BacktestConfig::default();

    let result = run_backtest(&strategy, &data, &config);

    // Compound: (1.01)^10 - 1 ≈ 0.1046
    assert_relative_eq!(result.total_return, 0.1046, epsilon = 1e-3);
}
```

### Documentation

Every public item needs doc comments with formulas where applicable:

```rust
/// Run a complete backtest simulation.
///
/// # Algorithm
/// 1. Initialize position = Flat, capital = initial_capital
/// 2. For each bar in data:
///    a. Generate signal from strategy
///    b. Execute trades based on signal and current position
///    c. Apply transaction costs
///    d. Record equity
/// 3. Close any open position at end
/// 4. Calculate performance metrics
///
/// # Arguments
/// * `strategy` - Trading strategy implementing `Strategy` trait
/// * `data` - OHLCV price data ordered chronologically
/// * `config` - Backtest configuration (capital, costs)
///
/// # Returns
/// `BacktestResult` with performance metrics and trade log
pub fn run_backtest<S: Strategy>(
    strategy: &S,
    data: &[Ohlcv],
    config: &BacktestConfig,
) -> BacktestResult {
    // ...
}
```

---

## Building the Book

The book uses the kaobook LaTeX template and compiles remotely via Podman.

### Remote Compilation (Preferred)

```bash
# From quant-lab-rust directory
# 1. Sync files to remote server
rsync -avz --delete book/ mvcorrea@lnx:~/podman/latex-remote/quant-lab/src/

# 2. Compile on remote
ssh mvcorrea@lnx "podman exec latex-compiler-shared sh -c '
    cd /workspace/quant-lab/src && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    biber --output-directory ../build main && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex && \
    pdflatex -interaction=nonstopmode -output-directory=../build main.tex
'"

# 3. Download PDF
scp mvcorrea@lnx:~/podman/latex-remote/quant-lab/build/main.pdf book/build/quant-finance-book.pdf
```

### Adding Chapter 5

1. Create `book/chapters/ch05.tex`
2. Add to `main.tex` after ch04:
   ```latex
   \input{chapters/ch05}
   ```
3. Follow existing chapter structure (see ch04.tex for reference)

---

## Completed: Phase 11 — Portfolio Optimization: Markowitz and Beyond

Phase 11 is **complete**. All exit criteria pass:

- Crate: `quant-portfolio` (47 tests: 30 unit + 16 integration + 1 doc test, clippy clean)
- Modules: `linalg`, `portfolio`, `frontier`, `tangency`, `capm`, `risk`
- Examples: `frontier.rs`, `capm.rs` — both run with verified numerical output
- Book: Chapter 11 complete (6 pages, 6 code listings, TikZ efficient-frontier figure)
- Key results:
  - Global min-variance: w_A=0.6923, mu_p=0.0846, sigma_p=0.1664 (inverse-variance weights)
  - Tangency (max Sharpe): w_A=0.8571, mu_tan=0.0929, sigma_tan=0.1767, Sharpe=0.4123
  - CML: mu = 0.02 + 0.4123 * sigma (linear, dominates risky-only frontier above tangency)
  - CAPM: beta_hat=1.175 (true 1.20), alpha_hat=-0.000196 (essentially zero, on SML)
  - Historical VaR=0.0405, CVaR=0.0450 (95% confidence, linear interpolation)
- Memory key: `quant-finance/experiment/phase11-markowitz-portfolio`

Key implementation insight: The entire Markowitz framework reduces to linear algebra
in Sigma^{-1}; no QP solver or gradient descent is needed because the constraints
are all equalities (budget + optional target return).

---

## Completed: Phase 12 — Factor Attribution and PCA

Phase 12 is **complete**. All exit criteria pass:

- Crate: `quant-factors` (28 tests: 12 unit + 15 integration + 1 doc test, clippy clean)
- Modules: `eigen`, `pca`, `fama_french`, `risk`, `error`
- Examples: `pca_demo.rs`, `fama_french.rs` — both run with verified numerical output
- Book: Chapter 12 complete (7 code listings, TikZ scree-plot figure)
- Key results:
  - PCA: PC1 captures 96.1% of variance, reconstruction SSE decreases from 8.81e-5 (k=1) to 4.08e-23 (k=3)
  - FF3: alpha=0.001003, beta_mkt=1.2008, beta_smb=0.4016, beta_hml=-0.2977 (all within 0.003 of DGP)
  - FF3 R^2 = 0.9893 vs CAPM R^2 = 0.9391 (5.34% improvement)
  - Risk attribution: systematic 98.9%, idiosyncratic 1.1%
- Memory key: `quant-finance/experiment/phase12-factors-pca-fama-french`

Key implementation insight: The power method's sign-flip artefact is
handled by aligning each iterate with its predecessor before the
convergence test. Deflation accumulates floating-point error, so the
smaller eigenvalues are less accurate (1e-3 relative error vs 1e-6
for the dominant pair).

---

## Completed: Phase 13 — Market Microstructure

Phase 13 is **complete**. All exit criteria pass:

- Crate `quant-microstructure` created with 5 modules: `types`, `orderbook`, `flow`, `impact`, `error`
- `OrderBook` uses `BTreeMap<u64, Vec<Order>>` per side + `HashMap` id index for O(1) cancel
- Integer tick prices (u64) avoid float comparison issues
- Price-time priority (FIFO at each price level) verified by test
- OFI (Cont, Kukanov & Stoikov 2014), VWAP, trade imbalance implemented
- Square-root (Almgren-Chriss) and linear market impact models implemented
- `execution_cost` combines half-spread + sqrt impact in basis points
- 42 tests passing (26 unit + 15 integration + 1 doc test); clippy clean
- 2 examples: `lob_demo` (book walk + fills), `ofi_demo` (OFI series + impact)
- Book chapter `ch13.tex` written with 7 listings, TikZ LOB-depth figure, 5 exercises
- Book recompiled to 112 pages (756K)

### Key Findings

- For a 2000-share order at 0.2% participation, sigma=2%, spread=2bps:
  total execution cost = 9.94 bps (impact 8.94 + half-spread 1.00)
- Impact dominates spread by ~9x at this participation rate
- Square-root law verified: doubling order size multiplies impact by sqrt(2)
- For small orders (200 shares), impact falls to 2.83 bps and spread becomes the larger cost

---

## Next Task: Phase 14 — AFML Backtesting

**Crate to create**: `quant-backtest`
**Book chapter**: `book/chapters/ch14.tex`
**Dependencies**: `quant-core`, `quant-timeseries`, `quant-portfolio`, `thiserror`
**Spec file**: `crates/quant-backtest/spec.md` (complete spec with all details)

### Overview

Phase 14 implements the AFML (Advances in Financial Machine Learning) backtesting
framework: triple-barrier labeling, purged k-fold cross-validation, sample weights,
and bet sizing. This phase bridges academic portfolio theory to production-grade
strategy development by addressing overfitting, leakage, and proper label generation.

**Key insight**: Traditional backtesting suffers from look-ahead bias, overlapping
samples, and arbitrary labeling. AFML provides a principled framework: labels are
generated via triple-barrier (profit-taking, stop-loss, time horizon), samples are
weighted by uniqueness, and cross-validation is purged to prevent leakage.

**Key constraint**: Hand-rolled math. No external ML or optimization libraries.
Reuse fractional differentiation from `quant-timeseries` and risk metrics from
`quant-portfolio`. The Kelly criterion is implemented inline.

### Requirements

#### R14.1: Create quant-backtest Crate

```bash
cd crates/quant-lab/crates
cargo new quant-backtest --lib
```

Add to workspace `Cargo.toml`:
```toml
members = [
    ...
    "crates/quant-microstructure",
    "crates/quant-backtest",  # ADD THIS
]
```

Dependencies:
```toml
[dependencies]
quant-core = { path = "../quant-core" }
quant-timeseries = { path = "../quant-timeseries" }
quant-portfolio = { path = "../quant-portfolio" }
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
quant-core = { path = "../quant-core" }
```

#### R14.2: Triple-Barrier Labeling

```rust
/// Triple-barrier label configuration
pub struct TripleBarrierConfig {
    pub upper_barrier: f64,  // profit-taking (e.g., 0.02 = 2%)
    pub lower_barrier: f64,  // stop-loss (e.g., -0.02 = -2%)
    pub time_barrier: usize, // max holding period in bars
    pub min_return: f64,     // min return at time barrier
}

/// Result of triple-barrier labeling
pub enum TripleBarrierLabel { Upper, Lower, Time }

/// Event with label and metadata
pub struct LabeledEvent {
    pub entry_index: usize,
    pub exit_index: usize,
    pub label: TripleBarrierLabel,
    pub return_pct: f64,
    pub holding_period: usize,
}

/// Apply triple-barrier labeling to a price series
pub fn triple_barrier_label(prices: &[f64], config: &TripleBarrierConfig) -> Vec<LabeledEvent>;
```

#### R14.3: Sample Weights (Uniqueness-based)

```rust
/// Compute sample weights based on average uniqueness
/// Weight_i = 1 / (average concurrent events during event i)
pub fn sample_weights(events: &[LabeledEvent], n_bars: usize) -> Vec<f64>;

/// Compute concurrent events at each bar
pub fn concurrent_events(events: &[LabeledEvent], n_bars: usize) -> Vec<usize>;

/// Average uniqueness of each event
pub fn average_uniqueness(events: &[LabeledEvent], n_bars: usize) -> Vec<f64>;
```

#### R14.4: Purged K-Fold Cross-Validation

```rust
/// Purged k-fold split configuration
pub struct PurgedKFoldConfig {
    pub n_folds: usize,
    pub embargo: usize,  // bars after test to purge
}

/// A train/test split with purging info
pub struct PurgedSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub purged_count: usize,
    pub embargoed_count: usize,
}

/// Generate purged k-fold splits (no train-test overlap)
pub fn purged_kfold_splits(
    events: &[LabeledEvent], n_bars: usize, config: &PurgedKFoldConfig
) -> Vec<PurgedSplit>;
```

#### R14.5: Bet Sizing (Kelly Criterion)

```rust
/// Kelly criterion: f* = p - q/b
/// p = win probability, q = 1-p, b = win/loss ratio
pub fn kelly_fraction(win_prob: f64, win_loss_ratio: f64) -> f64;

/// Fractional Kelly for risk management
pub fn fractional_kelly(win_prob: f64, win_loss_ratio: f64, fraction: f64) -> f64;

/// Kelly from historical returns
pub fn kelly_from_returns(returns: &[f64]) -> f64;

pub struct PositionSize {
    pub kelly_full: f64,
    pub kelly_half: f64,
    pub win_probability: f64,
    pub win_loss_ratio: f64,
}
```

#### R14.6: AFML Backtest Engine

```rust
pub struct AfmlBacktestConfig {
    pub barrier_config: TripleBarrierConfig,
    pub cv_config: PurgedKFoldConfig,
    pub use_weights: bool,
    pub bet_sizing: BetSizing,
}

pub enum BetSizing { Equal, KellyFull, KellyHalf, Fixed(f64) }

pub struct AfmlBacktestResult {
    pub events: Vec<LabeledEvent>,
    pub weights: Vec<f64>,
    pub cv_splits: Vec<PurgedSplit>,
    pub in_sample_sharpe: f64,
    pub out_of_sample_sharpe: f64,
    pub position_size: PositionSize,
    pub total_return: f64,
    pub max_drawdown: f64,
}

pub fn afml_backtest(prices: &[f64], config: &AfmlBacktestConfig) -> Result<AfmlBacktestResult, BacktestError>;
```

#### R14.7: Example Binaries

- `triple_barrier.rs`: Demonstrate triple-barrier labeling on synthetic price data,
  show distribution of labels, holding periods, and returns.
- `purged_cv.rs`: Show purged k-fold CV, visualize train/test splits, count
  purged and embargoed samples.
- `kelly_sizing.rs`: Demonstrate Kelly criterion bet sizing, compare full vs
  half Kelly, show optimal position sizes.

#### R14.8: Book Chapter

`book/chapters/ch14.tex` with:
1. Motivation: why traditional backtesting fails
2. Triple-barrier method: profit-taking, stop-loss, time barriers
3. Label generation: binary labels from barrier hits
4. Sample uniqueness and weighting
5. Purged k-fold cross-validation: preventing leakage
6. Embargo period: handling autocorrelation
7. Kelly criterion: optimal bet sizing
8. Fractional Kelly: risk management
9. Meta-labeling concept (brief introduction)
10. Rust implementation with code listings
11. Exercises: custom barrier functions, embargo calibration

### TDD Contract (15 tests)

**File**: `crates/quant-backtest/tests/backtest_tests.rs`

| Test | Given | Expects |
|---|---|---|
| `t01_triple_barrier_upper_hit` | prices cross upper barrier | label = Upper |
| `t02_triple_barrier_lower_hit` | prices cross lower barrier | label = Lower |
| `t03_triple_barrier_time_hit` | prices stay within barriers | label = Time |
| `t04_triple_barrier_immediate_exit` | barrier hit on first bar | exit_index = entry_index + 1 |
| `t05_labeled_events_count` | 100 prices, every 10th as entry | ~10 events |
| `t06_concurrent_events_overlap` | overlapping events | concurrent > 1 |
| `t07_sample_weights_sum` | any events | weights normalized |
| `t08_average_uniqueness_bounds` | events | uniqueness in (0, 1] |
| `t09_purged_kfold_no_leakage` | 5 folds | no train overlaps test |
| `t10_purged_kfold_embargo` | embargo = 5 | 5 bars after test purged |
| `t11_kelly_criterion_basic` | p=0.6, b=1.0 | f* = 0.2 |
| `t12_kelly_zero_edge` | p=0.5, b=1.0 | f* = 0 |
| `t13_kelly_from_returns` | 60% wins, 1:1 ratio | f* ~ 0.2 |
| `t14_position_size_half_kelly` | full Kelly 0.2 | half Kelly 0.1 |
| `t15_afml_backtest_smoke` | synthetic prices | no panics, finite output |

### Exit Criteria (Phase 14)

```bash
test -f crates/quant-backtest/Cargo.toml
grep -q "quant-backtest" Cargo.toml
cargo test -p quant-backtest 2>&1 | grep -E "test result.*0 failed"
cargo clippy -p quant-backtest --all-targets -- -D warnings
cargo run -p quant-backtest --example triple_barrier
cargo run -p quant-backtest --example purged_cv
cargo run -p quant-backtest --example kelly_sizing
test -f crates/quant-backtest/README.md
test -f book/chapters/ch14.tex
```

### Guardrails

- **Approved dependencies**: `quant-core`, `quant-timeseries`, `quant-portfolio`, `thiserror`
  - Dev: `approx`, `quant-core`
- **FORBIDDEN**: `rand`, `nalgebra`, `statrs`, external ML libraries, optimization crates
- **Package-scoped builds only**: `-p quant-backtest`
- **All math hand-rolled**: Kelly criterion, sample weights, purging logic
- **Reuse existing crates**: fractional differentiation from `quant-timeseries`, risk metrics from `quant-portfolio`
- **Deterministic**: same input produces same output

### Mathematical Reference

**Triple-Barrier Return**:
$$r_t = \frac{P_{exit} - P_{entry}}{P_{entry}}$$

Label: Upper if $r_t \geq$ upper_barrier first, Lower if $r_t \leq$ lower_barrier first, Time otherwise.

**Sample Weight** (uniqueness-based):
$$w_i = \frac{1}{\bar{c}_i}$$
where $\bar{c}_i$ is the average number of concurrent events during event $i$.

**Average Uniqueness**:
$$u_i = \frac{1}{t_i^{end} - t_i^{start}} \sum_{t=t_i^{start}}^{t_i^{end}} \frac{1}{c_t}$$

**Kelly Criterion**:
$$f^* = \frac{pb - q}{b} = p - \frac{q}{b}$$
where $p$ = win probability, $q = 1-p$, $b$ = win/loss ratio.

### Reference: Future Phases

| Phase | Crate | Key Concepts |
|-------|-------|--------------|
| 14 | `quant-backtest` | Triple-barrier, purged CV (AFML) |
| 15 | `quant-lib` | Unified library consolidation |

### References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 3: Triple-Barrier Method
  - Chapter 4: Sample Weights
  - Chapter 7: Cross-Validation in Finance
  - Chapter 10: Bet Sizing
- Kelly, J. L. (1956). "A New Interpretation of Information Rate". *Bell System Technical Journal*.

---

## Git Workflow

```bash
# Standard workflow
git add <files>
git commit -m "feat(qf-05-backtest): implement SMA crossover strategy"

# Commit message prefixes
# feat: new feature
# fix: bug fix
# docs: documentation
# test: tests
# refactor: code refactoring

# Push to GitHub
git push origin master
```

---

## Troubleshooting

### Tests fail with permission denied

Tests write to system temp dir. If issues persist:
```bash
rm -rf /tmp/quant-lab-tests
```

### LaTeX compilation fails

1. Check container is running:
   ```bash
   ssh mvcorrea@lnx "podman ps | grep latex-compiler"
   ```

2. If not running:
   ```bash
   ssh mvcorrea@lnx "cd ~/podman/latex-remote && podman-compose up -d"
   ```

3. Check logs:
   ```bash
   ssh mvcorrea@lnx "cat ~/podman/latex-remote/quant-lab/build/main.log | tail -50"
   ```

### Clippy warnings

Fix all warnings before committing:
```bash
cargo clippy -p qf-05-backtest --all-targets -- -D warnings
```

---

## Summary: Current State

| Phase | Crate | Status | Tests |
|-------|-------|--------|-------|
| 1 | qf-01-fraud | COMPLETE | 10 |
| 2 | qf-02-loan | COMPLETE | 21 |
| 3 | qf-03-stocks | COMPLETE | 20 |
| 4 | qf-04-returns | COMPLETE | 25 |
| 5 | qf-05-backtest | COMPLETE | 20 |
| 6 | quant-core | COMPLETE | 17 |
| 7 | quant-timeseries | COMPLETE | 18 |
| 8 | quant-vol | COMPLETE | 17 |
| 9 | quant-stochastic | COMPLETE | 16 |
| 10 | quant-options | COMPLETE | 16 |
| 11 | quant-portfolio | COMPLETE | 47 |
| 12 | quant-factors | COMPLETE | 28 |
| 13 | quant-microstructure | COMPLETE | 42 |
| 14 | quant-backtest | **NEXT** | - |
| 15 | **quant-lib** | PLANNED | - |

**Total tests passing**: 312 (Phases 1-13)

**Final target**: Phase 15 consolidates all crates into unified `quant-lib`.

---

## Contact

Repository: https://github.com/bitscrafts/quant-lab-rust
Author: Marcelo Correa
