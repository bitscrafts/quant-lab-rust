# Portfolio Examples - Wall Street Interview Implementations

25 quantitative finance implementations based on Zhou's "Practical Guide to
Quantitative Finance Interviews" and AFML. All implementations use quant-lib
as the foundation.

**Status**: Research complete, implementation pending.

---

## Table of Contents

| # | File | Name | quant-lib Base | Status |
|---|------|------|----------------|--------|
| 01 | `01_monte_carlo_foundations.rs` | Monte Carlo Foundations | `quant-core` | TO BE IMPLEMENTED |
| 02 | `02_bayesian_inference_engine.rs` | Bayesian Inference Engine | `quant-core` | TO BE IMPLEMENTED |
| 03 | `03_hypothesis_testing_suite.rs` | Hypothesis Testing Suite | `quant-timeseries` | TO BE IMPLEMENTED |
| 04 | `04_correlation_analysis.rs` | Correlation Analysis | `quant-core` | TO BE IMPLEMENTED |
| 05 | `05_distribution_fitting.rs` | Distribution Fitting | `quant-vol` | TO BE IMPLEMENTED |
| 06 | `06_markov_chain_analyzer.rs` | Markov Chain Analyzer | `quant-timeseries` | TO BE IMPLEMENTED |
| 07 | `07_random_walk_simulator.rs` | Random Walk Simulator | `quant-stochastic` | TO BE IMPLEMENTED |
| 08 | `08_brownian_motion_engine.rs` | Brownian Motion Engine | `quant-stochastic` | TO BE IMPLEMENTED |
| 09 | `09_ito_calculus_toolkit.rs` | Ito Calculus Toolkit | `quant-stochastic` | TO BE IMPLEMENTED |
| 10 | `10_jump_diffusion_models.rs` | Jump Diffusion Models | `quant-stochastic` | TO BE IMPLEMENTED |
| 11 | `11_black_scholes_suite.rs` | Black-Scholes Suite | `quant-options` | TO BE IMPLEMENTED |
| 12 | `12_binomial_tree_pricer.rs` | Binomial Tree Pricer | `quant-options` | TO BE IMPLEMENTED |
| 13 | `13_monte_carlo_options.rs` | Monte Carlo Options | `quant-options` | TO BE IMPLEMENTED |
| 14 | `14_greeks_calculator.rs` | Greeks Calculator | `quant-options` | TO BE IMPLEMENTED |
| 15 | `15_exotic_options_pricer.rs` | Exotic Options Pricer | `quant-options` | TO BE IMPLEMENTED |
| 16 | `16_markowitz_optimizer.rs` | Markowitz Optimizer | `quant-portfolio` | TO BE IMPLEMENTED |
| 17 | `17_black_litterman_model.rs` | Black-Litterman Model | `quant-portfolio` | TO BE IMPLEMENTED |
| 18 | `18_risk_parity_allocator.rs` | Risk Parity Allocator | `quant-portfolio` | TO BE IMPLEMENTED |
| 19 | `19_factor_model_portfolio.rs` | Factor Model Portfolio | `quant-factors` | TO BE IMPLEMENTED |
| 20 | `20_robust_optimization.rs` | Robust Optimization | `quant-portfolio` | TO BE IMPLEMENTED |
| 21 | `21_regime_detection_hmm.rs` | Regime Detection HMM | `quant-vol` | TO BE IMPLEMENTED |
| 22 | `22_alpha_factor_mining.rs` | Alpha Factor Mining | `quant-factors` | TO BE IMPLEMENTED |
| 23 | `23_neural_option_pricer.rs` | Neural Option Pricer | `quant-options` | TO BE IMPLEMENTED |
| 24 | `24_reinforcement_trader.rs` | Reinforcement Trader | `quant-backtest` | TO BE IMPLEMENTED |
| 25 | `25_ensemble_predictor.rs` | Ensemble Predictor | `quant-backtest` | TO BE IMPLEMENTED |

---

## Categories

### Phase 1: Probability and Statistics (01-05)

| # | Name | Zhou Chapter | New Features |
|---|------|--------------|--------------|
| 01 | Monte Carlo Foundations | Ch.4 | Variance reduction, Sobol sequences |
| 02 | Bayesian Inference Engine | Ch.4 | MCMC, conjugate priors |
| 03 | Hypothesis Testing Suite | Ch.4 | t-test, JB, Ljung-Box, KS |
| 04 | Correlation Analysis | Ch.3-4 | Stress correlation, PD validation |
| 05 | Distribution Fitting | Ch.4 | MLE fitting, model selection |

### Phase 2: Stochastic Processes (06-10)

| # | Name | Zhou Chapter | New Features |
|---|------|--------------|--------------|
| 06 | Markov Chain Analyzer | Ch.5 | Stationary dist, hitting times |
| 07 | Random Walk Simulator | Ch.5 | Gambler's ruin, barriers |
| 08 | Brownian Motion Engine | Ch.5 | Ito integral, quadratic variation |
| 09 | Ito Calculus Toolkit | Ch.5 | Euler-Maruyama, Milstein |
| 10 | Jump Diffusion Models | Ch.5-6 | Merton, Kou models |

### Phase 3: Option Pricing (11-15)

| # | Name | Zhou Chapter | New Features |
|---|------|--------------|--------------|
| 11 | Black-Scholes Suite | Ch.6 | Dividends, binary options |
| 12 | Binomial Tree Pricer | Ch.6-7 | CRR, American, exercise boundary |
| 13 | Monte Carlo Options | Ch.6-7 | LSM American, variance reduction |
| 14 | Greeks Calculator | Ch.6 | Vanna, volga, surfaces |
| 15 | Exotic Options Pricer | Ch.6 | Barriers, Asians, lookbacks |

### Phase 4: Portfolio Optimization (16-20)

| # | Name | Zhou Chapter | New Features |
|---|------|--------------|--------------|
| 16 | Markowitz Optimizer | Ch.6 | QP constraints, transaction costs |
| 17 | Black-Litterman Model | External | Views, posterior returns |
| 18 | Risk Parity Allocator | External | Equal risk contribution |
| 19 | Factor Model Portfolio | Ch.6 | Factor timing, exposures |
| 20 | Robust Optimization | External | Uncertainty sets, worst-case |

### Phase 5: Machine Learning (21-25)

| # | Name | Source | New Features |
|---|------|--------|--------------|
| 21 | Regime Detection HMM | AFML | Viterbi, Baum-Welch |
| 22 | Alpha Factor Mining | AFML | IC, factor returns, LASSO |
| 23 | Neural Option Pricer | Research | Autodiff Greeks, GPU |
| 24 | Reinforcement Trader | Research | PPO agent, Gym env |
| 25 | Ensemble Predictor | AFML | Stacking, feature importance |

---

## Running Examples

Once implemented, run with:

```bash
# Single example
cargo run -p quant-lib --example portfolio-01_monte_carlo_foundations

# With dataset
cargo run -p quant-lib --example portfolio-16_markowitz_optimizer -- --dataset sp500
```

---

## Kaggle Datasets

Download before running experiments:

```bash
kaggle datasets download -d camnugent/sandp500 -p data/
kaggle datasets download -d dgawlik/nyse -p data/
kaggle datasets download -d kenshoresearch/option-data -p data/
kaggle datasets download -d sudalairajkumar/cryptocurrencypricehistory -p data/
```

---

## References

**Primary Sources**:
- Zhou, X. "A Practical Guide to Quantitative Finance Interviews" (Red Book)
- Lopez de Prado, M. "Advances in Financial Machine Learning" (AFML)

**quant-lib Modules Used**:
- `quant-core`: Moments, rolling windows, RNG, risk metrics
- `quant-stochastic`: Brownian motion, GBM, Poisson, Monte Carlo
- `quant-options`: Black-Scholes, Greeks, implied volatility
- `quant-portfolio`: Markowitz, efficient frontier, VaR/CVaR
- `quant-factors`: PCA, Fama-French, factor models
- `quant-vol`: EWMA, ARCH, GARCH
- `quant-timeseries`: OLS, ACF, ADF, fractional differentiation
- `quant-backtest`: Triple-barrier, purged CV, Kelly sizing

**External Crates** (only when quant-lib lacks feature):
- `polars`: DataFrame loading
- `linfa`: ML algorithms (HMM, clustering)
- `burn`: Neural networks
- `argmin`: Constrained optimization

---

## Progress

| Category | Total | Implemented | % |
|----------|-------|-------------|---|
| Probability (01-05) | 5 | 0 | 0% |
| Stochastic (06-10) | 5 | 0 | 0% |
| Options (11-15) | 5 | 0 | 0% |
| Portfolio (16-20) | 5 | 0 | 0% |
| ML (21-25) | 5 | 0 | 0% |
| **Total** | **25** | **0** | **0%** |
