# Phase 16 Real-World Projects

Five standalone, runnable Rust examples that demonstrate real
quantitative-finance trading and research workflows on top of the
`quant-lib` facade. Each project ties together multiple sub-crates
(`quant-core`, `quant-timeseries`, `quant-vol`, `quant-stochastic`,
`quant-options`, `quant-portfolio`, `quant-factors`,
`quant-microstructure`, `quant-backtest`) into a single end-to-end
pipeline that loads real data, runs a strategy, and prints headline
metrics.

## Project index

| Project | File | Level | Crates used | Run command |
|---------|------|-------|------------|-------------|
| 1. Cross-Sectional Momentum | `01_momentum.rs` | Beginner | `core` | `cargo run -p quant-lib --example projects-01_momentum` |
| 2. Cointegration Pairs Trading | `02_pairs_trading.rs` | Intermediate | `core`, `timeseries` | `cargo run -p quant-lib --example projects-02_pairs_trading` |
| 3. Risk-Parity Portfolio | `03_risk_parity.rs` | Advanced | `core`, `portfolio` | `cargo run -p quant-lib --example projects-03_risk_parity` |
| 4. Options Market Making | `04_options_mm.rs` | Expert | `stochastic`, `options` | `cargo run -p quant-lib --example projects-04_options_mm` |
| 5. Full AFML Pipeline | `05_afml_pipeline.rs` | Expert | `core`, `timeseries`, `backtest`, `risk` | `cargo run -p quant-lib --example projects-05_afml_pipeline` |

## Project 1: Cross-Sectional Momentum

**What it demonstrates.** A long/short equity momentum strategy that
ranks 8 B3 stocks by their past 20-day return, goes long the top 2 and
short the bottom 2 with equal weights, and rebalances every 21 bars
(monthly). It is the smallest self-contained trading backtest in the
curriculum.

**Key concepts.** Cross-sectional ranking, equal-weight long/short
portfolio construction, monthly rebalancing, total return, annualised
Sharpe ratio, maximum drawdown.

**Expected output metrics.** Total return, annualised Sharpe, max
drawdown, and a 63-day rolling mean of portfolio returns as a smoothed
performance indicator. With the bundled B3 data the strategy loses
over the 2021-2024 window because pure cross-sectional momentum on
Brazilian equities is trend-following and the period was choppy.

## Project 2: Cointegration Pairs Trading

**What it demonstrates.** A statistical-arbitrage pairs trade on PETR4
vs VALE3. An OLS regression of PETR4 on VALE3 estimates the hedge
ratio; an ADF test on the residuals checks for cointegration; the
residual z-score drives entry (|z| > 2) and exit (|z| < 0.5) rules.

**Key concepts.** OLS regression, ADF unit-root test, spread z-score,
mean-reversion trading rules, hit rate, trade P&L.

**Expected output metrics.** OLS alpha/beta and R-squared, ADF
statistic vs the 5% critical value (-2.86), number of trades, hit
rate, total spread P&L, and annualised Sharpe. With the bundled data
the ADF does not reject the unit root at 5% (the pair is weakly
cointegrated), but the z-score rule still produces a positive Sharpe.

## Project 3: Risk-Parity Portfolio

**What it demonstrates.** Three portfolio allocations on 8 B3 stocks:
inverse-variance (risk parity), equal weight, and the Markowitz global
minimum-variance portfolio. All three are evaluated on the same
annualised covariance matrix at rf = 0.0.

**Key concepts.** Sample covariance estimation, inverse-variance
weighting, Markowitz minimum-variance frontier, Sharpe ratio
comparison, weight-budget sanity checks.

**Expected output metrics.** A weights table (RiskParity / EqualWgt
/ MinVar), a performance table (return, volatility, Sharpe, weight
sum), and per-asset annualised volatilities. The minimum-variance
portfolio achieves the highest Sharpe and lowest volatility, as
theory predicts.

## Project 4: Options Market Making

**What it demonstrates.** A market-maker sells an at-the-money
European call on a simulated GBM price path and delta-hedges it daily.
First an implied-volatility surface is built by pricing calls on a
grid of strikes and maturities at a "market" vol smile and inverting
with the implied-vol solver (a round-trip sanity check). Then the
short-call position is hedged by holding `+delta` shares, rebalanced
each day.

**Key concepts.** GBM simulation, Black-Scholes pricing, IV surface
construction, IV solver round-trip, delta-neutral hedging, hedge P&L
accounting, hedging error.

**Expected output metrics.** An IV surface table (strike, maturity,
market vol, recovered IV) with near-zero round-trip error; the option
premium; cumulative hedge cost; share liquidation value; option
payoff; total P&L (small but non-zero due to discrete daily
rebalancing gamma slippage); and the maximum hedging error.

## Project 5: Full AFML Pipeline

**What it demonstrates.** An end-to-end Lopez de Prado AFML pipeline
on PETR4 closes: fractional differentiation for stationary
memory-preserving features, triple-barrier labeling, walk-forward
cross-validation, Kelly bet sizing, walk-forward efficiency, and the
deflated Sharpe ratio.

**Key concepts.** Fractional differentiation (d=0.4), ADF stationarity
verification, triple-barrier labeling, walk-forward CV folds, Kelly
criterion, walk-forward efficiency (OOS/IS Sharpe), deflated Sharpe
ratio (multiple-testing adjustment).

**Expected output metrics.** FFD series length and ADF statistic;
labeled event counts (Upper/Lower/Time); number of walk-forward
folds; full and half-Kelly fractions; WFE; and the DSR. A summary
table at the end collects all six step outputs.

## Data

All projects use the bundled B3 stock JSON files in
`crates/quant-lab/data/`:

| File | Ticker | Bars | Window |
|------|--------|------|--------|
| `PETR4.json` | Petrobras | ~1247 | Jul 2021 - Jul 2024 |
| `VALE3.json` | Vale | ~1247 | Jul 2021 - Jul 2024 |
| `ITSA4.json` | Itausa | ~1247 | Jul 2021 - Jul 2024 |
| `BBDC4.json` | Bradesco | ~1247 | Jul 2021 - Jul 2024 |
| `B3SA3.json` | B3 | ~1247 | Jul 2021 - Jul 2024 |
| `ABEV3.json` | Ambev | ~1247 | Jul 2021 - Jul 2024 |
| `GGBR4.json` | Gerdau | ~1247 | Jul 2021 - Jul 2024 |
| `WEGE3.json` | Weg | ~1247 | Jul 2021 - Jul 2024 |

Project 4 is fully synthetic (GBM simulation) and does not load any
data files.

## Running

From the `crates/quant-lab` workspace root:

```bash
# Compile-check all projects
cargo check -p quant-lib --examples

# Lint all projects
cargo clippy -p quant-lib --examples -- -D warnings

# Run an individual project
cargo run -p quant-lib --example projects-01_momentum
cargo run -p quant-lib --example projects-02_pairs_trading
cargo run -p quant-lib --example projects-03_risk_parity
cargo run -p quant-lib --example projects-04_options_mm
cargo run -p quant-lib --example projects-05_afml_pipeline
```

All five projects are registered as explicit `[[example]]` entries in
`crates/quant-lib/Cargo.toml` under the `# Phase 16 real-world
projects.` header, because cargo does not auto-discover example files
in subdirectories.