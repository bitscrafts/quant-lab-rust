# Real-World Quantitative Finance Projects

This directory contains 5 production-like projects demonstrating multi-crate integration with `quant-lib`. Each project increases in complexity, building on concepts from the book chapters.

## Learning Path

```
Project 1 (Beginner)       Project 2 (Intermediate)    Project 3 (Advanced)
   Momentum Strategy  -->     Pairs Trading       -->    Risk Parity
   [qf-03, qf-04, qf-05]     [quant-timeseries]         [quant-portfolio]

                     Project 4 (Expert)           Project 5 (Expert)
                   -->  Options Market Making  -->  Full AFML Pipeline
                       [quant-options, micro]      [ALL crates]
```

## Project Index

| # | Project | Level | Crates Used | Concepts |
|---|---------|-------|-------------|----------|
| 1 | Momentum Strategy | Beginner | qf-03, qf-04, qf-05, quant-core | Cross-sectional momentum, decile portfolios, rebalancing |
| 2 | Pairs Trading | Intermediate | quant-timeseries, quant-core, qf-04 | Cointegration, z-score, mean reversion, spread trading |
| 3 | Risk Parity Portfolio | Advanced | quant-portfolio, quant-vol, quant-factors | Risk contribution, covariance estimation, optimization |
| 4 | Options Market Making | Expert | quant-options, quant-microstructure, quant-vol | IV surface, delta hedging, inventory management |
| 5 | AFML Pipeline | Expert | ALL | FFD, triple-barrier, purged CV, meta-labeling, Kelly, DSR |

## Quick Start

```bash
# Run any project
cargo run -p quant-lib --example project_01_momentum
cargo run -p quant-lib --example project_02_pairs
cargo run -p quant-lib --example project_03_risk_parity
cargo run -p quant-lib --example project_04_options_mm
cargo run -p quant-lib --example project_05_afml_pipeline
```

## Project Details

### Project 1: Cross-Sectional Momentum Strategy

**Level**: Beginner

**What You'll Learn**:
- Loading and processing OHLCV data
- Computing momentum (12-month returns)
- Ranking stocks by momentum
- Long-short portfolio construction
- Backtesting with transaction costs
- Performance metrics (Sharpe, max drawdown)

**Dataset**: S&P 500 daily prices (2010-2024)

**Strategy**: Buy top 10% momentum, sell bottom 10%, rebalance monthly.

**Key Insight**: Momentum is one of the most robust anomalies in finance, documented since Jegadeesh & Titman (1993). Winners tend to keep winning over 3-12 month horizons.

---

### Project 2: Statistical Arbitrage (Pairs Trading)

**Level**: Intermediate

**What You'll Learn**:
- Correlation vs cointegration
- ADF test for stationarity of spreads
- Z-score computation and normalization
- Mean-reversion trading signals
- Half-life of mean reversion
- Spread volatility for position sizing

**Dataset**: Sector ETFs (XLF, XLE, XLK, XLV, XLI, XLY, XLP, XLU)

**Strategy**: Find cointegrated pairs, trade when spread deviates > 2 standard deviations, close when it reverts to mean.

**Key Insight**: Pairs trading is market-neutral. The spread has a stationary distribution (if truly cointegrated), making mean-reversion predictable.

---

### Project 3: Risk Parity Portfolio

**Level**: Advanced

**What You'll Learn**:
- Risk contribution decomposition
- Marginal contribution to risk (MCR)
- Covariance estimation (EWMA, shrinkage)
- Optimization: equal risk contribution
- Comparison to Markowitz (60/40, min-variance)
- Regime analysis (performance in bull/bear markets)

**Dataset**: Multi-asset universe (equities, bonds, commodities)

**Strategy**: Allocate so each asset contributes equally to total portfolio risk, not equal weight.

**Key Insight**: Traditional 60/40 is actually ~90% equity risk. Risk parity diversifies *risk*, not *dollars*, leading to more stable drawdowns.

---

### Project 4: Options Market Making

**Level**: Expert

**What You'll Learn**:
- Implied volatility surface fitting
- Black-Scholes pricing and Greeks
- Delta-neutral hedging strategy
- Bid-ask spread determination
- Inventory risk management
- P&L decomposition by Greek

**Dataset**: SPY options chain (synthetic or CBOE)

**Strategy**: Quote two-sided markets, hedge delta continuously, profit from gamma/theta decay.

**Key Insight**: Market makers don't predict direction. They profit from bid-ask spread and volatility risk premium, hedging away directional exposure.

---

### Project 5: Full AFML Pipeline

**Level**: Expert

**What You'll Learn**:
- Fractional differentiation for stationarity
- Triple-barrier labeling with volatility scaling
- Purged k-fold cross-validation with embargo
- Meta-labeling for confidence filtering
- Kelly criterion with volatility scaling
- Walk-forward validation efficiency (WFE)
- Deflated Sharpe Ratio (DSR) for multiple testing

**Dataset**: E-mini S&P 500 futures (1-minute bars)

**Pipeline**:
```
Raw Data → FFD Transform → Triple-Barrier Labels → Features →
Purged CV → Primary Model → Meta-Model → Kelly Sizing →
Walk-Forward → DSR Assessment → VIABLE/NOT_VIABLE
```

**Key Insight**: This is the complete López de Prado workflow. Every step addresses a specific backtest pitfall: FFD handles non-stationarity, purged CV prevents leakage, meta-labeling improves precision, Kelly optimizes growth, WFE detects overfitting, and DSR corrects for selection bias.

---

## Data Sources

### Free Datasets

| Dataset | Source | Projects |
|---------|--------|----------|
| S&P 500 daily (50+ years) | [Kaggle](https://www.kaggle.com/datasets/samyakrajbayar/s-and-p-500-complete-historical-dataset-50-years) | 1, 3 |
| Minute bars (S&P, BTC, etc.) | [GitHub FutureSharks](https://github.com/FutureSharks/financial-data) | 5 |
| ETF prices | Yahoo Finance (via CSV export) | 2, 3 |
| FRED economic data | [FRED API](https://fred.stlouisfed.org/) | 3 |

### Synthetic Generation

If real data unavailable, projects include synthetic data generation:

```rust
use quant_lib::stochastic::GbmSimulator;

let sim = GbmSimulator::new(42); // deterministic seed
let prices = sim.simulate(S0: 100.0, mu: 0.08, sigma: 0.20, T: 10.0, n: 2520);
```

---

## Expected Outputs

### Project 1: Momentum Strategy
```
=== Momentum Strategy Results ===
Period: 2010-01-01 to 2024-12-31
Universe: 500 stocks

Performance:
  Annualized Return: 8.4%
  Sharpe Ratio: 0.72
  Max Drawdown: 18.3%
  Annual Turnover: 2.4x

Comparison (benchmark: equal-weight S&P 500):
  Strategy Sharpe: 0.72
  Benchmark Sharpe: 0.58
  Outperformance: +0.14
```

### Project 5: AFML Pipeline
```
=== AFML Pipeline Results ===

Data Preparation:
  FFD d*: 0.42 (ADF p < 0.01)
  Samples: 50,000 events

Labeling:
  Config: pt=2σ, sl=2σ, max_bars=20
  Distribution: +1=32%, 0=38%, -1=30%

Cross-Validation (Purged k=5, embargo=5):
  IS Sharpe: 1.82 ± 0.15
  OOS Sharpe: 1.05 ± 0.22

Meta-Labeling:
  Accuracy: 58%
  Filtered Sharpe: 1.35

Final Assessment:
  Walk-Forward Efficiency: 0.58 (> 0.5 ✓)
  Deflated Sharpe: 0.92 (p=0.04 < 0.05 ✓)
  Max Drawdown: 12.3%
  Calmar Ratio: 1.42

Conclusion: VIABLE
```

---

## Further Reading

### Books
- López de Prado, "Advances in Financial Machine Learning" (2018)
- Grinold & Kahn, "Active Portfolio Management" (1999)
- Pedersen, "Efficiently Inefficient" (2015)

### Papers
- Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers" (1993)
- Asness et al., "Value and Momentum Everywhere" (2013)
- Qian, "Risk Parity Portfolios" (2005)

### Online
- [QuantInsti Trading Strategies](https://www.quantinsti.com/articles/types-trading-strategies/)
- [AFML Implementations](https://github.com/kubahamerlik/afml-implementations)
- [Mean Reversion Strategies](https://blog.quantinsti.com/mean-reversion-strategies-introduction-building-blocks/)
