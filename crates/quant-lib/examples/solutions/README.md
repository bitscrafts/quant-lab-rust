# Chapter Exercise Solutions

Complete solutions for all 14 book chapters. Each file contains 10
exercises that test understanding of chapter concepts through practical
implementation and verification.

## Solution Index

| Chapter | File | Topic | Exercises |
|---------|------|-------|-----------|
| 1 | `ch01_fraud_solutions.rs` | Credit Card Fraud | Anomaly scoring, ROC curves, precision-recall |
| 2 | `ch02_loan_solutions.rs` | Loan Default | Logistic features, confusion matrix, AUC |
| 3 | `ch03_stocks_solutions.rs` | Stock Analysis | OHLCV processing, technical indicators |
| 4 | `ch04_returns_solutions.rs` | Returns | Simple/log returns, distributions, autocorrelation |
| 5 | `ch05_backtest_solutions.rs` | Backtesting | Walk-forward, overfitting tests, metrics |
| 6 | `ch06_moments_solutions.rs` | Moments | Skewness, kurtosis, normality tests |
| 7 | `ch07_timeseries_solutions.rs` | Time Series | ARIMA, stationarity, structural breaks |
| 8 | `ch08_volatility_solutions.rs` | Volatility | EWMA, GARCH, realized volatility |
| 9 | `ch09_stochastic_solutions.rs` | Stochastic | GBM, Poisson, jump diffusion |
| 10 | `ch10_options_solutions.rs` | Options | Black-Scholes, Greeks, implied vol |
| 11 | `ch11_portfolio_solutions.rs` | Portfolio | Mean-variance, risk parity, Sharpe |
| 12 | `ch12_factors_solutions.rs` | Factors | PCA, Fama-French, factor attribution |
| 13 | `ch13_microstructure_solutions.rs` | Microstructure | LOB, bid-ask, market impact |
| 14 | `ch14_afml_solutions.rs` | AFML | Triple barrier, purged CV, Kelly |

## Running Solutions

```bash
# From quant-lab workspace root

# Run all exercises in a chapter
cargo run -p quant-lib --example solutions-ch01_fraud_solutions

# Run tests to verify solutions
cargo test -p quant-lib --example solutions-ch01_fraud_solutions

# Run all solution tests
cargo test -p quant-lib --examples
```

## Chapter Details

### Chapter 1: Credit Card Fraud Detection

Anomaly detection on the Kaggle credit card fraud dataset.

**Exercises:**
1. Load and parse CSV data
2. Compute class imbalance ratio
3. Z-score anomaly detection
4. ROC curve construction
5. Precision-recall trade-off
6. Threshold optimization
7. Cost-sensitive evaluation
8. Feature importance ranking
9. Confusion matrix analysis
10. Model comparison

### Chapter 2: Loan Default Prediction

Binary classification for credit risk.

**Exercises:**
1. Feature engineering for credit data
2. Logistic regression features
3. Confusion matrix metrics
4. AUC-ROC computation
5. Calibration curves
6. Default probability estimation
7. Risk bucketing
8. Gini coefficient
9. KS statistic
10. Expected loss calculation

### Chapter 3: Stock Price Analysis

Working with OHLCV market data.

**Exercises:**
1. Parse JSON price data
2. Compute daily returns
3. Technical indicators (SMA, EMA)
4. Bollinger Bands
5. RSI calculation
6. Volume analysis
7. Gap detection
8. Trend identification
9. Correlation matrices
10. Beta estimation

### Chapter 4: Return Analysis

Statistical properties of financial returns.

**Exercises:**
1. Simple vs log returns
2. Return distributions
3. QQ plots
4. Autocorrelation functions
5. Ljung-Box test
6. Heteroskedasticity detection
7. Tail behavior analysis
8. VaR estimation
9. Expected Shortfall
10. Drawdown analysis

### Chapter 5: Backtesting Strategies

Rigorous strategy evaluation.

**Exercises:**
1. Walk-forward framework
2. In-sample/out-of-sample split
3. Overfitting detection
4. Multiple testing correction
5. Deflated Sharpe ratio
6. Probabilistic Sharpe ratio
7. Strategy capacity
8. Transaction cost modeling
9. Slippage estimation
10. Benchmark comparison

### Chapter 6: Statistical Moments

Higher-order statistics.

**Exercises:**
1. Sample moments
2. Skewness calculation
3. Kurtosis (excess)
4. Jarque-Bera test
5. Shapiro-Wilk test
6. D'Agostino test
7. Anderson-Darling test
8. Moment stability
9. Rolling moments
10. Co-skewness/co-kurtosis

### Chapter 7: Time Series Analysis

ARIMA and stationarity.

**Exercises:**
1. ACF/PACF computation
2. ADF test implementation
3. KPSS test
4. Differencing for stationarity
5. ARMA parameter selection
6. Forecast generation
7. Residual diagnostics
8. Structural break detection
9. Cointegration testing
10. Error correction models

### Chapter 8: Volatility Modeling

GARCH and realized volatility.

**Exercises:**
1. EWMA volatility
2. GARCH(1,1) estimation
3. Volatility clustering
4. Realized variance
5. Bipower variation
6. Jump detection
7. Volatility forecasting
8. VIX-style calculation
9. Term structure
10. Volatility smile

### Chapter 9: Stochastic Processes

Continuous-time finance.

**Exercises:**
1. GBM simulation
2. Euler-Maruyama discretization
3. Poisson process
4. Compound Poisson
5. Jump-diffusion (Merton)
6. Variance gamma
7. Heston model
8. Path-dependent payoffs
9. Monte Carlo Greeks
10. Antithetic variates

### Chapter 10: Option Pricing

Black-Scholes and beyond.

**Exercises:**
1. BS call/put pricing
2. Greeks (delta, gamma, vega, theta, rho)
3. Implied volatility solver
4. Put-call parity
5. American options (binomial)
6. Dividend-adjusted pricing
7. Local volatility
8. IV surface construction
9. Smile dynamics
10. Exotic payoffs

### Chapter 11: Portfolio Optimization

Mean-variance and beyond.

**Exercises:**
1. Covariance estimation
2. Mean-variance frontier
3. Minimum variance portfolio
4. Maximum Sharpe portfolio
5. Risk parity weights
6. Black-Litterman views
7. Resampling efficiency
8. Constraint handling
9. Transaction cost optimization
10. Dynamic rebalancing

### Chapter 12: Factor Models

PCA and Fama-French.

**Exercises:**
1. PCA decomposition
2. Eigenportfolios
3. Factor loadings
4. Fama-French 3-factor
5. Factor attribution
6. Alpha estimation
7. Information ratio
8. Factor timing
9. Risk decomposition
10. Style analysis

### Chapter 13: Market Microstructure

Order book and market impact.

**Exercises:**
1. LOB construction
2. Bid-ask spread
3. Depth profile
4. Order flow imbalance
5. VWAP calculation
6. Market impact models
7. Kyle's lambda
8. Almgren-Chriss
9. Optimal execution
10. High-frequency patterns

### Chapter 14: AFML Techniques

Advances in Financial ML.

**Exercises:**
1. Fractional differentiation
2. Triple-barrier labeling
3. Meta-labeling
4. Sample uniqueness
5. Sample weights
6. Purged K-fold CV
7. Embargo periods
8. Walk-forward efficiency
9. Kelly criterion
10. Deflated Sharpe ratio

## Data Requirements

Solutions use bundled data in `quant-lab/data/`:

- `creditcard_sample.csv` - Chapters 1, 2
- `stock_prices.csv` - Chapter 3
- `*.json` (B3 stocks) - Chapters 4-14

## Test Structure

Each solution file contains:
- `main()` function running all exercises sequentially
- `#[test]` functions for automated verification
- Assertions validating expected behavior

Run tests to verify correctness:
```bash
cargo test -p quant-lib --example solutions-ch04_returns_solutions
```
