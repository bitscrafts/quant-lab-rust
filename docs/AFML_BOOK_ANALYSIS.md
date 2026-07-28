# AFML Book Analysis

**Source**: "Advances in Financial Machine Learning" by Marcos Lopez de Prado (Wiley, 2018)
**ISBN**: 978-1-119-48208-6
**Analysis Date**: 2026-07-27

---

## Executive Summary

AFML is the definitive guide for applying machine learning to quantitative finance.
Unlike standard ML textbooks, it addresses the unique challenges of financial data:
non-stationarity, low signal-to-noise ratio, overlapping labels, and backtest overfitting.

**Key Thesis**: Standard ML techniques fail in finance because they ignore the temporal
structure of financial data. The book provides solutions for each failure mode.

**Implementation Status in quant-lab**: 6 of 22 chapters fully implemented.

---

## Part 1: Data Analysis (Chapters 2-5)

### Chapter 2: Financial Data Structures

**Core Problem**: Traditional time-based bars (OHLCV at fixed intervals) exhibit
heteroscedasticity - volatility varies dramatically across time periods.

**Solutions**:
1. **Tick bars**: Sample after N trades → more uniform information content
2. **Volume bars**: Sample after V units traded → normalize by activity
3. **Dollar bars**: Sample after $D traded → normalize by value
4. **Information-driven bars**: Sample when information arrives (entropy-based)

**Key Insight**: Dollar bars are most homoscedastic for liquid assets.

**Mathematical Foundation**:
- Volatility of time bars: σ_t varies with time of day, news events
- Volatility of dollar bars: σ_$ ≈ constant across bars
- Test: Jarque-Bera on returns should be closer to normal for dollar bars

**Implementation Status**: NOT IMPLEMENTED
- Priority: MEDIUM (useful for real data, not needed for synthetic)
- Suggested Phase: 26 (Data Structures)

---

### Chapter 3: Labeling

**Core Problem**: Fixed-time horizon labels (e.g., "price up 1% in 10 days") ignore
the path taken and don't match how trades are actually managed.

**Solution: Triple-Barrier Method**

Three barriers around entry price:
- **Upper barrier** (profit-taking): Exit when return ≥ +τ_u
- **Lower barrier** (stop-loss): Exit when return ≤ -τ_l
- **Vertical barrier** (time): Exit after h bars if neither horizontal hit

Label = sign of return at first barrier touch.

**Mathematical Formulation**:
```
r_t = (P_exit - P_entry) / P_entry

Label:
  +1 if upper barrier hit first (r ≥ τ_u)
  -1 if lower barrier hit first (r ≤ τ_l)
  sign(r) if vertical barrier hit first
```

**8 Barrier Configurations** [pt, sl, t1]:
- [1,1,1]: Standard (all three barriers) ← **Most realistic**
- [0,1,1]: Stop-loss only (let winners run) ← Useful
- [1,1,0]: Take profit or stop (no time limit) ← Risky
- [0,0,1]: Fixed-time horizon ← Traditional but flawed
- Others: Generally impractical

**Dynamic Thresholds** (Section 3.3):
- Use rolling EWMA volatility to set barrier widths
- τ = k × σ_EWMA where k is chosen per strategy
- Adapts to market conditions automatically

**Meta-Labeling** (Section 3.6):
- Primary model: Predicts direction (long/short/neutral)
- Secondary model: Predicts size/confidence of primary's bet
- Benefits: Better F1 score, reduced false positives, bet sizing

**Implementation Status**: IMPLEMENTED (Phase 14)
- `triple_barrier_label()` in quant-backtest
- Missing: Dynamic thresholds (use rolling vol from quant-vol)
- Missing: Meta-labeling (planned Phase 18)

---

### Chapter 4: Sample Weights

**Core Problem**: Overlapping labels violate IID assumption. If label_i and label_j
both depend on price at time t, they are correlated.

**Solution: Uniqueness-Based Weighting**

**Concurrent Events Count**:
```
c_t = Σ_i 𝟙_{[t_i^start, t_i^end]}(t)
```
Number of events "active" at time t.

**Average Uniqueness**:
```
u_i = (1 / (t_i^end - t_i^start)) × Σ_{t=t_i^start}^{t_i^end} (1 / c_t)
```
How unique is event i? Ranges from 0 (fully overlapping) to 1 (no overlap).

**Sample Weight**:
```
w_i = u_i / Σ_j u_j  (normalized)
```

**Time Decay** (Section 4.7):
- Recent observations more relevant than old ones
- Apply exponential decay: w_i × exp(-λ × (T - t_i))

**Implementation Status**: IMPLEMENTED (Phase 14)
- `concurrent_events()`, `average_uniqueness()`, `sample_weights()` in quant-backtest

---

### Chapter 5: Fractionally Differentiated Features

**Core Problem**: The stationarity vs. memory dilemma.
- Stationary series (d=1 differencing): Lose memory, can't predict
- Non-stationary series (d=0): Have memory but violate ML assumptions

**Solution: Fractional Differentiation**

Apply d ∈ (0, 1) differentiation:
```
(1 - B)^d × X_t = Σ_{k=0}^{∞} ω_k × X_{t-k}

where ω_k = (-1)^k × C(d, k) = (-1)^k × (d! / (k! × (d-k)!))
```

For d = 0.5: √differentiation - partial stationarity with partial memory.

**Fixed-Width Window (FFD)**:
- Truncate the infinite sum at window size w
- Weight threshold τ: stop when |ω_k| < τ

**Finding Optimal d**:
1. Run ADF test at various d values
2. Find minimum d where ADF p-value < 0.05 (stationary)
3. Use that d for maximum memory preservation

**Implementation Status**: IMPLEMENTED (Phase 7)
- `frac_diff()` in quant-timeseries
- Includes ADF test for finding optimal d

---

## Part 2: Modelling (Chapters 6-9)

### Chapter 6: Ensemble Methods

**Core Concepts**:

**Bias-Variance Tradeoff**:
- Bias: Error from overly simplistic assumptions
- Variance: Error from sensitivity to training set fluctuations
- Ensemble methods reduce variance (bagging) or bias (boosting)

**Bagging** (Bootstrap Aggregating):
1. Create B bootstrap samples
2. Train model on each sample
3. Average predictions (regression) or vote (classification)
4. Reduces variance without increasing bias

**Random Forest**:
- Bagging + feature randomization
- At each split, consider only m ≈ √p features
- Further decorrelates trees → more variance reduction

**Boosting**:
- Sequentially fit weak learners to residuals
- Each learner focuses on previous errors
- Reduces bias but can overfit

**Finance-Specific Considerations**:
- Bagging preferred over boosting (financial data is noisy)
- Boosting amplifies noise → overfitting
- Use sample weights from Chapter 4 in bagging

**Implementation Status**: NOT IMPLEMENTED
- Priority: HIGH (core ML)
- Suggested Phase: 21 (Random Forest), 22 (Boosting)

---

### Chapter 7: Cross-Validation in Finance

**Core Problem**: Standard k-fold CV fails in finance because:
1. Observations are not IID (temporal dependence)
2. Labels overlap in time (Chapter 4)
3. Serial correlation in features

**Solution: Purged K-Fold CV**

**Purging**:
Remove from training any observation whose label overlaps with test set.
```
Purge if: t_i^start < t_test^end AND t_i^end > t_test^start
```

**Embargo**:
After purging, also remove training observations immediately following test set.
Handles serial correlation in features.
```
Embargo period: [t_test^end, t_test^end + h_embargo]
```

**Why Standard CV Fails**:
- Test observation at t=100 with label spanning [100, 110]
- Training observation at t=105 with label spanning [105, 115]
- Overlap at [105, 110] → information leakage

**Implementation Status**: IMPLEMENTED (Phase 14)
- `purged_kfold_splits()` in quant-backtest
- Includes embargo parameter

---

### Chapter 8: Feature Importance

**Methods**:

**MDI (Mean Decrease Impurity)**:
- How much does each feature reduce Gini/entropy across all trees?
- Fast but biased toward high-cardinality features
- Prone to false positives with correlated features

**MDA (Mean Decrease Accuracy)**:
- Permute feature values, measure accuracy drop
- More reliable but computationally expensive
- Use OOB samples for unbiased estimate

**SFI (Single Feature Importance)**:
- Train separate model using only one feature
- Cross-validate each single-feature model
- Avoids substitution effects of correlated features

**Clustered Feature Importance**:
- Group correlated features into clusters
- Compute importance at cluster level
- Reduces false positives from redundant features

**Implementation Status**: NOT IMPLEMENTED
- Priority: MEDIUM
- Suggested Phase: 25

---

### Chapter 9: Hyper-Parameter Tuning

**Key Points**:
- Use purged CV (Chapter 7) for all hyperparameter search
- Grid search: Exhaustive but expensive
- Random search: Often more efficient for high-dimensional spaces
- Cross-validate on proper CV, not information-leaking CV

**Scoring Considerations**:
- Accuracy misleading for imbalanced classes
- Use F1, AUC-ROC, or log-loss
- Consider negative log-loss for probability calibration

**Implementation Status**: Partially covered by purged CV
- Priority: LOW (can use existing ML frameworks)

---

## Part 3: Backtesting (Chapters 10-16)

### Chapter 10: Bet Sizing

**Core Problem**: Even with perfect prediction accuracy, poor bet sizing leads to ruin.

**Kelly Criterion**:
```
f* = (p × b - q) / b = p - q/b

where:
  p = win probability
  q = 1 - p = loss probability
  b = win/loss ratio (mean win / mean loss)
```

**From Returns**:
```
p = count(r > 0) / count(r)
b = mean(r | r > 0) / |mean(r | r < 0)|
f* = p - (1-p) / b
```

**Fractional Kelly**:
- Full Kelly maximizes growth but has high variance
- Half-Kelly (f = 0.5 × f*): Halves variance, reduces growth by ~25%
- Practical choice: 0.3 to 0.5 of full Kelly

**Bet Sizing from Probabilities**:
- ML model outputs probability p of positive return
- Size = 2 × p - 1 (maps [0.5, 1] → [0, 1])
- Or use CDF of historical signals for calibration

**Implementation Status**: IMPLEMENTED (Phase 14)
- `kelly_fraction()`, `fractional_kelly()`, `kelly_from_returns()` in quant-backtest

---

### Chapter 11: The Dangers of Backtesting

**Seven Sins of Quantitative Investing** (Luo et al.):
1. Survivorship bias
2. Look-ahead bias
3. Storytelling
4. Data mining / p-hacking
5. Transaction costs
6. Outliers
7. Shorting

**Key Insight**: "Backtesting is not a research tool. It is a verification tool."

**Recommendations**:
- Never optimize on backtest results
- Specify model FULLY before backtesting
- If backtest fails, start over (don't tweak)
- Use walk-forward or CPCV, not single historical run

**Implementation Status**: Guidance, no code needed

---

### Chapter 12: Backtesting through Cross-Validation

**Methods Compared**:

**Walk-Forward (WF)**:
- Train on [0, t], test on [t, t+h], roll forward
- Single path through data
- High variance in performance estimate

**Cross-Validation (CV)**:
- Multiple train/test splits
- Tests many scenarios but still limited paths
- CPCV generalizes this

**Combinatorial Purged Cross-Validation (CPCV)**:

Generate all C(N, k) combinations of k test folds from N folds.
Each combination produces one backtest path.

```
φ = C(N, k) = N! / (k! × (N-k)!)

Example: N=6 folds, k=2 test folds
φ = C(6,2) = 15 paths
```

**Benefits**:
- φ paths give distribution of Sharpe ratios
- Variance of mean Sharpe decreases as 1/φ
- Defeats overfitting: researcher can't optimize for unknown paths

**Deflated Sharpe Ratio**:
```
DSR = SR × √(1 - γ × SR² × (T/4))

where γ accounts for multiple testing
```

**Implementation Status**: NOT IMPLEMENTED
- Priority: HIGH
- Suggested Phase: 17

---

### Chapter 13: Backtesting on Synthetic Data

**Approach**:
1. Generate synthetic data with known properties
2. Embed known signal with known Sharpe ratio
3. Test if strategy recovers expected performance
4. Validates entire pipeline before real data

**Implementation Status**: Use existing quant-stochastic for data generation

---

### Chapter 14: Backtest Statistics

**General Characteristics**:
- Time range, AUM, leverage
- Number of positions, turnover

**Performance Metrics**:
- Total return, CAGR
- Sharpe ratio (annualized)
- Sortino ratio (downside deviation only)
- Information ratio (vs benchmark)
- Maximum drawdown, Calmar ratio

**Runs Statistics**:
- Average win, average loss
- Win rate, profit factor
- Longest winning/losing streak
- Time under water

**Classification Scores**:
- Accuracy, precision, recall, F1
- AUC-ROC
- Log-loss

**Implementation Status**: PARTIAL (in quant-portfolio)
- `sharpe_ratio`, `max_drawdown` available
- Missing: Sortino, Calmar, runs statistics

---

### Chapter 15: Understanding Strategy Risk

**Symmetric Payouts**:
- Normal distribution of returns
- Standard deviation measures risk
- Sharpe ratio applies

**Asymmetric Payouts**:
- Skewed distributions common in finance
- Left tail (losses) matters more than right
- Use CVaR (Conditional VaR) not just VaR

**Probability of Strategy Failure**:
- Given win rate p and edge per trade
- How many consecutive losses until ruin?
- Kelly sizing prevents ruin

---

### Chapter 16: Machine Learning Asset Allocation (HRP)

**Markowitz's Curse**:
1. **Instability**: Small changes in inputs → large changes in weights
2. **Concentration**: Optimizer concentrates on few assets
3. **Underperformance**: In-sample optimal ≠ out-of-sample optimal

**Hierarchical Risk Parity (HRP)**:

**Stage 1: Tree Clustering**
- Compute correlation matrix → distance matrix
- d(i,j) = √(0.5 × (1 - ρ_ij))
- Apply hierarchical clustering (single/complete/ward linkage)

**Stage 2: Quasi-Diagonalization**
- Reorder correlation matrix by cluster hierarchy
- Similar assets are adjacent
- Largest correlations near diagonal

**Stage 3: Recursive Bisection**
- Split universe into two clusters
- Allocate between clusters based on inverse variance
- Recursively apply to each cluster
- Result: Weights that respect hierarchical structure

**Mathematical Result**:
- HRP Sharpe ratio 31.3% better than CLA out-of-sample
- HRP variance 72.5% lower than CLA out-of-sample
- HRP works on singular covariance matrices (CLA fails)

**Implementation Status**: NOT IMPLEMENTED
- Priority: HIGH
- Suggested Phase: 16

---

## Part 4: Useful Financial Features (Chapters 17-19)

### Chapter 17: Structural Breaks

**CUSUM Test**:
- Cumulative sum of residuals from regression
- Detects when relationship breaks down
- Good for detecting regime changes

**SADF (Supremum ADF)**:
- Run ADF test on expanding window
- Take supremum of test statistics
- Detects bubbles: unit root → explosive → unit root

**GSADF (Generalized SADF)**:
- Also varies start point, not just end
- More powerful for multiple bubbles
- Phillips, Shi, Yu (2015)

**Implementation Status**: NOT IMPLEMENTED
- Priority: MEDIUM
- Suggested Phase: 19

---

### Chapter 18: Entropy Features

**Shannon Entropy**:
```
H(X) = -Σ p(x) × log₂(p(x))
```
Measures uncertainty/information content.

**Lempel-Ziv Complexity**:
- Compression-based entropy estimate
- Count distinct patterns in sequence
- Works for non-stationary data

**Financial Applications**:
- Market efficiency: High entropy → harder to predict
- Order flow entropy: Detect informed trading
- Volatility clustering: Low entropy during crises

**Implementation Status**: NOT IMPLEMENTED
- Priority: LOW
- Suggested Phase: 24

---

### Chapter 19: Microstructural Features

**First Generation: Price Sequences**
- Roll model: Estimate bid-ask spread from price autocorrelation
- High-low estimators: Use range to estimate volatility

**Second Generation: Strategic Trade Models**
- Kyle lambda: Price impact per unit traded
- Amihud illiquidity: |r| / volume

**Third Generation: Sequential Trade Models**
- VPIN (Volume-Synchronized PIN): Probability of informed trading
- Order flow imbalance: Buy pressure vs sell pressure

**Additional Features**:
- Hasbrouck lambda: Permanent price impact
- Trade runs: Consecutive same-side trades
- Volume clock: Time measured in volume, not seconds

**Implementation Status**: IMPLEMENTED (Phase 13)
- `quant-microstructure` crate
- LOB, OFI, market impact models

---

## Part 5: High-Performance Computing (Chapters 20-22)

### Chapter 20: Multiprocessing and Vectorization

**Vectorization**:
- Use NumPy/SIMD operations instead of loops
- 10-100x speedup for numerical code

**Multiprocessing vs Multithreading**:
- Python GIL makes threading ineffective for CPU-bound
- Use multiprocessing for parallelism
- Map function across cores

**Molecules**:
- Split work into independent chunks
- Each chunk processed by one worker
- Combine results

**Implementation Status**: Rust naturally handles this
- Use Rayon for parallelism
- SIMD via std::simd or packed_simd

---

### Chapter 21-22: Advanced Computing

Topics on quantum computing and HPC infrastructure.
Not directly implementable in quant-lab.

---

## Summary: Implementation Roadmap

### Implemented (6 chapters)
| Chapter | Topic | Phase |
|---------|-------|-------|
| 3.4 | Triple-barrier | 14 |
| 4 | Sample weights | 14 |
| 5 | Fractional diff | 7 |
| 7.4 | Purged k-fold CV | 14 |
| 10 | Kelly criterion | 14 |
| 19 | Microstructure | 13 |

### High Priority (4 chapters)
| Chapter | Topic | Suggested Phase |
|---------|-------|-----------------|
| 16 | HRP | 16 |
| 12 | CPCV | 17 |
| 6 | Random Forest | 21 |
| 6 | Boosting | 22 |

### Medium Priority (4 chapters)
| Chapter | Topic | Suggested Phase |
|---------|-------|-----------------|
| 3.6 | Meta-labeling | 18 |
| 17 | Structural breaks | 19 |
| 8 | Feature importance | 25 |
| 2 | Data structures | 26 |

### Low Priority (2 chapters)
| Chapter | Topic | Suggested Phase |
|---------|-------|-----------------|
| 18 | Entropy | 24 |
| 9 | Hyperparameters | (use ML frameworks) |

---

## Exercise Solutions Plan

Each AFML chapter ends with exercises. The book appendices will include:

1. **Theoretical exercises**: Mathematical derivations
2. **Implementation exercises**: Rust code with tests
3. **Data exercises**: Using provided datasets

Format for each solution:
```latex
\exercise{Chapter X, Exercise Y}
\textbf{Problem}: [problem statement]

\textbf{Solution}:
[step-by-step solution]

\textbf{Code}:
\begin{rustcode}
// Implementation
\end{rustcode}

\textbf{Test}:
\begin{rustcode}
#[test]
fn test_exercise_x_y() { ... }
\end{rustcode}
```

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Kelly, J. L. (1956). "A New Interpretation of Information Rate". Bell System Technical Journal.
3. Phillips, P. C., Shi, S., & Yu, J. (2015). "Testing for Multiple Bubbles". International Economic Review.
4. De Miguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification". Review of Financial Studies.
