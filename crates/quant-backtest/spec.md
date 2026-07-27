# Phase 14 Spec: quant-backtest — AFML Backtesting Framework

**Project**: `/Users/mvcorrea/Private/PROJECTS/20260601-on-research/crates/quant-lab`
**Crate**: `quant-backtest`
**Book**: `book/chapters/ch14.tex`
**Dependencies**: `quant-core`, `quant-timeseries`, `quant-portfolio`, `thiserror`

---

## Overview

Phase 14 implements the AFML (Advances in Financial Machine Learning) backtesting
framework: triple-barrier labeling, purged k-fold cross-validation, sample weights,
and bet sizing. This phase bridges academic portfolio theory to production-grade
strategy development by addressing the critical problems of overfitting, leakage,
and proper label generation.

**Key insight**: Traditional backtesting suffers from look-ahead bias, overlapping
samples, and arbitrary labeling. AFML provides a principled framework: labels are
generated via triple-barrier (profit-taking, stop-loss, time horizon), samples are
weighted by uniqueness, and cross-validation is purged to prevent leakage.

**Key constraint**: Hand-rolled math. No external ML or optimization libraries.
Reuse fractional differentiation from `quant-timeseries` and risk metrics from
`quant-portfolio`. The Kelly criterion is implemented inline.

---

## Requirements

### R14.1: Create quant-backtest Crate

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

### R14.2: Triple-Barrier Labeling

```rust
/// Triple-barrier label configuration
#[derive(Debug, Clone)]
pub struct TripleBarrierConfig {
    /// Upper barrier: profit-taking threshold (e.g., 0.02 = 2%)
    pub upper_barrier: f64,
    /// Lower barrier: stop-loss threshold (e.g., -0.02 = -2%)
    pub lower_barrier: f64,
    /// Time barrier: maximum holding period in bars
    pub time_barrier: usize,
    /// Minimum return to label as positive at time barrier
    pub min_return: f64,
}

/// Result of triple-barrier labeling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TripleBarrierLabel {
    /// Upper barrier hit first (profit-taking)
    Upper,
    /// Lower barrier hit first (stop-loss)
    Lower,
    /// Time barrier hit (timeout)
    Time,
}

/// Event with label and metadata
#[derive(Debug, Clone)]
pub struct LabeledEvent {
    pub entry_index: usize,
    pub exit_index: usize,
    pub label: TripleBarrierLabel,
    pub return_pct: f64,
    pub holding_period: usize,
}

/// Apply triple-barrier labeling to a price series.
/// Returns a vector of labeled events, one per entry point.
pub fn triple_barrier_label(
    prices: &[f64],
    config: &TripleBarrierConfig,
) -> Vec<LabeledEvent>;

/// Convert TripleBarrierLabel to binary: Upper -> 1, Lower/Time -> 0
pub fn to_binary_label(label: TripleBarrierLabel, event: &LabeledEvent) -> i32;
```

### R14.3: Sample Weights (Uniqueness-based)

```rust
/// Compute sample weights based on average uniqueness.
/// Overlapping events reduce sample importance.
///
/// Weight_i = 1 / (average number of concurrent events during event i)
pub fn sample_weights(events: &[LabeledEvent], n_bars: usize) -> Vec<f64>;

/// Compute concurrent events at each bar.
/// Returns a vector of length n_bars with count of active events.
pub fn concurrent_events(events: &[LabeledEvent], n_bars: usize) -> Vec<usize>;

/// Average uniqueness of each event.
/// Uniqueness_i = mean(1 / concurrent_count) over event duration.
pub fn average_uniqueness(events: &[LabeledEvent], n_bars: usize) -> Vec<f64>;
```

### R14.4: Purged K-Fold Cross-Validation

```rust
/// Purged k-fold split configuration
#[derive(Debug, Clone)]
pub struct PurgedKFoldConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Embargo period after test set (in bars)
    pub embargo: usize,
}

/// A single train/test split with purging
#[derive(Debug, Clone)]
pub struct PurgedSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub purged_count: usize,
    pub embargoed_count: usize,
}

/// Generate purged k-fold splits for labeled events.
/// Purging: remove training samples that overlap with test period.
/// Embargo: remove training samples immediately after test period.
pub fn purged_kfold_splits(
    events: &[LabeledEvent],
    n_bars: usize,
    config: &PurgedKFoldConfig,
) -> Vec<PurgedSplit>;

/// Check if an event overlaps with a time range.
pub fn event_overlaps(event: &LabeledEvent, start: usize, end: usize) -> bool;
```

### R14.5: Bet Sizing (Kelly Criterion)

```rust
/// Kelly criterion for optimal position sizing.
/// f* = (p * b - q) / b = p - q/b
/// where p = win probability, q = 1-p, b = win/loss ratio
pub fn kelly_fraction(win_prob: f64, win_loss_ratio: f64) -> f64;

/// Fractional Kelly for risk management.
/// f = kelly_fraction * fraction (typically 0.5 for half-Kelly)
pub fn fractional_kelly(win_prob: f64, win_loss_ratio: f64, fraction: f64) -> f64;

/// Compute optimal bet size from historical returns.
pub fn kelly_from_returns(returns: &[f64]) -> f64;

/// Position sizing result
#[derive(Debug, Clone)]
pub struct PositionSize {
    pub kelly_full: f64,
    pub kelly_half: f64,
    pub win_probability: f64,
    pub win_loss_ratio: f64,
}

/// Compute position sizing from a series of trade returns.
pub fn compute_position_size(trade_returns: &[f64]) -> PositionSize;
```

### R14.6: Backtest Engine (AFML-style)

```rust
/// AFML backtest configuration
#[derive(Debug, Clone)]
pub struct AfmlBacktestConfig {
    /// Triple-barrier labeling config
    pub barrier_config: TripleBarrierConfig,
    /// Cross-validation config
    pub cv_config: PurgedKFoldConfig,
    /// Use sample weights
    pub use_weights: bool,
    /// Bet sizing method
    pub bet_sizing: BetSizing,
}

/// Bet sizing method
#[derive(Debug, Clone, Copy)]
pub enum BetSizing {
    /// Equal position sizes
    Equal,
    /// Full Kelly criterion
    KellyFull,
    /// Half Kelly (more conservative)
    KellyHalf,
    /// Fixed fraction
    Fixed(f64),
}

/// AFML backtest result
#[derive(Debug, Clone)]
pub struct AfmlBacktestResult {
    /// Labeled events with returns
    pub events: Vec<LabeledEvent>,
    /// Sample weights
    pub weights: Vec<f64>,
    /// Cross-validation splits
    pub cv_splits: Vec<PurgedSplit>,
    /// In-sample performance
    pub in_sample_sharpe: f64,
    /// Out-of-sample performance (average across folds)
    pub out_of_sample_sharpe: f64,
    /// Kelly position sizing
    pub position_size: PositionSize,
    /// Total return using bet sizing
    pub total_return: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
}

/// Run AFML-style backtest on price data.
pub fn afml_backtest(
    prices: &[f64],
    config: &AfmlBacktestConfig,
) -> Result<AfmlBacktestResult, BacktestError>;
```

### R14.7: Meta-Labeling (Optional Extension)

```rust
/// Meta-labeling: predict the size/confidence of a primary model's prediction.
/// The primary model generates signals; meta-labeling predicts if the signal is correct.
pub struct MetaLabel {
    /// Original signal (1 = long, -1 = short, 0 = no signal)
    pub signal: i32,
    /// Actual outcome (1 = correct, 0 = incorrect)
    pub outcome: i32,
    /// Confidence score from meta-model
    pub confidence: f64,
}

/// Compute meta-labels from signals and actual returns.
pub fn meta_labels(
    signals: &[i32],
    actual_returns: &[f64],
) -> Vec<MetaLabel>;
```

### R14.8: Example Binaries

- `triple_barrier.rs`: Demonstrate triple-barrier labeling on synthetic price data,
  show distribution of labels, holding periods, and returns.
- `purged_cv.rs`: Show purged k-fold CV, visualize train/test splits, count
  purged and embargoed samples.
- `kelly_sizing.rs`: Demonstrate Kelly criterion bet sizing, compare full vs
  half Kelly, show optimal position sizes.

### R14.9: Book Chapter

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

---

## TDD Contract (15 tests)

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

---

## Exit Criteria

```bash
# Crate exists
test -f crates/quant-backtest/Cargo.toml

# Workspace includes crate
grep -q "quant-backtest" Cargo.toml

# All tests pass
cargo test -p quant-backtest 2>&1 | grep -E "test result.*0 failed"

# Clippy clean
cargo clippy -p quant-backtest --all-targets -- -D warnings

# Examples run
cargo run -p quant-backtest --example triple_barrier
cargo run -p quant-backtest --example purged_cv
cargo run -p quant-backtest --example kelly_sizing

# README exists
test -f crates/quant-backtest/README.md

# Book chapter exists
test -f book/chapters/ch14.tex
```

---

## Guardrails

- **Approved dependencies**: `quant-core`, `quant-timeseries`, `quant-portfolio`, `thiserror`
  - Dev: `approx`, `quant-core`
- **FORBIDDEN**: `rand`, `nalgebra`, `statrs`, external ML libraries, optimization crates
- **Package-scoped builds only**: `-p quant-backtest`
- **All math hand-rolled**: Kelly criterion, sample weights, purging logic
- **Reuse existing crates**: fractional differentiation from `quant-timeseries`, risk metrics from `quant-portfolio`
- **No look-ahead bias**: triple-barrier uses only future prices from entry point (intentional for labeling)
- **Deterministic**: same input produces same output (use `quant-core` RNG if needed)

---

## Mathematical Reference

**Triple-Barrier Return**:
$$r_t = \frac{P_{exit} - P_{entry}}{P_{entry}}$$

Label:
- Upper: $r_t \geq \text{upper\_barrier}$ hit first
- Lower: $r_t \leq \text{lower\_barrier}$ hit first
- Time: neither barrier hit within $\text{time\_barrier}$ bars

**Sample Weight** (uniqueness-based):
$$w_i = \frac{1}{\bar{c}_i}$$
where $\bar{c}_i$ is the average number of concurrent events during event $i$.

**Concurrent Events**:
$$c_t = \sum_{i} \mathbb{1}_{[t_i^{start}, t_i^{end}]}(t)$$

**Average Uniqueness**:
$$u_i = \frac{1}{t_i^{end} - t_i^{start}} \sum_{t=t_i^{start}}^{t_i^{end}} \frac{1}{c_t}$$

**Purged K-Fold**: For test fold $[t_1, t_2]$, purge training samples where
$t_i^{start} < t_2$ and $t_i^{end} > t_1$ (overlap with test).

**Embargo**: Additionally purge training samples where
$t_i^{start} \in [t_2, t_2 + \text{embargo}]$.

**Kelly Criterion**:
$$f^* = \frac{pb - q}{b} = p - \frac{q}{b}$$
where $p$ = win probability, $q = 1-p$, $b$ = win/loss ratio.

**Win/Loss Ratio**:
$$b = \frac{\bar{r}^+}{|\bar{r}^-|}$$
where $\bar{r}^+$ is mean positive return, $\bar{r}^-$ is mean negative return.

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 3: Triple-Barrier Method
  - Chapter 4: Sample Weights
  - Chapter 7: Cross-Validation in Finance
  - Chapter 10: Bet Sizing
- Kelly, J. L. (1956). "A New Interpretation of Information Rate". *Bell System Technical Journal*.
