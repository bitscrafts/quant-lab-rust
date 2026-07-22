//! Drawdown analysis: peak-to-trough decline from a running maximum.
//!
//! See `README.md` in this directory for the module overview.

/// Computes the drawdown series of a price series.
///
/// The drawdown at time `t` is the percentage decline from the running maximum
/// of prices observed up to and including `t`:
///
/// ```text
/// dd_t = (P_t - max_{k<=t} P_k) / max_{k<=t} P_k
/// ```
///
/// Drawdowns are always non-positive (zero when `P_t` is a new high).
///
/// # Arguments
/// * `prices` - Slice of prices ordered chronologically.
///
/// # Returns
/// Vector of drawdown values with the same length as `prices`. An empty input
/// yields an empty output. A running maximum of zero yields `0.0` at that
/// index (avoids division by zero).
pub fn drawdown(prices: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(prices.len());
    let mut running_max = f64::NEG_INFINITY;
    for &p in prices {
        if p > running_max {
            running_max = p;
        }
        if running_max == 0.0 {
            out.push(0.0);
        } else {
            out.push((p - running_max) / running_max);
        }
    }
    out
}

/// Returns the maximum drawdown (most negative value) of a price series.
///
/// # Arguments
/// * `prices` - Slice of prices ordered chronologically.
///
/// # Returns
/// The most negative drawdown value (e.g. `-0.20` for a 20% peak-to-trough
/// decline), or `0.0` for empty input or a series that never declines.
pub fn max_drawdown(prices: &[f64]) -> f64 {
    drawdown(prices).iter().copied().fold(0.0_f64, f64::min)
}

/// Summary statistics for a drawdown series.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawdownStats {
    /// The most negative drawdown value (e.g. `-0.20` for a 20% drawdown).
    pub max_drawdown: f64,
    /// Longest run of consecutive strictly-negative drawdown observations
    /// before recovery to a new high.
    pub max_drawdown_duration: usize,
    /// Number of steps from the peak preceding the max drawdown until the
    /// series reaches a new high again. `None` if the series never recovered.
    pub recovery_time: Option<usize>,
}

/// Computes drawdown summary statistics for a price series.
///
/// # Arguments
/// * `prices` - Slice of prices ordered chronologically.
///
/// # Returns
/// A `DrawdownStats` describing the worst drawdown episode. For empty input
/// all fields are zero / `None`.
pub fn drawdown_stats(prices: &[f64]) -> DrawdownStats {
    let dd = drawdown(prices);
    if dd.is_empty() {
        return DrawdownStats {
            max_drawdown: 0.0,
            max_drawdown_duration: 0,
            recovery_time: None,
        };
    }

    // Max drawdown value and the index where it occurs.
    let mut max_dd = 0.0_f64;
    let mut trough_idx = 0;
    for (i, &v) in dd.iter().enumerate() {
        if v < max_dd {
            max_dd = v;
            trough_idx = i;
        }
    }

    // Longest run of strictly-negative drawdowns.
    let mut longest = 0_usize;
    let mut current = 0_usize;
    for &v in &dd {
        if v < 0.0 {
            current += 1;
            if current > longest {
                longest = current;
            }
        } else {
            current = 0;
        }
    }

    // Recovery time: steps from the trough back to a new high (drawdown == 0
    // after the trough). If the series never makes a new high after trough,
    // recovery_time is None.
    let mut recovery = None;
    for (i, &v) in dd.iter().enumerate().skip(trough_idx + 1) {
        if v == 0.0 {
            recovery = Some(i - trough_idx);
            break;
        }
    }

    DrawdownStats {
        max_drawdown: max_dd,
        max_drawdown_duration: longest,
        recovery_time: recovery,
    }
}