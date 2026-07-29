//! Kelly criterion bet sizing (AFML Ch. 10 + Kelly 1956).
//!
//! The Kelly criterion maximises the long-run growth rate of a bankroll
//! when betting a fixed fraction `f` of wealth on each trial. For a
//! binary outcome with win probability `p`, loss probability `q = 1-p`,
//! and win/loss ratio `b` (mean win divided by mean loss):
//!
//! $$f^* = \frac{pb - q}{b} = p - \frac{q}{b}.$$
//!
//! Fractional Kelly scales this down (`f = fraction * f*`) to reduce
//! variance at the cost of expected growth. Half-Kelly (`fraction=0.5`)
//! is the common practical choice.

/// Full Kelly fraction: `f* = p - q/b` where `p` is win probability,
/// `q = 1 - p`, and `b` is win/loss ratio.
pub fn kelly_fraction(win_prob: f64, win_loss_ratio: f64) -> f64 {
    if win_loss_ratio <= 0.0 {
        return 0.0;
    }
    let p = win_prob.clamp(0.0, 1.0);
    let q = 1.0 - p;
    p - q / win_loss_ratio
}

/// Fractional Kelly: `f = fraction * kelly_fraction(p, b)`.
pub fn fractional_kelly(win_prob: f64, win_loss_ratio: f64, fraction: f64) -> f64 {
    fraction * kelly_fraction(win_prob, win_loss_ratio)
}

/// Estimate Kelly from a series of trade returns. Win probability is
/// the fraction of positive returns; the win/loss ratio is the mean
/// positive return divided by the absolute mean negative return.
pub fn kelly_from_returns(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
    let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    let p = wins.len() as f64 / returns.len() as f64;
    if losses.is_empty() || wins.is_empty() {
        return 0.0;
    }
    let mean_win: f64 = wins.iter().sum::<f64>() / wins.len() as f64;
    let mean_loss: f64 = -losses.iter().sum::<f64>() / losses.len() as f64;
    if mean_loss == 0.0 {
        return 0.0;
    }
    let b = mean_win / mean_loss;
    kelly_fraction(p, b)
}

/// Position sizing result.
#[derive(Debug, Clone)]
pub struct PositionSize {
    pub kelly_full: f64,
    pub kelly_half: f64,
    pub win_probability: f64,
    pub win_loss_ratio: f64,
}

/// Compute position sizing from a series of trade returns.
pub fn compute_position_size(trade_returns: &[f64]) -> PositionSize {
    if trade_returns.is_empty() {
        return PositionSize {
            kelly_full: 0.0,
            kelly_half: 0.0,
            win_probability: 0.0,
            win_loss_ratio: 0.0,
        };
    }
    let wins: Vec<f64> = trade_returns
        .iter()
        .filter(|&&r| r > 0.0)
        .copied()
        .collect();
    let losses: Vec<f64> = trade_returns
        .iter()
        .filter(|&&r| r < 0.0)
        .copied()
        .collect();
    let p = wins.len() as f64 / trade_returns.len() as f64;
    let mean_win: f64 = if wins.is_empty() {
        0.0
    } else {
        wins.iter().sum::<f64>() / wins.len() as f64
    };
    let mean_loss: f64 = if losses.is_empty() {
        0.0
    } else {
        -losses.iter().sum::<f64>() / losses.len() as f64
    };
    let b = if mean_loss == 0.0 {
        0.0
    } else {
        mean_win / mean_loss
    };
    let kelly_full = kelly_fraction(p, b);
    PositionSize {
        kelly_full,
        kelly_half: 0.5 * kelly_full,
        win_probability: p,
        win_loss_ratio: b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_basic() {
        // p=0.6, b=1.0 -> f* = 0.6 - 0.4/1.0 = 0.2
        let f = kelly_fraction(0.6, 1.0);
        assert!((f - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_kelly_zero_edge() {
        // p=0.5, b=1.0 -> f* = 0.5 - 0.5/1.0 = 0
        let f = kelly_fraction(0.5, 1.0);
        assert!(f.abs() < 1e-9);
    }

    #[test]
    fn test_kelly_from_returns_sixty_percent() {
        // 60% wins at +1, 40% losses at -1: b=1, p=0.6 -> f*=0.2
        let returns = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let f = kelly_from_returns(&returns);
        assert!((f - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_half_kelly() {
        let full = kelly_fraction(0.6, 1.0);
        let half = fractional_kelly(0.6, 1.0, 0.5);
        assert!((half - 0.5 * full).abs() < 1e-9);
        assert!((half - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_compute_position_size() {
        let returns = vec![1.0, 1.0, 1.0, -1.0, -1.0];
        let ps = compute_position_size(&returns);
        // p = 3/5 = 0.6, b = 1, kelly = 0.2
        assert!((ps.win_probability - 0.6).abs() < 1e-9);
        assert!((ps.win_loss_ratio - 1.0).abs() < 1e-9);
        assert!((ps.kelly_full - 0.2).abs() < 1e-9);
        assert!((ps.kelly_half - 0.1).abs() < 1e-9);
    }
}
