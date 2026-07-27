//! Market impact models: square-root (Almgren-Chriss) and linear.
//!
//! The square-root law states that the impact of an order is
//! proportional to the volatility times the square root of the
//! participation rate `Q / V` (order size / daily volume). This is an
//! empirical regularity observed across equity markets.

/// Square-root market impact (Almgren-Chriss).
///
/// `Impact = sigma * sqrt(Q / V)` where `sigma` is the daily volatility,
/// `Q` is the order size, and `V` is the average daily volume. The
/// result is a fraction of the price (e.g., 0.001 = 10 bps).
pub fn sqrt_impact(volatility: f64, order_size: f64, daily_volume: f64) -> f64 {
    if daily_volume <= 0.0 {
        return 0.0;
    }
    let participation = order_size / daily_volume;
    volatility * participation.abs().sqrt()
}

/// Linear temporary impact.
///
/// `Impact = eta * (Q / V)` where `eta` is the impact coefficient and
/// `Q / V` is the participation rate.
pub fn linear_impact(eta: f64, order_size: f64, daily_volume: f64) -> f64 {
    if daily_volume <= 0.0 {
        return 0.0;
    }
    eta * (order_size / daily_volume)
}

/// Estimated execution cost for a given order size.
///
/// Combines the half-spread (paid by the aggressor) with the
/// square-root impact to give a total cost in basis points of the
/// mid price.
#[derive(Debug, Clone)]
pub struct ExecutionCost {
    pub spread_cost: f64,
    pub impact_cost: f64,
    pub total_cost: f64,
}

/// Estimate the execution cost for a market order of `order_size` shares
/// given the market conditions.
///
/// # Arguments
/// * `order_size` - Number of shares to execute.
/// * `daily_volume` - Average daily volume in shares.
/// * `volatility` - Daily volatility (e.g., 0.02 for 2%).
/// * `spread` - Bid-ask spread as a fraction of mid (e.g., 0.0002 = 2 bps).
pub fn execution_cost(
    order_size: f64,
    daily_volume: f64,
    volatility: f64,
    spread: f64,
) -> ExecutionCost {
    let spread_cost = spread / 2.0; // half-spread paid by aggressor
    let impact_cost = sqrt_impact(volatility, order_size, daily_volume);
    ExecutionCost {
        spread_cost,
        impact_cost,
        total_cost: spread_cost + impact_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_impact_basic() {
        // sigma=0.02, Q=1000, V=1_000_000 -> sqrt(0.001) * 0.02
        let impact = sqrt_impact(0.02, 1000.0, 1_000_000.0);
        let expected = 0.02 * (0.001_f64).sqrt();
        assert!((impact - expected).abs() < 1e-9);
    }

    #[test]
    fn test_sqrt_impact_zero_volume() {
        assert_eq!(sqrt_impact(0.02, 1000.0, 0.0), 0.0);
    }

    #[test]
    fn test_linear_impact() {
        let impact = linear_impact(0.1, 500.0, 10_000.0);
        assert!((impact - 0.005).abs() < 1e-9);
    }

    #[test]
    fn test_execution_cost() {
        let ec = execution_cost(1000.0, 1_000_000.0, 0.02, 0.0002);
        assert!(ec.spread_cost > 0.0);
        assert!(ec.impact_cost > 0.0);
        assert!((ec.total_cost - (ec.spread_cost + ec.impact_cost)).abs() < 1e-12);
    }

    #[test]
    fn test_sqrt_impact_scales_with_size() {
        // Doubling order size increases sqrt-impact by sqrt(2) ~ 1.414
        let small = sqrt_impact(0.02, 500.0, 1_000_000.0);
        let large = sqrt_impact(0.02, 1000.0, 1_000_000.0);
        let ratio = large / small;
        assert!((ratio - 2.0_f64.sqrt()).abs() < 1e-6, "ratio = {ratio}");
    }
}