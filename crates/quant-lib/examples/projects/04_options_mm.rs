//! Project 4: Options Market Making with Delta Hedging
//!
//! Level: Expert
//!
//! Simulates a market-maker who sells an at-the-money European call on
//! a geometric-Brownian-motion price path and delta-hedges it daily.
//! First an implied-volatility surface is built by pricing calls on a
//! grid of strikes and maturities at a "market" vol smile and inverting
//! with the implied-vol solver (a round-trip sanity check). Then the
//! short-ATM-call position is hedged by holding `-delta` shares,
//! rebalanced each day. The final option premium, cumulative hedge
//! P&L, total P&L (should be approximately zero), and maximum hedging
//! error are reported.
//!
//! Run: `cargo run -p quant-lib --example projects-04_options_mm`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::prelude::*;

const STEPS: usize = 252;
const DAYS_PER_YEAR: f64 = 252.0;
const R: f64 = 0.05;

fn main() {
    println!("=== Project 4: Options Market Making (Delta Hedge) ===\n");

    // --- Step 1: Simulate GBM path. ---
    let s0 = 100.0;
    let mu = 0.07;
    let sigma = 0.20;
    let t = 1.0;
    let mut rng = XorShift64::new(987);
    let prices = gbm(s0, mu, sigma, t, STEPS, &mut rng);
    let s_final = prices[STEPS];
    println!("1. GBM: S0={s0}, mu={mu}, sigma={sigma}, T={t}, S_T={s_final:.4}");

    // --- Step 2: IV surface round-trip. ---
    let strikes = [90.0_f64, 95.0, 100.0, 105.0, 110.0];
    let maturities_days = [30_usize, 60, 90, 180, 252];
    println!("\n2. IV surface (market vol smile):");
    println!(
        "   {:<10} {:>10} {:>10} {:>10}",
        "Strike", "T(days)", "MktVol", "RecIV"
    );
    let mut max_iv_err = 0.0_f64;
    for &k in &strikes {
        for &td in &maturities_days {
            let t_years = td as f64 / DAYS_PER_YEAR;
            let mkt_vol = 0.20 + 0.001 * (k - s0).abs();
            let price = bs_call(s0, k, R, mkt_vol, t_years);
            let iv = implied_vol(price, s0, k, R, t_years, true).expect("IV solve");
            let err = (iv - mkt_vol).abs();
            max_iv_err = max_iv_err.max(err);
            println!("   {:<10.0} {:>10} {:>10.4} {:>10.4}", k, td, mkt_vol, iv);
        }
    }
    println!("   max |IV - MktVol| = {max_iv_err:.2e}");

    // --- Step 3: Delta-hedge a short ATM call, daily rebalancing. ---
    let k_atm = s0;
    let t_total = STEPS as f64 / DAYS_PER_YEAR;
    let premium = bs_call(s0, k_atm, R, sigma, t_total);
    println!("\n3. Delta hedge short ATM call: K={k_atm}, T={t_total:.4}, premium={premium:.4}");

    // A short call has negative delta; the hedge is to HOLD +delta shares
    // (long the underlying) so the combined position is delta-neutral.
    // `hedge_cost` accumulates the cash spent building the long share
    // position (positive = cash outflow to buy shares, negative = proceeds
    // from selling shares when delta falls).
    let mut hedge_cost = 0.0_f64;
    let mut prev_delta = 0.0_f64;
    let mut max_err = 0.0_f64;

    for (day, &s) in prices.iter().enumerate() {
        let t_rem = (STEPS - day) as f64 / DAYS_PER_YEAR;
        let d = if t_rem > 1e-9 {
            delta(s, k_atm, R, sigma, t_rem, true)
        } else {
            // At expiry delta is the indicator of ITM.
            if s > k_atm { 1.0 } else { 0.0 }
        };
        // Cost of rebalancing to `d` shares (day 0 buys the initial position).
        let cost = (d - prev_delta) * s;
        hedge_cost += cost;
        // Mark-to-market hedging error: premium received minus current
        // option value plus the value of the share position minus cash spent.
        let opt_val = if t_rem > 1e-9 {
            bs_call(s, k_atm, R, sigma, t_rem)
        } else {
            (s - k_atm).max(0.0)
        };
        let share_value = d * s;
        let running_pnl = premium - opt_val + share_value - hedge_cost;
        max_err = max_err.max(running_pnl.abs());
        prev_delta = d;
    }

    // At expiry we hold `prev_delta` shares (0 if OTM, 1 if ITM); liquidate
    // at S_T. The short call owes `max(S_T - K, 0)` to the buyer.
    let liquidation = prev_delta * s_final;
    let option_payoff = (s_final - k_atm).max(0.0);
    let total_pnl = premium + liquidation - hedge_cost - option_payoff;

    println!("\nHedging results:");
    println!("  Option premium:          {premium:.4}");
    println!("  Cumulative hedge cost:   {hedge_cost:.4}");
    let liq = liquidation;
    println!("  Share liquidation value: {liq:.4}");
    let payoff = option_payoff;
    println!("  Option payoff (short):   {payoff:.4}");
    println!("  Total P&L:               {total_pnl:.4} (should be ~0)");
    println!("  Max hedging error:       {max_err:.4}");
}
