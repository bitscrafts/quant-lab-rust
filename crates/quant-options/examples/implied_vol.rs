//! Implied volatility demo.
//!
//! Recovers sigma from a BS call price, then demonstrates the volatility
//! smile by inverting a synthetic quote table (strikes shifted away from
//! ATM with a smile premium). Prints the smile as a table and a coarse text
//! plot.
//!
//! Run: cargo run -p quant-options --example implied_vol

use quant_options::{bs_call, bs_put, implied_vol};

fn main() {
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let t = 1.0;
    let true_sigma = 0.2;

    println!("==============================");
    println!("Implied Volatility");
    println!("==============================");
    println!();
    println!("Parameters: S0={s0}, K={k}, r={r}, T={t}, true sigma={true_sigma}");
    println!();

    // 1. Recovery: price a call at true_sigma, recover sigma from the price.
    let price = bs_call(s0, k, r, true_sigma, t);
    let iv = implied_vol(price, s0, k, r, t, true).expect("IV should converge");
    println!("Recovery (ATM call):");
    println!("  Market price  = {price:.6}");
    println!("  Implied vol   = {iv:.8}");
    println!("  True vol      = {true_sigma:.8}");
    println!("  |iv - true|   = {:.2e}", (iv - true_sigma).abs());
    println!();

    // 2. Put price gives the same IV (put-call parity).
    let put_price = bs_put(s0, k, r, true_sigma, t);
    let iv_put = implied_vol(put_price, s0, k, r, t, false).expect("put IV should converge");
    println!("Put-call parity IV:");
    println!("  IV from call = {iv:.8}");
    println!("  IV from put  = {iv_put:.8}");
    println!("  |iv_c - iv_p| = {:.2e}", (iv - iv_put).abs());
    println!();

    // 3. Volatility smile. Generate synthetic call prices by adding a smile
    // premium that is quadratic in log-moneyness:
    //   sigma(K) = true_sigma + a * (log(K/S0))^2
    // so ATM vol equals true_sigma and wings have higher vol (the classic
    // equity smile).
    println!("Volatility smile (synthetic quotes):");
    println!("  Model: sigma(K) = true_sigma + a * (ln(K/S0))^2, a = 0.5");
    let a_smile = 0.5;
    println!(
        "  {:>6}  {:>10}  {:>12}  {:>10}",
        "K", "moneyness", "market price", "IV"
    );
    println!("  {}", "-".repeat(46));

    let mut smile_rows: Vec<(f64, f64, f64)> = Vec::new();
    for &k_i in &[70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0] {
        let log_m = (k_i / s0).ln();
        let smile_sigma = true_sigma + a_smile * log_m * log_m;
        let p = bs_call(s0, k_i, r, smile_sigma, t);
        let iv_i = implied_vol(p, s0, k_i, r, t, true).unwrap_or(f64::NAN);
        println!(
            "  {:>6.1}  {:>10.4}  {:>12.6}  {:>10.6}",
            k_i,
            k_i / s0,
            p,
            iv_i
        );
        smile_rows.push((k_i, smile_sigma, iv_i));
    }
    println!();

    // 4. Coarse text plot of the smile.
    println!("Smile plot (IV vs strike, * = market-implied, + = model):");
    let iv_min = smile_rows
        .iter()
        .map(|(_, _, iv)| iv)
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::INFINITY, f64::min);
    let iv_max = smile_rows
        .iter()
        .map(|(_, _, iv)| iv)
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let width: usize = 50;
    let span = (iv_max - iv_min).max(1e-12);
    for (k_i, model, iv_i) in &smile_rows {
        if !iv_i.is_finite() {
            continue;
        }
        let pos_iv = ((iv_i - iv_min) / span * width as f64) as usize;
        let pos_model = ((model - iv_min) / span * width as f64) as usize;
        let mut line: Vec<char> = vec![' '; width + 1];
        if pos_model < line.len() {
            line[pos_model] = '+';
        }
        if pos_iv < line.len() {
            line[pos_iv] = '*';
        }
        println!("  K={:>5.1} |{}|", k_i, line.iter().collect::<String>());
    }
    println!();
    println!("==============================");
}
