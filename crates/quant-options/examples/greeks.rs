//! Black-Scholes Greeks demo.
//!
//! Prints all five Greeks for an ATM call and put, and validates the
//! analytical formulas against central finite differences. The two should
//! match to ~1e-4 with `h = 1e-4` for spot bumps and `h = 1e-3` for vol/time
//! bumps.
//!
//! Run: cargo run -p quant-options --example greeks

use quant_options::{
    bs_call, bs_put, delta, delta_fd, gamma, gamma_fd, normal_pdf, rho, theta, theta_fd, vega,
    vega_fd,
};

fn main() {
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;

    let call = bs_call(s0, k, r, sigma, t);
    let put = bs_put(s0, k, r, sigma, t);

    println!("==============================");
    println!("Black-Scholes Greeks");
    println!("==============================");
    println!();
    println!("Parameters: S0={s0}, K={k}, r={r}, sigma={sigma}, T={t}");
    println!("Call price: {call:.6}");
    println!("Put  price: {put:.6}");
    println!("phi(0)    = {:.10}  (1/sqrt(2 pi))", normal_pdf(0.0));
    println!();

    let h_spot = 1e-4;
    let h_vol = 1e-4;
    let h_t = 1e-4;

    println!("{:<8} {:>12} {:>12} {:>12}", "Greek", "Analytical", "FiniteDiff", "|diff|");
    println!("{}", "-".repeat(52));

    // Delta (call and put)
    for &(is_call, label) in &[(true, "Delta_c"), (false, "Delta_p")] {
        let a = delta(s0, k, r, sigma, t, is_call);
        let f = delta_fd(s0, k, r, sigma, t, is_call, h_spot);
        println!("{:<8} {:>12.6} {:>12.6} {:>12.2e}", label, a, f, (a - f).abs());
    }

    // Gamma (same for call/put)
    let g_a = gamma(s0, k, r, sigma, t);
    let g_f = gamma_fd(s0, k, r, sigma, t, 1e-3);
    println!("{:<8} {:>12.6} {:>12.6} {:>12.2e}", "Gamma", g_a, g_f, (g_a - g_f).abs());

    // Vega (same for call/put)
    let v_a = vega(s0, k, r, sigma, t);
    let v_f = vega_fd(s0, k, r, sigma, t, h_vol);
    println!("{:<8} {:>12.6} {:>12.6} {:>12.2e}", "Vega", v_a, v_f, (v_a - v_f).abs());

    // Theta (call and put). Forward difference in t.
    for &(is_call, label) in &[(true, "Theta_c"), (false, "Theta_p")] {
        let a = theta(s0, k, r, sigma, t, is_call);
        let f = theta_fd(s0, k, r, sigma, t, is_call, h_t);
        println!("{:<8} {:>12.6} {:>12.6} {:>12.2e}", label, a, f, (a - f).abs());
    }

    // Rho (no finite-difference version in this crate; analytical only).
    for &(is_call, label) in &[(true, "Rho_c"), (false, "Rho_p")] {
        let a = rho(s0, k, r, sigma, t, is_call);
        println!("{:<8} {:>12.6} {:>12} {:>12}", label, a, "-", "-");
    }

    println!();
    println!("Gamma is identical for calls and puts (d1 is).");
    println!("Vega  is identical for calls and puts (d1 is).");
    println!("Theta_p = Theta_c + r * K * exp(-rT) = {:.6}", r * k * (-r * t).exp());
    println!();
    println!("==============================");
}