//! Example: Stock price analysis with moving averages.
//!
//! Demonstrates loading OHLCV data, computing statistics,
//! calculating moving averages, and identifying crossovers.
//!
//! Run from quant-lab directory:
//!   cargo run -p qf-03-stocks --example stock_analysis

use qf_03_stocks::*;
use std::path::PathBuf;

fn main() {
    println!("Stock Price Analysis");
    println!("====================\n");

    // Sample dataset bundled with the repository
    let data_path = PathBuf::from("data/stock_prices.csv");

    match load_ohlcv(&data_path) {
        Ok(data) => {
            println!("Loaded {} trading days", data.len());

            // Compute basic statistics
            println!("\n=== Basic Statistics ===");
            println!("Average Volume: {:.0}", average_volume(&data));
            println!("Price Change: {:.2}%", price_change(&data) * 100.0);
            println!("Highest High: {:.2}", highest_high(&data));
            println!("Lowest Low: {:.2}", lowest_low(&data));
            println!("Average Daily Range: {:.2}", average_daily_range(&data));

            // Use TimeSeries trait
            let closes = data.closes();

            // Calculate moving averages
            let sma_20 = sma(&closes, 20);
            let sma_50 = sma(&closes, 50);
            let ema_20 = ema(&closes, 20);

            println!("\n=== Moving Averages ===");
            if let Some(&last_sma20) = sma_20.last() {
                println!("SMA(20): {:.2}", last_sma20);
            }
            if let Some(&last_sma50) = sma_50.last() {
                println!("SMA(50): {:.2}", last_sma50);
            }
            if let Some(&last_ema20) = ema_20.last() {
                println!("EMA(20): {:.2}", last_ema20);
            }

            // Identify crossovers
            println!("\n=== Crossovers (SMA20 vs SMA50) ===");
            let mut crossovers = 0;

            if sma_20.len() > 1 && sma_50.len() > 1 {
                let min_len = sma_20.len().min(sma_50.len());

                for i in 1..min_len {
                    let prev_20 = sma_20[sma_20.len() - min_len + i - 1];
                    let curr_20 = sma_20[sma_20.len() - min_len + i];
                    let prev_50 = sma_50[sma_50.len() - min_len + i - 1];
                    let curr_50 = sma_50[sma_50.len() - min_len + i];

                    if prev_20 <= prev_50 && curr_20 > curr_50 {
                        crossovers += 1;
                        println!("  Golden Cross at day {}", closes.len() - min_len + i);
                    }
                    if prev_20 >= prev_50 && curr_20 < curr_50 {
                        crossovers += 1;
                        println!("  Death Cross at day {}", closes.len() - min_len + i);
                    }
                }
            }

            if crossovers == 0 {
                println!("  No crossovers detected in sample period");
            }
            println!("Total Crossovers: {}", crossovers);

            // Show first few candles
            println!("\n=== Sample Candlesticks ===");
            for (i, candle) in data.iter().take(5).enumerate() {
                println!(
                    "Day {}: O={:.2} H={:.2} L={:.2} C={:.2} | Body={:.2} | {}",
                    i + 1,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.body(),
                    if candle.is_bullish() { "Bullish" } else { "Bearish" }
                );
            }
        }
        Err(e) => {
            eprintln!("Error loading stock data: {}", e);
            eprintln!("\nMake sure you run from the quant-lab directory:");
            eprintln!("  cd crates/quant-lab");
            eprintln!("  cargo run -p qf-03-stocks --example stock_analysis");
            std::process::exit(1);
        }
    }
}
