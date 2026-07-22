//! Integration tests for qf-03-stocks crate.
//!
//! Tests cover OHLCV operations, statistics, moving averages, and the TimeSeries trait.

use approx::assert_relative_eq;
use qf_03_stocks::*;

#[test]
fn test_ohlcv_creation() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 105.0,
        low: 95.0,
        close: 102.0,
        volume: 1000000,
        adj_close: Some(101.5),
    };
    
    assert_eq!(ohlcv.date, "2024-01-01");
    assert_eq!(ohlcv.open, 100.0);
    assert_eq!(ohlcv.high, 105.0);
    assert_eq!(ohlcv.low, 95.0);
    assert_eq!(ohlcv.close, 102.0);
    assert_eq!(ohlcv.volume, 1000000);
    assert_eq!(ohlcv.adj_close, Some(101.5));
}

#[test]
fn test_daily_range() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 105.0,
        low: 95.0,
        close: 102.0,
        volume: 1000,
        adj_close: None,
    };
    
    assert_eq!(ohlcv.daily_range(), 10.0);
}

#[test]
fn test_body_bullish() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 106.0,
        low: 99.0,
        close: 105.0,
        volume: 1000,
        adj_close: None,
    };
    
    assert_eq!(ohlcv.body(), 5.0);
}

#[test]
fn test_body_bearish() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 105.0,
        high: 106.0,
        low: 99.0,
        close: 100.0,
        volume: 1000,
        adj_close: None,
    };
    
    assert_eq!(ohlcv.body(), -5.0);
}

#[test]
fn test_upper_shadow() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 105.0,
        low: 98.0,
        close: 102.0,
        volume: 1000,
        adj_close: None,
    };
    
    // Upper shadow = high - max(open, close) = 105 - 102 = 3
    assert_eq!(ohlcv.upper_shadow(), 3.0);
}

#[test]
fn test_lower_shadow() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 105.0,
        low: 98.0,
        close: 102.0,
        volume: 1000,
        adj_close: None,
    };
    
    // Lower shadow = min(open, close) - low = 100 - 98 = 2
    assert_eq!(ohlcv.lower_shadow(), 2.0);
}

#[test]
fn test_is_bullish_true() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 105.0,
        low: 99.0,
        close: 103.0,
        volume: 1000,
        adj_close: None,
    };
    
    assert!(ohlcv.is_bullish());
}

#[test]
fn test_is_bullish_false() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 101.0,
        low: 95.0,
        close: 97.0,
        volume: 1000,
        adj_close: None,
    };
    
    assert!(!ohlcv.is_bullish());
}

#[test]
fn test_typical_price() {
    let ohlcv = Ohlcv {
        date: "2024-01-01".to_string(),
        open: 100.0,
        high: 105.0,
        low: 95.0,
        close: 102.0,
        volume: 1000,
        adj_close: None,
    };
    
    // (105 + 95 + 102) / 3 = 302 / 3 = 100.666...
    assert_relative_eq!(ohlcv.typical_price(), 100.66666666666667, epsilon = 1e-10);
}

#[test]
fn test_average_volume() {
    let data = vec![
        Ohlcv {
            date: "2024-01-01".to_string(),
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-02".to_string(),
            open: 102.0,
            high: 107.0,
            low: 101.0,
            close: 105.0,
            volume: 2000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-03".to_string(),
            open: 105.0,
            high: 108.0,
            low: 104.0,
            close: 107.0,
            volume: 3000,
            adj_close: None,
        },
    ];
    
    assert_eq!(average_volume(&data), 2000.0);
}

#[test]
fn test_price_change() {
    let data = vec![
        Ohlcv {
            date: "2024-01-01".to_string(),
            open: 98.0,
            high: 105.0,
            low: 95.0,
            close: 100.0,
            volume: 1000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-02".to_string(),
            open: 100.0,
            high: 112.0,
            low: 99.0,
            close: 110.0,
            volume: 2000,
            adj_close: None,
        },
    ];
    
    // (110 - 100) / 100 = 0.10
    assert_eq!(price_change(&data), 0.10);
}

#[test]
fn test_highest_high() {
    let data = vec![
        Ohlcv {
            date: "2024-01-01".to_string(),
            open: 98.0,
            high: 100.0,
            low: 95.0,
            close: 99.0,
            volume: 1000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-02".to_string(),
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 2000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-03".to_string(),
            open: 103.0,
            high: 102.0,
            low: 100.0,
            close: 101.0,
            volume: 1500,
            adj_close: None,
        },
    ];
    
    assert_eq!(highest_high(&data), 105.0);
}

#[test]
fn test_lowest_low() {
    let data = vec![
        Ohlcv {
            date: "2024-01-01".to_string(),
            open: 98.0,
            high: 100.0,
            low: 95.0,
            close: 99.0,
            volume: 1000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-02".to_string(),
            open: 100.0,
            high: 105.0,
            low: 92.0,
            close: 103.0,
            volume: 2000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-03".to_string(),
            open: 103.0,
            high: 106.0,
            low: 98.0,
            close: 104.0,
            volume: 1500,
            adj_close: None,
        },
    ];
    
    assert_eq!(lowest_low(&data), 92.0);
}

#[test]
fn test_sma_basic() {
    let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sma(&prices, 3);
    
    // Window 1: [1,2,3] -> avg = 2.0
    // Window 2: [2,3,4] -> avg = 3.0
    // Window 3: [3,4,5] -> avg = 4.0
    assert_eq!(result, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_sma_single_period() {
    let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sma(&prices, 1);
    
    // Period 1 means each price is its own average
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sma_full_period() {
    let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sma(&prices, 5);
    
    // One window: [1,2,3,4,5] -> avg = 3.0
    assert_eq!(result, vec![3.0]);
}

#[test]
fn test_sma_period_too_large() {
    let prices = vec![1.0, 2.0, 3.0];
    let result = sma(&prices, 5);
    
    // Period > length, should return empty
    assert!(result.is_empty());
}

#[test]
fn test_ema_basic() {
    let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = ema(&prices, 3);
    
    // alpha = 2 / (3 + 1) = 0.5
    // EMA[0] = 1.0
    // EMA[1] = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    // EMA[2] = 0.5 * 3.0 + 0.5 * 1.5 = 2.25
    // EMA[3] = 0.5 * 4.0 + 0.5 * 2.25 = 3.125
    // EMA[4] = 0.5 * 5.0 + 0.5 * 3.125 = 4.0625
    
    assert_eq!(result.len(), 5);
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(result[1], 1.5, epsilon = 1e-10);
    assert_relative_eq!(result[2], 2.25, epsilon = 1e-10);
    assert_relative_eq!(result[3], 3.125, epsilon = 1e-10);
    assert_relative_eq!(result[4], 4.0625, epsilon = 1e-10);
}

#[test]
fn test_ema_smoothing() {
    // EMA should be more responsive to recent price changes than SMA
    let prices = vec![10.0, 10.0, 10.0, 20.0, 20.0];
    
    let sma_result = sma(&prices, 3);
    let ema_result = ema(&prices, 3);
    
    // After the price jump to 20, EMA should react faster
    // SMA at index 2 (4th element): (10+20+20)/3 = 16.67
    // EMA at index 4 (5th element) should be closer to 20 than SMA
    
    assert_eq!(sma_result.len(), 3);
    assert_eq!(ema_result.len(), 5);
    
    // Check that EMA is more responsive
    // alpha = 2/(3+1) = 0.5
    // EMA[0] = 10
    // EMA[1] = 0.5*10 + 0.5*10 = 10
    // EMA[2] = 0.5*10 + 0.5*10 = 10
    // EMA[3] = 0.5*20 + 0.5*10 = 15
    // EMA[4] = 0.5*20 + 0.5*15 = 17.5
    
    assert_relative_eq!(ema_result[4], 17.5, epsilon = 1e-10);
    assert_relative_eq!(sma_result[2], 16.666666666666668, epsilon = 1e-10);
    
    // EMA should be closer to the recent price (20) than SMA
    assert!(ema_result[4] > sma_result[2]);
}
