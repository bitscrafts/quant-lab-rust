//! Sample weights from event uniqueness (AFML Ch. 4).
//!
//! When events overlap in time, each sample carries less unique
//! information. The uniqueness framework weights each event by the
//! inverse of the average number of concurrent events during its
//! lifetime:
//!   - `concurrent_events[t]` = number of events active at bar `t`.
//!   - `average_uniqueness[i]` = mean over event `i`'s lifetime of
//!     `1 / concurrent_events[t]`.
//!   - `sample_weights[i]` = `1 / avg_concurrent_i` (a simpler form
//!     equivalent to uniqueness up to scale).

use crate::triple_barrier::LabeledEvent;

/// Compute the number of concurrent (active) events at each bar.
/// Returns a vector of length `n_bars` with the count of events whose
/// `[entry_index, exit_index]` interval contains bar `t`.
pub fn concurrent_events(events: &[LabeledEvent], n_bars: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_bars];
    for ev in events {
        let start = ev.entry_index.min(n_bars);
        let end = ev.exit_index.min(n_bars.saturating_sub(1));
        if start >= n_bars {
            continue;
        }
        for c in counts.iter_mut().take(end + 1).skip(start) {
            *c += 1;
        }
    }
    counts
}

/// Compute the average uniqueness of each event.
/// Uniqueness_i = mean over event duration of `1 / concurrent[t]`.
/// Returns a vector of length `events.len()`, each value in `(0, 1]`.
pub fn average_uniqueness(events: &[LabeledEvent], n_bars: usize) -> Vec<f64> {
    let concurrent = concurrent_events(events, n_bars);
    events
        .iter()
        .map(|ev| {
            let start = ev.entry_index;
            let end = ev.exit_index.min(n_bars.saturating_sub(1));
            if end < start {
                return 0.0;
            }
            let mut sum_inv = 0.0;
            let mut count = 0;
            for &c in concurrent.iter().take(end + 1).skip(start) {
                if c > 0 {
                    sum_inv += 1.0 / c as f64;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                sum_inv / count as f64
            }
        })
        .collect()
}

/// Compute sample weights as the inverse of the average number of
/// concurrent events during each event. Weights are then normalised
/// to sum to 1 across the sample.
pub fn sample_weights(events: &[LabeledEvent], n_bars: usize) -> Vec<f64> {
    let concurrent = concurrent_events(events, n_bars);
    let mut weights: Vec<f64> = events
        .iter()
        .map(|ev| {
            let start = ev.entry_index;
            let end = ev.exit_index.min(n_bars.saturating_sub(1));
            if end < start {
                return 0.0;
            }
            let mut sum_c = 0.0;
            let mut count = 0;
            for &c in concurrent.iter().take(end + 1).skip(start) {
                if c > 0 {
                    sum_c += c as f64;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                1.0 / (sum_c / count as f64)
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    if total > 0.0 {
        for w in &mut weights {
            *w /= total;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triple_barrier::{LabeledEvent, TripleBarrierLabel};

    fn ev(entry: usize, exit: usize) -> LabeledEvent {
        LabeledEvent {
            entry_index: entry,
            exit_index: exit,
            label: TripleBarrierLabel::Upper,
            return_pct: 0.02,
            holding_period: exit - entry,
        }
    }

    #[test]
    fn test_concurrent_events_overlap() {
        let events = vec![ev(0, 5), ev(3, 8), ev(7, 10)];
        let counts = concurrent_events(&events, 12);
        // Bars 3-5 should have count 2; bars 7-8 should have count 2; rest 1 or 0.
        assert_eq!(counts[0], 1);
        assert_eq!(counts[3], 2);
        assert_eq!(counts[7], 2);
        assert_eq!(counts[10], 1);
        assert_eq!(counts[11], 0);
    }

    #[test]
    fn test_average_uniqueness_bounds() {
        let events = vec![ev(0, 5), ev(3, 8), ev(7, 10)];
        let u = average_uniqueness(&events, 12);
        assert_eq!(u.len(), 3);
        for &v in &u {
            assert!(v > 0.0 && v <= 1.0, "uniqueness out of bounds: {v}");
        }
        // An event with no overlaps has uniqueness 1.0.
        let single = vec![ev(100, 105)];
        let u2 = average_uniqueness(&single, 200);
        assert!((u2[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_weights_normalized() {
        let events = vec![ev(0, 5), ev(3, 8), ev(7, 10)];
        let w = sample_weights(&events, 12);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "weights must sum to 1, got {sum}");
        for &wi in &w {
            assert!(wi > 0.0 && wi <= 1.0);
        }
    }
}
