//! ROC-AUC evaluation metrics.

/// ROC curve result with AUC and optimal threshold.
#[derive(Debug, Clone)]
pub struct RocResult {
    pub auc: f64,
    pub optimal_threshold: f64,
    pub points: Vec<(f64, f64)>,  // (FPR, TPR) pairs
}

/// Compute ROC curve: (FPR, TPR) pairs at various thresholds.
///
/// # Arguments
/// * `scores` - Predicted scores (higher = more likely positive)
/// * `labels` - True labels (true = positive, false = negative)
///
/// # Returns
/// Vector of (FPR, TPR) pairs sorted by increasing FPR.
pub fn roc_curve(scores: &[f64], labels: &[bool]) -> Vec<(f64, f64)> {
    if scores.len() != labels.len() {
        return vec![];
    }
    
    if scores.is_empty() {
        return vec![];
    }
    
    // Combine scores and labels, then sort by score descending
    let mut pairs: Vec<(f64, bool)> = scores.iter()
        .copied()
        .zip(labels.iter().copied())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    // Count total positives and negatives
    let total_pos = labels.iter().filter(|&&l| l).count();
    let total_neg = labels.len() - total_pos;
    
    if total_pos == 0 || total_neg == 0 {
        // Degenerate case: all one class
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }
    
    let mut roc_points = Vec::new();
    let mut tp = 0;
    let mut fp = 0;
    
    // Start at (0, 0): threshold = infinity, predict all negative
    roc_points.push((0.0, 0.0));
    
    // Walk through sorted predictions
    for (_score, label) in pairs {
        if label {
            tp += 1;
        } else {
            fp += 1;
        }
        
        let fpr = fp as f64 / total_neg as f64;
        let tpr = tp as f64 / total_pos as f64;
        roc_points.push((fpr, tpr));
    }
    
    roc_points
}

/// Compute AUC via trapezoidal integration.
///
/// # Arguments
/// * `roc_points` - (FPR, TPR) pairs from `roc_curve`
///
/// # Returns
/// Area under curve (0.0 to 1.0).
pub fn auc(roc_points: &[(f64, f64)]) -> f64 {
    if roc_points.len() < 2 {
        return 0.0;
    }
    
    let mut area = 0.0;
    for i in 1..roc_points.len() {
        let (x0, y0) = roc_points[i - 1];
        let (x1, y1) = roc_points[i];
        
        // Trapezoidal rule: width × average height
        let width = x1 - x0;
        let avg_height = (y0 + y1) / 2.0;
        area += width * avg_height;
    }
    
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_auc_perfect_classifier() {
        // Perfect: all positives scored higher than all negatives
        let points = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let a = auc(&points);
        assert!((a - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_auc_random_classifier() {
        // Random: diagonal line
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let a = auc(&points);
        assert!((a - 0.5).abs() < 1e-10);
    }
}
