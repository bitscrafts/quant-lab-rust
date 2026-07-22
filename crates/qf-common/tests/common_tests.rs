use approx::assert_relative_eq;
use qf_common::{compute_stats, load_transactions, CommonError};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn create_test_csv(filename: &str, content: &str) -> PathBuf {
    let path = PathBuf::from(format!("../../../../_tmp/{}", filename));
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(content.as_bytes()).unwrap();
    path
}

#[test]
fn test_load_transactions_valid() {
    let content = "Time,V1,V2,Amount,Class\n\
                   0,1.5,2.5,100.0,0\n\
                   1,3.0,4.0,200.0,1\n\
                   2,5.5,6.5,300.0,0\n";
    
    let path = create_test_csv("valid_transactions.csv", content);
    
    let result = load_transactions(&path);
    assert!(result.is_ok());
    
    let transactions = result.unwrap();
    assert_eq!(transactions.len(), 3);
    
    assert_eq!(transactions[0].features, vec![1.5, 2.5]);
    assert_relative_eq!(transactions[0].amount, 100.0);
    assert_eq!(transactions[0].class, 0);
    
    assert_eq!(transactions[1].features, vec![3.0, 4.0]);
    assert_relative_eq!(transactions[1].amount, 200.0);
    assert_eq!(transactions[1].class, 1);
    
    fs::remove_file(path).ok();
}

#[test]
fn test_load_transactions_missing_file() {
    let result = load_transactions("_tmp/nonexistent_file.csv");
    assert!(result.is_err());
    
    match result.unwrap_err() {
        CommonError::FileNotFound(_) => {},
        other => panic!("Expected FileNotFound, got {:?}", other),
    }
}

#[test]
fn test_load_transactions_malformed() {
    let content = "Time,V1,V2,Amount,Class\n\
                   0,not_a_number,2.5,100.0,0\n";
    
    let path = create_test_csv("malformed_transactions.csv", content);
    
    let result = load_transactions(&path);
    assert!(result.is_err());
    
    match result.unwrap_err() {
        CommonError::ParseError(_) => {},
        other => panic!("Expected ParseError, got {:?}", other),
    }
    
    fs::remove_file(path).ok();
}

#[test]
fn test_compute_stats_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = compute_stats(&data);
    
    assert_relative_eq!(stats.mean, 3.0, epsilon = 1e-12);
    assert_relative_eq!(stats.std, 1.5811388300841898, epsilon = 1e-12);
    assert_relative_eq!(stats.min, 1.0, epsilon = 1e-12);
    assert_relative_eq!(stats.max, 5.0, epsilon = 1e-12);
}

#[test]
fn test_compute_stats_single() {
    let data = vec![42.0];
    let stats = compute_stats(&data);
    
    assert_relative_eq!(stats.mean, 42.0, epsilon = 1e-12);
    assert_relative_eq!(stats.std, 0.0, epsilon = 1e-12);
    assert_relative_eq!(stats.min, 42.0, epsilon = 1e-12);
    assert_relative_eq!(stats.max, 42.0, epsilon = 1e-12);
}

#[test]
fn test_compute_stats_empty() {
    let data: Vec<f64> = vec![];
    let stats = compute_stats(&data);
    
    assert!(stats.mean.is_nan());
    assert_relative_eq!(stats.std, 0.0, epsilon = 1e-12);
    assert!(stats.min.is_nan());
    assert!(stats.max.is_nan());
}
