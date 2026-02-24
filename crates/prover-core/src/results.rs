//! Result types for evaluation and iteration tracking.

use serde::{Deserialize, Serialize};

/// Results from evaluating a model on a theorem set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationResult {
    /// Expert iteration number (None for baseline evaluation).
    pub iteration: Option<u32>,
    /// ISO 8601 timestamp of when the evaluation was run.
    pub timestamp: String,
    /// Path to the LLM model used.
    pub llm_path: String,
    /// Path to the EBM checkpoint (if any).
    pub ebm_path: Option<String>,
    /// Name/path of the benchmark theorem set.
    pub benchmark: String,
    /// Total number of theorems in the benchmark.
    pub total_theorems: u32,
    /// Maximum search nodes per theorem.
    pub max_nodes: u32,
    /// Number of theorems proved.
    pub solved: u32,
    /// Total theorems attempted.
    pub total: u32,
    /// Fraction solved (solved / total).
    pub rate: f64,
    /// Average nodes expanded per theorem.
    pub avg_nodes: f64,
    /// Average wall-clock time per theorem in seconds.
    pub avg_time_secs: f64,
    /// Median wall-clock time per theorem in seconds.
    pub median_time_secs: f64,
    /// Per-theorem results.
    pub per_theorem: Vec<TheoremResult>,
}

/// Result for a single theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoremResult {
    /// Theorem name.
    pub name: String,
    /// Whether the theorem was proved.
    pub proved: bool,
    /// Number of search nodes expanded.
    pub nodes_used: u32,
    /// Wall-clock time in seconds.
    pub time_secs: f64,
    /// Why search ended: "proved", "budget_exhausted", "frontier_exhausted",
    /// "timeout", or "error".
    #[serde(default)]
    pub failure_reason: String,
}

/// Compute the median of a slice of f64 values.
///
/// Returns 0.0 for empty slices.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Compute p50, p95, p99 percentiles from a mutable slice of microsecond latencies.
///
/// Returns `(p50, p95, p99)`. Returns `(0, 0, 0)` for empty input.
pub fn percentiles(data: &mut [u64]) -> (u64, u64, u64) {
    if data.is_empty() {
        return (0, 0, 0);
    }
    data.sort_unstable();
    let n = data.len();
    let p50 = data[n / 2];
    let p95 = data[((n as f64 * 0.95) as usize).min(n - 1)];
    let p99 = data[((n as f64 * 0.99) as usize).min(n - 1)];
    (p50, p95, p99)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iteration_result_serde_roundtrip() {
        let result = IterationResult {
            iteration: Some(1),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            llm_path: "models/test".to_string(),
            ebm_path: Some("checkpoints/ebm".to_string()),
            benchmark: "data/test.json".to_string(),
            total_theorems: 10,
            max_nodes: 100,
            solved: 5,
            total: 10,
            rate: 0.5,
            avg_nodes: 42.0,
            avg_time_secs: 3.5,
            median_time_secs: 3.0,
            per_theorem: vec![TheoremResult {
                name: "thm1".to_string(),
                proved: true,
                nodes_used: 42,
                time_secs: 3.5,
                failure_reason: String::new(),
            }],
        };

        let json = serde_json::to_string_pretty(&result).unwrap();
        let loaded: IterationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.iteration, Some(1));
        assert_eq!(loaded.total_theorems, 10);
        assert_eq!(loaded.max_nodes, 100);
        assert_eq!(loaded.per_theorem[0].name, "thm1");
    }

    #[test]
    fn test_median_helper() {
        assert!((median(&mut []) - 0.0).abs() < 1e-9);
        assert!((median(&mut [5.0]) - 5.0).abs() < 1e-9);
        assert!((median(&mut [1.0, 3.0]) - 2.0).abs() < 1e-9);
        assert!((median(&mut [3.0, 1.0, 2.0]) - 2.0).abs() < 1e-9);
        assert!((median(&mut [4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_percentiles_empty() {
        assert_eq!(percentiles(&mut []), (0, 0, 0));
    }

    #[test]
    fn test_percentiles_single() {
        assert_eq!(percentiles(&mut [42]), (42, 42, 42));
    }

    #[test]
    fn test_percentiles_known_data() {
        // 100 values: 1..=100
        let mut data: Vec<u64> = (1..=100).collect();
        let (p50, p95, p99) = percentiles(&mut data);
        assert_eq!(p50, 51);  // index 50
        assert_eq!(p95, 96);  // index 95
        assert_eq!(p99, 100); // index 99
    }

    #[test]
    fn test_percentiles_unsorted_input() {
        let mut data = vec![100, 1, 50, 99, 2, 51, 95, 3];
        let (p50, p95, p99) = percentiles(&mut data);
        // sorted: [1, 2, 3, 50, 51, 95, 99, 100], n=8
        // p50 = index 4 = 51
        // p95 = index min(7, 7) = 100
        // p99 = index min(7, 7) = 100
        assert_eq!(p50, 51);
        assert_eq!(p95, 100);
        assert_eq!(p99, 100);
    }
}
