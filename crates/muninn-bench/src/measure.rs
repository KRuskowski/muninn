//! Accuracy and latency figures, computed the same way for every engine.
//!
//! The Python runners (`scripts/run_ann.py`) implement the same
//! definitions; keep the two in step.

use serde::Serialize;

use crate::dataset::Matrix;

/// Latency distribution of one query pass, in microseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Latency {
  /// Arithmetic mean.
  pub mean_us: f64,
  /// Median.
  pub p50_us: f64,
  /// 95th percentile.
  pub p95_us: f64,
  /// 99th percentile.
  pub p99_us: f64,
  /// Slowest query.
  pub max_us: f64,
}

/// Nearest-rank percentile `p` (0..=100) of ascending `sorted` values:
/// the smallest value with at least `p`% of values at or below it.
/// Zero for an empty slice.
pub fn percentile(sorted: &[u64], p: f64) -> u64 {
  if sorted.is_empty() {
    return 0;
  }
  let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
  sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Summarises per-query latencies given in nanoseconds (sorted in place).
pub fn summarize(lat_ns: &mut [u64]) -> Latency {
  if lat_ns.is_empty() {
    return Latency::default();
  }
  lat_ns.sort_unstable();
  let us = |ns: u64| ns as f64 / 1e3;
  let sum: u128 = lat_ns.iter().map(|&v| u128::from(v)).sum();
  Latency {
    mean_us: sum as f64 / lat_ns.len() as f64 / 1e3,
    p50_us: us(percentile(lat_ns, 50.0)),
    p95_us: us(percentile(lat_ns, 95.0)),
    p99_us: us(percentile(lat_ns, 99.0)),
    max_us: us(*lat_ns.last().unwrap()),
  }
}

/// Mean recall@k over queries: the fraction of each query's first `k`
/// ground-truth ids found among its first `k` results. Id-based (a
/// result at exactly the same distance as a true neighbour but with a
/// different id counts as a miss), which is what hnswlib and faiss
/// report and slightly stricter than ann-benchmarks' distance-based
/// recall.
pub fn recall_at_k(results: &[Vec<u32>], gt: &Matrix<u32>, k: usize) -> f64 {
  assert!(k <= gt.cols, "ground truth has only {} neighbours", gt.cols);
  assert!(results.len() <= gt.rows, "more results than queries");
  if results.is_empty() || k == 0 {
    return 0.0;
  }
  let mut found = 0usize;
  for (q, got) in results.iter().enumerate() {
    let truth = &gt.row(q)[..k];
    let got = &got[..got.len().min(k)];
    found += got.iter().filter(|id| truth.contains(id)).count();
  }
  found as f64 / (results.len() * k) as f64
}

/// Overlap of two result lists: shared ids divided by the longer list's
/// length (1 when both are empty).
pub fn overlap(a: &[u32], b: &[u32]) -> f64 {
  let n = a.len().max(b.len());
  if n == 0 {
    return 1.0;
  }
  a.iter().filter(|id| b.contains(id)).count() as f64 / n as f64
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn nearest_rank_percentiles() {
    let v: Vec<u64> = (1..=100).collect();
    assert_eq!(percentile(&v, 50.0), 50);
    assert_eq!(percentile(&v, 95.0), 95);
    assert_eq!(percentile(&v, 99.0), 99);
    assert_eq!(percentile(&v, 100.0), 100);
    assert_eq!(percentile(&v, 0.0), 1);
    assert_eq!(percentile(&[7], 99.0), 7);
    assert_eq!(percentile(&[], 50.0), 0);
  }

  #[test]
  fn summary_in_microseconds() {
    let mut ns = vec![3000, 1000, 2000, 4000];
    let s = summarize(&mut ns);
    assert_eq!(s.p50_us, 2.0);
    assert_eq!(s.max_us, 4.0);
    assert_eq!(s.mean_us, 2.5);
    assert_eq!(summarize(&mut []), Latency::default());
  }

  #[test]
  fn recall_counts_ids_in_the_first_k() {
    let gt = Matrix {
      rows: 2,
      cols: 3,
      data: vec![1, 2, 3, 4, 5, 6],
    };
    // Query 0 finds 1 and 2 of {1, 2}; query 1 finds 5 of {4, 5}.
    let res = vec![vec![2, 1, 9], vec![5, 9, 4]];
    assert_eq!(recall_at_k(&res, &gt, 2), 0.75);
    assert_eq!(recall_at_k(&res, &gt, 3), 4.0 / 6.0);
    // Short result lists count as misses.
    assert_eq!(recall_at_k(&[vec![1]], &gt, 2), 0.5);
  }

  #[test]
  fn overlap_of_result_lists() {
    assert_eq!(overlap(&[1, 2, 3, 4], &[4, 3, 9, 8]), 0.5);
    assert_eq!(overlap(&[1, 2], &[1, 2, 3, 4]), 0.5);
    assert_eq!(overlap(&[], &[]), 1.0);
  }
}
