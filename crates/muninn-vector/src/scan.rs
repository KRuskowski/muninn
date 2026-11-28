//! Exact scans shared by every index: blocked, filter-aware, parallel.

use std::ops::Range;

use muninn_core::{DocFilter, DocId, Metric, TopK};
use rayon::prelude::*;

/// Rows scored per `distance_batch` call. Small enough that the output
/// buffer lives on the stack, large enough to amortise the call.
pub(crate) const BLOCK: usize = 256;

/// Scans with fewer `rows * dim` floats than this stay on one thread:
/// below it, rayon's fork/join overhead outweighs the work.
const PAR_MIN_FLOATS: usize = 1 << 19;

/// Rows per parallel task (before rounding to whole blocks).
const PAR_MIN_ROWS: usize = 4 * BLOCK;

/// Splits `0..n` into chunks and folds each with `scan`, in parallel when
/// `work` (an estimate in floats touched) is large enough, then merges the
/// per-chunk collectors. `TopK` breaks ties by id, so the result does not
/// depend on the split or the thread count.
pub(crate) fn par_scan<F>(n: usize, k: usize, work: usize, scan: F) -> TopK
where
  F: Fn(Range<usize>, &mut TopK) + Sync,
{
  if k == 0 || n == 0 {
    return TopK::new(k);
  }
  let threads = rayon::current_num_threads();
  if work < PAR_MIN_FLOATS || threads <= 1 {
    let mut top = TopK::new(k);
    scan(0..n, &mut top);
    return top;
  }
  // A few chunks per thread so stealing evens out uneven filters.
  let chunk = n
    .div_ceil(threads * 4)
    .max(PAR_MIN_ROWS)
    .next_multiple_of(BLOCK);
  let chunks = n.div_ceil(chunk);
  (0..chunks)
    .into_par_iter()
    .map(|c| {
      let mut top = TopK::new(k);
      scan(c * chunk..((c + 1) * chunk).min(n), &mut top);
      top
    })
    .reduce(
      || TopK::new(k),
      |mut a, b| {
        a.merge(b);
        a
      },
    )
}

/// Exact distances from `query` to rows `range` of `data` (row-major,
/// `dim` floats per row), pushed into `top`. Rows failing `filter` are
/// skipped; when most of a block fails, only passing rows are scored.
pub(crate) fn scan_rows(
  metric: Metric,
  dim: usize,
  data: &[f32],
  query: &[f32],
  range: Range<usize>,
  filter: Option<&dyn DocFilter>,
  top: &mut TopK,
) {
  let mut out = [0.0f32; BLOCK];
  let mut pass = [0 as DocId; BLOCK];
  let mut start = range.start;
  while start < range.end {
    let end = (start + BLOCK).min(range.end);
    let rows = end - start;
    match filter {
      None => {
        muninn_simd::distance_batch(
          metric,
          query,
          &data[start * dim..end * dim],
          &mut out[..rows],
        );
        let mut thr = top.threshold();
        for (i, &d) in out[..rows].iter().enumerate() {
          // Ids rise within a scan, so an equal distance never wins.
          if d < thr {
            top.push((start + i) as DocId, d);
            thr = top.threshold();
          }
        }
      }
      Some(f) => {
        let mut np = 0;
        for id in start..end {
          if f.matches(id as DocId) {
            pass[np] = id as DocId;
            np += 1;
          }
        }
        if np * 2 >= rows {
          // Mostly passing: one batched call, then drop the failures.
          muninn_simd::distance_batch(
            metric,
            query,
            &data[start * dim..end * dim],
            &mut out[..rows],
          );
          for &id in &pass[..np] {
            let d = out[id as usize - start];
            if d < top.threshold() {
              top.push(id, d);
            }
          }
        } else {
          for &id in &pass[..np] {
            let s = id as usize * dim;
            let d = muninn_simd::distance(metric, query, &data[s..s + dim]);
            if d < top.threshold() {
              top.push(id, d);
            }
          }
        }
      }
    }
    start = end;
  }
}

/// Exact top-`k` over all `n` rows of `data`, parallel when large.
pub(crate) fn exact_search(
  metric: Metric,
  dim: usize,
  data: &[f32],
  query: &[f32],
  k: usize,
  filter: Option<&dyn DocFilter>,
) -> TopK {
  let n = if dim == 0 { 0 } else { data.len() / dim };
  par_scan(n, k, n * dim.max(1), |range, top| {
    scan_rows(metric, dim, data, query, range, filter, top)
  })
}

/// Re-scores `candidates` with exact distances and keeps the best `k`.
pub(crate) fn rerank(
  metric: Metric,
  dim: usize,
  data: &[f32],
  query: &[f32],
  candidates: impl IntoIterator<Item = DocId>,
  k: usize,
) -> TopK {
  let mut top = TopK::new(k);
  for id in candidates {
    let s = id as usize * dim;
    top.push(id, muninn_simd::distance(metric, query, &data[s..s + dim]));
  }
  top
}

#[cfg(test)]
mod tests {
  use super::*;

  fn data(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
      .map(|i| ((i * 7919 + 13) % 1009) as f32 / 1009.0 - 0.5)
      .collect()
  }

  fn naive(
    metric: Metric,
    dim: usize,
    data: &[f32],
    q: &[f32],
    k: usize,
    f: Option<&dyn DocFilter>,
  ) -> Vec<(DocId, f32)> {
    let mut top = TopK::new(k);
    for (i, row) in data.chunks_exact(dim).enumerate() {
      if f.is_none_or(|f| f.matches(i as DocId)) {
        top.push(i as DocId, muninn_simd::distance(metric, q, row));
      }
    }
    top.into_sorted_vec()
  }

  #[test]
  fn blocked_parallel_scan_equals_naive() {
    let (n, dim) = (9000, 70);
    let d = data(n, dim);
    let q: Vec<f32> = d[5 * dim..6 * dim].to_vec();
    let sparse = |id: DocId| id.is_multiple_of(13);
    let dense = |id: DocId| !id.is_multiple_of(3);
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
      let filters: [Option<&dyn DocFilter>; 3] =
        [None, Some(&sparse), Some(&dense)];
      for f in filters {
        for k in [0, 1, 10, 300] {
          let got = exact_search(metric, dim, &d, &q, k, f).into_sorted_vec();
          assert_eq!(got, naive(metric, dim, &d, &q, k, f));
        }
      }
    }
  }

  #[test]
  fn rerank_keeps_best() {
    let d = data(100, 4);
    let q = [0.0f32; 4];
    let top = rerank(Metric::L2, 4, &d, &q, [3, 50, 7, 99], 2);
    let all = naive(Metric::L2, 4, &d, &q, 100, None);
    let order: Vec<DocId> = all
      .iter()
      .map(|x| x.0)
      .filter(|id| [3, 50, 7, 99].contains(id))
      .take(2)
      .collect();
    let got: Vec<DocId> = top.into_sorted_vec().iter().map(|x| x.0).collect();
    assert_eq!(got, order);
  }
}
