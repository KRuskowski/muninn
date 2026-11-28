//! Exact scans shared by every index: blocked, filter-aware, parallel.

use std::ops::Range;

use muninn_core::{BitSet, DocFilter, DocId, Metric, TopK};
use rayon::prelude::*;

use crate::hnsw::lookup;

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
/// `dim` floats per row), pushed into `top`. Row `r` is reported as id
/// `ext[r]` (as `r` when `ext` is empty), and the filter is asked about
/// that id. Rows failing `filter` are skipped; when most of a block
/// fails, only passing rows are scored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_rows(
  metric: Metric,
  dim: usize,
  data: &[f32],
  query: &[f32],
  range: Range<usize>,
  filter: Option<&dyn DocFilter>,
  ext: &[u32],
  top: &mut TopK,
) {
  let id_of = |r: usize| lookup(ext, r as DocId);
  let mut out = [0.0f32; BLOCK];
  let mut pass = [0usize; BLOCK];
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
          // Unmapped ids rise within a scan, so an equal distance never
          // wins; mapped ones may tie with a larger kept id.
          if d < thr || (d == thr && !ext.is_empty()) {
            top.push(id_of(start + i), d);
            thr = top.threshold();
          }
        }
      }
      Some(f) => {
        let mut np = 0;
        for r in start..end {
          if f.matches(id_of(r)) {
            pass[np] = r;
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
          for &r in &pass[..np] {
            top.push(id_of(r), out[r - start]);
          }
        } else {
          for &r in &pass[..np] {
            let s = r * dim;
            let d = muninn_simd::distance(metric, query, &data[s..s + dim]);
            top.push(id_of(r), d);
          }
        }
      }
    }
    start = end;
  }
}

/// A filter whose candidate set holds at most `1 / SPARSE_DIV` of the
/// rows is scanned by walking the set's bits; denser ones by blocks of
/// rows, where batched distance calls win.
const SPARSE_DIV: usize = 8;

/// Like [`scan_rows`] with a filter, but visits only the ids set in
/// `bits` (a superset of the passing ids, see `DocFilter::candidates`),
/// so the cost follows the number of candidates rather than the range.
/// `range` is in ids; id `e` is row `int[e]` (row `e` when `int` is
/// empty).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_bits(
  metric: Metric,
  dim: usize,
  data: &[f32],
  query: &[f32],
  range: Range<usize>,
  bits: &BitSet,
  filter: &dyn DocFilter,
  int: &[u32],
  top: &mut TopK,
) {
  let n = data.len().checked_div(dim).unwrap_or(0);
  let words = bits.words();
  let end = range.end.min(bits.capacity());
  if range.start >= end {
    return;
  }
  let (w0, w1) = (range.start / 64, end.div_ceil(64).min(words.len()));
  // A tight walk over the words: the zero words dominate when the set is
  // selective, so they must cost next to nothing.
  for (k, &word) in words[w0..w1].iter().enumerate() {
    if word == 0 {
      continue;
    }
    let wi = w0 + k;
    let mut w = word;
    while w != 0 {
      let id = wi * 64 + w.trailing_zeros() as usize;
      w &= w - 1;
      if id < range.start || id >= end {
        continue;
      }
      let r = lookup(int, id as DocId) as usize;
      if r < n && filter.matches(id as DocId) {
        let s = r * dim;
        let d = muninn_simd::distance(metric, query, &data[s..s + dim]);
        if d <= top.threshold() {
          top.push(id as DocId, d);
        }
      }
    }
  }
}

/// Exact top-`k` over all `n` rows of `data`, parallel when large. A
/// selective filter with a candidate set is scanned by its set bits.
/// With `map = Some((ext, int))`, row `r` is reported as id `ext[r]`, and
/// the filter speaks those ids (see `hnsw::order`); results tie-break on
/// them.
pub(crate) fn exact_search(
  metric: Metric,
  dim: usize,
  data: &[f32],
  query: &[f32],
  k: usize,
  filter: Option<&dyn DocFilter>,
  map: Option<(&[u32], &[u32])>,
) -> TopK {
  let n = if dim == 0 { 0 } else { data.len() / dim };
  let (ext, int) = map.unwrap_or((&[], &[]));
  if let Some(f) = filter
    && let Some(bits) = f.candidates()
  {
    let count = bits.count();
    if count.saturating_mul(SPARSE_DIV) <= n {
      return par_scan(n, k, count * dim.max(1), |range, top| {
        scan_bits(metric, dim, data, query, range, bits, f, int, top)
      });
    }
  }
  par_scan(n, k, n * dim.max(1), |range, top| {
    scan_rows(metric, dim, data, query, range, filter, ext, top)
  })
}

/// Re-scores `candidates` with exact distances and keeps the best `k`.
/// Candidates are row numbers; row `r` is reported as id `ext[r]` (as `r`
/// when `ext` is empty).
pub(crate) fn rerank(
  metric: Metric,
  data: &[f32],
  query: &[f32],
  candidates: impl IntoIterator<Item = DocId>,
  k: usize,
  ext: &[u32],
) -> TopK {
  let rows: Vec<DocId> = candidates.into_iter().collect();
  let mut out = vec![0.0f32; rows.len()];
  muninn_simd::distance_gather(metric, query, data, &rows, &mut out);
  let mut top = TopK::new(k);
  for (&r, &d) in rows.iter().zip(&out) {
    top.push(lookup(ext, r), d);
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
          let got =
            exact_search(metric, dim, &d, &q, k, f, None).into_sorted_vec();
          assert_eq!(got, naive(metric, dim, &d, &q, k, f));
        }
      }
    }
  }

  /// A filter that counts its `matches` calls.
  struct Counting<'a> {
    bits: &'a BitSet,
    calls: std::sync::atomic::AtomicUsize,
  }

  impl DocFilter for Counting<'_> {
    fn matches(&self, id: DocId) -> bool {
      self
        .calls
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
      self.bits.contains(id) && !id.is_multiple_of(3)
    }

    fn candidates(&self) -> Option<&BitSet> {
      Some(self.bits)
    }
  }

  /// A selective filter with a candidate set is answered by visiting the
  /// set's ids only (asked about each once), with the same results as the
  /// row scan, sequentially and in parallel.
  #[test]
  fn selective_filter_scans_only_its_candidates() {
    let (dim, q) = (8, [0.1f32; 8]);
    for n in [1000, 300_000] {
      let d = data(n, dim);
      let mut bits = BitSet::new(n + 100);
      for id in (0..n as DocId).filter(|i| i % 97 == 5) {
        bits.insert(id);
      }
      // Ids past the data must be ignored.
      bits.insert(n as DocId + 50);
      let f = Counting {
        bits: &bits,
        calls: Default::default(),
      };
      for metric in [Metric::L2, Metric::Cosine] {
        f.calls.store(0, std::sync::atomic::Ordering::Relaxed);
        let got = exact_search(metric, dim, &d, &q, 10, Some(&f), None);
        let calls = f.calls.load(std::sync::atomic::Ordering::Relaxed);
        assert!(calls <= bits.count(), "{calls} matches calls for {n} rows");
        let pred = |id: DocId| bits.contains(id) && !id.is_multiple_of(3);
        let want = naive(metric, dim, &d, &q, 10, Some(&pred));
        assert_eq!(got.into_sorted_vec(), want);
      }
    }
  }

  #[test]
  fn rerank_keeps_best() {
    let d = data(100, 4);
    let q = [0.0f32; 4];
    let top = rerank(Metric::L2, &d, &q, [3, 50, 7, 99], 2, &[]);
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
