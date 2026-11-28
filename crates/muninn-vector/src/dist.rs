//! Distance oracles: "how far is stored vector `id` from the query".
//!
//! Graph search and quantised scans are generic over [`Dist`], so the
//! same traversal code runs on full-precision vectors, SQ8 codes or
//! binary codes, each monomorphised with no indirect calls.

use muninn_core::{DocFilter, DocId, Metric, TopK};

/// Distance from a fixed query to stored vectors, lower is closer.
pub(crate) trait Dist: Sync {
  /// Distance to stored vector `id` (which must be in range).
  fn dist(&self, id: DocId) -> f32;

  /// Hints that `id` will be scored soon. Never faults.
  fn prefetch(&self, id: DocId);
}

/// Exact distance on full-precision rows.
pub(crate) struct FullDist<'a> {
  pub metric: Metric,
  pub dim: usize,
  pub query: &'a [f32],
  pub data: &'a [f32],
}

impl FullDist<'_> {
  /// The stored row for `id`.
  #[inline]
  pub fn row(&self, id: DocId) -> &[f32] {
    let s = id as usize * self.dim;
    &self.data[s..s + self.dim]
  }
}

impl Dist for FullDist<'_> {
  #[inline]
  fn dist(&self, id: DocId) -> f32 {
    muninn_simd::distance(self.metric, self.query, self.row(id))
  }

  #[inline]
  fn prefetch(&self, id: DocId) {
    prefetch_lines(self.data.as_ptr(), id as usize * self.dim, self.dim);
  }
}

/// Prefetches the first cache lines of `len` elements at `base + offset`.
/// The hardware prefetcher picks up the rest of a row once it starts.
#[inline(always)]
pub(crate) fn prefetch_lines<T>(base: *const T, offset: usize, len: usize) {
  let bytes = len * std::mem::size_of::<T>();
  let p = base.wrapping_add(offset).cast::<i8>();
  prefetch(p);
  if bytes > 64 {
    prefetch(p.wrapping_add(64));
  }
}

#[inline(always)]
fn prefetch(p: *const i8) {
  #[cfg(target_arch = "x86_64")]
  {
    use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
    // SAFETY: SSE is part of the x86-64 baseline, and a prefetch is only
    // a hint: it never dereferences in the language sense and never
    // faults, even on an invalid address.
    unsafe { _mm_prefetch::<_MM_HINT_T0>(p) };
  }
  #[cfg(not(target_arch = "x86_64"))]
  let _ = p;
}

/// Scores rows `range` with `dist`, skipping rows failing `filter`.
pub(crate) fn scan_dist<D: Dist>(
  dist: &D,
  range: std::ops::Range<usize>,
  filter: Option<&dyn DocFilter>,
  top: &mut TopK,
) {
  let mut thr = top.threshold();
  for id in range {
    let id = id as DocId;
    if filter.is_some_and(|f| !f.matches(id)) {
      continue;
    }
    let d = dist.dist(id);
    if d < thr {
      top.push(id, d);
      thr = top.threshold();
    }
  }
}
