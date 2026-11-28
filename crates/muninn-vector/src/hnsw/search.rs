//! Greedy descent and beam search over a [`Neighbors`] graph.
//!
//! Unfiltered beam search (every query without a filter, and every insert
//! during a build) keeps its candidates in a [`Pool`]: one sorted array
//! of the best `ef` nodes, each flagged once expanded. It replaces the
//! textbook pair of heaps (closest-first candidates, farthest-first
//! results) with a single structure that stays in a few cache lines and
//! has no unpredictable sift branches. Filtered search keeps the two
//! heaps, because there the candidates (any node) and the results (only
//! passing nodes) are different sets.
//!
//! Both score the unvisited neighbours of an expanded node in one
//! [`Dist::dist_many`] call, which for full-precision vectors scores
//! several rows per kernel call while prefetching the next ones.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use muninn_core::{DocFilter, DocId, TopK};

use super::graph::Neighbors;
use crate::dist::Dist;

/// Visited marks: one bit per node. The words set during a search are
/// remembered, so starting the next search clears only those (a search
/// touches a few thousand nodes, not the whole index), and the bit array
/// is small enough (125 KB per million nodes) to stay in L2 next to the
/// vectors being scored.
#[derive(Default)]
pub(crate) struct Visited {
  bits: Vec<u64>,
  touched: Vec<u32>,
}

impl Visited {
  /// Starts a new search over `n` nodes.
  #[inline]
  pub(crate) fn reset(&mut self, n: usize) {
    for &w in &self.touched {
      self.bits[w as usize] = 0;
    }
    self.touched.clear();
    let words = n.div_ceil(64);
    if self.bits.len() < words {
      self.bits.resize(words, 0);
    }
  }

  /// Marks `id` (which must be below the `n` of the last reset); returns
  /// whether it was not yet visited.
  #[inline]
  pub(crate) fn insert(&mut self, id: DocId) -> bool {
    let wi = id as usize >> 6;
    let w = &mut self.bits[wi];
    let m = 1u64 << (id & 63);
    if *w & m != 0 {
      return false;
    }
    if *w == 0 {
      self.touched.push(wi as u32);
    }
    *w |= m;
    true
  }
}

/// A candidate ordered so that `BinaryHeap` pops the *closest* first.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Near {
  pub d: f32,
  pub id: DocId,
}

impl PartialEq for Near {
  fn eq(&self, o: &Self) -> bool {
    self.cmp(o) == Ordering::Equal
  }
}
impl Eq for Near {}
impl PartialOrd for Near {
  fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
    Some(self.cmp(o))
  }
}
impl Ord for Near {
  fn cmp(&self, o: &Self) -> Ordering {
    o.d.total_cmp(&self.d).then_with(|| o.id.cmp(&self.id))
  }
}

/// Sort key of a candidate: the distance's bits mapped so that unsigned
/// order is float order (negative values flipped, positive ones offset),
/// then the id. One `u64` compare ranks by distance, then by smaller id.
/// `-0.0` is folded into `+0.0` first, so the two tie (as in `TopK`).
#[inline]
fn key(d: f32, id: DocId) -> u64 {
  let b = (d + 0.0).to_bits();
  let k = if b >> 31 == 1 { !b } else { b | 0x8000_0000 };
  ((k as u64) << 32) | id as u64
}

/// The candidate list of an unfiltered beam search: the best `cap` nodes
/// seen so far, sorted by [`key`] (distance, then id, so results are
/// deterministic), each flagged once its neighbours have been scored.
/// `cur` is at or before the first unexpanded slot.
#[derive(Default)]
pub(crate) struct Pool {
  keys: Vec<u64>,
  dists: Vec<f32>,
  expanded: Vec<bool>,
  cap: usize,
  cur: usize,
}

impl Pool {
  /// Empties the pool and sets its capacity.
  #[inline]
  fn reset(&mut self, cap: usize) {
    self.keys.clear();
    self.dists.clear();
    self.expanded.clear();
    let room = cap.saturating_add(1).min(1 << 16);
    self.keys.reserve(room);
    self.dists.reserve(room);
    self.expanded.reserve(room);
    self.cap = cap;
    self.cur = 0;
  }

  /// The distance a new candidate must beat: the worst kept one once the
  /// pool is full, infinity before.
  #[inline]
  fn threshold(&self) -> f32 {
    if self.keys.len() >= self.cap {
      self.dists.last().copied().unwrap_or(f32::NEG_INFINITY)
    } else {
      f32::INFINITY
    }
  }

  /// Offers a candidate; keeps it if it ranks among the best `cap`.
  /// NaN distances are never kept.
  #[inline]
  fn insert(&mut self, id: DocId, d: f32) {
    if self.cap == 0 || d.is_nan() {
      return;
    }
    let k = key(d, id);
    if self.keys.len() >= self.cap {
      if self.keys.last().is_some_and(|&last| k >= last) {
        return;
      }
      self.keys.pop();
      self.dists.pop();
      self.expanded.pop();
    }
    let pos = lower_bound(&self.keys, k);
    self.keys.insert(pos, k);
    self.dists.insert(pos, d);
    self.expanded.insert(pos, false);
    if pos < self.cur {
      self.cur = pos;
    }
  }

  /// The closest unexpanded candidate, now flagged as expanded.
  #[inline]
  fn pop(&mut self) -> Option<DocId> {
    while self.cur < self.keys.len() {
      let i = self.cur;
      self.cur += 1;
      if !self.expanded[i] {
        self.expanded[i] = true;
        return Some(self.keys[i] as u32);
      }
    }
    None
  }

  /// The closest unexpanded candidate, without taking it.
  #[inline]
  fn peek(&self) -> Option<DocId> {
    (self.cur..self.keys.len())
      .find(|&i| !self.expanded[i])
      .map(|i| self.keys[i] as u32)
  }

  /// The kept candidates, closest first.
  fn to_vec(&self) -> Vec<(DocId, f32)> {
    self
      .keys
      .iter()
      .zip(&self.dists)
      .map(|(&k, &d)| (k as u32, d))
      .collect()
  }
}

/// Index of the first key `>= k` in sorted `keys`: a binary search whose
/// steps compile to conditional moves, since the outcome of each compare
/// is a coin flip that a branch would mispredict half the time.
#[inline]
fn lower_bound(keys: &[u64], k: u64) -> usize {
  let mut base = 0;
  let mut len = keys.len();
  while len > 1 {
    let half = len / 2;
    // `base + half < keys.len()` because `half < len`.
    base = if keys[base + half] < k {
      base + half
    } else {
      base
    };
    len -= half;
  }
  base + (len == 1 && keys[base] < k) as usize
}

/// Per-thread reusable search state, so a query allocates nothing that
/// scales with the index.
#[derive(Default)]
pub(crate) struct Scratch {
  pub visited: Visited,
  pub cands: BinaryHeap<Near>,
  pub pool: Pool,
  pub buf: Vec<u32>,
  pub fresh: Vec<u32>,
  pub dists: Vec<f32>,
}

thread_local! {
  static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Runs `f` with this thread's scratch. `f` must not start rayon work
/// (a stolen job could need the same scratch); if the scratch is busy
/// anyway, a temporary one is used instead of panicking.
pub(crate) fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
  SCRATCH.with(|c| match c.try_borrow_mut() {
    Ok(mut s) => f(&mut s),
    Err(_) => f(&mut Scratch::default()),
  })
}

/// Moves greedily to the closest neighbour at `level` until no neighbour
/// improves on `cur`.
#[inline]
pub(crate) fn greedy<G: Neighbors, D: Dist>(
  g: &G,
  dist: &D,
  mut cur: (DocId, f32),
  level: usize,
  buf: &mut Vec<u32>,
) -> (DocId, f32) {
  let n = g.n();
  loop {
    let mut moved = false;
    let at = cur.0;
    for &x in g.neighbors(at, level, buf) {
      if (x as usize) < n {
        let d = dist.dist(x);
        if d < cur.1 {
          cur = (x, d);
          moved = true;
        }
      }
    }
    if !moved {
      return cur;
    }
  }
}

/// Cost model for filtered search, in units of one distance computation:
///
/// - graph: about `GRAPH_COST * ef` distances unfiltered, and a filter
///   passing fraction `s` stretches the walk by `1 / s` (the beam must
///   see `ef / s` nodes to collect `ef` passing ones);
/// - exact scan: `s * n` distances plus `n` filter checks at about
///   `1 / FILTER_RATIO` of a distance each.
///
/// The constants come from `examples/vector_bench.rs` (128 dimensions,
/// `BitSet` filter); the crossover moves with `n` and `ef`, which a fixed
/// selectivity threshold cannot capture.
const GRAPH_COST: f32 = 25.0;
const FILTER_RATIO: f32 = 30.0;

/// Whether an exact scan over the passing documents is predicted to be
/// cheaper than graph search for selectivity `s`.
#[inline]
pub(crate) fn prefer_exact(n: usize, s: f32, ef: usize) -> bool {
  let s = s.max(1e-6);
  let n = n as f32;
  n * (s + 1.0 / FILTER_RATIO) < GRAPH_COST * ef as f32 / s
}

/// Stop rule for filtered graph search without a selectivity hint: once
/// `min_visits` nodes were scored, estimate the selectivity from how many
/// passed and give up on the graph if [`prefer_exact`] says so.
///
/// The estimate is pessimistic (about two standard deviations above the
/// observed pass rate). The rule is re-checked after every expansion, so
/// with the raw rate, sampling noise near the crossover sent a few
/// percent of queries to a needless exact scan (measured: 10% slower at
/// 10% selectivity than graph-only search).
#[derive(Clone, Copy)]
pub(crate) struct Abort {
  pub n: usize,
  pub ef: usize,
  pub min_visits: usize,
}

/// Result of [`beam`].
pub(crate) struct Beam {
  /// The best `ef` nodes that pass the filter, closest first (ties by
  /// smaller id).
  pub top: Vec<(DocId, f32)>,
  /// The filter turned out too selective for the graph; scan instead.
  pub aborted: bool,
}

/// Collects the unvisited, in-range neighbours of `node` at `level` into
/// `fresh`, marking them visited and prefetching their rows.
#[inline]
fn fresh_neighbors<G: Neighbors, D: Dist>(
  g: &G,
  dist: &D,
  node: DocId,
  level: usize,
  visited: &mut Visited,
  buf: &mut Vec<u32>,
  fresh: &mut Vec<u32>,
) {
  let n = g.n();
  fresh.clear();
  for &x in g.neighbors(node, level, buf) {
    if (x as usize) < n && visited.insert(x) {
      dist.prefetch(x);
      fresh.push(x);
    }
  }
}

/// Beam search at `level` from `entries` with a candidate list of `ef`.
///
/// Nodes failing `filter` are still traversed (so the search can cross
/// filtered-out regions) but never enter the results.
#[allow(clippy::too_many_arguments)]
pub(crate) fn beam<G: Neighbors, D: Dist>(
  g: &G,
  dist: &D,
  entries: &[(DocId, f32)],
  ef: usize,
  level: usize,
  filter: Option<&dyn DocFilter>,
  abort: Option<Abort>,
  s: &mut Scratch,
) -> Beam {
  if filter.is_none() && abort.is_none() {
    return Beam {
      top: beam_plain(g, dist, entries, ef, level, s),
      aborted: false,
    };
  }
  let n = g.n();
  let Scratch {
    visited,
    cands,
    buf,
    fresh,
    dists,
    ..
  } = s;
  visited.reset(n);
  cands.clear();
  let mut top = TopK::new(ef);
  let pass = |id: DocId| filter.is_none_or(|f| f.matches(id));
  let (mut visits, mut passes) = (0usize, 0usize);
  for &(id, d) in entries {
    if (id as usize) < n && visited.insert(id) {
      visits += 1;
      cands.push(Near { d, id });
      if pass(id) {
        passes += 1;
        top.push(id, d);
      }
    }
  }
  while let Some(c) = cands.pop() {
    if c.d > top.threshold() {
      break;
    }
    fresh_neighbors(g, dist, c.id, level, visited, buf, fresh);
    dists.resize(fresh.len(), 0.0);
    dist.dist_many(fresh, dists);
    for (&x, &d) in fresh.iter().zip(dists.iter()) {
      visits += 1;
      let ok = pass(x);
      passes += ok as usize;
      if d < top.threshold() {
        cands.push(Near { d, id: x });
        if ok {
          top.push(x, d);
        }
      }
    }
    if let Some(a) = abort {
      let p = passes as f32;
      let s = (p + 2.0 * p.sqrt() + 1.0) / visits as f32;
      if visits >= a.min_visits && prefer_exact(a.n, s, a.ef) {
        return Beam {
          top: top.into_sorted_vec(),
          aborted: true,
        };
      }
    }
  }
  Beam {
    top: top.into_sorted_vec(),
    aborted: false,
  }
}

/// Unfiltered beam search on a [`Pool`]: expand the closest unexpanded
/// candidate until every one of the best `ef` has been expanded (the
/// HNSW stop rule: the next candidate is farther than the `ef`-th best).
fn beam_plain<G: Neighbors, D: Dist>(
  g: &G,
  dist: &D,
  entries: &[(DocId, f32)],
  ef: usize,
  level: usize,
  s: &mut Scratch,
) -> Vec<(DocId, f32)> {
  let n = g.n();
  let Scratch {
    visited,
    pool,
    buf,
    fresh,
    dists,
    ..
  } = s;
  visited.reset(n);
  pool.reset(ef);
  for &(id, d) in entries {
    if (id as usize) < n && visited.insert(id) {
      pool.insert(id, d);
    }
  }
  while let Some(c) = pool.pop() {
    fresh_neighbors(g, dist, c, level, visited, buf, fresh);
    // The next expansion most likely starts from the current best
    // unexpanded candidate: fetch its list while this one is scored.
    if let Some(next) = pool.peek() {
      g.prefetch(next, level);
    }
    dists.resize(fresh.len(), 0.0);
    dist.dist_many(fresh, dists);
    for (&x, &d) in fresh.iter().zip(dists.iter()) {
      if d < pool.threshold() {
        pool.insert(x, d);
      }
    }
  }
  pool.to_vec()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn visited_marks_and_resets() {
    let mut v = Visited::default();
    v.reset(200);
    assert!(v.insert(2));
    assert!(!v.insert(2));
    assert!(v.insert(130));
    assert!(v.insert(131));
    v.reset(200);
    assert!(v.insert(2));
    assert!(v.insert(131));
    assert!(!v.insert(131));
    // Growing keeps working, and old marks never leak into a new search.
    v.reset(100_000);
    assert!(v.insert(99_999));
    assert!(v.insert(2));
    for _ in 0..3 {
      v.reset(100_000);
      assert!(v.insert(99_999) && v.insert(2) && v.insert(131));
    }
  }

  #[test]
  fn near_pops_closest_then_smallest_id() {
    let mut h = BinaryHeap::new();
    for (d, id) in [(2.0, 1), (1.0, 5), (1.0, 3), (3.0, 0)] {
      h.push(Near { d, id });
    }
    let order: Vec<DocId> =
      std::iter::from_fn(|| h.pop().map(|x| x.id)).collect();
    assert_eq!(order, vec![3, 5, 1, 0]);
  }

  /// The pool keeps the best `cap` by (distance, id), expands closest
  /// first, and goes back for a closer candidate inserted later.
  #[test]
  fn pool_keeps_best_and_expands_closest_first() {
    let mut p = Pool::default();
    p.reset(3);
    assert_eq!(p.threshold(), f32::INFINITY);
    for (id, d) in [(10, 5.0), (11, 1.0), (12, 3.0), (13, 9.0), (14, f32::NAN)]
    {
      p.insert(id, d);
    }
    assert_eq!(p.to_vec(), vec![(11, 1.0), (12, 3.0), (10, 5.0)]);
    assert_eq!(p.threshold(), 5.0);
    // Equal distance: the smaller id ranks first; a larger one is out.
    p.insert(9, 5.0);
    p.insert(15, 3.0);
    assert_eq!(p.to_vec(), vec![(11, 1.0), (12, 3.0), (15, 3.0)]);
    assert_eq!(p.pop(), Some(11));
    assert_eq!(p.pop(), Some(12));
    // A closer late arrival is expanded next; it pushes 15 out.
    p.insert(1, 0.5);
    assert_eq!(p.peek(), Some(1));
    assert_eq!(p.pop(), Some(1));
    assert_eq!(p.pop(), None);
    assert_eq!(p.to_vec(), vec![(1, 0.5), (11, 1.0), (12, 3.0)]);
    p.reset(0);
    p.insert(1, 0.0);
    assert_eq!(p.pop(), None);
    // Negative distances (inner product) order below positive ones.
    p.reset(4);
    for (id, d) in [(1, 0.5), (2, -2.0), (3, -0.0), (4, 0.0), (5, -1.0)] {
      p.insert(id, d);
    }
    let ids: Vec<DocId> = p.to_vec().iter().map(|x| x.0).collect();
    assert_eq!(ids, vec![2, 5, 3, 4]);
    // -0.0 and +0.0 tie, so the smaller id ranks first either way.
    p.reset(2);
    p.insert(9, 0.0);
    p.insert(8, -0.0);
    let ids: Vec<DocId> = p.to_vec().iter().map(|x| x.0).collect();
    assert_eq!(ids, vec![8, 9]);
    p.reset(2);
    p.insert(8, 0.0);
    p.insert(9, -0.0);
    let ids: Vec<DocId> = p.to_vec().iter().map(|x| x.0).collect();
    assert_eq!(ids, vec![8, 9]);
  }

  #[test]
  fn lower_bound_matches_partition_point() {
    let keys: Vec<u64> = (0..37).map(|i| i * 3).collect();
    for len in 0..keys.len() {
      for k in 0..115 {
        let s = &keys[..len];
        assert_eq!(lower_bound(s, k), s.partition_point(|&x| x < k));
      }
    }
  }
}
