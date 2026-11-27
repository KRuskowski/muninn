//! Greedy descent and beam search over a [`Neighbors`] graph.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use muninn_core::{DocFilter, DocId, TopK};

use super::graph::Neighbors;
use crate::dist::Dist;

/// Generation-stamped visited marks: `marks[i] == gen` means visited in
/// the current search. Starting a search is one increment, not a clear;
/// the array is cleared only when the 16-bit generation wraps.
#[derive(Default)]
pub(crate) struct Visited {
  marks: Vec<u16>,
  generation: u16,
}

impl Visited {
  /// Starts a new search over `n` nodes.
  #[inline]
  pub(crate) fn reset(&mut self, n: usize) {
    if self.marks.len() < n {
      self.marks.resize(n, 0);
    }
    self.generation = self.generation.wrapping_add(1);
    if self.generation == 0 {
      self.marks.fill(0);
      self.generation = 1;
    }
  }

  /// Marks `id`; returns whether it was not yet visited.
  #[inline]
  pub(crate) fn insert(&mut self, id: DocId) -> bool {
    let m = &mut self.marks[id as usize];
    let fresh = *m != self.generation;
    *m = self.generation;
    fresh
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

/// Per-thread reusable search state, so a query allocates nothing that
/// scales with the index.
#[derive(Default)]
pub(crate) struct Scratch {
  pub visited: Visited,
  pub cands: BinaryHeap<Near>,
  pub buf: Vec<u32>,
  pub fresh: Vec<u32>,
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
#[derive(Clone, Copy)]
pub(crate) struct Abort {
  pub n: usize,
  pub ef: usize,
  pub min_visits: usize,
}

/// Result of [`beam`].
pub(crate) struct Beam {
  /// The best `ef` nodes that pass the filter.
  pub top: TopK,
  /// The filter turned out too selective for the graph; scan instead.
  pub aborted: bool,
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
  let n = g.n();
  let Scratch {
    visited,
    cands,
    buf,
    fresh,
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
    fresh.clear();
    for &x in g.neighbors(c.id, level, buf) {
      if (x as usize) < n && visited.insert(x) {
        dist.prefetch(x);
        fresh.push(x);
      }
    }
    for &x in fresh.iter() {
      let d = dist.dist(x);
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
      let s = passes as f32 / visits as f32;
      if visits >= a.min_visits && prefer_exact(a.n, s, a.ef) {
        return Beam { top, aborted: true };
      }
    }
  }
  Beam {
    top,
    aborted: false,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn visited_generations_and_wrap() {
    let mut v = Visited::default();
    v.reset(4);
    assert!(v.insert(2));
    assert!(!v.insert(2));
    for _ in 0..70000 {
      v.reset(4);
    }
    assert!(v.insert(2));
    assert!(v.insert(3));
    assert!(!v.insert(3));
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
}
