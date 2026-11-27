//! Insertion (Malkov & Yashunin, Algorithm 1) with the neighbour
//! selection heuristic (Algorithm 4), safe to run on many threads at once
//! over a [`BuildView`].

use std::sync::Mutex;

use muninn_core::{DocId, Metric};

use super::graph::{BuildView, read_record, write_record};
use super::search::{Scratch, beam, greedy, with_scratch};
use crate::dist::{Dist, FullDist};

/// Everything an insert needs, shared by all build threads.
pub(crate) struct Builder<'a> {
  pub view: BuildView<'a>,
  pub data: &'a [f32],
  pub dim: usize,
  pub metric: Metric,
  pub ef_construction: usize,
  /// Current entry point and top level.
  pub entry: &'a Mutex<(DocId, u8)>,
}

impl Builder<'_> {
  #[inline]
  fn row(&self, id: DocId) -> &[f32] {
    let s = id as usize * self.dim;
    &self.data[s..s + self.dim]
  }

  #[inline]
  fn d(&self, a: DocId, b: DocId) -> f32 {
    muninn_simd::distance(self.metric, self.row(a), self.row(b))
  }

  /// Links node `q` (already stored, with its level assigned and empty
  /// records) into the graph.
  pub(crate) fn insert(&self, q: DocId) {
    let dist = FullDist {
      metric: self.metric,
      dim: self.dim,
      query: self.row(q),
      data: self.data,
    };
    let lq = self.view.levels[q as usize];
    let (ep, top) = *self.entry.lock().unwrap_or_else(|e| e.into_inner());
    if ep == q {
      return;
    }
    let mut cur = (ep, dist.dist(ep));
    let mut buf = Vec::new();
    for lc in (lq as usize + 1..=top as usize).rev() {
      cur = greedy(&self.view, &dist, cur, lc, &mut buf);
    }
    let mut entries = vec![cur];
    let mut selected = Vec::with_capacity(self.view.m);
    for lc in (0..=lq.min(top) as usize).rev() {
      let found = with_scratch(|s: &mut Scratch| {
        beam(
          &self.view,
          &dist,
          &entries,
          self.ef_construction,
          lc,
          None,
          None,
          s,
        )
      });
      let mut w = found.top.into_sorted_vec();
      w.retain(|&(id, _)| id != q);
      self.select(&w, self.view.m, &mut selected);
      self.view.set(q, lc, &selected);
      for &e in &selected {
        self.connect(e, q, lc);
      }
      entries = w;
    }
    if lq > top {
      let mut g = self.entry.lock().unwrap_or_else(|e| e.into_inner());
      if lq > g.1 {
        *g = (q, lq);
      }
    }
  }

  /// Algorithm 4 without candidate extension: walk candidates from the
  /// closest and keep one only if it is closer to the base node than to
  /// every neighbour already kept. Spreads edges across directions, which
  /// keeps clustered data navigable.
  ///
  /// Inner product is not a metric: large-norm vectors "dominate" every
  /// candidate and the rule above leaves nodes with a handful of edges
  /// (measured: average level-0 degree 7 instead of 16), which hurts
  /// filtered traversal. For [`Metric::Dot`] the list is therefore topped
  /// up with the closest pruned candidates (`keepPrunedConnections`).
  fn select(&self, sorted: &[(DocId, f32)], m: usize, out: &mut Vec<DocId>) {
    out.clear();
    let mut pruned = Vec::new();
    for &(c, dc) in sorted {
      if out.len() >= m {
        break;
      }
      let rc = self.row(c);
      let keep = out
        .iter()
        .all(|&r| muninn_simd::distance(self.metric, rc, self.row(r)) >= dc);
      if keep {
        out.push(c);
      } else if self.metric == Metric::Dot {
        pruned.push(c);
      }
    }
    let room = m.saturating_sub(out.len());
    out.extend(pruned.into_iter().take(room));
  }

  /// Adds the back-link `e -> q` at `level`, re-pruning `e`'s list with
  /// the heuristic when it is full.
  fn connect(&self, e: DocId, q: DocId, level: usize) {
    let _g = self.view.lock(e);
    let Some((rec, cap)) = self.view.record(e, level) else {
      return;
    };
    let mut ids = Vec::with_capacity(cap + 1);
    read_record(rec, cap, &mut ids);
    if ids.contains(&q) {
      return;
    }
    if ids.len() < cap {
      ids.push(q);
      write_record(rec, cap, &ids);
      return;
    }
    let mut cands: Vec<(DocId, f32)> =
      ids.iter().map(|&x| (x, self.d(e, x))).collect();
    cands.push((q, self.d(e, q)));
    cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut keep = Vec::with_capacity(cap);
    self.select(&cands, cap, &mut keep);
    write_record(rec, cap, &keep);
  }
}

/// Level of node `id`: `floor(-ln(U) / ln(M))` with `U` uniform in
/// `(0, 1]`, drawn from a hash of `(seed, id)`. Stateless, so it is the
/// same however and whenever the node is inserted (parallel builds,
/// inserts after reopening a file).
pub(crate) fn level_for(seed: u64, id: DocId, m: usize) -> u8 {
  let h = splitmix64(seed ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
  let u = ((h >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
  let l = -u.ln() / (m as f64).ln();
  l.min(MAX_LEVEL as f64) as u8
}

/// Highest level a node can get. At `M = 2` the chance of exceeding it is
/// below 2^-30 per node; capping only bounds memory for hostile seeds.
pub(crate) const MAX_LEVEL: u8 = 30;

fn splitmix64(mut x: u64) -> u64 {
  x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
  x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn levels_follow_the_geometric_law() {
    let n = 200_000u32;
    let mut counts = [0usize; 8];
    for id in 0..n {
      let l = level_for(42, id, 16) as usize;
      counts[l.min(7)] += 1;
    }
    // P(level >= 1) = 1/16, P(level >= 2) = 1/256.
    let ge1 = n as usize - counts[0];
    assert!((ge1 as f64 / n as f64 - 1.0 / 16.0).abs() < 0.005);
    let ge2: usize = counts[2..].iter().sum();
    assert!((ge2 as f64 / n as f64 - 1.0 / 256.0).abs() < 0.001);
    assert_eq!(level_for(42, 7, 16), level_for(42, 7, 16));
  }
}
