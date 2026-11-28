//! Insertion (Malkov & Yashunin, Algorithm 1) with the neighbour
//! selection heuristic (Algorithm 4), safe to run on many threads at once
//! over a [`BuildView`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use muninn_core::{DocId, Metric};

use super::graph::{BuildView, NONE, Neighbors, read_record, write_record};
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
  /// Level-0 in-degree of every node (how many lists contain it).
  pub indeg: &'a [AtomicU32],
  /// Nodes that may have lost their last incoming level-0 edge during
  /// this batch; see [`Builder::reconnect_orphans`].
  pub orphans: &'a Mutex<Vec<DocId>>,
  /// Per node, the neighbour whose level-0 edge to it was forced in by
  /// [`Builder::ensure_backlink`] ([`NONE`] if none, e.g. for nodes read
  /// from a file). Prunes never drop that edge.
  pub prot: &'a [AtomicU32],
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

  /// Whether the edge `e -> y` is `y`'s protected edge.
  #[inline]
  fn protected(&self, y: DocId, e: DocId) -> bool {
    self.prot[y as usize].load(Relaxed) == e
  }

  /// After node `e`'s candidates (`sorted`, closest first) were pruned to
  /// `keep`, puts back every dropped candidate whose nearest neighbour is
  /// `e`, evicting the farthest unprotected kept entries.
  fn keep_protected(
    &self,
    e: DocId,
    sorted: &[(DocId, f32)],
    keep: &mut Vec<DocId>,
    cap: usize,
  ) {
    for &(y, _) in sorted {
      if !self.protected(y, e) || keep.contains(&y) {
        continue;
      }
      if keep.len() < cap {
        keep.push(y);
        continue;
      }
      let victim = sorted
        .iter()
        .rev()
        .map(|c| c.0)
        .find(|&z| !self.protected(z, e) && keep.contains(&z));
      match victim.and_then(|z| keep.iter().position(|&k| k == z)) {
        Some(i) => keep[i] = y,
        None => break,
      }
    }
  }

  fn note_orphan(&self, x: DocId) {
    self
      .orphans
      .lock()
      .unwrap_or_else(|e| e.into_inner())
      .push(x);
  }

  /// Updates level-0 in-degrees after a list changed from `old` to `new`.
  fn note_change(&self, level: usize, old: &[DocId], new: &[DocId]) {
    if level != 0 {
      return;
    }
    for &x in new {
      if !old.contains(&x) {
        self.indeg[x as usize].fetch_add(1, Relaxed);
      }
    }
    for &x in old {
      if !new.contains(&x) {
        let prev = self.indeg[x as usize]
          .fetch_update(Relaxed, Relaxed, |v| v.checked_sub(1));
        if prev == Ok(1) {
          self.note_orphan(x);
        }
      }
    }
  }

  /// Writes the new node `q`'s list at `level`: `selected` merged with
  /// any back-links other threads added meanwhile (they can reach `q` as
  /// a search entry from the level above before `q`'s own insert gets
  /// here), re-pruned with the heuristic if the union overflows. A plain
  /// overwrite would silently drop those edges.
  fn set_merged(&self, q: DocId, level: usize, selected: &[DocId]) {
    let _g = self.view.lock(q);
    let Some((rec, cap)) = self.view.record(q, level) else {
      return;
    };
    let mut old = Vec::with_capacity(cap);
    read_record(rec, cap, self.data.len() / self.dim, &mut old);
    let mut ids = old.clone();
    for &s in selected {
      if !ids.contains(&s) {
        ids.push(s);
      }
    }
    let keep = if ids.len() <= cap {
      ids
    } else {
      let mut cands: Vec<(DocId, f32)> =
        ids.iter().map(|&x| (x, self.d(q, x))).collect();
      cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
      let mut keep = Vec::with_capacity(cap);
      self.select(&cands, cap, &mut keep);
      if level == 0 {
        self.keep_protected(q, &cands, &mut keep, cap);
      }
      keep
    };
    write_record(rec, cap, &keep);
    self.note_change(level, &old, &keep);
  }

  /// Whether `q` is the current entry point.
  pub(crate) fn is_entry(&self, q: DocId) -> bool {
    self.entry.lock().unwrap_or_else(|e| e.into_inner()).0 == q
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
    let mut links = Vec::with_capacity(lq.min(top) as usize + 1);
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
      self.set_merged(q, lc, &selected);
      links.push((lc, selected.clone()));
      entries = w;
    }
    // Back-links only once every level of `q` is written. Linked level by
    // level, `q` became reachable at its upper levels while its lower
    // lists were still empty, and another thread descending through it
    // got stuck there: its level-0 beam could not expand, so its own node
    // was linked to `q` alone, far from its real neighbours.
    for (lc, sel) in links {
      for e in sel {
        self.connect(e, q, lc);
      }
    }
    if self.indeg[q as usize].load(Relaxed) == 0 {
      // Every neighbour pruned the back-link to `q` straight away.
      self.note_orphan(q);
    }
    if lq > top {
      let mut g = self.entry.lock().unwrap_or_else(|e| e.into_inner());
      if lq > g.1 {
        // The old entry point may have had no incoming edge: it did not
        // need one while searches started there.
        let old = g.0;
        *g = (q, lq);
        drop(g);
        self.note_orphan(old);
      }
    }
  }

  /// Re-links nodes left without any incoming level-0 edge.
  ///
  /// Re-pruning a full list with the heuristic can drop a node from the
  /// last list that pointed to it; no search can reach it after that
  /// (measured on 20k clustered vectors: 3 to 12 such nodes per build,
  /// at any thread count). Each orphan is searched for from the entry
  /// point like a new insert and added to the closest reachable node's
  /// list, replacing that node's farthest neighbour that has another
  /// incoming edge if the list is full. Run serially after a batch;
  /// orphans are rare, so it costs little.
  pub(crate) fn reconnect_orphans(&self) {
    let mut orphans = std::mem::take(
      &mut *self.orphans.lock().unwrap_or_else(|e| e.into_inner()),
    );
    orphans.sort_unstable();
    orphans.dedup();
    for x in orphans {
      if self.indeg[x as usize].load(Relaxed) == 0 && !self.is_entry(x) {
        let _ = self.adopt(x);
      }
    }
  }

  /// Makes every node reachable from the entry point at level 0.
  ///
  /// [`Builder::reconnect_orphans`] catches nodes with no in-edge at all,
  /// but a few nodes can also end up linking only to each other, with no
  /// edge from the reachable part (measured: 4 to 5 per million, and 27
  /// in 20k with M = 4). A breadth-first walk from the entry point finds
  /// them; each one left unreachable gets an edge from the closest
  /// reachable node, which also brings in everything it links to.
  /// O(n * M) reads, so callers run it only for large enough batches.
  pub(crate) fn reconnect_unreachable(&self) {
    let n = self.view.levels.len();
    if n == 0 {
      return;
    }
    let entry = self.entry.lock().unwrap_or_else(|e| e.into_inner()).0;
    let mut seen = vec![false; n];
    let mut buf = Vec::new();
    let mut stack = Vec::new();
    let mut walk = |from: DocId, seen: &mut [bool], stack: &mut Vec<u32>| {
      if seen[from as usize] {
        return;
      }
      seen[from as usize] = true;
      stack.push(from);
      while let Some(x) = stack.pop() {
        for &y in self.view.neighbors(x, 0, &mut buf) {
          if (y as usize) < n && !seen[y as usize] {
            seen[y as usize] = true;
            stack.push(y);
          }
        }
      }
    };
    walk(entry, &mut seen, &mut stack);
    for x in 0..n as DocId {
      if !seen[x as usize] && self.adopt(x) {
        walk(x, &mut seen, &mut stack);
      }
    }
  }

  /// Gives orphan `x` one incoming level-0 edge from the closest node a
  /// search from the entry point finds (so a reachable one).
  fn adopt(&self, x: DocId) -> bool {
    let dist = FullDist {
      metric: self.metric,
      dim: self.dim,
      query: self.row(x),
      data: self.data,
    };
    let (ep, top) = *self.entry.lock().unwrap_or_else(|e| e.into_inner());
    let mut cur = (ep, dist.dist(ep));
    let mut buf = Vec::new();
    for lc in (1..=top as usize).rev() {
      cur = greedy(&self.view, &dist, cur, lc, &mut buf);
    }
    let found = with_scratch(|s: &mut Scratch| {
      beam(
        &self.view,
        &dist,
        &[cur],
        self.ef_construction,
        0,
        None,
        None,
        s,
      )
    });
    for (e, _) in found.top.into_sorted_vec() {
      if e != x && self.force_link(e, x) {
        return true;
      }
    }
    false
  }

  /// Makes sure the closest of `x`'s level-0 neighbours (that has room)
  /// links back to it, and from then on every prune keeps that edge.
  ///
  /// The heuristic drops a back-link `e -> x` when some neighbour `r` of
  /// `e` is closer to `x` than `e` is, assuming `x` is reachable through
  /// `r`; but `r` need not link to `x`. When that happens at every node
  /// near `x`, `x`'s only in-links come from far away and a search for
  /// `x`'s own vector arrives next to it without seeing it (measured: a
  /// few nodes per 20k, more in parallel builds). One forced edge from
  /// its nearest neighbour fixes that.
  pub(crate) fn ensure_backlink(&self, x: DocId) {
    if self.prot[x as usize].load(Relaxed) != NONE {
      return;
    }
    let mut out = Vec::new();
    {
      let _g = self.view.lock(x);
      if let Some((rec, cap)) = self.view.record(x, 0) {
        read_record(rec, cap, self.data.len() / self.dim, &mut out);
      }
    }
    let mut by_dist: Vec<(DocId, f32)> = out
      .into_iter()
      .filter(|&e| e != x)
      .map(|e| (e, self.d(x, e)))
      .collect();
    by_dist.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    // The nearest neighbour that can take the edge (a hub whose list is
    // all protected edges cannot).
    for (e, _) in by_dist {
      if self.force_link(e, x) {
        self.prot[x as usize].store(e, Relaxed);
        return;
      }
    }
  }

  /// Adds `x` to `e`'s level-0 list if absent: appended if there is room,
  /// else replacing `e`'s farthest neighbour that has another in-edge.
  /// Returns whether `e` now links to `x`.
  fn force_link(&self, e: DocId, x: DocId) -> bool {
    let n = self.data.len() / self.dim;
    let _g = self.view.lock(e);
    let Some((rec, cap)) = self.view.record(e, 0) else {
      return false;
    };
    let mut ids = Vec::with_capacity(cap + 1);
    read_record(rec, cap, n, &mut ids);
    if ids.contains(&x) {
      return true;
    }
    let old = ids.clone();
    if ids.len() < cap {
      ids.push(x);
    } else {
      // Never evict a node's last in-edge or its protected one.
      let victim = (0..ids.len())
        .filter(|&i| {
          self.indeg[ids[i] as usize].load(Relaxed) >= 2
            && !self.protected(ids[i], e)
        })
        .max_by(|&a, &b| self.d(e, ids[a]).total_cmp(&self.d(e, ids[b])));
      let Some(i) = victim else {
        return false;
      };
      ids[i] = x;
    }
    write_record(rec, cap, &ids);
    self.note_change(0, &old, &ids);
    true
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
  pub(crate) fn connect(&self, e: DocId, q: DocId, level: usize) {
    let _g = self.view.lock(e);
    let Some((rec, cap)) = self.view.record(e, level) else {
      return;
    };
    let mut ids = Vec::with_capacity(cap + 1);
    read_record(rec, cap, self.data.len() / self.dim, &mut ids);
    if ids.contains(&q) {
      return;
    }
    if ids.len() < cap {
      ids.push(q);
      write_record(rec, cap, &ids);
      self.note_change(level, &[], &[q]);
      return;
    }
    let mut cands: Vec<(DocId, f32)> =
      ids.iter().map(|&x| (x, self.d(e, x))).collect();
    cands.push((q, self.d(e, q)));
    cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut keep = Vec::with_capacity(cap);
    self.select(&cands, cap, &mut keep);
    if level == 0 {
      self.keep_protected(e, &cands, &mut keep, cap);
    }
    write_record(rec, cap, &keep);
    self.note_change(level, &ids, &keep);
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
