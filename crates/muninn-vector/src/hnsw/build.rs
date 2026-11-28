//! Insertion (Malkov & Yashunin, Algorithm 1) with the neighbour
//! selection heuristic (Algorithm 4), safe to run on many threads at once
//! over a [`BuildView`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use muninn_core::{BitSet, DocId, Metric};
use rayon::prelude::*;

use super::graph::{BuildView, NONE, read_record, write_record};
use super::search::{Scratch, beam, greedy, with_scratch};
use crate::dist::{Dist, FullDist};
use crate::quant::{Quantizer, with_code_dist};

/// Pending nodes whose candidate parents are ranked in one parallel step
/// of [`Builder::anchor`]; bounds that step's memory.
const ANCHOR_CHUNK: usize = 1 << 16;

/// Everything an insert needs, shared by all build threads.
pub(crate) struct Builder<'a> {
  pub view: BuildView<'a>,
  pub data: &'a [f32],
  pub dim: usize,
  pub metric: Metric,
  pub ef_construction: usize,
  /// Current entry point and top level.
  pub entry: &'a Mutex<(DocId, u8)>,
  /// Per node, its parent in the protected spanning tree: the node whose
  /// level-0 edge to it no prune or eviction may drop ([`NONE`] for the
  /// root). See [`Builder::anchor`].
  pub prot: &'a [AtomicU32],
  /// Codes to search with while inserting (candidates are re-scored
  /// exactly before selection), or `None` to search full vectors.
  pub codes: Option<&'a Quantizer>,
}

impl Builder<'_> {
  #[inline]
  fn row(&self, id: DocId) -> &[f32] {
    let s = id as usize * self.dim;
    &self.data[s..s + self.dim]
  }

  /// `(id, distance from a)` for every id (all stored), from batched
  /// distance calls.
  fn d_pairs(
    &self,
    a: DocId,
    ids: impl Iterator<Item = DocId>,
  ) -> Vec<(DocId, f32)> {
    let ids: Vec<DocId> = ids.collect();
    let mut out = vec![0.0f32; ids.len()];
    muninn_simd::distance_gather(
      self.metric,
      self.row(a),
      self.data,
      &ids,
      &mut out,
    );
    ids.into_iter().zip(out).collect()
  }

  /// Sorts candidates closest first, ties by smaller id.
  fn sort_cands(c: &mut [(DocId, f32)]) {
    c.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
  }

  /// Whether the edge `e -> y` is `y`'s protected edge.
  #[inline]
  fn protected(&self, y: DocId, e: DocId) -> bool {
    self.prot[y as usize].load(Relaxed) == e
  }

  /// After node `e`'s candidates (`sorted`, closest first) were pruned to
  /// `keep`, puts back every dropped candidate whose protected parent is
  /// `e`, evicting the farthest unprotected kept entries. Always succeeds:
  /// all of `e`'s protected children are in its list, so there are never
  /// more than `cap` of them.
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
    let mut ids = Vec::with_capacity(cap + selected.len());
    read_record(rec, cap, self.data.len() / self.dim, &mut ids);
    for &s in selected {
      if !ids.contains(&s) {
        ids.push(s);
      }
    }
    let keep = if ids.len() <= cap {
      ids
    } else {
      let mut cands = self.d_pairs(q, ids.iter().copied());
      Self::sort_cands(&mut cands);
      let mut keep = Vec::with_capacity(cap);
      self.select(&cands, cap, &mut keep);
      if level == 0 {
        self.keep_protected(q, &cands, &mut keep, cap);
      }
      keep
    };
    write_record(rec, cap, &keep);
  }

  /// Whether `q` is the current entry point.
  pub(crate) fn is_entry(&self, q: DocId) -> bool {
    self.entry.lock().unwrap_or_else(|e| e.into_inner()).0 == q
  }

  /// Links node `q` (already stored, with its level assigned and empty
  /// records) into the graph.
  pub(crate) fn insert(&self, q: DocId) {
    match self.codes {
      Some(qz) => {
        let qc = qz.prepare(self.metric, self.row(q));
        with_code_dist!(qz.dist(&qc), |d| self.insert_with(q, &d, true))
      }
      None => {
        let dist = FullDist {
          metric: self.metric,
          dim: self.dim,
          query: self.row(q),
          data: self.data,
        };
        self.insert_with(q, &dist, false)
      }
    }
  }

  /// [`Builder::insert`] searching with `dist`. With `rescore`, `dist` is
  /// approximate (codes): the candidates it finds are re-scored exactly
  /// before neighbours are selected from them.
  fn insert_with<D: Dist>(&self, q: DocId, dist: &D, rescore: bool) {
    let lq = self.view.levels[q as usize];
    let (ep, top) = *self.entry.lock().unwrap_or_else(|e| e.into_inner());
    if ep == q {
      return;
    }
    let mut cur = (ep, dist.dist(ep));
    let mut buf = Vec::new();
    for lc in (lq as usize + 1..=top as usize).rev() {
      cur = greedy(&self.view, dist, cur, lc, &mut buf);
    }
    let mut entries = vec![cur];
    let mut selected = Vec::with_capacity(self.view.m);
    let mut links = Vec::with_capacity(lq.min(top) as usize + 1);
    for lc in (0..=lq.min(top) as usize).rev() {
      let found = with_scratch(|s: &mut Scratch| {
        beam(
          &self.view,
          dist,
          &entries,
          self.ef_construction,
          lc,
          None,
          None,
          s,
        )
      });
      let mut w = found.top;
      w.retain(|&(id, _)| id != q);
      // A new node takes as many neighbours as its level holds: `2M` at
      // level 0, like faiss (hnswlib takes `M` everywhere and leaves the
      // rest of level 0 to back-links). Measured on SIFT-1M at M = 16:
      // recall@10 at ef 32 rose from 0.903 to 0.916, for 7% more
      // distance computations per search.
      let want = if lc == 0 { self.view.m0 } else { self.view.m };
      if rescore {
        let mut exact = self.d_pairs(q, w.iter().map(|&(id, _)| id));
        Self::sort_cands(&mut exact);
        self.select(&exact, want, &mut selected);
      } else {
        self.select(&w, want, &mut selected);
      }
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
    if lq > top {
      let mut g = self.entry.lock().unwrap_or_else(|e| e.into_inner());
      if lq > g.1 {
        *g = (q, lq);
      }
    }
  }

  /// Gives every node in `pending` a protected parent: a level-0 edge
  /// from a node already in the protected spanning tree.
  ///
  /// This is what makes every node reachable from the entry point, always:
  /// the tree is rooted at the entry, every node except the root has one
  /// protected in-edge from a node that was already in the tree when the
  /// edge was made (so the edges cannot form a cycle cut off from the
  /// root), and no prune or eviction ever drops a protected edge. It runs
  /// serially after all concurrent linking of the batch has finished, so
  /// nothing changes the graph while it works.
  ///
  /// Repairing reachability after the fact was not enough: re-pruning can
  /// drop a node's last in-edge, or leave a small group linking only to
  /// each other, and a repair that links a stranded node into a full list
  /// must evict some other edge, which can strand another node.
  ///
  /// The parent is the nearest out-neighbour already in the tree (an edge
  /// from a node's own neighbourhood also keeps it findable by a search
  /// for its own vector); failing that, the nearest tree node a search
  /// finds; failing that, any tree node with room.
  ///
  /// `old_root` was the entry point when the batch started. If the batch
  /// raised the entry point, the new root gets a protected edge to the old
  /// one, so the whole existing tree hangs below it. `unanchored` holds
  /// exactly the nodes not yet in the tree.
  pub(crate) fn anchor(
    &self,
    old_root: DocId,
    pending: &[DocId],
    unanchored: &mut BitSet,
  ) {
    let root = self.entry.lock().unwrap_or_else(|e| e.into_inner()).0;
    unanchored.remove(root);
    self.prot[root as usize].store(NONE, Relaxed);
    if root != old_root {
      let linked = self.adopt_under(root, old_root);
      debug_assert!(linked, "a new root always has an unprotected slot");
    }
    let n = self.data.len() / self.dim;
    for chunk in pending.chunks(ANCHOR_CHUNK) {
      // The expensive part, ranking each node's out-neighbours by
      // distance, runs in parallel. The ranking may be a little stale by
      // the time the serial pass below uses it (that pass adds edges), but
      // it only suggests parents: `adopt_under` checks each one.
      let seen: &BitSet = unanchored;
      let ranked: Vec<Vec<DocId>> = chunk
        .par_iter()
        .map(|&x| {
          if !seen.contains(x) {
            return Vec::new();
          }
          let mut out = Vec::new();
          {
            let _g = self.view.lock(x);
            if let Some((rec, cap)) = self.view.record(x, 0) {
              read_record(rec, cap, n, &mut out);
            }
          }
          let mut near = self.d_pairs(x, out.into_iter().filter(|&e| e != x));
          Self::sort_cands(&mut near);
          near.into_iter().map(|(e, _)| e).collect()
        })
        .collect();
      for (&x, near) in chunk.iter().zip(ranked) {
        if !unanchored.contains(x) {
          continue;
        }
        let in_tree = |e: DocId| e != x && !unanchored.contains(e);
        let done = near.iter().any(|&e| in_tree(e) && self.adopt_under(e, x))
          || self
            .search_level0(x)
            .into_iter()
            .any(|(e, _)| in_tree(e) && self.adopt_under(e, x))
          || (0..n as DocId).any(|e| in_tree(e) && self.adopt_under(e, x));
        // Protected edges number fewer than the nodes, so some tree node
        // always has an unprotected slot.
        debug_assert!(done, "no tree node could take {x}");
        if done {
          unanchored.remove(x);
        }
      }
    }
  }

  /// Makes `e` the protected parent of `x` if `e` can take the edge.
  fn adopt_under(&self, e: DocId, x: DocId) -> bool {
    if self.force_link(e, x) {
      self.prot[x as usize].store(e, Relaxed);
      true
    } else {
      false
    }
  }

  /// The nodes closest to `x` that a search from the entry point finds at
  /// level 0, closest first.
  fn search_level0(&self, x: DocId) -> Vec<(DocId, f32)> {
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
    found.top
  }

  /// Adds `x` to `e`'s level-0 list if absent: appended if there is room,
  /// else replacing `e`'s farthest neighbour whose edge is not protected
  /// (dropping an unprotected edge never disconnects anything, since every
  /// node keeps its protected one). Returns whether `e` now links to `x`.
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
    if ids.len() < cap {
      ids.push(x);
    } else {
      let d = self.d_pairs(e, ids.iter().copied());
      let victim = (0..ids.len())
        .filter(|&i| !self.protected(ids[i], e))
        .max_by(|&a, &b| d[a].1.total_cmp(&d[b].1));
      let Some(i) = victim else {
        return false;
      };
      ids[i] = x;
    }
    write_record(rec, cap, &ids);
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
  ///
  /// A candidate is checked against the kept neighbours four at a time
  /// (one batched distance call), stopping at the first group holding a
  /// closer one.
  fn select(&self, sorted: &[(DocId, f32)], m: usize, out: &mut Vec<DocId>) {
    out.clear();
    let mut pruned = Vec::new();
    let mut d4 = [0.0f32; 4];
    for &(c, dc) in sorted {
      if out.len() >= m {
        break;
      }
      let rc = self.row(c);
      let keep = out.chunks(4).all(|group| {
        let d = &mut d4[..group.len()];
        muninn_simd::distance_gather(self.metric, rc, self.data, group, d);
        d.iter().all(|&x| x >= dc)
      });
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
      return;
    }
    let mut cands = self.d_pairs(e, ids.iter().copied().chain([q]));
    // `q` farther than every current neighbour: re-pruning a list the
    // heuristic produced would keep the list and drop `q` (the heuristic
    // walks closest first and the list is full before it reaches `q`), so
    // the re-prune is skipped. It differs only for lists that grew by
    // plain appends, whose re-prune could drop a redundant neighbour and
    // keep `q`. Measured on NYTimes-256: the build took 50 s instead of
    // 58 s, with recall@10 at ef 64 and 256 unchanged (0.858, 0.917).
    let dq = cands[cands.len() - 1].1;
    if cands[..cands.len() - 1].iter().all(|&(_, d)| d <= dq) {
      return;
    }
    Self::sort_cands(&mut cands);
    let mut keep = Vec::with_capacity(cap);
    self.select(&cands, cap, &mut keep);
    if level == 0 {
      self.keep_protected(e, &cands, &mut keep, cap);
    }
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
