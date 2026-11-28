//! HNSW graph storage.
//!
//! - **Level 0**: one `u32` array with a fixed-width record per node,
//!   `[count, slot_0 .. slot_{M0-1}]`. Fixed width means node `i`'s list
//!   is at `i * (1 + M0)`, no index needed, and the whole array is one
//!   container section read in place.
//! - **Upper levels** (about `1/M` of nodes): `upoff[i]` is the offset of
//!   node `i`'s first upper record in `upper` (or [`NONE`]); its record
//!   for level `l >= 1` is `[count, slot_0 .. slot_{M-1}]` at
//!   `upoff[i] + (l - 1) * (1 + M)`.
//!
//! Reads never trust the data: counts are clamped and out-of-range
//! offsets yield empty lists, so a corrupt file gives bad results rather
//! than a panic. Search additionally skips neighbour ids `>= n`.
//!
//! During a parallel build the same arrays are viewed as `AtomicU32` and
//! each node's records are guarded by a striped lock ([`BuildView`]).

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, PoisonError};

use muninn_core::DocId;

use crate::dist::prefetch_lines;
use crate::slab::Slab;

/// "No node" / "no upper levels".
pub(crate) const NONE: u32 = u32::MAX;

/// Neighbour lists of one graph, read during search or build.
pub(crate) trait Neighbors: Sync {
  /// Number of nodes.
  fn n(&self) -> usize;

  /// Node `node`'s neighbours at `level`. Implementations either return
  /// a slice of their own storage or copy into `buf` and return that.
  fn neighbors<'a>(
    &'a self,
    node: DocId,
    level: usize,
    buf: &'a mut Vec<u32>,
  ) -> &'a [u32];

  /// Hints that `node`'s list at `level` will be read soon. Never faults.
  #[inline]
  fn prefetch(&self, _node: DocId, _level: usize) {}
}

/// Owned or mapped graph arrays.
#[derive(Clone, Debug, Default)]
pub(crate) struct Graph {
  pub levels: Slab<u8>,
  pub l0: Slab<u32>,
  pub upoff: Slab<u32>,
  pub upper: Slab<u32>,
}

impl Graph {
  /// A read-only view for search.
  pub(crate) fn view(&self, m: usize) -> GraphView<'_> {
    GraphView {
      m,
      m0: 2 * m,
      levels: self.levels.as_slice(),
      l0: self.l0.as_slice(),
      upoff: self.upoff.as_slice(),
      upper: self.upper.as_slice(),
    }
  }

  /// Bytes held by the graph arrays.
  pub(crate) fn bytes(&self) -> usize {
    self.levels.bytes()
      + self.l0.bytes()
      + self.upoff.bytes()
      + self.upper.bytes()
  }
}

/// Plain slices of a frozen graph.
#[derive(Clone, Copy)]
pub(crate) struct GraphView<'a> {
  pub m: usize,
  pub m0: usize,
  pub levels: &'a [u8],
  pub l0: &'a [u32],
  pub upoff: &'a [u32],
  pub upper: &'a [u32],
}

/// Where node `node`'s record for `level >= 1` starts in `upper`.
#[inline]
pub(crate) fn upper_base(
  levels: &[u8],
  upoff: &[u32],
  m: usize,
  node: usize,
  level: usize,
) -> Option<usize> {
  let off = *upoff.get(node)?;
  if off == NONE || level > *levels.get(node)? as usize {
    return None;
  }
  Some(off as usize + (level - 1) * (1 + m))
}

#[inline]
fn record(arr: &[u32], base: usize, cap: usize) -> &[u32] {
  match arr.get(base) {
    Some(&c) => {
      let c = (c as usize).min(cap);
      arr.get(base + 1..base + 1 + c).unwrap_or(&[])
    }
    None => &[],
  }
}

impl Neighbors for GraphView<'_> {
  #[inline]
  fn n(&self) -> usize {
    self.levels.len()
  }

  #[inline]
  fn neighbors<'a>(
    &'a self,
    node: DocId,
    level: usize,
    _buf: &'a mut Vec<u32>,
  ) -> &'a [u32] {
    let node = node as usize;
    if level == 0 {
      record(self.l0, node * (1 + self.m0), self.m0)
    } else {
      match upper_base(self.levels, self.upoff, self.m, node, level) {
        Some(b) => record(self.upper, b, self.m),
        None => &[],
      }
    }
  }

  #[inline]
  fn prefetch(&self, node: DocId, level: usize) {
    if level == 0 {
      let w = 1 + self.m0;
      prefetch_lines(self.l0.as_ptr(), node as usize * w, w);
    }
  }
}

/// Views an exclusively borrowed `u32` slice as atomics.
pub(crate) fn as_atomic(s: &mut [u32]) -> &[AtomicU32] {
  const _: () = assert!(
    std::mem::size_of::<AtomicU32>() == std::mem::size_of::<u32>()
      && std::mem::align_of::<AtomicU32>() == std::mem::align_of::<u32>()
  );
  // SAFETY: AtomicU32 has the same size and bit validity as u32, and the
  // same alignment (asserted above). `s` is exclusively borrowed for the
  // returned lifetime, so no non-atomic access can race with it.
  unsafe { &*(s as *mut [u32] as *const [AtomicU32]) }
}

/// The graph during a (possibly parallel) build: atomics plus striped
/// per-node locks. A thread holds at most one lock at a time, so stripes
/// shared by several nodes cannot deadlock.
pub(crate) struct BuildView<'a> {
  pub m: usize,
  pub m0: usize,
  pub levels: &'a [u8],
  pub upoff: &'a [u32],
  pub l0: &'a [AtomicU32],
  pub upper: &'a [AtomicU32],
  pub locks: &'a [Mutex<()>],
}

impl BuildView<'_> {
  /// Locks node `node`'s records.
  #[inline]
  pub(crate) fn lock(&self, node: DocId) -> MutexGuard<'_, ()> {
    let i = node as usize & (self.locks.len() - 1);
    self.locks[i].lock().unwrap_or_else(PoisonError::into_inner)
  }

  /// Node `node`'s record at `level` (count then slots) and its capacity.
  /// Callers hold the node's lock. Levels above the node's own, and
  /// records outside the arrays, are `None`.
  #[inline]
  pub(crate) fn record(
    &self,
    node: DocId,
    level: usize,
  ) -> Option<(&[AtomicU32], usize)> {
    let node = node as usize;
    if level == 0 {
      let b = node.checked_mul(1 + self.m0)?;
      Some((self.l0.get(b..b.checked_add(1 + self.m0)?)?, self.m0))
    } else {
      let b = upper_base(self.levels, self.upoff, self.m, node, level)?;
      Some((self.upper.get(b..b.checked_add(1 + self.m)?)?, self.m))
    }
  }
}

/// Writes `ids` (truncated to `cap`) into a locked record.
#[inline]
pub(crate) fn write_record(rec: &[AtomicU32], cap: usize, ids: &[u32]) {
  let c = ids.len().min(cap);
  for (slot, &id) in rec[1..].iter().zip(&ids[..c]) {
    slot.store(id, Relaxed);
  }
  rec[0].store(c as u32, Relaxed);
}

/// The ids currently in a locked record, without any `>= n` (which only
/// a corrupt file can contain).
#[inline]
pub(crate) fn read_record(
  rec: &[AtomicU32],
  cap: usize,
  n: usize,
  out: &mut Vec<u32>,
) {
  let c = (rec[0].load(Relaxed) as usize).min(cap);
  out.clear();
  out.extend(
    rec[1..1 + c]
      .iter()
      .map(|a| a.load(Relaxed))
      .filter(|&id| (id as usize) < n),
  );
}

impl Neighbors for BuildView<'_> {
  #[inline]
  fn n(&self) -> usize {
    self.levels.len()
  }

  /// Reads the list without taking the node's lock (writers still lock
  /// each other out). A concurrent rewrite can be seen half done: the
  /// snapshot may mix ids of the old and the new list, or repeat one, but
  /// every id in it is a real node (slots only ever hold node ids, and
  /// `read_record` drops anything out of range), which is all a search
  /// needs: it skips repeats and treats the list as a hint. hnswlib locks
  /// here; faiss does not. Skipping the lock took a tenth off build time.
  #[inline]
  fn neighbors<'a>(
    &'a self,
    node: DocId,
    level: usize,
    buf: &'a mut Vec<u32>,
  ) -> &'a [u32] {
    match self.record(node, level) {
      Some((rec, cap)) => read_record(rec, cap, self.levels.len(), buf),
      None => buf.clear(),
    }
    buf
  }

  #[inline]
  fn prefetch(&self, node: DocId, level: usize) {
    if level == 0 {
      let w = 1 + self.m0;
      prefetch_lines(self.l0.as_ptr(), node as usize * w, w);
    }
  }
}
