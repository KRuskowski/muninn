//! Node order: renumbering a built graph for memory locality, and the id
//! map that keeps callers' ids unchanged.
//!
//! A graph search scores the unvisited neighbours of each node it
//! expands. In insertion order those neighbours are scattered over the
//! whole vector array, so every one costs its own cache and TLB misses.
//! After a bulk build the nodes are renumbered breadth-first from the
//! entry point over level 0: a node's neighbours then get consecutive
//! numbers, so their vectors, codes and lists sit next to each other.
//!
//! Callers keep using the ids they inserted with (`DocId`s are the
//! engine's document ids): [`IdMap`] translates between the two orders at
//! the edges of the index (filters, results, `vector(id)`). Nodes added
//! after the renumbering keep equal internal and external ids.

use muninn_core::{DocFilter, DocId};

use crate::slab::Slab;

/// Internal (graph) order to external (caller) ids and back. Both empty
/// means the two orders are the same. Lookups never trust the data: an
/// out-of-range entry (only a corrupt file has one) maps to the id
/// itself, which gives wrong answers but no panic.
#[derive(Clone, Debug, Default)]
pub(crate) struct IdMap {
  /// `ext[internal]`.
  pub ext: Slab<u32>,
  /// `int[external]`.
  pub int: Slab<u32>,
}

impl IdMap {
  /// Whether internal and external ids are the same.
  #[inline]
  pub(crate) fn is_identity(&self) -> bool {
    self.ext.len() == 0
  }

  /// The external id of internal node `i`.
  #[inline]
  pub(crate) fn ext(&self, i: DocId) -> DocId {
    lookup(self.ext.as_slice(), i)
  }

  /// The internal node of external id `e`.
  #[inline]
  pub(crate) fn int(&self, e: DocId) -> DocId {
    lookup(self.int.as_slice(), e)
  }

  /// Extends a non-identity map for nodes `first..end` appended with equal
  /// internal and external ids.
  pub(crate) fn push_identity(&mut self, first: usize, end: usize) {
    if self.is_identity() {
      return;
    }
    let ids = first as DocId..end as DocId;
    self.ext.to_mut().extend(ids.clone());
    self.int.to_mut().extend(ids);
  }

  /// Composes the map with a renumbering of the internal order: new node
  /// `i` is old node `order[i]`.
  pub(crate) fn renumber(&mut self, order: &[u32]) {
    let n = order.len();
    let ext: Vec<u32> = order.iter().map(|&o| self.ext(o)).collect();
    let mut int = vec![0u32; n];
    for (i, &e) in ext.iter().enumerate() {
      if let Some(slot) = int.get_mut(e as usize) {
        *slot = i as u32;
      }
    }
    self.ext = Slab::owned(ext);
    self.int = Slab::owned(int);
  }
}

#[inline]
pub(crate) fn lookup(map: &[u32], id: DocId) -> DocId {
  map.get(id as usize).copied().unwrap_or(id)
}

/// A caller's filter seen from the graph's internal order.
pub(crate) struct Mapped<'a> {
  pub filter: &'a dyn DocFilter,
  pub ext: &'a [u32],
}

impl DocFilter for Mapped<'_> {
  #[inline]
  fn matches(&self, id: DocId) -> bool {
    self.filter.matches(lookup(self.ext, id))
  }

  fn selectivity_hint(&self) -> Option<f32> {
    self.filter.selectivity_hint()
  }
}

/// Breadth-first order of level 0 from `entry`: `order[new] = old`.
/// `neighbors(v, out)` fills `out` with `v`'s list. Nodes the walk cannot
/// reach (none in a healthy graph) follow in id order.
pub(crate) fn bfs_order(
  n: usize,
  entry: DocId,
  mut neighbors: impl FnMut(DocId, &mut Vec<u32>),
) -> Vec<u32> {
  let mut order = Vec::with_capacity(n);
  if n == 0 {
    return order;
  }
  let mut seen = vec![false; n];
  let mut buf = Vec::new();
  let mut next_unseen = 0;
  let mut head = 0;
  let mut start = (entry as usize).min(n - 1);
  loop {
    seen[start] = true;
    order.push(start as u32);
    while head < order.len() {
      let v = order[head];
      head += 1;
      neighbors(v, &mut buf);
      for &y in &buf {
        if (y as usize) < n && !seen[y as usize] {
          seen[y as usize] = true;
          order.push(y);
        }
      }
    }
    while next_unseen < n && seen[next_unseen] {
      next_unseen += 1;
    }
    if next_unseen == n {
      return order;
    }
    start = next_unseen;
  }
}

/// Reorders `data`, rows of `width` values, in place so that new row `i`
/// is old row `order[i]`. `order` must be a permutation of the row
/// indexes. Follows the permutation's cycles with one spare row, so no
/// second copy of the array is ever held.
pub(crate) fn permute_rows<T: Copy + Default>(
  data: &mut [T],
  width: usize,
  order: &[u32],
) {
  let n = order.len();
  if width == 0 || n == 0 {
    return;
  }
  debug_assert_eq!(data.len(), n * width);
  let mut done = vec![false; n];
  let mut tmp = vec![T::default(); width];
  for s in 0..n {
    if done[s] || order[s] as usize == s {
      done[s] = true;
      continue;
    }
    tmp.copy_from_slice(&data[s * width..(s + 1) * width]);
    let mut i = s;
    loop {
      done[i] = true;
      let j = order[i] as usize;
      if j == s {
        data[i * width..(i + 1) * width].copy_from_slice(&tmp);
        break;
      }
      data.copy_within(j * width..(j + 1) * width, i * width);
      i = j;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn permute_rows_gathers_in_place() {
    let order = [3u32, 0, 4, 1, 2, 5];
    let mut data: Vec<u32> = (0..12).collect();
    permute_rows(&mut data, 2, &order);
    assert_eq!(data, vec![6, 7, 0, 1, 8, 9, 2, 3, 4, 5, 10, 11]);
  }

  #[test]
  fn bfs_visits_everything_once() {
    // 0 - 1 - 2, 3 isolated, 4 -> 0.
    let adj: Vec<Vec<u32>> =
      vec![vec![1], vec![0, 2, 9], vec![1], vec![], vec![0]];
    let order = bfs_order(5, 2, |v, out| {
      out.clear();
      out.extend(&adj[v as usize]);
    });
    assert_eq!(order, vec![2, 1, 0, 3, 4]);
  }

  #[test]
  fn id_map_round_trips() {
    let mut m = IdMap::default();
    assert!(m.is_identity());
    assert_eq!((m.ext(5), m.int(5)), (5, 5));
    m.renumber(&[2, 0, 1]);
    assert_eq!((m.ext(0), m.ext(1), m.ext(2)), (2, 0, 1));
    assert_eq!((m.int(2), m.int(0), m.int(1)), (0, 1, 2));
    m.renumber(&[1, 2, 0]);
    assert_eq!((m.ext(0), m.ext(1), m.ext(2)), (0, 1, 2));
    m.push_identity(3, 5);
    assert_eq!((m.ext(4), m.int(3)), (4, 3));
  }
}
