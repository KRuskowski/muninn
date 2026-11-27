use crate::{DocId, Hit, Metric};

/// Keeps the `k` candidates with the smallest distance seen so far.
///
/// A hand-rolled binary max-heap on `(distance, id)`: the root is the
/// current worst kept candidate, so rejecting a candidate is one compare
/// against [`TopK::threshold`]. Ties on distance are broken by the
/// smaller id, which makes results deterministic across thread counts and
/// segment orders. NaN distances are never kept.
#[derive(Clone, Debug)]
pub struct TopK {
  k: usize,
  heap: Vec<(f32, DocId)>,
}

#[inline]
fn worse(a: (f32, DocId), b: (f32, DocId)) -> bool {
  // `a` ranks after `b`.
  a.0 > b.0 || (a.0 == b.0 && a.1 > b.1)
}

impl TopK {
  /// Creates an empty collector for `k` results.
  pub fn new(k: usize) -> Self {
    Self {
      k,
      heap: Vec::with_capacity(k),
    }
  }

  /// The configured `k`.
  #[inline]
  pub fn k(&self) -> usize {
    self.k
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.heap.len()
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.heap.is_empty()
  }

  #[inline]
  pub fn is_full(&self) -> bool {
    self.heap.len() >= self.k
  }

  /// The distance a candidate must beat to be kept: the worst kept
  /// distance once full, `f32::INFINITY` before that.
  #[inline]
  pub fn threshold(&self) -> f32 {
    if self.is_full() && self.k > 0 {
      self.heap[0].0
    } else {
      f32::INFINITY
    }
  }

  /// Offers a candidate. Returns whether it was kept.
  #[inline]
  pub fn push(&mut self, id: DocId, distance: f32) -> bool {
    if self.k == 0 || distance.is_nan() {
      return false;
    }
    let item = (distance, id);
    if self.heap.len() < self.k {
      self.heap.push(item);
      self.sift_up(self.heap.len() - 1);
      return true;
    }
    if !worse(self.heap[0], item) {
      return false;
    }
    self.heap[0] = item;
    self.sift_down(0);
    true
  }

  /// Merges another collector into this one.
  pub fn merge(&mut self, other: TopK) {
    for (d, id) in other.heap {
      self.push(id, d);
    }
  }

  /// Iterates kept candidates in no particular order.
  pub fn iter(&self) -> impl Iterator<Item = (DocId, f32)> + '_ {
    self.heap.iter().map(|&(d, id)| (id, d))
  }

  /// Consumes the collector and returns `(id, distance)` pairs sorted from
  /// closest to farthest.
  pub fn into_sorted_vec(self) -> Vec<(DocId, f32)> {
    let mut v = self.heap;
    v.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    v.into_iter().map(|(d, id)| (id, d)).collect()
  }

  /// Consumes the collector and returns hits, best first, with distances
  /// converted to public scores.
  pub fn into_hits(self, metric: Metric) -> Vec<Hit> {
    self
      .into_sorted_vec()
      .into_iter()
      .map(|(id, d)| Hit::new(id, metric.score(d)))
      .collect()
  }

  fn sift_up(&mut self, mut i: usize) {
    while i > 0 {
      let parent = (i - 1) / 2;
      if worse(self.heap[i], self.heap[parent]) {
        self.heap.swap(i, parent);
        i = parent;
      } else {
        break;
      }
    }
  }

  fn sift_down(&mut self, mut i: usize) {
    let n = self.heap.len();
    loop {
      let l = 2 * i + 1;
      if l >= n {
        break;
      }
      let r = l + 1;
      let mut c = l;
      if r < n && worse(self.heap[r], self.heap[l]) {
        c = r;
      }
      if worse(self.heap[c], self.heap[i]) {
        self.heap.swap(c, i);
        i = c;
      } else {
        break;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn keeps_smallest_k_sorted() {
    let mut t = TopK::new(3);
    for (id, d) in [(0, 5.0), (1, 1.0), (2, 4.0), (3, 0.5), (4, 9.0)] {
      t.push(id, d);
    }
    assert_eq!(t.threshold(), 4.0);
    let v = t.into_sorted_vec();
    assert_eq!(v, vec![(3, 0.5), (1, 1.0), (2, 4.0)]);
  }

  #[test]
  fn ties_prefer_smaller_id_and_nan_is_rejected() {
    let mut t = TopK::new(2);
    assert!(!t.push(7, f32::NAN));
    t.push(5, 1.0);
    t.push(3, 1.0);
    t.push(4, 1.0);
    assert_eq!(t.into_sorted_vec(), vec![(3, 1.0), (4, 1.0)]);
  }

  #[test]
  fn zero_k_keeps_nothing() {
    let mut t = TopK::new(0);
    assert!(!t.push(1, 0.0));
    assert!(t.is_empty());
    assert_eq!(t.threshold(), f32::INFINITY);
  }

  #[test]
  fn merge_matches_single_pass() {
    let data: Vec<(u32, f32)> =
      (0..100).map(|i| (i, ((i * 37) % 101) as f32)).collect();
    let mut one = TopK::new(10);
    let (mut a, mut b) = (TopK::new(10), TopK::new(10));
    for &(id, d) in &data {
      one.push(id, d);
      if id % 2 == 0 {
        a.push(id, d)
      } else {
        b.push(id, d)
      };
    }
    a.merge(b);
    assert_eq!(a.into_sorted_vec(), one.into_sorted_vec());
  }
}
