use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::DocId;

/// Decides whether a document may appear in results.
///
/// Indexes call this inside their search loops, so implementations must be
/// cheap. [`BitSet`] is the canonical implementation; closures work too.
pub trait DocFilter: Sync {
  fn matches(&self, id: DocId) -> bool;

  /// Rough fraction of documents that pass, if known. Indexes may use it
  /// to switch strategy (for example brute force over a tiny allow-list
  /// instead of graph traversal). Called once or twice per search, so it
  /// should not scan the whole id range each time.
  fn selectivity_hint(&self) -> Option<f32> {
    None
  }

  /// A set holding every id that passes (it may hold more), if the filter
  /// has one at hand. Exact scans over a selective filter then visit only
  /// these ids, still checking each with [`DocFilter::matches`], instead of
  /// testing every id in the index.
  fn candidates(&self) -> Option<&BitSet> {
    None
  }
}

impl<F: Fn(DocId) -> bool + Sync> DocFilter for F {
  #[inline]
  fn matches(&self, id: DocId) -> bool {
    self(id)
  }
}

/// Marks [`BitSet::count`] as not computed yet.
const UNCOUNTED: usize = usize::MAX;

/// A fixed-size set of document ids backed by `u64` words.
///
/// The population count is cached: [`BitSet::count`] (and so
/// [`DocFilter::selectivity_hint`]) scans the words once, and later calls
/// are free until the set changes. Single-id changes keep the cache
/// up to date; bulk operations reset it.
#[derive(Default)]
pub struct BitSet {
  words: Vec<u64>,
  len: usize,
  /// Number of set bits, or [`UNCOUNTED`]. Atomic only so that `count`
  /// can fill it through `&self` while the set is shared between threads.
  count: AtomicUsize,
}

impl Clone for BitSet {
  fn clone(&self) -> Self {
    Self {
      words: self.words.clone(),
      len: self.len,
      count: AtomicUsize::new(self.count.load(Relaxed)),
    }
  }
}

impl PartialEq for BitSet {
  fn eq(&self, other: &Self) -> bool {
    self.len == other.len && self.words == other.words
  }
}

impl Eq for BitSet {}

impl std::fmt::Debug for BitSet {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("BitSet")
      .field("words", &self.words)
      .field("len", &self.len)
      .finish()
  }
}

impl BitSet {
  /// An empty set able to hold ids `0..len`.
  pub fn new(len: usize) -> Self {
    Self {
      words: vec![0; len.div_ceil(64)],
      len,
      count: AtomicUsize::new(0),
    }
  }

  /// A set holding every id `0..len`.
  pub fn full(len: usize) -> Self {
    let mut s = Self {
      words: vec![u64::MAX; len.div_ceil(64)],
      len,
      count: AtomicUsize::new(len),
    };
    s.clear_tail();
    s
  }

  /// Number of ids the set can hold.
  #[inline]
  pub fn capacity(&self) -> usize {
    self.len
  }

  /// Grows the set so it can hold ids `0..len`. Never shrinks.
  pub fn grow(&mut self, len: usize) {
    if len > self.len {
      self.words.resize(len.div_ceil(64), 0);
      self.len = len;
    }
  }

  #[inline]
  pub fn insert(&mut self, id: DocId) {
    let i = id as usize;
    assert!(i < self.len, "id {i} out of range {}", self.len);
    let w = &mut self.words[i >> 6];
    let bit = 1 << (i & 63);
    let added = *w & bit == 0;
    *w |= bit;
    let c = self.count.get_mut();
    if added && *c != UNCOUNTED {
      *c += 1;
    }
  }

  #[inline]
  pub fn remove(&mut self, id: DocId) {
    let i = id as usize;
    if i < self.len {
      let w = &mut self.words[i >> 6];
      let bit = 1 << (i & 63);
      let removed = *w & bit != 0;
      *w &= !bit;
      let c = self.count.get_mut();
      if removed && *c != UNCOUNTED {
        *c -= 1;
      }
    }
  }

  #[inline]
  pub fn contains(&self, id: DocId) -> bool {
    let i = id as usize;
    i < self.len && (self.words[i >> 6] >> (i & 63)) & 1 == 1
  }

  /// Number of ids in the set. Counted on the first call after a bulk
  /// change, cached otherwise.
  pub fn count(&self) -> usize {
    let c = self.count.load(Relaxed);
    if c != UNCOUNTED {
      return c;
    }
    let c = self.words.iter().map(|w| w.count_ones() as usize).sum();
    // Racing callers store the same value.
    self.count.store(c, Relaxed);
    c
  }

  /// Keeps only ids present in both sets.
  pub fn intersect_with(&mut self, other: &BitSet) {
    for (i, w) in self.words.iter_mut().enumerate() {
      *w &= other.words.get(i).copied().unwrap_or(0);
    }
    *self.count.get_mut() = UNCOUNTED;
  }

  /// Adds every id present in `other`.
  pub fn union_with(&mut self, other: &BitSet) {
    self.grow(other.len);
    for (w, o) in self.words.iter_mut().zip(&other.words) {
      *w |= o;
    }
    *self.count.get_mut() = UNCOUNTED;
  }

  /// Removes every id present in `other`.
  pub fn difference_with(&mut self, other: &BitSet) {
    for (w, o) in self.words.iter_mut().zip(&other.words) {
      *w &= !o;
    }
    *self.count.get_mut() = UNCOUNTED;
  }

  /// Iterates set ids in ascending order.
  pub fn iter(&self) -> impl Iterator<Item = DocId> + '_ {
    self.words.iter().enumerate().flat_map(|(wi, &w)| {
      let mut w = w;
      std::iter::from_fn(move || {
        if w == 0 {
          return None;
        }
        let bit = w.trailing_zeros();
        w &= w - 1;
        Some((wi as u32) * 64 + bit)
      })
    })
  }

  /// The raw words, for serialisation.
  pub fn words(&self) -> &[u64] {
    &self.words
  }

  /// Rebuilds a set from [`BitSet::words`].
  pub fn from_words(words: Vec<u64>, len: usize) -> Self {
    let mut s = Self {
      words,
      len,
      count: AtomicUsize::new(UNCOUNTED),
    };
    s.words.resize(len.div_ceil(64), 0);
    s.clear_tail();
    s
  }

  fn clear_tail(&mut self) {
    let rem = self.len % 64;
    if rem != 0
      && let Some(last) = self.words.last_mut()
    {
      *last &= (1u64 << rem) - 1;
    }
  }
}

impl DocFilter for BitSet {
  #[inline]
  fn matches(&self, id: DocId) -> bool {
    self.contains(id)
  }

  fn selectivity_hint(&self) -> Option<f32> {
    (self.len > 0).then(|| self.count() as f32 / self.len as f32)
  }

  fn candidates(&self) -> Option<&BitSet> {
    Some(self)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn insert_contains_iter() {
    let mut s = BitSet::new(130);
    for id in [0, 63, 64, 129] {
      s.insert(id);
    }
    assert!(s.contains(64) && !s.contains(65) && !s.contains(1000));
    assert_eq!(s.iter().collect::<Vec<_>>(), vec![0, 63, 64, 129]);
    assert_eq!(s.count(), 4);
    s.remove(63);
    assert_eq!(s.count(), 3);
  }

  #[test]
  fn full_respects_len() {
    let s = BitSet::full(70);
    assert_eq!(s.count(), 70);
    assert_eq!(s.selectivity_hint(), Some(1.0));
  }

  #[test]
  fn set_algebra() {
    let mut a = BitSet::new(10);
    let mut b = BitSet::new(10);
    a.insert(1);
    a.insert(2);
    b.insert(2);
    b.insert(3);
    let mut i = a.clone();
    i.intersect_with(&b);
    assert_eq!(i.iter().collect::<Vec<_>>(), vec![2]);
    let mut u = a.clone();
    u.union_with(&b);
    assert_eq!(u.iter().collect::<Vec<_>>(), vec![1, 2, 3]);
    a.difference_with(&b);
    assert_eq!(a.iter().collect::<Vec<_>>(), vec![1]);
  }

  /// The cached count follows every kind of change.
  #[test]
  fn cached_count_tracks_changes() {
    let naive = |s: &BitSet| s.iter().count();
    let mut a = BitSet::new(200);
    assert_eq!(a.count(), 0);
    for id in [3, 3, 64, 199] {
      a.insert(id);
    }
    assert_eq!(a.count(), 3);
    a.remove(3);
    a.remove(3);
    a.remove(500);
    assert_eq!(a.count(), 2);
    let mut b = BitSet::full(150);
    assert_eq!(b.count(), 150);
    b.intersect_with(&a);
    assert_eq!(b.count(), naive(&b));
    b.union_with(&BitSet::full(10));
    assert_eq!(b.count(), naive(&b));
    b.insert(140);
    assert_eq!(b.count(), naive(&b));
    b.difference_with(&a);
    assert_eq!(b.count(), naive(&b));
    let c = BitSet::from_words(vec![u64::MAX, 7], 66);
    assert_eq!(c.count(), 66);
    let mut d = c.clone();
    d.remove(0);
    assert_eq!((c.count(), d.count()), (66, 65));
    d.grow(1000);
    d.insert(999);
    assert_eq!(d.count(), naive(&d));
    assert_eq!(d.selectivity_hint(), Some(66.0 / 1000.0));
    // The cache is not part of equality.
    let mut e = BitSet::new(66);
    e.union_with(&c);
    assert_eq!(e, c);
    assert!(std::ptr::eq(c.candidates().unwrap(), &c));
  }

  #[test]
  fn closures_are_filters() {
    let even = |id: DocId| id.is_multiple_of(2);
    assert!(even.matches(4) && !even.matches(5));
  }
}
