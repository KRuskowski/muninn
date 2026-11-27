use crate::DocId;

/// Decides whether a document may appear in results.
///
/// Indexes call this inside their search loops, so implementations must be
/// cheap. [`BitSet`] is the canonical implementation; closures work too.
pub trait DocFilter: Sync {
  fn matches(&self, id: DocId) -> bool;

  /// Rough fraction of documents that pass, if known. Indexes may use it
  /// to switch strategy (for example brute force over a tiny allow-list
  /// instead of graph traversal).
  fn selectivity_hint(&self) -> Option<f32> {
    None
  }
}

impl<F: Fn(DocId) -> bool + Sync> DocFilter for F {
  #[inline]
  fn matches(&self, id: DocId) -> bool {
    self(id)
  }
}

/// A fixed-size set of document ids backed by `u64` words.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitSet {
  words: Vec<u64>,
  len: usize,
}

impl BitSet {
  /// An empty set able to hold ids `0..len`.
  pub fn new(len: usize) -> Self {
    Self {
      words: vec![0; len.div_ceil(64)],
      len,
    }
  }

  /// A set holding every id `0..len`.
  pub fn full(len: usize) -> Self {
    let mut s = Self {
      words: vec![u64::MAX; len.div_ceil(64)],
      len,
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
    self.words[i >> 6] |= 1 << (i & 63);
  }

  #[inline]
  pub fn remove(&mut self, id: DocId) {
    let i = id as usize;
    if i < self.len {
      self.words[i >> 6] &= !(1 << (i & 63));
    }
  }

  #[inline]
  pub fn contains(&self, id: DocId) -> bool {
    let i = id as usize;
    i < self.len && (self.words[i >> 6] >> (i & 63)) & 1 == 1
  }

  /// Number of ids in the set.
  pub fn count(&self) -> usize {
    self.words.iter().map(|w| w.count_ones() as usize).sum()
  }

  /// Keeps only ids present in both sets.
  pub fn intersect_with(&mut self, other: &BitSet) {
    for (i, w) in self.words.iter_mut().enumerate() {
      *w &= other.words.get(i).copied().unwrap_or(0);
    }
  }

  /// Adds every id present in `other`.
  pub fn union_with(&mut self, other: &BitSet) {
    self.grow(other.len);
    for (w, o) in self.words.iter_mut().zip(&other.words) {
      *w |= o;
    }
  }

  /// Removes every id present in `other`.
  pub fn difference_with(&mut self, other: &BitSet) {
    for (w, o) in self.words.iter_mut().zip(&other.words) {
      *w &= !o;
    }
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
    let mut s = Self { words, len };
    s.words.resize(len.div_ceil(64), 0);
    s.clear_tail();
    s
  }

  fn clear_tail(&mut self) {
    let rem = self.len % 64;
    if rem != 0 {
      if let Some(last) = self.words.last_mut() {
        *last &= (1u64 << rem) - 1;
      }
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

  #[test]
  fn closures_are_filters() {
    let even = |id: DocId| id % 2 == 0;
    assert!(even.matches(4) && !even.matches(5));
  }
}
