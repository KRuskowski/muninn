//! Word-level bitset used while evaluating, converted to a
//! [`muninn_core::BitSet`] (without copying) at the end.

use muninn_core::{BitSet, DocId};

/// A set of ids `0..len`. Ids at or beyond `len` are silently ignored by
/// every insertion, which is how postings are truncated to `num_docs`.
#[derive(Clone, Debug)]
pub(crate) struct Bits {
  words: Vec<u64>,
  len: usize,
}

impl Bits {
  pub(crate) fn new(len: usize) -> Self {
    Self {
      words: vec![0; len.div_ceil(64)],
      len,
    }
  }

  pub(crate) fn full(len: usize) -> Self {
    let mut b = Self {
      words: vec![u64::MAX; len.div_ceil(64)],
      len,
    };
    b.clear_tail();
    b
  }

  #[inline]
  pub(crate) fn set(&mut self, id: DocId) {
    let i = id as usize;
    if i < self.len {
      self.words[i >> 6] |= 1 << (i & 63);
    }
  }

  /// Inserts sorted ids; ids past `len` are cut off with one search.
  pub(crate) fn set_sorted(&mut self, ids: &[DocId]) {
    let len = self.len;
    let end = match ids.last() {
      Some(&last) if (last as usize) < len => ids.len(),
      _ => ids.partition_point(|&d| (d as usize) < len),
    };
    for &d in &ids[..end] {
      let i = d as usize;
      self.words[i >> 6] |= 1 << (i & 63);
    }
  }

  /// ORs in a bitset stored as little-endian `u32` halves of `u64` words.
  pub(crate) fn or_words32(&mut self, words: &[u32]) {
    for (w, pair) in self.words.iter_mut().zip(words.chunks_exact(2)) {
      *w |= u64::from(pair[0]) | (u64::from(pair[1]) << 32);
    }
    self.clear_tail();
  }

  pub(crate) fn or_with(&mut self, other: &Bits) {
    for (w, o) in self.words.iter_mut().zip(&other.words) {
      *w |= o;
    }
  }

  pub(crate) fn and_with(&mut self, other: &Bits) {
    for (w, o) in self.words.iter_mut().zip(&other.words) {
      *w &= o;
    }
  }

  pub(crate) fn and_not(&mut self, other: &Bits) {
    for (w, o) in self.words.iter_mut().zip(&other.words) {
      *w &= !o;
    }
  }

  pub(crate) fn negate(&mut self) {
    for w in &mut self.words {
      *w = !*w;
    }
    self.clear_tail();
  }

  pub(crate) fn is_empty(&self) -> bool {
    self.words.iter().all(|&w| w == 0)
  }

  pub(crate) fn is_full(&self) -> bool {
    let full = self.len / 64;
    self.words[..full].iter().all(|&w| w == u64::MAX)
      && (self.len % 64 == 0
        || self.words[full] == (1u64 << (self.len % 64)) - 1)
  }

  pub(crate) fn into_bitset(self) -> BitSet {
    BitSet::from_words(self.words, self.len)
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

#[cfg(test)]
mod tests {
  use super::*;

  fn ids(b: Bits) -> Vec<DocId> {
    b.into_bitset().iter().collect()
  }

  #[test]
  fn truncates_and_combines() {
    let mut a = Bits::new(70);
    a.set_sorted(&[1, 5, 69, 70, 200]);
    a.set(100);
    assert_eq!(ids(a.clone()), [1, 5, 69]);
    let mut b = Bits::new(70);
    b.or_words32(&[0b10, 0, u32::MAX, 0]);
    assert_eq!(ids(b.clone()), [1, 64, 65, 66, 67, 68, 69]);
    let mut c = a.clone();
    c.and_with(&b);
    assert_eq!(ids(c), [1, 69]);
    let mut c = a.clone();
    c.and_not(&b);
    assert_eq!(ids(c), [5]);
    let mut c = a.clone();
    c.or_with(&b);
    assert_eq!(ids(c).len(), 8);
    let mut n = a;
    n.negate();
    assert_eq!(n.clone().into_bitset().count(), 67);
    assert!(!n.is_full() && !n.is_empty());
    assert!(Bits::full(70).is_full() && Bits::full(64).is_full());
    assert!(Bits::new(0).is_full() && Bits::new(0).is_empty());
    assert!(Bits::new(5).is_empty() && !Bits::new(5).is_full());
  }
}
