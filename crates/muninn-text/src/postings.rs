//! Posting lists: the block encoder and the block cursor.
//!
//! A term's postings are split into blocks of 128 documents. A block is
//!
//! ```text
//! u8 doc_bits, u8 tf_bits,
//! doc-id deltas packed at doc_bits  (first delta is from the previous
//!                                     block's last doc, or from 0)
//! tf - 1 packed at tf_bits
//! ```
//!
//! and each block has a [`SkipRec`]: its last doc id, its byte offset in
//! the term's data, and its impacts (see [`crate::impacts`]). Every block
//! holds 128 documents except the term's last one.
//!
//! [`PostingList`] is the live, appendable form used by the in-memory
//! index and by merging: full blocks are packed as soon as they fill, the
//! open block is a varint buffer. Freezing writes the packed blocks
//! verbatim plus the open block packed, so a frozen index built from
//! memory and one built by merging are byte-identical.
//!
//! [`Cursor`] iterates any posting list (in memory or zero-copy from a
//! file) block by block, with shallow skipping over block metadata.

use muninn_core::{DocId, Pod};

use crate::bitpack::{self, BLOCK, packed_len};
use crate::bm25::{LenNorms, bound};
use crate::impacts::{Impacts, frontier};
use crate::varint;

/// Sentinel doc id: the cursor is exhausted. Never a valid document.
pub(crate) const END: DocId = DocId::MAX;

/// Skip data for one block. 32 bytes, stored as an array in the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct SkipRec {
  pub last_doc: u32,
  pub offset: u32,
  pub impacts: Impacts,
}

// SAFETY: `SkipRec` is `repr(C)` with eight `u32` fields (32 bytes,
// alignment 4, no padding), holds no pointers, and every bit pattern is
// a valid value.
unsafe impl Pod for SkipRec {}

/// Packs one block of absolute doc ids (deltas from `base`) and tfs.
pub(crate) fn pack_block(
  base: u32,
  docs: &[u32],
  tfs: &[u32],
  out: &mut Vec<u8>,
) {
  debug_assert!(docs.len() == tfs.len() && docs.len() <= BLOCK);
  let mut deltas = [0u32; BLOCK];
  let mut prev = base;
  let mut max_d = 0;
  for (d, &doc) in deltas.iter_mut().zip(docs) {
    *d = doc - prev;
    max_d |= *d;
    prev = doc;
  }
  let mut tfm = [0u32; BLOCK];
  let mut max_t = 0;
  for (t, &tf) in tfm.iter_mut().zip(tfs) {
    *t = tf - 1;
    max_t |= *t;
  }
  let wd = bitpack::bits_needed(max_d);
  let wt = bitpack::bits_needed(max_t);
  out.push(wd as u8);
  out.push(wt as u8);
  bitpack::pack(&deltas[..docs.len()], wd, out);
  bitpack::pack(&tfm[..docs.len()], wt, out);
}

/// Full (packed) blocks of a live posting list.
#[derive(Clone, Debug, Default)]
pub(crate) struct Closed {
  pub data: Vec<u8>,
  pub skips: Vec<SkipRec>,
}

/// An appendable posting list.
#[derive(Clone, Debug, Default)]
pub(crate) struct PostingList {
  /// Open block: varint pairs (doc delta, tf).
  tail: Vec<u8>,
  closed: Option<Box<Closed>>,
  df: u32,
  last_doc: u32,
  tail_count: u32,
}

/// A posting list's frozen form, borrowed where possible.
pub(crate) struct Frozen<'a> {
  pub df: u32,
  pub last_doc: u32,
  /// Already packed full blocks.
  pub head: &'a [u8],
  /// The open block, packed (empty if there is none).
  pub tail_block: Vec<u8>,
  /// One record per block, `head` blocks first.
  pub skips: Vec<SkipRec>,
}

impl PostingList {
  /// Number of documents.
  #[inline]
  pub fn df(&self) -> u32 {
    self.df
  }

  /// Appends a posting. `doc` must exceed every doc already added and
  /// `norms[doc]` must be its final length norm.
  pub fn push(&mut self, doc: DocId, tf: u32, norms: &[u8]) {
    debug_assert!(self.df == 0 || doc > self.last_doc);
    debug_assert!(tf >= 1);
    let delta = if self.df == 0 {
      doc
    } else {
      doc - self.last_doc
    };
    varint::put(&mut self.tail, u64::from(delta));
    varint::put(&mut self.tail, u64::from(tf));
    self.df += 1;
    self.last_doc = doc;
    self.tail_count += 1;
    if self.tail_count as usize == BLOCK {
      self.close_tail(norms);
    }
  }

  fn closed_skips(&self) -> &[SkipRec] {
    self.closed.as_ref().map_or(&[], |c| &c.skips)
  }

  fn tail_base(&self) -> u32 {
    self.closed_skips().last().map_or(0, |s| s.last_doc)
  }

  /// Decodes the open block into absolute doc ids and tfs.
  fn decode_tail(
    &self,
    docs: &mut [u32; BLOCK],
    tfs: &mut [u32; BLOCK],
  ) -> usize {
    let n = self.tail_count as usize;
    let mut pos = 0;
    let mut doc = self.tail_base();
    for i in 0..n {
      // The tail is written by `push` in this process.
      let d = varint::get_u32(&self.tail, &mut pos).expect("valid tail");
      let tf = varint::get_u32(&self.tail, &mut pos).expect("valid tail");
      doc += d;
      docs[i] = doc;
      tfs[i] = tf;
    }
    n
  }

  fn tail_impacts(&self, docs: &[u32], tfs: &[u32], norms: &[u8]) -> Impacts {
    let pts: Vec<(u32, u8)> = docs
      .iter()
      .zip(tfs)
      .map(|(&d, &tf)| (tf, norms[d as usize]))
      .collect();
    frontier(&pts)
  }

  fn close_tail(&mut self, norms: &[u8]) {
    let mut docs = [0u32; BLOCK];
    let mut tfs = [0u32; BLOCK];
    let n = self.decode_tail(&mut docs, &mut tfs);
    let base = self.tail_base();
    let impacts = self.tail_impacts(&docs[..n], &tfs[..n], norms);
    let closed = self.closed.get_or_insert_with(Default::default);
    let offset = closed.data.len() as u32;
    pack_block(base, &docs[..n], &tfs[..n], &mut closed.data);
    closed.skips.push(SkipRec {
      last_doc: docs[n - 1],
      offset,
      impacts,
    });
    self.tail.clear();
    self.tail.shrink_to(32);
    self.tail_count = 0;
  }

  /// The frozen form, without modifying the list.
  pub fn freeze(&self, norms: &[u8]) -> Frozen<'_> {
    let head: &[u8] = self.closed.as_ref().map_or(&[], |c| &c.data);
    let mut skips = self.closed_skips().to_vec();
    let mut tail_block = Vec::new();
    if self.tail_count > 0 {
      let mut docs = [0u32; BLOCK];
      let mut tfs = [0u32; BLOCK];
      let n = self.decode_tail(&mut docs, &mut tfs);
      let impacts = self.tail_impacts(&docs[..n], &tfs[..n], norms);
      pack_block(self.tail_base(), &docs[..n], &tfs[..n], &mut tail_block);
      skips.push(SkipRec {
        last_doc: self.last_doc,
        offset: head.len() as u32,
        impacts,
      });
    }
    Frozen {
      df: self.df,
      last_doc: self.last_doc,
      head,
      tail_block,
      skips,
    }
  }

  /// A cursor over the list. `norms` covers every doc id in it.
  pub fn cursor<'a>(&'a self, norms: &[u8]) -> Cursor<'a> {
    let (data, skips): (&[u8], &[SkipRec]) = match &self.closed {
      Some(c) => (&c.data, &c.skips),
      None => (&[], &[]),
    };
    let mut c = Cursor::empty(data, skips, self.df, norms.len());
    if self.tail_count > 0 {
      let mut docs = [0u32; BLOCK];
      let mut tfs = [0u32; BLOCK];
      let n = self.decode_tail(&mut docs, &mut tfs);
      c.last = SkipRec {
        last_doc: self.last_doc,
        offset: 0,
        impacts: self.tail_impacts(&docs[..n], &tfs[..n], norms),
      };
      c.last_kind = LastKind::Varint;
      c.tail = &self.tail;
      c.nblocks += 1;
    }
    c.start();
    c
  }

  /// Approximate heap bytes owned.
  pub fn heap_bytes(&self) -> usize {
    self.tail.capacity()
      + self.closed.as_ref().map_or(0, |c| {
        std::mem::size_of::<Closed>()
          + c.data.capacity()
          + c.skips.capacity() * std::mem::size_of::<SkipRec>()
      })
  }
}

/// How the cursor's final block (if not in `skips`) is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LastKind {
  None,
  /// Packed in `data`, described by `last` (single-block frozen terms).
  Packed,
  /// Varint pairs in `tail` (the open block of a live list).
  Varint,
}

/// Iterates one term's postings in doc-id order.
///
/// Positioned on a document (`doc()`), or [`END`]. Block metadata can be
/// skipped without decoding (`shallow_advance`), which is what lets the
/// search skip whole blocks whose score bound cannot reach the top-k.
///
/// Corrupt blocks (only possible in files) never panic: the cursor marks
/// itself `corrupt`, becomes exhausted, and the search reports an error.
pub(crate) struct Cursor<'a> {
  data: &'a [u8],
  skips: &'a [SkipRec],
  last: SkipRec,
  last_kind: LastKind,
  tail: &'a [u8],
  df: u32,
  nblocks: usize,
  num_docs: usize,
  /// Block the cursor is (shallowly) in.
  block: usize,
  /// Block whose docs are in `docs`, or `usize::MAX`.
  decoded: usize,
  pos: usize,
  len: usize,
  doc: DocId,
  tf_off: usize,
  tf_bits: u32,
  tfs_ready: bool,
  bound_block: usize,
  bound: f32,
  /// Query weight `idf * (k1 + 1)`.
  pub weight: f32,
  /// Position of the term in the query, for canonical summation.
  pub idx: usize,
  pub corrupt: bool,
  docs: [u32; BLOCK],
  tfs: [u32; BLOCK],
}

impl<'a> Cursor<'a> {
  /// A cursor over `skips.len()` packed blocks in `data`. Callers may add
  /// a final block via `last`/`last_kind`, then call `start`.
  pub fn empty(
    data: &'a [u8],
    skips: &'a [SkipRec],
    df: u32,
    num_docs: usize,
  ) -> Self {
    Self {
      data,
      skips,
      last: SkipRec::default(),
      last_kind: LastKind::None,
      tail: &[],
      df,
      nblocks: skips.len(),
      num_docs,
      block: 0,
      decoded: usize::MAX,
      pos: 0,
      len: 0,
      doc: END,
      tf_off: 0,
      tf_bits: 0,
      tfs_ready: false,
      bound_block: usize::MAX,
      bound: 0.0,
      weight: 0.0,
      idx: 0,
      corrupt: false,
      docs: [0; BLOCK],
      tfs: [0; BLOCK],
    }
  }

  /// A cursor over a single packed block at the start of `data`.
  pub fn single(
    data: &'a [u8],
    df: u32,
    last_doc: DocId,
    impacts: Impacts,
    num_docs: usize,
  ) -> Self {
    let mut c = Self::empty(data, &[], df, num_docs);
    c.last = SkipRec {
      last_doc,
      offset: 0,
      impacts,
    };
    c.last_kind = LastKind::Packed;
    c.nblocks = 1;
    c.start();
    c
  }

  /// Validates the block count and positions on the first document.
  pub fn start(&mut self) {
    let want = (self.df as usize).div_ceil(BLOCK);
    if self.nblocks != want || self.df == 0 {
      self.fail();
      return;
    }
    if self.decode(0) {
      self.pos = 0;
      self.doc = self.docs[0];
    }
  }

  fn fail(&mut self) {
    self.corrupt = true;
    self.block = self.nblocks;
    self.doc = END;
  }

  /// Number of documents in the list.
  #[inline]
  pub fn df(&self) -> u32 {
    self.df
  }

  #[inline]
  fn rec(&self, i: usize) -> &SkipRec {
    if i < self.skips.len() {
      &self.skips[i]
    } else {
      &self.last
    }
  }

  #[inline]
  fn block_len(&self, i: usize) -> usize {
    if i + 1 < self.nblocks {
      BLOCK
    } else {
      self.df as usize - BLOCK * (self.nblocks - 1)
    }
  }

  /// Current document, or [`END`].
  #[inline]
  pub fn doc(&self) -> DocId {
    self.doc
  }

  /// Whether no blocks remain.
  #[inline]
  pub fn exhausted(&self) -> bool {
    self.block >= self.nblocks
  }

  /// Last doc id of the current block. Only valid if not exhausted.
  #[inline]
  pub fn block_last(&self) -> DocId {
    self.rec(self.block).last_doc
  }

  /// Score bound of the current block (cached per block).
  #[inline]
  pub fn block_bound(&mut self, norms: &LenNorms) -> f32 {
    if self.bound_block != self.block {
      self.bound = bound(&self.rec(self.block).impacts, self.weight, norms);
      self.bound_block = self.block;
    }
    self.bound
  }

  /// Moves to the first block whose last doc is `>= target`, without
  /// decoding it. Never moves backwards.
  pub fn shallow_advance(&mut self, target: DocId) {
    if self.block >= self.nblocks || self.rec(self.block).last_doc >= target {
      return;
    }
    let mut b = self.block + 1;
    let n = self.skips.len();
    if b < n {
      // Gallop, then binary search the bracket.
      let mut step = 1;
      let mut lo = b;
      while b < n && self.skips[b].last_doc < target {
        lo = b + 1;
        b += step;
        step *= 2;
      }
      let hi = b.min(n);
      b = lo + self.skips[lo..hi].partition_point(|r| r.last_doc < target);
    }
    if b >= n && b < self.nblocks && self.last.last_doc < target {
      b = self.nblocks;
    }
    self.block = b;
    if b >= self.nblocks {
      self.doc = END;
    }
  }

  /// Moves to the first document `>= target` and returns it, or [`END`].
  #[inline]
  pub fn advance(&mut self, target: DocId) -> DocId {
    if self.doc >= target {
      return self.doc;
    }
    self.shallow_advance(target);
    if self.block >= self.nblocks {
      return END;
    }
    if self.decoded != self.block {
      if !self.decode(self.block) {
        return END;
      }
      self.pos = 0;
    }
    let mut p = self.pos;
    // The block's last doc is >= target (validated by `decode`).
    while p + 1 < self.len && self.docs[p] < target {
      p += 1;
    }
    self.pos = p;
    self.doc = self.docs[p];
    self.doc
  }

  /// Moves to the next document and returns it, or [`END`].
  #[inline]
  pub fn next(&mut self) -> DocId {
    if self.doc == END {
      return END;
    }
    self.pos += 1;
    if self.pos >= self.len {
      self.block += 1;
      if self.block >= self.nblocks || !self.decode(self.block) {
        self.doc = END;
        return END;
      }
      self.pos = 0;
    }
    self.doc = self.docs[self.pos];
    self.doc
  }

  /// Term frequency of the current document.
  #[inline]
  pub fn tf(&mut self) -> u32 {
    if !self.tfs_ready {
      self.decode_tfs();
    }
    self.tfs[self.pos]
  }

  #[cold]
  fn decode_tfs(&mut self) {
    let src = self.data.get(self.tf_off..).unwrap_or(&[]);
    if !bitpack::unpack(src, self.tf_bits, self.len, &mut self.tfs) {
      // Lengths were checked in `decode`; unreachable on any input.
      self.tfs.fill(1);
      self.corrupt = true;
    } else {
      for t in &mut self.tfs {
        *t = t.wrapping_add(1);
      }
    }
    self.tfs_ready = true;
  }

  /// Decodes block `i` into `docs` (tfs lazily). Returns `false` and marks
  /// the cursor corrupt if the block does not check out.
  fn decode(&mut self, i: usize) -> bool {
    let n = self.block_len(i);
    let base = if i == 0 { 0 } else { self.rec(i - 1).last_doc };
    let rec_last = self.rec(i).last_doc;
    let ok = if i >= self.skips.len() && self.last_kind == LastKind::Varint {
      self.decode_varint(n)
    } else {
      self.decode_packed(self.rec(i).offset as usize, n)
    };
    if !ok {
      self.fail();
      return false;
    }
    // Prefix sums with validation: strictly increasing ids ending exactly
    // at the skip record's last doc, inside the id space.
    let mut acc = u64::from(base);
    let mut min_delta = u32::MAX;
    for (k, d) in self.docs[..n].iter_mut().enumerate() {
      if k > 0 || i > 0 {
        min_delta = min_delta.min(*d);
      }
      acc += u64::from(*d);
      *d = acc as u32;
    }
    if min_delta == 0
      || acc != u64::from(rec_last)
      || rec_last as usize >= self.num_docs
    {
      self.fail();
      return false;
    }
    self.decoded = i;
    self.len = n;
    true
  }

  fn decode_packed(&mut self, off: usize, n: usize) -> bool {
    let Some(src) = self.data.get(off..) else {
      return false;
    };
    if src.len() < 2 {
      return false;
    }
    let (wd, wt) = (u32::from(src[0]), u32::from(src[1]));
    if wd > 32 || wt > 32 {
      return false;
    }
    let dlen = packed_len(n, wd);
    let tlen = packed_len(n, wt);
    if src.len() < 2 + dlen + tlen {
      return false;
    }
    if !bitpack::unpack(&src[2..], wd, n, &mut self.docs) {
      return false;
    }
    self.tf_off = off + 2 + dlen;
    self.tf_bits = wt;
    self.tfs_ready = false;
    true
  }

  fn decode_varint(&mut self, n: usize) -> bool {
    let mut pos = 0;
    for k in 0..n {
      let (Some(d), Some(tf)) = (
        varint::get_u32(self.tail, &mut pos),
        varint::get_u32(self.tail, &mut pos),
      ) else {
        return false;
      };
      self.docs[k] = d;
      self.tfs[k] = tf;
    }
    self.tfs_ready = true;
    true
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn build(docs: &[(u32, u32)], num_docs: usize) -> (PostingList, Vec<u8>) {
    let norms: Vec<u8> = (0..num_docs).map(|i| (i % 40) as u8).collect();
    let mut p = PostingList::default();
    for &(d, tf) in docs {
      p.push(d, tf, &norms);
    }
    (p, norms)
  }

  fn collect(mut c: Cursor<'_>) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    while c.doc() != END {
      let tf = c.tf();
      out.push((c.doc(), tf));
      c.next();
    }
    assert!(!c.corrupt);
    out
  }

  fn sample(n: usize, gap: u32) -> Vec<(u32, u32)> {
    (0..n as u32)
      .map(|i| (i * gap + i % 3, 1 + (i * 7) % 200))
      .collect()
  }

  #[test]
  fn live_and_frozen_cursors_return_every_posting() {
    for n in [1usize, 5, 127, 128, 129, 256, 300, 1000] {
      let docs = sample(n, 5);
      let num_docs = docs.last().unwrap().0 as usize + 1;
      let (p, norms) = build(&docs, num_docs);
      assert_eq!(collect(p.cursor(&norms)), docs, "live n={n}");
      let f = p.freeze(&norms);
      let mut data = f.head.to_vec();
      data.extend_from_slice(&f.tail_block);
      let c = if f.skips.len() == 1 {
        Cursor::single(&data, f.df, f.last_doc, f.skips[0].impacts, num_docs)
      } else {
        let mut c = Cursor::empty(&data, &f.skips, f.df, num_docs);
        c.start();
        c
      };
      assert_eq!(collect(c), docs, "frozen n={n}");
    }
  }

  #[test]
  fn advance_and_shallow_skip() {
    let docs = sample(2000, 3);
    let num_docs = docs.last().unwrap().0 as usize + 1;
    let (p, norms) = build(&docs, num_docs);
    let mut c = p.cursor(&norms);
    let mut i = 0;
    for target in (0..num_docs as u32).step_by(97) {
      while docs[i].0 < target {
        i += 1;
      }
      assert_eq!(c.advance(target), docs[i].0);
      assert_eq!(c.tf(), docs[i].1);
    }
    assert_eq!(c.advance(num_docs as u32), END);
    let mut c = p.cursor(&norms);
    c.shallow_advance(docs[1000].0);
    assert!(c.block_last() >= docs[1000].0);
    assert_eq!(c.advance(docs[1000].0), docs[1000].0);
    c.shallow_advance(END - 1);
    assert!(c.exhausted());
    assert_eq!(c.doc(), END);
  }

  #[test]
  fn corrupt_block_is_reported_not_panicking() {
    let docs = sample(300, 4);
    let num_docs = docs.last().unwrap().0 as usize + 1;
    let (p, norms) = build(&docs, num_docs);
    let f = p.freeze(&norms);
    let mut data = f.head.to_vec();
    data.extend_from_slice(&f.tail_block);
    for flip in 0..data.len() {
      let mut bad = data.clone();
      bad[flip] ^= 0x5a;
      let mut c = Cursor::empty(&bad, &f.skips, f.df, num_docs);
      c.start();
      let mut steps = 0;
      while c.doc() != END {
        c.tf();
        c.next();
        steps += 1;
        assert!(steps <= 300);
      }
    }
    // Wrong block count for the df.
    let mut c = Cursor::empty(&data, &f.skips, 5000, num_docs);
    c.start();
    assert!(c.corrupt && c.doc() == END);
  }
}
