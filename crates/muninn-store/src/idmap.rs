//! User-facing document ids and their mapping to [`DocId`]s.
//!
//! Users name documents with a `u64` or a string ([`ExternalId`]); indexes
//! only see dense, segment-local [`DocId`]s. [`IdMap`] is the mutable form
//! used by the memtable; [`FrozenIdMap`] is the immutable per-segment form,
//! opened zero-copy from a container of kind `idmap`.
//!
//! # File format (`idmap`, version 1)
//!
//! | section    | contents                                                |
//! |------------|---------------------------------------------------------|
//! | `meta`     | `[u64; 8]`: docs, u64 keys, string keys, u64 table      |
//! |            | capacity, string table capacity, hash seed, 0, 0        |
//! | `fwd_u64`  | `[u64; docs]` numeric id per doc (absent if none)       |
//! | `str_off`  | `[u64; docs + 1]` byte offsets into `str_data`          |
//! | `str_data` | UTF-8 string ids, concatenated (both absent if none)    |
//! | `is_str`   | bitset over docs, set = string id (only when mixed)     |
//! | `u64_tab`  | `[u64; 2 * cap]` slots `(key, doc)`, doc `u64::MAX` =   |
//! |            | empty                                                   |
//! | `str_tab`  | `[u32; 2 * cap]` slots `(fingerprint, doc)`, doc        |
//! |            | `u32::MAX` = empty                                      |
//!
//! The reverse lookup is an open-addressing hash table with linear probing
//! (capacity a power of two, load factor at most 3/4), not sorted keys
//! with binary search. For a million ids a binary search takes about 20
//! dependent probes, the lower half of them cache misses; a hash lookup
//! touches one 64-byte line of the table for `u64` ids (key and doc are
//! stored together, four slots per line) and, for strings, that line plus
//! the stored string to confirm the match. `benches/idmap.rs` measures
//! both on the same keys. Only live ids are in the tables; every doc is in
//! the forward arrays, so a deleted doc can still name its external id.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{DocId, Error, Region, Result};

use crate::hash::{IdBuildHasher, hash_bytes, hash_u64};
use crate::view::View;

/// Container kind of a frozen id map.
pub const IDMAP_KIND: [u8; 8] = tag("idmap");
/// Current id-map format version.
pub const IDMAP_VERSION: u32 = 1;

const EMPTY_U64: u64 = u64::MAX;
const EMPTY_U32: u32 = u32::MAX;

/// A user-facing document id.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExternalId {
  U64(u64),
  Str(String),
}

/// A borrowed [`ExternalId`], for lookups without allocating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalIdRef<'a> {
  U64(u64),
  Str(&'a str),
}

impl ExternalId {
  /// Borrows this id.
  pub fn as_borrowed(&self) -> ExternalIdRef<'_> {
    match self {
      Self::U64(v) => ExternalIdRef::U64(*v),
      Self::Str(s) => ExternalIdRef::Str(s),
    }
  }
}

impl ExternalIdRef<'_> {
  /// Copies into an owned id.
  pub fn into_owned(self) -> ExternalId {
    match self {
      Self::U64(v) => ExternalId::U64(v),
      Self::Str(s) => ExternalId::Str(s.to_owned()),
    }
  }
}

impl From<u64> for ExternalId {
  fn from(v: u64) -> Self {
    Self::U64(v)
  }
}

impl From<&str> for ExternalId {
  fn from(s: &str) -> Self {
    Self::Str(s.to_owned())
  }
}

impl From<String> for ExternalId {
  fn from(s: String) -> Self {
    Self::Str(s)
  }
}

impl From<u64> for ExternalIdRef<'_> {
  fn from(v: u64) -> Self {
    Self::U64(v)
  }
}

impl<'a> From<&'a str> for ExternalIdRef<'a> {
  fn from(s: &'a str) -> Self {
    Self::Str(s)
  }
}

/// Mutable id map for the memtable.
///
/// [`IdMap::insert`] assigns the next dense [`DocId`]. Inserting an id
/// that is already mapped (an update) re-points it at the new doc and
/// returns the old one, which the caller tombstones. The forward direction
/// keeps every doc ever assigned, so `external_id` works for deleted docs.
#[derive(Clone, Debug, Default)]
pub struct IdMap {
  by_u64: HashMap<u64, DocId, IdBuildHasher>,
  by_str: HashMap<Box<str>, DocId, IdBuildHasher>,
  ids: Vec<ExternalId>,
}

impl IdMap {
  /// An empty map.
  pub fn new() -> Self {
    Self::default()
  }

  /// Number of [`DocId`]s assigned (live or not).
  pub fn len(&self) -> usize {
    self.ids.len()
  }

  /// True if no doc was ever assigned.
  pub fn is_empty(&self) -> bool {
    self.ids.is_empty()
  }

  /// Number of external ids currently mapped.
  pub fn live_len(&self) -> usize {
    self.by_u64.len() + self.by_str.len()
  }

  /// Assigns the next [`DocId`] to `id`. Returns it, and the doc `id` was
  /// mapped to before, if any. Fails once 2^32 - 1 docs are assigned.
  pub fn insert(&mut self, id: ExternalId) -> Result<(DocId, Option<DocId>)> {
    let doc = DocId::try_from(self.ids.len())
      .ok()
      .filter(|&d| d != DocId::MAX)
      .ok_or_else(|| Error::invalid("id map is full (2^32 - 1 docs)"))?;
    let old = match &id {
      ExternalId::U64(v) => self.by_u64.insert(*v, doc),
      ExternalId::Str(s) => self.by_str.insert(s.as_str().into(), doc),
    };
    self.ids.push(id);
    Ok((doc, old))
  }

  /// The doc `id` is mapped to.
  pub fn get(&self, id: ExternalIdRef<'_>) -> Option<DocId> {
    match id {
      ExternalIdRef::U64(v) => self.by_u64.get(&v).copied(),
      ExternalIdRef::Str(s) => self.by_str.get(s).copied(),
    }
  }

  /// Unmaps `id` (a delete) and returns the doc it named.
  pub fn remove(&mut self, id: ExternalIdRef<'_>) -> Option<DocId> {
    match id {
      ExternalIdRef::U64(v) => self.by_u64.remove(&v),
      ExternalIdRef::Str(s) => self.by_str.remove(s),
    }
  }

  /// The external id `doc` was assigned to.
  pub fn external_id(&self, doc: DocId) -> Option<ExternalIdRef<'_>> {
    self.ids.get(doc as usize).map(ExternalId::as_borrowed)
  }

  /// Writes the frozen form as an `idmap` container into `out`.
  ///
  /// The tables are hashed with a fresh random seed (stored in the file),
  /// so nobody can choose ids that all collide: with a known seed,
  /// colliding ids make building the table quadratic. Use
  /// [`IdMap::write_to_with_seed`] for byte-identical output.
  pub fn write_to<W: Write>(&self, out: W) -> Result<W> {
    self.write_to_with_seed(out, random_seed())
  }

  /// [`IdMap::write_to`] with a caller-chosen hash seed, for reproducible
  /// files in tests. Production files should use a random seed, so that
  /// ids cannot be chosen to collide.
  pub fn write_to_with_seed<W: Write>(&self, out: W, seed: u64) -> Result<W> {
    let n = self.ids.len();
    let n_str = self
      .ids
      .iter()
      .filter(|i| matches!(i, ExternalId::Str(_)))
      .count();
    let n_u64 = n - n_str;
    let u64_cap = table_capacity(self.by_u64.len());
    let str_cap = table_capacity(self.by_str.len());
    let mut w = ContainerWriter::new(out, IDMAP_KIND, IDMAP_VERSION)?;
    let meta: [u64; 8] = [
      n as u64,
      self.by_u64.len() as u64,
      self.by_str.len() as u64,
      u64_cap as u64,
      str_cap as u64,
      seed,
      0,
      0,
    ];
    w.pod_section(tag("meta"), &meta)?;

    if n_u64 > 0 {
      let fwd: Vec<u64> = self
        .ids
        .iter()
        .map(|i| match i {
          ExternalId::U64(v) => *v,
          ExternalId::Str(_) => 0,
        })
        .collect();
      w.pod_section(tag("fwd_u64"), &fwd)?;
    }
    if n_str > 0 {
      let mut off = Vec::with_capacity(n + 1);
      let mut at = 0u64;
      off.push(0u64);
      for i in &self.ids {
        if let ExternalId::Str(s) = i {
          at += s.len() as u64;
        }
        off.push(at);
      }
      w.pod_section(tag("str_off"), &off)?;
      w.begin_section(tag("str_data"))?;
      for i in &self.ids {
        if let ExternalId::Str(s) = i {
          w.write(s.as_bytes())?;
        }
      }
      w.end_section()?;
    }
    if n_str > 0 && n_u64 > 0 {
      let mut bits = vec![0u64; n.div_ceil(64)];
      for (d, i) in self.ids.iter().enumerate() {
        if matches!(i, ExternalId::Str(_)) {
          bits[d >> 6] |= 1 << (d & 63);
        }
      }
      w.pod_section(tag("is_str"), &bits)?;
    }

    // Insert in doc order so identical maps produce identical files.
    if u64_cap > 0 {
      let mask = u64_cap - 1;
      let mut tab = vec![0u64; 2 * u64_cap];
      for s in tab.chunks_exact_mut(2) {
        s[1] = EMPTY_U64;
      }
      for (d, i) in self.ids.iter().enumerate() {
        let ExternalId::U64(k) = i else { continue };
        if self.by_u64.get(k) != Some(&(d as DocId)) {
          continue;
        }
        let mut s = hash_u64(*k, seed) as usize & mask;
        while tab[2 * s + 1] != EMPTY_U64 {
          s = (s + 1) & mask;
        }
        tab[2 * s] = *k;
        tab[2 * s + 1] = d as u64;
      }
      w.pod_section(tag("u64_tab"), &tab)?;
    }
    if str_cap > 0 {
      let mask = str_cap - 1;
      let mut tab = vec![0u32; 2 * str_cap];
      for s in tab.chunks_exact_mut(2) {
        s[1] = EMPTY_U32;
      }
      for (d, i) in self.ids.iter().enumerate() {
        let ExternalId::Str(k) = i else { continue };
        if self.by_str.get(k.as_str()) != Some(&(d as DocId)) {
          continue;
        }
        let h = hash_bytes(k.as_bytes(), seed);
        let mut s = h as usize & mask;
        while tab[2 * s + 1] != EMPTY_U32 {
          s = (s + 1) & mask;
        }
        tab[2 * s] = (h >> 32) as u32;
        tab[2 * s + 1] = d as u32;
      }
      w.pod_section(tag("str_tab"), &tab)?;
    }
    w.finish()
  }

  /// Durably writes the frozen form to `path` (see [`crate::fs`]).
  pub fn write_file(&self, path: &Path) -> Result<u64> {
    crate::fs::write_atomic_with(path, |out| {
      self.write_to(out)?;
      Ok(())
    })
  }
}

/// A seed from the standard library's per-process random keys, mixed
/// with a counter so successive files differ.
fn random_seed() -> u64 {
  use std::hash::BuildHasher;
  use std::sync::atomic::{AtomicU64, Ordering};
  static N: AtomicU64 = AtomicU64::new(0);
  std::collections::hash_map::RandomState::new()
    .hash_one(N.fetch_add(1, Ordering::Relaxed))
}

/// Smallest power of two holding `keys` at a load factor of at most 3/4.
fn table_capacity(keys: usize) -> usize {
  if keys == 0 {
    0
  } else {
    (keys * 4).div_ceil(3).next_power_of_two()
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kinds {
  None,
  U64,
  Str,
  Mixed,
}

/// Immutable id map read in place from an `idmap` container.
#[derive(Clone)]
pub struct FrozenIdMap {
  n: usize,
  live: usize,
  seed: u64,
  kinds: Kinds,
  fwd_u64: View<u64>,
  str_off: View<u64>,
  str_data: View<u8>,
  is_str: View<u64>,
  u64_tab: View<u64>,
  u64_mask: usize,
  str_tab: View<u32>,
  str_mask: usize,
}

impl std::fmt::Debug for FrozenIdMap {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("FrozenIdMap")
      .field("len", &self.n)
      .field("live", &self.live)
      .finish()
  }
}

fn optional<T: muninn_core::Pod>(
  c: &Container,
  name: &str,
  expect: usize,
) -> Result<View<T>> {
  match c.try_section(tag(name)) {
    Some(r) => {
      let v = View::<T>::new(r)?;
      if v.as_slice().len() != expect {
        return Err(Error::corrupt(format!(
          "idmap section {name} has {} entries, expected {expect}",
          v.as_slice().len()
        )));
      }
      Ok(v)
    }
    None if expect == 0 => Ok(View::empty()),
    None => Err(Error::corrupt(format!("idmap section {name} missing"))),
  }
}

/// The table capacity a writer uses for `keys` keys, checked against the
/// stored value. Bounding it this way (keys never exceed docs, which are
/// fewer than 2^32) keeps every size computed from it far from overflow.
fn checked_cap(stored: u64, keys: u64, n: usize) -> Result<usize> {
  let keys = usize::try_from(keys)
    .ok()
    .filter(|&k| k <= n)
    .ok_or_else(|| Error::corrupt("idmap key count out of range"))?;
  let cap = table_capacity(keys);
  if stored != cap as u64 {
    return Err(Error::corrupt(format!(
      "idmap table capacity {stored} does not match {keys} keys"
    )));
  }
  Ok(cap)
}

impl FrozenIdMap {
  /// Opens a map from the bytes of an `idmap` container (usually a memory
  /// map). Checks structure and section sizes, not the checksum; reads
  /// nothing proportional to the number of ids.
  pub fn open(region: Region) -> Result<Self> {
    let c = Container::open_kind(region, IDMAP_KIND, IDMAP_VERSION)?;
    let meta_r = c.section(tag("meta"))?;
    let meta = meta_r.cast::<u64>()?;
    if meta.len() != 8 {
      return Err(Error::corrupt("idmap meta has the wrong size"));
    }
    let n = usize::try_from(meta[0])
      .ok()
      .filter(|&n| n < DocId::MAX as usize)
      .ok_or_else(|| Error::corrupt("idmap doc count out of range"))?;
    let (u64_keys, str_keys) = (meta[1], meta[2]);
    let u64_cap = checked_cap(meta[3], u64_keys, n)?;
    let str_cap = checked_cap(meta[4], str_keys, n)?;
    // Both counts are at most n < 2^32, so the sum cannot overflow.
    let live = (u64_keys + str_keys) as usize;
    if live > n {
      return Err(Error::corrupt("idmap key counts exceed the doc count"));
    }
    let has_u64 = c.try_section(tag("fwd_u64")).is_some();
    let has_str = c.try_section(tag("str_off")).is_some();
    let kinds = match (has_u64, has_str) {
      (false, false) => Kinds::None,
      (true, false) => Kinds::U64,
      (false, true) => Kinds::Str,
      (true, true) => Kinds::Mixed,
    };
    if kinds == Kinds::None && n != 0 {
      return Err(Error::corrupt("idmap has docs but no ids"));
    }
    let str_data = match c.try_section(tag("str_data")) {
      Some(r) => View::new(r)?,
      None if !has_str => View::empty(),
      None => return Err(Error::corrupt("idmap section str_data missing")),
    };
    let dim = |present: bool, len: usize| if present { len } else { 0 };
    Ok(Self {
      n,
      live,
      seed: meta[5],
      kinds,
      fwd_u64: optional(&c, "fwd_u64", dim(has_u64, n))?,
      str_off: optional(&c, "str_off", dim(has_str, n + 1))?,
      str_data,
      is_str: optional(
        &c,
        "is_str",
        dim(kinds == Kinds::Mixed, n.div_ceil(64)),
      )?,
      u64_tab: optional(&c, "u64_tab", 2 * u64_cap)?,
      u64_mask: u64_cap.wrapping_sub(1),
      str_tab: optional(&c, "str_tab", 2 * str_cap)?,
      str_mask: str_cap.wrapping_sub(1),
    })
  }

  /// Number of docs in the segment (live or not).
  pub fn len(&self) -> usize {
    self.n
  }

  /// True if the segment has no docs.
  pub fn is_empty(&self) -> bool {
    self.n == 0
  }

  /// Number of external ids that resolve to a doc.
  pub fn live_len(&self) -> usize {
    self.live
  }

  /// The doc `id` maps to.
  #[inline]
  pub fn get(&self, id: ExternalIdRef<'_>) -> Option<DocId> {
    match id {
      ExternalIdRef::U64(v) => self.get_u64(v),
      ExternalIdRef::Str(s) => self.get_str(s),
    }
  }

  /// The doc numeric id `key` maps to.
  #[inline]
  pub fn get_u64(&self, key: u64) -> Option<DocId> {
    let tab = self.u64_tab.as_slice();
    if tab.is_empty() {
      return None;
    }
    let mask = self.u64_mask;
    let mut s = hash_u64(key, self.seed) as usize & mask;
    // Bounded so a corrupt table with no empty slot cannot loop forever.
    for _ in 0..=mask {
      let (k, d) = (tab[2 * s], tab[2 * s + 1]);
      if d == EMPTY_U64 {
        return None;
      }
      if k == key && d < self.n as u64 {
        return Some(d as DocId);
      }
      s = (s + 1) & mask;
    }
    None
  }

  /// The doc string id `key` maps to.
  #[inline]
  pub fn get_str(&self, key: &str) -> Option<DocId> {
    let tab = self.str_tab.as_slice();
    if tab.is_empty() {
      return None;
    }
    let mask = self.str_mask;
    let h = hash_bytes(key.as_bytes(), self.seed);
    let fp = (h >> 32) as u32;
    let mut s = h as usize & mask;
    for _ in 0..=mask {
      let (f, d) = (tab[2 * s], tab[2 * s + 1]);
      if d == EMPTY_U32 {
        return None;
      }
      if f == fp && self.str_bytes(d) == Some(key.as_bytes()) {
        return Some(d);
      }
      s = (s + 1) & mask;
    }
    None
  }

  fn str_bytes(&self, doc: DocId) -> Option<&[u8]> {
    let off = self.str_off.as_slice();
    let d = doc as usize;
    let (a, b) = (*off.get(d)?, *off.get(d + 1)?);
    let a = usize::try_from(a).ok()?;
    let b = usize::try_from(b).ok()?;
    self.str_data.as_slice().get(a..b)
  }

  fn is_str_doc(&self, doc: usize) -> bool {
    match self.kinds {
      Kinds::None | Kinds::U64 => false,
      Kinds::Str => true,
      Kinds::Mixed => (self.is_str.as_slice()[doc >> 6] >> (doc & 63)) & 1 == 1,
    }
  }

  /// The external id of `doc`. Fails for docs outside the segment and for
  /// corrupt string data.
  pub fn external_id(&self, doc: DocId) -> Result<ExternalIdRef<'_>> {
    let d = doc as usize;
    if d >= self.n {
      return Err(Error::invalid(format!(
        "doc {doc} outside id map of {} docs",
        self.n
      )));
    }
    if self.is_str_doc(d) {
      let b = self
        .str_bytes(doc)
        .ok_or_else(|| Error::corrupt("idmap string offsets out of range"))?;
      let s = std::str::from_utf8(b)
        .map_err(|_| Error::corrupt("idmap string id is not UTF-8"))?;
      Ok(ExternalIdRef::Str(s))
    } else {
      Ok(ExternalIdRef::U64(self.fwd_u64.as_slice()[d]))
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use muninn_core::AlignedBuf;

  fn freeze(m: &IdMap) -> FrozenIdMap {
    let buf = m.write_to(AlignedBuf::new()).unwrap();
    let c = Container::open(Region::new(buf.clone())).unwrap();
    c.verify_checksum().unwrap();
    FrozenIdMap::open(Region::new(buf)).unwrap()
  }

  #[test]
  fn table_capacity_bounds_load() {
    assert_eq!(table_capacity(0), 0);
    assert_eq!(table_capacity(1), 2);
    assert_eq!(table_capacity(3), 4);
    assert_eq!(table_capacity(4), 8);
    for n in 1..2000 {
      let c = table_capacity(n);
      assert!(c.is_power_of_two() && n * 4 <= c * 3);
    }
  }

  #[test]
  fn identical_maps_freeze_to_identical_bytes() {
    let mut a = IdMap::new();
    let mut b = IdMap::new();
    for i in 0..500u64 {
      a.insert(ExternalId::U64(i * 7)).unwrap();
      b.insert(ExternalId::U64(i * 7)).unwrap();
      a.insert(format!("s{i}").into()).unwrap();
      b.insert(format!("s{i}").into()).unwrap();
    }
    let x = a.write_to_with_seed(Vec::new(), 7).unwrap();
    let y = b.write_to_with_seed(Vec::new(), 7).unwrap();
    assert_eq!(x, y);
  }

  #[test]
  fn empty_map_round_trips() {
    let f = freeze(&IdMap::new());
    assert!(f.is_empty());
    assert_eq!(f.get_u64(0), None);
    assert_eq!(f.get_str(""), None);
    assert!(f.external_id(0).is_err());
  }
}
