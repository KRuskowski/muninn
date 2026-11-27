//! Search filters and the hook payload-filter crates plug into.
//!
//! A [`Filter`] is evaluated once per *source* of a search snapshot (each
//! segment and memtable) into a [`Selection`], which the engine ANDs with
//! "not deleted" and hands to the vector and text indexes as their
//! `DocFilter`, so filters apply during the search and `k` results come
//! back whenever `k` documents match.
//!
//! Built-in filters select by external id. Payload filters (field
//! equality, ranges) are provided by implementing [`FilterPlugin`]: given
//! a [`FilterSource`] (one segment or memtable), return the set of
//! matching doc ids. A plugin with its own per-segment indexes looks them
//! up by [`FilterSource::segment_id`].

use std::collections::HashSet;
use std::sync::Arc;

use muninn_core::{BitSet, DocFilter, DocId, Result};
use muninn_store::ExternalIdRef;

use crate::codec;
use crate::memtable::MemInner;
use crate::segment::Segment;
use crate::types::Id;

/// A borrowed [`Id`], passed to predicates without allocating.
pub type IdRef<'a> = ExternalIdRef<'a>;

/// What a [`FilterPlugin`] can see of one source of a collection.
pub trait FilterSource {
  /// Doc ids of this source are `0..len()`.
  fn len(&self) -> u32;
  /// True for an empty source.
  fn is_empty(&self) -> bool {
    self.len() == 0
  }
  /// The segment id, or `None` for in-memory data.
  fn segment_id(&self) -> Option<u64>;
  /// The external id of `doc`.
  fn external_id(&self, doc: DocId) -> Result<Id>;
  /// The doc of a live external id in this source, if any. A doc returned
  /// here may still have been deleted since; the engine excludes deleted
  /// docs on its own.
  fn lookup(&self, id: &Id) -> Option<DocId>;
  /// The stored payload of `doc` (`Null` if none).
  fn payload(&self, doc: DocId) -> Result<serde_json::Value>;
}

/// A filter that can be evaluated per source: the extension point for
/// payload filtering.
pub trait FilterPlugin: Send + Sync {
  /// The docs of `src` that match. Ids at or above `src.len()` are
  /// ignored.
  fn select(&self, src: &dyn FilterSource) -> Result<BitSet>;
}

type Predicate = dyn Fn(IdRef<'_>) -> bool + Send + Sync;

#[derive(Clone)]
enum Kind {
  Ids(Arc<HashSet<Id>>),
  Predicate(Arc<Predicate>),
  Plugin(Arc<dyn FilterPlugin>),
  And(Vec<Filter>),
  Or(Vec<Filter>),
  Not(Box<Filter>),
}

/// Restricts which documents a search may return.
///
/// ```
/// use muninn::Filter;
/// let f = Filter::ids([1u64, 2, 3]).and(Filter::ids([2u64, 3]).not());
/// # let _ = f;
/// ```
#[derive(Clone)]
pub struct Filter {
  kind: Kind,
}

impl std::fmt::Debug for Filter {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match &self.kind {
      Kind::Ids(s) => write!(f, "Ids({} ids)", s.len()),
      Kind::Predicate(_) => write!(f, "Predicate"),
      Kind::Plugin(_) => write!(f, "Plugin"),
      Kind::And(v) => f.debug_tuple("And").field(v).finish(),
      Kind::Or(v) => f.debug_tuple("Or").field(v).finish(),
      Kind::Not(x) => f.debug_tuple("Not").field(x).finish(),
    }
  }
}

impl Filter {
  /// Only documents with one of these ids.
  pub fn ids<I: Into<Id>>(ids: impl IntoIterator<Item = I>) -> Self {
    Self {
      kind: Kind::Ids(Arc::new(ids.into_iter().map(Into::into).collect())),
    }
  }

  /// Only documents whose id passes `f`. Evaluated lazily per candidate
  /// on segments, so keep it cheap.
  pub fn predicate(
    f: impl Fn(IdRef<'_>) -> bool + Send + Sync + 'static,
  ) -> Self {
    Self {
      kind: Kind::Predicate(Arc::new(f)),
    }
  }

  /// A filter evaluated by a [`FilterPlugin`].
  pub fn plugin(p: Arc<dyn FilterPlugin>) -> Self {
    Self {
      kind: Kind::Plugin(p),
    }
  }

  /// Documents matching both.
  pub fn and(self, other: Filter) -> Self {
    Self {
      kind: Kind::And(vec![self, other]),
    }
  }

  /// Documents matching either.
  pub fn or(self, other: Filter) -> Self {
    Self {
      kind: Kind::Or(vec![self, other]),
    }
  }

  /// Documents not matching.
  #[allow(clippy::should_implement_trait)]
  pub fn not(self) -> Self {
    Self {
      kind: Kind::Not(Box::new(self)),
    }
  }

  /// Evaluates the filter on one source. Lazy predicates may borrow only
  /// `lazy_seg` (the segment, when `src` is one), so a memtable's
  /// selection does not borrow its lock guard.
  pub(crate) fn select<'s>(
    &self,
    src: &SourceRef<'_>,
    lazy_seg: Option<&'s Segment>,
  ) -> Result<Selection<'s>> {
    let len = src.len();
    Ok(match &self.kind {
      Kind::Ids(set) => {
        let mut b = BitSet::new(len as usize);
        for id in set.iter() {
          if let Some(d) = src.lookup(id)
            && d < len
          {
            b.insert(d);
          }
        }
        Selection::bits(b)
      }
      Kind::Predicate(p) => match (src, lazy_seg) {
        (SourceRef::Seg(_), Some(seg)) => {
          let p = p.clone();
          Selection::Pred(Box::new(move |d| {
            seg.ids.external_id(d).map(|e| p(e)).unwrap_or(false)
          }))
        }
        (SourceRef::Seg(seg), None) => {
          let mut b = BitSet::new(len as usize);
          for d in 0..len {
            if seg.ids.external_id(d).is_ok_and(|e| p(e)) {
              b.insert(d);
            }
          }
          Selection::bits(b)
        }
        (SourceRef::Mem(g, _), _) => {
          // Memtables are small: evaluate eagerly, under the lock the
          // caller already holds.
          let mut b = BitSet::new(len as usize);
          for d in 0..len {
            if g.ids.external_id(d).is_some_and(|e| p(e)) {
              b.insert(d);
            }
          }
          Selection::bits(b)
        }
      },
      Kind::Plugin(p) => {
        let mut b = p.select(src)?;
        if b.capacity() > len as usize {
          let mut t = BitSet::new(len as usize);
          for d in b.iter().take_while(|&d| d < len) {
            t.insert(d);
          }
          b = t;
        }
        Selection::bits(b)
      }
      Kind::And(v) => {
        let parts = v
          .iter()
          .map(|f| f.select(src, lazy_seg))
          .collect::<Result<Vec<_>>>()?;
        Selection::and(parts)
      }
      Kind::Or(v) => {
        let parts = v
          .iter()
          .map(|f| f.select(src, lazy_seg))
          .collect::<Result<Vec<_>>>()?;
        Selection::or(parts, len)
      }
      Kind::Not(f) => match f.select(src, lazy_seg)? {
        Selection::Bits(b, _) => {
          let mut all = BitSet::full(len as usize);
          all.difference_with(&b);
          Selection::bits(all)
        }
        Selection::All => Selection::bits(BitSet::new(len as usize)),
        other => Selection::Not(Box::new(other)),
      },
    })
  }
}

/// One source, as filters see it.
pub(crate) enum SourceRef<'a> {
  /// A memtable's locked contents and its visible length.
  Mem(&'a MemInner, u32),
  Seg(&'a Segment),
}

impl FilterSource for SourceRef<'_> {
  fn len(&self) -> u32 {
    match self {
      Self::Mem(_, n) => *n,
      Self::Seg(s) => s.len,
    }
  }

  fn segment_id(&self) -> Option<u64> {
    match self {
      Self::Mem(..) => None,
      Self::Seg(s) => Some(s.id),
    }
  }

  fn external_id(&self, doc: DocId) -> Result<Id> {
    let missing = || muninn_core::Error::NotFound(format!("doc {doc}"));
    match self {
      Self::Mem(g, n) => {
        if doc >= *n {
          return Err(missing());
        }
        g.ids
          .external_id(doc)
          .map(|e| e.into_owned())
          .ok_or_else(missing)
      }
      Self::Seg(s) => Ok(s.ids.external_id(doc)?.into_owned()),
    }
  }

  fn lookup(&self, id: &Id) -> Option<DocId> {
    match self {
      Self::Mem(g, n) => g.ids.get(id.as_borrowed()).filter(|d| d < n),
      Self::Seg(s) => s.ids.get(id.as_borrowed()),
    }
  }

  fn payload(&self, doc: DocId) -> Result<serde_json::Value> {
    let rec = match self {
      Self::Mem(g, n) => {
        if doc >= *n {
          return Err(muninn_core::Error::NotFound(format!("doc {doc}")));
        }
        g.stored
          .get(doc)
          .ok_or_else(|| muninn_core::Error::NotFound(format!("doc {doc}")))?
      }
      Self::Seg(s) => s.stored.get(doc)?,
    };
    codec::parse_payload(codec::decode_stored(rec)?.payload)
  }
}

/// A filter evaluated on one source.
pub(crate) enum Selection<'a> {
  All,
  /// Allowed docs and their count.
  Bits(BitSet, usize),
  Pred(Box<dyn Fn(DocId) -> bool + Send + Sync + 'a>),
  And(Vec<Selection<'a>>),
  Or(Vec<Selection<'a>>),
  Not(Box<Selection<'a>>),
}

impl<'a> Selection<'a> {
  fn bits(b: BitSet) -> Self {
    let n = b.count();
    Self::Bits(b, n)
  }

  fn and(parts: Vec<Selection<'a>>) -> Self {
    let mut bits: Option<BitSet> = None;
    let mut rest = Vec::new();
    for p in parts {
      match p {
        Self::All => {}
        Self::Bits(b, _) => match &mut bits {
          Some(acc) => acc.intersect_with(&b),
          None => bits = Some(b),
        },
        other => rest.push(other),
      }
    }
    match (bits, rest.is_empty()) {
      (None, true) => Self::All,
      (Some(b), true) => Self::bits(b),
      (b, false) => {
        if let Some(b) = b {
          rest.insert(0, Self::bits(b));
        }
        Self::And(rest)
      }
    }
  }

  fn or(parts: Vec<Selection<'a>>, len: u32) -> Self {
    let mut bits = BitSet::new(len as usize);
    let mut rest = Vec::new();
    for p in parts {
      match p {
        Self::All => return Self::All,
        Self::Bits(b, _) => bits.union_with(&b),
        other => rest.push(other),
      }
    }
    if rest.is_empty() {
      return Self::bits(bits);
    }
    rest.insert(0, Self::bits(bits));
    Self::Or(rest)
  }

  #[inline]
  pub(crate) fn matches(&self, d: DocId) -> bool {
    match self {
      Self::All => true,
      Self::Bits(b, _) => b.contains(d),
      Self::Pred(p) => p(d),
      Self::And(v) => v.iter().all(|s| s.matches(d)),
      Self::Or(v) => v.iter().any(|s| s.matches(d)),
      Self::Not(s) => !s.matches(d),
    }
  }

  /// Estimated fraction of `len` docs that match, if known.
  fn hint(&self, len: u32) -> Option<f32> {
    match self {
      Self::All => Some(1.0),
      Self::Bits(_, n) => Some(if len == 0 {
        0.0
      } else {
        *n as f32 / len as f32
      }),
      Self::And(v) => v
        .iter()
        .filter_map(|s| s.hint(len))
        .min_by(|a, b| a.total_cmp(b)),
      _ => None,
    }
  }

  pub(crate) fn is_all(&self) -> bool {
    matches!(self, Self::All)
  }
}

/// The filter handed to an index: the user's selection AND not deleted
/// AND below the snapshot's length.
pub(crate) struct SourceFilter<'a> {
  pub sel: &'a Selection<'a>,
  pub tomb: &'a BitSet,
  pub deleted: usize,
  pub len: u32,
}

impl SourceFilter<'_> {
  /// Whether this filter excludes nothing from `0..len`.
  pub(crate) fn passes_all(&self) -> bool {
    self.sel.is_all() && self.deleted == 0
  }
}

impl DocFilter for SourceFilter<'_> {
  #[inline]
  fn matches(&self, d: DocId) -> bool {
    d < self.len && !self.tomb.contains(d) && self.sel.matches(d)
  }

  fn selectivity_hint(&self) -> Option<f32> {
    let live = if self.len == 0 {
      0.0
    } else {
      1.0 - self.deleted as f32 / self.len as f32
    };
    match self.sel {
      Selection::All => Some(live),
      // A bitset's count already excludes nothing deleted; assume
      // independence otherwise.
      s => s.hint(self.len).map(|h| h * live),
    }
  }
}
