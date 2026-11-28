//! Filters: per-source evaluation of payload filters and id filters.
//!
//! A query may carry a payload [`Filter`] (fields of the JSON payload,
//! evaluated by the per-source payload indexes of `muninn-filter`) and an
//! [`IdFilter`] (external ids). Both are evaluated once per *source* of a
//! search snapshot (each segment and memtable) into a [`Selection`],
//! which the engine ANDs with "not deleted" and hands to the vector and
//! text indexes as their `DocFilter`. Filters therefore apply during the
//! search, and `k` results come back whenever `k` documents match.

use std::collections::HashSet;
use std::sync::Arc;

use muninn_core::{BitSet, DocFilter, DocId, Result};
use muninn_filter::{Filter, FilterIndex, MemIndex};
use muninn_store::ExternalIdRef;

use crate::memtable::MemInner;
use crate::segment::Segment;
use crate::types::Id;

/// A borrowed [`Id`], passed to predicates without allocating.
pub type IdRef<'a> = ExternalIdRef<'a>;

type Predicate = dyn Fn(IdRef<'_>) -> bool + Send + Sync;

#[derive(Clone)]
enum Kind {
  Ids(Arc<HashSet<Id>>),
  Predicate(Arc<Predicate>),
  And(Vec<IdFilter>),
  Or(Vec<IdFilter>),
  Not(Box<IdFilter>),
}

/// Restricts a search to documents by external id. Combine with a
/// payload [`Filter`] on the same query; both must match.
///
/// ```
/// use muninn::IdFilter;
/// let f = IdFilter::ids([1u64, 2, 3]).and(IdFilter::ids([2u64]).not());
/// # let _ = f;
/// ```
#[derive(Clone)]
pub struct IdFilter {
  kind: Kind,
}

impl std::fmt::Debug for IdFilter {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match &self.kind {
      Kind::Ids(s) => write!(f, "Ids({} ids)", s.len()),
      Kind::Predicate(_) => write!(f, "Predicate"),
      Kind::And(v) => f.debug_tuple("And").field(v).finish(),
      Kind::Or(v) => f.debug_tuple("Or").field(v).finish(),
      Kind::Not(x) => f.debug_tuple("Not").field(x).finish(),
    }
  }
}

impl IdFilter {
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

  /// Documents matching both.
  pub fn and(self, other: IdFilter) -> Self {
    Self {
      kind: Kind::And(vec![self, other]),
    }
  }

  /// Documents matching either.
  pub fn or(self, other: IdFilter) -> Self {
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

impl SourceRef<'_> {
  fn len(&self) -> u32 {
    match self {
      Self::Mem(_, n) => *n,
      Self::Seg(s) => s.len,
    }
  }

  fn lookup(&self, id: &Id) -> Option<DocId> {
    match self {
      Self::Mem(g, n) => g.ids.get(id.as_borrowed()).filter(|d| d < n),
      Self::Seg(s) => s.ids.get(id.as_borrowed()),
    }
  }

  /// Evaluates a payload filter (already checked against the schema) on
  /// this source's payload index.
  pub(crate) fn payload(&self, f: &Filter) -> Result<Selection<'static>> {
    let len = self.len() as usize;
    let idx: Option<&dyn FilterIndex> = match self {
      Self::Mem(g, _) => g.pindex.as_ref().map(|i| i as &dyn FilterIndex),
      Self::Seg(s) => s.pindex.as_ref().map(|i| i as &dyn FilterIndex),
    };
    let bits = match idx {
      Some(i) => i.evaluate(f, len)?,
      // No filterable fields: only field-free filters (empty `and`/`or`
      // and negations of them) pass the schema check, and an empty index
      // evaluates those.
      None => MemIndex::new(muninn_filter::Schema::new()).evaluate(f, len)?,
    };
    Ok(Selection::bits(bits))
  }
}

/// A filter evaluated on one source.
pub(crate) enum Selection<'a> {
  All,
  /// Allowed docs and their count (counted once, not per query step).
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

  pub(crate) fn and(parts: Vec<Selection<'a>>) -> Self {
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

/// The filter handed to an index: the query's selection AND not deleted
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
      // Assume deletes are independent of the selection.
      s => s.hint(self.len).map(|h| h * live),
    }
  }
}
