//! The mutable, in-memory part of a collection.
//!
//! Vectors live in a lock-free [`VecArena`]; everything else (id map,
//! stored records, one [`MemTextIndex`] per text field) lives in
//! [`MemInner`] behind an `RwLock`.
//!
//! Why a lock for the text side: `MemTextIndex` needs `&mut` to add and
//! its corpus statistics (document count, total length, document
//! frequencies) always describe *every* added document. A query must
//! score against statistics that match the snapshot it searches, so the
//! engine holds the read lock while it takes the snapshot, gathers
//! statistics and searches the memtable's text, and writers hold the
//! write lock while they apply a batch and publish the version that makes
//! it visible. Under the lock, the memtable's contents therefore equal
//! exactly what the newest version says. Both critical sections are
//! short: one batch on the write side, one memtable-sized text search on
//! the read side. Vector scans never take the lock.
//!
//! Trade-off: a slow text query over a large memtable delays writers for
//! its duration, and because `std::sync::RwLock` lets a waiting writer go
//! first on most platforms, new text searches then wait behind that
//! writer too. Memtables are bounded (`Options::memtable_max_docs`), which
//! bounds both waits.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use muninn_core::{DocId, Metric, Result};
use muninn_filter::MemIndex;
use muninn_store::{IdMap, PayloadBuilder};
use muninn_text::{Analyzer, MemTextIndex};

use crate::arena::{ArenaWriter, VecArena};
use crate::codec::{self, DocRef, STORED_FORMAT};
use crate::schema::Schema;
use crate::types::Id;

/// Takes a read lock, ignoring poisoning: a panic in a writer already
/// poisons the collection for writes (see `collection.rs`).
pub(crate) fn read<T>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
  l.read().unwrap_or_else(|e| e.into_inner())
}

/// Takes a write lock, ignoring poisoning (see [`read`]).
pub(crate) fn write<T>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
  l.write().unwrap_or_else(|e| e.into_inner())
}

/// A memtable: active (taking writes) or frozen (waiting to be flushed).
pub(crate) struct Memtable {
  /// Unique within the collection, for identity checks.
  pub id: u64,
  pub vectors: Option<VecArena>,
  pub metric: Metric,
  pub inner: RwLock<MemInner>,
}

/// The lock-protected part of a [`Memtable`].
pub(crate) struct MemInner {
  /// The arena's writer; `None` without a vector field.
  pub writer: Option<ArenaWriter>,
  /// External id per doc, and live external id to doc.
  pub ids: IdMap,
  /// Stored record (text and payload) per doc.
  pub stored: PayloadBuilder,
  /// One index per schema text field.
  pub text: Vec<MemTextIndex>,
  /// Payload index of the filterable fields; `None` if there are none.
  pub pindex: Option<MemIndex>,
  /// Docs added.
  pub docs: u32,
  /// Approximate memory held, for flush decisions.
  pub bytes: usize,
  scratch: Vec<f32>,
  record: Vec<u8>,
}

impl Memtable {
  /// An empty memtable for `schema`.
  pub(crate) fn new(id: u64, schema: &Schema) -> Result<Memtable> {
    let analyzer = Analyzer::new(schema.analyzer.clone())?;
    let (vectors, writer, metric) = match &schema.vector {
      Some(v) => {
        let (a, w) = VecArena::new(v.dim);
        (Some(a), Some(w), v.metric)
      }
      None => (None, None, Metric::L2),
    };
    let text = schema
      .text_fields
      .iter()
      .map(|_| MemTextIndex::new(analyzer.clone()))
      .collect();
    let pindex = if schema.filterable.is_empty() {
      None
    } else {
      Some(MemIndex::new(schema.filter_schema()?))
    };
    Ok(Memtable {
      id,
      vectors,
      metric,
      inner: RwLock::new(MemInner {
        writer,
        ids: IdMap::new(),
        stored: PayloadBuilder::new(),
        text,
        pindex,
        docs: 0,
        bytes: 0,
        scratch: Vec::new(),
        record: Vec::new(),
      }),
    })
  }

  /// Adds a decoded document as the next doc. Returns its id and the doc
  /// of this memtable the external id named before (now replaced), if
  /// any. Not visible to readers until the arena is published and a new
  /// version names the new length.
  pub(crate) fn insert(
    &self,
    g: &mut MemInner,
    doc: &DocRef<'_>,
  ) -> Result<(DocId, Option<DocId>)> {
    let next = g.docs;
    if let (Some(arena), Some(w)) = (&self.vectors, g.writer.as_mut()) {
      debug_assert_eq!(w.len(), next as usize);
      doc.vector_into(&mut g.scratch);
      if self.metric.normalizes() {
        muninn_simd::normalize(&mut g.scratch);
      }
      arena.push(w, &g.scratch);
    }
    let MemInner {
      record,
      stored,
      text,
      pindex,
      ..
    } = &mut *g;
    record.clear();
    record.push(STORED_FORMAT);
    record.extend_from_slice(doc.stored);
    let st = codec::decode_stored(record)?;
    let mut text_bytes = 0;
    for &(f, t) in &st.text {
      text[f].add(next, t)?;
      text_bytes += t.len();
    }
    if let Some(pi) = pindex
      && !st.payload.is_empty()
    {
      pi.add(next, &codec::parse_payload(st.payload)?)?;
    }
    let pushed = stored.push(record);
    debug_assert_eq!(pushed, next);
    let (id, old) = g.ids.insert(doc.id.clone())?;
    debug_assert_eq!(id, next);
    g.docs += 1;
    g.bytes += doc.vector.len()
      + 2 * doc.stored.len()
      + text_bytes
      + std::mem::size_of::<Id>()
      + 16;
    Ok((id, old))
  }

  /// Makes rows pushed by [`Memtable::insert`] scannable.
  pub(crate) fn publish(&self, g: &MemInner) {
    if let (Some(a), Some(w)) = (&self.vectors, &g.writer) {
      a.publish(w);
    }
  }
}
