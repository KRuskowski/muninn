//! Documents, ids and search results.

use std::collections::BTreeMap;

/// A user-facing document id: a `u64` or a string.
///
/// `Id::from(7u64)`, `Id::from("doc-7")` and `"doc-7".into()` all work.
/// Numeric and string ids live in separate spaces: `7` and `"7"` are
/// different documents.
pub type Id = muninn_store::ExternalId;

/// One document: an id, an optional vector, named text fields and an
/// arbitrary JSON payload.
///
/// When the collection's schema has a vector field every document must
/// carry a vector of that dimension; otherwise none may. Text fields must
/// be declared in the schema; a document may leave any of them out.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
  /// The document's id. Upserting an existing id replaces the document.
  pub id: Id,
  /// The vector, for collections with a vector field.
  pub vector: Option<Vec<f32>>,
  /// Text per field name.
  pub text: BTreeMap<String, String>,
  /// Stored JSON, returned with results on request. `Null` stores
  /// nothing.
  pub payload: serde_json::Value,
}

impl Document {
  /// A document with only an id.
  pub fn new(id: impl Into<Id>) -> Self {
    Self {
      id: id.into(),
      vector: None,
      text: BTreeMap::new(),
      payload: serde_json::Value::Null,
    }
  }

  /// Sets the vector.
  pub fn vector(mut self, v: impl Into<Vec<f32>>) -> Self {
    self.vector = Some(v.into());
    self
  }

  /// Sets the text of `field`.
  pub fn text(
    mut self,
    field: impl Into<String>,
    text: impl Into<String>,
  ) -> Self {
    self.text.insert(field.into(), text.into());
    self
  }

  /// Sets the payload.
  pub fn payload(mut self, payload: serde_json::Value) -> Self {
    self.payload = payload;
    self
  }
}

/// One search result.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
  /// The document's id.
  pub id: Id,
  /// The final score, higher is better: the vector similarity for
  /// vector queries (see [`crate::Metric`]), BM25 for text queries, the
  /// fused score for hybrid queries (see [`crate::Fusion`]).
  pub score: f32,
  /// The vector similarity, when the vector part of the query found the
  /// document.
  pub vector_score: Option<f32>,
  /// The BM25 score, when the text part of the query found the document.
  pub text_score: Option<f32>,
  /// The stored payload, when the query asked for it
  /// ([`crate::Query::with_payload`]).
  pub payload: Option<serde_json::Value>,
}

/// Sizes and counters of one collection, for monitoring.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollectionStats {
  /// Documents that can be found (not deleted or replaced).
  pub live_docs: u64,
  /// Deleted or replaced documents still taking space until compaction.
  pub deleted_docs: u64,
  /// Documents in memory, not yet written to a segment (live or not).
  pub memtable_docs: u64,
  /// Memtables waiting to be written to segments.
  pub pending_flushes: usize,
  /// Immutable on-disk segments.
  pub segments: Vec<SegmentStats>,
  /// Log sequence number of the last applied write.
  pub last_lsn: u64,
  /// Every write up to this log sequence number is on stable storage.
  pub durable_lsn: u64,
  /// Every write up to this log sequence number is in segments.
  pub checkpoint_lsn: u64,
}

/// One segment, as reported by [`CollectionStats`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentStats {
  /// Segment id (unique within the collection, never reused).
  pub id: u64,
  /// Documents in the segment, deleted ones included.
  pub docs: u64,
  /// Deleted documents.
  pub deleted: u64,
  /// `"hnsw"`, `"flat"` or `"none"` (no vector field).
  pub vector_index: &'static str,
}
