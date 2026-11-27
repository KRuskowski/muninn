//! Payload filters for muninn: a filter language over JSON payload
//! fields, and per-segment payload indexes that evaluate filters to a
//! [`BitSet`] the vector and text indexes consume as a
//! [`muninn_core::DocFilter`].
//!
//! # Pieces
//!
//! - [`Filter`]: `eq`, `in`, `range`, `exists`, `and`, `or`, `not`, with
//!   builder helpers, `&` / `|` / `!` operators and a JSON form
//!   ([`Filter::from_json`], [`Filter::to_json`]).
//! - [`Schema`]: the filterable fields of a collection, each a
//!   dot-separated payload path with a [`FieldType`].
//! - [`MemIndex`]: the memtable's mutable index; [`MemIndex::add`] one
//!   payload per document, ids increasing.
//! - [`FrozenIndex`]: a segment's immutable index, written by
//!   [`MemIndex::write_to`] or [`FrozenIndex::merge`] and opened
//!   zero-copy from a [`muninn_core::Region`].
//!
//! Both indexes implement [`FilterIndex`]: `evaluate(filter, num_docs)`
//! returns the matching ids as a [`BitSet`], `estimate` the expected
//! fraction of matches without evaluating.
//!
//! # Semantics
//!
//! - **Multi-valued fields.** Arrays anywhere on a field's path are
//!   flattened. `eq`, `in` and `range` match when any one value matches
//!   (for `range`, one value must satisfy both bounds). `not` is the
//!   complement within `0..num_docs`, so `not eq tag "a"` keeps documents
//!   with no `"a"` value, including those without the field.
//! - **Types.** A payload value of the wrong type is ignored for that
//!   document, never an error: it neither matches nor counts for
//!   `exists`. A *filter* value of the wrong type (a string compared with
//!   an integer field, `range` on a keyword) is an `InvalidArgument`
//!   error, as is any undeclared field: the engine decides whether to
//!   fall back to scanning payloads.
//! - **Integer fields** hold JSON numbers that are whole and fit in an
//!   `i64` (`3`, `3.0` and `3e0` are all 3; `3.5` and `2^64` are
//!   skipped). A float filter value compares exactly: `eq 3.0` matches 3,
//!   `eq 3.5` matches nothing, `gt 2.5` means `>= 3`.
//! - **Float fields** hold any JSON number as an `f64` (integers above
//!   2^53 round to the nearest float). `eq` on a float field is exact
//!   equality after that conversion, with `-0.0 == 0.0`; use a `range`
//!   for tolerance. Integer filter values are converted to `f64` too.
//!   JSON cannot carry NaN or infinities, so payloads never hold them.
//!   NaN in a filter is an `InvalidArgument` error; infinite bounds are
//!   fine (`lt inf` matches every value).
//! - **Empty combinators.** `and []` matches every document, `or []` and
//!   `in []` match none.
//!
//! # Speed
//!
//! Evaluation folds constants, runs the most selective `and` clause
//! first (estimated from posting lengths and range counts), stops as soon
//! as an `and` is empty or an `or` is full, and applies `and not x` as a
//! set difference. Frozen keyword postings are sorted `u32` lists, or
//! bitsets when a term is in more than 1 of 32 documents; numeric ranges
//! are two binary searches over sorted values.

mod bits;
mod filter;
mod frozen;
mod json;
mod mem;
mod plan;
mod schema;
mod value;
mod write;

use muninn_core::{BitSet, Result};

pub use filter::Filter;
pub use frozen::{DELETED, FrozenIndex};
pub use mem::MemIndex;
pub use plan::MAX_DEPTH;
pub use schema::{Field, FieldType, MAX_FIELDS, Schema};
pub use value::Value;

/// What the engine needs from a payload index, mutable or frozen.
pub trait FilterIndex: Send + Sync {
  /// The declared filterable fields.
  fn schema(&self) -> &Schema;

  /// Ids `0..num_docs` whose payload matches `filter`. Fails with
  /// `InvalidArgument` for undeclared fields or mistyped values.
  fn evaluate(&self, filter: &Filter, num_docs: usize) -> Result<BitSet>;

  /// Estimated fraction of `0..num_docs` matching `filter`, in `0..=1`,
  /// computed without materialising any set. Returns 1.0 for filters
  /// that `evaluate` would reject.
  fn estimate(&self, filter: &Filter, num_docs: usize) -> f32;
}

impl FilterIndex for MemIndex {
  fn schema(&self) -> &Schema {
    MemIndex::schema(self)
  }

  fn evaluate(&self, filter: &Filter, num_docs: usize) -> Result<BitSet> {
    MemIndex::evaluate(self, filter, num_docs)
  }

  fn estimate(&self, filter: &Filter, num_docs: usize) -> f32 {
    MemIndex::estimate(self, filter, num_docs)
  }
}

impl FilterIndex for FrozenIndex {
  fn schema(&self) -> &Schema {
    FrozenIndex::schema(self)
  }

  fn evaluate(&self, filter: &Filter, num_docs: usize) -> Result<BitSet> {
    FrozenIndex::evaluate(self, filter, num_docs)
  }

  fn estimate(&self, filter: &Filter, num_docs: usize) -> f32 {
    FrozenIndex::estimate(self, filter, num_docs)
  }
}
