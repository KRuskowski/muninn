//! muninn: an embedded vector database and full-text search engine.
//!
//! ```no_run
//! use muninn::{Database, Document, Metric, Options, Query, Schema,
//!              VectorField};
//! use serde_json::json;
//!
//! # fn main() -> muninn::Result<()> {
//! let db = Database::open("data", Options::default())?;
//! let docs = db.create_collection(
//!   "docs",
//!   Schema::vector(VectorField::hnsw(3, Metric::Cosine))
//!     .with_text_fields(["title"]),
//! )?;
//! docs.upsert(&[Document::new("a")
//!   .vector(vec![0.1, 0.2, 0.3])
//!   .text("title", "an embedded database")
//!   .payload(json!({"lang": "en"}))])?;
//! let hits = docs.search(
//!   &Query::hybrid(vec![0.1, 0.2, 0.3], "database").k(10).with_payload(true),
//! )?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```
//!
//! # How it works
//!
//! A [`Collection`] keeps recent writes in memory (a *memtable*: exact
//! vectors plus an in-memory text index) and everything older in
//! immutable on-disk *segments* (HNSW or flat vector index, one inverted
//! index per text field, stored documents, id map), all memory-mapped.
//! Every write is first recorded in a write-ahead log. Full memtables are
//! written to segments in the background; segments are merged in the
//! background too. Searches run on an immutable snapshot, in parallel
//! over all parts, and never block or get blocked by flushes and merges.
//! See `docs/ARCHITECTURE.md` for the design.

mod arena;
mod codec;
mod collection;
mod database;
mod filter;
mod memtable;
mod options;
mod query;
mod schema;
mod search;
mod segment;
mod types;
mod version;

pub use muninn_core as core;
pub use muninn_simd as simd;
pub use muninn_store as store;
pub use muninn_text as text;
pub use muninn_vector as vector;

pub use collection::Collection;
pub use database::Database;
pub use filter::{Filter, FilterPlugin, FilterSource, IdRef};
#[doc(hidden)]
pub use options::Failpoint;
pub use options::{Durability, Options};
pub use query::{Fusion, Query};
pub use schema::{FieldKind, FieldSpec, IndexKind, Schema, VectorField};
pub use types::{CollectionStats, Document, Id, SearchHit, SegmentStats};

pub use muninn_core::{BitSet, DocId, Error, Metric, Result};
pub use muninn_text::AnalyzerConfig;
pub use muninn_vector::Quantization;
