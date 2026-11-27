//! muninn: an embedded vector database and full-text search engine.
//!
//! The engine (collections, segments, hybrid search) lands in a later
//! workstream; for now this crate re-exports the building blocks.

pub use muninn_core as core;
pub use muninn_simd as simd;
pub use muninn_store as store;
pub use muninn_text as text;
pub use muninn_vector as vector;

pub use muninn_core::{DocFilter, DocId, Error, Hit, Metric, Result};
