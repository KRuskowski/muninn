//! Benchmark harness comparing muninn with established engines.
//!
//! The `muninn-bench` binary measures muninn's indexes on standard
//! datasets and writes one JSON file per run. Competitors run in the same
//! harness (tantivy, linked in) or in Python scripts next to it (hnswlib,
//! faiss, usearch) that write the same JSON schema. `docs/BENCHMARKS.md`
//! describes the whole pipeline and how to reproduce it.
//!
//! - [`dataset`]: the on-disk vector dataset format and its loader.
//! - [`measure`]: recall, overlap and latency percentiles.
//! - [`sys`]: machine description and memory figures for reports.
//! - [`vector`]: muninn HNSW and flat search sweeps, ground-truth checks.
//! - [`text`]: muninn and tantivy BM25 indexing and query benchmarks.
//! - [`run`]: shared helpers for timing query passes on thread pools.

pub mod dataset;
pub mod measure;
pub mod run;
pub mod sys;
pub mod text;
pub mod vector;

/// Error type of the harness: any error, with a message.
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Result type of the harness.
pub type Result<T> = std::result::Result<T, Error>;
