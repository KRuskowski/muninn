//! Vector indexes.
//!
//! Every index implements [`VectorIndex`], so the engine can hold a mix of
//! index kinds across segments and search them uniformly.
//!
//! - [`FlatIndex`]: exact, blocked and multi-threaded scan; the recall
//!   oracle for every approximate index and the memtable index.
//! - [`HnswIndex`]: approximate graph search with parallel build,
//!   incremental insert and filter-aware search.
//! - [`Quantization`]: SQ8 or binary codes for either index's first pass,
//!   re-ranked with full-precision vectors.
//!
//! Every index writes one `muninn_core::container` file in a single pass
//! and opens from a `Region` without copying its arrays.

mod dist;
mod flat;
mod hnsw;
mod meta;
mod quant;
mod scan;
mod slab;

pub use flat::{FLAT_KIND, FLAT_VERSION, FlatIndex};
pub use hnsw::{
  FilterStrategy, HNSW_KIND, HNSW_VERSION, HnswIndex, HnswParams, HnswStats,
  MAX_M,
};
pub use quant::Quantization;

use muninn_core::{DocFilter, Hit, Metric, Result};

/// Knobs for one search call. Zero means "use the index's default".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchParams {
  /// Number of results wanted.
  pub k: usize,
  /// Candidate-list size for graph search (HNSW `ef`). Larger is slower
  /// and more accurate. Ignored by exact indexes.
  pub ef: usize,
  /// How many quantised candidates to re-score with full-precision
  /// vectors (default: [`Quantization::default_rerank`]). Ignored by
  /// unquantised indexes.
  pub rerank: usize,
}

impl SearchParams {
  /// Asks for `k` results with every other knob at its default.
  pub fn new(k: usize) -> Self {
    Self {
      k,
      ef: 0,
      rerank: 0,
    }
  }

  /// Sets the graph candidate-list size.
  pub fn with_ef(mut self, ef: usize) -> Self {
    self.ef = ef;
    self
  }

  /// Sets how many quantised candidates are re-scored exactly.
  pub fn with_rerank(mut self, rerank: usize) -> Self {
    self.rerank = rerank;
    self
  }
}

/// A searchable set of vectors keyed by dense [`muninn_core::DocId`]s.
pub trait VectorIndex: Send + Sync {
  /// Dimensions of every vector.
  fn dim(&self) -> usize;

  fn metric(&self) -> Metric;

  /// Number of vectors, which is also one past the largest id.
  fn len(&self) -> usize;

  fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// The `params.k` nearest vectors to `query` that pass `filter`, best
  /// first, as [`Hit`]s with public scores. The index normalises the query
  /// itself for [`Metric::Cosine`]; callers pass raw vectors.
  ///
  /// Returns fewer than `k` hits only when fewer than `k` vectors pass the
  /// filter (exact indexes) or the graph cannot reach more (approximate
  /// indexes with a very selective filter).
  fn search(
    &self,
    query: &[f32],
    params: &SearchParams,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>>;
}
