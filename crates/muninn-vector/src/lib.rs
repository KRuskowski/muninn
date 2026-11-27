//! Vector indexes.
//!
//! Every index implements [`VectorIndex`], so the engine can hold a mix of
//! index kinds across segments and search them uniformly. [`FlatIndex`] is
//! the exact baseline and the recall oracle for every approximate index.

mod flat;

pub use flat::FlatIndex;

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
  /// vectors. Ignored by unquantised indexes.
  pub rerank: usize,
}

impl SearchParams {
  pub fn new(k: usize) -> Self {
    Self {
      k,
      ef: 0,
      rerank: 0,
    }
  }

  pub fn with_ef(mut self, ef: usize) -> Self {
    self.ef = ef;
    self
  }

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
