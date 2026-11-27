//! Search requests.

use crate::filter::Filter;

/// How a hybrid query combines its vector and text results.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fusion {
  /// Reciprocal-rank fusion: a document scores `sum 1 / (k + rank)` over
  /// the result lists it appears in (rank 1 is the best). Ignores score
  /// scales entirely, so it needs no tuning. The usual `k` is 60.
  Rrf {
    /// Rank offset; larger values flatten the advantage of top ranks.
    k: u32,
  },
  /// Weighted sum of normalised scores. Each list's scores are min-max
  /// normalised over the candidates it returned: the best becomes 1, the
  /// worst 0 (all 1 if they are equal). A document missing from a list
  /// gets 0 for it. The fused score is
  /// `vector * norm_vector + text * norm_text`.
  Weighted {
    /// Weight of the vector similarity.
    vector: f32,
    /// Weight of the BM25 score.
    text: f32,
  },
}

impl Default for Fusion {
  fn default() -> Self {
    Self::Rrf { k: 60 }
  }
}

/// A search request: vector, text, or both (hybrid).
///
/// ```
/// use muninn::{Fusion, Query};
/// let q = Query::hybrid(vec![0.1, 0.2], "rust database")
///   .k(10)
///   .fusion(Fusion::Rrf { k: 60 })
///   .with_payload(true);
/// # let _ = q;
/// ```
#[derive(Clone, Debug)]
pub struct Query {
  pub(crate) vector: Option<Vec<f32>>,
  pub(crate) text: Option<String>,
  pub(crate) text_fields: Option<Vec<String>>,
  pub(crate) k: usize,
  pub(crate) ef: usize,
  pub(crate) rerank: usize,
  pub(crate) candidates: usize,
  pub(crate) filter: Option<Filter>,
  pub(crate) fusion: Fusion,
  pub(crate) with_payload: bool,
}

impl Query {
  fn new(vector: Option<Vec<f32>>, text: Option<String>) -> Self {
    Self {
      vector,
      text,
      text_fields: None,
      k: 10,
      ef: 0,
      rerank: 0,
      candidates: 0,
      filter: None,
      fusion: Fusion::default(),
      with_payload: false,
    }
  }

  /// Nearest neighbours of `v`.
  pub fn vector(v: impl Into<Vec<f32>>) -> Self {
    Self::new(Some(v.into()), None)
  }

  /// Best BM25 matches for `q` (any of its words), over every text field
  /// unless [`Query::fields`] narrows it. With several fields a document
  /// scores its best field's BM25.
  pub fn text(q: impl Into<String>) -> Self {
    Self::new(None, Some(q.into()))
  }

  /// Both, fused (see [`Fusion`]).
  pub fn hybrid(v: impl Into<Vec<f32>>, q: impl Into<String>) -> Self {
    Self::new(Some(v.into()), Some(q.into()))
  }

  /// Number of results (default 10).
  pub fn k(mut self, k: usize) -> Self {
    self.k = k;
    self
  }

  /// HNSW candidate-list size (default: the index's, 64); larger is more
  /// accurate and slower.
  pub fn ef(mut self, ef: usize) -> Self {
    self.ef = ef;
    self
  }

  /// Quantised candidates re-scored exactly (default: 4k for SQ8, 16k for
  /// binary).
  pub fn rerank(mut self, n: usize) -> Self {
    self.rerank = n;
    self
  }

  /// For hybrid queries, how many candidates each side retrieves before
  /// fusion (default `max(4 k, 40)`).
  pub fn candidates(mut self, n: usize) -> Self {
    self.candidates = n;
    self
  }

  /// Searches only these text fields.
  pub fn fields<S: Into<String>>(
    mut self,
    fields: impl IntoIterator<Item = S>,
  ) -> Self {
    self.text_fields = Some(fields.into_iter().map(Into::into).collect());
    self
  }

  /// Only returns documents matching `f`.
  pub fn filter(mut self, f: Filter) -> Self {
    self.filter = Some(f);
    self
  }

  /// Sets the fusion for hybrid queries.
  pub fn fusion(mut self, f: Fusion) -> Self {
    self.fusion = f;
    self
  }

  /// Returns each hit's payload.
  pub fn with_payload(mut self, on: bool) -> Self {
    self.with_payload = on;
    self
  }

  /// Candidates per side for a hybrid query.
  pub(crate) fn side_k(&self) -> usize {
    if self.vector.is_some() && self.text.is_some() {
      match self.candidates {
        0 => (self.k.saturating_mul(4)).max(40),
        n => n.max(self.k),
      }
    } else {
      self.k
    }
  }
}
