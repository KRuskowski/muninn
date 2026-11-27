//! Full-text search.
//!
//! Scope of this crate (see `docs/ARCHITECTURE.md`): text analysis
//! (tokenising, normalising), an in-memory index writer, an immutable
//! on-disk index format opened from a [`muninn_core::Region`], and BM25
//! top-k retrieval with dynamic pruning.
//!
//! Every searchable text index implements [`TextSearcher`].

use muninn_core::{DocFilter, Hit, Result};

/// A searchable full-text index over one text field.
pub trait TextSearcher: Send + Sync {
  /// Number of documents indexed, which is also one past the largest id.
  fn num_docs(&self) -> usize;

  /// The `k` best documents for `query` by BM25 that pass `filter`, best
  /// first. Query syntax: whitespace-separated terms, OR semantics,
  /// analysed with the same analyser as the indexed text.
  fn search(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>>;
}
