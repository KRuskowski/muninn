//! Full-text search.
//!
//! Scope of this crate (see `docs/ARCHITECTURE.md`): text analysis
//! (tokenising, normalising), an in-memory index writer, an immutable
//! on-disk index format opened from a [`muninn_core::Region`], and BM25
//! top-k retrieval with dynamic pruning.
//!
//! * [`Analyzer`] turns text into terms; its [`AnalyzerConfig`] is stored
//!   in every index file.
//! * [`MemTextIndex`] is appendable and searchable after every add (the
//!   memtable's text index).
//! * [`TextIndex`] is the frozen form: written from a [`MemTextIndex`]
//!   or merged from several [`TextIndex`]es, opened zero-copy.
//! * Both implement [`TextSearcher`]: BM25 ([`bm25`]) with OR semantics,
//!   block-max MaxScore pruning, and filters applied during traversal.
//!
//! Scores from different segments are only comparable when they use the
//! same corpus statistics; see [`CorpusStats`] and
//! [`TextSearcher::search_with_stats`].

mod analyzer;
mod bitpack;
pub mod bm25;
mod frozen;
mod hash;
mod impacts;
mod mem;
pub mod norm;
mod porter;
mod postings;
mod search;
mod stats;
mod varint;

pub use analyzer::{
  Analyzer, AnalyzerConfig, PorterStemmer, Stemmer, StemmerChoice, Tokens,
};
pub use frozen::{DELETED, FORMAT_VERSION, KIND, MergeInput, TextIndex};
pub use mem::MemTextIndex;
pub use stats::{CorpusStats, TermDocFreq};

use muninn_core::{DocFilter, Hit, Result};

/// Most distinct analysed terms a query may have. Longer queries are
/// rejected with `InvalidArgument`; repeated terms count once.
pub const MAX_QUERY_TERMS: usize = 1024;

/// A searchable full-text index over one text field.
///
/// Queries are analysed with the index's analyzer; the distinct terms
/// are OR-ed (a term repeated in the query counts once). A query may have
/// at most [`MAX_QUERY_TERMS`] distinct terms; longer ones are rejected
/// with `InvalidArgument`. A `k` larger than [`TextSearcher::num_docs`]
/// is treated as `num_docs`. Results are the
/// `k` best documents by BM25 that pass the filter, best first, ties
/// broken by smaller id. Pruned search returns exactly what scoring
/// every matching document would.
pub trait TextSearcher: Send + Sync {
  /// Size of the id space: one past the largest id added. Ids that were
  /// skipped are absent and never returned.
  fn num_docs(&self) -> usize;

  /// The `k` best documents for `query` by BM25 that pass `filter`, best
  /// first, scored with this index's own statistics.
  fn search(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>>;

  /// This index's BM25 statistics for `query`: document count, total
  /// length, and the document frequency of each distinct analysed query
  /// term (0 if absent). Sum these over segments with
  /// [`CorpusStats::merge`].
  fn stats_for(&self, query: &str) -> Result<CorpusStats>;

  /// Like [`TextSearcher::search`], but scores with the supplied
  /// statistics (typically summed over every segment of a collection),
  /// so scores are comparable across indexes. Every analysed query term
  /// must appear in `stats.terms`; otherwise this is an
  /// `InvalidArgument` error.
  fn search_with_stats(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
    stats: &CorpusStats,
  ) -> Result<Vec<Hit>>;
}
