//! Corpus statistics for scoring across segments.

/// Document frequency of one analysed query term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TermDocFreq {
  /// The analysed term, as the index stores it.
  pub term: String,
  /// Number of documents containing it.
  pub doc_freq: u64,
}

/// BM25 statistics for one query over some set of documents.
///
/// A collection is split into many segments, and per-segment IDF and
/// average length make scores from different segments incomparable. The
/// engine therefore asks every segment for [`crate::TextSearcher::stats_for`],
/// sums them with [`CorpusStats::merge`], and passes the total to
/// [`crate::TextSearcher::search_with_stats`] on each segment. Every
/// segment then scores with the same `N`, average length and document
/// frequencies, exactly as if the collection were one index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CorpusStats {
  /// Number of documents (`N`).
  pub doc_count: u64,
  /// Sum of exact document lengths, in tokens.
  pub total_doc_len: u64,
  /// Document frequency per distinct analysed query term, in query
  /// order. Terms absent from an index have `doc_freq` 0.
  pub terms: Vec<TermDocFreq>,
}

impl CorpusStats {
  /// Adds `other` into `self`: document counts, lengths and per-term
  /// frequencies are summed; terms only in `other` are appended.
  pub fn merge(&mut self, other: &CorpusStats) {
    self.doc_count += other.doc_count;
    self.total_doc_len += other.total_doc_len;
    for t in &other.terms {
      match self.terms.iter_mut().find(|s| s.term == t.term) {
        Some(s) => s.doc_freq += t.doc_freq,
        None => self.terms.push(t.clone()),
      }
    }
  }

  /// Document frequency of `term`, if these statistics cover it.
  pub fn doc_freq(&self, term: &str) -> Option<u64> {
    self
      .terms
      .iter()
      .find(|t| t.term == term)
      .map(|t| t.doc_freq)
  }

  /// Average document length used by BM25.
  pub fn avg_doc_len(&self) -> f32 {
    crate::bm25::avg_doc_len(self.total_doc_len, self.doc_count)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn t(term: &str, doc_freq: u64) -> TermDocFreq {
    TermDocFreq {
      term: term.into(),
      doc_freq,
    }
  }

  #[test]
  fn merge_sums_and_unions() {
    let mut a = CorpusStats {
      doc_count: 10,
      total_doc_len: 100,
      terms: vec![t("x", 3), t("y", 0)],
    };
    let b = CorpusStats {
      doc_count: 5,
      total_doc_len: 20,
      terms: vec![t("y", 2), t("z", 1)],
    };
    a.merge(&b);
    assert_eq!(a.doc_count, 15);
    assert_eq!(a.total_doc_len, 120);
    assert_eq!(a.doc_freq("x"), Some(3));
    assert_eq!(a.doc_freq("y"), Some(2));
    assert_eq!(a.doc_freq("z"), Some(1));
    assert_eq!(a.doc_freq("w"), None);
    assert_eq!(a.avg_doc_len(), 8.0);
  }
}
