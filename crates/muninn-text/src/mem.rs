//! The in-memory, appendable text index (the memtable's text field).

use std::collections::HashMap;
use std::io::Write;

use muninn_core::{AlignedBuf, DocFilter, DocId, Error, Hit, Region, Result};

use crate::TextSearcher;
use crate::analyzer::{Analyzer, Tokens};
use crate::bm25::NormsCache;
use crate::frozen::{ABSENT, TextIndex, Writer};
use crate::hash::FoldBuild;
use crate::norm::encode_doc_len;
use crate::postings::{Cursor, END, PostingList};
use crate::search::{self, Source};
use crate::stats::CorpusStats;

/// An appendable full-text index, searchable after every [`add`].
///
/// Postings are compressed as they arrive: every 128 postings of a term
/// are bit-packed into the same block format the on-disk index uses,
/// with the same skip data and score bounds, so search here uses the
/// same pruned algorithm as [`TextIndex`]. Writing it out
/// ([`MemTextIndex::write_to`]) copies the packed blocks verbatim.
///
/// Ids must be strictly increasing across [`add`] calls; gaps are
/// allowed (documents without this field) and count as absent.
///
/// [`add`]: MemTextIndex::add
#[derive(Debug)]
pub struct MemTextIndex {
  analyzer: Analyzer,
  terms: HashMap<Box<str>, u32, FoldBuild>,
  lists: Vec<PostingList>,
  /// Per term: 1 + its slot in `doc_terms` while a document is being
  /// added, otherwise 0.
  slots: Vec<u32>,
  norms: Vec<u8>,
  lens: Vec<u32>,
  doc_count: u64,
  total_len: u64,
  tokens: Tokens,
  doc_terms: Vec<(u32, u32)>,
  norms_cache: NormsCache,
}

impl MemTextIndex {
  /// An empty index using `analyzer`.
  pub fn new(analyzer: Analyzer) -> Self {
    Self {
      analyzer,
      terms: HashMap::with_hasher(FoldBuild),
      lists: Vec::new(),
      slots: Vec::new(),
      norms: Vec::new(),
      lens: Vec::new(),
      doc_count: 0,
      total_len: 0,
      tokens: Tokens::new(),
      doc_terms: Vec::new(),
      norms_cache: NormsCache::default(),
    }
  }

  /// The analyzer applied to documents and queries.
  pub fn analyzer(&self) -> &Analyzer {
    &self.analyzer
  }

  /// Indexes `text` as document `id`. `id` must be larger than every id
  /// added before and below `u32::MAX`.
  pub fn add(&mut self, id: DocId, text: &str) -> Result<()> {
    if (id as usize) < self.norms.len() {
      return Err(Error::invalid(format!(
        "document id {id} is not above the last added id {}",
        self.norms.len() - 1
      )));
    }
    if id == END {
      return Err(Error::invalid("document id u32::MAX is reserved"));
    }
    let mut tokens = std::mem::take(&mut self.tokens);
    self.analyzer.analyze(text, &mut tokens);
    self.doc_terms.clear();
    for tok in tokens.iter() {
      let tid = match self.terms.get(tok) {
        Some(&t) => t,
        None => {
          let t = self.lists.len() as u32;
          self.terms.insert(Box::from(tok), t);
          self.lists.push(PostingList::default());
          self.slots.push(0);
          t
        }
      };
      let slot = &mut self.slots[tid as usize];
      if *slot == 0 {
        self.doc_terms.push((tid, 1));
        *slot = self.doc_terms.len() as u32;
      } else {
        self.doc_terms[*slot as usize - 1].1 += 1;
      }
    }
    let len = u32::try_from(tokens.len())
      .unwrap_or(ABSENT - 1)
      .min(ABSENT - 1);
    self.tokens = tokens;
    let i = id as usize;
    self.norms.resize(i + 1, 0);
    self.lens.resize(i + 1, ABSENT);
    self.norms[i] = encode_doc_len(len);
    self.lens[i] = len;
    for &(tid, tf) in &self.doc_terms {
      self.slots[tid as usize] = 0;
      self.lists[tid as usize].push(id, tf, &self.norms);
    }
    self.doc_count += 1;
    self.total_len += u64::from(len);
    Ok(())
  }

  /// Number of documents added (gaps excluded).
  pub fn doc_count(&self) -> u64 {
    self.doc_count
  }

  /// Sum of document lengths in tokens.
  pub fn total_doc_len(&self) -> u64 {
    self.total_len
  }

  /// Number of distinct terms.
  pub fn num_terms(&self) -> usize {
    self.lists.len()
  }

  /// Length in tokens of document `id`, or `None` if it was not added.
  pub fn doc_len(&self, id: DocId) -> Option<u32> {
    self.lens.get(id as usize).copied().filter(|&l| l != ABSENT)
  }

  /// Approximate heap memory in bytes, for flush decisions.
  pub fn heap_bytes(&self) -> usize {
    let term_bytes: usize = self.terms.keys().map(|k| k.len() + 16).sum();
    let lists: usize = self.lists.iter().map(PostingList::heap_bytes).sum();
    term_bytes
      + self.terms.capacity() * (std::mem::size_of::<(Box<str>, u32)>() + 1)
      + lists
      + self.lists.capacity() * std::mem::size_of::<PostingList>()
      + self.slots.capacity() * 4
      + self.norms.capacity()
      + self.lens.capacity() * 4
  }

  /// Writes the index as a frozen [`TextIndex`] container in one
  /// streaming pass and returns the writer (flushing is the caller's).
  pub fn write_to<W: Write>(&self, out: W) -> Result<W> {
    let mut sorted: Vec<(&str, u32)> =
      self.terms.iter().map(|(k, &v)| (&**k, v)).collect();
    sorted.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut w = Writer::new(
      out,
      self.analyzer.config(),
      &self.norms,
      &self.lens,
      self.doc_count,
      self.total_len,
    )?;
    for (term, tid) in sorted {
      let f = self.lists[tid as usize].freeze(&self.norms);
      w.add_term(term.as_bytes(), &f)?;
    }
    w.finish()
  }

  /// Writes the index to memory and opens it as a [`TextIndex`].
  pub fn freeze(&self) -> Result<TextIndex> {
    let buf = self.write_to(AlignedBuf::new())?;
    TextIndex::open_with_analyzer(Region::new(buf), self.analyzer.clone())
  }

  /// BM25 search without pruning: scores every matching document. Same
  /// results as [`TextSearcher::search`]; a baseline for tests and
  /// benchmarks.
  pub fn search_exhaustive(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
    stats: Option<&CorpusStats>,
  ) -> Result<Vec<Hit>> {
    search::search(self, query, k, filter, stats, false)
  }
}

impl Source for MemTextIndex {
  fn analyzer(&self) -> &Analyzer {
    &self.analyzer
  }

  fn doc_count(&self) -> u64 {
    self.doc_count
  }

  fn total_doc_len(&self) -> u64 {
    self.total_len
  }

  fn norms(&self) -> &[u8] {
    &self.norms
  }

  fn doc_freq(&self, term: &str) -> Result<u32> {
    Ok(
      self
        .terms
        .get(term)
        .map_or(0, |&t| self.lists[t as usize].df()),
    )
  }

  fn norms_cache(&self) -> &NormsCache {
    &self.norms_cache
  }

  fn cursor(&self, term: &str) -> Result<Option<Cursor<'_>>> {
    Ok(
      self
        .terms
        .get(term)
        .map(|&t| self.lists[t as usize].cursor(&self.norms)),
    )
  }
}

impl TextSearcher for MemTextIndex {
  fn num_docs(&self) -> usize {
    self.norms.len()
  }

  fn search(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>> {
    search::search(self, query, k, filter, None, true)
  }

  fn stats_for(&self, query: &str) -> Result<CorpusStats> {
    search::stats_for(self, query)
  }

  fn search_with_stats(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
    stats: &CorpusStats,
  ) -> Result<Vec<Hit>> {
    search::search(self, query, k, filter, Some(stats), true)
  }
}
