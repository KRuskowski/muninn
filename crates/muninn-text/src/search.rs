//! Top-k BM25 retrieval with block-max MaxScore pruning.
//!
//! The doc-id space is processed in windows. A window ends where the
//! first current block of any query term ends, so inside a window every
//! term has one block and that block's impacts give an upper bound on
//! the term's contribution to any document in the window. Terms are
//! sorted by that bound; the longest prefix whose bounds sum to no more
//! than the current k-th best score is *non-essential*: a document
//! matching only those terms cannot enter the top-k. Candidates are
//! therefore drawn only from the *essential* terms, and non-essential
//! terms are probed (with skipping) only while the candidate can still
//! beat the threshold. A window where every term is non-essential is
//! skipped without decoding a single block.
//!
//! Correctness details:
//!
//! * Documents are visited in increasing id order, so a later document
//!   with a score equal to the threshold can never displace a kept one
//!   (ties go to the smaller id). Pruning on `bound <= threshold` is
//!   therefore exact.
//! * Bounds are compared after multiplying by `1 + 2 (n + 4) eps`, which
//!   covers `f32` rounding differences between summing bounds and summing
//!   actual contributions in another order.
//! * Final scores are summed in query-term order, independent of the
//!   traversal, so equal documents get bit-identical scores.
//! * The filter is checked before a candidate is scored, so filtered-out
//!   documents never raise the threshold.

use muninn_core::{DocFilter, DocId, Error, Hit, Result, TopK};

use crate::analyzer::{Analyzer, query_terms};
use crate::bm25::{K1, LenNorms, avg_doc_len, contrib, idf};
use crate::postings::{Cursor, END};
use crate::stats::{CorpusStats, TermDocFreq};

/// What the search needs from an index.
pub(crate) trait Source {
  fn analyzer(&self) -> &Analyzer;
  fn doc_count(&self) -> u64;
  fn total_doc_len(&self) -> u64;
  /// One length norm per doc id.
  fn norms(&self) -> &[u8];
  fn doc_freq(&self, term: &str) -> Result<u32>;
  fn cursor(&self, term: &str) -> Result<Option<Cursor<'_>>>;
}

pub(crate) fn stats_for<S: Source>(s: &S, query: &str) -> Result<CorpusStats> {
  let mut terms = Vec::new();
  for term in query_terms(s.analyzer(), query) {
    let doc_freq = u64::from(s.doc_freq(&term)?);
    terms.push(TermDocFreq { term, doc_freq });
  }
  Ok(CorpusStats {
    doc_count: s.doc_count(),
    total_doc_len: s.total_doc_len(),
    terms,
  })
}

pub(crate) fn search<S: Source>(
  s: &S,
  query: &str,
  k: usize,
  filter: Option<&dyn DocFilter>,
  stats: Option<&CorpusStats>,
  prune: bool,
) -> Result<Vec<Hit>> {
  if k == 0 {
    return Ok(Vec::new());
  }
  let terms = query_terms(s.analyzer(), query);
  let (n, total) = match stats {
    Some(st) => (st.doc_count, st.total_doc_len),
    None => (s.doc_count(), s.total_doc_len()),
  };
  let mut cursors = Vec::with_capacity(terms.len());
  for (i, term) in terms.iter().enumerate() {
    let global_df = match stats {
      Some(st) => Some(st.doc_freq(term).ok_or_else(|| {
        Error::invalid(format!(
          "corpus statistics have no entry for query term {term:?}"
        ))
      })?),
      None => None,
    };
    if let Some(mut c) = s.cursor(term)? {
      if c.corrupt {
        return Err(corrupt());
      }
      let df = global_df.unwrap_or(u64::from(c.df()));
      c.weight = idf(df, n) * (K1 + 1.0);
      c.idx = i;
      cursors.push(c);
    }
  }
  let norms = LenNorms::new(avg_doc_len(total, n));
  top_k(cursors, terms.len(), s.norms(), &norms, k, filter, prune)
}

fn corrupt() -> Error {
  Error::corrupt("text index postings are corrupt")
}

/// Runs the query over `cs`. With `prune == false` every matching
/// document is scored (the exhaustive reference).
fn top_k(
  mut cs: Vec<Cursor<'_>>,
  nterms: usize,
  doc_norms: &[u8],
  norms: &LenNorms,
  k: usize,
  filter: Option<&dyn DocFilter>,
  prune: bool,
) -> Result<Vec<Hit>> {
  let slack = 1.0 + 2.0 * (nterms as f32 + 4.0) * f32::EPSILON;
  let mut top = TopK::new(k);
  let mut parts = vec![0f32; nterms];
  let mut order: Vec<usize> = Vec::with_capacity(cs.len());
  let mut wb = vec![0f32; cs.len()];
  let mut prefix = vec![0f32; cs.len()];
  let mut corrupt_seen = false;
  let threshold = |top: &TopK| {
    if prune {
      -top.threshold()
    } else {
      f32::NEG_INFINITY
    }
  };
  let mut window_min: DocId = 0;
  loop {
    let mut i = 0;
    while i < cs.len() {
      cs[i].shallow_advance(window_min);
      if cs[i].exhausted() {
        corrupt_seen |= cs[i].corrupt;
        cs.swap_remove(i);
      } else {
        i += 1;
      }
    }
    if cs.is_empty() {
      break;
    }
    let window_max = cs.iter().map(|c| c.block_last()).min().unwrap_or(END);
    let alive = cs.len();
    for (b, c) in wb.iter_mut().zip(cs.iter_mut()) {
      // A cursor already positioned past the window has nothing in it.
      *b = if c.doc() > window_max {
        0.0
      } else {
        c.block_bound(norms)
      };
    }
    order.clear();
    order.extend(0..alive);
    order.sort_unstable_by(|&a, &b| wb[a].total_cmp(&wb[b]).then(a.cmp(&b)));
    let mut acc = 0f32;
    for (p, &o) in prefix.iter_mut().zip(&order) {
      acc += wb[o];
      *p = acc;
    }
    let mut theta = threshold(&top);
    let mut ne = 0;
    while ne < alive && prefix[ne] * slack <= theta {
      ne += 1;
    }
    if ne < alive {
      for &o in &order[ne..] {
        cs[o].advance(window_min);
      }
      loop {
        let mut d = END;
        for &o in &order[ne..] {
          d = d.min(cs[o].doc());
        }
        if d > window_max {
          break;
        }
        if filter.is_some_and(|f| !f.matches(d)) {
          for &o in &order[ne..] {
            if cs[o].doc() == d {
              cs[o].next();
            }
          }
          continue;
        }
        let kc = norms.kc[usize::from(doc_norms[d as usize])];
        let mut partial = 0f32;
        for &o in &order[ne..] {
          let c = &mut cs[o];
          if c.doc() == d {
            let v = contrib(c.weight, c.tf(), kc);
            parts[c.idx] = v;
            partial += v;
            c.next();
          }
        }
        let mut pruned = false;
        for j in (0..ne).rev() {
          if (partial + prefix[j]) * slack <= theta {
            pruned = true;
            break;
          }
          let c = &mut cs[order[j]];
          if c.advance(d) == d {
            let v = contrib(c.weight, c.tf(), kc);
            parts[c.idx] = v;
            partial += v;
          }
        }
        if !pruned {
          let mut score = 0f32;
          for &v in &parts {
            score += v;
          }
          if top.push(d, -score) && prune {
            theta = threshold(&top);
            while ne < alive && prefix[ne] * slack <= theta {
              ne += 1;
            }
          }
        }
        parts.fill(0.0);
      }
    }
    if window_max >= END - 1 {
      break;
    }
    window_min = window_max + 1;
  }
  corrupt_seen |= cs.iter().any(|c| c.corrupt);
  if corrupt_seen {
    return Err(corrupt());
  }
  Ok(
    top
      .into_sorted_vec()
      .into_iter()
      .map(|(id, d)| Hit::new(id, -d))
      .collect(),
  )
}
