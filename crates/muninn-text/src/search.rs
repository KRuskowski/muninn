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
//!   therefore exact, which matters: with short documents many scores
//!   tie exactly, and a bound that ties the threshold must prune.
//! * Final scores are summed in query-term order, independent of the
//!   traversal, so equal documents get bit-identical scores.
//! * Bounds are summed in that same canonical order, with each term
//!   contributing its bound, its actual contribution, or zero. `f32`
//!   addition is monotonic in each operand, so if every per-term bound
//!   is at least the term's actual contribution, the bound sum is at
//!   least the final score bit for bit, with no rounding slack. Queries
//!   with more than [`CANONICAL_MAX_TERMS`] distinct terms instead sum
//!   bounds in any order and multiply by `1 + 2 (n + 4) eps`.
//! * Per-term bounds are evaluated with the same arithmetic as scores,
//!   and that arithmetic is monotonic under `f32` rounding in tf and in
//!   length (see [`crate::bm25`]). So a document dominated by a frontier
//!   point (no higher tf, no shorter) scores at most the point's bound,
//!   bit for bit, and one equal to it scores exactly the bound.
//! * The filter is checked before a candidate is scored, so filtered-out
//!   documents never raise the threshold.

use std::collections::HashMap;

use muninn_core::{DocFilter, DocId, Error, Hit, Result, TopK};

use crate::analyzer::{Analyzer, query_terms};
use crate::bm25::{K1, LenNorms, avg_doc_len, contrib, idf};
use crate::hash::FoldBuild;
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
  for term in query_terms(s.analyzer(), query)? {
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
  // More results than documents cannot exist; this also keeps a huge k
  // from sizing the top-k heap.
  let k = k.min(s.norms().len());
  let terms = query_terms(s.analyzer(), query)?;
  if k == 0 {
    return Ok(Vec::new());
  }
  let global: Option<HashMap<&str, u64, FoldBuild>> = stats.map(|st| {
    let mut m = HashMap::with_capacity_and_hasher(st.terms.len(), FoldBuild);
    for t in &st.terms {
      m.entry(t.term.as_str()).or_insert(t.doc_freq);
    }
    m
  });
  let (n, total) = match stats {
    Some(st) => (st.doc_count, st.total_doc_len),
    None => (s.doc_count(), s.total_doc_len()),
  };
  let mut cursors = Vec::with_capacity(terms.len());
  for (i, term) in terms.iter().enumerate() {
    let global_df = match &global {
      Some(g) => Some(g.get(term.as_str()).copied().ok_or_else(|| {
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

/// Up to this many distinct query terms, bound sums are computed in
/// canonical order (exact, `O(n^2)` per window); beyond it, with slack.
pub(crate) const CANONICAL_MAX_TERMS: usize = 32;

/// Sums `v` in index order: the canonical order of scores and bounds.
#[inline]
fn canonical_sum(v: &[f32]) -> f32 {
  let mut s = 0f32;
  for &x in v {
    s += x;
  }
  s
}

/// Per-window pruning state.
struct Partition {
  canonical: bool,
  slack: f32,
  /// Cursor indexes sorted by window bound, ascending.
  order: Vec<usize>,
  /// Window bound per cursor index.
  wb: Vec<f32>,
  /// Query-term index per cursor index.
  tidx: Vec<usize>,
  /// `prefix[j]`: sum of the `j + 1` smallest bounds (slack mode).
  prefix: Vec<f32>,
  /// Bounds of the non-essential terms by query index, 0 elsewhere
  /// (canonical mode).
  ne_vals: Vec<f32>,
  /// Number of non-essential terms: `order[..ne]`.
  ne: usize,
}

impl Partition {
  /// Moves terms to non-essential while a document matching only
  /// non-essential terms provably cannot beat `theta`.
  fn grow(&mut self, theta: f32) {
    while self.ne < self.order.len() {
      let o = self.order[self.ne];
      let fits = if self.canonical {
        let t = self.tidx[o];
        self.ne_vals[t] = self.wb[o];
        let fits = canonical_sum(&self.ne_vals) <= theta;
        if !fits {
          self.ne_vals[t] = 0.0;
        }
        fits
      } else {
        let scale = if self.ne == 0 { 1.0 } else { self.slack };
        self.prefix[self.ne] * scale <= theta
      };
      if !fits {
        break;
      }
      self.ne += 1;
    }
  }
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
  let mut top = TopK::new(k);
  let mut p = Partition {
    canonical: nterms <= CANONICAL_MAX_TERMS,
    slack: 1.0 + 2.0 * (nterms as f32 + 4.0) * f32::EPSILON,
    order: Vec::with_capacity(cs.len()),
    wb: Vec::with_capacity(cs.len()),
    tidx: Vec::with_capacity(cs.len()),
    prefix: Vec::with_capacity(cs.len()),
    ne_vals: vec![0f32; nterms],
    ne: 0,
  };
  // Per candidate, by query index: actual contribution, bound of a
  // not-yet-probed non-essential term, or 0.
  let mut cur = vec![0f32; nterms];
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
    p.wb.clear();
    p.tidx.clear();
    for c in cs.iter_mut() {
      // A cursor already positioned past the window has nothing in it.
      p.wb.push(if c.doc() > window_max {
        0.0
      } else {
        c.block_bound(norms)
      });
      p.tidx.push(c.idx);
    }
    p.order.clear();
    p.order.extend(0..alive);
    let wb = &p.wb;
    p.order
      .sort_unstable_by(|&a, &b| wb[a].total_cmp(&wb[b]).then(a.cmp(&b)));
    p.prefix.clear();
    let mut acc = 0f32;
    for &o in &p.order {
      acc += p.wb[o];
      p.prefix.push(acc);
    }
    let mut theta = threshold(&top);
    p.ne_vals.fill(0.0);
    p.ne = 0;
    p.grow(theta);
    if p.ne == 1 && alive == 1 {
      // A lone term whose block cannot beat the threshold: scan the skip
      // records for the next block that can.
      window_min = cs[0].skip_blocks_at_most(theta, norms);
      if window_min == END {
        break;
      }
      continue;
    }
    if p.ne == 0 && alive == 1 {
      // One term in this window: walk its block directly.
      let c = &mut cs[p.order[0]];
      let block_bound = p.wb[p.order[0]];
      let mut d = c.advance(window_min);
      while d <= window_max {
        if filter.is_none_or(|f| f.matches(d)) {
          let kc = norms.kc[usize::from(doc_norms[d as usize])];
          let v = contrib(c.weight, c.tf(), kc);
          if v > theta {
            top.push(d, -v);
            theta = threshold(&top);
            if block_bound <= theta {
              break;
            }
          }
        }
        d = c.next();
      }
    } else if p.ne < alive {
      for &o in &p.order[p.ne..] {
        cs[o].advance(window_min);
      }
      loop {
        let ne = p.ne;
        let mut d = END;
        for &o in &p.order[ne..] {
          d = d.min(cs[o].doc());
        }
        if d > window_max {
          break;
        }
        if filter.is_some_and(|f| !f.matches(d)) {
          for &o in &p.order[ne..] {
            if cs[o].doc() == d {
              cs[o].next();
            }
          }
          continue;
        }
        let kc = norms.kc[usize::from(doc_norms[d as usize])];
        if p.canonical {
          cur.copy_from_slice(&p.ne_vals);
        } else {
          cur.fill(0.0);
        }
        let mut partial = 0f32;
        for &o in &p.order[ne..] {
          let c = &mut cs[o];
          if c.doc() == d {
            let v = contrib(c.weight, c.tf(), kc);
            cur[c.idx] = v;
            partial += v;
            c.next();
          }
        }
        let mut pruned = false;
        for j in (0..ne).rev() {
          let hopeless = if p.canonical {
            canonical_sum(&cur) <= theta
          } else {
            (partial + p.prefix[j]) * p.slack <= theta
          };
          if hopeless {
            pruned = true;
            break;
          }
          let c = &mut cs[p.order[j]];
          let v = if c.advance(d) == d {
            contrib(c.weight, c.tf(), kc)
          } else {
            0.0
          };
          cur[c.idx] = v;
          partial += v;
        }
        if !pruned && top.push(d, -canonical_sum(&cur)) && prune {
          theta = threshold(&top);
          p.grow(theta);
        }
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
