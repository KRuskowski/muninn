//! Top-k BM25 retrieval with block-max MaxScore pruning.
//!
//! **Windows.** The doc-id space is processed in windows that end where
//! the current block of an *essential* term ends (one that draws
//! candidates; initially every term). In a window each term's bound is
//! the largest bound of its blocks that overlap the window
//! (`Cursor::window_bound`). Terms are sorted by that bound; the longest
//! prefix whose bounds together cannot beat the current k-th best score
//! is *non-essential*: a document matching only those terms cannot enter
//! the top k. A window where every term is non-essential is skipped
//! without decoding anything. Otherwise the window shrinks to the
//! essential terms' current blocks (bounds over a smaller window stay
//! valid), so each of them has exactly one decoded block in it.
//!
//! **Candidates** come from the *accumulated* terms: the essential ones
//! plus non-essential ones with no more postings than those (probing a
//! term for every candidate costs more than adding its postings, and
//! treating a term as essential is always exact). Three ways to walk a
//! window:
//!
//! * one accumulated term (`lead_window`): its block is scored at once,
//!   documents whose score plus the non-essential bounds cannot compete
//!   are dropped, and if some non-essential term is *required* (without
//!   it even the best lead score cannot compete) the candidates are
//!   intersected with that term by leapfrogging;
//! * several (`taat_window`): term at a time into per-document
//!   accumulators over chunks of `CHUNK` ids, then every matched
//!   document in id order;
//! * the last remaining term (`lone`): block by block, skipping blocks
//!   whose bound cannot compete without decoding them.
//!
//! **Evaluation** of a candidate (`Pruner::evaluate`): the non-essential
//! terms, highest bound first, first tightened to the bound of the block
//! that would hold the candidate (no decoding), then probed, while the
//! candidate can still beat the threshold.
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
//! * A pruning test sums bounds and contributions in whatever order is
//!   cheap and compares the sum, scaled by `1 +- 2 (n + 4) eps`, with the
//!   threshold: rounding cannot move an `n`-term sum of non-negative
//!   values further than that from the canonical sum. Only a sum within
//!   that band is re-summed in canonical order, with each term
//!   contributing its bound, its contribution or zero; `f32` addition is
//!   monotonic in each operand, so if every per-term bound is at least the
//!   term's contribution, that sum is at least the final score bit for
//!   bit. Queries with more than [`CANONICAL_MAX_TERMS`] distinct terms
//!   skip the re-sum and keep only the slack test.
//! * Per-term bounds are evaluated with the same arithmetic as scores,
//!   and that arithmetic is monotonic under `f32` rounding in tf and in
//!   length (see [`crate::bm25`]). So a document dominated by a frontier
//!   point (no higher tf, no shorter) scores at most the point's bound,
//!   bit for bit, and one equal to it scores exactly the bound.
//! * The filter is checked before a candidate is pushed, so filtered-out
//!   documents never raise the threshold.
//! * The exhaustive reference (`exhaustive`) shares only the cursors and
//!   the arithmetic with all of this.

use std::collections::HashMap;

use muninn_core::{DocFilter, DocId, Error, Hit, Result, TopK};

use crate::analyzer::{Analyzer, query_terms};
use crate::bitpack::BLOCK;
use crate::bm25::{K1, LenNorms, NormsCache, avg_doc_len, contrib, idf};
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
  /// Cache of the query-time length tables.
  fn norms_cache(&self) -> &NormsCache;
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
  if cursors.is_empty() {
    return Ok(Vec::new());
  }
  let norms = s.norms_cache().get(avg_doc_len(total, n));
  top_k(cursors, terms.len(), s.norms(), &norms, k, filter, prune)
}

fn corrupt() -> Error {
  Error::corrupt("text index postings are corrupt")
}

/// Up to this many distinct query terms, bound sums that come close to
/// the threshold are re-checked in canonical order (exact); beyond it,
/// only the slack test is used.
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

/// Runs the query over `cs`. With `prune == false` every matching
/// document is scored (the exhaustive reference).
fn top_k(
  cs: Vec<Cursor<'_>>,
  nterms: usize,
  doc_norms: &[u8],
  norms: &LenNorms,
  k: usize,
  filter: Option<&dyn DocFilter>,
  prune: bool,
) -> Result<Vec<Hit>> {
  let mut top = TopK::new(k);
  let corrupt_seen = if prune {
    let mut p = Pruner::new(cs, nterms, doc_norms, norms, filter, &mut top);
    p.run();
    p.corrupt
  } else {
    exhaustive(cs, nterms, doc_norms, norms, filter, &mut top)
  };
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

/// Scores every matching document, in doc-id order. Deliberately shares
/// nothing with the pruned search but the cursors and the arithmetic.
fn exhaustive(
  mut cs: Vec<Cursor<'_>>,
  nterms: usize,
  doc_norms: &[u8],
  norms: &LenNorms,
  filter: Option<&dyn DocFilter>,
  top: &mut TopK,
) -> bool {
  let mut cur = vec![0f32; nterms];
  loop {
    let d = cs.iter().map(Cursor::doc).min().unwrap_or(END);
    if d == END {
      break;
    }
    let pass = filter.is_none_or(|f| f.matches(d));
    let kc = norms.kc[usize::from(doc_norms[d as usize])];
    cur.fill(0.0);
    for c in cs.iter_mut() {
      if c.doc() == d {
        if pass {
          cur[c.idx] = contrib(c.weight, c.tf(), kc);
        }
        c.next();
      }
    }
    if pass {
      top.push(d, -canonical_sum(&cur));
    }
  }
  cs.iter().any(|c| c.corrupt)
}

/// The state of one pruned query (see the module docs).
struct Pruner<'a, 'c> {
  cs: Vec<Cursor<'c>>,
  doc_norms: &'a [u8],
  norms: &'a LenNorms,
  filter: Option<&'a dyn DocFilter>,
  top: &'a mut TopK,
  /// Score a document must beat (`-inf` until `k` are found).
  theta: f32,
  canonical: bool,
  slack: f32,
  inv_slack: f32,
  /// Cursor indexes sorted by window bound, ascending.
  order: Vec<usize>,
  /// Window bound per cursor index.
  wb: Vec<f32>,
  /// `prefix[j]`: sum of the `j + 1` smallest window bounds.
  prefix: Vec<f32>,
  /// Window bounds of the non-essential terms by query index, else 0.
  ne_vals: Vec<f32>,
  /// Number of non-essential terms: `order[..ne]`.
  ne: usize,
  /// Per candidate, by query index: contribution, bound, or 0.
  cur: Vec<f32>,
  scores: [f32; BLOCK],
  cands: [(DocId, f32); BLOCK],
  /// Chunk accumulators of the term-at-a-time path.
  acc: Vec<f32>,
  bits: [u64; CHUNK / 64],
  /// Next doc of each essential term in the window, by position.
  edoc: Vec<DocId>,
  /// Required non-essential terms of a lead block (cursor indexes).
  req: [usize; 8],
  corrupt: bool,
}

/// A non-essential term is accumulated with the essential ones when its
/// document frequency is at most this times theirs (see
/// `Pruner::accumulate_from`).
const ACCUMULATE: f64 = 1.0;

/// Documents per chunk of the term-at-a-time path.
const CHUNK: usize = 4096;

impl<'a, 'c> Pruner<'a, 'c> {
  fn new(
    cs: Vec<Cursor<'c>>,
    nterms: usize,
    doc_norms: &'a [u8],
    norms: &'a LenNorms,
    filter: Option<&'a dyn DocFilter>,
    top: &'a mut TopK,
  ) -> Self {
    let n = cs.len();
    let slack = 2.0 * (nterms as f32 + 4.0) * f32::EPSILON;
    Self {
      cs,
      doc_norms,
      norms,
      filter,
      top,
      theta: f32::NEG_INFINITY,
      canonical: nterms <= CANONICAL_MAX_TERMS,
      slack: 1.0 + slack,
      inv_slack: 1.0 - slack,
      order: Vec::with_capacity(n),
      wb: Vec::with_capacity(n),
      prefix: Vec::with_capacity(n),
      ne_vals: vec![0f32; nterms],
      ne: 0,
      cur: vec![0f32; nterms],
      scores: [0f32; BLOCK],
      cands: [(0, 0f32); BLOCK],
      acc: Vec::new(),
      bits: [0u64; CHUNK / 64],
      edoc: Vec::with_capacity(n),
      req: [0; 8],
      corrupt: false,
    }
  }

  /// Whether a document whose per-term values (contributions or bounds,
  /// by query index) are `vals`, summing to `sum` in some order, cannot
  /// beat the threshold.
  #[inline]
  fn hopeless(&self, sum: f32, vals: &[f32]) -> bool {
    match self.quick(sum) {
      Some(h) => h,
      None => canonical_sum(vals) <= self.theta,
    }
  }

  /// The slack test alone: `Some(hopeless)` when it decides, `None` when
  /// `sum` is within the slack of the threshold and only the canonical
  /// sum can.
  #[inline]
  fn quick(&self, sum: f32) -> Option<bool> {
    if sum * self.slack <= self.theta {
      Some(true)
    } else if !self.canonical || sum * self.inv_slack > self.theta {
      Some(false)
    } else {
      None
    }
  }

  /// Offers a fully scored document; raises the threshold and moves
  /// terms to non-essential if it was kept.
  #[inline]
  fn push(&mut self, d: DocId, score: f32) {
    if self.top.push(d, -score) {
      self.theta = -self.top.threshold();
      self.grow();
    }
  }

  /// Moves terms to non-essential while a document matching only
  /// non-essential terms provably cannot beat the threshold.
  fn grow(&mut self) {
    while self.ne < self.order.len() {
      let o = self.order[self.ne];
      let t = self.cs[o].idx;
      self.ne_vals[t] = self.wb[o];
      if !self.hopeless(self.prefix[self.ne], &self.ne_vals) {
        self.ne_vals[t] = 0.0;
        break;
      }
      self.ne += 1;
    }
  }

  fn run(&mut self) {
    let mut window_min: DocId = 0;
    loop {
      let mut i = 0;
      while i < self.cs.len() {
        self.cs[i].shallow_advance(window_min);
        if self.cs[i].exhausted() {
          let c = self.cs.swap_remove(i);
          self.corrupt |= c.corrupt;
          self.cur[c.idx] = 0.0;
        } else {
          i += 1;
        }
      }
      match self.cs.len() {
        0 => break,
        1 => {
          self.lone(window_min);
          break;
        }
        _ => {}
      }
      let window_max = self
        .cs
        .iter()
        .filter(|c| c.essential)
        .map(Cursor::block_last)
        .min()
        .unwrap_or_else(|| {
          self.cs.iter().map(Cursor::block_last).min().unwrap_or(END)
        });
      let alive = self.cs.len();
      self.wb.clear();
      for c in self.cs.iter_mut() {
        // A cursor already positioned past the window has nothing in it
        // (a stale doc is below `window_min`).
        let b = if c.doc() > window_max {
          0.0
        } else if c.essential {
          // Its current block holds the whole window.
          c.block_bound(self.norms)
        } else {
          c.window_bound(window_max, self.norms)
        };
        self.wb.push(b);
      }
      self.order.clear();
      self.order.extend(0..alive);
      let wb = &self.wb;
      self
        .order
        .sort_unstable_by(|&a, &b| wb[a].total_cmp(&wb[b]).then(a.cmp(&b)));
      self.prefix.clear();
      let mut acc = 0f32;
      for &o in &self.order {
        acc += self.wb[o];
        self.prefix.push(acc);
      }
      self.ne_vals.fill(0.0);
      self.ne = 0;
      self.grow();
      let mut window_max = window_max;
      for (j, &o) in self.order.iter().enumerate() {
        let c = &mut self.cs[o];
        c.essential = j >= self.ne;
        if c.essential {
          // Bounds over a smaller window stay valid: end the window
          // where an essential term's block ends, so each has one
          // decoded block in it.
          window_max = window_max.min(c.block_last());
        }
      }
      if self.ne == alive {
        // Nothing in this window can make the top k. The term with the
        // highest bound ends the next window.
        self.cs[self.order[alive - 1]].essential = true;
      } else {
        let from = self.accumulate_from(window_max);
        for &o in &self.order[from..self.ne] {
          window_max = window_max.min(self.cs[o].block_last());
        }
        if alive - from == 1 {
          self.lead_window(window_min, window_max);
        } else {
          self.taat_window(window_min, window_max, from);
        }
      }
      if window_max >= END - 1 {
        break;
      }
      window_min = window_max + 1;
    }
    // Every exit: a cursor that failed to decode in the last window is
    // still in `cs` and must be reported.
    self.corrupt |= self.cs.iter().any(|c| c.corrupt);
  }

  /// Sum of the non-essential terms' window bounds (any order).
  #[inline]
  fn ne_sum(&self) -> f32 {
    if self.ne == 0 {
      0.0
    } else {
      self.prefix[self.ne - 1]
    }
  }

  /// Finishes scoring candidate `d`. `order[..ne]` are the terms to
  /// probe; the others' values sum to `partial` and are in `cur`, or,
  /// with `lazy`, are looked up in their decoded blocks when needed.
  fn evaluate(&mut self, d: DocId, partial: f32, ne: usize, lazy: bool) {
    let n = self.doc_norms[d as usize];
    // Unvisited non-essential terms `order[..j]` count with their window
    // bounds, which `prefix[j - 1]` sums.
    let mut ready = !lazy;
    let mut filtered = false;
    let mut actual = partial;
    for j in (0..ne).rev() {
      let rest = if j == 0 { 0.0 } else { self.prefix[j - 1] };
      let c = &mut self.cs[self.order[j]];
      let b = c.peek_bound(d, n, self.norms);
      self.cur[c.idx] = b;
      if b == 0.0 {
        continue;
      }
      if self.check(actual + b + rest, d, n, ne, j, &mut ready) {
        return;
      }
      if !filtered {
        if self.filter.is_some_and(|f| !f.matches(d)) {
          return;
        }
        filtered = true;
      }
      let c = &mut self.cs[self.order[j]];
      let v = if c.advance(d) == d {
        self.norms.score(c.weight, c.tf(), n)
      } else {
        0.0
      };
      self.cur[c.idx] = v;
      actual += v;
      if self.check(actual + rest, d, n, ne, j, &mut ready) {
        return;
      }
    }
    if self.check(actual, d, n, ne, 0, &mut ready) {
      return;
    }
    if !filtered && self.filter.is_some_and(|f| !f.matches(d)) {
      return;
    }
    if !ready {
      self.fill(d, n, ne);
    }
    let score = canonical_sum(&self.cur);
    self.push(d, score);
  }

  /// `hopeless` for `evaluate`. If the canonical sum is needed, first
  /// completes `cur`: the essential terms' values (unless `ready`) and
  /// the window bounds of the unvisited terms `order[..unvisited]`.
  #[inline]
  fn check(
    &mut self,
    sum: f32,
    d: DocId,
    n: u8,
    ne: usize,
    unvisited: usize,
    ready: &mut bool,
  ) -> bool {
    match self.quick(sum) {
      Some(h) => h,
      None => {
        if !*ready {
          self.fill(d, n, ne);
          *ready = true;
        }
        for &o in &self.order[..unvisited] {
          self.cur[self.cs[o].idx] = self.wb[o];
        }
        canonical_sum(&self.cur) <= self.theta
      }
    }
  }

  /// Puts the contributions to `d` of the terms `order[ne..]`, whose
  /// blocks holding `d` are decoded, into `cur`.
  fn fill(&mut self, d: DocId, n: u8, ne: usize) {
    for &o in &self.order[ne..] {
      let c = &mut self.cs[o];
      self.cur[c.idx] = match c.decoded_tf(d) {
        Some(tf) => self.norms.score(c.weight, tf, n),
        None => 0.0,
      };
    }
  }

  /// Where the accumulated terms of a window start: the essential terms
  /// `order[ne..]`, plus non-essential terms (highest bound first) with
  /// at most [`ACCUMULATE`] times as many postings as the essential ones.
  /// Probing such a term for every candidate costs more than adding its
  /// postings; accumulating a non-essential term is always exact.
  fn accumulate_from(&self, window_max: DocId) -> usize {
    let mut from = self.ne;
    let df: u64 = self.order[from..]
      .iter()
      .map(|&o| u64::from(self.cs[o].df()))
      .sum();
    let limit = ACCUMULATE * df as f64;
    while from > 0 {
      let c = &self.cs[self.order[from - 1]];
      if c.doc() > window_max || f64::from(c.df()) > limit {
        break;
      }
      from -= 1;
    }
    from
  }

  /// Finds the non-essential terms (of `order[..lpos]`) that every lead
  /// candidate in `cands[..n]` needs to compete: those whose absence
  /// leaves the best lead score plus all other non-essential window
  /// bounds unable to beat the threshold. Puts them in `req`, sparsest
  /// first, and returns how many there are.
  fn required(&mut self, lpos: usize, n: usize) -> usize {
    let smax = self.cands[..n].iter().fold(0f32, |m, c| m.max(c.1));
    let mut nreq = 0;
    for j in 0..lpos {
      let mut rest = smax;
      for i in 0..lpos {
        if i != j {
          rest += self.wb[self.order[i]];
        }
      }
      if rest * self.slack <= self.theta && nreq < self.req.len() {
        self.req[nreq] = self.order[j];
        nreq += 1;
      }
    }
    let cs = &self.cs;
    self.req[..nreq].sort_unstable_by_key(|&o| cs[o].df());
    nreq
  }

  /// One essential term: walks its blocks in the window, scoring a block
  /// at a time and evaluating only documents that can still compete.
  fn lead_window(&mut self, window_min: DocId, window_max: DocId) {
    let lpos = self.ne;
    let li = self.order[lpos];
    if self.cs[li].advance(window_min) > window_max {
      return;
    }
    loop {
      let ne_sum = self.ne_sum();
      let (slack, theta) = (self.slack, self.theta);
      let c = &mut self.cs[li];
      let idx = c.idx;
      let w = c.weight;
      let (docs, tfs, pos) = c.decoded_block();
      let end = pos + docs[pos..].partition_point(|&d| d <= window_max);
      let len = docs.len();
      score_docs(
        w,
        &docs[pos..end],
        &tfs[pos..end],
        self.doc_norms,
        self.norms,
        &mut self.scores[..end - pos],
      );
      let mut n = 0;
      for (&d, &s) in docs[pos..end].iter().zip(&self.scores) {
        self.cands[n] = (d, s);
        n += usize::from((s + ne_sum) * slack > theta);
      }
      c.seek_pos(end);
      let nreq = self.required(lpos, n);
      let mut i = 0;
      'cand: while i < n {
        let (d, s) = self.cands[i];
        // A document without a required term cannot compete: leapfrog
        // over the candidates below the term's next document.
        for r in 0..nreq {
          let x = self.cs[self.req[r]].advance(d);
          if x != d {
            i += self.cands[i..n].partition_point(|c| c.0 < x);
            continue 'cand;
          }
        }
        self.cur[idx] = s;
        self.evaluate(d, s, lpos, false);
        if self.ne > lpos {
          // The lead became non-essential: nothing left can compete.
          return;
        }
        i += 1;
      }
      if end < len {
        return;
      }
      // The block ended inside the window: continue with the next one.
      let c = &mut self.cs[li];
      if c.exhausted() {
        return;
      }
      let first = c.block_first();
      if c.advance(first) > window_max {
        return;
      }
    }
  }

  /// Several essential terms: term at a time over chunks of the window.
  /// Each essential term has one decoded block in the window; its
  /// contributions are added into per-document accumulators, then every
  /// document that any of them matched is evaluated in id order.
  fn taat_window(&mut self, window_min: DocId, window_max: DocId, from: usize) {
    if self.acc.is_empty() {
      self.acc = vec![0f32; CHUNK];
    }
    let alive = self.order.len();
    self.edoc.clear();
    self.edoc.resize(alive, END);
    for j in from..alive {
      self.edoc[j] = self.cs[self.order[j]].advance(window_min);
    }
    let ne = from;
    loop {
      if self.ne == alive {
        return;
      }
      let lo = self.edoc[ne..].iter().copied().min().unwrap_or(END);
      if lo > window_max {
        return;
      }
      if alive - self.ne == 1 && self.ne >= ne {
        self.lead_window(lo, window_max);
        return;
      }
      let hi = (u64::from(lo) + CHUNK as u64).min(u64::from(window_max) + 1);
      let hi = hi as DocId;
      let mut top_slot = 0;
      for j in ne..alive {
        if self.edoc[j] >= hi {
          continue;
        }
        let c = &mut self.cs[self.order[j]];
        let w = c.weight;
        let (docs, tfs, pos) = c.decoded_block();
        let mut p = pos;
        while p < docs.len() && docs[p] < hi {
          let d = docs[p];
          let slot = (d - lo) as usize;
          let n = self.doc_norms[d as usize];
          self.acc[slot] += self.norms.score(w, tfs[p], n);
          self.bits[slot >> 6] |= 1 << (slot & 63);
          p += 1;
        }
        if p > pos {
          top_slot = top_slot.max((docs[p - 1] - lo) as usize);
        }
        self.edoc[j] = docs.get(p).copied().unwrap_or(END);
        c.seek_pos(p);
      }
      let ne_sum = if ne == 0 { 0.0 } else { self.prefix[ne - 1] };
      for wi in 0..=top_slot / 64 {
        let mut word = std::mem::take(&mut self.bits[wi]);
        while word != 0 {
          let slot = wi * 64 + word.trailing_zeros() as usize;
          word &= word - 1;
          let partial = std::mem::take(&mut self.acc[slot]);
          if (partial + ne_sum) * self.slack > self.theta {
            self.evaluate(lo + slot as DocId, partial, ne, true);
          }
        }
      }
    }
  }

  /// The last remaining term: block by block, skipping blocks whose
  /// bound cannot beat the threshold without decoding them.
  fn lone(&mut self, window_min: DocId) {
    let c = &mut self.cs[0];
    let idx = c.idx;
    let w = c.weight;
    loop {
      if c.exhausted() {
        break;
      }
      if c.block_bound(self.norms) <= self.theta {
        c.end_block();
        continue;
      }
      let target = c.block_first().max(window_min);
      if c.advance(target) == END {
        break;
      }
      let (docs, tfs, pos) = c.decoded_block();
      let n = docs.len() - pos;
      score_docs(
        w,
        &docs[pos..],
        &tfs[pos..],
        self.doc_norms,
        self.norms,
        &mut self.scores[..n],
      );
      for (i, &s) in self.scores[..n].iter().enumerate() {
        if s > self.theta {
          let d = docs[pos + i];
          if self.filter.is_none_or(|f| f.matches(d)) {
            // With one term alive, the canonical sum is `s` itself.
            self.cur[idx] = s;
            if self.top.push(d, -canonical_sum(&self.cur)) {
              self.theta = -self.top.threshold();
            }
          }
        }
      }
      c.end_block();
    }
    self.corrupt |= c.corrupt;
  }
}

/// `out[i] = contrib(w, tfs[i], kc[norm of docs[i]])`.
#[inline]
fn score_docs(
  w: f32,
  docs: &[u32],
  tfs: &[u32],
  doc_norms: &[u8],
  norms: &LenNorms,
  out: &mut [f32],
) {
  for ((o, &d), &tf) in out.iter_mut().zip(docs).zip(tfs) {
    *o = norms.score(w, tf, doc_norms[d as usize]);
  }
}
