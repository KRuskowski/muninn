//! Shared test helpers: a PRNG, a Zipfian corpus generator, and an
//! exhaustive BM25 reference that works from raw text, independent of
//! every index structure.

#![allow(dead_code)]

use std::collections::HashMap;

use muninn_core::{DocFilter, DocId, Hit};
use muninn_text::norm::{decode_doc_len, encode_doc_len};
use muninn_text::{Analyzer, CorpusStats, Tokens};

/// xorshift64* PRNG.
pub struct Rng(u64);

impl Rng {
  pub fn new(seed: u64) -> Self {
    Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
  }

  pub fn next_u64(&mut self) -> u64 {
    self.0 ^= self.0 >> 12;
    self.0 ^= self.0 << 25;
    self.0 ^= self.0 >> 27;
    self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
  }

  pub fn below(&mut self, n: u64) -> u64 {
    self.next_u64() % n
  }

  pub fn chance(&mut self, p: f64) -> bool {
    ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
  }
}

/// Samples ranks `0..n` with probability proportional to `1 / (r+1)^s`.
pub struct Zipf {
  cdf: Vec<f64>,
}

impl Zipf {
  pub fn new(n: usize, s: f64) -> Self {
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for r in 0..n {
      acc += 1.0 / ((r + 1) as f64).powf(s);
      cdf.push(acc);
    }
    for c in &mut cdf {
      *c /= acc;
    }
    Self { cdf }
  }

  pub fn sample(&self, rng: &mut Rng) -> usize {
    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    self.cdf.partition_point(|&c| c < u).min(self.cdf.len() - 1)
  }
}

/// A deterministic pseudo-word for a vocabulary rank. Some words carry
/// non-ASCII letters or mixed case, to exercise the analyzer.
pub fn word(rank: usize) -> String {
  const SYL: [&str; 16] = [
    "ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "ba", "de", "fo", "gu",
    "ha", "je", "pe", "zo",
  ];
  let mut w = String::new();
  let mut r = rank + 1;
  while r > 0 {
    w.push_str(SYL[r % 16]);
    r /= 16;
  }
  match rank % 23 {
    5 => w.push('é'),
    11 => w.insert(0, 'Ü'),
    17 => w = w.to_uppercase(),
    _ => {}
  }
  w
}

/// A corpus: `docs[i] = (id, text)` with strictly increasing ids.
pub struct Corpus {
  pub docs: Vec<(DocId, String)>,
  pub vocab: usize,
}

/// Generates `n` documents over a Zipfian vocabulary of `vocab` words.
/// Lengths mix short, medium and long documents; some documents repeat
/// a word many times (high tf in long documents), and ids have gaps.
pub fn zipf_corpus(seed: u64, n: usize, vocab: usize) -> Corpus {
  let mut rng = Rng::new(seed);
  let zipf = Zipf::new(vocab, 1.05);
  let mut docs = Vec::with_capacity(n);
  let mut id: DocId = rng.below(3) as DocId;
  let seps = [" ", "  ", ", ", ". ", "\n", " - ", "; "];
  for _ in 0..n {
    let len = match rng.below(100) {
      0..=69 => 1 + rng.below(12),
      70..=94 => 12 + rng.below(50),
      _ => 60 + rng.below(340),
    } as usize;
    let mut text = String::new();
    for i in 0..len {
      if i > 0 {
        text.push_str(seps[rng.below(seps.len() as u64) as usize]);
      }
      text.push_str(&word(zipf.sample(&mut rng)));
    }
    if rng.chance(0.08) {
      let w = word(zipf.sample(&mut rng));
      for _ in 0..1 + rng.below(60) {
        text.push(' ');
        text.push_str(&w);
      }
    }
    if rng.chance(0.01) {
      text.clear();
    }
    docs.push((id, text));
    id += 1
      + if rng.chance(0.1) {
        rng.below(5) as DocId
      } else {
        0
      };
  }
  Corpus { docs, vocab }
}

/// A random query: 1..=6 terms mixing frequent, rare, unknown and
/// repeated words, with arbitrary separators and case.
pub fn random_query(rng: &mut Rng, vocab: usize) -> String {
  let zipf = Zipf::new(vocab, 1.05);
  let n = 1 + rng.below(6);
  let mut words: Vec<String> = Vec::new();
  for _ in 0..n {
    let w = match rng.below(10) {
      0..=4 => word(zipf.sample(rng)),
      5..=7 => word(rng.below(vocab as u64) as usize),
      8 => format!("unknownqx{}", rng.below(1000)),
      _ => words.last().cloned().unwrap_or_else(|| word(0)),
    };
    words.push(if rng.chance(0.3) { w.to_uppercase() } else { w });
  }
  words.join(if rng.chance(0.5) { " " } else { ", " })
}

/// Per-document analysed term frequencies, precomputed for the reference.
pub struct Reference {
  analyzer: Analyzer,
  docs: Vec<(DocId, u32, HashMap<String, u32>)>,
}

impl Reference {
  pub fn new(analyzer: &Analyzer, docs: &[(DocId, String)]) -> Self {
    let mut toks = Tokens::new();
    let docs = docs
      .iter()
      .map(|(id, text)| {
        analyzer.analyze(text, &mut toks);
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in toks.iter() {
          *tf.entry(t.to_string()).or_default() += 1;
        }
        (*id, toks.len() as u32, tf)
      })
      .collect();
    Self {
      analyzer: analyzer.clone(),
      docs,
    }
  }

  /// Distinct analysed query terms in order of first occurrence.
  pub fn terms(&self, query: &str) -> Vec<String> {
    let mut toks = Tokens::new();
    self.analyzer.analyze(query, &mut toks);
    let mut out: Vec<String> = Vec::new();
    for t in toks.iter() {
      if !out.iter().any(|o| o == t) {
        out.push(t.to_string());
      }
    }
    out
  }

  /// Local statistics, computed from the raw documents.
  pub fn stats(&self, query: &str) -> CorpusStats {
    let terms = self
      .terms(query)
      .into_iter()
      .map(|t| {
        let df = self.docs.iter().filter(|d| d.2.contains_key(&t)).count();
        muninn_text::TermDocFreq {
          term: t,
          doc_freq: df as u64,
        }
      })
      .collect();
    CorpusStats {
      doc_count: self.docs.len() as u64,
      total_doc_len: self.docs.iter().map(|d| u64::from(d.1)).sum(),
      terms,
    }
  }

  /// Exhaustive BM25 exactly as documented in `muninn_text::bm25`.
  pub fn search(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
    stats: Option<&CorpusStats>,
  ) -> Vec<Hit> {
    let local;
    let stats = match stats {
      Some(s) => s,
      None => {
        local = self.stats(query);
        &local
      }
    };
    let terms = self.terms(query);
    let n = stats.doc_count;
    let avgdl = if n == 0 || stats.total_doc_len == 0 {
      1.0f32
    } else {
      (stats.total_doc_len as f64 / n as f64) as f32
    };
    let weights: Vec<f32> = terms
      .iter()
      .map(|t| {
        let df = stats.doc_freq(t).unwrap().min(n) as f64;
        let nf = n as f64;
        let idf = (1.0 + (nf - df + 0.5) / (df + 0.5)).ln() as f32;
        idf * (1.2 + 1.0)
      })
      .collect();
    let mut hits = Vec::new();
    for (id, len, tfs) in &self.docs {
      if filter.is_some_and(|f| !f.matches(*id)) {
        continue;
      }
      let qlen = decode_doc_len(encode_doc_len(*len)) as f32;
      let kc = 1.2 * (1.0 - 0.75 + 0.75 * (qlen / avgdl));
      let mut score = 0f32;
      let mut any = false;
      for (t, &w) in terms.iter().zip(&weights) {
        if let Some(&tf) = tfs.get(t) {
          score += w / (1.0 + kc / tf as f32);
          any = true;
        }
      }
      if any {
        hits.push(Hit::new(*id, score));
      }
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    hits.truncate(k);
    hits
  }
}

/// Asserts identical ids (and order) and scores within float tolerance.
pub fn assert_same(got: &[Hit], want: &[Hit], ctx: &str) {
  let ids = |h: &[Hit]| h.iter().map(|h| h.id).collect::<Vec<_>>();
  assert_eq!(ids(got), ids(want), "ids differ: {ctx}");
  for (g, w) in got.iter().zip(want) {
    assert!(
      (g.score - w.score).abs() <= 1e-5 * w.score.abs().max(1.0),
      "score of {} differs ({} vs {}): {ctx}",
      g.id,
      g.score,
      w.score
    );
  }
}
