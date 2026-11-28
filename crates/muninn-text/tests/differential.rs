//! Differential test: pruned search, in memory and frozen, against a
//! naive BM25 written straight from the formula documented in
//! `muninn_text::bm25`, compared bit for bit (ids, order and score bits).
//!
//! Random corpora (sizes around block and block-group boundaries, Zipf
//! skews, equal and mixed lengths, repeated words, id gaps), random
//! queries (repeats, unknown words, upper case, long ones), every k from
//! 0 to `usize::MAX`, filters, and supplied statistics unrelated to the
//! index.
//!
//! The default run (a seed range plus seeds pinned because they once
//! found bugs) takes a few seconds. For longer runs:
//! `MUNINN_TEXT_DIFF_ITERS=4000 MUNINN_TEXT_DIFF_SEED=1 cargo test
//! --release -p muninn-text --test differential`; set
//! `MUNINN_TEXT_DIFF_DENSE=1` for denser corpora and longer queries.

use std::collections::HashMap;

use muninn_core::{BitSet, DocFilter, DocId, Hit};
use muninn_text::norm::{decode_doc_len, encode_doc_len};
use muninn_text::{
  Analyzer, CorpusStats, MemTextIndex, TermDocFreq, TextSearcher,
};

struct Rng(u64);
impl Rng {
  fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
  }
  fn below(&mut self, n: u64) -> u64 {
    if n == 0 { 0 } else { self.next() % n }
  }
  fn chance(&mut self, p: f64) -> bool {
    (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
  }
  fn pick<T: Copy>(&mut self, v: &[T]) -> T {
    v[self.below(v.len() as u64) as usize]
  }
}

fn word(r: usize) -> String {
  let mut s = String::from("q");
  let mut x = r;
  loop {
    s.push((b'a' + (x % 26) as u8) as char);
    x /= 26;
    if x == 0 {
      break;
    }
  }
  s
}

struct Doc {
  id: DocId,
  len: u32,
  tf: HashMap<String, u32>,
}

struct Corpus {
  texts: Vec<(DocId, String)>,
  docs: Vec<Doc>,
  vocab: usize,
  cum: Vec<f64>,
}

fn sample(rng: &mut Rng, cum: &[f64]) -> usize {
  let x = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * cum[cum.len() - 1];
  cum.partition_point(|&c| c < x).min(cum.len() - 1)
}

fn gen_corpus(rng: &mut Rng) -> Corpus {
  let n = rng.pick(&[
    1usize, 2, 3, 7, 64, 127, 128, 129, 200, 257, 500, 1000, 2000, 4000, 4100,
    6000, 9000,
  ]);
  let dense = dense_env() && rng.chance(0.7);
  let vm = rng.pick(&[3u64, 10, 30, 80]);
  let vocab = if dense {
    20 + rng.below(50) as usize
  } else {
    1 + rng.below(vm) as usize
  };
  let skew = if dense {
    rng.pick(&[0.0f64, 0.3, 0.6])
  } else {
    rng.pick(&[0.0f64, 0.8, 1.2, 2.0, 3.0])
  };
  let mut cum = Vec::with_capacity(vocab);
  let mut acc = 0.0;
  for r in 0..vocab {
    acc += 1.0 / ((r + 1) as f64).powf(skew);
    cum.push(acc);
  }
  let mode = if dense { 4 } else { rng.below(4) };
  let equal_len = 1 + rng.below(4) as usize;
  let rep_p = rng.pick(&[0.0, 0.02, 0.2]);
  let gap_p = rng.pick(&[0.0, 0.1, 0.5]);
  let mut id: DocId = rng.below(3) as DocId;
  let mut texts = Vec::with_capacity(n);
  let mut docs = Vec::with_capacity(n);
  for _ in 0..n {
    let len = match mode {
      0 => equal_len,
      1 => 1 + rng.below(3) as usize,
      2 => match rng.below(10) {
        0..=6 => 1 + rng.below(6) as usize,
        7..=8 => 6 + rng.below(40) as usize,
        _ => 50 + rng.below(400) as usize,
      },
      4 => 8 + rng.below(50) as usize,
      _ => {
        if rng.chance(0.02) {
          0
        } else {
          1 + rng.below(20) as usize
        }
      }
    };
    let mut toks: Vec<String> =
      (0..len).map(|_| word(sample(rng, &cum))).collect();
    if rng.chance(rep_p) {
      let w = word(sample(rng, &cum));
      let rm = rng.pick(&[3u64, 30, 300]);
      for _ in 0..1 + rng.below(rm) {
        toks.push(w.clone());
      }
    }
    let mut tf: HashMap<String, u32> = HashMap::new();
    for t in &toks {
      *tf.entry(t.clone()).or_default() += 1;
    }
    let sep = rng.pick(&[" ", ", ", "\n"]);
    texts.push((id, toks.join(sep)));
    docs.push(Doc {
      id,
      len: toks.len() as u32,
      tf,
    });
    id += 1
      + if rng.chance(gap_p) {
        rng.below(4) as DocId
      } else {
        0
      };
  }
  Corpus {
    texts,
    docs,
    vocab,
    cum,
  }
}

fn distinct_terms(query: &str) -> Vec<String> {
  let mut out: Vec<String> = Vec::new();
  for t in query
    .split(|c: char| !c.is_alphanumeric())
    .filter(|t| !t.is_empty())
  {
    let t = t.to_lowercase();
    if !out.contains(&t) {
      out.push(t);
    }
  }
  out
}

fn local_stats(c: &Corpus, terms: &[String]) -> CorpusStats {
  CorpusStats {
    doc_count: c.docs.len() as u64,
    total_doc_len: c.docs.iter().map(|d| u64::from(d.len)).sum(),
    terms: terms
      .iter()
      .map(|t| TermDocFreq {
        term: t.clone(),
        doc_freq: c.docs.iter().filter(|d| d.tf.contains_key(t)).count() as u64,
      })
      .collect(),
  }
}

/// BM25 straight from the formula in `muninn_text::bm25`'s docs.
fn naive(
  c: &Corpus,
  query: &str,
  k: usize,
  filter: Option<&dyn DocFilter>,
  stats: &CorpusStats,
) -> Vec<Hit> {
  let terms = distinct_terms(query);
  let n = stats.doc_count;
  let avgdl = if n == 0 || stats.total_doc_len == 0 {
    1.0f32
  } else {
    (stats.total_doc_len as f64 / n as f64) as f32
  };
  let w: Vec<f32> = terms
    .iter()
    .map(|t| {
      let df = stats
        .terms
        .iter()
        .find(|x| &x.term == t)
        .unwrap()
        .doc_freq
        .min(n) as f64;
      let nf = n as f64;
      ((1.0 + (nf - df + 0.5) / (df + 0.5)).ln() as f32) * (1.2f32 + 1.0)
    })
    .collect();
  let mut hits = Vec::new();
  for d in &c.docs {
    if filter.is_some_and(|f| !f.matches(d.id)) {
      continue;
    }
    let len = decode_doc_len(encode_doc_len(d.len)) as f32;
    let kc = 1.2f32 * (1.0 - 0.75 + 0.75 * (len / avgdl));
    let mut s = 0f32;
    let mut any = false;
    for (t, &wt) in terms.iter().zip(&w) {
      if let Some(&tf) = d.tf.get(t) {
        s += wt / (1.0 + kc / tf as f32);
        any = true;
      }
    }
    if any {
      hits.push(Hit::new(d.id, s));
    }
  }
  hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
  hits.truncate(k);
  hits
}

fn gen_query(rng: &mut Rng, c: &Corpus) -> String {
  let tm = if dense_env() && rng.chance(0.5) {
    rng.pick(&[16u64, 34, 45])
  } else {
    rng.pick(&[2u64, 4, 8, 12])
  };
  let nt = 1 + rng.below(tm) as usize;
  let mut ws: Vec<String> = Vec::new();
  for _ in 0..nt {
    let w = match rng.below(10) {
      0..=3 => word(sample(rng, &c.cum)),
      4..=6 => word(rng.below(c.vocab as u64) as usize),
      7 => format!("zz{}", rng.below(5)),
      _ => ws.last().cloned().unwrap_or_else(|| word(0)),
    };
    ws.push(if rng.chance(0.2) { w.to_uppercase() } else { w });
  }
  ws.join(" ")
}

fn same(got: &[Hit], want: &[Hit]) -> bool {
  got.len() == want.len()
    && got
      .iter()
      .zip(want)
      .all(|(g, w)| g.id == w.id && g.score.to_bits() == w.score.to_bits())
}

fn run(seed: u64) -> usize {
  let mut rng = Rng(seed);
  let c = gen_corpus(&mut rng);
  let mut mem = MemTextIndex::new(Analyzer::default());
  for (id, t) in &c.texts {
    mem.add(*id, t).unwrap();
  }
  for d in &c.docs {
    assert_eq!(mem.doc_len(d.id), Some(d.len), "seed {seed}");
  }
  let frozen = mem.freeze().unwrap();
  let nd = mem.num_docs();
  let mut filters: Vec<Box<dyn DocFilter>> = Vec::new();
  for p in [0.5, 0.05, 0.95] {
    let mut b = BitSet::new(nd);
    for i in 0..nd as DocId {
      if rng.chance(p) {
        b.insert(i);
      }
    }
    filters.push(Box::new(b));
  }
  filters.push(Box::new(|id: DocId| id % 3 != 1));
  let mut checked = 0;
  for _ in 0..25 {
    let q = gen_query(&mut rng, &c);
    let terms = distinct_terms(&q);
    let local = local_stats(&c, &terms);
    let k = match rng.below(10) {
      0 => 0,
      1 => 1000,
      2 => usize::MAX,
      _ => 1 + rng.below(50) as usize,
    };
    let global = {
      let dc = local.doc_count * (1 + rng.below(4)) + rng.below(50);
      let avg = rng.pick(&[0.3f64, 1.0, 3.0, 40.0, 500.0]);
      CorpusStats {
        doc_count: dc,
        total_doc_len: (dc as f64 * avg) as u64,
        terms: local
          .terms
          .iter()
          .map(|t| TermDocFreq {
            term: t.term.clone(),
            doc_freq: rng.below(dc + 1),
          })
          .collect(),
      }
    };
    let fi = rng.below(filters.len() as u64 + 1) as usize;
    let filter: Option<&dyn DocFilter> = filters.get(fi).map(|f| f.as_ref());
    for gs in [false, true] {
      let st = if gs { &global } else { &local };
      let want = naive(&c, &q, k, filter, st);
      let indexes: [(&str, &dyn TextSearcher); 2] =
        [("mem", &mem), ("frozen", &frozen)];
      for (name, idx) in indexes {
        let got = if gs {
          idx.search_with_stats(&q, k, filter, st).unwrap()
        } else {
          idx.search(&q, k, filter).unwrap()
        };
        if !same(&got, &want) {
          let first = got
            .iter()
            .zip(&want)
            .position(|(g, w)| {
              g.id != w.id || g.score.to_bits() != w.score.to_bits()
            })
            .unwrap_or(got.len().min(want.len()));
          panic!(
            "MISMATCH seed {seed} {name} n={} vocab={} q={q:?} k={k} \
             filter={fi} global={gs} lens {}/{} first diff at {first}: \
             got {:?} want {:?}",
            c.docs.len(),
            c.vocab,
            got.len(),
            want.len(),
            got.get(first),
            want.get(first)
          );
        }
        checked += 1;
      }
    }
  }
  checked
}

fn dense_env() -> bool {
  std::env::var_os("MUNINN_TEXT_DIFF_DENSE").is_some()
}

fn env_u64(name: &str) -> Option<u64> {
  std::env::var(name).ok().and_then(|s| s.parse().ok())
}

/// Seeds that found bugs, always run by default: all of these hit a
/// memtable term with `16m + 15` closed blocks plus an open one, whose
/// window bound read a block group past the closed blocks (index out of
/// bounds).
const PINNED: &[u64] = &[872, 1081, 1647, 1718, 2766, 3775];

/// Seeds run by default: `1..=DEFAULT_ITERS` and [`PINNED`].
const DEFAULT_ITERS: u64 = 200;

#[test]
fn pruned_equals_naive_bm25_bit_for_bit() {
  let seeds: Vec<u64> = match env_u64("MUNINN_TEXT_DIFF_ITERS") {
    Some(iters) => {
      let base = env_u64("MUNINN_TEXT_DIFF_SEED").unwrap_or(1);
      (base..base + iters).collect()
    }
    None => (1..=DEFAULT_ITERS).chain(PINNED.iter().copied()).collect(),
  };
  let mut total = 0;
  let mut fails = Vec::new();
  for &s in &seeds {
    match std::panic::catch_unwind(|| run(s)) {
      Ok(n) => total += n,
      Err(e) => {
        let msg = e
          .downcast_ref::<String>()
          .cloned()
          .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
          .unwrap_or_default();
        eprintln!("seed {s}: {msg}");
        fails.push(s);
      }
    }
  }
  eprintln!(
    "differential: {} corpora, {total} searches, failing seeds {fails:?}",
    seeds.len()
  );
  assert!(fails.is_empty(), "failing seeds {fails:?}");
}

/// Concurrent searches on one index with alternating statistics (so the
/// cached length table keeps being replaced) equal sequential ones.
#[test]
fn concurrent_searches_with_changing_stats_agree() {
  let mut rng = Rng(4242);
  let c = gen_corpus(&mut rng);
  let mut mem = MemTextIndex::new(Analyzer::default());
  for (id, t) in &c.texts {
    mem.add(*id, t).unwrap();
  }
  let frozen = mem.freeze().unwrap();
  let qs: Vec<(String, CorpusStats)> = (0..40)
    .map(|i| {
      let q = gen_query(&mut rng, &c);
      let mut st = frozen.stats_for(&q).unwrap();
      st.total_doc_len = st.total_doc_len * (1 + i % 5) + i;
      (q, st)
    })
    .collect();
  let want: Vec<Vec<Hit>> = qs
    .iter()
    .map(|(q, st)| frozen.search_with_stats(q, 10, None, st).unwrap())
    .collect();
  std::thread::scope(|s| {
    for t in 0..4 {
      let (qs, want, frozen) = (&qs, &want, &frozen);
      s.spawn(move || {
        for r in 0..50 {
          for (i, (q, st)) in qs.iter().enumerate() {
            if (i + r + t) % 2 == 0 {
              let got = frozen.search_with_stats(q, 10, None, st).unwrap();
              assert_eq!(&got, &want[i]);
            } else {
              let got = frozen.search(q, 10, None).unwrap();
              assert!(got.len() <= 10);
            }
          }
        }
      });
    }
  });
}
