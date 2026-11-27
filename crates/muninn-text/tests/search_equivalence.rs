//! Pruned search must return exactly what exhaustive BM25 returns.

mod common;

use common::{Corpus, Reference, Rng, assert_same, random_query, zipf_corpus};
use muninn_core::{BitSet, DocFilter, DocId, Hit};
use muninn_text::{
  Analyzer, AnalyzerConfig, CorpusStats, MemTextIndex, TermDocFreq, TextIndex,
  TextSearcher,
};

fn build(corpus: &Corpus, analyzer: &Analyzer) -> MemTextIndex {
  let mut m = MemTextIndex::new(analyzer.clone());
  for (id, text) in &corpus.docs {
    m.add(*id, text).unwrap();
  }
  m
}

fn filters(
  num_docs: usize,
  rng: &mut Rng,
) -> Vec<(String, Box<dyn DocFilter>)> {
  let mut half = BitSet::new(num_docs);
  let mut rare = BitSet::new(num_docs);
  for id in 0..num_docs as DocId {
    if rng.chance(0.5) {
      half.insert(id);
    }
    if rng.chance(0.03) {
      rare.insert(id);
    }
  }
  vec![
    ("half".into(), Box::new(half)),
    ("rare".into(), Box::new(rare)),
    ("mod7".into(), Box::new(|id: DocId| id % 7 != 3)),
  ]
}

/// Statistics unrelated to the index: different N, average length and
/// document frequencies, including extremes, to check that block bounds
/// hold under any supplied statistics.
fn synthetic_stats(local: &CorpusStats, rng: &mut Rng) -> CorpusStats {
  let doc_count = local.doc_count * (1 + rng.below(5)) + rng.below(100);
  let avg = match rng.below(4) {
    0 => 0.3,
    1 => 2.0,
    2 => 40.0,
    _ => 500.0,
  };
  CorpusStats {
    doc_count,
    total_doc_len: (doc_count as f64 * avg) as u64,
    terms: local
      .terms
      .iter()
      .map(|t| TermDocFreq {
        term: t.term.clone(),
        doc_freq: rng.below(doc_count + 1),
      })
      .collect(),
  }
}

/// The exhaustive (unpruned) search of one index.
type Exhaustive<'a> = dyn Fn(&Case<'_>) -> Vec<Hit> + 'a;

/// One query configuration.
struct Case<'a> {
  query: &'a str,
  k: usize,
  filter: Option<&'a dyn DocFilter>,
  stats: Option<&'a CorpusStats>,
}

/// Checks pruned and exhaustive search of `index` against the reference
/// and returns the pruned result.
fn check(
  name: &str,
  index: &dyn TextSearcher,
  exhaustive: &Exhaustive<'_>,
  reference: &Reference,
  c: &Case<'_>,
) -> Vec<Hit> {
  let ctx = format!(
    "{name} q={:?} k={} stats={}",
    c.query,
    c.k,
    c.stats.is_some()
  );
  let want = reference.search(c.query, c.k, c.filter, c.stats);
  let got = match c.stats {
    Some(s) => index.search_with_stats(c.query, c.k, c.filter, s).unwrap(),
    None => index.search(c.query, c.k, c.filter).unwrap(),
  };
  assert_same(&got, &want, &ctx);
  assert_same(&exhaustive(c), &want, &ctx);
  got
}

fn run_corpus(seed: u64, n: usize, vocab: usize, config: AnalyzerConfig) {
  let analyzer = Analyzer::new(config).unwrap();
  let corpus = zipf_corpus(seed, n, vocab);
  let mem = build(&corpus, &analyzer);
  let frozen = mem.freeze().unwrap();
  let reference = Reference::new(&analyzer, &corpus.docs);
  let mut rng = Rng::new(seed ^ 0xabcdef);
  let filters = filters(mem.num_docs(), &mut rng);
  let mem_ex = |c: &Case<'_>| {
    mem
      .search_exhaustive(c.query, c.k, c.filter, c.stats)
      .unwrap()
  };
  let fro_ex = |c: &Case<'_>| {
    frozen
      .search_exhaustive(c.query, c.k, c.filter, c.stats)
      .unwrap()
  };
  let mut checked = 0;
  for qi in 0..150 {
    let query = random_query(&mut rng, vocab);
    let k = [1, 3, 10, 10, 50, 1000][qi % 6];
    let local = mem.stats_for(&query).unwrap();
    assert_eq!(local, reference.stats(&query), "stats for {query:?}");
    assert_eq!(frozen.stats_for(&query).unwrap(), local);
    let global = synthetic_stats(&local, &mut rng);
    let filter_choice: Vec<Option<&dyn DocFilter>> = std::iter::once(None)
      .chain(filters.iter().map(|(_, f)| Some(f.as_ref())))
      .collect();
    for filter in filter_choice {
      for stats in [None, Some(&global)] {
        let case = Case {
          query: &query,
          k,
          filter,
          stats,
        };
        let a = check("mem", &mem, &mem_ex, &reference, &case);
        let b = check("frozen", &frozen, &fro_ex, &reference, &case);
        // In-memory and frozen scores are bit-identical.
        assert_eq!(a, b, "mem vs frozen q={query:?}");
        checked += 1;
      }
    }
  }
  assert_eq!(checked, 150 * 4 * 2);
}

#[test]
fn pruned_equals_exhaustive_plain_analyzer() {
  run_corpus(1, 4000, 3000, AnalyzerConfig::default());
}

#[test]
fn pruned_equals_exhaustive_english_analyzer() {
  run_corpus(2, 3000, 2000, AnalyzerConfig::english());
}

#[test]
fn pruned_equals_exhaustive_small_vocabulary_long_lists() {
  // Few distinct words: every list spans many blocks and ties abound.
  run_corpus(3, 5000, 60, AnalyzerConfig::default());
}

#[test]
fn pruned_equals_exhaustive_tiny_corpora() {
  for seed in 10..40 {
    run_corpus(seed, 1 + (seed as usize * 7) % 200, 50, Default::default());
  }
}

/// The engine's use: one collection split into segments, statistics
/// summed over all of them, per-segment top-k merged. Must equal one
/// index over everything.
#[test]
fn global_stats_make_segments_equal_one_index() {
  let analyzer = Analyzer::default();
  let corpus = zipf_corpus(7, 6000, 2500);
  let whole = build(&corpus, &analyzer).freeze().unwrap();
  // Split into three segments with local ids; remember the global id.
  let mut segs: Vec<(MemTextIndex, Vec<DocId>)> = (0..3)
    .map(|_| (MemTextIndex::new(analyzer.clone()), Vec::new()))
    .collect();
  for (i, (id, text)) in corpus.docs.iter().enumerate() {
    let s = &mut segs[(i * 7 / 3) % 3];
    let local = s.1.len() as DocId;
    s.0.add(local, text).unwrap();
    s.1.push(*id);
  }
  // Segment 0 stays in memory, as the engine's memtable would.
  let frozen: Vec<TextIndex> =
    segs[1..].iter().map(|(m, _)| m.freeze().unwrap()).collect();
  let mut searchers: Vec<(&dyn TextSearcher, &Vec<DocId>)> =
    vec![(&segs[0].0, &segs[0].1)];
  for (f, (_, ids)) in frozen.iter().zip(&segs[1..]) {
    searchers.push((f, ids));
  }
  let mut rng = Rng::new(99);
  for qi in 0..200 {
    let query = random_query(&mut rng, 2500);
    let k = [1, 5, 10, 100][qi % 4];
    let mut stats = CorpusStats::default();
    for (s, _) in &searchers {
      stats.merge(&s.stats_for(&query).unwrap());
    }
    assert_eq!(stats.doc_count, whole.doc_count());
    let mut merged: Vec<Hit> = Vec::new();
    for (s, ids) in &searchers {
      for h in s.search_with_stats(&query, k, None, &stats).unwrap() {
        merged.push(Hit::new(ids[h.id as usize], h.score));
      }
    }
    merged.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    merged.truncate(k);
    let want = whole.search(&query, k, None).unwrap();
    assert_eq!(merged, want, "q={query:?}");
  }
}

/// Queries with more distinct terms than the canonical-summation limit
/// take the slack path; it must be exact too.
#[test]
fn pruned_equals_exhaustive_long_queries() {
  let analyzer = Analyzer::default();
  let corpus = zipf_corpus(11, 3000, 400);
  let mem = build(&corpus, &analyzer);
  let frozen = mem.freeze().unwrap();
  let reference = Reference::new(&analyzer, &corpus.docs);
  let mut rng = Rng::new(12);
  let half = |id: DocId| id.is_multiple_of(2);
  for qi in 0..60 {
    let n = [20, 33, 40, 80][qi % 4];
    let query: Vec<String> = (0..n)
      .map(|_| common::word(rng.below(400) as usize))
      .collect();
    let query = query.join(" ");
    assert!(reference.terms(&query).len() > 12);
    for k in [1, 10, 100] {
      for filter in [None, Some(&half as &dyn DocFilter)] {
        let want = reference.search(&query, k, filter, None);
        let ctx = format!("n={n} k={k}");
        assert_same(&mem.search(&query, k, filter).unwrap(), &want, &ctx);
        assert_same(&frozen.search(&query, k, filter).unwrap(), &want, &ctx);
      }
    }
  }
}

/// Review finding: `w * tf / (tf + kc)` is not monotonic in tf under f32
/// rounding, so a document with tf - 1 (same length norm) scored an ulp
/// above its block's bound and pruning dropped the true top hit.
#[test]
fn tf_rounding_cannot_lift_a_document_above_its_block_bound() {
  let big = "x ".repeat(29_894);
  let almost = format!("{}y", "x ".repeat(29_893));
  let mut mem = MemTextIndex::new(Analyzer::default());
  for id in 0..256u32 {
    let text = match id {
      0 | 128 => big.as_str(),
      129 => almost.as_str(),
      _ => "x",
    };
    mem.add(id, text).unwrap();
  }
  let frozen = mem.freeze().unwrap();
  for (name, idx) in [
    ("mem", &mem as &dyn TextSearcher),
    ("frozen", &frozen as &dyn TextSearcher),
  ] {
    let exhaustive = |k| match name {
      "mem" => mem.search_exhaustive("x", k, None, None).unwrap(),
      _ => frozen.search_exhaustive("x", k, None, None).unwrap(),
    };
    // Before the fix, doc 129 (one "x" fewer, same length) out-scored
    // docs 0 and 128 by an ulp and pruning returned doc 0.
    let all = exhaustive(256);
    let score = |id| all.iter().find(|h| h.id == id).unwrap().score;
    assert!(score(129) <= score(0), "{name}: tf - 1 scored higher");
    assert_eq!(score(0), score(128));
    for k in [1, 2, 3, 256] {
      assert_eq!(idx.search("x", k, None).unwrap(), exhaustive(k), "{name}");
    }
  }
}

/// The same property under supplied statistics with a huge average
/// length. For each average length, finds a tf where the old formula
/// `w * tf / (tf + K)` ranks tf - 1 above tf at equal length, and puts
/// exactly those two documents first in one block.
#[test]
fn tf_rounding_with_large_supplied_stats() {
  use muninn_text::bm25::{B, K1, idf};
  use muninn_text::norm::{decode_doc_len, encode_doc_len};
  let old = |w: f32, tf: u32, kc: f32| {
    let tf = tf as f32;
    w * tf / (tf + kc)
  };
  let analyzer = Analyzer::default();
  let n = 10_000_000u64;
  let mut found = 0;
  for avgdl in [1.0e5f64, 1.0e6, 1.0e7, 1.0e8, 1.0e9] {
    let stats = CorpusStats {
      doc_count: n,
      total_doc_len: (n as f64 * avgdl) as u64,
      terms: vec![TermDocFreq {
        term: "x".into(),
        doc_freq: 5_000,
      }],
    };
    let w = idf(5_000, n) * (K1 + 1.0);
    let avg = (stats.total_doc_len as f64 / n as f64) as f32;
    let kc = |len: u32| {
      let len = decode_doc_len(encode_doc_len(len)) as f32;
      K1 * (1.0 - B + B * (len / avg))
    };
    let Some(tf) =
      (2..20_000u32).find(|&tf| old(w, tf - 1, kc(tf)) > old(w, tf, kc(tf)))
    else {
      continue;
    };
    found += 1;
    let mut docs: Vec<(DocId, String)> = vec![
      (0, "x ".repeat(tf as usize)),
      (1, format!("{}y", "x ".repeat(tf as usize - 1))),
    ];
    docs.extend((2..128).map(|id| (id, "x".to_string())));
    let mem = build(
      &Corpus {
        docs: docs.clone(),
        vocab: 2,
      },
      &analyzer,
    );
    let frozen = mem.freeze().unwrap();
    let reference = Reference::new(&analyzer, &docs);
    for k in [1, 2, 3] {
      let want = reference.search("x", k, None, Some(&stats));
      let ctx = format!("avgdl {avgdl} tf {tf} k {k}");
      let m = mem.search_with_stats("x", k, None, &stats).unwrap();
      let f = frozen.search_with_stats("x", k, None, &stats).unwrap();
      assert_same(&m, &want, &ctx);
      assert_same(&f, &want, &ctx);
      let ex = mem.search_exhaustive("x", k, None, Some(&stats)).unwrap();
      assert_eq!(m, ex, "{ctx}");
    }
  }
  assert!(found >= 3, "only {found} averages produced an inversion");
}
