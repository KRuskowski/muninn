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

/// A query mixing `nfreq` frequent words (Zipf ranks below 8), `nmid`
/// mid-frequency ones and `nrare` rare ones.
fn mixed_query(rng: &mut Rng, nfreq: u64, nmid: u64, nrare: u64) -> String {
  let mut words = Vec::new();
  for _ in 0..nfreq {
    words.push(common::word(rng.below(8) as usize));
  }
  for _ in 0..nmid {
    words.push(common::word(8 + rng.below(60) as usize));
  }
  for _ in 0..nrare {
    words.push(common::word(150 + rng.below(250) as usize));
  }
  words.join(" ")
}

/// Long posting lists (hundreds of blocks, so window bounds use cached
/// block groups), windows wider than one term-at-a-time chunk, and query
/// shapes that drive every path: several frequent terms (accumulation
/// of dense non-essential terms, term at a time), frequent terms with a
/// rare one (one lead term with required-term leapfrogging, rescanned
/// window bounds), and single terms.
#[test]
fn pruned_equals_exhaustive_long_lists_every_path() {
  let analyzer = Analyzer::default();
  let corpus = zipf_corpus(21, 30_000, 400);
  let mem = build(&corpus, &analyzer);
  let frozen = mem.freeze().unwrap();
  let reference = Reference::new(&analyzer, &corpus.docs);
  let top = frozen.stats_for(&common::word(0)).unwrap();
  assert!(top.terms[0].doc_freq > 16 * 128 * 4, "{top:?}");
  let mut rng = Rng::new(22);
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
  let shapes = [
    (2, 0, 0),
    (3, 0, 0),
    (2, 1, 0),
    (1, 2, 0),
    (0, 3, 0),
    (2, 0, 1),
    (4, 1, 1),
    (0, 1, 1),
    (1, 0, 0),
    (0, 0, 1),
  ];
  for qi in 0..60 {
    let (f, m, r) = shapes[qi % shapes.len()];
    let query = mixed_query(&mut rng, f, m, r);
    let k = [1, 10, 100][qi % 3];
    let local = mem.stats_for(&query).unwrap();
    let global = synthetic_stats(&local, &mut rng);
    let filter = match qi % 4 {
      0 | 1 => None,
      i => Some(filters[i - 2].1.as_ref()),
    };
    for stats in [None, Some(&global)] {
      let case = Case {
        query: &query,
        k,
        filter,
        stats,
      };
      let a = check("mem", &mem, &mem_ex, &reference, &case);
      let b = check("frozen", &frozen, &fro_ex, &reference, &case);
      assert_eq!(a, b, "mem vs frozen q={query:?}");
    }
  }
}

/// Documents built from a few templates, so very many scores tie
/// exactly and multi-term bound sums land exactly on the threshold:
/// pruning must decide those ties exactly (canonical re-sum), never
/// dropping a document that the exhaustive order keeps.
#[test]
fn pruned_equals_exhaustive_exact_ties_multi_term() {
  let analyzer = Analyzer::default();
  let templates = ["x y", "x", "y z", "x y z", "y", "z x", "w x y", "x x y"];
  let docs: Vec<(DocId, String)> = (0..6000u32)
    .map(|i| (i, templates[(i as usize * 7 + i as usize / 13) % 8].into()))
    .collect();
  let corpus = Corpus {
    docs: docs.clone(),
    vocab: 4,
  };
  let mem = build(&corpus, &analyzer);
  let frozen = mem.freeze().unwrap();
  let reference = Reference::new(&analyzer, &docs);
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
  let mod5 = |id: DocId| id % 5 != 2;
  for query in ["x y", "y x z", "z w", "x z", "w y", "x y z w"] {
    for k in [1, 3, 10, 64, 500] {
      for filter in [None, Some(&mod5 as &dyn DocFilter)] {
        let case = Case {
          query,
          k,
          filter,
          stats: None,
        };
        let a = check("mem", &mem, &mem_ex, &reference, &case);
        let b = check("frozen", &frozen, &fro_ex, &reference, &case);
        assert_eq!(a, b, "q={query:?} k={k}");
      }
    }
  }
}

/// Two documents whose scores differ by one ulp, the better one later:
/// its bound sums land within the slack band of the threshold set by the
/// earlier one, so only the exact canonical re-sum can tell that it must
/// be kept.
#[test]
fn near_tie_one_ulp_above_threshold_is_kept() {
  use muninn_text::bm25::{B, K1, idf};
  use muninn_text::norm::{decode_doc_len, encode_doc_len};
  let n = 1_000_000u64;
  let stats = CorpusStats {
    doc_count: n,
    total_doc_len: n * 37,
    terms: ["x", "y"]
      .iter()
      .zip([31_000u64, 47_000])
      .map(|(t, df)| TermDocFreq {
        term: t.to_string(),
        doc_freq: df,
      })
      .collect(),
  };
  let (wx, wy) = (idf(31_000, n) * (K1 + 1.0), idf(47_000, n) * (K1 + 1.0));
  let avg = 37.0f32;
  let score = |tx: u32, ty: u32, len: u32| {
    let len = decode_doc_len(encode_doc_len(len)) as f32;
    let kc = K1 * (1.0 - B + B * (len / avg));
    wx / (1.0 + kc / tx as f32) + wy / (1.0 + kc / ty as f32)
  };
  // Every (tf of x, tf of y, length) combination, sorted by score.
  let mut combos: Vec<(f32, u32, u32, u32)> = Vec::new();
  for tx in 1..24 {
    for ty in 1..24 {
      for len in tx + ty..tx + ty + 120 {
        combos.push((score(tx, ty, len), tx, ty, len));
      }
    }
  }
  combos.sort_by(|a, b| a.0.total_cmp(&b.0));
  let pairs: Vec<_> = combos
    .windows(2)
    .filter(|p| p[1].0.to_bits() == p[0].0.to_bits() + 1)
    .map(|p| (p[0], p[1]))
    .collect();
  assert!(pairs.len() >= 4, "only {} one-ulp pairs", pairs.len());
  let text = |tx: u32, ty: u32, len: u32| {
    let mut t = vec!["x"; tx as usize];
    t.extend(vec!["y"; ty as usize]);
    t.extend(vec!["z"; (len - tx - ty) as usize]);
    t.join(" ")
  };
  let analyzer = Analyzer::default();
  for (lo, hi) in pairs.iter().take(20) {
    // Long fillers matching one term once (dominated by both documents
    // on every block frontier), the lower-scoring document, more than a
    // block of fillers per term, then the higher-scoring document. Its
    // blocks hold nothing better than itself, so even its window bounds
    // sum to exactly its score, one ulp above the threshold.
    let filler = |i: u32| {
      let t = if i.is_multiple_of(2) { "x" } else { "y" };
      format!("{t}{}", " z".repeat(200))
    };
    let mut docs: Vec<(DocId, String)> =
      (0..300).map(|i| (i, filler(i))).collect();
    docs.push((300, text(lo.1, lo.2, lo.3)));
    docs.extend((301..701).map(|i| (i, filler(i))));
    docs.push((701, text(hi.1, hi.2, hi.3)));
    let corpus = Corpus {
      docs: docs.clone(),
      vocab: 3,
    };
    let mem = build(&corpus, &analyzer);
    let frozen = mem.freeze().unwrap();
    let reference = Reference::new(&analyzer, &docs);
    for k in [1, 2] {
      let want = reference.search("x y", k, None, Some(&stats));
      assert_eq!(want[0].id, 701, "the later document scores higher");
      assert_eq!(want[0].score.to_bits(), hi.0.to_bits());
      let ctx = format!("{lo:?} {hi:?} k={k}");
      for idx in [&mem as &dyn TextSearcher, &frozen] {
        let got = idx.search_with_stats("x y", k, None, &stats).unwrap();
        assert_same(&got, &want, &ctx);
      }
    }
  }
}

/// A live (memtable) posting list with 31 closed blocks and an open
/// one, and a window over blocks 16..=31: the window bound may take a
/// cached group of closed blocks only, never one reaching the open
/// block (which once indexed past the group cache and panicked).
#[test]
fn memtable_window_bound_stays_within_closed_blocks() {
  let mut mem = MemTextIndex::new(Analyzer::default());
  for id in 0..4000u32 {
    let text = match id {
      0 => "qa qb",
      300 => "qb",
      _ => "qa",
    };
    mem.add(id, text).unwrap();
  }
  for id in 4000..5000u32 {
    mem.add(id, if id == 4999 { "qb" } else { "qc" }).unwrap();
  }
  for k in [1, 2, 10] {
    let want = mem.search_exhaustive("qa qb", k, None, None).unwrap();
    assert_eq!(mem.search("qa qb", k, None).unwrap(), want, "k={k}");
    let frozen = mem.freeze().unwrap();
    assert_eq!(frozen.search("qa qb", k, None).unwrap(), want, "k={k}");
  }
}
