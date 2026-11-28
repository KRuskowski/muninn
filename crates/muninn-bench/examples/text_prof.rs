//! Development tool for profiling muninn's text search on MS MARCO.
//!
//! Not part of the benchmark protocol (that is `muninn-bench text`): it
//! keeps the index files between runs so a query pass can be repeated
//! under `perf`, and it checks pruned search against exhaustive BM25 on
//! every real query.
//!
//! ```text
//! text_prof build <collection.tsv> <limit> <default|english> <out.idx>
//! text_prof query <index.idx> <queries.tsv> <passes> [check] [slow]
//! text_prof one <index.idx> <query>
//! text_prof churn <index.idx> <queries.tsv> <passes>
//! text_prof tbuild <collection.tsv> <limit> <default|english> <dir>
//! text_prof cmp <index.idx> <tantivy dir> <analyzer> <queries.tsv> <passes>
//! ```
//!
//! `query` reports one-thread QPS and latency percentiles (best of
//! `passes`); `check` first compares pruned and exhaustive results for
//! k = 10 and 100, `slow` lists the slowest queries. `one` prints a
//! query's terms, best latency and hits. `churn` is `query` with supplied
//! statistics whose average length changes on every query (as for a
//! memtable taking writes), so no per-query table can be reused. `cmp` times every query in both
//! engines (best of `passes`, tantivy configured as in `muninn-bench
//! text`) and lists the queries where muninn loses most.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use muninn_bench::Result;
use muninn_bench::text::read_tsv;
use muninn_core::DocId;
use muninn_store::map_file;
use muninn_text::{
  Analyzer, AnalyzerConfig, MemTextIndex, TextIndex, TextSearcher,
};
use tantivy::collector::TopDocs;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::BooleanQuery;
use tantivy::schema::{
  IndexRecordOption, Schema, TextFieldIndexing, TextOptions,
};
use tantivy::tokenizer::{
  AsciiFoldingFilter, Language, LowerCaser, RemoveLongFilter, SimpleTokenizer,
  Stemmer, StopWordFilter, TextAnalyzer,
};
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, Term};

fn build(args: &[String]) -> Result<()> {
  let [coll, limit, analyzer, out] = args else {
    return Err("build <collection> <limit> <analyzer> <out>".into());
  };
  let config = match analyzer.as_str() {
    "default" => AnalyzerConfig::default(),
    "english" => AnalyzerConfig::english(),
    _ => return Err("analyzer: default or english".into()),
  };
  let (_, docs) = read_tsv(Path::new(coll), Some(limit.parse()?))?;
  let t = Instant::now();
  let mut mem = MemTextIndex::new(Analyzer::new(config)?);
  for (i, d) in docs.iter().enumerate() {
    mem.add(i as DocId, d)?;
  }
  let w = std::io::BufWriter::new(std::fs::File::create(out)?);
  mem.write_to(w)?;
  eprintln!(
    "built {} docs in {:.1} s",
    docs.len(),
    t.elapsed().as_secs_f64()
  );
  Ok(())
}

fn query(args: &[String]) -> Result<()> {
  let (idx, qs, passes) = match args {
    [i, q, p, ..] => (i, q, p.parse::<usize>()?),
    _ => return Err("query <index> <queries> <passes> [check] [slow]".into()),
  };
  let check = args.iter().any(|a| a == "check");
  let slow = args.iter().any(|a| a == "slow");
  let index = TextIndex::open(map_file(Path::new(idx))?)?;
  let (_, queries) = read_tsv(Path::new(qs), None)?;
  if check {
    let t = Instant::now();
    let mut bad = 0;
    for q in &queries {
      for k in [10, 100] {
        let a = index.search(q, k, None)?;
        let b = index.search_exhaustive(q, k, None, None)?;
        if a != b {
          bad += 1;
          if bad < 10 {
            eprintln!("MISMATCH k={k} q={q:?}\n  pruned {a:?}\n  exh {b:?}");
          }
        }
      }
    }
    eprintln!(
      "check: {} queries x 2 k, {bad} mismatches ({:.1} s)",
      queries.len(),
      t.elapsed().as_secs_f64()
    );
    if bad > 0 {
      return Err("pruned search differs from exhaustive".into());
    }
  }
  for q in &queries {
    black_box(index.search(q, 10, None)?);
  }
  let mut best = f64::MAX;
  let mut best_lat = Vec::new();
  for _ in 0..passes {
    let mut lat = Vec::with_capacity(queries.len());
    let t = Instant::now();
    for q in &queries {
      let s = Instant::now();
      black_box(index.search(q, 10, None)?);
      lat.push(s.elapsed().as_nanos() as u64);
    }
    let secs = t.elapsed().as_secs_f64();
    if secs < best {
      best = secs;
      best_lat = lat;
    }
  }
  let mut sorted = best_lat.clone();
  sorted.sort_unstable();
  let pct = |p: f64| {
    sorted[((sorted.len() as f64 * p) as usize).min(sorted.len() - 1)] as f64
      / 1e3
  };
  println!(
    "qps {:.0}  mean {:.1} us  p50 {:.1} us  p90 {:.1} us  p99 {:.1} us  \
     max {:.1} us",
    queries.len() as f64 / best,
    best * 1e6 / queries.len() as f64,
    pct(0.5),
    pct(0.9),
    pct(0.99),
    pct(1.0)
  );
  if slow {
    let mut order: Vec<usize> = (0..queries.len()).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(best_lat[i]));
    for &i in order.iter().take(25) {
      println!("{:>9.1} us  {:?}", best_lat[i] as f64 / 1e3, queries[i]);
    }
  }
  Ok(())
}

fn tantivy_analyzer(kind: &str) -> TextAnalyzer {
  let base = TextAnalyzer::builder(SimpleTokenizer::default())
    .filter(LowerCaser)
    .dynamic();
  match kind {
    "default" => base.filter_dynamic(RemoveLongFilter::limit(256)),
    _ => base
      .filter_dynamic(AsciiFoldingFilter)
      .filter_dynamic(RemoveLongFilter::limit(256))
      .filter_dynamic(StopWordFilter::new(Language::English).unwrap())
      .filter_dynamic(Stemmer::new(Language::English)),
  }
  .build()
}

fn tbuild(args: &[String]) -> Result<()> {
  let [coll, limit, analyzer, dir] = args else {
    return Err("tbuild <collection> <limit> <analyzer> <dir>".into());
  };
  let (_, docs) = read_tsv(Path::new(coll), Some(limit.parse()?))?;
  let dir = Path::new(dir);
  if dir.exists() {
    std::fs::remove_dir_all(dir)?;
  }
  std::fs::create_dir_all(dir)?;
  let mut sb = Schema::builder();
  let text = sb.add_text_field(
    "text",
    TextOptions::default().set_indexing_options(
      TextFieldIndexing::default()
        .set_tokenizer("bench")
        .set_index_option(IndexRecordOption::WithFreqs),
    ),
  );
  let index = Index::create_in_dir(dir, sb.build())?;
  index
    .tokenizers()
    .register("bench", tantivy_analyzer(analyzer));
  let mut writer: IndexWriter<TantivyDocument> =
    index.writer_with_num_threads(1, 1 << 30)?;
  writer.set_merge_policy(Box::new(NoMergePolicy));
  for d in &docs {
    writer.add_document(tantivy::doc!(text => d.as_str()))?;
  }
  writer.commit()?;
  let segments = index.searchable_segment_ids()?;
  if segments.len() > 1 {
    writer.merge(&segments).wait()?;
  }
  writer.garbage_collect_files().wait()?;
  writer.wait_merging_threads()?;
  Ok(())
}

fn cmp(args: &[String]) -> Result<()> {
  let [idx, tdir, analyzer, qs, passes] = args else {
    return Err(
      "cmp <index> <tantivy dir> <analyzer> <queries> <passes>".into(),
    );
  };
  let passes: usize = passes.parse()?;
  let index = TextIndex::open(map_file(Path::new(idx))?)?;
  let (_, queries) = read_tsv(Path::new(qs), None)?;
  let tindex = Index::open_in_dir(tdir)?;
  tindex
    .tokenizers()
    .register("bench", tantivy_analyzer(analyzer));
  let reader = tindex
    .reader_builder()
    .reload_policy(ReloadPolicy::Manual)
    .try_into()?;
  let searcher = reader.searcher();
  let field = tindex.schema().get_field("text")?;
  let tan = tindex.tokenizers().get("bench").ok_or("no tokenizer")?;
  let tsearch = |q: &str| {
    let mut a = tan.clone();
    let mut stream = a.token_stream(q);
    let mut terms: Vec<Term> = Vec::new();
    while let Some(tok) = stream.next() {
      let term = Term::from_field_text(field, &tok.text);
      if !terms.contains(&term) {
        terms.push(term);
      }
    }
    if terms.is_empty() {
      return 0;
    }
    let query = BooleanQuery::new_multiterms_query(terms);
    searcher
      .search(&query, &TopDocs::with_limit(10).order_by_score())
      .unwrap()
      .len()
  };
  let mut mt = vec![u64::MAX; queries.len()];
  let mut tt = vec![u64::MAX; queries.len()];
  for pass in 0..=passes {
    for (i, q) in queries.iter().enumerate() {
      let s = Instant::now();
      black_box(index.search(q, 10, None)?);
      let m = s.elapsed().as_nanos() as u64;
      let s = Instant::now();
      black_box(tsearch(q));
      let t = s.elapsed().as_nanos() as u64;
      if pass > 0 {
        mt[i] = mt[i].min(m);
        tt[i] = tt[i].min(t);
      }
    }
  }
  let sum = |v: &[u64]| v.iter().sum::<u64>() as f64 / 1e3;
  let pct = |v: &[u64], p: f64| {
    let mut s = v.to_vec();
    s.sort_unstable();
    s[((s.len() as f64 * p) as usize).min(s.len() - 1)] as f64 / 1e3
  };
  println!(
    "sum of best: muninn {:.0} us, tantivy {:.0} us; p50 {:.1}/{:.1} p90 \
     {:.1}/{:.1} p99 {:.1}/{:.1}",
    sum(&mt),
    sum(&tt),
    pct(&mt, 0.5),
    pct(&tt, 0.5),
    pct(&mt, 0.9),
    pct(&tt, 0.9),
    pct(&mt, 0.99),
    pct(&tt, 0.99)
  );
  let wins = mt.iter().zip(&tt).filter(|(m, t)| m < t).count();
  println!("muninn faster on {wins} of {} queries", queries.len());
  // Loss by bucket of tantivy latency.
  for (lo, hi) in [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800)]
    .into_iter()
    .chain([(800, 1600), (1600, u64::MAX / 1000)])
  {
    let sel: Vec<usize> = (0..queries.len())
      .filter(|&i| tt[i] >= lo * 1000 && tt[i] < hi * 1000)
      .collect();
    let m: u64 = sel.iter().map(|&i| mt[i]).sum();
    let t: u64 = sel.iter().map(|&i| tt[i]).sum();
    println!(
      "  tantivy {lo:>5}..{hi:<5} us: {:>5} queries, muninn/tantivy time \
       {:.2}",
      sel.len(),
      m as f64 / t.max(1) as f64
    );
  }
  let mut order: Vec<usize> = (0..queries.len()).collect();
  order
    .sort_unstable_by_key(|&i| std::cmp::Reverse(mt[i] as i64 - tt[i] as i64));
  println!("largest losses (muninn us, tantivy us, query):");
  for &i in order.iter().take(25) {
    println!(
      "{:>9.1} {:>9.1}  {:?}",
      mt[i] as f64 / 1e3,
      tt[i] as f64 / 1e3,
      queries[i]
    );
  }
  Ok(())
}

fn one(args: &[String]) -> Result<()> {
  let [idx, q] = args else {
    return Err("one <index> <query>".into());
  };
  let index = TextIndex::open(map_file(Path::new(idx))?)?;
  let st = index.stats_for(q)?;
  for t in &st.terms {
    println!("  term {:?} df {}", t.term, t.doc_freq);
  }
  let mut best = u64::MAX;
  for _ in 0..50 {
    let s = Instant::now();
    black_box(index.search(q, 10, None)?);
    best = best.min(s.elapsed().as_nanos() as u64);
  }
  println!("best {:.1} us", best as f64 / 1e3);
  println!("{:?}", index.search(q, 10, None)?);
  Ok(())
}

fn churn(args: &[String]) -> Result<()> {
  let [idx, qs, passes] = args else {
    return Err("churn <index> <queries> <passes>".into());
  };
  let index = TextIndex::open(map_file(Path::new(idx))?)?;
  let (_, queries) = read_tsv(Path::new(qs), None)?;
  let stats: Vec<_> = queries
    .iter()
    .map(|q| index.stats_for(q))
    .collect::<muninn_core::Result<_>>()?;
  let mut best = f64::MAX;
  for pass in 0..passes.parse::<u64>()? {
    let t = Instant::now();
    for (i, (q, st)) in queries.iter().zip(&stats).enumerate() {
      let mut st = st.clone();
      st.total_doc_len += pass * queries.len() as u64 + i as u64 + 1;
      black_box(index.search_with_stats(q, 10, None, &st)?);
    }
    best = best.min(t.elapsed().as_secs_f64());
  }
  println!("churn qps {:.0}", queries.len() as f64 / best);
  Ok(())
}

fn main() -> Result<()> {
  let args: Vec<String> = std::env::args().skip(1).collect();
  match args.first().map(String::as_str) {
    Some("build") => build(&args[1..]),
    Some("query") => query(&args[1..]),
    Some("tbuild") => tbuild(&args[1..]),
    Some("one") => one(&args[1..]),
    Some("churn") => churn(&args[1..]),
    Some("cmp") => cmp(&args[1..]),
    _ => Err("usage: text_prof build|query|one|tbuild|cmp ...".into()),
  }
}
