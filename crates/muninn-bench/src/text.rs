//! Full-text benchmark: muninn and tantivy, BM25 top-k on MS MARCO.
//!
//! Both engines index the same passages with matching analysis, both
//! score with BM25 (k1 = 1.2, b = 0.75, Lucene-style one-byte length
//! norms) and OR semantics, and both are searched from a single
//! memory-mapped segment. Settings:
//!
//! - **Indexing.** muninn: one [`MemTextIndex`] on one thread, written
//!   to one file and fsynced. tantivy: an `IndexWriter` with the given
//!   number of indexing threads (1 GiB budget each), merges disabled
//!   while indexing, one commit, then all segments merged into one and
//!   obsolete files removed. Indexing figures include writing (muninn)
//!   and committing plus merging (tantivy), so both end with one durable
//!   segment.
//! - **Fields.** tantivy indexes term frequencies without positions and
//!   stores no document text, like muninn; it additionally keeps a
//!   `u64` fast field with the passage number to map its results back
//!   (reported separately in the index size details).
//! - **Queries.** Each query string is analysed by the engine's own
//!   analyzer inside the timed region. tantivy gets a boolean OR of one
//!   term query per distinct term (`BooleanQuery::new_multiterms_query`),
//!   collected with `TopDocs::with_limit(k).order_by_score()`; muninn
//!   gets `TextSearcher::search`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use muninn_core::DocId;
use muninn_store::map_file;
use muninn_text::TextSearcher;
use muninn_text::{Analyzer, AnalyzerConfig, MemTextIndex, TextIndex};
use serde::Serialize;
use serde_json::json;
use tantivy::collector::TopDocs;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::BooleanQuery;
use tantivy::schema::{
  FAST, IndexRecordOption, Schema, TextFieldIndexing, TextOptions,
};
use tantivy::tokenizer::{
  AsciiFoldingFilter, Language, LowerCaser, RemoveLongFilter, SimpleTokenizer,
  Stemmer, StopWordFilter, TextAnalyzer,
};
use tantivy::{
  DocAddress, Index, IndexWriter, ReloadPolicy, TantivyDocument, Term,
};

use crate::Result;
use crate::measure::{Latency, overlap};
use crate::run::{Pools, measure, write_file, write_json};
use crate::sys::{SystemInfo, dir_size, rss_bytes, rss_delta};
use crate::vector::muninn_version;

/// Text analysis variant, configured equivalently in both engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalyzerKind {
  /// Split on non-alphanumeric characters, lowercase, drop tokens over
  /// 255 bytes.
  Default,
  /// `Default` plus ASCII diacritic folding, Lucene's English stop words
  /// and an English stemmer (Porter in muninn, Snowball English, also
  /// called Porter2, in tantivy).
  English,
}

impl AnalyzerKind {
  /// Parses `default` or `english`.
  pub fn parse(s: &str) -> Result<Self> {
    match s {
      "default" => Ok(Self::Default),
      "english" => Ok(Self::English),
      _ => Err(format!("unknown analyzer {s:?}").into()),
    }
  }

  /// Name used in file names and reports.
  pub fn name(self) -> &'static str {
    match self {
      Self::Default => "default",
      Self::English => "english",
    }
  }

  fn muninn(self) -> Result<Analyzer> {
    Ok(Analyzer::new(match self {
      Self::Default => AnalyzerConfig::default(),
      Self::English => AnalyzerConfig::english(),
    })?)
  }

  fn muninn_description(self) -> &'static str {
    match self {
      Self::Default => {
        "alphanumeric runs, Unicode lowercase, tokens > 255 bytes dropped"
      }
      Self::English => {
        "alphanumeric runs, Unicode lowercase, Latin diacritic folding, \
         tokens > 255 bytes dropped, Lucene English stop words, Porter \
         stemmer"
      }
    }
  }

  fn tantivy(self) -> TextAnalyzer {
    let base = TextAnalyzer::builder(SimpleTokenizer::default())
      .filter(LowerCaser)
      .dynamic();
    match self {
      // RemoveLongFilter keeps tokens shorter than the limit.
      Self::Default => base.filter_dynamic(RemoveLongFilter::limit(256)),
      Self::English => base
        .filter_dynamic(AsciiFoldingFilter)
        .filter_dynamic(RemoveLongFilter::limit(256))
        .filter_dynamic(
          StopWordFilter::new(Language::English).expect("English stop words"),
        )
        .filter_dynamic(Stemmer::new(Language::English)),
    }
    .build()
  }

  fn tantivy_description(self) -> &'static str {
    match self {
      Self::Default => {
        "SimpleTokenizer, LowerCaser, RemoveLongFilter(256) (tokens of \
         at most 255 bytes kept)"
      }
      Self::English => {
        "SimpleTokenizer, LowerCaser, AsciiFoldingFilter, \
         RemoveLongFilter(256), StopWordFilter(English, Lucene list), \
         Stemmer(English, Snowball)"
      }
    }
  }
}

/// Reads a `number<TAB>text` file (MS MARCO collection or queries),
/// keeping at most `limit` rows.
pub fn read_tsv(
  path: &Path,
  limit: Option<usize>,
) -> Result<(Vec<u64>, Vec<String>)> {
  let r = BufReader::with_capacity(
    1 << 20,
    File::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
  );
  let mut ids = Vec::new();
  let mut texts = Vec::new();
  for (n, line) in r.lines().enumerate() {
    if limit.is_some_and(|l| texts.len() >= l) {
      break;
    }
    let line = line?;
    let (id, text) = line
      .split_once('\t')
      .ok_or_else(|| format!("{}:{}: no tab", path.display(), n + 1))?;
    ids.push(
      id.parse()
        .map_err(|_| format!("{}:{}: bad id {id:?}", path.display(), n + 1))?,
    );
    texts.push(text.to_string());
  }
  Ok((ids, texts))
}

/// Settings of a text benchmark.
#[derive(Clone, Debug)]
pub struct TextRun {
  /// `collection.tsv`: `pid<TAB>passage`.
  pub collection: PathBuf,
  /// Queries file: `qid<TAB>query`.
  pub queries: PathBuf,
  /// Index only the first this many passages.
  pub limit: Option<usize>,
  /// Use only the first this many queries.
  pub query_limit: Option<usize>,
  /// Analysis variant.
  pub analyzer: AnalyzerKind,
  /// Results per query.
  pub k: usize,
  /// Measurement repetitions.
  pub runs: usize,
  /// tantivy indexing thread counts to time. The first one builds the
  /// index that is searched. Empty skips tantivy.
  pub tantivy_index_threads: Vec<usize>,
  /// Where index files go.
  pub work_dir: PathBuf,
  /// Where result JSON goes.
  pub out_dir: PathBuf,
}

/// Indexing figures for one engine and thread count.
#[derive(Clone, Debug, Serialize)]
pub struct TextIndexInfo {
  /// Indexing threads.
  pub threads: usize,
  /// Adding all documents (tantivy: including the commit).
  pub index_seconds: f64,
  /// Producing the final single segment: writing and fsyncing the file
  /// (muninn), or merging and removing obsolete files (tantivy).
  pub finish_seconds: f64,
  /// `index_seconds + finish_seconds`.
  pub total_seconds: f64,
  /// Documents divided by `total_seconds`.
  pub docs_per_sec: f64,
  /// Size of the final index on disk.
  pub index_bytes: u64,
  /// Growth of resident memory while indexing, when measurable.
  pub rss_delta_bytes: Option<i64>,
  /// Engine-specific figures.
  pub details: serde_json::Value,
}

/// Query figures for one engine.
#[derive(Clone, Debug, Serialize)]
pub struct TextSearchInfo {
  /// Results per query.
  pub k: usize,
  /// Queries measured.
  pub queries: usize,
  /// Queries per second, one thread.
  pub qps_1t: f64,
  /// Queries per second, all threads.
  pub qps_nt: f64,
  /// One-thread latency distribution.
  pub latency: Latency,
  /// Mean number of results returned.
  pub mean_hits: f64,
  /// Whether all-thread and one-thread passes agreed.
  pub deterministic: bool,
}

/// One engine's results.
#[derive(Clone, Debug, Serialize)]
pub struct TextEngine {
  /// `muninn` or `tantivy`.
  pub engine: String,
  /// Engine version.
  pub engine_version: String,
  /// Analysis chain.
  pub analyzer: String,
  /// Indexing figures of the searched index.
  pub index: TextIndexInfo,
  /// Indexing figures with other thread counts (tantivy).
  pub other_index_runs: Vec<TextIndexInfo>,
  /// Query figures.
  pub search: TextSearchInfo,
}

/// Agreement of two engines' top-k results.
#[derive(Clone, Debug, Serialize)]
pub struct OverlapInfo {
  /// The two engines compared.
  pub engines: [String; 2],
  /// Mean overlap@k (shared ids / longer list).
  pub mean: f64,
  /// Fraction of queries with identical top-k id sets.
  pub identical_sets: f64,
  /// Fraction of queries with the same best result.
  pub same_top1: f64,
}

/// A text benchmark result file.
#[derive(Clone, Debug, Serialize)]
pub struct TextReport {
  /// Always `"text"`.
  pub kind: &'static str,
  /// Dataset description.
  pub dataset: serde_json::Value,
  /// Analysis variant name.
  pub analyzer: String,
  /// Results per query.
  pub k: usize,
  /// Threads in the all-thread pool.
  pub threads: usize,
  /// Per-engine results.
  pub engines: Vec<TextEngine>,
  /// muninn vs tantivy agreement.
  pub overlap: Option<OverlapInfo>,
  /// Machine description.
  pub system: SystemInfo,
  /// Free-form remarks.
  pub notes: Vec<String>,
}

fn muninn_engine(
  cfg: &TextRun,
  docs: &[String],
  queries: &[String],
  pools: &Pools,
) -> Result<(TextEngine, Vec<Vec<u32>>)> {
  let file = cfg
    .work_dir
    .join(format!("muninn-text-{}.idx", cfg.analyzer.name()));
  let rss0 = rss_bytes();
  let t = Instant::now();
  let mut mem = MemTextIndex::new(cfg.analyzer.muninn()?);
  for (i, d) in docs.iter().enumerate() {
    mem.add(i as DocId, d)?;
  }
  let index_s = t.elapsed().as_secs_f64();
  let rss = rss_delta(rss0, rss_bytes());
  let heap = mem.heap_bytes();
  let terms = mem.num_terms();
  let (write_s, bytes) = write_file(&file, |w| mem.write_to(w))?;
  drop(mem);
  let idx = TextIndex::open(map_file(&file)?)?;
  let total = index_s + write_s;
  eprintln!(
    "  muninn: indexed in {index_s:.1} s + write {write_s:.2} s \
     ({:.0} docs/s), {:.1} MB, {terms} terms",
    docs.len() as f64 / total,
    bytes as f64 / 1e6
  );
  let k = cfg.k;
  let m = measure(pools, queries, cfg.runs, |q| {
    idx
      .search(q, k, None)
      .expect("muninn search")
      .into_iter()
      .map(|h| h.id)
      .collect::<Vec<u32>>()
  });
  let engine = TextEngine {
    engine: "muninn".into(),
    engine_version: muninn_version(),
    analyzer: cfg.analyzer.muninn_description().into(),
    index: TextIndexInfo {
      threads: 1,
      index_seconds: index_s,
      finish_seconds: write_s,
      total_seconds: total,
      docs_per_sec: docs.len() as f64 / total,
      index_bytes: bytes,
      rss_delta_bytes: rss,
      details: json!({ "terms": terms, "in_memory_index_bytes": heap }),
    },
    other_index_runs: Vec::new(),
    search: search_info(k, &m),
  };
  drop(idx);
  std::fs::remove_file(&file)?;
  Ok((engine, m.results))
}

fn search_info(k: usize, m: &crate::run::Measured<Vec<u32>>) -> TextSearchInfo {
  let hits: usize = m.results.iter().map(Vec::len).sum();
  TextSearchInfo {
    k,
    queries: m.results.len(),
    qps_1t: m.qps_1t,
    qps_nt: m.qps_nt,
    latency: m.latency,
    mean_hits: hits as f64 / m.results.len().max(1) as f64,
    deterministic: m.same_results,
  }
}

/// Builds a tantivy index of `docs` in `dir` with `threads` indexing
/// threads and merges it into one segment.
fn tantivy_build(
  docs: &[String],
  kind: AnalyzerKind,
  threads: usize,
  dir: &Path,
) -> Result<(Index, TextIndexInfo)> {
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
  let id = sb.add_u64_field("id", FAST);
  let index = Index::create_in_dir(dir, sb.build())?;
  index.tokenizers().register("bench", kind.tantivy());
  let mut writer: IndexWriter<TantivyDocument> =
    index.writer_with_num_threads(threads, threads << 30)?;
  writer.set_merge_policy(Box::new(NoMergePolicy));
  let rss0 = rss_bytes();
  let t = Instant::now();
  for (i, d) in docs.iter().enumerate() {
    writer.add_document(tantivy::doc!(text => d.as_str(), id => i as u64))?;
  }
  writer.commit()?;
  let index_s = t.elapsed().as_secs_f64();
  let rss = rss_delta(rss0, rss_bytes());
  let t = Instant::now();
  let segments = index.searchable_segment_ids()?;
  if segments.len() > 1 {
    writer.merge(&segments).wait()?;
  }
  writer.garbage_collect_files().wait()?;
  writer.wait_merging_threads()?;
  let merge_s = t.elapsed().as_secs_f64();
  let bytes = dir_size(dir)?;
  let total = index_s + merge_s;
  eprintln!(
    "  tantivy ({threads} indexing threads): indexed in {index_s:.1} s \
     ({} segments) + merge {merge_s:.1} s ({:.0} docs/s), {:.1} MB",
    segments.len(),
    docs.len() as f64 / total,
    bytes as f64 / 1e6
  );
  Ok((
    index,
    TextIndexInfo {
      threads,
      index_seconds: index_s,
      finish_seconds: merge_s,
      total_seconds: total,
      docs_per_sec: docs.len() as f64 / total,
      index_bytes: bytes,
      rss_delta_bytes: rss,
      details: json!({ "segments_before_merge": segments.len() }),
    },
  ))
}

fn tantivy_engine(
  cfg: &TextRun,
  docs: &[String],
  queries: &[String],
  pools: &Pools,
) -> Result<(TextEngine, Vec<Vec<u32>>)> {
  let mut threads = cfg.tantivy_index_threads.iter().copied();
  let first = threads.next().expect("at least one thread count");
  let dir = |t: usize| {
    cfg
      .work_dir
      .join(format!("tantivy-{}-{t}t", cfg.analyzer.name()))
  };
  let mut other = Vec::new();
  for t in threads {
    let (index, info) = tantivy_build(docs, cfg.analyzer, t, &dir(t))?;
    drop(index);
    std::fs::remove_dir_all(dir(t))?;
    other.push(info);
  }
  let (index, mut info) =
    tantivy_build(docs, cfg.analyzer, first, &dir(first))?;
  let reader = index
    .reader_builder()
    .reload_policy(ReloadPolicy::Manual)
    .try_into()?;
  let searcher = reader.searcher();
  let usage = searcher.space_usage()?;
  let fast_bytes: u64 = usage
    .segments()
    .iter()
    .map(|s| s.fast_fields().total().get_bytes())
    .sum();
  info.details["segments"] = json!(searcher.segment_readers().len());
  info.details["id_fast_field_bytes"] = json!(fast_bytes);
  let field = index.schema().get_field("text")?;
  let analyzer = index
    .tokenizers()
    .get("bench")
    .ok_or("tokenizer not registered")?;
  let k = cfg.k;
  let m = measure(pools, queries, cfg.runs, |q| {
    let mut a = analyzer.clone();
    let mut stream = a.token_stream(q);
    let mut terms: Vec<Term> = Vec::new();
    while let Some(tok) = stream.next() {
      let term = Term::from_field_text(field, &tok.text);
      if !terms.contains(&term) {
        terms.push(term);
      }
    }
    if terms.is_empty() {
      return Vec::new();
    }
    let query = BooleanQuery::new_multiterms_query(terms);
    searcher
      .search(&query, &TopDocs::with_limit(k).order_by_score())
      .expect("tantivy search")
      .into_iter()
      .map(|(_, addr)| addr)
      .collect::<Vec<DocAddress>>()
  });
  // Map document addresses to passage numbers outside the timed passes.
  let columns = searcher
    .segment_readers()
    .iter()
    .map(|r| r.fast_fields().u64("id"))
    .collect::<tantivy::Result<Vec<_>>>()?;
  let ids: Vec<Vec<u32>> = m
    .results
    .iter()
    .map(|addrs| {
      addrs
        .iter()
        .map(|a| {
          columns[a.segment_ord as usize]
            .first(a.doc_id)
            .expect("every document has an id") as u32
        })
        .collect()
    })
    .collect();
  let hits: usize = ids.iter().map(Vec::len).sum();
  let engine = TextEngine {
    engine: "tantivy".into(),
    engine_version: tantivy::version_string().to_string(),
    analyzer: cfg.analyzer.tantivy_description().into(),
    index: info,
    other_index_runs: other,
    search: TextSearchInfo {
      k,
      queries: ids.len(),
      qps_1t: m.qps_1t,
      qps_nt: m.qps_nt,
      latency: m.latency,
      mean_hits: hits as f64 / ids.len().max(1) as f64,
      deterministic: m.same_results,
    },
  };
  drop(searcher);
  drop(reader);
  drop(index);
  std::fs::remove_dir_all(dir(first))?;
  Ok((engine, ids))
}

/// Agreement between two engines' per-query results.
pub fn compare(
  a: &[Vec<u32>],
  b: &[Vec<u32>],
  names: [&str; 2],
) -> OverlapInfo {
  assert_eq!(a.len(), b.len(), "same queries");
  let n = a.len().max(1) as f64;
  let mut sum = 0.0;
  let mut same = 0usize;
  let mut top1 = 0usize;
  for (x, y) in a.iter().zip(b) {
    let o = overlap(x, y);
    sum += o;
    same += usize::from(o == 1.0);
    top1 += usize::from(x.first() == y.first());
  }
  OverlapInfo {
    engines: names.map(String::from),
    mean: sum / n,
    identical_sets: same as f64 / n,
    same_top1: top1 as f64 / n,
  }
}

/// Runs the text benchmark and writes its JSON. Returns the path.
pub fn run_text(cfg: &TextRun, pools: &Pools) -> Result<PathBuf> {
  let t = Instant::now();
  let (_, docs) = read_tsv(&cfg.collection, cfg.limit)?;
  let (_, queries) = read_tsv(&cfg.queries, cfg.query_limit)?;
  let bytes: usize = docs.iter().map(String::len).sum();
  eprintln!(
    "msmarco-passage: {} passages ({:.0} MB of text), {} queries loaded \
     in {:.1} s; analyzer {}",
    docs.len(),
    bytes as f64 / 1e6,
    queries.len(),
    t.elapsed().as_secs_f64(),
    cfg.analyzer.name()
  );
  std::fs::create_dir_all(&cfg.work_dir)?;
  let (mun, mun_ids) = muninn_engine(cfg, &docs, &queries, pools)?;
  log_search("muninn", &mun.search, pools.threads);
  let mut engines = vec![mun];
  let mut overlap = None;
  if !cfg.tantivy_index_threads.is_empty() {
    let (tan, tan_ids) = tantivy_engine(cfg, &docs, &queries, pools)?;
    log_search("tantivy", &tan.search, pools.threads);
    let o = compare(&mun_ids, &tan_ids, ["muninn", "tantivy"]);
    eprintln!(
      "  overlap@{}: mean {:.4}, identical sets {:.4}, same top-1 {:.4}",
      cfg.k, o.mean, o.identical_sets, o.same_top1
    );
    overlap = Some(o);
    engines.push(tan);
  }
  let report = TextReport {
    kind: "text",
    dataset: json!({
      "name": "msmarco-passage",
      "docs": docs.len(),
      "text_bytes": bytes,
      "queries": queries.len(),
      "collection": cfg.collection.file_name().map(|f| f.to_string_lossy()),
      "queries_file": cfg.queries.file_name().map(|f| f.to_string_lossy()),
    }),
    analyzer: cfg.analyzer.name().into(),
    k: cfg.k,
    threads: pools.threads,
    engines,
    overlap,
    system: SystemInfo::collect(),
    notes: vec![
      "BM25 k1=1.2 b=0.75 in both engines; OR of distinct query terms.".into(),
      "Both searched from one memory-mapped segment after a warm-up pass."
        .into(),
    ],
  };
  let out = cfg.out_dir.join(format!(
    "msmarco-passage-{}__text-{}.json",
    docs.len(),
    cfg.analyzer.name()
  ));
  write_json(&out, &report)?;
  Ok(out)
}

fn log_search(name: &str, s: &TextSearchInfo, threads: usize) {
  eprintln!(
    "  {name}: 1t {:.0} q/s, {threads}t {:.0} q/s, p50 {:.1} us, p99 {:.1} \
     us, {:.2} hits/query",
    s.qps_1t, s.qps_nt, s.latency.p50_us, s.latency.p99_us, s.mean_hits
  );
}

#[cfg(test)]
mod tests {
  use super::*;

  const DOCS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog",
    "A fox is a small omnivorous mammal",
    "Dogs are loyal companions and good runners",
    "Running foxes were seen near the river bank",
    "Café culture in Paris: coffee, croissants and conversation",
    "The river flows quickly past the old mill",
    "Brown bears and red foxes share the forest",
    "Quick facts about running shoes for marathon runners",
  ];

  fn setup(name: &str) -> (PathBuf, TextRun) {
    let root = std::env::temp_dir()
      .join(format!("muninn-bench-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let coll: String = DOCS
      .iter()
      .enumerate()
      .map(|(i, d)| format!("{i}\t{d}\n"))
      .collect();
    std::fs::write(root.join("collection.tsv"), coll).unwrap();
    std::fs::write(
      root.join("queries.tsv"),
      "1\tquick fox\n2\trunning river\n3\tcafe\n4\tthe\n5\tzzz\n",
    )
    .unwrap();
    let cfg = TextRun {
      collection: root.join("collection.tsv"),
      queries: root.join("queries.tsv"),
      limit: None,
      query_limit: None,
      analyzer: AnalyzerKind::Default,
      k: 3,
      runs: 1,
      tantivy_index_threads: vec![1, 2],
      work_dir: root.join("work"),
      out_dir: root.join("out"),
    };
    (root, cfg)
  }

  #[test]
  fn reads_tsv_with_limit() {
    let (root, cfg) = setup("tsv");
    let (ids, texts) = read_tsv(&cfg.collection, Some(3)).unwrap();
    assert_eq!(ids, vec![0, 1, 2]);
    assert_eq!(texts[1], DOCS[1]);
    std::fs::write(root.join("bad.tsv"), "no tab here\n").unwrap();
    assert!(read_tsv(&root.join("bad.tsv"), None).is_err());
    std::fs::remove_dir_all(root).unwrap();
  }

  #[test]
  fn muninn_and_tantivy_agree_with_the_default_analyzer() {
    let pools = Pools::new(2).unwrap();
    let (root, cfg) = setup("agree");
    let out = run_text(&cfg, &pools).unwrap();
    let v: serde_json::Value =
      serde_json::from_reader(File::open(out).unwrap()).unwrap();
    assert_eq!(v["engines"].as_array().unwrap().len(), 2);
    assert_eq!(v["engines"][1]["other_index_runs"][0]["threads"], 2);
    // Same BM25 and equivalent analysis: the same documents come back.
    assert_eq!(v["overlap"]["mean"], 1.0, "{v:#}");
    std::fs::remove_dir_all(root).unwrap();
  }

  #[test]
  fn english_analyzers_both_stem_and_drop_stop_words() {
    let pools = Pools::new(1).unwrap();
    let (root, mut cfg) = setup("english");
    cfg.analyzer = AnalyzerKind::English;
    cfg.tantivy_index_threads = vec![1];
    let docs: Vec<String> = DOCS.iter().map(|d| d.to_string()).collect();
    let queries: Vec<String> =
      vec!["runners".into(), "the".into(), "cafe".into()];
    let (_, m) = muninn_engine(&cfg, &docs, &queries, &pools).unwrap();
    let (_, t) = tantivy_engine(&cfg, &docs, &queries, &pools).unwrap();
    // "runners" stems to "runner" in both; "the" is a stop word; "café"
    // is folded at index time.
    assert_eq!(m[0], t[0]);
    assert!(m[1].is_empty() && t[1].is_empty());
    assert_eq!(m[2], vec![4]);
    assert_eq!(t[2], vec![4]);
    std::fs::remove_dir_all(root).unwrap();
  }

  #[test]
  fn compares_result_lists() {
    let o = compare(
      &[vec![1, 2], vec![3, 4], vec![]],
      &[vec![1, 2], vec![4, 5], vec![]],
      ["a", "b"],
    );
    assert_eq!(o.mean, 2.5 / 3.0);
    assert_eq!(o.identical_sets, 2.0 / 3.0);
    assert_eq!(o.same_top1, 2.0 / 3.0);
  }
}
