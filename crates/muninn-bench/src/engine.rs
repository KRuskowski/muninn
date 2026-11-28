//! End-to-end benchmark of the `muninn` engine: documents go in through
//! `Collection::upsert` (write-ahead log, memtable, background flushes
//! and merges) and come out through `Collection::search`, exactly as an
//! application uses muninn.
//!
//! Ingest is measured in three parts: `upsert` of every batch (each
//! acknowledged as durable as the chosen durability promises), `flush`
//! (everything into segments), and `compact` (all segments merged into
//! one, the state the component benchmarks measure). Searches then run
//! against that single segment with the same protocol as every other
//! engine (see [`crate::run`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use muninn::store::{ExternalId, MANIFEST_PREFIX, Manifest, map_file};
use muninn::{
  Collection, Database, Document, Durability, IndexKind, Options, Query,
  Schema, VectorField,
};
use serde::Serialize;
use serde_json::json;

use crate::Result;
use crate::dataset::Dataset;
use crate::measure::recall_at_k;
use crate::run::{Pools, measure, write_json};
use crate::sys::{SystemInfo, dir_size, peak_rss_bytes, rss_bytes, rss_delta};
use crate::text::{
  AnalyzerKind, OverlapInfo, TextEngine, TextIndexInfo, TextReport, TextRun,
  TextSearchInfo, compare, muninn_component, read_tsv,
};
use crate::vector::{
  BuildInfo, DatasetInfo, SweepPoint, VectorReport, muninn_version,
};

/// Parses `each` (fsync every upsert), `batch` (fsync at most 10 ms
/// later) or `none` (leave it to the OS).
pub fn parse_durability(s: &str) -> Result<Durability> {
  match s {
    "each" => Ok(Durability::SyncEachWrite),
    "batch" => Ok(Durability::SyncBatch {
      max_delay: Duration::from_millis(10),
    }),
    "none" => Ok(Durability::NoSync),
    _ => Err(format!("unknown durability {s:?}").into()),
  }
}

/// Ingest figures.
#[derive(Clone, Debug, Serialize)]
pub struct Ingest {
  /// Documents written.
  pub docs: usize,
  /// Documents per `upsert` call.
  pub batch: usize,
  /// Wall time of all `upsert` calls (including building the
  /// `Document`s, as an application would).
  pub upsert_seconds: f64,
  /// `flush`: memtable and pending flushes into segments.
  pub flush_seconds: f64,
  /// `compact`: all segments merged into one.
  pub compact_seconds: f64,
  /// Segments right after `flush` (background merges may have run).
  pub segments_after_flush: usize,
  /// Segments after `compact`.
  pub segments_after_compact: usize,
  /// Size of the database directory right after `compact`, while the
  /// database is still open (may include files that are no longer
  /// live but not yet removed).
  pub dir_bytes_after_compact: u64,
  /// Time to close the database and open it again.
  pub reopen_seconds: f64,
  /// Size of the database directory after reopening. Includes the
  /// segments of the previous manifest generation, which muninn keeps
  /// as a crash fallback until the next commit.
  pub dir_bytes: u64,
  /// Bytes of the files the current manifest references (the live
  /// segments), plus the manifest itself and the write-ahead log.
  pub live_bytes: u64,
  /// `dir_bytes` by file extension (`vec`, `ids`, `docs`, `wal`, ...).
  pub bytes_by_kind: BTreeMap<String, u64>,
  /// Resident memory growth over the whole ingest, when measurable.
  pub rss_delta_bytes: Option<i64>,
}

/// Bytes of the files referenced by the newest manifest in `coll_dir`,
/// plus that manifest and the write-ahead log directory.
fn live_bytes(coll_dir: &Path) -> Result<u64> {
  let mut newest: Option<(u64, PathBuf)> = None;
  for entry in std::fs::read_dir(coll_dir)? {
    let p = entry?.path();
    let name = p.file_name().unwrap_or_default().to_string_lossy();
    if let Some(g) = name.strip_prefix(MANIFEST_PREFIX)
      && let Ok(g) = g.parse::<u64>()
      && newest.as_ref().is_none_or(|n| g > n.0)
    {
      newest = Some((g, p.clone()));
    }
  }
  let (_, path) = newest.ok_or("no manifest found")?;
  let manifest = Manifest::decode(map_file(&path)?)?;
  let mut total = std::fs::metadata(&path)?.len();
  for f in manifest.referenced_files() {
    total += std::fs::metadata(coll_dir.join(f))?.len();
  }
  let wal = coll_dir.join("wal");
  if wal.exists() {
    total += dir_size(&wal)?;
  }
  Ok(total)
}

fn bytes_by_kind(dir: &Path, out: &mut BTreeMap<String, u64>) -> Result<()> {
  for entry in std::fs::read_dir(dir)? {
    let p = entry?.path();
    if p.is_dir() {
      bytes_by_kind(&p, out)?;
    } else {
      let kind = p
        .extension()
        .map_or("other".to_string(), |e| e.to_string_lossy().into_owned());
      *out.entry(kind).or_default() += std::fs::metadata(&p)?.len();
    }
  }
  Ok(())
}

/// Creates a fresh database in `work_dir/name` with one collection,
/// ingests `n` documents, flushes, compacts to one segment, closes and
/// reopens it. Returns the reopened database and collection.
fn ingest(
  work_dir: &Path,
  name: &str,
  durability: Durability,
  schema: Schema,
  n: usize,
  batch: usize,
  make: impl Fn(usize) -> Document,
) -> Result<(PathBuf, Database, Collection, Ingest)> {
  let dir = work_dir.join(name);
  if dir.exists() {
    std::fs::remove_dir_all(&dir)?;
  }
  std::fs::create_dir_all(&dir)?;
  let opts = Options::default().durability(durability);
  let db = Database::open(&dir, opts.clone())?;
  let coll = db.create_collection("bench", schema)?;
  let rss0 = rss_bytes();
  let t = Instant::now();
  let mut start = 0;
  while start < n {
    let end = (start + batch).min(n);
    let docs: Vec<Document> = (start..end).map(&make).collect();
    coll.upsert(&docs)?;
    start = end;
  }
  let upsert_s = t.elapsed().as_secs_f64();
  eprintln!(
    "  upserted {n} docs in {upsert_s:.1} s ({:.0} docs/s)",
    n as f64 / upsert_s
  );
  let t = Instant::now();
  coll.flush()?;
  let flush_s = t.elapsed().as_secs_f64();
  let after_flush = coll.stats()?.segments.len();
  let t = Instant::now();
  coll.compact()?;
  let compact_s = t.elapsed().as_secs_f64();
  let stats = coll.stats()?;
  let raw_bytes = dir_size(&dir)?;
  let rss = rss_delta(rss0, rss_bytes());
  if stats.live_docs != n as u64 {
    return Err(
      format!(
        "{} live documents after ingest, expected {n}",
        stats.live_docs
      )
      .into(),
    );
  }
  drop(coll);
  drop(db);
  let t = Instant::now();
  let db = Database::open(&dir, opts)?;
  let coll = db.collection("bench")?;
  let reopen_s = t.elapsed().as_secs_f64();
  let bytes = dir_size(&dir)?;
  let live = live_bytes(&dir.join("collections").join("bench"))?;
  let mut kinds = BTreeMap::new();
  bytes_by_kind(&dir, &mut kinds)?;
  eprintln!(
    "  flush {flush_s:.1} s ({after_flush} segments), compact \
     {compact_s:.1} s ({} segments); {:.1} MB on disk after compact, \
     {:.1} MB after reopening ({reopen_s:.2} s), {:.1} MB live",
    stats.segments.len(),
    raw_bytes as f64 / 1e6,
    bytes as f64 / 1e6,
    live as f64 / 1e6
  );
  let ing = Ingest {
    docs: n,
    batch,
    upsert_seconds: upsert_s,
    flush_seconds: flush_s,
    compact_seconds: compact_s,
    segments_after_flush: after_flush,
    segments_after_compact: stats.segments.len(),
    dir_bytes_after_compact: raw_bytes,
    reopen_seconds: reopen_s,
    dir_bytes: bytes,
    live_bytes: live,
    bytes_by_kind: kinds,
    rss_delta_bytes: rss,
  };
  Ok((dir, db, coll, ing))
}

fn hit_ids(hits: Vec<muninn::SearchHit>) -> Vec<u32> {
  hits
    .into_iter()
    .map(|h| match h.id {
      ExternalId::U64(v) => v as u32,
      ExternalId::Str(s) => panic!("unexpected string id {s:?}"),
    })
    .collect()
}

/// Settings of an end-to-end vector benchmark.
#[derive(Clone, Debug)]
pub struct EngineVectorRun {
  /// Dataset directory.
  pub data: PathBuf,
  /// HNSW `M`.
  pub m: usize,
  /// HNSW `ef_construction`.
  pub ef_construction: usize,
  /// Search `ef` values to sweep.
  pub efs: Vec<usize>,
  /// Results per query.
  pub k: usize,
  /// Use only the first this many queries.
  pub query_limit: Option<usize>,
  /// Measurement repetitions.
  pub runs: usize,
  /// Documents per `upsert`.
  pub batch: usize,
  /// Write durability (`each`, `batch`, `none`).
  pub durability: String,
  /// Where the database goes.
  pub work_dir: PathBuf,
  /// Where result JSON goes.
  pub out_dir: PathBuf,
}

/// Ingests a vector dataset into a collection and sweeps `ef`.
pub fn run_engine_vector(
  cfg: &EngineVectorRun,
  pools: &Pools,
) -> Result<PathBuf> {
  let ds = Dataset::load(&cfg.data)?;
  let metric = ds.metric()?;
  let name = ds.meta.name.clone();
  let nq = cfg
    .query_limit
    .map_or(ds.query.rows, |l| l.min(ds.query.rows));
  let qs: Vec<&[f32]> = (0..nq).map(|i| ds.query.row(i)).collect();
  eprintln!(
    "{name}: engine ingest of {} vectors in batches of {} (durability \
     {}), HNSW M={} efC={}",
    ds.base.rows, cfg.batch, cfg.durability, cfg.m, cfg.ef_construction
  );
  let field = VectorField {
    index: IndexKind::Hnsw {
      m: cfg.m,
      ef_construction: cfg.ef_construction,
    },
    ..VectorField::hnsw(ds.dim(), metric)
  };
  let durability = parse_durability(&cfg.durability)?;
  let (dir, db, coll, ing) = pools.all.install(|| {
    ingest(
      &cfg.work_dir,
      &format!("engine-{name}"),
      durability,
      Schema::vector(field),
      ds.base.rows,
      cfg.batch,
      |i| Document::new(i as u64).vector(ds.base.row(i).to_vec()),
    )
  })?;
  let mut results = Vec::new();
  for &ef in &cfg.efs {
    let m = measure(pools, &qs, cfg.runs, |q| {
      hit_ids(
        coll
          .search(&Query::vector(q.to_vec()).k(cfg.k).ef(ef))
          .expect("engine search"),
      )
    });
    let recall = recall_at_k(&m.results, &ds.gt, cfg.k);
    eprintln!(
      "  ef {ef:4}: recall {recall:.4}  1t {:9.0} q/s  {}t {:9.0} q/s  \
       p50 {:7.1} us  p99 {:7.1} us",
      m.qps_1t, pools.threads, m.qps_nt, m.latency.p50_us, m.latency.p99_us
    );
    results.push(SweepPoint {
      ef,
      recall,
      qps_1t: m.qps_1t,
      qps_nt: m.qps_nt,
      latency: m.latency,
      deterministic: m.same_results,
    });
  }
  let report = VectorReport {
    kind: "vector",
    engine: "muninn-engine".into(),
    engine_version: muninn_version(),
    dataset: DatasetInfo::new(&ds, nq),
    k: cfg.k,
    threads: pools.threads,
    params: json!({
      "M": cfg.m,
      "ef_construction": cfg.ef_construction,
      "level0_neighbours": 2 * cfg.m,
      "batch": cfg.batch,
      "durability": cfg.durability,
      "memtable_max_docs": Options::default().memtable_max_docs,
    }),
    build: BuildInfo {
      seconds: ing.upsert_seconds + ing.flush_seconds + ing.compact_seconds,
      threads: pools.threads,
      index_bytes: ing.live_bytes,
      rss_delta_bytes: ing.rss_delta_bytes,
      peak_rss_bytes: peak_rss_bytes(),
      details: serde_json::to_value(&ing)?,
    },
    results,
    system: SystemInfo::collect(),
    notes: vec![
      "End to end through Collection::upsert / flush / compact / search; \
       build seconds = upsert + flush + compact (to one segment)."
        .into(),
      "Index size counts the files the current manifest references \
       (vectors, graph, id map, stored documents) plus manifest and log, \
       after closing and reopening the database; dir_bytes also counts \
       the previous generation kept as a crash fallback. Searches run on \
       the reopened collection."
        .into(),
    ],
  };
  drop(coll);
  drop(db);
  std::fs::remove_dir_all(&dir)?;
  let out = cfg.out_dir.join(format!(
    "{name}__muninn-engine__M{}_efc{}.json",
    cfg.m, cfg.ef_construction
  ));
  write_json(&out, &report)?;
  Ok(out)
}

/// Settings of an end-to-end text benchmark.
#[derive(Clone, Debug)]
pub struct EngineTextRun {
  /// The component-level settings (files, limits, analyzer, k, runs).
  pub text: TextRun,
  /// Documents per `upsert`.
  pub batch: usize,
  /// Write durability (`each`, `batch`, `none`).
  pub durability: String,
}

/// Ingests MS MARCO passages into a collection and measures BM25 top-k
/// through `Collection::search`, comparing results with the
/// component-level text index as a sanity check.
pub fn run_engine_text(cfg: &EngineTextRun, pools: &Pools) -> Result<PathBuf> {
  let tc = &cfg.text;
  let (_, docs) = read_tsv(&tc.collection, tc.limit)?;
  let (_, queries) = read_tsv(&tc.queries, tc.query_limit)?;
  let bytes: usize = docs.iter().map(String::len).sum();
  eprintln!(
    "msmarco-passage: engine ingest of {} passages in batches of {} \
     (durability {}), analyzer {}",
    docs.len(),
    cfg.batch,
    cfg.durability,
    tc.analyzer.name()
  );
  let durability = parse_durability(&cfg.durability)?;
  let schema = Schema::text(["text"]).with_analyzer(tc.analyzer.config());
  let (dir, db, coll, ing) = pools.all.install(|| {
    ingest(
      &tc.work_dir,
      &format!("engine-text-{}", tc.analyzer.name()),
      durability,
      schema,
      docs.len(),
      cfg.batch,
      |i| Document::new(i as u64).text("text", docs[i].as_str()),
    )
  })?;
  let k = tc.k;
  let m = measure(pools, &queries, tc.runs, |q| {
    hit_ids(
      coll
        .search(&Query::text(q.as_str()).k(k))
        .expect("engine search"),
    )
  });
  let hits: usize = m.results.iter().map(Vec::len).sum();
  let total = ing.upsert_seconds + ing.flush_seconds + ing.compact_seconds;
  let engine = TextEngine {
    engine: "muninn-engine".into(),
    engine_version: muninn_version(),
    analyzer: tc.analyzer.muninn_description().into(),
    index: TextIndexInfo {
      threads: 1,
      index_seconds: ing.upsert_seconds,
      finish_seconds: ing.flush_seconds + ing.compact_seconds,
      total_seconds: total,
      docs_per_sec: docs.len() as f64 / total,
      index_bytes: ing.live_bytes,
      rss_delta_bytes: ing.rss_delta_bytes,
      details: serde_json::to_value(&ing)?,
    },
    other_index_runs: Vec::new(),
    search: TextSearchInfo {
      k,
      queries: m.results.len(),
      qps_1t: m.qps_1t,
      qps_nt: m.qps_nt,
      latency: m.latency,
      mean_hits: hits as f64 / m.results.len().max(1) as f64,
      deterministic: m.same_results,
    },
  };
  eprintln!(
    "  muninn-engine: 1t {:.0} q/s, {}t {:.0} q/s, p50 {:.1} us, p99 {:.1} \
     us",
    m.qps_1t, pools.threads, m.qps_nt, m.latency.p50_us, m.latency.p99_us
  );
  drop(coll);
  drop(db);
  std::fs::remove_dir_all(&dir)?;
  // The engine must rank exactly like the component index it wraps.
  let (component, comp_ids) = muninn_component(tc, &docs, &queries, pools)?;
  let overlap: OverlapInfo =
    compare(&m.results, &comp_ids, ["muninn-engine", "muninn"]);
  eprintln!(
    "  overlap@{k} with the component index: mean {:.4}, identical sets \
     {:.4}",
    overlap.mean, overlap.identical_sets
  );
  let report = TextReport {
    kind: "text",
    dataset: json!({
      "name": "msmarco-passage",
      "docs": docs.len(),
      "text_bytes": bytes,
      "queries": queries.len(),
      "collection": tc.collection.file_name().map(|f| f.to_string_lossy()),
      "queries_file": tc.queries.file_name().map(|f| f.to_string_lossy()),
    }),
    analyzer: tc.analyzer.name().into(),
    k,
    threads: pools.threads,
    engines: vec![engine, component],
    overlap: Some(overlap),
    system: SystemInfo::collect(),
    notes: vec![
      "End to end through Collection::upsert / flush / compact / search, \
       next to the component-level index on the same passages."
        .into(),
      format!(
        "Engine: batches of {} documents, durability {}.",
        cfg.batch, cfg.durability
      ),
    ],
  };
  let out = tc.out_dir.join(format!(
    "msmarco-passage-{}__engine-text-{}.json",
    docs.len(),
    tc.analyzer.name()
  ));
  write_json(&out, &report)?;
  Ok(out)
}

impl AnalyzerKind {
  /// The muninn analyzer configuration for this variant.
  pub fn config(self) -> muninn_text::AnalyzerConfig {
    match self {
      Self::Default => muninn_text::AnalyzerConfig::default(),
      Self::English => muninn_text::AnalyzerConfig::english(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_durability() {
    assert!(matches!(
      parse_durability("each").unwrap(),
      Durability::SyncEachWrite
    ));
    assert!(matches!(
      parse_durability("none").unwrap(),
      Durability::NoSync
    ));
    assert!(parse_durability("sometimes").is_err());
  }

  fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir()
      .join(format!("muninn-bench-{}-{name}", std::process::id()))
  }

  #[test]
  fn vector_ingest_and_search_through_a_collection() {
    let pools = Pools::new(2).unwrap();
    let root = tmp("engine-vec");
    let data = root.join("data");
    crate::vector::tests::tiny_dataset(&data, muninn::Metric::L2);
    let cfg = EngineVectorRun {
      data,
      m: 8,
      ef_construction: 64,
      efs: vec![64],
      k: 10,
      query_limit: None,
      runs: 1,
      batch: 100,
      durability: "none".into(),
      work_dir: root.join("work"),
      out_dir: root.join("out"),
    };
    let out = run_engine_vector(&cfg, &pools).unwrap();
    let v: serde_json::Value =
      serde_json::from_reader(std::fs::File::open(out).unwrap()).unwrap();
    assert_eq!(v["engine"], "muninn-engine");
    // 600 documents stay below the HNSW threshold: an exact segment.
    assert_eq!(v["results"][0]["recall"], 1.0);
    let d = &v["build"]["details"];
    assert_eq!(d["segments_after_compact"], 1);
    let live = d["live_bytes"].as_u64().unwrap();
    assert!(live > 600 * 8 * 4 && live <= d["dir_bytes"].as_u64().unwrap());
    std::fs::remove_dir_all(root).unwrap();
  }

  #[test]
  fn text_ingest_matches_the_component_index() {
    let pools = Pools::new(2).unwrap();
    let root = tmp("engine-text");
    std::fs::create_dir_all(&root).unwrap();
    let docs = [
      "the quick brown fox",
      "a lazy dog sleeps",
      "quick dogs and slow foxes",
      "rivers flow to the sea",
    ];
    let coll: String = docs
      .iter()
      .enumerate()
      .map(|(i, d)| format!("{i}\t{d}\n"))
      .collect();
    std::fs::write(root.join("c.tsv"), coll).unwrap();
    std::fs::write(root.join("q.tsv"), "1\tquick fox\n2\tsea\n").unwrap();
    let cfg = EngineTextRun {
      text: TextRun {
        collection: root.join("c.tsv"),
        queries: root.join("q.tsv"),
        limit: None,
        query_limit: None,
        analyzer: AnalyzerKind::Default,
        k: 3,
        runs: 1,
        tantivy_index_threads: Vec::new(),
        work_dir: root.join("work"),
        out_dir: root.join("out"),
      },
      batch: 2,
      durability: "each".into(),
    };
    let out = run_engine_text(&cfg, &pools).unwrap();
    let v: serde_json::Value =
      serde_json::from_reader(std::fs::File::open(out).unwrap()).unwrap();
    assert_eq!(v["overlap"]["mean"], 1.0, "{v:#}");
    std::fs::remove_dir_all(root).unwrap();
  }
}
