//! muninn vector benchmarks: HNSW (full precision and quantised) and flat
//! search sweeps, and ground-truth verification.
//!
//! Every index is built in memory on the all-thread pool, written to a
//! file (fsynced) to measure its size, then memory-mapped and searched
//! from the mapped file, the way a muninn segment is served.

use std::path::{Path, PathBuf};
use std::time::Instant;

use muninn_core::{DocId, Metric};
use muninn_store::map_file;
use muninn_vector::{
  FlatIndex, HnswIndex, HnswParams, Quantization, SearchParams, VectorIndex,
};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

use crate::Result;
use crate::dataset::Dataset;
use crate::measure::{Latency, recall_at_k};
use crate::run::{Pools, measure, write_file, write_json};
use crate::sys::{SystemInfo, peak_rss_bytes, rss_bytes, rss_delta};

/// The dataset part of a report.
#[derive(Clone, Debug, Serialize)]
pub struct DatasetInfo {
  /// Dataset name.
  pub name: String,
  /// Metric name as in `meta.json`.
  pub metric: String,
  /// Indexed vectors.
  pub n: usize,
  /// Dimensions.
  pub dim: usize,
  /// Queries measured.
  pub queries: usize,
  /// The rest of `meta.json` (sources, checksums).
  pub meta: serde_json::Map<String, serde_json::Value>,
}

impl DatasetInfo {
  pub(crate) fn new(ds: &Dataset, queries: usize) -> Self {
    Self {
      name: ds.meta.name.clone(),
      metric: ds.meta.metric.clone(),
      n: ds.base.rows,
      dim: ds.dim(),
      queries,
      meta: ds.meta.extra.clone(),
    }
  }
}

/// Index construction figures.
#[derive(Clone, Debug, Serialize)]
pub struct BuildInfo {
  /// Wall time to build the searchable in-memory index.
  pub seconds: f64,
  /// Threads used for the build.
  pub threads: usize,
  /// Size of the index file on disk.
  pub index_bytes: u64,
  /// Growth of resident memory during the build (excludes the dataset
  /// itself, which was loaded before), when measurable.
  pub rss_delta_bytes: Option<i64>,
  /// Peak resident memory of the process so far.
  pub peak_rss_bytes: Option<u64>,
  /// Engine-specific timings (file write, quantiser training).
  pub details: serde_json::Value,
}

/// One point of a search sweep.
#[derive(Clone, Debug, Serialize)]
pub struct SweepPoint {
  /// Candidate-list size (HNSW `ef`); 0 for exact search.
  pub ef: usize,
  /// Mean recall@k against the dataset's ground truth.
  pub recall: f64,
  /// Queries per second, one thread.
  pub qps_1t: f64,
  /// Queries per second, all threads.
  pub qps_nt: f64,
  /// One-thread latency distribution.
  pub latency: Latency,
  /// Whether all-thread and one-thread passes returned the same ids.
  pub deterministic: bool,
}

/// A vector benchmark result file.
#[derive(Clone, Debug, Serialize)]
pub struct VectorReport {
  /// Always `"vector"`.
  pub kind: &'static str,
  /// Engine name, e.g. `muninn-hnsw`.
  pub engine: String,
  /// Engine version.
  pub engine_version: String,
  /// The dataset.
  pub dataset: DatasetInfo,
  /// Results per query.
  pub k: usize,
  /// Threads in the all-thread pool.
  pub threads: usize,
  /// Index parameters.
  pub params: serde_json::Value,
  /// Build figures.
  pub build: BuildInfo,
  /// Search sweep.
  pub results: Vec<SweepPoint>,
  /// Machine and build description.
  pub system: SystemInfo,
  /// Free-form remarks.
  pub notes: Vec<String>,
}

/// The muninn version string recorded in reports.
pub fn muninn_version() -> String {
  let v = env!("CARGO_PKG_VERSION");
  match std::env::var("BENCH_GIT_COMMIT") {
    Ok(c) if !c.is_empty() => format!("{v} ({})", &c[..c.len().min(12)]),
    _ => v.to_string(),
  }
}

/// Settings of an HNSW benchmark.
#[derive(Clone, Debug)]
pub struct HnswRun {
  /// Dataset directory.
  pub data: PathBuf,
  /// HNSW `M`.
  pub m: usize,
  /// HNSW `ef_construction`.
  pub ef_construction: usize,
  /// Search `ef` values to sweep.
  pub efs: Vec<usize>,
  /// Traversal variants measured on the same graph.
  pub quant: Vec<Quantization>,
  /// Results per query.
  pub k: usize,
  /// Use only the first this many queries.
  pub query_limit: Option<usize>,
  /// Measurement repetitions (best is kept).
  pub runs: usize,
  /// Where index files go.
  pub work_dir: PathBuf,
  /// Where result JSON goes.
  pub out_dir: PathBuf,
}

fn engine_name(q: Quantization) -> &'static str {
  match q {
    Quantization::None => "muninn-hnsw",
    Quantization::Sq8 => "muninn-hnsw-sq8",
    Quantization::Binary => "muninn-hnsw-binary",
  }
}

/// Parses `none`, `sq8` or `binary`.
pub fn parse_quantization(s: &str) -> Result<Quantization> {
  match s {
    "none" => Ok(Quantization::None),
    "sq8" => Ok(Quantization::Sq8),
    "binary" => Ok(Quantization::Binary),
    _ => Err(format!("unknown quantisation {s:?}").into()),
  }
}

fn query_rows(ds: &Dataset, limit: Option<usize>) -> Vec<&[f32]> {
  let n = limit.map_or(ds.query.rows, |l| l.min(ds.query.rows));
  (0..n).map(|i| ds.query.row(i)).collect()
}

fn sweep(
  idx: &dyn VectorIndex,
  qs: &[&[f32]],
  ds: &Dataset,
  k: usize,
  efs: &[usize],
  runs: usize,
  pools: &Pools,
) -> Result<Vec<SweepPoint>> {
  let mut out = Vec::new();
  for &ef in efs {
    let p = SearchParams::new(k).with_ef(ef);
    let m = measure(pools, qs, runs, |q| {
      idx
        .search(q, &p, None)
        .expect("search failed")
        .into_iter()
        .map(|h| h.id)
        .collect::<Vec<DocId>>()
    });
    let recall = recall_at_k(&m.results, &ds.gt, k);
    eprintln!(
      "  ef {ef:4}: recall {recall:.4}  1t {:9.0} q/s  {}t {:9.0} q/s  \
       p50 {:7.1} us  p99 {:7.1} us",
      m.qps_1t, pools.threads, m.qps_nt, m.latency.p50_us, m.latency.p99_us
    );
    out.push(SweepPoint {
      ef,
      recall,
      qps_1t: m.qps_1t,
      qps_nt: m.qps_nt,
      latency: m.latency,
      deterministic: m.same_results,
    });
  }
  Ok(out)
}

/// Builds one HNSW graph and sweeps `ef` for each traversal variant.
/// Writes one JSON file per variant and returns their paths.
pub fn run_hnsw(cfg: &HnswRun, pools: &Pools) -> Result<Vec<PathBuf>> {
  let ds = Dataset::load(&cfg.data)?;
  let metric = ds.metric()?;
  let qs = query_rows(&ds, cfg.query_limit);
  let name = ds.meta.name.clone();
  eprintln!(
    "{name}: {} x {} ({metric:?}), {} queries; HNSW M={} efC={} on {} \
     threads",
    ds.base.rows,
    ds.dim(),
    qs.len(),
    cfg.m,
    cfg.ef_construction,
    pools.threads
  );
  let params = HnswParams::default()
    .with_m(cfg.m)
    .with_ef_construction(cfg.ef_construction);
  let rss0 = rss_bytes();
  let t = Instant::now();
  let mut built = pools
    .all
    .install(|| HnswIndex::build(ds.dim(), metric, params, &ds.base.data))?;
  let build_s = t.elapsed().as_secs_f64();
  let rss_build = rss_delta(rss0, rss_bytes());
  let st = built.stats();
  eprintln!(
    "  built in {build_s:.1} s (graph {:.1} MB, max level {}, avg degree \
     {})",
    st.graph_bytes as f64 / 1e6,
    st.max_level,
    st.avg_degree
  );
  let mut written = Vec::new();
  for &q in &cfg.quant {
    let engine = engine_name(q);
    let t = Instant::now();
    if q != Quantization::None {
      pools.all.install(|| built.set_quantization(q));
    }
    let quant_s = t.elapsed().as_secs_f64();
    let file = cfg.work_dir.join(format!(
      "{name}__{engine}__M{}_efc{}.idx",
      cfg.m, cfg.ef_construction
    ));
    let (write_s, bytes) = write_file(&file, |w| built.write_to(w))?;
    let opened = HnswIndex::open(map_file(&file)?)?;
    eprintln!(
      "  {engine}: quantiser {quant_s:.2} s, file {:.1} MB written in \
       {write_s:.2} s",
      bytes as f64 / 1e6
    );
    let results = sweep(&opened, &qs, &ds, cfg.k, &cfg.efs, cfg.runs, pools)?;
    let stats = opened.stats();
    drop(opened);
    let report = VectorReport {
      kind: "vector",
      engine: engine.to_string(),
      engine_version: muninn_version(),
      dataset: DatasetInfo::new(&ds, qs.len()),
      k: cfg.k,
      threads: pools.threads,
      params: json!({
        "M": cfg.m,
        "ef_construction": cfg.ef_construction,
        "level0_neighbours": 2 * cfg.m,
        "quantization": format!("{q:?}"),
        "rerank": match q {
          Quantization::None => serde_json::Value::Null,
          _ => json!(q.default_rerank(cfg.k)),
        },
      }),
      build: BuildInfo {
        seconds: build_s + quant_s,
        threads: pools.threads,
        index_bytes: bytes,
        rss_delta_bytes: rss_build,
        peak_rss_bytes: peak_rss_bytes(),
        details: json!({
          "graph_seconds": build_s,
          "quantizer_seconds": quant_s,
          "write_seconds": write_s,
          "vector_bytes": stats.vector_bytes,
          "graph_bytes": stats.graph_bytes,
          "code_bytes": stats.code_bytes,
          "max_level": stats.max_level,
          "avg_level0_degree": stats.avg_degree,
        }),
      },
      results,
      system: SystemInfo::collect(),
      notes: vec![
        "Searched from the memory-mapped index file.".into(),
        "Graph built in parallel with all threads; queries measured one \
         at a time on one thread, and spread over all threads."
          .into(),
      ],
    };
    let out = cfg.out_dir.join(format!(
      "{name}__{engine}__M{}_efc{}.json",
      cfg.m, cfg.ef_construction
    ));
    write_json(&out, &report)?;
    std::fs::remove_file(&file)?;
    written.push(out);
  }
  Ok(written)
}

/// Settings of a flat (exact) benchmark.
#[derive(Clone, Debug)]
pub struct FlatRun {
  /// Dataset directory.
  pub data: PathBuf,
  /// Results per query.
  pub k: usize,
  /// Use only the first this many queries (exact search is slow).
  pub query_limit: Option<usize>,
  /// Measurement repetitions.
  pub runs: usize,
  /// Where the index file goes.
  pub work_dir: PathBuf,
  /// Where result JSON goes.
  pub out_dir: PathBuf,
}

/// Measures exact search with [`FlatIndex`]. Returns the result path.
pub fn run_flat(cfg: &FlatRun, pools: &Pools) -> Result<PathBuf> {
  let ds = Dataset::load(&cfg.data)?;
  let metric = ds.metric()?;
  let qs = query_rows(&ds, cfg.query_limit);
  let name = ds.meta.name.clone();
  eprintln!(
    "{name}: flat scan, {} queries on {} threads",
    qs.len(),
    pools.threads
  );
  let rss0 = rss_bytes();
  let t = Instant::now();
  let flat = FlatIndex::from_vectors(ds.dim(), metric, &ds.base.data)?;
  let build_s = t.elapsed().as_secs_f64();
  let rss_build = rss_delta(rss0, rss_bytes());
  let file = cfg.work_dir.join(format!("{name}__muninn-flat.idx"));
  let (write_s, bytes) = write_file(&file, |w| flat.write_to(w))?;
  drop(flat);
  let opened = FlatIndex::open(map_file(&file)?)?;
  let results = sweep(&opened, &qs, &ds, cfg.k, &[0], cfg.runs, pools)?;
  drop(opened);
  let report = VectorReport {
    kind: "vector",
    engine: "muninn-flat".into(),
    engine_version: muninn_version(),
    dataset: DatasetInfo::new(&ds, qs.len()),
    k: cfg.k,
    threads: pools.threads,
    params: json!({ "exact": true }),
    build: BuildInfo {
      seconds: build_s,
      threads: 1,
      index_bytes: bytes,
      rss_delta_bytes: rss_build,
      peak_rss_bytes: peak_rss_bytes(),
      details: json!({ "write_seconds": write_s }),
    },
    results,
    system: SystemInfo::collect(),
    notes: vec![
      "Exact scan; recall below 1 only from distance ties.".into(),
      "Searched from the memory-mapped index file.".into(),
    ],
  };
  let out = cfg.out_dir.join(format!("{name}__muninn-flat.json"));
  write_json(&out, &report)?;
  std::fs::remove_file(&file)?;
  Ok(out)
}

/// Distance in `f64`, computed independently of muninn's kernels: squared
/// L2, negative dot product, or cosine distance `1 - cos`.
pub fn reference_distance(metric: Metric, a: &[f32], b: &[f32]) -> f64 {
  let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
  match metric {
    Metric::L2 => a
      .iter()
      .zip(b)
      .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
      .sum(),
    Metric::Dot => -dot,
    Metric::Cosine => {
      let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
      let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
      if na == 0.0 || nb == 0.0 {
        1.0
      } else {
        1.0 - dot / (na.sqrt() * nb.sqrt())
      }
    }
  }
}

/// Outcome of checking a dataset's ground truth against muninn's exact
/// search.
#[derive(Clone, Debug, Serialize)]
pub struct GtCheck {
  /// Always `"gt-check"`.
  pub kind: &'static str,
  /// Dataset name.
  pub dataset: String,
  /// Metric.
  pub metric: String,
  /// Queries checked (evenly spaced over the query set).
  pub sampled: usize,
  /// Neighbours compared per query.
  pub k: usize,
  /// Fraction of queries whose top-k id set equals the ground truth's.
  pub id_set_match: f64,
  /// Fraction of queries where every differing id lies at the same
  /// distance as the k-th true neighbour (a tie, so both are correct).
  pub match_with_ties: f64,
  /// Mean recall@k of muninn's exact search against the ground truth.
  pub recall: f64,
  /// Query numbers that matched neither way (at most 10).
  pub mismatches: Vec<usize>,
  /// Machine description.
  pub system: SystemInfo,
}

/// Recomputes exact neighbours with [`FlatIndex`] for `sample` queries
/// and compares them with the dataset's ground truth.
pub fn verify_ground_truth(
  data: &Path,
  sample: usize,
  k: usize,
  pools: &Pools,
) -> Result<GtCheck> {
  let ds = Dataset::load(data)?;
  let metric = ds.metric()?;
  if k > ds.gt.cols {
    return Err(
      format!("ground truth has only {} neighbours", ds.gt.cols).into(),
    );
  }
  let nq = ds.query.rows;
  let sample = sample.clamp(1, nq);
  let picks: Vec<usize> = (0..sample).map(|i| i * nq / sample).collect();
  let flat = FlatIndex::from_vectors(ds.dim(), metric, &ds.base.data)?;
  let p = SearchParams::new(k);
  let found: Vec<Vec<DocId>> = pools.all.install(|| {
    picks
      .par_iter()
      .map(|&q| {
        flat
          .search(ds.query.row(q), &p, None)
          .expect("flat search")
          .into_iter()
          .map(|h| h.id)
          .collect()
      })
      .collect()
  });
  let mut exact = 0usize;
  let mut tied = 0usize;
  let mut hits = 0usize;
  let mut mismatches = Vec::new();
  for (&q, got) in picks.iter().zip(&found) {
    let truth = &ds.gt.row(q)[..k];
    let qv = ds.query.row(q);
    hits += got.iter().filter(|id| truth.contains(id)).count();
    if got.len() == k && got.iter().all(|id| truth.contains(id)) {
      exact += 1;
      tied += 1;
      continue;
    }
    let kth =
      reference_distance(metric, qv, ds.base.row(truth[k - 1] as usize));
    let tol = 1e-5 * kth.abs().max(1e-3);
    let all_tied = got.len() == k
      && got.iter().all(|&id| {
        truth.contains(&id)
          || reference_distance(metric, qv, ds.base.row(id as usize))
            <= kth + tol
      });
    if all_tied {
      tied += 1;
    } else if mismatches.len() < 10 {
      mismatches.push(q);
    }
  }
  Ok(GtCheck {
    kind: "gt-check",
    dataset: ds.meta.name.clone(),
    metric: ds.meta.metric.clone(),
    sampled: sample,
    k,
    id_set_match: exact as f64 / sample as f64,
    match_with_ties: tied as f64 / sample as f64,
    recall: hits as f64 / (sample * k) as f64,
    mismatches,
    system: SystemInfo::collect(),
  })
}

#[cfg(test)]
pub(crate) mod tests {
  use super::*;
  use std::fs::File;

  use crate::dataset::{Matrix, write_fbin, write_ibin};

  /// A small clustered dataset with exact ground truth, on disk.
  pub(crate) fn tiny_dataset(dir: &Path, metric: Metric) {
    std::fs::create_dir_all(dir).unwrap();
    let (n, dim, nq, k) = (600, 8, 20, 10);
    let mut s = 0x1234_5678u64;
    let mut next = move || {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let base: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let query: Vec<f32> = (0..nq * dim).map(|_| next()).collect();
    let mut gt = Vec::new();
    for q in query.chunks(dim) {
      let mut d: Vec<(f64, u32)> = base
        .chunks(dim)
        .enumerate()
        .map(|(i, b)| (reference_distance(metric, q, b), i as u32))
        .collect();
      d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
      gt.extend(d[..k].iter().map(|x| x.1));
    }
    let name = match metric {
      Metric::Cosine => "cosine",
      Metric::Dot => "dot",
      Metric::L2 => "l2",
    };
    std::fs::write(
      dir.join("meta.json"),
      format!(r#"{{"name": "tiny-{name}", "metric": "{name}"}}"#),
    )
    .unwrap();
    let m = |rows, data| Matrix {
      rows,
      cols: dim,
      data,
    };
    write_fbin(&dir.join("base.fbin"), &m(n, base)).unwrap();
    write_fbin(&dir.join("query.fbin"), &m(nq, query)).unwrap();
    write_ibin(
      &dir.join("gt.ibin"),
      &Matrix {
        rows: nq,
        cols: k,
        data: gt,
      },
    )
    .unwrap();
  }

  fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir()
      .join(format!("muninn-bench-{}-{name}", std::process::id()))
  }

  #[test]
  fn ground_truth_check_agrees_with_flat_search() {
    let pools = Pools::new(2).unwrap();
    for metric in [Metric::L2, Metric::Cosine] {
      let dir = tmp(&format!("gt-{metric:?}"));
      tiny_dataset(&dir, metric);
      let c = verify_ground_truth(&dir, 20, 10, &pools).unwrap();
      assert_eq!(c.sampled, 20);
      assert_eq!(c.match_with_ties, 1.0, "{metric:?}: {c:?}");
      assert!(c.recall > 0.99);
      std::fs::remove_dir_all(dir).unwrap();
    }
  }

  #[test]
  fn hnsw_and_flat_runs_write_reports() {
    let pools = Pools::new(2).unwrap();
    let root = tmp("runs");
    let data = root.join("data");
    tiny_dataset(&data, Metric::L2);
    let cfg = HnswRun {
      data: data.clone(),
      m: 8,
      ef_construction: 64,
      efs: vec![10, 64],
      quant: vec![Quantization::None, Quantization::Sq8],
      k: 10,
      query_limit: Some(10),
      runs: 1,
      work_dir: root.join("work"),
      out_dir: root.join("out"),
    };
    let files = run_hnsw(&cfg, &pools).unwrap();
    assert_eq!(files.len(), 2);
    let v: serde_json::Value =
      serde_json::from_reader(File::open(&files[0]).unwrap()).unwrap();
    assert_eq!(v["engine"], "muninn-hnsw");
    assert_eq!(v["dataset"]["queries"], 10);
    assert_eq!(v["results"].as_array().unwrap().len(), 2);
    assert!(v["results"][1]["recall"].as_f64().unwrap() > 0.9);
    assert!(v["build"]["index_bytes"].as_u64().unwrap() > 600 * 8 * 4);
    let flat = run_flat(
      &FlatRun {
        data,
        k: 10,
        query_limit: None,
        runs: 1,
        work_dir: root.join("work"),
        out_dir: root.join("out"),
      },
      &pools,
    )
    .unwrap();
    let v: serde_json::Value =
      serde_json::from_reader(File::open(flat).unwrap()).unwrap();
    assert_eq!(v["results"][0]["recall"], 1.0);
    std::fs::remove_dir_all(root).unwrap();
  }
}
