//! Timing query passes: the one measurement protocol every engine in the
//! Rust harness goes through (the Python runners mirror it).
//!
//! For each configuration:
//!
//! 1. **Warm-up**: every query once on all threads, untimed, so caches
//!    and page mappings are in the same state for every engine.
//! 2. **One thread**: every query in order on a one-thread pool, timing
//!    each query; QPS is queries divided by the pass's wall time. Repeated
//!    `runs` times; the fastest pass is kept, with its latencies.
//! 3. **All threads**: every query on the full pool (one query per task);
//!    QPS from wall time, best of `runs`.

use std::fs::File;
use std::hint::black_box;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::Serialize;

use crate::Result;
use crate::measure::{Latency, summarize};

/// A one-thread pool and a pool with every benchmark thread.
pub struct Pools {
  /// One worker, for single-thread figures.
  pub one: ThreadPool,
  /// `threads` workers, for builds and all-thread figures.
  pub all: ThreadPool,
  /// Size of `all`.
  pub threads: usize,
}

impl Pools {
  /// Builds both pools. `threads == 0` means one per available CPU.
  pub fn new(threads: usize) -> Result<Self> {
    let threads = if threads == 0 {
      std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
      threads
    };
    Ok(Self {
      one: ThreadPoolBuilder::new()
        .num_threads(1)
        .thread_name(|_| "bench-1t".into())
        .build()?,
      all: ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("bench-{i}"))
        .build()?,
      threads,
    })
  }
}

/// Outcome of measuring one query configuration.
#[derive(Clone, Debug)]
pub struct Measured<R> {
  /// Per-query results from the kept one-thread pass.
  pub results: Vec<R>,
  /// Queries per second on one thread.
  pub qps_1t: f64,
  /// Queries per second on all threads.
  pub qps_nt: f64,
  /// One-thread latencies of the kept pass.
  pub latency: Latency,
  /// Whether the all-thread pass returned exactly the one-thread results.
  pub same_results: bool,
}

/// Measures `search` over `queries` following the module's protocol.
pub fn measure<Q, R, F>(
  pools: &Pools,
  queries: &[Q],
  runs: usize,
  search: F,
) -> Measured<R>
where
  Q: Sync,
  R: Send + PartialEq,
  F: Fn(&Q) -> R + Sync,
{
  let n = queries.len().max(1) as f64;
  pools.all.install(|| {
    queries.par_iter().for_each(|q| {
      black_box(search(q));
    })
  });
  let mut best: Option<(f64, Vec<R>, Vec<u64>)> = None;
  for _ in 0..runs.max(1) {
    let (res, lat, secs) = pools.one.install(|| {
      let mut res = Vec::with_capacity(queries.len());
      let mut lat = Vec::with_capacity(queries.len());
      let t = Instant::now();
      for q in queries {
        let s = Instant::now();
        let r = search(q);
        lat.push(s.elapsed().as_nanos() as u64);
        res.push(r);
      }
      (res, lat, t.elapsed().as_secs_f64())
    });
    let qps = n / secs;
    if best.as_ref().is_none_or(|b| qps > b.0) {
      best = Some((qps, res, lat));
    }
  }
  let (qps_1t, results, mut lat) = best.expect("at least one run");
  let mut qps_nt = 0.0f64;
  let mut same_results = true;
  for _ in 0..runs.max(1) {
    let (res, secs) = pools.all.install(|| {
      let t = Instant::now();
      let res: Vec<R> = queries.par_iter().map(&search).collect();
      (res, t.elapsed().as_secs_f64())
    });
    qps_nt = qps_nt.max(n / secs);
    same_results &= res == results;
  }
  Measured {
    results,
    qps_1t,
    qps_nt,
    latency: summarize(&mut lat),
    same_results,
  }
}

/// Writes `value` as pretty JSON to `path`, creating parent directories.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
  if let Some(dir) = path.parent() {
    std::fs::create_dir_all(dir)?;
  }
  let mut w = BufWriter::new(File::create(path)?);
  serde_json::to_writer_pretty(&mut w, value)?;
  std::io::Write::write_all(&mut w, b"\n")?;
  Ok(())
}

/// Writes a file through `write`, flushes and fsyncs it. Returns the
/// seconds taken and the file size.
pub fn write_file<F>(path: &Path, write: F) -> Result<(f64, u64)>
where
  F: FnOnce(BufWriter<File>) -> muninn_core::Result<BufWriter<File>>,
{
  if let Some(dir) = path.parent() {
    std::fs::create_dir_all(dir)?;
  }
  let t = Instant::now();
  let w = write(BufWriter::with_capacity(1 << 20, File::create(path)?))?;
  let file = w.into_inner().map_err(|e| e.into_error())?;
  file.sync_all()?;
  let secs = t.elapsed().as_secs_f64();
  Ok((secs, std::fs::metadata(path)?.len()))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn measures_and_keeps_results_in_query_order() {
    let pools = Pools::new(2).unwrap();
    let qs: Vec<u32> = (0..100).collect();
    let m = measure(&pools, &qs, 2, |&q| q * 2);
    assert_eq!(m.results, (0..100).map(|q| q * 2).collect::<Vec<_>>());
    assert!(m.qps_1t > 0.0 && m.qps_nt > 0.0);
    assert!(m.same_results);
    assert!(m.latency.p99_us >= m.latency.p50_us);
  }

  #[test]
  fn one_thread_pass_runs_on_one_thread() {
    let pools = Pools::new(3).unwrap();
    assert_eq!(pools.threads, 3);
    let qs = [0u8; 8];
    let m = measure(&pools, &qs, 1, |_| rayon::current_num_threads());
    assert!(m.results.iter().all(|&t| t == 1));
  }
}
