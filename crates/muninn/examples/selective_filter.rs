//! Vector search through `Collection::search` with an id filter that
//! keeps only a small fraction of the documents: the case where scanning
//! the allowed documents beats walking the graph.
//!
//! ```text
//! cargo run --release -p muninn --example selective_filter -- \
//!   <dataset dir with base.fbin and query.fbin> [docs] [ef]
//! ```
//!
//! Loads the first `docs` vectors (default 1 000 000) into a collection,
//! compacts it into one segment, then for each filter fraction runs the
//! same 2000 queries on one thread and on all threads and prints the best
//! throughput of three rounds.

use std::path::{Path, PathBuf};
use std::time::Instant;

use muninn::{
  Database, Document, Durability, IdFilter, Metric, Options, Query, Schema,
  VectorField,
};
use rayon::prelude::*;

/// Reads a `.fbin` file: u32 rows, u32 dim, then row-major f32.
fn read_fbin(path: &Path, limit: usize) -> (usize, Vec<f32>) {
  let bytes = std::fs::read(path).expect("read fbin");
  let rows = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
  let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
  let rows = rows.min(limit);
  let v = bytes[8..8 + rows * dim * 4]
    .chunks_exact(4)
    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
    .collect();
  (dim, v)
}

fn qps(queries: &[&[f32]], all: bool, f: &(dyn Fn(&[f32]) + Sync)) -> f64 {
  let t = Instant::now();
  if all {
    queries.par_iter().for_each(|q| f(q));
  } else {
    queries.iter().for_each(|q| f(q));
  }
  queries.len() as f64 / t.elapsed().as_secs_f64()
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  let data = PathBuf::from(args.get(1).expect("dataset directory"));
  let n: usize = args.get(2).map_or(1_000_000, |s| s.parse().unwrap());
  let ef: usize = args.get(3).map_or(64, |s| s.parse().unwrap());
  let (dim, base) = read_fbin(&data.join("base.fbin"), n);
  let n = base.len() / dim;
  let (_, qv) = read_fbin(&data.join("query.fbin"), 2000);
  let queries: Vec<&[f32]> = qv.chunks_exact(dim).collect();
  let dir = PathBuf::from("target/selective-filter");
  let _ = std::fs::remove_dir_all(&dir);
  let db =
    Database::open(&dir, Options::default().durability(Durability::NoSync))
      .unwrap();
  let c = db
    .create_collection("c", Schema::vector(VectorField::hnsw(dim, Metric::L2)))
    .unwrap();
  let t = Instant::now();
  for (b, chunk) in base.chunks(1000 * dim).enumerate() {
    let docs: Vec<Document> = chunk
      .chunks_exact(dim)
      .enumerate()
      .map(|(i, v)| Document::new((b * 1000 + i) as u64).vector(v.to_vec()))
      .collect();
    c.upsert(&docs).unwrap();
  }
  c.compact().unwrap();
  println!(
    "{n} docs loaded and compacted in {:.1} s",
    t.elapsed().as_secs_f64()
  );
  for frac in [0.0005, 0.002, 0.008] {
    // Every `step`-th id, offset so it is not aligned with anything.
    let step = (1.0 / frac) as usize;
    let ids: Vec<u64> = (7..n as u64).step_by(step).collect();
    let filter = IdFilter::ids(ids.iter().copied());
    let search = |q: &[f32]| {
      let r = c
        .search(
          &Query::vector(q.to_vec())
            .k(10)
            .ef(ef)
            .id_filter(filter.clone()),
        )
        .unwrap();
      assert_eq!(r.len(), 10.min(ids.len()));
      std::hint::black_box(r);
    };
    let mut best = [0f64; 2];
    for _ in 0..3 {
      best[0] = best[0].max(qps(&queries, false, &search));
      best[1] = best[1].max(qps(&queries, true, &search));
    }
    println!(
      "filter keeps {} of {n} ({frac}): 1 thread {:.0} q/s, {} threads {:.0} \
       q/s",
      ids.len(),
      best[0],
      rayon::current_num_threads(),
      best[1]
    );
  }
  drop(c);
  drop(db);
  let _ = std::fs::remove_dir_all(&dir);
}
