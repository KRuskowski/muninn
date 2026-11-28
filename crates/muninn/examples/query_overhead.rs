//! What a vector search through `Collection::search` costs on top of
//! searching the segment's HNSW index directly.
//!
//! ```text
//! cargo run --release -p muninn --example query_overhead -- \
//!   <dataset dir with base.fbin and query.fbin> [docs] [ef]
//! ```
//!
//! Loads the first `docs` vectors (default 200 000) into a collection,
//! compacts it into one segment, then alternates rounds of the same
//! queries through the engine and through the segment's index file
//! opened on its own, on one thread and on all threads, and prints the
//! best throughput of each.

use std::path::{Path, PathBuf};
use std::time::Instant;

use muninn::store::map_file;
use muninn::vector::{HnswIndex, SearchParams, VectorIndex};
use muninn::{
  Database, Document, Durability, Metric, Options, Query, Schema, VectorField,
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

/// One way to run a query.
type Search<'a> = &'a (dyn Fn(&[f32]) + Sync);

fn qps(queries: &[&[f32]], all: bool, f: Search<'_>) -> f64 {
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
  let n: usize = args.get(2).map_or(200_000, |s| s.parse().unwrap());
  let ef: usize = args.get(3).map_or(64, |s| s.parse().unwrap());
  let (dim, base) = read_fbin(&data.join("base.fbin"), n);
  let (_, qv) = read_fbin(&data.join("query.fbin"), 2000);
  // Ten passes over the queries per round, for stable figures.
  let queries: Vec<&[f32]> =
    qv.chunks_exact(dim).cycle().take(20_000).collect();
  let dir = PathBuf::from("target/query-overhead");
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
    "{} docs loaded and compacted in {:.1} s",
    base.len() / dim,
    t.elapsed().as_secs_f64()
  );
  let coll_dir = dir.join("collections").join("c");
  let vec_file = std::fs::read_dir(&coll_dir)
    .unwrap()
    .map(|e| e.unwrap().path())
    .find(|p| p.extension().is_some_and(|e| e == "vec"))
    .expect("segment vector file");
  let raw = HnswIndex::open(map_file(&vec_file).unwrap()).unwrap();
  let params = SearchParams::new(10).with_ef(ef);
  let engine = |q: &[f32]| {
    std::hint::black_box(
      c.search(&Query::vector(q.to_vec()).k(10).ef(ef)).unwrap(),
    );
  };
  let direct = |q: &[f32]| {
    std::hint::black_box(raw.search(q, &params, None).unwrap());
  };
  let mut best = [0f64; 4];
  // Alternate the order so neither side always runs on a warmer cache.
  for round in 0..6 {
    let sides: [(usize, Search<'_>); 2] = if round % 2 == 0 {
      [(0, &direct), (1, &engine)]
    } else {
      [(1, &engine), (0, &direct)]
    };
    for (i, f) in sides {
      best[i] = best[i].max(qps(&queries, false, f));
      best[i + 2] = best[i + 2].max(qps(&queries, true, f));
    }
  }
  println!(
    "ef {ef}: 1 thread  index {:.0} q/s, engine {:.0} q/s ({:+.1}%)",
    best[0],
    best[1],
    (best[1] / best[0] - 1.0) * 100.0
  );
  println!(
    "ef {ef}: {} threads index {:.0} q/s, engine {:.0} q/s ({:+.1}%)",
    rayon::current_num_threads(),
    best[2],
    best[3],
    (best[3] / best[2] - 1.0) * 100.0
  );
  drop(c);
  drop(db);
  let _ = std::fs::remove_dir_all(&dir);
}
