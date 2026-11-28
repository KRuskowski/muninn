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
//! best throughput of each. With `FRESH=1` it also builds a graph over
//! the same vectors in one go and compares it with the one the engine
//! built (through flushes and merges): recall@10 against exact search
//! and single-thread throughput.

use std::path::{Path, PathBuf};
use std::time::Instant;

use muninn::store::map_file;
use muninn::vector::{
  FlatIndex, HnswIndex, HnswParams, SearchParams, VectorIndex,
};
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
  if std::env::var_os("FRESH").is_some() {
    compare_fresh(dim, &base, &queries[..1000], &raw, &params);
  }
  drop(c);
  drop(db);
  let _ = std::fs::remove_dir_all(&dir);
}

/// Recall@10 and single-thread throughput of the engine's graph against
/// one built over the same vectors in a single batch.
fn compare_fresh(
  dim: usize,
  base: &[f32],
  queries: &[&[f32]],
  engine_graph: &HnswIndex,
  params: &SearchParams,
) {
  let t = Instant::now();
  let fresh = HnswIndex::build(
    dim,
    Metric::L2,
    HnswParams::default().with_m(16).with_ef_construction(200),
    base,
  )
  .unwrap();
  println!("fresh graph built in {:.1} s", t.elapsed().as_secs_f64());
  let flat = FlatIndex::from_vectors(dim, Metric::L2, base).unwrap();
  let exact: Vec<Vec<u32>> = queries
    .iter()
    .map(|q| {
      let h = flat.search(q, &SearchParams::new(10), None).unwrap();
      h.iter().map(|x| x.id).collect()
    })
    .collect();
  let recall = |g: &HnswIndex| {
    let found: usize = queries
      .iter()
      .zip(&exact)
      .map(|(q, e)| {
        let h = g.search(q, params, None).unwrap();
        h.iter().filter(|x| e.contains(&x.id)).count()
      })
      .sum();
    found as f64 / (queries.len() * 10) as f64
  };
  let a = |q: &[f32]| {
    std::hint::black_box(engine_graph.search(q, params, None).unwrap());
  };
  let b = |q: &[f32]| {
    std::hint::black_box(fresh.search(q, params, None).unwrap());
  };
  let mut best = [0f64; 2];
  for _ in 0..6 {
    best[0] = best[0].max(qps(queries, false, &a));
    best[1] = best[1].max(qps(queries, false, &b));
  }
  println!(
    "engine-built graph: recall@10 {:.4}, {:.0} q/s; fresh graph: recall@10 \
     {:.4}, {:.0} q/s",
    recall(engine_graph),
    best[0],
    recall(&fresh),
    best[1]
  );
}
