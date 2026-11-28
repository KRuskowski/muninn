//! HNSW recall through the engine, on clustered data, against exact
//! brute force: across several segments plus a memtable, after a full
//! compaction into one segment (which extends the first segment's graph),
//! with SQ8 quantisation, and after a bulk load whose flushes leave their
//! graphs to background merges.

mod common;

use std::time::{Duration, Instant};

use common::{Rng, TestDir};
use muninn::{
  Collection, Database, Document, Durability, Id, Metric, Options,
  Quantization, Query, Schema, VectorField,
};

const DIM: usize = 32;
const N: usize = 24_000;
const CLUSTERS: usize = 40;
const QUERIES: usize = 200;
const K: usize = 10;

fn data(rng: &mut Rng) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
  let centres: Vec<Vec<f32>> = (0..CLUSTERS)
    .map(|_| (0..DIM).map(|_| rng.normal() * 4.0).collect())
    .collect();
  let point = |rng: &mut Rng| -> Vec<f32> {
    let c = &centres[rng.below(CLUSTERS as u64) as usize];
    c.iter().map(|x| x + rng.normal()).collect()
  };
  let docs = (0..N).map(|_| point(rng)).collect();
  let queries = (0..QUERIES).map(|_| point(rng)).collect();
  (docs, queries)
}

fn exact(docs: &[Vec<f32>], q: &[f32], metric: Metric) -> Vec<u64> {
  let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
  let mut d: Vec<(f32, u64)> = docs
    .iter()
    .enumerate()
    .map(|(i, v)| {
      let s = match metric {
        Metric::L2 => v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(),
        _ => {
          let dot: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
          -dot / (norm(v) * norm(q))
        }
      };
      (s, i as u64)
    })
    .collect();
  d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
  d.iter().take(K).map(|x| x.1).collect()
}

fn recall(
  c: &Collection,
  docs: &[Vec<f32>],
  qs: &[Vec<f32>],
  m: Metric,
) -> f64 {
  let mut found = 0;
  for q in qs {
    let want = exact(docs, q, m);
    let got: Vec<u64> = c
      .search(&Query::vector(q.clone()).k(K))
      .unwrap()
      .iter()
      .map(|h| match h.id {
        muninn::Id::U64(v) => v,
        _ => unreachable!(),
      })
      .collect();
    found += want.iter().filter(|w| got.contains(w)).count();
  }
  found as f64 / (qs.len() * K) as f64
}

fn run(metric: Metric, quant: Quantization) {
  let mut rng = Rng::new(42);
  let (docs, queries) = data(&mut rng);
  let dir = TestDir::new("recall");
  let opts = Options {
    durability: Durability::NoSync,
    memtable_max_docs: 7_000,
    hnsw_min_docs: 5_000,
    auto_compact: false,
    ..Options::default()
  };
  let db = Database::open(dir.path(), opts).unwrap();
  let field = VectorField {
    dim: DIM,
    metric,
    index: muninn::IndexKind::Hnsw {
      m: 16,
      ef_construction: 100,
    },
    quantization: quant,
  };
  let c = db.create_collection("r", Schema::vector(field)).unwrap();
  for (chunk_i, chunk) in docs.chunks(1000).enumerate() {
    let batch: Vec<Document> = chunk
      .iter()
      .enumerate()
      .map(|(j, v)| {
        Document::new((chunk_i * 1000 + j) as u64).vector(v.clone())
      })
      .collect();
    c.upsert(&batch).unwrap();
  }
  // Wait for the background flushes; the tail stays in the memtable.
  let st = loop {
    let st = c.stats().unwrap();
    if st.pending_flushes == 0 {
      break st;
    }
    std::thread::yield_now();
  };
  assert!(st.segments.len() >= 3, "{st:?}");
  assert!(
    st.segments.iter().all(|s| s.vector_index == "hnsw"),
    "{st:?}"
  );
  assert!(st.memtable_docs > 0, "{st:?}");
  let r = recall(&c, &docs, &queries, metric);
  println!("{metric:?} {quant:?} segments+memtable recall@{K} = {r:.4}");
  assert!(
    r >= 0.95,
    "recall {r} across segments ({metric:?} {quant:?})"
  );
  c.compact().unwrap();
  assert_eq!(c.stats().unwrap().segments.len(), 1);
  let r = recall(&c, &docs, &queries, metric);
  println!("{metric:?} {quant:?} one segment recall@{K} = {r:.4}");
  assert!(
    r >= 0.95,
    "recall {r} after compaction ({metric:?} {quant:?})"
  );
}

#[test]
fn hnsw_recall_l2() {
  run(Metric::L2, Quantization::None);
}

#[test]
fn hnsw_recall_cosine_sq8() {
  run(Metric::Cosine, Quantization::Sq8);
}

/// A bulk load: writes outpace the worker, so flushes leave their graphs
/// to merges, which insert those vectors into one growing graph; when the
/// writes stop, the rest is merged after the idle delay. Every document
/// keeps its id and vector, and recall holds before and after `compact`.
#[test]
fn bulk_load_builds_graphs_in_merges() {
  let mut rng = Rng::new(7);
  let (docs, queries) = data(&mut rng);
  let dir = TestDir::new("recall-bulk");
  let opts = Options {
    durability: Durability::NoSync,
    memtable_max_docs: 1_500,
    hnsw_min_docs: 1_000,
    merge_idle_delay: Duration::from_millis(50),
    ..Options::default()
  };
  let db = Database::open(dir.path(), opts).unwrap();
  let field = VectorField {
    dim: DIM,
    metric: Metric::L2,
    index: muninn::IndexKind::Hnsw {
      m: 16,
      ef_construction: 100,
    },
    quantization: Quantization::None,
  };
  let c = db.create_collection("r", Schema::vector(field)).unwrap();
  let mut flat_seen = false;
  for (chunk_i, chunk) in docs.chunks(500).enumerate() {
    let batch: Vec<Document> = chunk
      .iter()
      .enumerate()
      .map(|(j, v)| Document::new((chunk_i * 500 + j) as u64).vector(v.clone()))
      .collect();
    c.upsert(&batch).unwrap();
    let st = c.stats().unwrap();
    flat_seen |= st
      .segments
      .iter()
      .any(|s| s.vector_index == "flat" && s.docs >= 1_000);
  }
  assert!(flat_seen, "no flush left its graph to a merge");
  c.flush().unwrap();
  // The idle delay folds whatever is left into graphs.
  let t = Instant::now();
  let st = loop {
    let st = c.stats().unwrap();
    if st.segments.iter().all(|s| s.vector_index == "hnsw") {
      break st;
    }
    assert!(t.elapsed() < Duration::from_secs(60), "{st:?}");
    std::thread::sleep(Duration::from_millis(10));
  };
  assert_eq!(st.live_docs, N as u64, "{st:?}");
  assert!(st.segments.len() <= 4, "{st:?}");
  for (i, v) in docs.iter().enumerate().step_by(97) {
    let d = c.get(&Id::U64(i as u64)).unwrap().unwrap();
    assert_eq!(d.vector.as_deref(), Some(v.as_slice()), "doc {i}");
  }
  let r = recall(&c, &docs, &queries, Metric::L2);
  println!(
    "bulk load: {} segments, recall@{K} = {r:.4}",
    st.segments.len()
  );
  assert!(r >= 0.95, "recall {r} after a bulk load");
  c.compact().unwrap();
  let r = recall(&c, &docs, &queries, Metric::L2);
  println!("bulk load compacted: recall@{K} = {r:.4}");
  assert!(r >= 0.95, "recall {r} after compaction");
}
