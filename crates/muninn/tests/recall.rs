//! HNSW recall through the engine, on clustered data, against exact
//! brute force: across several segments plus a memtable, after a full
//! compaction into one segment, and with SQ8 quantisation.

mod common;

use common::{Rng, TestDir};
use muninn::{
  Collection, Database, Document, Durability, Metric, Options, Quantization,
  Query, Schema, VectorField,
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
