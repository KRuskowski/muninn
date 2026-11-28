//! End-to-end engine benchmark: ingest, flush, compaction, and vector,
//! text and hybrid query latency and throughput, with recall@10.
//!
//! ```text
//! cargo run --release -p muninn --example engine_bench -- [dir]
//! ```
//!
//! Environment: `N` documents (default 500 000), `DIM` (128),
//! `QUERIES` (1000), `INGEST_N` docs per durability-mode run (100 000),
//! `THREADS` for the multi-threaded runs (4), `LABEL` for the machine.
//! Documents: clustered vectors, 8 to 12 words of text from a Zipf-like
//! 5000-word vocabulary, a small JSON payload.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use muninn::vector::{FlatIndex, SearchParams, VectorIndex};
use muninn::{
  Collection, Database, Document, Durability, IndexKind, Metric, Options,
  Query, Schema, VectorField,
};
use serde_json::json;

fn env(name: &str, default: usize) -> usize {
  std::env::var(name)
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(default)
}

struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
  }
  fn f32(&mut self) -> f32 {
    (self.next() >> 40) as f32 / (1u64 << 24) as f32
  }
  fn normal(&mut self) -> f32 {
    (0..6).map(|_| self.f32()).sum::<f32>() - 3.0
  }
  fn below(&mut self, n: usize) -> usize {
    (self.next() % n as u64) as usize
  }
  /// Zipf-like word index: small indexes are much more frequent.
  fn word(&mut self, vocab: usize) -> usize {
    let u = self.f32().max(1e-6);
    ((vocab as f32).powf(u) as usize - 1).min(vocab - 1)
  }
}

struct Data {
  dim: usize,
  vectors: Vec<f32>,
  texts: Vec<String>,
  queries: Vec<Vec<f32>>,
  text_queries: Vec<String>,
}

fn word(i: usize) -> String {
  // Pronounceable, distinct words.
  const S: [&str; 16] = [
    "ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "da", "fe", "go", "hu",
    "ji", "pe", "qo", "ze",
  ];
  let mut w = String::new();
  let mut x = i + 16;
  while x > 0 {
    w.push_str(S[x % 16]);
    x /= 16;
  }
  w
}

fn make_data(n: usize, dim: usize, nq: usize) -> Data {
  const CLUSTERS: usize = 256;
  const VOCAB: usize = 5000;
  let mut rng = Rng(7);
  let centres: Vec<Vec<f32>> = (0..CLUSTERS)
    .map(|_| (0..dim).map(|_| rng.normal()).collect())
    .collect();
  let point = |rng: &mut Rng, out: &mut Vec<f32>| {
    let c = &centres[rng.below(CLUSTERS)];
    out.extend(c.iter().map(|x| x + 0.35 * rng.normal()));
  };
  let mut vectors = Vec::with_capacity(n * dim);
  for _ in 0..n {
    point(&mut rng, &mut vectors);
  }
  let words: Vec<String> = (0..VOCAB).map(word).collect();
  let texts = (0..n)
    .map(|_| {
      let len = 8 + rng.below(5);
      (0..len)
        .map(|_| words[rng.word(VOCAB)].as_str())
        .collect::<Vec<_>>()
        .join(" ")
    })
    .collect();
  let queries = (0..nq)
    .map(|_| {
      let mut q = Vec::with_capacity(dim);
      point(&mut rng, &mut q);
      q
    })
    .collect();
  let text_queries = (0..nq)
    .map(|_| {
      let len = 2 + rng.below(2);
      (0..len)
        .map(|_| words[rng.word(VOCAB)].as_str())
        .collect::<Vec<_>>()
        .join(" ")
    })
    .collect();
  Data {
    dim,
    vectors,
    texts,
    queries,
    text_queries,
  }
}

fn doc(d: &Data, i: usize) -> Document {
  Document::new(i as u64)
    .vector(d.vectors[i * d.dim..(i + 1) * d.dim].to_vec())
    .text("body", d.texts[i].clone())
    .payload(json!({"i": i, "group": i % 100}))
}

fn schema(dim: usize) -> Schema {
  Schema::vector(VectorField {
    dim,
    metric: Metric::L2,
    index: IndexKind::Hnsw {
      m: 16,
      ef_construction: 100,
    },
    quantization: muninn::Quantization::None,
  })
  .with_text_fields(["body"])
}

fn fresh(dir: &Path, name: &str) -> PathBuf {
  let p = dir.join(name);
  let _ = std::fs::remove_dir_all(&p);
  p
}

/// Ingests docs `0..n` with `threads` writers in batches of `batch`.
/// Returns acknowledged docs/s.
fn ingest(
  c: &Collection,
  d: &Arc<Data>,
  n: usize,
  batch: usize,
  threads: usize,
) -> f64 {
  let next = Arc::new(AtomicUsize::new(0));
  let t0 = Instant::now();
  std::thread::scope(|s| {
    for _ in 0..threads {
      let (c, d, next) = (c.clone(), d.clone(), next.clone());
      s.spawn(move || {
        loop {
          let start = next.fetch_add(batch, Ordering::Relaxed);
          if start >= n {
            break;
          }
          let docs: Vec<Document> = (start..(start + batch).min(n))
            .map(|i| doc(&d, i))
            .collect();
          c.upsert(&docs).unwrap();
        }
      });
    }
  });
  n as f64 / t0.elapsed().as_secs_f64()
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
  let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
  sorted[i].as_secs_f64() * 1e3
}

/// Runs `q(i)` for every query index on `threads` threads; returns
/// (p50 ms, p99 ms, QPS).
fn run_queries<F>(n: usize, threads: usize, q: F) -> (f64, f64, f64)
where
  F: Fn(usize) + Sync,
{
  let next = AtomicUsize::new(0);
  let lat = std::sync::Mutex::new(Vec::with_capacity(n));
  let t0 = Instant::now();
  std::thread::scope(|s| {
    for _ in 0..threads {
      s.spawn(|| {
        let mut mine = Vec::new();
        loop {
          let i = next.fetch_add(1, Ordering::Relaxed);
          if i >= n {
            break;
          }
          let t = Instant::now();
          q(i);
          mine.push(t.elapsed());
        }
        lat.lock().unwrap().extend(mine);
      });
    }
  });
  let wall = t0.elapsed().as_secs_f64();
  let mut lat = lat.into_inner().unwrap();
  lat.sort();
  (pct(&lat, 0.5), pct(&lat, 0.99), n as f64 / wall)
}

fn main() {
  let n = env("N", 500_000);
  let dim = env("DIM", 128);
  let nq = env("QUERIES", 1000);
  let ingest_n = env("INGEST_N", 100_000).min(n);
  let threads = env("THREADS", 4);
  let label =
    std::env::var("LABEL").unwrap_or_else(|_| "shared VM, pinned cores".into());
  let dir = PathBuf::from(
    std::env::args()
      .nth(1)
      .unwrap_or_else(|| "target/engine-bench".into()),
  );
  std::fs::create_dir_all(&dir).unwrap();
  println!("# muninn engine benchmark");
  println!();
  println!(
    "{label}; N = {n}, dim = {dim}, {nq} queries, ISA {}, rayon threads {}",
    muninn::simd::active_isa(),
    rayon::current_num_threads()
  );
  let t = Instant::now();
  let data = Arc::new(make_data(n, dim, nq));
  eprintln!("data generated in {:.1}s", t.elapsed().as_secs_f64());

  // Ingest rate per durability mode.
  println!();
  println!("## Ingest ({ingest_n} docs per run, acknowledged docs/s)");
  println!();
  println!(
    "| durability | 1 writer, batch 1000 | {threads} writers, batch 10 |"
  );
  println!("|---|---|---|");
  let modes = [
    ("fsync each write", Durability::SyncEachWrite),
    (
      "fsync every 10 ms",
      Durability::SyncBatch {
        max_delay: Duration::from_millis(10),
      },
    ),
    ("no fsync", Durability::NoSync),
  ];
  for (name, mode) in modes {
    let mut rates = Vec::new();
    for (batch, th, count) in
      [(1000, 1, ingest_n), (10, threads, ingest_n.min(20_000))]
    {
      let p = fresh(&dir, "ingest");
      let opts = Options::default().durability(mode);
      let db = Database::open(&p, opts).unwrap();
      let c = db.create_collection("c", schema(dim)).unwrap();
      rates.push(ingest(&c, &data, count, batch, th));
      drop(c);
      drop(db);
      let _ = std::fs::remove_dir_all(&p);
    }
    println!("| {name} | {:.0} | {:.0} |", rates[0], rates[1]);
  }

  // Full load with background flushes and merges.
  let p = fresh(&dir, "main");
  let opts = Options {
    durability: Durability::SyncBatch {
      max_delay: Duration::from_millis(10),
    },
    memtable_max_docs: 100_000,
    ..Options::default()
  };
  let db = Database::open(&p, opts).unwrap();
  let c = db.create_collection("c", schema(dim)).unwrap();
  let rate = ingest(&c, &data, n, 1000, 1);
  let t = Instant::now();
  c.flush().unwrap();
  let flush_s = t.elapsed().as_secs_f64();
  let st = c.stats().unwrap();
  println!();
  println!("## Build");
  println!();
  println!("| step | result |");
  println!("|---|---|");
  println!(
    "| ingest {n} docs (fsync every 10 ms, background flush and merge) | {rate:.0} docs/s |"
  );
  println!("| flush() of the remainder | {flush_s:.2} s |");
  println!(
    "| segments after flush | {} ({}) |",
    st.segments.len(),
    st.segments
      .iter()
      .map(|s| format!("{} {}", s.docs, s.vector_index))
      .collect::<Vec<_>>()
      .join(", ")
  );

  // Ground truth.
  let t = Instant::now();
  let flat = FlatIndex::from_vectors(dim, Metric::L2, &data.vectors).unwrap();
  let truth: Vec<Vec<u64>> = data
    .queries
    .iter()
    .map(|q| {
      flat
        .search(q, &SearchParams::new(10), None)
        .unwrap()
        .iter()
        .map(|h| h.id as u64)
        .collect()
    })
    .collect();
  drop(flat);
  eprintln!("ground truth in {:.1}s", t.elapsed().as_secs_f64());

  let report = |c: &Collection, stage: &str| {
    println!();
    println!("## Queries: {stage}");
    println!();
    println!("| query | threads | p50 ms | p99 ms | QPS | recall@10 |");
    println!("|---|---|---|---|---|---|");
    for ef in [64usize, 128] {
      let mut found = 0usize;
      for (i, q) in data.queries.iter().enumerate() {
        let hits = c.search(&Query::vector(q.clone()).k(10).ef(ef)).unwrap();
        found += hits
          .iter()
          .filter(
            |h| matches!(h.id, muninn::Id::U64(v) if truth[i].contains(&v)),
          )
          .count();
      }
      let recall = found as f64 / (10 * nq) as f64;
      for th in [1, threads] {
        let (p50, p99, qps) = run_queries(nq, th, |i| {
          c.search(&Query::vector(data.queries[i].clone()).k(10).ef(ef))
            .unwrap();
        });
        println!(
          "| vector k=10 ef={ef} | {th} | {p50:.3} | {p99:.3} | {qps:.0} | {recall:.4} |"
        );
      }
    }
    for th in [1, threads] {
      let (p50, p99, qps) = run_queries(nq, th, |i| {
        c.search(&Query::text(data.text_queries[i].clone()).k(10))
          .unwrap();
      });
      println!("| text k=10 | {th} | {p50:.3} | {p99:.3} | {qps:.0} | - |");
    }
    for th in [1, threads] {
      let (p50, p99, qps) = run_queries(nq, th, |i| {
        c.search(
          &Query::hybrid(data.queries[i].clone(), data.text_queries[i].clone())
            .k(10),
        )
        .unwrap();
      });
      println!(
        "| hybrid k=10 (RRF, 40 per side) | {th} | {p50:.3} | {p99:.3} | {qps:.0} | - |"
      );
    }
  };
  report(&c, &format!("{} segments", st.segments.len()));

  let t = Instant::now();
  c.compact().unwrap();
  let compact_s = t.elapsed().as_secs_f64();
  println!();
  println!(
    "compact() into one segment of {n} docs (HNSW rebuild): {compact_s:.1} s"
  );
  report(&c, "1 segment after compact()");
  drop(c);
  drop(db);
  let t = Instant::now();
  let db = Database::open(&p, Options::default()).unwrap();
  let c = db.collection("c").unwrap();
  println!();
  println!(
    "reopen (mmap, nothing to replay): {:.3} s, {} live docs",
    t.elapsed().as_secs_f64(),
    c.stats().unwrap().live_docs
  );
  drop(c);
  drop(db);
  let _ = std::fs::remove_dir_all(&p);
}
