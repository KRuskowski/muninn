//! Build time, QPS and recall@10 for HNSW (full precision, SQ8, binary),
//! flat-scan throughput, and filtered search, on clustered synthetic data.
//!
//! ```text
//! cargo run --release -p muninn-vector --example vector_bench -- \
//!   [--n 200000] [--dim 128] [--queries 1000] [--clusters 256] \
//!   [--spread 0.6] [--metric l2] [--m 16] [--efc 200]
//! ```
//!
//! Threads come from rayon (`RAYON_NUM_THREADS`). Numbers from a shared
//! machine are indicative only; report the machine with them.

use std::time::{Duration, Instant};

use muninn_core::{AlignedBuf, BitSet, DocFilter, DocId, Hit, Metric, Region};
use muninn_vector::{
  FilterStrategy, FlatIndex, HnswIndex, HnswParams, Quantization, SearchParams,
  VectorIndex,
};
use rayon::prelude::*;

const K: usize = 10;

struct Args {
  n: usize,
  dim: usize,
  queries: usize,
  clusters: usize,
  spread: f32,
  metric: Metric,
  m: usize,
  efc: usize,
}

fn args() -> Args {
  let mut a = Args {
    n: 200_000,
    dim: 128,
    queries: 1000,
    clusters: 256,
    spread: 0.6,
    metric: Metric::L2,
    m: 16,
    efc: 200,
  };
  let v: Vec<String> = std::env::args().skip(1).collect();
  for pair in v.chunks(2) {
    let val = pair.get(1).map(String::as_str).unwrap_or("");
    let num = || val.parse::<usize>().expect("numeric argument");
    match pair[0].as_str() {
      "--n" => a.n = num(),
      "--dim" => a.dim = num(),
      "--queries" => a.queries = num(),
      "--clusters" => a.clusters = num(),
      "--spread" => a.spread = val.parse().expect("float"),
      "--metric" => a.metric = Metric::parse(val).expect("l2|dot|cosine"),
      "--m" => a.m = num(),
      "--efc" => a.efc = num(),
      other => panic!("unknown argument {other}"),
    }
  }
  a
}

struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  fn gauss(&mut self) -> f32 {
    let u = 1.0 - (self.next() >> 11) as f64 / (1u64 << 53) as f64;
    let v = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
  }
}

/// Gaussian clusters, generated in parallel (deterministic per row).
fn clustered(a: &Args, n: usize, seed: u64) -> Vec<f32> {
  let mut r = Rng(1);
  let centres: Vec<f32> = (0..a.clusters * a.dim).map(|_| r.gauss()).collect();
  let mut out = vec![0.0f32; n * a.dim];
  out.par_chunks_mut(a.dim).enumerate().for_each(|(i, row)| {
    let mut r = Rng(seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    let c = (r.next() % a.clusters as u64) as usize;
    let centre = &centres[c * a.dim..(c + 1) * a.dim];
    for (x, &m) in row.iter_mut().zip(centre) {
      *x = m + a.spread * r.gauss();
    }
  });
  out
}

fn recall(got: &[Vec<Hit>], truth: &[Vec<Hit>]) -> f64 {
  let mut s = 0.0;
  for (g, t) in got.iter().zip(truth) {
    if t.is_empty() {
      s += 1.0;
      continue;
    }
    let hit = t.iter().filter(|h| g.iter().any(|x| x.id == h.id)).count();
    s += hit as f64 / t.len() as f64;
  }
  s / truth.len() as f64
}

/// Runs every query on this thread, then all queries across the pool.
/// Returns (results, single-thread QPS, all-thread QPS).
fn run(
  idx: &dyn VectorIndex,
  queries: &[f32],
  dim: usize,
  p: SearchParams,
  f: Option<&dyn DocFilter>,
) -> (Vec<Vec<Hit>>, f64, f64) {
  let qs: Vec<&[f32]> = queries.chunks_exact(dim).collect();
  // Untimed warm-up so every configuration starts with the same cache
  // state (otherwise whichever runs first after a flat scan looks slow).
  qs.par_iter().for_each(|q| {
    idx.search(q, &p, f).unwrap();
  });
  let t = Instant::now();
  let res: Vec<Vec<Hit>> =
    qs.iter().map(|q| idx.search(q, &p, f).unwrap()).collect();
  let single = qs.len() as f64 / t.elapsed().as_secs_f64();
  let t = Instant::now();
  let par: Vec<Vec<Hit>> = qs
    .par_iter()
    .map(|q| idx.search(q, &p, f).unwrap())
    .collect();
  let multi = qs.len() as f64 / t.elapsed().as_secs_f64();
  assert_eq!(par, res, "results must not depend on threading");
  (res, single, multi)
}

fn secs(d: Duration) -> String {
  format!("{:.2} s", d.as_secs_f64())
}

fn main() {
  let a = args();
  let threads = rayon::current_num_threads();
  println!(
    "# muninn-vector bench: n={} dim={} queries={} clusters={} spread={} \
     metric={:?} M={} efC={} threads={} simd={}",
    a.n,
    a.dim,
    a.queries,
    a.clusters,
    a.spread,
    a.metric,
    a.m,
    a.efc,
    threads,
    muninn_simd::active_isa()
  );
  let t = Instant::now();
  let data = clustered(&a, a.n, 2);
  let queries = clustered(&a, a.queries, 3);
  println!("data generated in {}", secs(t.elapsed()));

  // Flat: the oracle, and its scan throughput.
  let flat = FlatIndex::from_vectors(a.dim, a.metric, &data).unwrap();
  let qs: Vec<&[f32]> = queries.chunks_exact(a.dim).collect();
  let p = SearchParams::new(K);
  let t = Instant::now();
  let truth: Vec<Vec<Hit>> = qs
    .par_iter()
    .map(|q| flat.search(q, &p, None).unwrap())
    .collect();
  let flat_multi = qs.len() as f64 / t.elapsed().as_secs_f64();
  let few = qs.len().min(50);
  let t = Instant::now();
  // One query at a time; each scan is itself split across the pool.
  for q in &qs[..few] {
    flat.search(q, &p, None).unwrap();
  }
  let flat_split = few as f64 / t.elapsed().as_secs_f64();
  let gb = (a.n * a.dim * 4) as f64 / 1e9;
  println!("\n## Flat scan (exact)\n");
  println!("| mode | QPS | scanned GB/s |\n|---|---|---|");
  println!(
    "| one query at a time, scan split over {threads} threads | {:.1} | \
     {:.1} |",
    flat_split,
    flat_split * gb
  );
  println!(
    "| {threads} queries in parallel | {:.1} | {:.1} |",
    flat_multi,
    flat_multi * gb
  );

  // HNSW build.
  let params = HnswParams::default()
    .with_m(a.m)
    .with_ef_construction(a.efc);
  let t = Instant::now();
  let mut hnsw = HnswIndex::build(a.dim, a.metric, params, &data).unwrap();
  let build = t.elapsed();
  let st = hnsw.stats();
  println!(
    "\n## HNSW build\n\n{} with {threads} threads ({:.0} vectors/s); \
     max level {}, avg level-0 degree {}, {} unreachable, graph {:.1} MB, \
     vectors {:.1} MB",
    secs(build),
    a.n as f64 / build.as_secs_f64(),
    st.max_level,
    st.avg_degree,
    st.unreachable,
    st.graph_bytes as f64 / 1e6,
    st.vector_bytes as f64 / 1e6
  );

  // Persistence: one streaming write, zero-copy open.
  let t = Instant::now();
  let buf = hnsw.write_to(AlignedBuf::new()).unwrap();
  let write = t.elapsed();
  let bytes = buf.len();
  let region = Region::new(buf);
  let t = Instant::now();
  let opened = HnswIndex::open(region).unwrap();
  let open = t.elapsed();
  assert!(opened.is_mapped());
  println!(
    "file {:.1} MB written in {:.0} ms, opened (zero-copy) in {:.1} us",
    bytes as f64 / 1e6,
    write.as_secs_f64() * 1e3,
    open.as_secs_f64() * 1e6
  );
  let (r0, _, _) = run(&opened, &queries, a.dim, p.with_ef(64), None);
  let (r1, _, _) = run(&hnsw, &queries, a.dim, p.with_ef(64), None);
  assert_eq!(r0, r1, "opened index must answer identically");
  drop(opened);

  for q in [Quantization::None, Quantization::Sq8, Quantization::Binary] {
    hnsw.set_quantization(q);
    let code_mb = hnsw.stats().code_bytes as f64 / 1e6;
    println!("\n## HNSW search, traversal {q:?} (codes {code_mb:.1} MB)\n");
    println!(
      "| ef | rerank | recall@10 | QPS 1 thread | QPS {threads} threads |\n\
       |---|---|---|---|---|"
    );
    for ef in [10, 16, 32, 64, 128, 256] {
      let reranks: &[usize] = match q {
        Quantization::None => &[0],
        _ => &[0, 8 * K],
      };
      for &rr in reranks {
        let sp = p.with_ef(ef).with_rerank(rr);
        let (res, single, multi) = run(&hnsw, &queries, a.dim, sp, None);
        let rr_shown = match (q, rr) {
          (Quantization::None, _) => "-".to_string(),
          (_, 0) => format!("{} (default)", q.default_rerank(K)),
          _ => rr.to_string(),
        };
        println!(
          "| {ef} | {rr_shown} | {:.4} | {single:.0} | {multi:.0} |",
          recall(&res, &truth)
        );
      }
    }
  }
  hnsw.set_quantization(Quantization::None);

  // Filtered search: automatic strategy, and graph-only vs exact-only to
  // show where the crossover lies.
  println!("\n## Filtered search (ef 64, filter = random BitSet)\n");
  println!("| pass | strategy | recall@10 | QPS 1 thread |\n|---|---|---|---|");
  let mut r = Rng(99);
  for fraction in [0.5, 0.1, 0.05, 0.02, 0.01, 0.001] {
    let mut set = BitSet::new(a.n);
    for id in 0..a.n {
      if ((r.next() >> 11) as f64 / (1u64 << 53) as f64) < fraction {
        set.insert(id as DocId);
      }
    }
    let ft: Vec<Vec<Hit>> = qs
      .par_iter()
      .map(|q| flat.search(q, &p, Some(&set)).unwrap())
      .collect();
    for strategy in [
      FilterStrategy::Auto,
      FilterStrategy::Graph,
      FilterStrategy::Exact,
    ] {
      hnsw.set_filter_strategy(strategy);
      let sp = p.with_ef(64);
      let (res, single, _) = run(&hnsw, &queries, a.dim, sp, Some(&set));
      println!(
        "| {:.1}% | {strategy:?} | {:.4} | {single:.0} |",
        fraction * 100.0,
        recall(&res, &ft)
      );
    }
    hnsw.set_filter_strategy(FilterStrategy::Auto);
    // A filter without a selectivity hint: Auto judges it in flight.
    let opaque = |id: DocId| set.contains(id);
    let sp = p.with_ef(64);
    let (res, single, _) = run(&hnsw, &queries, a.dim, sp, Some(&opaque));
    println!(
      "| {:.1}% | Auto, no hint | {:.4} | {single:.0} |",
      fraction * 100.0,
      recall(&res, &ft)
    );
  }
}
