//! Build-time robustness: large batches search with compact codes while
//! linking (see `HnswIndex::add_batch`); a few extreme vectors must not
//! degrade the graph for everyone else.

mod common;

use common::{Clusters, recall};
use muninn_core::Metric;
use muninn_vector::{
  FlatIndex, HnswIndex, HnswParams, SearchParams, VectorIndex,
};

const DIM: usize = 32;
const N: usize = 12_000;
const QUERIES: usize = 100;

fn pool() -> rayon::ThreadPool {
  rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()
    .unwrap()
}

/// Recall@10 at `ef` of `idx` against `flat`.
fn mean_recall(
  idx: &HnswIndex,
  flat: &FlatIndex,
  qs: &[f32],
  ef: usize,
) -> f64 {
  let sp = SearchParams::new(10).with_ef(ef);
  let mut r = 0.0;
  for q in qs.chunks_exact(DIM) {
    let t = flat.search(q, &sp, None).unwrap();
    r += recall(&idx.search(q, &sp, None).unwrap(), &t);
  }
  r / (qs.len() / DIM) as f64
}

/// One bulk batch (searched with codes while linking) against the same
/// rows added in batches too small for codes (full-precision linking), on
/// clustered data plus `outliers` rows at about `+-mag` in every
/// dimension. One
/// such row used to stretch the codes' per-dimension range until normal
/// rows fell onto one or two code levels; recall@10 at ef 32 then dropped
/// from 0.97 to 0.15 (L2) and 0.26 (Dot). Many outliers (2% of rows) must
/// not break it either.
#[test]
fn coded_build_recall_with_outliers() {
  let c = Clusters::new(DIM, 30, 0.4, 21);
  let queries = c.sample(QUERIES, 77);
  for (outliers, mag) in [(0usize, 0.0f32), (1, 1e3), (1, 1e4), (240, 1e4)] {
    let mut d = c.sample(N, 1);
    // Distinct outliers (sign pattern and size vary), so that recall is
    // not decided by how ties between identical vectors are broken.
    for o in 0..outliers {
      // Even rows: the code-quality check samples every other row here.
      let row = (o * 998 + 14) % N;
      let m = mag * (1.0 + o as f32 / 512.0);
      for j in 0..DIM {
        let up = (o * 31 + j * 7) % 5 < 2;
        d[row * DIM + j] = if up { m } else { -m };
      }
    }
    for metric in [Metric::L2, Metric::Dot] {
      let p = HnswParams::default();
      let bulk = pool()
        .install(|| HnswIndex::build(DIM, metric, p, &d))
        .unwrap();
      let mut chunked = HnswIndex::new(DIM, metric, p).unwrap();
      pool().install(|| {
        for ch in d.chunks(3000 * DIM) {
          chunked.add_batch(ch).unwrap();
        }
      });
      let flat = FlatIndex::from_vectors(DIM, metric, &d).unwrap();
      for ef in [32, 64] {
        let rb = mean_recall(&bulk, &flat, &queries, ef);
        let rc = mean_recall(&chunked, &flat, &queries, ef);
        println!(
          "{outliers} outliers x{mag} {metric:?} ef {ef}: bulk {rb:.4} \
           chunked {rc:.4}"
        );
        assert!(
          rb >= rc - 0.03,
          "{outliers} outliers x{mag} {metric:?} ef {ef}: bulk {rb:.4} \
           vs chunked {rc:.4}"
        );
      }
    }
  }
}

/// Two populations interleaved row by row (as alternating kinds of
/// documents arrive): code ranges used to be fitted, and checked, on every
/// `ceil(n / 8192)`-th row, so with an even stride one population was
/// never sampled, saturated to a single code and the check still passed
/// (recall@10 at ef 16 for its queries: 0.62 against 0.98).
#[test]
fn coded_build_sees_interleaved_populations() {
  // Above the size where every row is sampled, with an even stride
  // (`ceil(32768 / 8192) = 4`) for the old sampling to miss a population.
  let n = 32_768;
  let a = Clusters::new(DIM, 30, 0.4, 21);
  let b = Clusters::new(DIM, 30, 0.4, 22);
  let (sa, sb) = (a.sample(n / 2, 1), b.sample(n / 2, 2));
  let shift = |x: &f32| x * 20.0 + 50.0;
  let mut d = Vec::with_capacity(n * DIM);
  for i in 0..n / 2 {
    d.extend_from_slice(&sa[i * DIM..(i + 1) * DIM]);
    d.extend(sb[i * DIM..(i + 1) * DIM].iter().map(shift));
  }
  let qa = a.sample(QUERIES, 77);
  let qb: Vec<f32> = b.sample(QUERIES, 78).iter().map(shift).collect();
  let metric = Metric::L2;
  let p = HnswParams::default().with_ef_construction(100);
  let bulk = pool()
    .install(|| HnswIndex::build(DIM, metric, p, &d))
    .unwrap();
  let mut chunked = HnswIndex::new(DIM, metric, p).unwrap();
  pool().install(|| {
    for ch in d.chunks(3000 * DIM) {
      chunked.add_batch(ch).unwrap();
    }
  });
  let flat = FlatIndex::from_vectors(DIM, metric, &d).unwrap();
  for (name, qs) in [("first", &qa), ("second", &qb)] {
    for ef in [16, 32] {
      let rb = mean_recall(&bulk, &flat, qs, ef);
      let rc = mean_recall(&chunked, &flat, qs, ef);
      println!("{name} population ef {ef}: bulk {rb:.4} chunked {rc:.4}");
      assert!(rb >= rc - 0.03, "{name} ef {ef}: bulk {rb:.4} vs {rc:.4}");
    }
  }
}
