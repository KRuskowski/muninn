//! Quantised search (SQ8 and binary) with full-precision re-ranking, for
//! HNSW traversal and flat scans, and persistence of the codes.

mod common;

use common::{Clusters, init_pool, random_set, recall};
use muninn_core::{AlignedBuf, BitSet, DocId, Metric, Region};
use muninn_vector::{
  FlatIndex, HnswIndex, HnswParams, Quantization, SearchParams, VectorIndex,
};

const N: usize = 20_000;
const K: usize = 10;
const QUERIES: usize = 200;

struct Data {
  dim: usize,
  data: Vec<f32>,
  queries: Vec<f32>,
}

/// Tight clusters (spread 0.35 around N(0, 1) centres) suit SQ8 tests.
/// Binary codes keep one sign per dimension, which cannot separate points
/// inside a tight cluster, so binary tests use spread 1.0: neighbours
/// differ in many dimensions, as in real embeddings.
fn data_with(dim: usize, spread: f32) -> Data {
  init_pool();
  let c = Clusters::new(dim, 64, spread, 7);
  Data {
    dim,
    data: c.sample(N, 8),
    queries: c.sample(QUERIES, 9),
  }
}

fn data(dim: usize) -> Data {
  data_with(dim, 0.35)
}

fn mean_recall(
  d: &Data,
  oracle: &FlatIndex,
  idx: &dyn VectorIndex,
  p: SearchParams,
) -> f64 {
  let mut sum = 0.0;
  for q in d.queries.chunks_exact(d.dim) {
    let truth = oracle.search(q, &SearchParams::new(p.k), None).unwrap();
    sum += recall(&idx.search(q, &p, None).unwrap(), &truth);
  }
  sum / QUERIES as f64
}

fn check_hnsw(dim: usize, metric: Metric, q: Quantization, min: f64) {
  let spread = if q == Quantization::Binary { 1.0 } else { 0.35 };
  let d = data_with(dim, spread);
  let oracle = FlatIndex::from_vectors(dim, metric, &d.data).unwrap();
  let params = HnswParams::default().with_quantization(q);
  let idx = HnswIndex::build(dim, metric, params, &d.data).unwrap();
  let stats = idx.stats();
  assert!(stats.code_bytes > 0);
  let mut line =
    format!("hnsw {q:?} {metric:?} dim {dim} spread {spread} recall@{K}:");
  for rerank in [K, 2 * K, 4 * K, 8 * K, 16 * K] {
    let p = SearchParams::new(K).with_ef(64).with_rerank(rerank);
    line +=
      &format!(" rerank={rerank}:{:.4}", mean_recall(&d, &oracle, &idx, p));
  }
  // Default re-rank depth.
  let r = mean_recall(&d, &oracle, &idx, SearchParams::new(K).with_ef(64));
  println!("{line} default:{r:.4} codes {} B", stats.code_bytes);
  assert!(r >= min, "{q:?} {metric:?} recall {r}");

  // Codes persist and open in place; results are identical.
  let back =
    HnswIndex::open(Region::new(idx.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  assert!(back.is_mapped());
  let p = SearchParams::new(K);
  for qv in d.queries.chunks_exact(dim).take(50) {
    assert_eq!(
      back.search(qv, &p, None).unwrap(),
      idx.search(qv, &p, None).unwrap()
    );
  }
}

#[test]
fn hnsw_sq8_l2_recall_with_rerank() {
  check_hnsw(64, Metric::L2, Quantization::Sq8, 0.95);
}

#[test]
fn hnsw_sq8_cosine_recall_with_rerank() {
  check_hnsw(64, Metric::Cosine, Quantization::Sq8, 0.95);
}

#[test]
fn hnsw_sq8_dot_recall_with_rerank() {
  check_hnsw(64, Metric::Dot, Quantization::Sq8, 0.95);
}

#[test]
fn hnsw_binary_cosine_recall_with_rerank() {
  check_hnsw(128, Metric::Cosine, Quantization::Binary, 0.9);
}

#[test]
fn hnsw_binary_l2_recall_with_rerank() {
  check_hnsw(128, Metric::L2, Quantization::Binary, 0.9);
}

#[test]
fn quantised_hnsw_filtered_search_returns_only_passing() {
  let d = data(64);
  let params = HnswParams::default().with_quantization(Quantization::Sq8);
  let idx = HnswIndex::build(64, Metric::L2, params, &d.data).unwrap();
  let oracle = FlatIndex::from_vectors(64, Metric::L2, &d.data).unwrap();
  for fraction in [0.3, 0.05, 0.005] {
    let set = random_set(N, fraction, 31);
    let p = SearchParams::new(K);
    let mut sum = 0.0;
    for q in d.queries.chunks_exact(64) {
      let got = idx.search(q, &p, Some(&set)).unwrap();
      assert_eq!(got.len(), K.min(set.count()));
      assert!(got.iter().all(|h| set.contains(h.id)));
      sum += recall(&got, &oracle.search(q, &p, Some(&set)).unwrap());
    }
    let r = sum / QUERIES as f64;
    println!("hnsw sq8 filtered {fraction}: recall@{K} {r:.4}");
    assert!(r >= 0.9);
  }
}

fn check_flat(dim: usize, metric: Metric, q: Quantization, min: f64) {
  let spread = if q == Quantization::Binary { 1.0 } else { 0.35 };
  let d = data_with(dim, spread);
  let oracle = FlatIndex::from_vectors(dim, metric, &d.data).unwrap();
  let mut idx = oracle.clone();
  idx.quantize(q);
  assert_eq!(idx.quantization(), q);
  let r = mean_recall(&d, &oracle, &idx, SearchParams::new(K));
  println!(
    "flat {q:?} {metric:?} dim {dim} spread {spread}: recall@{K} {r:.4}"
  );
  assert!(r >= min, "flat {q:?} {metric:?} recall {r}");
  // Codes persist; the mapped index answers identically.
  let buf = idx.write_to(AlignedBuf::new()).unwrap();
  let back = FlatIndex::open(Region::new(buf)).unwrap();
  assert!(back.is_mapped());
  assert_eq!(back.quantization(), q);
  let odd = |id: DocId| id % 2 == 1;
  for qv in d.queries.chunks_exact(dim).take(50) {
    let p = SearchParams::new(K);
    assert_eq!(
      back.search(qv, &p, Some(&odd)).unwrap(),
      idx.search(qv, &p, Some(&odd)).unwrap()
    );
  }
}

#[test]
fn flat_sq8_recall_with_rerank() {
  check_flat(64, Metric::L2, Quantization::Sq8, 0.97);
  check_flat(64, Metric::Cosine, Quantization::Sq8, 0.97);
}

#[test]
fn flat_binary_recall_with_rerank() {
  check_flat(128, Metric::Cosine, Quantization::Binary, 0.9);
}

#[test]
fn quantised_index_grows_one_insert_at_a_time() {
  init_pool();
  let c = Clusters::new(32, 16, 0.35, 40);
  let data = c.sample(3000, 41);
  let params = HnswParams::default()
    .with_quantization(Quantization::Sq8)
    .with_ef_construction(64);
  let mut idx = HnswIndex::new(32, Metric::L2, params).unwrap();
  let mut flat = FlatIndex::new(32, Metric::L2);
  flat.quantize(Quantization::Sq8);
  for row in data.chunks_exact(32) {
    idx.insert(row).unwrap();
    flat.add(row).unwrap();
  }
  let oracle = FlatIndex::from_vectors(32, Metric::L2, &data).unwrap();
  let p = SearchParams::new(K);
  let (mut rh, mut rf) = (0.0, 0.0);
  let queries = c.sample(100, 42);
  for q in queries.chunks_exact(32) {
    let truth = oracle.search(q, &p, None).unwrap();
    rh += recall(&idx.search(q, &p, None).unwrap(), &truth);
    rf += recall(&flat.search(q, &p, None).unwrap(), &truth);
  }
  println!("grown sq8: hnsw {:.4} flat {:.4}", rh / 100.0, rf / 100.0);
  // Retraining as the index grows keeps codes useful.
  assert!(rh / 100.0 >= 0.95 && rf / 100.0 >= 0.97);
}

/// SQ8 search codes of a quantised index: one vector at +-1e4 among
/// clustered vectors used to stretch every dimension's range, so all the
/// others shared one or two code levels (recall@10 0.008 at ef 32..128,
/// against 1.0 at full precision). The ranges now come from clipped
/// quantiles and the outliers are scored at full precision: recall stays
/// within 0.03 of the same graph searched at full precision, with 2% of
/// outliers too, and a query next to an outlier finds it (at ef 256; at
/// ef 64 a far outlier is missed now and then at full precision too).
#[test]
fn sq8_index_keeps_recall_with_outliers() {
  let dim = 32;
  let n = 12_000;
  let c = common::Clusters::new(dim, 30, 0.4, 21);
  let qs = c.sample(100, 77);
  for outliers in [0usize, 1, 240] {
    let mut d = c.sample(n, 1);
    let mut rows = Vec::new();
    for o in 0..outliers {
      let row = (o * 997 + 14) % n;
      rows.push(row);
      for j in 0..dim {
        let up = (o * 31 + j * 7) % 5 < 2;
        let m = 1e4 * (1.0 + o as f32 / 512.0);
        d[row * dim + j] = if up { m } else { -m };
      }
    }
    for metric in [Metric::L2, Metric::Dot] {
      let p = HnswParams::default();
      let mut idx = HnswIndex::build(dim, metric, p, &d).unwrap();
      let full = idx.clone();
      idx.set_quantization(Quantization::Sq8);
      let exact = idx.stats().exact_rows;
      assert!(exact >= outliers && exact <= outliers + n / 100, "{exact}");
      let flat = FlatIndex::from_vectors(dim, metric, &d).unwrap();
      for ef in [32, 64] {
        let sp = SearchParams::new(10).with_ef(ef);
        let (mut rq, mut rf) = (0.0, 0.0);
        for q in qs.chunks_exact(dim) {
          let t = flat.search(q, &sp, None).unwrap();
          rq += common::recall(&idx.search(q, &sp, None).unwrap(), &t);
          rf += common::recall(&full.search(q, &sp, None).unwrap(), &t);
        }
        let (rq, rf) = (rq / 100.0, rf / 100.0);
        println!(
          "{outliers} outliers {metric:?} ef {ef}: sq8 {rq:.4} full {rf:.4}"
        );
        assert!(
          rq >= rf - 0.03,
          "{outliers} {metric:?} ef {ef}: {rq:.4} vs {rf:.4}"
        );
      }
      if metric == Metric::L2 {
        let sp = SearchParams::new(1).with_ef(256);
        for &row in rows.iter().take(40) {
          let q: Vec<f32> = d[row * dim..(row + 1) * dim]
            .iter()
            .map(|x| x + 0.5)
            .collect();
          let hit = idx.search(&q, &sp, None).unwrap();
          assert_eq!(hit[0].id as usize, row, "outlier {row} not found");
        }
      }
    }
  }
}

/// A selective filter with an id set on a quantised flat index is
/// answered by scoring those ids at full precision: exact, and cheaper
/// than scanning every code. Before, binary codes found 0.964 of the true
/// ten nearest of 600 passing documents.
#[test]
fn quantised_flat_scores_selective_filters_exactly() {
  let (n, dim) = (20_000, 16);
  // Unclustered data, where binary codes rank poorly.
  let mut g = common::Rng::new(5);
  let d: Vec<f32> = (0..n * dim).map(|_| g.gauss()).collect();
  let qs: Vec<f32> = (0..30 * dim).map(|_| g.gauss()).collect();
  let exact = FlatIndex::from_vectors(dim, Metric::L2, &d).unwrap();
  let mut set = BitSet::new(n);
  let mut rng = common::Rng::new(4);
  for _ in 0..600 {
    set.insert((rng.next_u64() % n as u64) as DocId);
  }
  for q in [Quantization::Sq8, Quantization::Binary] {
    let mut f = FlatIndex::from_vectors(dim, Metric::L2, &d).unwrap();
    f.quantize(q);
    for qv in qs.chunks_exact(dim) {
      let sp = SearchParams::new(10);
      let want = exact.search(qv, &sp, Some(&set)).unwrap();
      assert_eq!(f.search(qv, &sp, Some(&set)).unwrap(), want, "{q:?}");
    }
  }
}

/// The section table of a container file: (tag, entry offset, section
/// offset, section length).
fn sections(b: &[u8]) -> Vec<([u8; 8], usize, usize, usize)> {
  let f = b.len() - 64;
  let toff = u64::from_le_bytes(b[f..f + 8].try_into().unwrap()) as usize;
  let cnt = u32::from_le_bytes(b[f + 8..f + 12].try_into().unwrap()) as usize;
  (0..cnt)
    .map(|i| {
      let e = toff + i * 24;
      let tag: [u8; 8] = b[e..e + 8].try_into().unwrap();
      let off = u64::from_le_bytes(b[e + 8..e + 16].try_into().unwrap());
      let len = u64::from_le_bytes(b[e + 16..e + 24].try_into().unwrap());
      (tag, e, off as usize, len as usize)
    })
    .collect()
}

fn find(
  t: &[([u8; 8], usize, usize, usize)],
  s: &str,
) -> (usize, usize, usize) {
  let e = t
    .iter()
    .find(|e| e.0 == muninn_core::container::tag(s))
    .unwrap();
  (e.1, e.2, e.3)
}

/// An SQ8 index file as written before ranges were fitted (format v2,
/// min/max ranges, no `qexact`/`qcoarse`), holding one vector at +-1e4:
/// its collapsed codes used to be trusted for good (recall@10 0.01, and
/// a merge-like `add_batch` built a damaged graph on them: 0.84 at full
/// precision). Opening now checks the ranges against the vectors and
/// refits them.
#[test]
fn old_min_max_sq8_file_is_refitted_on_open() {
  let dim = 32;
  let n = 12_000;
  let c = Clusters::new(dim, 30, 0.4, 21);
  let mut d = c.sample(n, 1);
  for j in 0..dim {
    d[17 * dim + j] = if j % 2 == 0 { 1e4 } else { -1e4 };
  }
  let p = HnswParams::default().with_quantization(Quantization::Sq8);
  let idx = HnswIndex::build(dim, Metric::L2, p, &d).unwrap();
  let buf = idx.write_to(AlignedBuf::new()).unwrap();
  let t = sections(buf.as_slice());
  let (_, voff, vlen) = find(&t, "vectors");
  let vecs: Vec<f32> = buf.as_slice()[voff..voff + vlen]
    .chunks_exact(4)
    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
    .collect();
  // Main's min/max codes over the stored (internal-order) vectors.
  let mut lo = vec![f32::INFINITY; dim];
  let mut hi = vec![f32::NEG_INFINITY; dim];
  for row in vecs.chunks_exact(dim) {
    for j in 0..dim {
      lo[j] = lo[j].min(row[j]);
      hi[j] = hi[j].max(row[j]);
    }
  }
  let step: Vec<f32> = lo
    .iter()
    .zip(&hi)
    .map(|(&l, &h)| if h > l { h / 255.0 - l / 255.0 } else { 0.0 })
    .collect();
  let (mut codes, mut sq) = (Vec::new(), Vec::new());
  for row in vecs.chunks_exact(dim) {
    let mut acc = 0.0f32;
    for j in 0..dim {
      let c = if step[j] > 0.0 {
        ((row[j] - lo[j]) / step[j]).round().clamp(0.0, 255.0) as u8
      } else {
        0
      };
      let v = step[j] * c as f32;
      acc += v * v;
      codes.push(c);
    }
    sq.push(acc);
  }
  let mut b = buf.clone();
  let f2b =
    |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
  for (name, bytes) in [
    ("qmin", f2b(&lo)),
    ("qstep", f2b(&step)),
    ("qcodes", codes),
    ("qsq", f2b(&sq)),
  ] {
    let (_, off, len) = find(&t, name);
    assert_eq!(len, bytes.len(), "{name}");
    b.as_mut_slice()[off..off + len].copy_from_slice(&bytes);
  }
  b.as_mut_slice()[16..20].copy_from_slice(&2u32.to_le_bytes());
  for (name, gone) in [("qexact", "zzold1"), ("qcoarse", "zzold2")] {
    let (e, _, _) = find(&t, name);
    b.as_mut_slice()[e..e + 8]
      .copy_from_slice(&muninn_core::container::tag(gone));
  }
  let old = HnswIndex::open(Region::new(b)).unwrap();
  let flat = FlatIndex::from_vectors(dim, Metric::L2, &d).unwrap();
  let qs = c.sample(100, 77);
  let rec = |ix: &HnswIndex, fl: &FlatIndex| {
    let sp = SearchParams::new(10).with_ef(64);
    let mut r = 0.0;
    for q in qs.chunks_exact(dim) {
      let t = fl.search(q, &sp, None).unwrap();
      r += recall(&ix.search(q, &sp, None).unwrap(), &t);
    }
    r / 100.0
  };
  let (rn, ro) = (rec(&idx, &flat), rec(&old, &flat));
  println!("recall new file {rn:.4}, old file {ro:.4}");
  assert!(ro >= rn - 0.03, "old file {ro:.4} vs {rn:.4}");
  assert_eq!(old.stats().exact_rows, 1);
  // A merge-like extension builds a sound graph.
  let more = c.sample(5000, 2);
  let mut ext = old.clone();
  ext.add_batch(&more).unwrap();
  ext.set_quantization(Quantization::None);
  let all: Vec<f32> = d.iter().chain(&more).copied().collect();
  let flat2 = FlatIndex::from_vectors(dim, Metric::L2, &all).unwrap();
  let r = rec(&ext, &flat2);
  assert!(r >= 0.97, "extended graph at full precision {r:.4}");
}

/// Exact-row and coarse sections are fully validated at open: a bit past
/// the last vector, a coarse flag other than 0 or 1, and sections of the
/// wrong length are corrupt files, not silently accepted.
#[test]
fn bad_sq8_sections_are_rejected() {
  let dim = 8;
  let n = 5000 + 7;
  let mut d = Clusters::new(dim, 10, 0.4, 3).sample(n, 1);
  for j in 0..dim {
    d[17 * dim + j] = 1e4;
  }
  let p = HnswParams::default().with_quantization(Quantization::Sq8);
  let idx = HnswIndex::build(dim, Metric::L2, p, &d).unwrap();
  assert_eq!(idx.stats().exact_rows, 1);
  let buf = idx.write_to(AlignedBuf::new()).unwrap();
  assert!(HnswIndex::open(Region::new(buf.clone())).is_ok());
  let t = sections(buf.as_slice());
  let (xe, xoff, xlen) = find(&t, "qexact");
  let (ce, coff, _) = find(&t, "qcoarse");
  let mut cases: Vec<(&str, AlignedBuf)> = Vec::new();
  let mut b = buf.clone();
  let last = xoff + xlen - 8;
  let w = u64::from_le_bytes(b.as_slice()[last..last + 8].try_into().unwrap());
  b.as_mut_slice()[last..last + 8]
    .copy_from_slice(&(w | 1 << 63).to_le_bytes());
  cases.push(("bit past n", b));
  let mut b = buf.clone();
  b.as_mut_slice()[coff..coff + 8].copy_from_slice(&2u64.to_le_bytes());
  cases.push(("coarse = 2", b));
  let mut b = buf.clone();
  b.as_mut_slice()[ce + 16..ce + 24].copy_from_slice(&0u64.to_le_bytes());
  cases.push(("coarse empty", b));
  for delta in [8u64, 4, 1] {
    let mut b = buf.clone();
    b.as_mut_slice()[xe + 16..xe + 24]
      .copy_from_slice(&(xlen as u64 - delta).to_le_bytes());
    cases.push(("exact too short", b));
  }
  for (what, b) in cases {
    assert!(HnswIndex::open(Region::new(b)).is_err(), "{what}: opened");
  }
}

/// Sparse data (each value nonzero with probability 10% or 20%, so every
/// dimension's interquartile range is 0) keeps its SQ8 codes: few vectors
/// at full precision, recall within 0.03 of full precision.
#[test]
fn sparse_data_keeps_sq8_codes() {
  let (n, dim) = (12_000, 32);
  for dens in [0.1f64, 0.2] {
    let mut g = common::Rng::new(4);
    let mut sparse = |m: usize| -> Vec<f32> {
      (0..m * dim)
        .map(|_| {
          if g.uniform() < dens {
            g.uniform() as f32
          } else {
            0.0
          }
        })
        .collect()
    };
    let d = sparse(n);
    let qs = sparse(100);
    let full =
      HnswIndex::build(dim, Metric::L2, HnswParams::default(), &d).unwrap();
    let mut idx = full.clone();
    idx.set_quantization(Quantization::Sq8);
    assert!(
      idx.stats().exact_rows < n / 100,
      "{}",
      idx.stats().exact_rows
    );
    let flat = FlatIndex::from_vectors(dim, Metric::L2, &d).unwrap();
    let sp = SearchParams::new(10).with_ef(64);
    let (mut rq, mut rf) = (0.0, 0.0);
    for q in qs.chunks_exact(dim) {
      let t = flat.search(q, &sp, None).unwrap();
      rq += recall(&idx.search(q, &sp, None).unwrap(), &t);
      rf += recall(&full.search(q, &sp, None).unwrap(), &t);
    }
    let (rq, rf) = (rq / 100.0, rf / 100.0);
    println!("density {dens}: sq8 {rq:.4} full {rf:.4}");
    assert!(rq >= rf - 0.03, "density {dens}: {rq:.4} vs {rf:.4}");
  }
}
