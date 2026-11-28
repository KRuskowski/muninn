//! HNSW correctness: recall against the exact `FlatIndex` on clustered
//! data for every metric, filtered search, persistence, incremental
//! inserts, and bad input.

mod common;

use std::sync::OnceLock;

use common::{Clusters, init_pool, random_set, recall};
use muninn_core::{AlignedBuf, DocFilter, DocId, Error, Metric, Region};
use muninn_vector::{
  FilterStrategy, FlatIndex, HnswIndex, HnswParams, SearchParams, VectorIndex,
};

const N: usize = 20_000;
const DIM: usize = 64;
const QUERIES: usize = 200;
const K: usize = 10;
/// The `ef` at which recall@10 is asserted.
const EF: usize = 64;
const MIN_RECALL: f64 = 0.95;

struct Fixture {
  queries: Vec<f32>,
  flat: FlatIndex,
  hnsw: HnswIndex,
}

fn clusters() -> Clusters {
  Clusters::new(DIM, 64, 0.35, 1)
}

fn build(metric: Metric) -> Fixture {
  init_pool();
  let c = clusters();
  let data = c.sample(N, 2);
  let queries = c.sample(QUERIES, 3);
  let flat = FlatIndex::from_vectors(DIM, metric, &data).unwrap();
  let hnsw =
    HnswIndex::build(DIM, metric, HnswParams::default(), &data).unwrap();
  Fixture {
    queries,
    flat,
    hnsw,
  }
}

fn fixture(metric: Metric) -> &'static Fixture {
  static L2: OnceLock<Fixture> = OnceLock::new();
  static DOT: OnceLock<Fixture> = OnceLock::new();
  static COS: OnceLock<Fixture> = OnceLock::new();
  match metric {
    Metric::L2 => L2.get_or_init(|| build(metric)),
    Metric::Dot => DOT.get_or_init(|| build(metric)),
    Metric::Cosine => COS.get_or_init(|| build(metric)),
  }
}

fn query(f: &Fixture, i: usize) -> &[f32] {
  &f.queries[i * DIM..(i + 1) * DIM]
}

/// Mean recall@k of `idx` against the flat oracle.
fn mean_recall(
  f: &Fixture,
  idx: &dyn VectorIndex,
  p: SearchParams,
  filter: Option<&dyn DocFilter>,
) -> f64 {
  let mut sum = 0.0;
  for i in 0..QUERIES {
    let truth = f.flat.search(query(f, i), &p, filter).unwrap();
    let got = idx.search(query(f, i), &p, filter).unwrap();
    sum += recall(&got, &truth);
  }
  sum / QUERIES as f64
}

fn check_recall(metric: Metric) {
  let f = fixture(metric);
  let mut line = format!("{metric:?} recall@{K}:");
  for ef in [16, 32, 64, 128, 256] {
    let r = mean_recall(f, &f.hnsw, SearchParams::new(K).with_ef(ef), None);
    line += &format!(" ef={ef}:{r:.4}");
    if ef == EF {
      assert!(r >= MIN_RECALL, "{metric:?} recall {r} at ef {ef}");
    }
  }
  // ef = 0 means the index default (64).
  let r = mean_recall(f, &f.hnsw, SearchParams::new(K), None);
  assert!(r >= MIN_RECALL);
  println!("{line} ({:?})", f.hnsw.stats());
}

#[test]
fn recall_l2() {
  check_recall(Metric::L2);
}

#[test]
fn recall_cosine() {
  check_recall(Metric::Cosine);
}

#[test]
fn recall_dot() {
  check_recall(Metric::Dot);
  // Inner-product pruning alone leaves ~7 edges per node; pruned
  // candidates top lists back up so the graph stays navigable.
  let st = fixture(Metric::Dot).hnsw.stats();
  assert!(st.avg_degree >= HnswParams::default().m, "{st:?}");
}

/// Every hit passes, exactly `min(k, passing)` come back, and recall
/// against the exact filtered result is reported (and asserted loosely).
fn check_filtered(metric: Metric, with_hint: bool, strategy: FilterStrategy) {
  let f = fixture(metric);
  let mut idx = f.hnsw.clone();
  idx.set_filter_strategy(strategy);
  let p = SearchParams::new(K).with_ef(EF);
  for (fraction, seed) in [(0.5, 10), (0.1, 11), (0.01, 12), (0.001, 13)] {
    let set = random_set(N, fraction, seed);
    let passing = set.count();
    let closure = |id: DocId| set.contains(id);
    let filter: &dyn DocFilter = if with_hint { &set } else { &closure };
    let mut sum = 0.0;
    for i in 0..QUERIES {
      let got = idx.search(query(f, i), &p, Some(filter)).unwrap();
      assert_eq!(got.len(), K.min(passing), "selectivity {fraction}");
      assert!(got.iter().all(|h| set.contains(h.id)));
      let truth = f.flat.search(query(f, i), &p, Some(filter)).unwrap();
      sum += recall(&got, &truth);
    }
    let r = sum / QUERIES as f64;
    println!(
      "{metric:?} filtered {:.1}% ({strategy:?}, hint {with_hint}, \
       {passing} pass): recall@{K} {r:.4}",
      fraction * 100.0
    );
    assert!(r >= 0.9, "filtered recall {r} at selectivity {fraction}");
  }
}

#[test]
fn filtered_search_l2_with_selectivity_hint() {
  check_filtered(Metric::L2, true, FilterStrategy::Auto);
}

#[test]
fn filtered_search_l2_without_hint() {
  check_filtered(Metric::L2, false, FilterStrategy::Auto);
}

#[test]
fn filtered_search_cosine_with_selectivity_hint() {
  check_filtered(Metric::Cosine, true, FilterStrategy::Auto);
}

/// Forces graph traversal at every selectivity, so the in-beam filter
/// (traverse failing nodes, collect only passing ones) is what is tested.
/// Dot product is left out: inner-product graphs leave some low-norm
/// vectors without incoming edges, so a forced graph walk cannot reach
/// every passing document under a 0.1% filter (`Auto` scans instead).
#[test]
fn filtered_search_graph_strategy() {
  check_filtered(Metric::L2, true, FilterStrategy::Graph);
  check_filtered(Metric::Cosine, false, FilterStrategy::Graph);
}

#[test]
fn filtered_search_exact_strategy() {
  check_filtered(Metric::Cosine, true, FilterStrategy::Exact);
}

#[test]
fn filter_rejecting_everything_returns_nothing() {
  let f = fixture(Metric::L2);
  let none = |_: DocId| false;
  let hits = f
    .hnsw
    .search(query(f, 0), &SearchParams::new(K), Some(&none))
    .unwrap();
  assert!(hits.is_empty());
}

#[test]
fn persistence_round_trip_is_zero_copy_and_identical() -> Result<(), Error> {
  for metric in [Metric::L2, Metric::Cosine] {
    let f = fixture(metric);
    let buf = f.hnsw.write_to(AlignedBuf::new()).unwrap();
    let region = Region::new(buf);
    let back = HnswIndex::open(region.clone()).unwrap();
    assert!(back.is_mapped());
    // The vectors are the file's bytes, not a copy.
    let v = back.vector(0).as_ptr() as usize;
    let r = region.bytes().as_ptr() as usize;
    assert!(v >= r && v < r + region.len());
    assert_eq!(back.len(), f.hnsw.len());
    assert_eq!(back.stats(), f.hnsw.stats());
    let set = random_set(N, 0.1, 20);
    let p = SearchParams::new(K);
    for i in 0..QUERIES {
      let q = query(f, i);
      assert_eq!(back.search(q, &p, None)?, f.hnsw.search(q, &p, None)?);
      assert_eq!(
        back.search(q, &p, Some(&set))?,
        f.hnsw.search(q, &p, Some(&set))?
      );
    }
  }
  Ok(())
}

#[test]
fn incremental_insert_after_load() {
  let f = fixture(Metric::L2);
  let buf = f.hnsw.write_to(AlignedBuf::new()).unwrap();
  let mut idx = HnswIndex::open(Region::new(buf)).unwrap();
  let mut flat = f.flat.clone();
  let extra = clusters().sample(1500, 4);
  for row in extra.chunks_exact(DIM).take(500) {
    let id = idx.insert(row).unwrap();
    assert_eq!(flat.add(row).unwrap(), id);
  }
  assert!(!idx.is_mapped());
  let more = idx.add_batch(&extra[500 * DIM..]).unwrap();
  assert_eq!(more, (N + 500) as DocId..(N + 1500) as DocId);
  flat.add_batch(&extra[500 * DIM..]).unwrap();
  // The new vectors themselves are found first.
  let p = SearchParams::new(K);
  for (i, row) in extra.chunks_exact(DIM).enumerate().step_by(37) {
    let hits = idx.search(row, &p, None).unwrap();
    assert_eq!(hits[0].id, (N + i) as DocId);
  }
  let mut sum = 0.0;
  for i in 0..QUERIES {
    let truth = flat.search(query(f, i), &p, None).unwrap();
    sum += recall(&idx.search(query(f, i), &p, None).unwrap(), &truth);
  }
  let r = sum / QUERIES as f64;
  println!("recall@{K} after 1500 inserts into a loaded index: {r:.4}");
  assert!(r >= MIN_RECALL);
}

/// Checks reachability (hard) and self-search misses at ef 64 (bounded);
/// prints the counts and, for each miss, whether a larger ef finds it.
fn check_reachable(idx: &HnswIndex, label: &str) {
  let st = idx.stats();
  let p = SearchParams::new(1).with_ef(EF);
  let mut misses = 0;
  for id in 0..idx.len() as DocId {
    let hits = idx.search(idx.vector(id), &p, None).unwrap();
    if hits[0].id != id {
      misses += 1;
      let f: Vec<bool> = [256, 1024]
        .iter()
        .map(|&ef| {
          let p = SearchParams::new(1).with_ef(ef);
          idx.search(idx.vector(id), &p, None).unwrap()[0].id == id
        })
        .collect();
      println!(
        "  {label} miss {id} top {} score {} found@256/1024 {f:?}",
        hits[0].id, hits[0].score
      );
    }
  }
  println!(
    "{label}: {} unreachable, {misses} self-query misses",
    st.unreachable
  );
  assert_eq!(st.unreachable, 0, "{label}: unreachable nodes");
  assert!(misses * 2000 <= idx.len(), "{label}: {misses} self misses");
}

/// Parallel `add_batch` on a fresh and on a reopened index, at several
/// thread counts, leaves no node unreachable from the entry point, and
/// practically every node is found by a search for its own vector.
///
/// Reachability is a hard invariant (orphans are re-linked). Self-search
/// at ef 64 is not guaranteed by any HNSW: a few nodes next to "hub"
/// nodes are only found at a larger ef (measured: at most 2 of 20-23k per
/// build), so that part is bounded at 0.05%.
#[test]
fn every_node_is_reachable_after_parallel_builds() {
  let c = clusters();
  let data = c.sample(N, 2);
  let extra = c.sample(3000, 4);
  for threads in [1, 2, 3, 4, 8] {
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(threads)
      .build()
      .unwrap();
    pool.install(|| {
      let p = HnswParams::default();
      let idx = HnswIndex::build(DIM, Metric::L2, p, &data).unwrap();
      check_reachable(&idx, &format!("fresh, {threads} threads"));
      let buf = idx.write_to(AlignedBuf::new()).unwrap();
      let mut back = HnswIndex::open(Region::new(buf)).unwrap();
      back.add_batch(&extra).unwrap();
      check_reachable(&back, &format!("reopened, {threads} threads"));
    });
  }
}

#[test]
fn serial_build_is_deterministic() {
  init_pool();
  let c = clusters();
  let data = c.sample(2000, 5);
  let params = HnswParams::default().with_ef_construction(64);
  let mut a = HnswIndex::new(DIM, Metric::L2, params).unwrap();
  let mut b = HnswIndex::new(DIM, Metric::L2, params).unwrap();
  for row in data.chunks_exact(DIM) {
    a.insert(row).unwrap();
    b.insert(row).unwrap();
  }
  let fa = a.write_to(Vec::new()).unwrap();
  let fb = b.write_to(Vec::new()).unwrap();
  assert_eq!(fa, fb);
}

#[test]
fn small_and_degenerate_indexes() {
  init_pool();
  let p = HnswParams::default();
  let mut idx = HnswIndex::new(3, Metric::L2, p).unwrap();
  assert!(idx.is_empty());
  let sp = SearchParams::new(5);
  assert!(idx.search(&[0.0, 0.0, 0.0], &sp, None).unwrap().is_empty());
  idx.insert(&[1.0, 0.0, 0.0]).unwrap();
  idx.insert(&[0.0, 1.0, 0.0]).unwrap();
  idx.insert(&[0.0, 0.0, 1.0]).unwrap();
  // k > n returns everything, best first.
  let hits = idx.search(&[0.9, 0.1, 0.0], &sp, None).unwrap();
  assert_eq!(hits.len(), 3);
  assert_eq!(hits[0].id, 0);
  assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
  // k = 0 returns nothing.
  let none = idx.search(&[1.0, 0.0, 0.0], &SearchParams::new(0), None);
  assert!(none.unwrap().is_empty());
  // Duplicates are fine.
  for _ in 0..100 {
    idx.insert(&[1.0, 0.0, 0.0]).unwrap();
  }
  assert_eq!(idx.search(&[1.0, 0.0, 0.0], &sp, None).unwrap().len(), 5);
  // An empty index round-trips.
  let empty = HnswIndex::new(3, Metric::Dot, p).unwrap();
  let back =
    HnswIndex::open(Region::new(empty.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  assert!(back.is_empty());
  assert!(back.search(&[1.0, 2.0, 3.0], &sp, None).unwrap().is_empty());
}

#[test]
fn bad_input_is_an_error() {
  let p = HnswParams::default();
  let mut idx = HnswIndex::new(4, Metric::Cosine, p).unwrap();
  assert!(matches!(
    idx.insert(&[1.0, 2.0]),
    Err(Error::DimensionMismatch {
      expected: 4,
      got: 2
    })
  ));
  assert!(idx.add_batch(&[1.0; 6]).is_err());
  assert!(idx.is_empty());
  idx.insert(&[1.0; 4]).unwrap();
  assert!(matches!(
    idx.search(&[1.0; 5], &SearchParams::new(1), None),
    Err(Error::DimensionMismatch {
      expected: 4,
      got: 5
    })
  ));
  assert!(HnswIndex::new(0, Metric::L2, p).is_err());
  assert!(HnswIndex::new(4, Metric::L2, p.with_m(1)).is_err());
  assert!(HnswIndex::new(4, Metric::L2, p.with_ef_construction(0)).is_err());
}

#[test]
fn corrupt_files_are_rejected_or_harmless() {
  init_pool();
  let c = clusters();
  let data = c.sample(1000, 6);
  let idx =
    HnswIndex::build(DIM, Metric::L2, HnswParams::default(), &data).unwrap();
  let buf = idx.write_to(AlignedBuf::new()).unwrap();
  // Wrong kind, truncation and bad meta are errors.
  let flat = FlatIndex::new(DIM, Metric::L2);
  let fbuf = flat.write_to(AlignedBuf::new()).unwrap();
  assert!(HnswIndex::open(Region::new(fbuf)).is_err());
  let short = Region::copy_from(&buf.as_slice()[..buf.len() / 2]);
  assert!(HnswIndex::open(short).is_err());
  let mut bad_meta = buf.clone();
  bad_meta.as_mut_slice()[64 + 3 * 8] = 0; // m = 0
  assert!(HnswIndex::open(Region::new(bad_meta)).is_err());
  // Garbage neighbour ids and counts in the graph never panic.
  let back = HnswIndex::open(Region::new(buf.clone())).unwrap();
  let l0_tag = muninn_core::container::tag("l0");
  let whole = Region::new(buf.clone());
  let c0 = muninn_core::container::Container::open(whole.clone()).unwrap();
  let l0 = c0.section(l0_tag).unwrap();
  let start = l0.bytes().as_ptr() as usize - whole.bytes().as_ptr() as usize;
  let mut bad = buf.clone();
  for (i, b) in bad.as_mut_slice()[start..start + l0.len()]
    .iter_mut()
    .enumerate()
  {
    if i % 7 == 0 {
      *b = 0xff;
    }
  }
  let broken = HnswIndex::open(Region::new(bad)).unwrap();
  for row in data.chunks_exact(DIM).take(20) {
    let p = SearchParams::new(K);
    let good = back.search(row, &p, None).unwrap();
    let hits = broken.search(row, &p, None).unwrap();
    assert_eq!(good.len(), K);
    assert!(hits.len() <= K);
    assert!(hits.iter().all(|h| (h.id as usize) < 1000));
  }
}
