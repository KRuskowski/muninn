//! FlatIndex: the parallel blocked scan equals a naive reference, at a
//! size that takes the multi-threaded path, and persistence round-trips.

mod common;

use common::{Clusters, init_pool, random_set};
use muninn_core::{AlignedBuf, DocFilter, DocId, Error, Metric, Region};
use muninn_vector::{FlatIndex, SearchParams, VectorIndex};

const DIM: usize = 96;

fn naive(
  metric: Metric,
  data: &[f32],
  q: &[f32],
  k: usize,
  f: Option<&dyn DocFilter>,
) -> Vec<DocId> {
  let mut q = q.to_vec();
  if metric.normalizes() {
    muninn_simd::normalize(&mut q);
  }
  let mut all: Vec<(f32, DocId)> = data
    .chunks_exact(DIM)
    .enumerate()
    .filter(|(i, _)| f.is_none_or(|f| f.matches(*i as DocId)))
    .map(|(i, row)| {
      let mut r = row.to_vec();
      if metric.normalizes() {
        muninn_simd::normalize(&mut r);
      }
      (muninn_simd::distance(metric, &q, &r), i as DocId)
    })
    .collect();
  all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
  all.into_iter().take(k).map(|x| x.1).collect()
}

#[test]
fn parallel_scan_matches_naive_reference() {
  init_pool();
  let c = Clusters::new(DIM, 32, 0.4, 50);
  let data = c.sample(30_000, 51);
  let queries = c.sample(10, 52);
  let set = random_set(30_000, 0.03, 53);
  for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
    let idx = FlatIndex::from_vectors(DIM, metric, &data).unwrap();
    assert_eq!(idx.len(), 30_000);
    for q in queries.chunks_exact(DIM) {
      for k in [1, 10, 100] {
        let filters: [Option<&dyn DocFilter>; 2] = [None, Some(&set)];
        for f in filters {
          let got: Vec<DocId> = idx
            .search(q, &SearchParams::new(k), f)
            .unwrap()
            .iter()
            .map(|h| h.id)
            .collect();
          assert_eq!(got, naive(metric, &data, q, k, f), "{metric:?} k={k}");
        }
      }
    }
  }
}

#[test]
fn edge_cases() {
  let mut idx = FlatIndex::new(DIM, Metric::L2);
  let q = vec![0.5; DIM];
  assert!(
    idx
      .search(&q, &SearchParams::new(3), None)
      .unwrap()
      .is_empty()
  );
  idx.add(&vec![1.0; DIM]).unwrap();
  idx.add(&vec![0.0; DIM]).unwrap();
  assert_eq!(
    idx.search(&q, &SearchParams::new(9), None).unwrap().len(),
    2
  );
  assert!(
    idx
      .search(&q, &SearchParams::new(0), None)
      .unwrap()
      .is_empty()
  );
  assert!(matches!(
    idx.search(&q[1..], &SearchParams::new(1), None),
    Err(Error::DimensionMismatch { .. })
  ));
}

#[test]
fn round_trip_large() -> Result<(), Error> {
  init_pool();
  let c = Clusters::new(DIM, 8, 0.4, 60);
  let data = c.sample(5000, 61);
  let idx = FlatIndex::from_vectors(DIM, Metric::Cosine, &data)?;
  let region = Region::new(idx.write_to(AlignedBuf::new())?);
  let back = FlatIndex::open(region.clone())?;
  assert!(back.is_mapped());
  assert_eq!(back.vectors(), idx.vectors());
  let p = region.bytes().as_ptr() as usize;
  let v = back.vectors().as_ptr() as usize;
  assert!(v > p && v < p + region.len(), "vectors must not be copied");
  for q in c.sample(20, 62).chunks_exact(DIM) {
    let s = SearchParams::new(10);
    assert_eq!(back.search(q, &s, None)?, idx.search(q, &s, None)?);
  }
  Ok(())
}
