//! Regression tests from the independent review of PR #15. Each one
//! failed before its fix.

use muninn_core::container::{Container, tag};
use muninn_core::{AlignedBuf, BitSet, DocFilter, DocId, Metric, Region};
use muninn_vector::{
  FlatIndex, HnswIndex, HnswParams, Quantization, SearchParams, VectorIndex,
};

fn data(n: usize, dim: usize) -> Vec<f32> {
  let mut s = 12345u64;
  (0..n * dim)
    .map(|_| {
      s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    })
    .collect()
}

/// Byte range of section `t` inside `buf`.
fn section_range(buf: &AlignedBuf, t: &str) -> std::ops::Range<usize> {
  let whole = Region::new(buf.clone());
  let c = Container::open(whole.clone()).unwrap();
  let s = c.section(tag(t)).unwrap();
  let start = s.bytes().as_ptr() as usize - whole.bytes().as_ptr() as usize;
  start..start + s.len()
}

/// Rewrites every little-endian `u32` in `bytes` with `f(index, value)`.
fn edit_u32s(bytes: &mut [u8], mut f: impl FnMut(usize, u32) -> u32) {
  for (i, w) in bytes.chunks_exact_mut(4).enumerate() {
    let v = u32::from_le_bytes(w.try_into().unwrap());
    w.copy_from_slice(&f(i, v).to_le_bytes());
  }
}

fn no_panic(what: &str, f: impl FnOnce()) {
  let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
  assert!(res.is_ok(), "{what} panicked");
}

/// Review item 1 (r1): an empty index built with quantisation must
/// reopen.
#[test]
fn empty_quantised_hnsw_round_trips() {
  for q in [Quantization::Sq8, Quantization::Binary] {
    let p = HnswParams::default().with_quantization(q);
    let idx = HnswIndex::new(4, Metric::L2, p).unwrap();
    let buf = idx.write_to(AlignedBuf::new()).unwrap();
    let mut back = HnswIndex::open(Region::new(buf)).unwrap();
    assert_eq!(back.params().quantization, q);
    back.insert(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let hits = back.search(&[1.0; 4], &SearchParams::new(1), None).unwrap();
    assert_eq!(hits.len(), 1);
  }
}

/// Review item 2 (r2): garbage level-0 neighbour ids in a file must not
/// make a later insert panic.
#[test]
fn insert_after_garbage_level0_ids() {
  let dim = 8;
  let d = data(500, dim);
  let p = HnswParams::default().with_m(4);
  let idx = HnswIndex::build(dim, Metric::L2, p, &d).unwrap();
  let mut buf = idx.write_to(AlignedBuf::new()).unwrap();
  let r = section_range(&buf, "l0");
  let m0 = 8;
  edit_u32s(&mut buf.as_mut_slice()[r], |i, _| {
    if i % (1 + m0) == 0 {
      m0 as u32
    } else {
      0x7fff_0000
    }
  });
  let mut back = HnswIndex::open(Region::new(buf)).unwrap();
  let _ = back.search(&d[..dim], &SearchParams::new(5), None).unwrap();
  no_panic("insert into opened corrupt graph", || {
    for row in d.chunks_exact(dim).take(50) {
      back.insert(row).unwrap();
    }
  });
  let hits = back.search(&d[..dim], &SearchParams::new(5), None).unwrap();
  assert_eq!(hits.len(), 5);
}

/// Review item 2 (r3): upper-level offsets pointing past the array must
/// not make a later insert panic.
#[test]
fn insert_after_bad_upper_offsets() {
  let dim = 8;
  let d = data(500, dim);
  let p = HnswParams::default().with_m(4);
  let idx = HnswIndex::build(dim, Metric::L2, p, &d).unwrap();
  let mut buf = idx.write_to(AlignedBuf::new()).unwrap();
  let r = section_range(&buf, "upoff");
  edit_u32s(&mut buf.as_mut_slice()[r], |_, o| {
    if o != u32::MAX { 0x0fff_ffff } else { o }
  });
  let mut back = HnswIndex::open(Region::new(buf)).unwrap();
  let h = back.search(&d[..dim], &SearchParams::new(5), None).unwrap();
  assert_eq!(h.len(), 5);
  no_panic("insert after bad upoff", || {
    back.insert(&d[..dim]).unwrap();
  });
}

/// Review item 2: levels above the maximum in a file are harmless too.
#[test]
fn insert_after_bad_levels() {
  let dim = 8;
  let d = data(300, dim);
  let p = HnswParams::default().with_m(4);
  let idx = HnswIndex::build(dim, Metric::L2, p, &d).unwrap();
  let mut buf = idx.write_to(AlignedBuf::new()).unwrap();
  let r = section_range(&buf, "levels");
  let entry_level = {
    let back = HnswIndex::open(Region::new(buf.clone())).unwrap();
    back.stats().max_level as u8
  };
  for b in &mut buf.as_mut_slice()[r] {
    // Keep the entry point's level (validated at open), break the rest.
    if *b != entry_level {
      *b = 200;
    }
  }
  let mut back = HnswIndex::open(Region::new(buf)).unwrap();
  no_panic("insert after bad levels", || {
    for row in d.chunks_exact(dim).take(20) {
      back.insert(row).unwrap();
    }
  });
}

/// Review item 3 (r4): absurd `k` on a flat index.
#[test]
fn huge_k_flat() {
  let d = data(100, 4);
  let mut idx = FlatIndex::from_vectors(4, Metric::L2, &d).unwrap();
  for k in [usize::MAX, 10_000_000] {
    let h = idx.search(&d[..4], &SearchParams::new(k), None).unwrap();
    assert_eq!(h.len(), 100);
  }
  idx.quantize(Quantization::Binary);
  let p = SearchParams::new(usize::MAX);
  assert_eq!(idx.search(&d[..4], &p, None).unwrap().len(), 100);
  let p = SearchParams::new(3).with_rerank(usize::MAX);
  assert_eq!(idx.search(&d[..4], &p, None).unwrap().len(), 3);
}

/// Review item 3 (r5): absurd `ef`, `k` and `rerank` on HNSW.
#[test]
fn huge_ef_k_rerank_hnsw() {
  let d = data(300, 4);
  for q in [Quantization::None, Quantization::Sq8, Quantization::Binary] {
    let p = HnswParams::default().with_quantization(q);
    let idx = HnswIndex::build(4, Metric::L2, p, &d).unwrap();
    for sp in [
      SearchParams::new(1).with_ef(usize::MAX),
      SearchParams::new(usize::MAX),
      SearchParams::new(2).with_rerank(usize::MAX),
    ] {
      let h = idx.search(&d[..4], &sp, None).unwrap();
      assert_eq!(h.len(), sp.k.min(300), "{q:?} {sp:?}");
    }
  }
  assert_eq!(Quantization::Binary.default_rerank(usize::MAX), usize::MAX);
}

/// Review item 4 (r6): a zero-dimensional flat index cannot be quantised,
/// and asking must not panic.
#[test]
fn flat_dim0_quantize() {
  no_panic("dim 0 flat quantize", || {
    let mut f = FlatIndex::new(0, Metric::L2);
    f.quantize(Quantization::Sq8);
    f.add_batch(&[]).unwrap();
    assert_eq!(f.quantization(), Quantization::None);
  });
}

/// r7: even when the selectivity hint lies, a filtered search returns
/// exactly the passing documents, best first.
#[test]
fn filtered_min_k_passing_with_lying_hint() {
  struct Lying(BitSet);
  impl DocFilter for Lying {
    fn matches(&self, id: DocId) -> bool {
      self.0.contains(id)
    }
    fn selectivity_hint(&self) -> Option<f32> {
      Some(1.0)
    }
  }
  let dim = 16;
  let n = 5000;
  let d = data(n, dim);
  for q in [Quantization::None, Quantization::Sq8, Quantization::Binary] {
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
      let p = HnswParams::default().with_quantization(q);
      let idx = HnswIndex::build(dim, metric, p, &d).unwrap();
      let mut bs = BitSet::new(n);
      for id in [7u32, 1234, 4999] {
        bs.insert(id);
      }
      let f = Lying(bs);
      let p = SearchParams::new(10);
      let h = idx.search(&d[..dim], &p, Some(&f)).unwrap();
      assert_eq!(h.len(), 3, "{q:?} {metric:?}");
      assert!(h.windows(2).all(|w| w[0].score >= w[1].score));
    }
  }
}

/// r8: HNSW reports the same exact scores as the flat index, whatever
/// the traversal quantisation (results are re-ranked at full precision).
#[test]
fn scores_consistent_flat_vs_hnsw() {
  let dim = 37;
  let n = 3000;
  let d = data(n, dim);
  for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
    let flat = FlatIndex::from_vectors(dim, metric, &d).unwrap();
    for q in [Quantization::None, Quantization::Sq8, Quantization::Binary] {
      let p = HnswParams::default().with_quantization(q);
      let h = HnswIndex::build(dim, metric, p, &d).unwrap();
      let mut maxdiff = 0f32;
      for i in 0..20 {
        let qv = &d[i * dim..(i + 1) * dim];
        let a = flat.search(qv, &SearchParams::new(10), None).unwrap();
        let b = h.search(qv, &SearchParams::new(10), None).unwrap();
        for hb in &b {
          if let Some(ha) = a.iter().find(|x| x.id == hb.id) {
            maxdiff = maxdiff.max((ha.score - hb.score).abs());
          }
        }
      }
      assert!(maxdiff < 1e-4, "{metric:?} {q:?} {maxdiff:e}");
    }
  }
}

/// Format version 2 adds the protected-edge section and version 3 the
/// node-order maps; version 1 and 2 files (without them) still open,
/// unknown versions do not.
#[test]
fn hnsw_format_versions() {
  let d = data(200, 4);
  let idx = HnswIndex::build(4, Metric::L2, HnswParams::default(), &d).unwrap();
  let buf = idx.write_to(AlignedBuf::new()).unwrap();
  for (version, ok) in
    [(1u32, true), (2, true), (3, true), (4, false), (0, false)]
  {
    let mut b = buf.clone();
    b.as_mut_slice()[16..20].copy_from_slice(&version.to_le_bytes());
    let r = HnswIndex::open(Region::new(b));
    assert_eq!(r.is_ok(), ok, "version {version}");
  }
}
