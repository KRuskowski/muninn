use std::io::Write;
use std::ops::Range;

use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{DocFilter, DocId, Error, Hit, Metric, Region, Result};

use crate::dist::{FullDist, scan_dist};
use crate::quant::{Quantization, Quantizer, with_code_dist};
use crate::scan::{exact_search, par_scan, rerank, sparse_candidates};
use crate::slab::Slab;
use crate::{SearchParams, VectorIndex, meta};

/// Container kind of a [`FlatIndex`] file.
pub const FLAT_KIND: [u8; 8] = tag("flat");
/// Current [`FlatIndex`] file format version. Version 2 adds the SQ8
/// sections `qexact` and `qcoarse` (vectors scored at full precision);
/// version 1 files still open.
pub const FLAT_VERSION: u32 = 2;

const T_META: [u8; 8] = tag("meta");
const T_VECTORS: [u8; 8] = tag("vectors");

/// Exact brute-force index over full-precision vectors.
///
/// The recall oracle for every approximate index and the memtable index
/// of the engine: inserts are O(1) and, below tens of thousands of
/// vectors, a blocked scan beats any graph. Search scans blocks of rows
/// with [`muninn_simd::distance_batch`], applies the filter during the
/// scan, and splits large scans across rayon's thread pool with one
/// [`muninn_core::TopK`] per task merged at the end.
///
/// Vectors are either owned (growable) or read in place from a mapped
/// [`Region`] after [`FlatIndex::open`]; adding to a mapped index copies
/// it first.
///
/// Optionally ([`FlatIndex::quantize`]) the scan runs over SQ8 or binary
/// codes and the best `params.rerank` candidates are re-scored exactly.
/// Unquantised (the default), results are exact.
#[derive(Clone, Debug)]
pub struct FlatIndex {
  dim: usize,
  metric: Metric,
  data: Slab<f32>,
  quant: Option<Quantizer>,
  trained_on: usize,
}

impl FlatIndex {
  /// An empty index of `dim`-dimensional vectors compared by `metric`.
  pub fn new(dim: usize, metric: Metric) -> Self {
    Self {
      dim,
      metric,
      data: Slab::default(),
      quant: None,
      trained_on: 0,
    }
  }

  /// Builds an index from row-major `vectors` (`dim` floats per row).
  pub fn from_vectors(
    dim: usize,
    metric: Metric,
    vectors: &[f32],
  ) -> Result<Self> {
    let mut idx = Self::new(dim, metric);
    idx.add_batch(vectors)?;
    Ok(idx)
  }

  /// Appends a vector and returns its id.
  pub fn add(&mut self, vector: &[f32]) -> Result<DocId> {
    check_dim(self.dim, vector)?;
    Ok(self.add_batch(vector)?.start)
  }

  /// Appends row-major `vectors` (`dim` floats per row) and returns the
  /// ids they were given.
  pub fn add_batch(&mut self, vectors: &[f32]) -> Result<Range<DocId>> {
    let rows = check_rows(self.dim, vectors)?;
    let first = self.len();
    check_capacity(first, rows)?;
    let data = self.data.to_mut();
    let start = data.len();
    data.extend_from_slice(vectors);
    if self.metric.normalizes() {
      for row in data[start..].chunks_exact_mut(self.dim) {
        muninn_simd::normalize(row);
      }
    }
    if let Some(q) = &mut self.quant {
      q.append(&data[start..]);
      self.trained_on =
        q.maybe_retrain(self.metric, self.dim, data, self.trained_on);
    }
    Ok(first as DocId..(first + rows) as DocId)
  }

  /// The stored (possibly normalised) vector for `id`. Panics if `id` is
  /// out of range.
  pub fn vector(&self, id: DocId) -> &[f32] {
    let s = id as usize * self.dim;
    &self.data.as_slice()[s..s + self.dim]
  }

  /// All stored vectors, row-major (normalised for cosine).
  pub fn vectors(&self) -> &[f32] {
    self.data.as_slice()
  }

  /// Switches the scan to quantised codes trained on the current vectors
  /// (re-trained automatically while the index is small and growing), or
  /// back to exact search with [`Quantization::None`].
  pub fn quantize(&mut self, kind: Quantization) {
    self.quant =
      Quantizer::train(kind, self.dim, self.metric, self.data.as_slice());
    self.trained_on = self.len();
  }

  /// The active quantisation.
  pub fn quantization(&self) -> Quantization {
    self.quant.as_ref().map_or(Quantization::None, |q| q.kind())
  }

  /// Whether the vectors are read in place from a mapped region.
  pub fn is_mapped(&self) -> bool {
    self.data.is_mapped()
  }

  /// Bytes held for vectors and codes (owned or mapped).
  pub fn memory_bytes(&self) -> usize {
    self.data.bytes() + self.quant.as_ref().map_or(0, |q| q.bytes())
  }

  /// Writes the index as one container file in a single streaming pass.
  pub fn write_to<W: Write>(&self, out: W) -> Result<W> {
    let mut w = ContainerWriter::new(out, FLAT_KIND, FLAT_VERSION)?;
    self.write_sections(&mut w)?;
    w.finish()
  }

  /// Writes the index's sections into an already open container, for
  /// callers that bundle several structures in one file.
  pub fn write_sections<W: Write>(
    &self,
    w: &mut ContainerWriter<W>,
  ) -> Result<()> {
    let m = [
      self.dim as u64,
      self.metric.to_u8() as u64,
      self.len() as u64,
      self.quantization().to_u8() as u64,
      self.trained_on as u64,
    ];
    w.pod_section(T_META, &m)?;
    w.pod_section(T_VECTORS, self.data.as_slice())?;
    if let Some(q) = &self.quant {
      q.write_sections(w)?;
    }
    Ok(())
  }

  /// Opens a file written by [`FlatIndex::write_to`]. Vectors and codes
  /// are read in place from `region`; nothing is copied.
  pub fn open(region: Region) -> Result<Self> {
    let c = Container::open(region)?;
    if c.kind() != FLAT_KIND {
      return Err(Error::corrupt("not a flat index file"));
    }
    if !(1..=FLAT_VERSION).contains(&c.version()) {
      return Err(Error::corrupt(format!(
        "flat format version {} is not supported (expected 1..={FLAT_VERSION})",
        c.version()
      )));
    }
    Self::from_container(&c)
  }

  /// Opens the sections written by [`FlatIndex::write_sections`].
  pub fn from_container(c: &Container) -> Result<Self> {
    let m = meta::read(c.section(T_META)?, 5)?;
    let dim = meta::usize_field(m[0], "dimension")?;
    let metric = meta::metric(m[1])?;
    let n = meta::usize_field(m[2], "vector count")?;
    let kind = meta::quantization(m[3])?;
    let data = Slab::from_region(c.section(T_VECTORS)?)?;
    if Some(data.len()) != n.checked_mul(dim) {
      return Err(Error::corrupt("flat vectors section has the wrong size"));
    }
    let mut quant = Quantizer::open(kind, c, dim, metric, n)?;
    if let Some(q) = &mut quant {
      q.upgrade_legacy(metric, data.as_slice());
    }
    Ok(Self {
      dim,
      metric,
      data,
      quant,
      trained_on: meta::usize_field(m[4], "training size")?,
    })
  }
}

pub(crate) fn check_dim(dim: usize, v: &[f32]) -> Result<()> {
  if v.len() != dim {
    return Err(Error::DimensionMismatch {
      expected: dim,
      got: v.len(),
    });
  }
  Ok(())
}

/// Checks a row-major batch and returns its row count.
pub(crate) fn check_rows(dim: usize, vectors: &[f32]) -> Result<usize> {
  if dim == 0 {
    return if vectors.is_empty() {
      Ok(0)
    } else {
      Err(Error::DimensionMismatch {
        expected: 0,
        got: vectors.len(),
      })
    };
  }
  if !vectors.len().is_multiple_of(dim) {
    return Err(Error::DimensionMismatch {
      expected: dim,
      got: vectors.len() % dim,
    });
  }
  Ok(vectors.len() / dim)
}

/// Ids are `u32`; refuses batches that would overflow them.
pub(crate) fn check_capacity(len: usize, more: usize) -> Result<()> {
  if len + more > DocId::MAX as usize {
    return Err(Error::invalid("index would exceed u32::MAX vectors"));
  }
  Ok(())
}

/// The query, normalised for cosine.
pub(crate) fn prepare_query(metric: Metric, query: &[f32]) -> Vec<f32> {
  let mut q = query.to_vec();
  if metric.normalizes() {
    muninn_simd::normalize(&mut q);
  }
  q
}

impl VectorIndex for FlatIndex {
  fn dim(&self) -> usize {
    self.dim
  }

  fn metric(&self) -> Metric {
    self.metric
  }

  fn len(&self) -> usize {
    self.data.len().checked_div(self.dim).unwrap_or(0)
  }

  fn search(
    &self,
    query: &[f32],
    params: &SearchParams,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>> {
    check_dim(self.dim, query)?;
    // No search can return more than every vector: clamping keeps absurd
    // `k`/`rerank` values from sizing allocations.
    let n = self.len();
    let k = params.k.min(n);
    if k == 0 {
      return Ok(Vec::new());
    }
    let q = prepare_query(self.metric, query);
    let data = self.data.as_slice();
    let top = match &self.quant {
      None => exact_search(self.metric, self.dim, data, &q, k, filter, None),
      // A selective filter with an id set: scoring just those ids at full
      // precision is cheaper than scanning every code, and exact.
      Some(_) if filter.is_some_and(|f| sparse_candidates(f, n)) => {
        exact_search(self.metric, self.dim, data, &q, k, filter, None)
      }
      Some(qz) => {
        let r = match params.rerank {
          0 => qz.kind().default_rerank(k),
          r => r,
        }
        .clamp(k, n);
        let qc = qz.prepare(self.metric, &q);
        let full = FullDist {
          metric: self.metric,
          dim: self.dim,
          query: &q,
          data,
        };
        let cd = qz.dist(&qc, full);
        let cands = with_code_dist!(cd, |d| {
          par_scan(n, r, n * self.dim / 4, |range, top| {
            scan_dist(&d, range, filter, top)
          })
        });
        let ids = cands.iter().map(|(id, _)| id);
        rerank(self.metric, data, &q, ids, k, &[])
      }
    };
    Ok(top.into_hits(self.metric))
  }
}

#[cfg(test)]
mod tests {
  use muninn_core::AlignedBuf;

  use super::*;

  #[test]
  fn finds_exact_neighbours_with_filter() {
    let mut idx = FlatIndex::new(2, Metric::L2);
    for v in [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [0.9, 0.1]] {
      idx.add(&v).unwrap();
    }
    let p = SearchParams::new(2);
    let hits = idx.search(&[1.0, 0.0], &p, None).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 3]);
    assert_eq!(hits[0].score, 0.0);
    let odd = |id: DocId| id % 2 == 1;
    let hits = idx.search(&[0.0, 0.0], &p, Some(&odd)).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![3, 1]);
  }

  #[test]
  fn cosine_normalises_and_rejects_bad_dims() {
    let mut idx = FlatIndex::new(2, Metric::Cosine);
    idx.add(&[10.0, 0.0]).unwrap();
    idx.add(&[0.0, 3.0]).unwrap();
    let hits = idx
      .search(&[2.0, 0.0], &SearchParams::new(1), None)
      .unwrap();
    assert_eq!(hits[0].id, 0);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    assert!(matches!(
      idx.add(&[1.0]),
      Err(Error::DimensionMismatch {
        expected: 2,
        got: 1
      })
    ));
    assert!(idx.add_batch(&[1.0, 2.0, 3.0]).is_err());
    assert_eq!(idx.len(), 2);
  }

  #[test]
  fn round_trip_is_zero_copy_and_growable() -> Result<()> {
    let mut idx = FlatIndex::new(3, Metric::Dot);
    idx.add_batch(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).unwrap();
    let buf = idx.write_to(AlignedBuf::new()).unwrap();
    let mut back = FlatIndex::open(Region::new(buf)).unwrap();
    assert!(back.is_mapped());
    assert_eq!(back.vectors(), idx.vectors());
    let p = SearchParams::new(2);
    let q = [0.5, 1.0, 0.0];
    assert_eq!(back.search(&q, &p, None)?, idx.search(&q, &p, None)?);
    assert_eq!(back.add(&[0.0, 0.0, 9.0]).unwrap(), 2);
    assert!(!back.is_mapped());
    assert_eq!(back.search(&[0.0, 0.0, 1.0], &p, None)?[0].id, 2);
    Ok(())
  }

  #[test]
  fn open_rejects_garbage() {
    let idx = FlatIndex::new(3, Metric::L2);
    let buf = idx.write_to(AlignedBuf::new()).unwrap();
    let mut bad = buf.clone();
    // Corrupt the metric byte in the meta section (offset 64 + 8).
    bad.as_mut_slice()[72] = 9;
    assert!(FlatIndex::open(Region::new(bad)).is_err());
    let short = Region::copy_from(&buf.as_slice()[..buf.len() - 3]);
    assert!(FlatIndex::open(short).is_err());
  }
}
