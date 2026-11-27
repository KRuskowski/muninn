use muninn_core::{DocFilter, DocId, Error, Hit, Metric, Result, TopK};

use crate::{SearchParams, VectorIndex};

/// Exact brute-force index over full-precision vectors.
///
/// Reference implementation: simple and obviously correct, used as the
/// ground truth in recall tests. Optimised scanning (parallelism,
/// blocking, quantised pre-filtering) belongs in the vector workstream.
#[derive(Clone, Debug)]
pub struct FlatIndex {
  dim: usize,
  metric: Metric,
  data: Vec<f32>,
}

impl FlatIndex {
  pub fn new(dim: usize, metric: Metric) -> Self {
    Self {
      dim,
      metric,
      data: Vec::new(),
    }
  }

  /// Appends a vector and returns its id.
  pub fn add(&mut self, vector: &[f32]) -> Result<DocId> {
    check_dim(self.dim, vector)?;
    let id = self.len() as DocId;
    let start = self.data.len();
    self.data.extend_from_slice(vector);
    if self.metric.normalizes() {
      muninn_simd::normalize(&mut self.data[start..]);
    }
    Ok(id)
  }

  /// The stored (possibly normalised) vector for `id`.
  pub fn vector(&self, id: DocId) -> &[f32] {
    let s = id as usize * self.dim;
    &self.data[s..s + self.dim]
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

impl VectorIndex for FlatIndex {
  fn dim(&self) -> usize {
    self.dim
  }

  fn metric(&self) -> Metric {
    self.metric
  }

  fn len(&self) -> usize {
    if self.dim == 0 {
      0
    } else {
      self.data.len() / self.dim
    }
  }

  fn search(
    &self,
    query: &[f32],
    params: &SearchParams,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>> {
    check_dim(self.dim, query)?;
    let mut q = query.to_vec();
    if self.metric.normalizes() {
      muninn_simd::normalize(&mut q);
    }
    let mut top = TopK::new(params.k);
    for id in 0..self.len() as DocId {
      if filter.is_some_and(|f| !f.matches(id)) {
        continue;
      }
      let d = muninn_simd::distance(self.metric, &q, self.vector(id));
      top.push(id, d);
    }
    Ok(top.into_hits(self.metric))
  }
}

#[cfg(test)]
mod tests {
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
  }
}
