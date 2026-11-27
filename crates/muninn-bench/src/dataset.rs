//! Vector datasets in the harness's on-disk format.
//!
//! `scripts/prepare_datasets.py` converts every source (TEXMEX
//! fvecs/ivecs, ann-benchmarks HDF5) into one directory per dataset:
//!
//! - `base.fbin`, `query.fbin`: `u32` rows, `u32` columns, then
//!   `rows * columns` little-endian `f32` values, row-major.
//! - `gt.ibin`: `u32` rows, `u32` columns, then `rows * columns`
//!   little-endian `i32` neighbour ids per query, nearest first.
//! - `meta.json`: name, metric, source URLs and checksums.
//!
//! The same layout ("fbin"/"ibin") is used by big-ann-benchmarks, and
//! numpy reads it with one `fromfile` call.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use muninn_core::Metric;
use serde::{Deserialize, Serialize};

use crate::Result;

/// A dense row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
  /// Number of rows.
  pub rows: usize,
  /// Values per row.
  pub cols: usize,
  /// `rows * cols` values, row-major.
  pub data: Vec<T>,
}

impl<T> Matrix<T> {
  /// Row `i`.
  pub fn row(&self, i: usize) -> &[T] {
    &self.data[i * self.cols..(i + 1) * self.cols]
  }

  /// Keeps only the first `rows` rows (no-op if there are fewer).
  pub fn truncate(&mut self, rows: usize) {
    if rows < self.rows {
      self.rows = rows;
      self.data.truncate(rows * self.cols);
    }
  }
}

/// A 4-byte little-endian value stored in the binary files.
trait Word: Copy + Default {
  fn from_le(b: [u8; 4]) -> Self;
  fn to_le(self) -> [u8; 4];
}

impl Word for f32 {
  fn from_le(b: [u8; 4]) -> Self {
    f32::from_le_bytes(b)
  }

  fn to_le(self) -> [u8; 4] {
    self.to_le_bytes()
  }
}

impl Word for i32 {
  fn from_le(b: [u8; 4]) -> Self {
    i32::from_le_bytes(b)
  }

  fn to_le(self) -> [u8; 4] {
    self.to_le_bytes()
  }
}

fn read_bin<T: Word>(path: &Path) -> Result<Matrix<T>> {
  let file = File::open(path)
    .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
  let file_len = file.metadata()?.len();
  let mut r = BufReader::with_capacity(1 << 22, file);
  let mut head = [0u8; 8];
  r.read_exact(&mut head)?;
  let rows = u32::from_le_bytes(head[..4].try_into().unwrap()) as usize;
  let cols = u32::from_le_bytes(head[4..].try_into().unwrap()) as usize;
  let len = rows
    .checked_mul(cols)
    .ok_or_else(|| format!("{}: header overflows", path.display()))?;
  let want = 8 + 4 * len as u64;
  if file_len != want {
    return Err(
      format!(
        "{}: header says {rows}x{cols} ({want} bytes), file has {file_len}",
        path.display()
      )
      .into(),
    );
  }
  let mut data = Vec::with_capacity(len);
  let mut buf = vec![0u8; 1 << 22];
  let mut left = len * 4;
  while left > 0 {
    let n = left.min(buf.len());
    r.read_exact(&mut buf[..n])?;
    data.extend(
      buf[..n]
        .chunks_exact(4)
        .map(|c| T::from_le(c.try_into().unwrap())),
    );
    left -= n;
  }
  Ok(Matrix { rows, cols, data })
}

fn write_bin<T: Word>(path: &Path, m: &Matrix<T>) -> Result<()> {
  assert_eq!(m.data.len(), m.rows * m.cols, "matrix shape");
  let mut w = BufWriter::new(File::create(path)?);
  w.write_all(&u32::try_from(m.rows)?.to_le_bytes())?;
  w.write_all(&u32::try_from(m.cols)?.to_le_bytes())?;
  for &v in &m.data {
    w.write_all(&v.to_le())?;
  }
  w.flush()?;
  Ok(())
}

/// Reads an `.fbin` file.
pub fn read_fbin(path: &Path) -> Result<Matrix<f32>> {
  read_bin(path)
}

/// Writes an `.fbin` file.
pub fn write_fbin(path: &Path, m: &Matrix<f32>) -> Result<()> {
  write_bin(path, m)
}

/// Reads an `.ibin` file of neighbour ids. Negative ids (padding in some
/// sources) are rejected.
pub fn read_ibin(path: &Path) -> Result<Matrix<u32>> {
  let m: Matrix<i32> = read_bin(path)?;
  let data = m
    .data
    .iter()
    .map(|&v| u32::try_from(v))
    .collect::<std::result::Result<Vec<u32>, _>>()
    .map_err(|_| format!("{}: negative neighbour id", path.display()))?;
  Ok(Matrix {
    rows: m.rows,
    cols: m.cols,
    data,
  })
}

/// Writes an `.ibin` file of neighbour ids.
pub fn write_ibin(path: &Path, m: &Matrix<u32>) -> Result<()> {
  let data = m
    .data
    .iter()
    .map(|&v| i32::try_from(v))
    .collect::<std::result::Result<Vec<i32>, _>>()?;
  write_bin(
    path,
    &Matrix {
      rows: m.rows,
      cols: m.cols,
      data,
    },
  )
}

/// Contents of a dataset's `meta.json` that the harness reads. Other
/// fields (sources, checksums) are carried through to reports untouched.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetMeta {
  /// Dataset name, e.g. `sift-128-euclidean`.
  pub name: String,
  /// `l2`, `cosine` or `dot`.
  pub metric: String,
  /// Everything else in the file.
  #[serde(flatten)]
  pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A loaded vector dataset.
#[derive(Clone, Debug)]
pub struct Dataset {
  /// Its `meta.json`.
  pub meta: DatasetMeta,
  /// Vectors to index.
  pub base: Matrix<f32>,
  /// Query vectors.
  pub query: Matrix<f32>,
  /// Exact neighbours of each query, nearest first.
  pub gt: Matrix<u32>,
}

impl Dataset {
  /// Loads the dataset in `dir`, checking that the files agree in shape.
  pub fn load(dir: &Path) -> Result<Self> {
    let meta: DatasetMeta = serde_json::from_reader(BufReader::new(
      File::open(dir.join("meta.json"))
        .map_err(|e| format!("{}/meta.json: {e}", dir.display()))?,
    ))?;
    let base = read_fbin(&dir.join("base.fbin"))?;
    let query = read_fbin(&dir.join("query.fbin"))?;
    let gt = read_ibin(&dir.join("gt.ibin"))?;
    if base.cols != query.cols {
      return Err(
        format!("base has {} dims, queries {}", base.cols, query.cols).into(),
      );
    }
    if gt.rows != query.rows {
      return Err(
        format!("{} queries but {} ground-truth rows", query.rows, gt.rows)
          .into(),
      );
    }
    if let Some(&bad) = gt.data.iter().find(|&&id| id as usize >= base.rows) {
      return Err(format!("ground-truth id {bad} out of range").into());
    }
    let ds = Self {
      meta,
      base,
      query,
      gt,
    };
    ds.metric()?;
    Ok(ds)
  }

  /// The dataset's metric.
  pub fn metric(&self) -> Result<Metric> {
    Metric::parse(&self.meta.metric).ok_or_else(|| {
      format!("unknown metric {:?} in meta.json", self.meta.metric).into()
    })
  }

  /// Vector dimension.
  pub fn dim(&self) -> usize {
    self.base.cols
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
      .join(format!("muninn-bench-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  #[test]
  fn fbin_and_ibin_round_trip() {
    let dir = tmp("roundtrip");
    let f = Matrix {
      rows: 3,
      cols: 2,
      data: vec![1.0, -2.5, 3.25, 0.0, f32::MAX, 1e-9],
    };
    write_fbin(&dir.join("a.fbin"), &f).unwrap();
    assert_eq!(read_fbin(&dir.join("a.fbin")).unwrap(), f);
    let i = Matrix {
      rows: 2,
      cols: 3,
      data: vec![0, 1, 2, 7, 8, 9],
    };
    write_ibin(&dir.join("a.ibin"), &i).unwrap();
    assert_eq!(read_ibin(&dir.join("a.ibin")).unwrap(), i);
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[test]
  fn rejects_wrong_length_and_negative_ids() {
    let dir = tmp("bad");
    let p = dir.join("bad.fbin");
    // Header claims 2x2 floats, body holds one.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    std::fs::write(&p, &bytes).unwrap();
    assert!(read_fbin(&p).is_err());
    let p = dir.join("neg.ibin");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(-1i32).to_le_bytes());
    std::fs::write(&p, &bytes).unwrap();
    assert!(read_ibin(&p).is_err());
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[test]
  fn loads_a_dataset_directory() {
    let dir = tmp("ds");
    std::fs::write(
      dir.join("meta.json"),
      r#"{"name": "tiny", "metric": "l2", "source": "test"}"#,
    )
    .unwrap();
    let base = Matrix {
      rows: 4,
      cols: 2,
      data: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0],
    };
    let query = Matrix {
      rows: 1,
      cols: 2,
      data: vec![0.9, 0.1],
    };
    let gt = Matrix {
      rows: 1,
      cols: 2,
      data: vec![1, 0],
    };
    write_fbin(&dir.join("base.fbin"), &base).unwrap();
    write_fbin(&dir.join("query.fbin"), &query).unwrap();
    write_ibin(&dir.join("gt.ibin"), &gt).unwrap();
    let ds = Dataset::load(&dir).unwrap();
    assert_eq!(ds.dim(), 2);
    assert_eq!(ds.metric().unwrap(), Metric::L2);
    assert_eq!(ds.meta.extra["source"], "test");
    // An out-of-range neighbour id is caught on load.
    let bad = Matrix {
      rows: 1,
      cols: 1,
      data: vec![4],
    };
    write_ibin(&dir.join("gt.ibin"), &bad).unwrap();
    assert!(Dataset::load(&dir).is_err());
    std::fs::remove_dir_all(dir).unwrap();
  }
}
