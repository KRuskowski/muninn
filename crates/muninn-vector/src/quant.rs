//! Vector quantisation: compact codes scanned first, then re-ranked with
//! full-precision vectors.
//!
//! - **SQ8** maps every dimension to a byte using that dimension's
//!   min/max. Distances are *asymmetric*: the query stays `f32` and the
//!   distance to a dequantised vector reduces to one `f32 x u8` dot
//!   product plus per-query and per-vector constants (derivation below),
//!   so no query precision is lost and codes are never decoded.
//! - **Binary** keeps one bit per dimension, the sign of the value after
//!   subtracting the per-dimension mean, packed into `u64` words; 32x
//!   smaller than `f32`. A code decodes to `mean + scale * (+-1)` with a
//!   per-dimension `scale = E|x - mean|`. Distances are asymmetric too:
//!   the query is folded into a 256-entry table per code byte, so a
//!   128-dimensional candidate costs 16 lookups. On clustered test data
//!   this found far more true neighbours than Hamming distance between
//!   sign codes at the same re-rank depth (Hamming throws away the
//!   query's magnitudes and ties constantly), so Hamming is not used.
//!
//! SQ8 distance with `x_i ~ m_i + d_i c_i` (min `m`, step `d`, code `c`):
//!
//! ```text
//! L2:  |q - x|^2 = sum (q_i - m_i)^2 + sum d_i^2 c_i^2
//!                  + sum (-2 (q_i - m_i) d_i) c_i
//! Dot: -q.x      = -sum q_i m_i + sum (-q_i d_i) c_i
//! Cos: 1 - q.x   = 1 - sum q_i m_i + sum (-q_i d_i) c_i
//! ```
//!
//! The middle L2 term depends only on the vector and is stored with it.

use std::io::Write;

use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{DocId, Error, Metric, Result};

use crate::dist::{Dist, prefetch_lines};
use crate::slab::Slab;

/// How an index compresses vectors for its first-pass distance
/// computations. Full-precision vectors are always kept for re-ranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Quantization {
  /// Full-precision `f32` distances only.
  #[default]
  None,
  /// One byte per dimension (per-dimension min/max), 4x smaller.
  Sq8,
  /// One bit per dimension (sign after centring), 32x smaller.
  Binary,
}

impl Quantization {
  /// Stable on-disk tag.
  pub fn to_u8(self) -> u8 {
    match self {
      Self::None => 0,
      Self::Sq8 => 1,
      Self::Binary => 2,
    }
  }

  /// Inverse of [`Quantization::to_u8`].
  pub fn from_u8(t: u8) -> Option<Self> {
    match t {
      0 => Some(Self::None),
      1 => Some(Self::Sq8),
      2 => Some(Self::Binary),
      _ => None,
    }
  }

  /// Candidates re-ranked with full precision when the caller does not
  /// say: binary codes are much coarser, so they need a deeper re-rank.
  pub fn default_rerank(self, k: usize) -> usize {
    match self {
      Self::None => k,
      Self::Sq8 => 4 * k,
      Self::Binary => 16 * k,
    }
  }
}

/// Retrain while the index is smaller than this whenever it has doubled
/// since the last training, so codes built from a handful of early
/// inserts do not clamp everything inserted later.
const RETRAIN_BELOW: usize = 1 << 16;

const T_MIN: [u8; 8] = tag("qmin");
const T_STEP: [u8; 8] = tag("qstep");
const T_CODES: [u8; 8] = tag("qcodes");
const T_SQ: [u8; 8] = tag("qsq");
const T_MEAN: [u8; 8] = tag("qmean");
const T_SCALE: [u8; 8] = tag("qscale");
const T_BITS: [u8; 8] = tag("qbits");

/// Trained codes for every stored vector.
#[derive(Clone, Debug)]
pub(crate) enum Quantizer {
  Sq8(Sq8),
  Binary(Binary),
}

/// SQ8 parameters and codes.
#[derive(Clone, Debug)]
pub(crate) struct Sq8 {
  dim: usize,
  l2: bool,
  min: Slab<f32>,
  step: Slab<f32>,
  codes: Slab<u8>,
  /// Per vector `sum (d_i c_i)^2`; only for L2, empty otherwise.
  sq: Slab<f32>,
}

/// Binary parameters and codes.
#[derive(Clone, Debug)]
pub(crate) struct Binary {
  dim: usize,
  words: usize,
  mean: Slab<f32>,
  /// Per dimension `E|x_i - mean_i|`: the magnitude a set or clear bit
  /// stands for, so a code decodes to `mean + scale * (+1 or -1)`.
  scale: Slab<f32>,
  bits: Slab<u64>,
}

/// A query prepared for one quantiser.
pub(crate) enum QueryCodes {
  Sq8 {
    w: Vec<f32>,
    base: f32,
  },
  /// `lut[j * 256 + b]` is the distance contribution of code byte `j`
  /// having value `b`; the distance is `base + sum_j lut[j][byte_j]`.
  Binary {
    lut: Vec<f32>,
    base: f32,
  },
}

impl Quantizer {
  /// Trains on `data` (row-major, `dim` per row) and encodes every row.
  /// Returns `None` for [`Quantization::None`].
  pub(crate) fn train(
    kind: Quantization,
    dim: usize,
    metric: Metric,
    data: &[f32],
  ) -> Option<Self> {
    let n = data.len().checked_div(dim).unwrap_or(0);
    let mut q = match kind {
      Quantization::None => return None,
      Quantization::Sq8 => {
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for row in data.chunks_exact(dim) {
          for ((l, h), &x) in lo.iter_mut().zip(hi.iter_mut()).zip(row) {
            *l = l.min(x);
            *h = h.max(x);
          }
        }
        let step: Vec<f32> = lo
          .iter()
          .zip(&hi)
          .map(|(&l, &h)| if h > l { (h - l) / 255.0 } else { 0.0 })
          .collect();
        let lo = lo
          .into_iter()
          .map(|l| if l.is_finite() { l } else { 0.0 })
          .collect();
        Self::Sq8(Sq8 {
          dim,
          l2: metric == Metric::L2,
          min: Slab::owned(lo),
          step: Slab::owned(step),
          codes: Slab::default(),
          sq: Slab::default(),
        })
      }
      Quantization::Binary => {
        let mut mean = vec![0.0f64; dim];
        for row in data.chunks_exact(dim) {
          for (m, &x) in mean.iter_mut().zip(row) {
            *m += x as f64;
          }
        }
        let mean: Vec<f32> = mean
          .into_iter()
          .map(|m| if n > 0 { (m / n as f64) as f32 } else { 0.0 })
          .collect();
        let mut dev = vec![0.0f64; dim];
        for row in data.chunks_exact(dim) {
          for ((s, &x), &m) in dev.iter_mut().zip(row).zip(&mean) {
            *s += (x - m).abs() as f64;
          }
        }
        let scale = dev
          .into_iter()
          .map(|s| if n > 0 { (s / n as f64) as f32 } else { 0.0 })
          .collect();
        Self::Binary(Binary {
          dim,
          words: dim.div_ceil(64),
          mean: Slab::owned(mean),
          scale: Slab::owned(scale),
          bits: Slab::default(),
        })
      }
    };
    q.append(data);
    Some(q)
  }

  /// Which quantisation this is.
  pub(crate) fn kind(&self) -> Quantization {
    match self {
      Self::Sq8(_) => Quantization::Sq8,
      Self::Binary(_) => Quantization::Binary,
    }
  }

  /// Number of encoded vectors.
  pub(crate) fn len(&self) -> usize {
    match self {
      Self::Sq8(s) => s.codes.len().checked_div(s.dim).unwrap_or(0),
      Self::Binary(b) => b.bits.len().checked_div(b.words).unwrap_or(0),
    }
  }

  /// Encodes and appends `rows` using the trained parameters (values
  /// outside the trained range are clamped).
  pub(crate) fn append(&mut self, rows: &[f32]) {
    match self {
      Self::Sq8(s) => {
        let (min, step) = (s.min.as_slice(), s.step.as_slice());
        let codes = s.codes.to_mut();
        let mut sq = Vec::new();
        for row in rows.chunks_exact(s.dim) {
          let mut acc = 0.0f32;
          for ((&x, &m), &d) in row.iter().zip(min).zip(step) {
            let c = if d > 0.0 {
              ((x - m) / d).round().clamp(0.0, 255.0) as u8
            } else {
              0
            };
            let v = d * c as f32;
            acc += v * v;
            codes.push(c);
          }
          sq.push(acc);
        }
        if s.l2 {
          s.sq.to_mut().extend_from_slice(&sq);
        }
      }
      Self::Binary(b) => {
        let mean = b.mean.as_slice();
        let bits = b.bits.to_mut();
        for row in rows.chunks_exact(b.dim) {
          let at = bits.len();
          bits.resize(at + b.words, 0);
          for (i, (&x, &m)) in row.iter().zip(mean).enumerate() {
            if x > m {
              bits[at + i / 64] |= 1 << (i % 64);
            }
          }
        }
      }
    }
  }

  /// Retrains from all of `data` if the index has doubled since the codes
  /// were trained on `trained_on` vectors and is still small. Returns the
  /// new training size.
  pub(crate) fn maybe_retrain(
    &mut self,
    metric: Metric,
    dim: usize,
    data: &[f32],
    trained_on: usize,
  ) -> usize {
    let n = data.len() / dim;
    if trained_on < RETRAIN_BELOW && n >= 2 * trained_on.max(1) {
      if let Some(q) = Self::train(self.kind(), dim, metric, data) {
        *self = q;
      }
      return n;
    }
    trained_on
  }

  /// Prepares `query` (already normalised for cosine).
  pub(crate) fn prepare(&self, metric: Metric, query: &[f32]) -> QueryCodes {
    match self {
      Self::Sq8(s) => {
        let (min, step) = (s.min.as_slice(), s.step.as_slice());
        let mut w = Vec::with_capacity(s.dim);
        let mut base = 0.0f32;
        for ((&q, &m), &d) in query.iter().zip(min).zip(step) {
          if metric == Metric::L2 {
            let r = q - m;
            base += r * r;
            w.push(-2.0 * r * d);
          } else {
            base -= q * m;
            w.push(-q * d);
          }
        }
        if metric == Metric::Cosine {
          base += 1.0;
        }
        QueryCodes::Sq8 { w, base }
      }
      Self::Binary(b) => {
        // Per-dimension weight of a sign: distance = base + sum w_i s_i
        // with s_i = +1 for a set bit, -1 for a clear one.
        let (mean, scale) = (b.mean.as_slice(), b.scale.as_slice());
        let mut base = 0.0f32;
        let mut w = vec![0.0f32; b.words * 64];
        for (i, ((&q, &m), &sc)) in
          query.iter().zip(mean).zip(scale).enumerate()
        {
          if metric == Metric::L2 {
            let r = q - m;
            base += r * r + sc * sc;
            w[i] = -2.0 * r * sc;
          } else {
            base -= q * m;
            w[i] = -q * sc;
          }
        }
        if metric == Metric::Cosine {
          base += 1.0;
        }
        // s_i = 2 b_i - 1, so sum w_i s_i = sum_{b_i = 1} 2 w_i - sum w_i.
        base -= w.iter().sum::<f32>();
        let mut lut = vec![0.0f32; b.words * 8 * 256];
        for (j, table) in lut.chunks_exact_mut(256).enumerate() {
          let wj = &w[j * 8..j * 8 + 8];
          for (v, out) in table.iter_mut().enumerate() {
            *out = (0..8)
              .filter(|t| v >> t & 1 == 1)
              .map(|t| 2.0 * wj[t])
              .sum();
          }
        }
        QueryCodes::Binary { lut, base }
      }
    }
  }

  /// The code distance for a prepared query.
  pub(crate) fn dist<'a>(&'a self, qc: &'a QueryCodes) -> CodeDist<'a> {
    match (self, qc) {
      (Self::Sq8(s), QueryCodes::Sq8 { w, base }) => CodeDist::Sq8(Sq8Dist {
        dim: s.dim,
        w,
        base: *base,
        codes: s.codes.as_slice(),
        sq: s.sq.as_slice(),
      }),
      (Self::Binary(b), QueryCodes::Binary { lut, base }) => {
        CodeDist::Bin(BinDist {
          row_bytes: b.words * 8,
          lut,
          base: *base,
          bytes: muninn_core::as_bytes(b.bits.as_slice()),
        })
      }
      _ => unreachable!("query prepared by a different quantiser"),
    }
  }

  /// Writes the parameter and code sections.
  pub(crate) fn write_sections<W: Write>(
    &self,
    w: &mut ContainerWriter<W>,
  ) -> Result<()> {
    match self {
      Self::Sq8(s) => {
        w.pod_section(T_MIN, s.min.as_slice())?;
        w.pod_section(T_STEP, s.step.as_slice())?;
        w.pod_section(T_CODES, s.codes.as_slice())?;
        w.pod_section(T_SQ, s.sq.as_slice())
      }
      Self::Binary(b) => {
        w.pod_section(T_MEAN, b.mean.as_slice())?;
        w.pod_section(T_SCALE, b.scale.as_slice())?;
        w.pod_section(T_BITS, b.bits.as_slice())
      }
    }
  }

  /// Opens the sections written by [`Self::write_sections`], zero-copy,
  /// and checks they describe `n` vectors of `dim` dimensions.
  pub(crate) fn open(
    kind: Quantization,
    c: &Container,
    dim: usize,
    metric: Metric,
    n: usize,
  ) -> Result<Option<Self>> {
    let q = match kind {
      Quantization::None => return Ok(None),
      Quantization::Sq8 => Self::Sq8(Sq8 {
        dim,
        l2: metric == Metric::L2,
        min: Slab::from_region(c.section(T_MIN)?)?,
        step: Slab::from_region(c.section(T_STEP)?)?,
        codes: Slab::from_region(c.section(T_CODES)?)?,
        sq: Slab::from_region(c.section(T_SQ)?)?,
      }),
      Quantization::Binary => Self::Binary(Binary {
        dim,
        words: dim.div_ceil(64),
        mean: Slab::from_region(c.section(T_MEAN)?)?,
        scale: Slab::from_region(c.section(T_SCALE)?)?,
        bits: Slab::from_region(c.section(T_BITS)?)?,
      }),
    };
    let ok = match &q {
      Self::Sq8(s) => {
        s.min.len() == dim
          && s.step.len() == dim
          && s.codes.len() == n * dim
          && s.sq.len() == if s.l2 { n } else { 0 }
      }
      Self::Binary(b) => {
        b.mean.len() == dim
          && b.scale.len() == dim
          && b.bits.len() == n * b.words
      }
    };
    if !ok {
      return Err(Error::corrupt("quantiser sections have the wrong size"));
    }
    Ok(Some(q))
  }

  /// Bytes of codes and parameters.
  pub(crate) fn bytes(&self) -> usize {
    match self {
      Self::Sq8(s) => {
        s.min.bytes() + s.step.bytes() + s.codes.bytes() + s.sq.bytes()
      }
      Self::Binary(b) => b.mean.bytes() + b.scale.bytes() + b.bits.bytes(),
    }
  }

  /// Whether the codes are read in place from a mapped region.
  pub(crate) fn is_mapped(&self) -> bool {
    match self {
      Self::Sq8(s) => s.codes.is_mapped(),
      Self::Binary(b) => b.bits.is_mapped(),
    }
  }
}

/// A code distance of either kind; match on it (or use
/// [`with_code_dist!`]) to get one monomorphised hot loop per quantiser.
pub(crate) enum CodeDist<'a> {
  Sq8(Sq8Dist<'a>),
  Bin(BinDist<'a>),
}

/// Evaluates `$body` with `$x` bound to the concrete distance in a
/// [`CodeDist`].
macro_rules! with_code_dist {
  ($cd:expr, |$x:ident| $body:expr) => {
    match $cd {
      $crate::quant::CodeDist::Sq8($x) => $body,
      $crate::quant::CodeDist::Bin($x) => $body,
    }
  };
}
pub(crate) use with_code_dist;

/// SQ8 distance for one prepared query.
pub(crate) struct Sq8Dist<'a> {
  dim: usize,
  w: &'a [f32],
  base: f32,
  codes: &'a [u8],
  sq: &'a [f32],
}

impl Dist for Sq8Dist<'_> {
  #[inline]
  fn dist(&self, id: DocId) -> f32 {
    let i = id as usize;
    let s = i * self.dim;
    let extra = self.sq.get(i).copied().unwrap_or(0.0);
    self.base + extra + dot_f32_u8(self.w, &self.codes[s..s + self.dim])
  }

  #[inline]
  fn prefetch(&self, id: DocId) {
    prefetch_lines(self.codes.as_ptr(), id as usize * self.dim, self.dim);
  }
}

/// Asymmetric binary distance for one prepared query: one table lookup
/// per code byte.
pub(crate) struct BinDist<'a> {
  row_bytes: usize,
  lut: &'a [f32],
  base: f32,
  bytes: &'a [u8],
}

impl Dist for BinDist<'_> {
  #[inline]
  fn dist(&self, id: DocId) -> f32 {
    let s = id as usize * self.row_bytes;
    let code = &self.bytes[s..s + self.row_bytes];
    // Rows are whole u64 words, so always a multiple of 4 bytes.
    let mut acc = [0.0f32; 4];
    for (c, t) in code.chunks_exact(4).zip(self.lut.chunks_exact(4 * 256)) {
      for i in 0..4 {
        acc[i] += t[i * 256 + c[i] as usize];
      }
    }
    self.base + (acc[0] + acc[1]) + (acc[2] + acc[3])
  }

  #[inline]
  fn prefetch(&self, id: DocId) {
    let at = id as usize * self.row_bytes;
    prefetch_lines(self.bytes.as_ptr(), at, self.row_bytes);
  }
}

/// `sum w_i * c_i` with the codes widened to `f32`.
///
/// muninn-simd has no mixed `f32 x u8` kernel yet; this portable loop
/// keeps eight independent accumulators so the compiler vectorises it.
#[inline]
pub(crate) fn dot_f32_u8(w: &[f32], c: &[u8]) -> f32 {
  debug_assert_eq!(w.len(), c.len());
  let mut acc = [0.0f32; 8];
  let (cw, cc) = (w.chunks_exact(8), c.chunks_exact(8));
  let (rw, rc) = (cw.remainder(), cc.remainder());
  for (x, y) in cw.zip(cc) {
    for i in 0..8 {
      acc[i] += x[i] * y[i] as f32;
    }
  }
  let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
    + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  for (x, &y) in rw.iter().zip(rc) {
    s += x * y as f32;
  }
  s
}

#[cfg(test)]
mod tests {
  use super::*;

  fn rows(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
      .map(|i| ((i * 2654435761) % 10007) as f32 / 5000.0 - 1.0)
      .collect()
  }

  #[test]
  fn dot_f32_u8_matches_reference() {
    for len in 0..40 {
      let w: Vec<f32> = (0..len).map(|i| i as f32 * 0.37 - 3.0).collect();
      let c: Vec<u8> = (0..len).map(|i| (i * 53 % 256) as u8).collect();
      let r: f32 = w.iter().zip(&c).map(|(a, &b)| a * b as f32).sum();
      assert!((dot_f32_u8(&w, &c) - r).abs() <= 1e-3 * r.abs().max(1.0));
    }
  }

  #[test]
  fn sq8_distance_equals_distance_to_dequantised_vector() {
    let (n, dim) = (50, 19);
    let data = rows(n, dim);
    let q = &rows(1, dim)[..];
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
      let qz =
        Quantizer::train(Quantization::Sq8, dim, metric, &data).expect("sq8");
      let Quantizer::Sq8(s) = &qz else {
        unreachable!()
      };
      let qc = qz.prepare(metric, q);
      for id in 0..n {
        let deq: Vec<f32> = (0..dim)
          .map(|i| {
            s.min.as_slice()[i]
              + s.step.as_slice()[i] * s.codes.as_slice()[id * dim + i] as f32
          })
          .collect();
        let want = muninn_simd::distance(metric, q, &deq);
        let got = with_code_dist!(qz.dist(&qc), |x| x.dist(id as DocId));
        assert!((got - want).abs() < 1e-3, "{metric:?} {got} {want}");
        // Dequantisation error is at most half a step per dimension.
        let orig = &data[id * dim..(id + 1) * dim];
        for i in 0..dim {
          assert!((deq[i] - orig[i]).abs() <= s.step.as_slice()[i] * 0.51);
        }
      }
    }
  }

  #[test]
  fn binary_codes_are_centred_signs() {
    let data = [1.0, 10.0, 3.0, 20.0, 5.0, 30.0];
    let qz = Quantizer::train(Quantization::Binary, 2, Metric::L2, &data)
      .expect("binary");
    let Quantizer::Binary(b) = &qz else {
      unreachable!()
    };
    // Means are 3 and 20: row 0 is below both, row 2 above both, and row
    // 1 equals the mean (not above it).
    assert_eq!(b.bits.as_slice(), &[0b00, 0b00, 0b11]);
    assert_eq!(b.scale.as_slice(), &[4.0 / 3.0, 20.0 / 3.0]);
  }

  #[test]
  fn binary_distance_equals_distance_to_decoded_vector() {
    let (n, dim) = (40, 70);
    let data = rows(n, dim);
    let q = &rows(1, dim)[..];
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
      let qz = Quantizer::train(Quantization::Binary, dim, metric, &data)
        .expect("binary");
      let Quantizer::Binary(b) = &qz else {
        unreachable!()
      };
      let qc = qz.prepare(metric, q);
      for id in 0..n {
        let dec: Vec<f32> = (0..dim)
          .map(|i| {
            let bit = b.bits.as_slice()[id * b.words + i / 64] >> (i % 64) & 1;
            let sign = if bit == 1 { 1.0 } else { -1.0 };
            b.mean.as_slice()[i] + b.scale.as_slice()[i] * sign
          })
          .collect();
        let want = muninn_simd::distance(metric, q, &dec);
        let got = with_code_dist!(qz.dist(&qc), |x| x.dist(id as DocId));
        assert!((got - want).abs() < 1e-3, "{metric:?} {got} {want}");
      }
    }
  }

  #[test]
  fn retrain_on_growth() {
    let dim = 4;
    let mut data = rows(2, dim);
    let mut qz =
      Quantizer::train(Quantization::Sq8, dim, Metric::L2, &data).unwrap();
    let mut trained = 2;
    data.extend(rows(3, dim).iter().map(|x| x * 100.0));
    qz.append(&data[2 * dim..]);
    trained = qz.maybe_retrain(Metric::L2, dim, &data, trained);
    assert_eq!(trained, 5);
    assert_eq!(qz.len(), 5);
  }
}
