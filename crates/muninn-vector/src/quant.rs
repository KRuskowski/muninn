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

use crate::dist::{Dist, FullDist, prefetch_lines};
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
      Self::Sq8 => k.saturating_mul(4),
      Self::Binary => k.saturating_mul(16),
    }
  }
}

/// Fractions cut from each tail of every dimension when SQ8 ranges are
/// fitted, tried in turn until the codes resolve the bulk of the values
/// (see `fit_sq8`): a few extreme vectors then cannot stretch the range
/// until every normal vector falls onto one or two code levels.
const CLIPS: [f64; 3] = [0.001, 0.01, 0.05];

/// Largest coding error relative to the spread of the bulk of the values
/// (`error_ratio`) at which SQ8 codes are used. Well-spread data scores
/// about 1e-4; codes whose range an outlier stretched score near 1.
pub(crate) const MAX_CODE_ERROR: f64 = 1e-2;

/// A vector whose values saturate (fall outside the code range) by more
/// than this many times the codes' mean squared rounding error is scored
/// at full precision instead of from its codes (see `Sq8::exact`).
const EXACT_FACTOR: f64 = 64.0;

/// Fraction of vectors scored at full precision above which the codes are
/// retrained (when the index has also grown by half since the last
/// training), and above which a build does not search with them.
const EXACT_MAX_FRACTION: f64 = 0.05;

/// Rows sampled for fitting ranges, and again for checking them.
const SAMPLE_ROWS: usize = 8192;

/// Two disjoint samples of row indexes, about `SAMPLE_ROWS` each: one to
/// fit code ranges on, one to check them on. Rows are picked by a hash of
/// their index (fixed seed, so deterministic), never by a stride, so no
/// periodic order of the input (alternating kinds of documents) can hide
/// a population from either sample. Small inputs use every row for both.
fn sample_rows(n: usize) -> (Vec<usize>, Vec<usize>) {
  if n <= 2 * SAMPLE_ROWS {
    let all: Vec<usize> = (0..n).collect();
    return (all.clone(), all);
  }
  let p = SAMPLE_ROWS as f64 / n as f64;
  let (mut fit, mut check) = (Vec::new(), Vec::new());
  for i in 0..n {
    let u = (splitmix64(i as u64 ^ 0x5851_f42d_4c95_7f2d) >> 11) as f64
      / (1u64 << 53) as f64;
    if u < p {
      fit.push(i);
    } else if u < 2.0 * p {
      check.push(i);
    }
  }
  (fit, check)
}

fn splitmix64(mut x: u64) -> u64 {
  x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
  x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  x ^ (x >> 31)
}

/// The non-NaN values of each dimension over `rows`.
fn columns(dim: usize, data: &[f32], rows: &[usize]) -> Vec<Vec<f32>> {
  let mut cols = vec![Vec::with_capacity(rows.len()); dim];
  for &r in rows {
    for (c, &x) in cols.iter_mut().zip(&data[r * dim..(r + 1) * dim]) {
      if !x.is_nan() {
        c.push(x);
      }
    }
  }
  cols
}

/// The `a` and `b` quantiles (`0 <= a <= b <= 1`, nearest rank) of
/// `v`; `(0, 0)` for an empty vector.
fn quantiles(mut v: Vec<f32>, a: f64, b: f64) -> (f32, f32) {
  if v.is_empty() {
    return (0.0, 0.0);
  }
  let rank =
    |p: f64| ((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
  let (ia, ib) = (rank(a), rank(b));
  let (_, &mut hi, _) = v.select_nth_unstable_by(ib, f32::total_cmp);
  let (_, &mut lo, _) = v[..=ib].select_nth_unstable_by(ia, f32::total_cmp);
  (lo, hi)
}

/// Step per dimension for ranges `lo..=hi`: `h / 255 - l / 255` rather
/// than `(h - l) / 255`, since the difference of two finite floats can
/// overflow to infinity, and an infinite step turns `inf * 0` into NaN.
fn steps(lo: &[f32], hi: &[f32]) -> Vec<f32> {
  lo.iter()
    .zip(hi)
    .map(|(&l, &h)| if h > l { h / 255.0 - l / 255.0 } else { 0.0 })
    .collect()
}

/// How coarse SQ8 codes with ranges `min + [0, 255] * step` are for the
/// values in `cols`: estimated mean squared coding error (`step^2 / 12`
/// inside the range plus the squared excess of bulk values outside it,
/// which saturate) summed over dimensions, divided by the summed robust
/// variance of the bulk (`(IQR / 1.349)^2`, which outliers cannot
/// inflate). The bulk is what lies inside the `clip` quantiles: codes
/// fitted on a narrower part of the data fail by saturating it, while
/// clipping the outliers is intended. Computed in `f64`; anything
/// non-finite counts as infinitely coarse.
fn error_ratio(min: &[f32], step: &[f32], cols: &[Vec<f32>], clip: f64) -> f64 {
  let (mut err, mut var) = (0f64, 0f64);
  for ((col, &m), &d) in cols.iter().zip(min).zip(step) {
    if col.is_empty() {
      continue;
    }
    let (lo, d) = (m as f64, d as f64);
    let hi = lo + 255.0 * d;
    let (b0, b1) = quantiles(col.clone(), clip, 1.0 - clip);
    let out: f64 = col
      .iter()
      .filter(|&&x| x >= b0 && x <= b1)
      .map(|&x| {
        let e = (lo - x as f64).max(x as f64 - hi).max(0.0);
        e * e
      })
      .sum();
    err += d * d / 12.0 + out / col.len() as f64;
    let (q1, q3) = quantiles(col.clone(), 0.25, 0.75);
    let iqr = (q3 as f64 - q1 as f64) / 1.349;
    var += iqr * iqr;
  }
  let r = if err == 0.0 { 0.0 } else { err / var };
  if r.is_finite() { r } else { f64::INFINITY }
}

/// Widens the quantile range `(a, b)` to the smallest and largest values
/// of `col` within `EXTEND` times its width beyond either end: the tail of
/// the bulk stays inside the codes, while values far beyond (outliers)
/// stay out.
fn extend(col: &[f32], (a, b): (f32, f32)) -> (f32, f32) {
  let w = (b as f64 - a as f64) * EXTEND;
  let (lo_lim, hi_lim) = (a as f64 - w, b as f64 + w);
  let (mut lo, mut hi) = (a, b);
  for &x in col {
    let v = x as f64;
    if v >= lo_lim && v < lo as f64 {
      lo = x;
    }
    if v <= hi_lim && v > hi as f64 {
      hi = x;
    }
  }
  (lo, hi)
}

/// How far beyond the clipping quantiles (in multiples of their distance)
/// values still count as the bulk's tail; see `extend`.
const EXTEND: f64 = 0.5;

/// SQ8 ranges for `data`: the first clipping level of `CLIPS` whose ranges
/// (its quantiles widened by `extend`, fitted on one hashed sample) pass
/// the `error_ratio` check on another.
/// Returns `(min, max, coarse)`; `coarse` when no level passes (the bulk
/// itself cannot be resolved by 256 levels), with the last level's
/// ranges.
fn fit_sq8(dim: usize, data: &[f32]) -> (Vec<f32>, Vec<f32>, bool) {
  let n = data.len() / dim;
  let (fit, check) = sample_rows(n);
  let fcols = columns(dim, data, &fit);
  let ccols = columns(dim, data, &check);
  let (mut lo, mut hi) = (vec![0.0f32; dim], vec![0.0f32; dim]);
  for clip in CLIPS {
    for (j, col) in fcols.iter().enumerate() {
      (lo[j], hi[j]) = extend(col, quantiles(col.clone(), clip, 1.0 - clip));
    }
    if error_ratio(&lo, &steps(&lo, &hi), &ccols, clip) <= MAX_CODE_ERROR {
      return (lo, hi, false);
    }
  }
  (lo, hi, true)
}

/// Retrain while the index is smaller than this whenever it has doubled
/// since the last training, so codes built from a handful of early
/// inserts do not clamp everything inserted later.
const RETRAIN_BELOW: usize = 1 << 16;

const T_MIN: [u8; 8] = tag("qmin");
const T_STEP: [u8; 8] = tag("qstep");
const T_CODES: [u8; 8] = tag("qcodes");
const T_SQ: [u8; 8] = tag("qsq");
const T_EXACT: [u8; 8] = tag("qexact");
const T_COARSE: [u8; 8] = tag("qcoarse");
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
  /// Bit per vector: its codes misrepresent it (values far outside the
  /// code range), so searches score it at full precision. Empty means
  /// none (all files before this field existed).
  exact: Slab<u64>,
  /// The codes cannot resolve the bulk of the data at all (see
  /// `fit_sq8`): every vector is scored at full precision.
  coarse: bool,
}

impl Sq8 {
  fn exact_count(&self) -> usize {
    self
      .exact
      .as_slice()
      .iter()
      .map(|w| w.count_ones() as usize)
      .sum()
  }
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
    if dim == 0 {
      // Nothing to encode; a zero-dimensional index scans exactly.
      return None;
    }
    let n = data.len() / dim;
    let mut q = match kind {
      Quantization::None => return None,
      Quantization::Sq8 => {
        let (lo, hi, coarse) = fit_sq8(dim, data);
        Self::sq8_with_range(dim, metric, lo, &hi, coarse)
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

  /// Untrained SQ8 parameters covering `lo[i]..=hi[i]` per dimension.
  fn sq8_with_range(
    dim: usize,
    metric: Metric,
    lo: Vec<f32>,
    hi: &[f32],
    coarse: bool,
  ) -> Self {
    let step = steps(&lo, hi);
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
      exact: Slab::default(),
      coarse,
    })
  }

  /// SQ8 codes for searching while a graph is built, or `None` when SQ8
  /// cannot stand in for the full vectors (the bulk is too coarse for 256
  /// levels, or too many vectors would need full precision): the build
  /// then searches full vectors. The check runs on the fitted ranges
  /// before anything is encoded.
  pub(crate) fn train_sq8_for_build(
    dim: usize,
    metric: Metric,
    data: &[f32],
  ) -> Option<Self> {
    if dim == 0 {
      return None;
    }
    let (lo, hi, coarse) = fit_sq8(dim, data);
    if coarse {
      return None;
    }
    let mut q = Self::sq8_with_range(dim, metric, lo, &hi, false);
    q.append(data);
    q.good_for_build().then_some(q)
  }

  /// Whether these are SQ8 codes that resolve the data well enough to
  /// search with while building: not coarse, and few enough vectors that
  /// need full precision (codes trained on an early part of the data mark
  /// later vectors outside their range that way).
  pub(crate) fn good_for_build(&self) -> bool {
    match self {
      Self::Sq8(s) => {
        !s.coarse
          && (s.exact_count() as f64) <= EXACT_MAX_FRACTION * self.len() as f64
      }
      Self::Binary(_) => false,
    }
  }

  /// Number of vectors scored at full precision instead of from codes.
  pub(crate) fn exact_rows(&self) -> usize {
    match self {
      Self::Sq8(s) if s.coarse => self.len(),
      Self::Sq8(s) => s.exact_count(),
      Self::Binary(_) => 0,
    }
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
        let first = s.codes.len().checked_div(s.dim).unwrap_or(0);
        let (min, step) = (s.min.as_slice(), s.step.as_slice());
        // Mean squared rounding error of one vector's codes.
        let noise: f64 =
          step.iter().map(|&d| (d as f64) * (d as f64) / 12.0).sum();
        let codes = s.codes.to_mut();
        let mut sq = Vec::new();
        let mut flagged = Vec::new();
        for (r, row) in rows.chunks_exact(s.dim).enumerate() {
          let mut acc = 0.0f32;
          let mut out = 0.0f64;
          for ((&x, &m), &d) in row.iter().zip(min).zip(step) {
            let c = if d > 0.0 {
              ((x - m) / d).round().clamp(0.0, 255.0) as u8
            } else {
              0
            };
            let (lo, hi) = (m as f64, m as f64 + 255.0 * d as f64);
            let e = (lo - x as f64).max(x as f64 - hi);
            if e > 0.0 {
              out += e * e;
            }
            let v = d * c as f32;
            acc += v * v;
            codes.push(c);
          }
          sq.push(acc);
          if !s.coarse && out > EXACT_FACTOR * noise {
            flagged.push(first + r);
          }
        }
        if s.l2 {
          s.sq.to_mut().extend_from_slice(&sq);
        }
        if !flagged.is_empty() || !s.exact.as_slice().is_empty() {
          let n = first + rows.len() / s.dim;
          let exact = s.exact.to_mut();
          exact.resize(n.div_ceil(64), 0);
          for i in flagged {
            exact[i / 64] |= 1 << (i % 64);
          }
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

  /// Renumbers the encoded vectors: new vector `i` is old `order[i]`.
  pub(crate) fn permute(&mut self, order: &[u32]) {
    use crate::hnsw::permute_rows;
    match self {
      Self::Sq8(s) => {
        permute_rows(s.codes.to_mut(), s.dim, order);
        if s.l2 {
          permute_rows(s.sq.to_mut(), 1, order);
        }
        let old = s.exact.as_slice();
        if !old.is_empty() {
          let mut exact = vec![0u64; old.len()];
          for (i, &o) in order.iter().enumerate() {
            let o = o as usize;
            if old.get(o / 64).is_some_and(|w| w >> (o % 64) & 1 == 1) {
              exact[i / 64] |= 1 << (i % 64);
            }
          }
          s.exact = Slab::owned(exact);
        }
      }
      Self::Binary(b) => permute_rows(b.bits.to_mut(), b.words, order),
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
    let n = data.len().checked_div(dim).unwrap_or(0);
    // Data that drifted out of the code range shows up as vectors marked
    // for full precision; retrain once they are many and the index has
    // grown by half since, so retraining stays amortised.
    let drifted = matches!(self, Self::Sq8(_))
      && self.exact_rows() as f64 > EXACT_MAX_FRACTION * n as f64
      && 2 * n >= 3 * trained_on.max(1);
    if drifted || (trained_on < RETRAIN_BELOW && n >= 2 * trained_on.max(1)) {
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
  /// `full` scores the vectors whose codes misrepresent them.
  pub(crate) fn dist<'a>(
    &'a self,
    qc: &'a QueryCodes,
    full: FullDist<'a>,
  ) -> CodeDist<'a> {
    match (self, qc) {
      (Self::Sq8(s), QueryCodes::Sq8 { w, base }) => CodeDist::Sq8(Sq8Dist {
        dim: s.dim,
        w,
        base: *base,
        codes: s.codes.as_slice(),
        sq: s.sq.as_slice(),
        exact: s.exact.as_slice(),
        coarse: s.coarse,
        full,
      }),
      (Self::Binary(b), QueryCodes::Binary { lut, base }) => {
        CodeDist::Bin(BinDist {
          row_bytes: b.words * 8,
          lut,
          base: *base,
          // Byte `j` of a little-endian `u64` row holds bits 8j..8j+8,
          // which the table layout relies on; muninn-core refuses to
          // compile for big-endian targets.
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
        w.pod_section(T_SQ, s.sq.as_slice())?;
        w.pod_section(T_EXACT, s.exact.as_slice())?;
        w.pod_section(T_COARSE, &[s.coarse as u64])
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
        // Absent in files written before these sections existed.
        exact: match c.try_section(T_EXACT) {
          Some(r) => Slab::from_region(r)?,
          None => Slab::default(),
        },
        coarse: match c.try_section(T_COARSE) {
          Some(r) => Slab::<u64>::from_region(r)?.as_slice() == [1],
          None => false,
        },
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
          && (s.exact.len() == 0 || s.exact.len() == n.div_ceil(64))
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
        s.min.bytes()
          + s.step.bytes()
          + s.codes.bytes()
          + s.sq.bytes()
          + s.exact.bytes()
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

/// SQ8 distance for one prepared query. Vectors marked in `exact` (or
/// all, when `coarse`) are scored at full precision with `full`.
pub(crate) struct Sq8Dist<'a> {
  dim: usize,
  w: &'a [f32],
  base: f32,
  codes: &'a [u8],
  sq: &'a [f32],
  exact: &'a [u64],
  coarse: bool,
  full: FullDist<'a>,
}

impl Sq8Dist<'_> {
  #[inline]
  fn is_exact(&self, i: usize) -> bool {
    self.coarse
      || self
        .exact
        .get(i / 64)
        .is_some_and(|w| w >> (i % 64) & 1 == 1)
  }
}

impl Dist for Sq8Dist<'_> {
  #[inline]
  fn dist(&self, id: DocId) -> f32 {
    let i = id as usize;
    if self.is_exact(i) {
      return self.full.dist(id);
    }
    let s = i * self.dim;
    let extra = self.sq.get(i).copied().unwrap_or(0.0);
    self.base + extra + dot_f32_u8(self.w, &self.codes[s..s + self.dim])
  }

  #[inline]
  fn prefetch(&self, id: DocId) {
    prefetch_lines(self.codes.as_ptr(), id as usize * self.dim, self.dim);
    if !self.sq.is_empty() {
      prefetch_lines(self.sq.as_ptr(), id as usize, 1);
    }
    if !self.exact.is_empty() {
      prefetch_lines(self.exact.as_ptr(), id as usize / 64, 1);
    }
  }

  /// Same values as `dist`, bit for bit (`a + b == b + a` in floats).
  #[inline]
  fn dist_many(&self, ids: &[DocId], out: &mut [f32]) {
    if self.coarse {
      return self.full.dist_many(ids, out);
    }
    muninn_simd::dot_f32_u8_gather(self.w, self.codes, ids, out);
    if self.exact.is_empty() {
      for (o, &id) in out.iter_mut().zip(ids) {
        *o += self.base + self.sq.get(id as usize).copied().unwrap_or(0.0);
      }
      return;
    }
    for (o, &id) in out.iter_mut().zip(ids) {
      let i = id as usize;
      *o = if self.exact[i / 64] >> (i % 64) & 1 == 1 {
        self.full.dist(id)
      } else {
        *o + (self.base + self.sq.get(i).copied().unwrap_or(0.0))
      };
    }
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

/// `sum w_i * c_i` with the codes widened to `f32`, on the dispatched
/// SIMD kernel.
#[inline]
pub(crate) fn dot_f32_u8(w: &[f32], c: &[u8]) -> f32 {
  muninn_simd::dot_f32_u8(w, c)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Review item 7: a dimension spanning more than `f32::MAX` must still
  /// get a finite step (an infinite one turns `inf * 0` into NaN).
  #[test]
  fn sq8_step_is_finite_for_huge_ranges() {
    let data = [-f32::MAX, 1.0, f32::MAX, 2.0, 0.0, 3.0];
    let qz =
      Quantizer::train(Quantization::Sq8, 2, Metric::L2, &data).expect("sq8");
    let Quantizer::Sq8(s) = &qz else {
      unreachable!()
    };
    assert!(s.step.as_slice().iter().all(|d| d.is_finite()), "{s:?}");
    assert!(s.sq.as_slice().iter().all(|x| !x.is_nan()));
    // The middle row decodes near zero in the huge dimension (decoded in
    // f64: `step * code` alone exceeds f32 there).
    let c = s.codes.as_slice()[4] as f64;
    let step = s.step.as_slice()[0] as f64;
    let deq = s.min.as_slice()[0] as f64 + step * c;
    assert!(deq.abs() <= step, "{deq} {step}");
  }

  fn full<'a>(metric: Metric, q: &'a [f32], data: &'a [f32]) -> FullDist<'a> {
    FullDist {
      metric,
      dim: q.len(),
      query: q,
      data,
    }
  }

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
        let got = with_code_dist!(qz.dist(&qc, full(metric, q, &data)), |x| {
          x.dist(id as DocId)
        });
        assert!((got - want).abs() < 1e-3, "{metric:?} {got} {want}");
        // Inside the code range, dequantisation error is at most half a
        // step per dimension (values beyond it saturate).
        let orig = &data[id * dim..(id + 1) * dim];
        for i in 0..dim {
          let (m, d) = (s.min.as_slice()[i], s.step.as_slice()[i]);
          if orig[i] >= m && orig[i] <= m + 255.0 * d {
            assert!((deq[i] - orig[i]).abs() <= d * 0.51);
          }
        }
      }
    }
  }

  /// Batched code distances equal single ones bit for bit, for every
  /// quantiser and metric.
  #[test]
  fn dist_many_equals_dist() {
    let (n, dim) = (40, 37);
    let data = rows(n, dim);
    let q = &rows(1, dim)[..];
    let ids: Vec<DocId> = (0..n as DocId).rev().chain([3, 3, 0]).collect();
    for kind in [Quantization::Sq8, Quantization::Binary] {
      for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        let qz = Quantizer::train(kind, dim, metric, &data).unwrap();
        let qc = qz.prepare(metric, q);
        with_code_dist!(qz.dist(&qc, full(metric, q, &data)), |d| {
          let mut out = vec![0.0f32; ids.len()];
          d.dist_many(&ids, &mut out);
          for (&id, &o) in ids.iter().zip(&out) {
            assert_eq!(
              o.to_bits(),
              d.dist(id).to_bits(),
              "{kind:?} {metric:?}"
            );
          }
        });
      }
    }
  }

  /// Ranges ignore a single extreme row and keep the bulk's resolution;
  /// that row is scored at full precision; with many outliers a wider
  /// clip is used; the check rejects ranges that saturate the bulk, and
  /// values near `f32::MAX` cannot sneak past it as infinities.
  #[test]
  fn fitted_ranges_exact_rows_and_error_ratio() {
    let (n, dim) = (5000, 8);
    let mut data = rows(n, dim);
    for j in 0..dim {
      data[17 * dim + j] = 1e4;
    }
    let qz =
      Quantizer::train(Quantization::Sq8, dim, Metric::L2, &data).unwrap();
    let Quantizer::Sq8(s) = &qz else {
      unreachable!()
    };
    assert!(!s.coarse);
    assert!(s.step.as_slice().iter().all(|&d| d < 0.01), "{s:?}");
    assert_eq!(qz.exact_rows(), 1);
    assert!(s.exact.as_slice()[0] >> 17 & 1 == 1);
    assert!(qz.good_for_build());
    // The exact row is scored at full precision.
    let q = vec![1e4f32; dim];
    let qc = qz.prepare(Metric::L2, &q);
    let d = with_code_dist!(qz.dist(&qc, full(Metric::L2, &q, &data)), |x| {
      x.dist(17)
    });
    assert_eq!(d, 0.0);
    // 2% outliers at both ends: a wider clip resolves the bulk, and the
    // outliers are exact.
    for i in (0..n).step_by(50) {
      for j in 0..dim {
        data[i * dim + j] = if (i / 50) % 2 == 0 { -1e4 } else { 1e4 };
      }
    }
    let qz =
      Quantizer::train(Quantization::Sq8, dim, Metric::L2, &data).unwrap();
    let Quantizer::Sq8(s) = &qz else {
      unreachable!()
    };
    assert!(!s.coarse && s.step.as_slice().iter().all(|&d| d < 0.01));
    assert_eq!(qz.exact_rows(), 101);
    assert!(qz.good_for_build());
    // Ranges fitted on a narrow early part saturate the bulk of later
    // data: rejected, and later rows come out exact.
    let later = rows(n, dim);
    let early: Vec<f32> = later[..100 * dim].iter().map(|x| x * 0.05).collect();
    let (lo, hi, _) = fit_sq8(dim, &early);
    let ccols = columns(dim, &later, &(0..n).collect::<Vec<_>>());
    assert!(error_ratio(&lo, &steps(&lo, &hi), &ccols, CLIPS[0]) > 0.1);
    let mut narrow =
      Quantizer::train(Quantization::Sq8, dim, Metric::L2, &early).unwrap();
    narrow.append(&later);
    assert!(narrow.exact_rows() > n / 2);
    // Near f32::MAX the spread `q3 - q1` overflows f32; an infinite
    // spread made the ratio 0 and accepted ranges that saturate the whole
    // bulk. In f64 it is finite and the ratio rejects them; anything
    // non-finite rejects too.
    let big = vec![-f32::MAX, f32::MAX, -f32::MAX, f32::MAX];
    let c = columns(1, &big, &[0, 1, 2, 3]);
    let r = error_ratio(&[0.0], &[1.0], &c, CLIPS[0]);
    assert!(r > MAX_CODE_ERROR, "{r}");
    assert_eq!(
      error_ratio(&[0.0], &[f32::INFINITY], &c, CLIPS[0]),
      f64::INFINITY
    );
    assert_eq!(quantiles(vec![], 0.1, 0.9), (0.0, 0.0));
    assert_eq!(quantiles(vec![3.0, 1.0, 2.0], 0.0, 1.0), (1.0, 3.0));
  }

  /// Samples are hashed, not strided: with two populations interleaved,
  /// both samples hold both (stride 2 would have seen only one).
  #[test]
  fn samples_see_interleaved_populations() {
    let n = 40_000;
    let (fit, check) = sample_rows(n);
    for s in [&fit, &check] {
      let odd = s.iter().filter(|&&r| r % 2 == 1).count();
      assert!(s.len() > 7000 && s.len() < 9500, "{}", s.len());
      assert!(odd * 3 > s.len() && odd * 3 < 2 * s.len(), "{odd}");
    }
    assert!(fit.iter().all(|r| check.binary_search(r).is_err()));
    assert_eq!(sample_rows(10), ((0..10).collect(), (0..10).collect()));
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
        let got = with_code_dist!(qz.dist(&qc, full(metric, q, &data)), |x| {
          x.dist(id as DocId)
        });
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
