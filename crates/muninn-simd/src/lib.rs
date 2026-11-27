//! Distance kernels.
//!
//! This is the public API every index calls on its hottest path. Each
//! function picks the fastest implementation the running CPU supports
//! (AVX-512, AVX2+FMA, NEON, scalar) once, then calls it through a cached
//! function pointer.
//!
//! The scaffold ships only the scalar implementations in [`scalar`]; they
//! are also the reference every SIMD implementation is tested against.
//!
//! Contracts, for every function:
//! - slices passed together have equal length (checked with
//!   `debug_assert!`, not in release builds);
//! - any length is accepted, including 0 and non-multiples of the vector
//!   width;
//! - no alignment is required; aligned input may simply be faster.

use muninn_core::Metric;

pub mod scalar;

/// Dot product of two `f32` vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
  scalar::dot(a, b)
}

/// Squared Euclidean distance between two `f32` vectors.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
  scalar::l2sq(a, b)
}

/// Squared L2 norm of a vector.
#[inline]
pub fn norm_sq(a: &[f32]) -> f32 {
  dot(a, a)
}

/// Scales `a` to unit length in place. Zero vectors are left unchanged.
pub fn normalize(a: &mut [f32]) {
  let n = norm_sq(a).sqrt();
  if n > 0.0 {
    let inv = 1.0 / n;
    a.iter_mut().for_each(|x| *x *= inv);
  }
}

/// Internal distance for `metric` (lower is closer), see [`Metric`].
/// Cosine expects both inputs already normalised.
#[inline]
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
  match metric {
    Metric::L2 => l2sq(a, b),
    Metric::Dot => -dot(a, b),
    Metric::Cosine => 1.0 - dot(a, b),
  }
}

/// Distances from one query to `out.len()` consecutive rows of `dim`
/// floats in `rows`: `out[i] = distance(metric, query, row_i)`.
///
/// The batched form exists so implementations can prefetch and keep the
/// query in registers across rows; flat scans and re-ranking use it.
pub fn distance_batch(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  out: &mut [f32],
) {
  let dim = query.len();
  debug_assert_eq!(rows.len(), dim * out.len());
  if dim == 0 {
    out.fill(distance(metric, query, query));
    return;
  }
  for (o, row) in out.iter_mut().zip(rows.chunks_exact(dim)) {
    *o = distance(metric, query, row);
  }
}

/// Dot product of two `i8` vectors, accumulated exactly in `i32`.
/// Used by scalar-quantised (SQ8) indexes. Exact for lengths up to 2^17.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
  scalar::dot_i8(a, b)
}

/// Squared L2 distance of two `u8` vectors, exact in `u32`.
#[inline]
pub fn l2sq_u8(a: &[u8], b: &[u8]) -> u32 {
  scalar::l2sq_u8(a, b)
}

/// Hamming distance between two bit vectors packed into `u64` words.
/// Used by binary-quantised indexes.
#[inline]
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
  scalar::hamming(a, b)
}

/// Name of the instruction set the dispatched kernels use, for logs and
/// benchmark reports: `"avx512"`, `"avx2"`, `"neon"` or `"scalar"`.
pub fn active_isa() -> &'static str {
  "scalar"
}
