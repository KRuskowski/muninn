//! Distance kernels.
//!
//! This is the public API every index calls on its hottest path. Each
//! function picks the fastest implementation the running CPU supports
//! (AVX-512, AVX2+FMA, NEON, scalar) once, then calls it through a cached
//! function pointer. Hot loops can hoist even that lookup out with
//! [`kernels`], which returns the resolved table of function pointers.
//!
//! The portable implementations in [`scalar`] are the reference every SIMD
//! implementation is tested against.
//!
//! Contracts, for every function:
//! - slices passed together have equal length (checked with
//!   `debug_assert!`, not in release builds; in release builds a mismatch
//!   never reads out of bounds, the kernels use the shorter length);
//! - any length is accepted, including 0 and non-multiples of the vector
//!   width;
//! - no alignment is required; aligned input may simply be faster.
//!
//! Within one process all functions use the same instruction set, and
//! [`distance_batch`] returns bit-for-bit what [`distance`] returns for
//! each row, so results do not depend on how a scan is split into
//! batches. Results may differ between instruction sets in the last bits
//! (floating-point reassociation); integer kernels are exact everywhere.
//!
//! The environment variable `MUNINN_ISA` (`scalar`, `avx2`, `avx512` or
//! `neon`) caps the instruction set, for tests and benchmarks. It is read
//! once, on first use. See [`Isa`] for the fallback rule.
//!
//! Half-precision vectors are passed as IEEE 754 binary16 bit patterns in
//! `u16`; see [`dot_f16`], [`f32_to_f16_slice`] and friends.

use muninn_core::Metric;

mod dispatch;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(test)]
mod tests;

pub use dispatch::{GatherFn, GatherU8Fn, Isa, Kernels, kernels, kernels_for};

/// Dot product of two `f32` vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().dot)(a, b)
}

/// Squared Euclidean distance between two `f32` vectors.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().l2sq)(a, b)
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

/// Converts a raw kernel result (squared L2 for `Metric::L2`, the dot
/// product otherwise) into the internal distance for `metric`.
#[inline]
pub(crate) fn finish(metric: Metric, raw: f32) -> f32 {
  match metric {
    Metric::L2 => raw,
    Metric::Dot => -raw,
    Metric::Cosine => 1.0 - raw,
  }
}

/// Internal distance for `metric` (lower is closer), see [`Metric`].
/// Cosine expects both inputs already normalised.
#[inline]
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
  match metric {
    Metric::L2 => l2sq(a, b),
    Metric::Dot | Metric::Cosine => finish(metric, dot(a, b)),
  }
}

/// Distances from one query to `out.len()` consecutive rows of `dim`
/// floats in `rows`: `out[i] = distance(metric, query, row_i)`.
///
/// The batched form exists so implementations can prefetch and keep the
/// query in registers across rows; flat scans and re-ranking use it. Each
/// `out[i]` is bit-identical to [`distance`] for that row.
#[inline]
pub fn distance_batch(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  out: &mut [f32],
) {
  debug_assert_eq!(rows.len(), query.len() * out.len());
  (kernels().distance_batch)(metric, query, rows, out)
}

/// Distances from one query to rows picked by id from a row-major array:
/// `out[i] = distance(metric, query, row ids[i])`, where row `r` is
/// `rows[r * query.len()..(r + 1) * query.len()]`.
///
/// This is graph search's inner loop: the unvisited neighbours of a node
/// are scored together, several rows at a time with the query loaded
/// once, while the rows of the next group are prefetched. Each `out[i]` is
/// bit-identical to [`distance`] for that row.
///
/// # Panics
/// If an id does not name a whole row of `rows`, or `out` is shorter than
/// `ids`.
#[inline]
pub fn distance_gather(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  ids: &[u32],
  out: &mut [f32],
) {
  let dim = query.len();
  let n = rows.len().checked_div(dim).unwrap_or(usize::MAX);
  assert!(out.len() >= ids.len(), "output shorter than ids");
  assert!(
    ids.iter().all(|&id| (id as usize) < n),
    "row id out of range"
  );
  (kernels().distance_gather)(metric, query, rows, ids, &mut out[..ids.len()])
}

/// Dot product of `f32` weights with `u8` codes, each code widened to
/// `f32`: `sum a_i * b_i`. This is the inner loop of asymmetric SQ8
/// distances, where the query is kept in `f32` and each dimension's scale
/// and offset are folded into the weights `a`, so codes are never decoded.
/// Codes convert to `f32` exactly; results differ from the scalar
/// reference only by float reassociation.
#[inline]
pub fn dot_f32_u8(a: &[f32], b: &[u8]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().dot_f32_u8)(a, b)
}

/// [`dot_f32_u8`] of `weights` with code rows picked by id: `out[i] =
/// dot_f32_u8(weights, row ids[i])`, where row `r` is
/// `codes[r * weights.len()..(r + 1) * weights.len()]`. Graph search over
/// SQ8 codes scores a node's unvisited neighbours this way, several rows
/// per step with the weights loaded once. Each `out[i]` is bit-identical
/// to [`dot_f32_u8`] for that row.
///
/// # Panics
/// If an id does not name a whole row of `codes`, or `out` is shorter
/// than `ids`.
#[inline]
pub fn dot_f32_u8_gather(
  weights: &[f32],
  codes: &[u8],
  ids: &[u32],
  out: &mut [f32],
) {
  let dim = weights.len();
  let n = codes.len().checked_div(dim).unwrap_or(usize::MAX);
  assert!(out.len() >= ids.len(), "output shorter than ids");
  assert!(
    ids.iter().all(|&id| (id as usize) < n),
    "row id out of range"
  );
  (kernels().dot_f32_u8_gather)(weights, codes, ids, &mut out[..ids.len()])
}

/// Dot product of two `i8` vectors, accumulated exactly in `i32`.
/// Used by scalar-quantised (SQ8) indexes. Exact whenever the result fits
/// in `i32`, which holds for any input of length below 2^17; beyond that
/// every instruction set returns the same sum wrapped modulo 2^32.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().dot_i8)(a, b)
}

/// Squared L2 distance of two `u8` vectors, accumulated in `u32`. Exact
/// whenever the result fits in `u32`, which holds for any input of up to
/// 66,051 elements (each term is at most 255^2); beyond that every
/// instruction set returns the same sum wrapped modulo 2^32.
#[inline]
pub fn l2sq_u8(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().l2sq_u8)(a, b)
}

/// Hamming distance between two bit vectors packed into `u64` words.
/// Used by binary-quantised indexes. Exact for fewer than 2^26 words
/// (2^32 bits); beyond that the count wraps modulo 2^32 on every
/// instruction set.
#[inline]
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().hamming)(a, b)
}

/// Dot product of two half-precision vectors (binary16 bit patterns),
/// computed in `f32`.
#[inline]
pub fn dot_f16(a: &[u16], b: &[u16]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().dot_f16)(a, b)
}

/// Squared Euclidean distance between two half-precision vectors
/// (binary16 bit patterns), computed in `f32`.
#[inline]
pub fn l2sq_f16(a: &[u16], b: &[u16]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  (kernels().l2sq_f16)(a, b)
}

/// Converts one binary16 bit pattern to `f32`. Exact.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
  scalar::f16_to_f32(h)
}

/// Converts an `f32` to the nearest binary16 bit pattern (round to
/// nearest, ties to even). Values beyond the binary16 range become
/// infinity; NaN stays NaN.
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
  scalar::f32_to_f16(x)
}

/// Converts binary16 bit patterns to `f32`: `dst[i] = f16_to_f32(src[i])`.
pub fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
  debug_assert_eq!(src.len(), dst.len());
  (kernels().f16_to_f32)(src, dst)
}

/// Converts `f32` values to binary16 bit patterns:
/// `dst[i] = f32_to_f16(src[i])`. Bit-identical to [`f32_to_f16`] except
/// that NaN payloads may differ.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
  debug_assert_eq!(src.len(), dst.len());
  (kernels().f32_to_f16)(src, dst)
}

/// Name of the instruction set the dispatched kernels use, for logs and
/// benchmark reports: `"avx512"`, `"avx2"`, `"neon"` or `"scalar"`.
pub fn active_isa() -> &'static str {
  kernels().isa.name()
}
