//! AVX2 + FMA + F16C kernels for x86-64.
//!
//! The private wrappers stored in [`KERNELS`] call `#[target_feature]`
//! functions; [`kernels`] hands the table out only after [`detected`] has
//! confirmed the CPU supports every feature used here, so the wrappers are
//! never reachable on a CPU without them.

// Index loops over small const-size arrays read better than iterator
// chains in SIMD code, and unroll to the same machine code.
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::*;

use muninn_core::Metric;

use crate::dispatch::{Isa, Kernels};
use crate::{finish, scalar};

/// Rows per iteration in `distance_batch`: 2 rows x 4 accumulators keep 8
/// of the 16 ymm registers for sums, leaving room for the query and rows.
const BATCH_ROWS: usize = 2;

fn detected() -> bool {
  is_x86_feature_detected!("avx2")
    && is_x86_feature_detected!("fma")
    && is_x86_feature_detected!("f16c")
    && is_x86_feature_detected!("popcnt")
}

static KERNELS: Kernels = Kernels {
  isa: Isa::Avx2,
  features: "avx2+fma+f16c+popcnt",
  dot,
  l2sq,
  distance_batch,
  dot_i8,
  l2sq_u8,
  hamming,
  dot_f16,
  l2sq_f16,
  f16_to_f32: f16_to_f32_slice,
  f32_to_f16: f32_to_f16_slice,
};

/// The AVX2 kernels, if the running CPU supports them.
pub(crate) fn kernels() -> Option<&'static Kernels> {
  detected().then_some(&KERNELS)
}

// Safe wrappers. Each is reachable only through `KERNELS`, which
// `kernels()` returns only after `detected()` confirmed AVX2, FMA, F16C and
// POPCNT; that is the safety argument for every call below.

fn dot(a: &[f32], b: &[f32]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  let n = a.len().min(b.len());
  // SAFETY: CPU features confirmed (see above); both pointers are valid
  // for `n` reads.
  unsafe { rows_f32::<1, false, false>(a.as_ptr(), [b.as_ptr()], n, 0)[0] }
}

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  let n = a.len().min(b.len());
  // SAFETY: CPU features confirmed (see above); both pointers are valid
  // for `n` reads.
  unsafe { rows_f32::<1, true, false>(a.as_ptr(), [b.as_ptr()], n, 0)[0] }
}

fn distance_batch(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  out: &mut [f32],
) {
  let dim = query.len();
  debug_assert_eq!(rows.len(), dim * out.len());
  if dim == 0 {
    out.fill(finish(metric, 0.0));
    return;
  }
  let n = out.len().min(rows.len() / dim);
  let out = &mut out[..n];
  // SAFETY: CPU features confirmed (see above).
  unsafe {
    if metric == Metric::L2 {
      batch::<true>(metric, query, rows, out)
    } else {
      batch::<false>(metric, query, rows, out)
    }
  }
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { dot_i8_impl(a, b) }
}

fn l2sq_u8(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { l2sq_u8_impl(a, b) }
}

fn hamming(a: &[u64], b: &[u64]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { hamming_impl(a, b) }
}

fn dot_f16(a: &[u16], b: &[u16]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { f16_impl::<false>(a, b) }
}

fn l2sq_f16(a: &[u16], b: &[u16]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { f16_impl::<true>(a, b) }
}

fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
  debug_assert_eq!(src.len(), dst.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { f16_to_f32_impl(src, dst) }
}

fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
  debug_assert_eq!(src.len(), dst.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { f32_to_f16_impl(src, dst) }
}

// Helpers.

/// `acc + q * r`, or `acc + (q - r)^2` when `L2`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn step<const L2: bool>(acc: __m256, q: __m256, r: __m256) -> __m256 {
  if L2 {
    let d = _mm256_sub_ps(q, r);
    _mm256_fmadd_ps(d, d, acc)
  } else {
    _mm256_fmadd_ps(q, r, acc)
  }
}

/// Horizontal sum in a fixed order.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_ps(v: __m256) -> f32 {
  let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
  let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  let s = _mm_add_ss(s, _mm_movehdup_ps(s));
  _mm_cvtss_f32(s)
}

/// Sum of the four accumulators, then horizontal sum.
#[inline]
#[target_feature(enable = "avx2")]
fn reduce4(acc: [__m256; 4]) -> f32 {
  let s01 = _mm256_add_ps(acc[0], acc[1]);
  let s23 = _mm256_add_ps(acc[2], acc[3]);
  hsum_ps(_mm256_add_ps(s01, s23))
}

/// Wrapping horizontal sum of eight `i32` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_epi32(v: __m256i) -> i32 {
  let s =
    _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
  let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
  let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
  _mm_cvtsi128_si32(s)
}

/// Prefetches the cache line holding `p` into L1. `p` need not be valid.
#[inline]
#[target_feature(enable = "avx2")]
fn prefetch(p: *const f32) {
  // SAFETY: a prefetch is only a hint; it never faults or reads memory
  // architecturally, whatever the address.
  unsafe { _mm_prefetch::<_MM_HINT_T0>(p.cast()) }
}

/// Lanes `0..rem` set, for `_mm256_maskload_ps`; `rem` in `1..8`.
#[inline]
#[target_feature(enable = "avx2")]
fn tail_mask(rem: usize) -> __m256i {
  static TAIL: [i32; 16] =
    [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];
  debug_assert!((1..8).contains(&rem));
  // SAFETY: `8 - rem` is in `1..=7`, so the eight loaded `i32` lie inside
  // `TAIL`.
  unsafe { _mm256_loadu_si256(TAIL.as_ptr().add(8 - rem).cast()) }
}

/// The f32 kernel shared by `dot`, `l2sq` and `distance_batch`: the dot
/// product (or squared L2 distance when `L2`) of `q` with each of `R`
/// rows. Every row is accumulated in exactly the same order whatever `R`
/// is, so batched and single-row results are bit-identical. With `PF`,
/// prefetches the row data `pf` elements ahead of each row.
///
/// # Safety
/// The CPU supports AVX2 and FMA; `q` and every row pointer are valid for
/// `n` reads.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_f32<const R: usize, const L2: bool, const PF: bool>(
  q: *const f32,
  rows: [*const f32; R],
  n: usize,
  pf: usize,
) -> [f32; R] {
  let mut acc = [[_mm256_setzero_ps(); 4]; R];
  let mut j = 0;
  while j + 32 <= n {
    for u in 0..4 {
      // SAFETY: `j + 8 * u + 8 <= n` and every pointer is valid for `n`
      // reads.
      let qv = unsafe { _mm256_loadu_ps(q.add(j + 8 * u)) };
      for r in 0..R {
        // SAFETY: as above.
        let rv = unsafe { _mm256_loadu_ps(rows[r].add(j + 8 * u)) };
        acc[r][u] = step::<L2>(acc[r][u], qv, rv);
      }
    }
    if PF {
      for r in 0..R {
        prefetch(rows[r].wrapping_add(pf + j));
        prefetch(rows[r].wrapping_add(pf + j + 16));
      }
    }
    j += 32;
  }
  // At most three full vectors remain; give each its own accumulator.
  // Constant accumulator indices (not a running counter) keep the
  // accumulators in registers.
  for u in 0..3 {
    if j + 8 <= n {
      // SAFETY: `j + 8 <= n`.
      let qv = unsafe { _mm256_loadu_ps(q.add(j)) };
      for r in 0..R {
        // SAFETY: `j + 8 <= n`.
        let rv = unsafe { _mm256_loadu_ps(rows[r].add(j)) };
        acc[r][u] = step::<L2>(acc[r][u], qv, rv);
      }
      j += 8;
    }
  }
  if j < n {
    let mask = tail_mask(n - j);
    // SAFETY: `maskload` accesses only the `n - j` lanes whose mask bit is
    // set, all in bounds; masked-off lanes are never accessed and cannot
    // fault.
    let qv = unsafe { _mm256_maskload_ps(q.add(j), mask) };
    for r in 0..R {
      // SAFETY: as above.
      let rv = unsafe { _mm256_maskload_ps(rows[r].add(j), mask) };
      acc[r][3] = step::<L2>(acc[r][3], qv, rv);
    }
  }
  let mut out = [0.0; R];
  for r in 0..R {
    out[r] = reduce4(acc[r]);
  }
  out
}

/// `out[i] = finish(metric, kernel(query, row_i))` for every row.
/// Requires `rows.len() >= query.len() * out.len()` and a non-empty query.
#[target_feature(enable = "avx2,fma")]
fn batch<const L2: bool>(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  out: &mut [f32],
) {
  let dim = query.len();
  assert!(dim > 0 && rows.len() / dim >= out.len());
  let q = query.as_ptr();
  let base = rows.as_ptr();
  let mut i = 0;
  while i + BATCH_ROWS <= out.len() {
    let ptrs: [*const f32; BATCH_ROWS] =
      core::array::from_fn(|r| base.wrapping_add((i + r) * dim));
    // SAFETY: rows `i..i + BATCH_ROWS` lie inside `rows` (asserted above);
    // the prefetch target need not be valid.
    let d = unsafe {
      rows_f32::<BATCH_ROWS, L2, true>(q, ptrs, dim, BATCH_ROWS * dim)
    };
    for r in 0..BATCH_ROWS {
      out[i + r] = finish(metric, d[r]);
    }
    i += BATCH_ROWS;
  }
  while i < out.len() {
    // SAFETY: row `i` lies inside `rows` (asserted above).
    let d = unsafe {
      rows_f32::<1, L2, false>(q, [base.wrapping_add(i * dim)], dim, 0)
    };
    out[i] = finish(metric, d[0]);
    i += 1;
  }
}

/// Exact `i8` dot product: sign-extend to `i16`, `madd` pairs into `i32`.
/// (`maddubs` would saturate at the -128 * -128 extremes.)
#[target_feature(enable = "avx2")]
pub(crate) fn dot_i8_impl(a: &[i8], b: &[i8]) -> i32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = _mm256_setzero_si256();
  let mut acc1 = _mm256_setzero_si256();
  let mut j = 0;
  while j + 32 <= n {
    // SAFETY: `j + 32 <= n`, so all four 16-byte loads are in bounds.
    let (a0, b0, a1, b1) = unsafe {
      (
        _mm_loadu_si128(pa.add(j).cast()),
        _mm_loadu_si128(pb.add(j).cast()),
        _mm_loadu_si128(pa.add(j + 16).cast()),
        _mm_loadu_si128(pb.add(j + 16).cast()),
      )
    };
    let p0 =
      _mm256_madd_epi16(_mm256_cvtepi8_epi16(a0), _mm256_cvtepi8_epi16(b0));
    let p1 =
      _mm256_madd_epi16(_mm256_cvtepi8_epi16(a1), _mm256_cvtepi8_epi16(b1));
    acc0 = _mm256_add_epi32(acc0, p0);
    acc1 = _mm256_add_epi32(acc1, p1);
    j += 32;
  }
  if j + 16 <= n {
    // SAFETY: `j + 16 <= n`.
    let (a0, b0) = unsafe {
      (
        _mm_loadu_si128(pa.add(j).cast()),
        _mm_loadu_si128(pb.add(j).cast()),
      )
    };
    let p0 =
      _mm256_madd_epi16(_mm256_cvtepi8_epi16(a0), _mm256_cvtepi8_epi16(b0));
    acc0 = _mm256_add_epi32(acc0, p0);
    j += 16;
  }
  let mut s = hsum_epi32(_mm256_add_epi32(acc0, acc1));
  for k in j..n {
    s = s.wrapping_add(a[k] as i32 * b[k] as i32);
  }
  s
}

/// Exact `u8` squared L2: `|a - b|` as `u8`, widen to `i16`, `madd`.
#[target_feature(enable = "avx2")]
pub(crate) fn l2sq_u8_impl(a: &[u8], b: &[u8]) -> u32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let zero = _mm256_setzero_si256();
  let mut acc0 = zero;
  let mut acc1 = zero;
  let mut j = 0;
  while j + 32 <= n {
    // SAFETY: `j + 32 <= n`.
    let (av, bv) = unsafe {
      (
        _mm256_loadu_si256(pa.add(j).cast()),
        _mm256_loadu_si256(pb.add(j).cast()),
      )
    };
    let d = _mm256_sub_epi8(_mm256_max_epu8(av, bv), _mm256_min_epu8(av, bv));
    let lo = _mm256_unpacklo_epi8(d, zero);
    let hi = _mm256_unpackhi_epi8(d, zero);
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(lo, lo));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(hi, hi));
    j += 32;
  }
  let mut s = hsum_epi32(_mm256_add_epi32(acc0, acc1)) as u32;
  for k in j..n {
    let d = a[k] as i32 - b[k] as i32;
    s = s.wrapping_add((d * d) as u32);
  }
  s
}

/// Per-byte popcount via a nibble lookup table.
#[inline]
#[target_feature(enable = "avx2")]
fn popcnt_bytes(x: __m256i, lut: __m256i, low: __m256i) -> __m256i {
  let lo = _mm256_and_si256(x, low);
  let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), low);
  _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi))
}

#[target_feature(enable = "avx2,popcnt")]
fn hamming_impl(a: &[u64], b: &[u64]) -> u32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  #[rustfmt::skip]
  let lut = _mm256_setr_epi8(
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
  );
  let low = _mm256_set1_epi8(0x0f);
  let zero = _mm256_setzero_si256();
  let mut acc = zero;
  let mut j = 0;
  while j + 8 <= n {
    // SAFETY: `j + 8 <= n`, so the four 4-word loads are in bounds.
    let (x0, x1) = unsafe {
      (
        _mm256_xor_si256(
          _mm256_loadu_si256(pa.add(j).cast()),
          _mm256_loadu_si256(pb.add(j).cast()),
        ),
        _mm256_xor_si256(
          _mm256_loadu_si256(pa.add(j + 4).cast()),
          _mm256_loadu_si256(pb.add(j + 4).cast()),
        ),
      )
    };
    // At most 16 per byte, so the byte sum cannot overflow.
    let c =
      _mm256_add_epi8(popcnt_bytes(x0, lut, low), popcnt_bytes(x1, lut, low));
    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(c, zero));
    j += 8;
  }
  let lanes =
    _mm256_add_epi64(acc, _mm256_permute4x64_epi64::<0b01_00_11_10>(acc));
  let mut s = (_mm256_extract_epi64::<0>(lanes) as u64)
    .wrapping_add(_mm256_extract_epi64::<1>(lanes) as u64);
  for k in j..n {
    s += (a[k] ^ b[k]).count_ones() as u64;
  }
  s as u32
}

/// Loads eight binary16 values and widens them to `f32`.
///
/// # Safety
/// The CPU supports AVX and F16C; `p` is valid for 8 reads.
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn load_f16(p: *const u16) -> __m256 {
  // SAFETY: guaranteed by the caller.
  _mm256_cvtph_ps(unsafe { _mm_loadu_si128(p.cast()) })
}

#[target_feature(enable = "avx2,fma,f16c")]
fn f16_impl<const L2: bool>(a: &[u16], b: &[u16]) -> f32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc = [_mm256_setzero_ps(); 4];
  let mut j = 0;
  while j + 32 <= n {
    for u in 0..4 {
      // SAFETY: `j + 8 * u + 8 <= n`.
      let (x, y) =
        unsafe { (load_f16(pa.add(j + 8 * u)), load_f16(pb.add(j + 8 * u))) };
      acc[u] = step::<L2>(acc[u], x, y);
    }
    j += 32;
  }
  // Constant accumulator indices (not a running counter) keep the
  // accumulators in registers.
  for u in 0..3 {
    if j + 8 <= n {
      // SAFETY: `j + 8 <= n`.
      let (x, y) = unsafe { (load_f16(pa.add(j)), load_f16(pb.add(j))) };
      acc[u] = step::<L2>(acc[u], x, y);
      j += 8;
    }
  }
  if j < n {
    let (mut ta, mut tb) = ([0u16; 8], [0u16; 8]);
    ta[..n - j].copy_from_slice(&a[j..n]);
    tb[..n - j].copy_from_slice(&b[j..n]);
    // SAFETY: both buffers hold 8 values. Zero padding adds nothing.
    let (x, y) = unsafe { (load_f16(ta.as_ptr()), load_f16(tb.as_ptr())) };
    acc[3] = step::<L2>(acc[3], x, y);
  }
  reduce4(acc)
}

#[target_feature(enable = "avx2,f16c")]
fn f16_to_f32_impl(src: &[u16], dst: &mut [f32]) {
  let n = src.len().min(dst.len());
  let mut j = 0;
  while j + 8 <= n {
    // SAFETY: `j + 8 <= n` for both slices.
    unsafe {
      let v = load_f16(src.as_ptr().add(j));
      _mm256_storeu_ps(dst.as_mut_ptr().add(j), v);
    }
    j += 8;
  }
  for k in j..n {
    dst[k] = scalar::f16_to_f32(src[k]);
  }
}

#[target_feature(enable = "avx2,f16c")]
fn f32_to_f16_impl(src: &[f32], dst: &mut [u16]) {
  let n = src.len().min(dst.len());
  let mut j = 0;
  while j + 8 <= n {
    // SAFETY: `j + 8 <= n` for both slices.
    unsafe {
      let v = _mm256_loadu_ps(src.as_ptr().add(j));
      let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
      _mm_storeu_si128(dst.as_mut_ptr().add(j).cast(), h);
    }
    j += 8;
  }
  for k in j..n {
    dst[k] = scalar::f32_to_f16(src[k]);
  }
}
