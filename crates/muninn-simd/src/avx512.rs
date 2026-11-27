//! AVX-512 kernels for x86-64.
//!
//! Base requirement: AVX-512 F and BW (plus AVX2, FMA, F16C and POPCNT,
//! which every such CPU has). AVX-512 VNNI and VPOPCNTDQ select faster
//! integer kernels when present. [`kernels`] builds the table only after
//! runtime detection, so no wrapper stored in it is reachable on a CPU
//! without the features it uses.

// Index loops over small const-size arrays read better than iterator
// chains in SIMD code, and unroll to the same machine code.
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::*;
use std::sync::OnceLock;

use muninn_core::Metric;

use crate::dispatch::{Isa, Kernels};
use crate::{finish, scalar};

/// Accumulators per row in the f32 kernel. FMA has a 4-cycle latency and
/// two ports, so eight independent sums keep both busy.
const ACC: usize = 4;

/// Rows per iteration in `distance_batch`: 2 rows x 8 accumulators use 16
/// of the 32 zmm registers for sums; the query vector is loaded once per
/// pair of rows.
const BATCH_ROWS: usize = 4;

/// Optional features detected on top of the base requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extras {
  pub vnni: bool,
  pub vpopcntdq: bool,
}

fn base_detected() -> bool {
  is_x86_feature_detected!("avx512f")
    && is_x86_feature_detected!("avx512bw")
    && is_x86_feature_detected!("avx2")
    && is_x86_feature_detected!("fma")
    && is_x86_feature_detected!("f16c")
    && is_x86_feature_detected!("popcnt")
}

/// The optional features the running CPU has, or `None` if it lacks the
/// base requirement.
pub(crate) fn detected() -> Option<Extras> {
  base_detected().then(|| Extras {
    vnni: is_x86_feature_detected!("avx512vnni"),
    vpopcntdq: is_x86_feature_detected!("avx512vpopcntdq"),
  })
}

/// The AVX-512 kernels, if the running CPU supports them.
pub(crate) fn kernels() -> Option<&'static Kernels> {
  static KERNELS: OnceLock<Option<Kernels>> = OnceLock::new();
  KERNELS
    .get_or_init(|| {
      // SAFETY: `detected` confirmed the base features and reports exactly
      // the extras present.
      detected().map(|extras| unsafe { build(extras) })
    })
    .as_ref()
}

/// The kernel table using the base features plus `extras`.
///
/// # Safety
/// The CPU must support the base requirement and every extra set in
/// `extras`. Tests also build tables for subsets of the detected extras.
pub(crate) unsafe fn build(extras: Extras) -> Kernels {
  let features = match (extras.vnni, extras.vpopcntdq) {
    (false, false) => "avx512f+bw",
    (true, false) => "avx512f+bw+vnni",
    (false, true) => "avx512f+bw+vpopcntdq",
    (true, true) => "avx512f+bw+vnni+vpopcntdq",
  };
  Kernels {
    isa: Isa::Avx512,
    features,
    dot,
    l2sq,
    distance_batch,
    dot_i8: if extras.vnni { dot_i8_vnni } else { dot_i8_bw },
    l2sq_u8: if extras.vnni {
      l2sq_u8_vnni
    } else {
      l2sq_u8_bw
    },
    hamming: if extras.vpopcntdq {
      hamming_vpopcntdq
    } else {
      hamming_bw
    },
    dot_f16,
    l2sq_f16,
    f16_to_f32: f16_to_f32_slice,
    f32_to_f16: f32_to_f16_slice,
  }
}

// Safe wrappers. Each is reachable only through a table from `build`, whose
// contract guarantees the CPU supports the base features and, for the
// `_vnni` and `_vpopcntdq` variants, that extra; that is the safety
// argument for every call below.

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

fn dot_i8_vnni(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed, including VNNI (see above).
  unsafe { dot_i8_vnni_impl(a, b) }
}

fn dot_i8_bw(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { dot_i8_bw_impl(a, b) }
}

fn l2sq_u8_vnni(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed, including VNNI (see above).
  unsafe { l2sq_u8_vnni_impl(a, b) }
}

fn l2sq_u8_bw(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { l2sq_u8_bw_impl(a, b) }
}

fn hamming_vpopcntdq(a: &[u64], b: &[u64]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed, including VPOPCNTDQ (see above).
  unsafe { hamming_vpopcntdq_impl(a, b) }
}

fn hamming_bw(a: &[u64], b: &[u64]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { hamming_bw_impl(a, b) }
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
#[target_feature(enable = "avx512f")]
fn step<const L2: bool>(acc: __m512, q: __m512, r: __m512) -> __m512 {
  if L2 {
    let d = _mm512_sub_ps(q, r);
    _mm512_fmadd_ps(d, d, acc)
  } else {
    _mm512_fmadd_ps(q, r, acc)
  }
}

/// Pairwise sum of `N` (a power of two) accumulators, then a horizontal
/// sum, in a fixed order (spelled out rather than `_mm512_reduce_add_ps`,
/// whose order the compiler may choose, so every instantiation rounds
/// identically).
#[inline]
#[target_feature(enable = "avx512f")]
fn reduce<const N: usize>(mut acc: [__m512; N]) -> f32 {
  let mut w = N;
  while w > 1 {
    w /= 2;
    for i in 0..w {
      acc[i] = _mm512_add_ps(acc[2 * i], acc[2 * i + 1]);
    }
  }
  let v = acc[0];
  let lo = _mm512_castps512_ps256(v);
  let hi = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(v)));
  let s = _mm256_add_ps(lo, hi);
  let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
  let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  let s = _mm_add_ss(s, _mm_movehdup_ps(s));
  _mm_cvtss_f32(s)
}

/// Prefetches the cache line holding `p` into L1. `p` need not be valid.
#[inline]
#[target_feature(enable = "avx512f")]
fn prefetch(p: *const f32) {
  // SAFETY: a prefetch is only a hint; it never faults or reads memory
  // architecturally, whatever the address.
  unsafe { _mm_prefetch::<_MM_HINT_T0>(p.cast()) }
}

/// Mask with the low `rem` bits set, `rem` in `0..16`.
#[inline]
fn mask16(rem: usize) -> __mmask16 {
  ((1u32 << rem) - 1) as __mmask16
}

/// Mask with the low `rem` bits set, `rem` in `0..64`.
#[inline]
fn mask64(rem: usize) -> __mmask64 {
  (1u64 << rem) - 1
}

/// The f32 kernel shared by `dot`, `l2sq` and `distance_batch`: the dot
/// product (or squared L2 distance when `L2`) of `q` with each of `R`
/// rows. Every row is accumulated in exactly the same order whatever `R`
/// is, so batched and single-row results are bit-identical. With `PF`,
/// prefetches the row data `pf` elements ahead of each row.
///
/// # Safety
/// The CPU supports AVX-512 F; `q` and every row pointer are valid for `n`
/// reads.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn rows_f32<const R: usize, const L2: bool, const PF: bool>(
  q: *const f32,
  rows: [*const f32; R],
  n: usize,
  pf: usize,
) -> [f32; R] {
  let mut acc = [[_mm512_setzero_ps(); ACC]; R];
  let mut j = 0;
  while j + 16 * ACC <= n {
    for u in 0..ACC {
      // SAFETY: `j + 16 * u + 16 <= n` and every pointer is valid for `n`
      // reads.
      let qv = unsafe { _mm512_loadu_ps(q.add(j + 16 * u)) };
      for r in 0..R {
        // SAFETY: as above.
        let rv = unsafe { _mm512_loadu_ps(rows[r].add(j + 16 * u)) };
        acc[r][u] = step::<L2>(acc[r][u], qv, rv);
      }
    }
    if PF {
      // One zmm load is one cache line: prefetch the lines the same
      // position of the next group of rows will need.
      for r in 0..R {
        for u in 0..ACC {
          prefetch(rows[r].wrapping_add(pf + j + 16 * u));
        }
      }
    }
    j += 16 * ACC;
  }
  // At most `ACC - 1` full vectors remain; give each its own accumulator.
  // Constant accumulator indices (not a running counter) keep the
  // accumulators in registers.
  for u in 0..ACC - 1 {
    if j + 16 <= n {
      // SAFETY: `j + 16 <= n`.
      let qv = unsafe { _mm512_loadu_ps(q.add(j)) };
      for r in 0..R {
        // SAFETY: `j + 16 <= n`.
        let rv = unsafe { _mm512_loadu_ps(rows[r].add(j)) };
        acc[r][u] = step::<L2>(acc[r][u], qv, rv);
      }
      j += 16;
    }
  }
  if j < n {
    let m = mask16(n - j);
    // SAFETY: a masked load accesses only the `n - j` lanes whose mask bit
    // is set, all in bounds; masked-off lanes never fault.
    let qv = unsafe { _mm512_maskz_loadu_ps(m, q.add(j)) };
    for r in 0..R {
      // SAFETY: as above.
      let rv = unsafe { _mm512_maskz_loadu_ps(m, rows[r].add(j)) };
      acc[r][ACC - 1] = step::<L2>(acc[r][ACC - 1], qv, rv);
    }
  }
  let mut out = [0.0; R];
  for r in 0..R {
    out[r] = reduce(acc[r]);
  }
  out
}

/// `out[i] = finish(metric, kernel(query, row_i))` for every row.
/// Requires `rows.len() >= query.len() * out.len()` and a non-empty query.
#[target_feature(enable = "avx512f")]
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

/// Exact `i8` dot product with VNNI. `vpdpbusd` multiplies unsigned by
/// signed bytes, so `a` is biased to `a + 128` (`a ^ 0x80`) and
/// `128 * sum(b)` is subtracted at the end. All sums wrap, so the result
/// is exact whenever the true dot product fits in `i32`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2")]
fn dot_i8_vnni_impl(a: &[i8], b: &[i8]) -> i32 {
  let n = a.len().min(b.len());
  if n < 64 {
    // Below one zmm of data the masked tail costs more than it saves.
    return crate::avx2::dot_i8_impl(a, b);
  }
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let bias = _mm512_set1_epi8(i8::MIN);
  let zero = _mm512_setzero_si512();
  let (mut acc0, mut acc1, mut cor0, mut cor1) = (zero, zero, zero, zero);
  let mut j = 0;
  while j + 128 <= n {
    // SAFETY: `j + 128 <= n`.
    let (a0, b0, a1, b1) = unsafe {
      (
        _mm512_loadu_si512(pa.add(j).cast()),
        _mm512_loadu_si512(pb.add(j).cast()),
        _mm512_loadu_si512(pa.add(j + 64).cast()),
        _mm512_loadu_si512(pb.add(j + 64).cast()),
      )
    };
    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_xor_si512(a0, bias), b0);
    cor0 = _mm512_dpbusd_epi32(cor0, bias, b0);
    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_xor_si512(a1, bias), b1);
    cor1 = _mm512_dpbusd_epi32(cor1, bias, b1);
    j += 128;
  }
  if j + 64 <= n {
    // SAFETY: `j + 64 <= n`.
    let (a0, b0) = unsafe {
      (
        _mm512_loadu_si512(pa.add(j).cast()),
        _mm512_loadu_si512(pb.add(j).cast()),
      )
    };
    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_xor_si512(a0, bias), b0);
    cor0 = _mm512_dpbusd_epi32(cor0, bias, b0);
    j += 64;
  }
  if j < n {
    let m = mask64(n - j);
    // SAFETY: masked loads access only the `n - j` in-bounds lanes.
    // Masked-off lanes are zero in both, contributing `128 * 0` to both
    // sums.
    let (a0, b0) = unsafe {
      (
        _mm512_maskz_loadu_epi8(m, pa.add(j)),
        _mm512_maskz_loadu_epi8(m, pb.add(j)),
      )
    };
    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_xor_si512(a0, bias), b0);
    cor1 = _mm512_dpbusd_epi32(cor1, bias, b0);
  }
  let acc = _mm512_add_epi32(acc0, acc1);
  let cor = _mm512_add_epi32(cor0, cor1);
  _mm512_reduce_add_epi32(_mm512_sub_epi32(acc, cor))
}

/// `madd` of the sign-extended low and high halves of two byte vectors.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn madd_i8(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
  let alo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a));
  let ahi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(a));
  let blo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b));
  let bhi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(b));
  (_mm512_madd_epi16(alo, blo), _mm512_madd_epi16(ahi, bhi))
}

/// Exact `i8` dot product without VNNI: sign-extend to `i16`, `madd`.
#[target_feature(enable = "avx512f,avx512bw,avx2")]
fn dot_i8_bw_impl(a: &[i8], b: &[i8]) -> i32 {
  let n = a.len().min(b.len());
  if n < 64 {
    // Below one zmm of data the masked tail costs more than it saves.
    return crate::avx2::dot_i8_impl(a, b);
  }
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = _mm512_setzero_si512();
  let mut acc1 = _mm512_setzero_si512();
  let mut j = 0;
  while j + 64 <= n {
    // SAFETY: `j + 64 <= n`.
    let (av, bv) = unsafe {
      (
        _mm512_loadu_si512(pa.add(j).cast()),
        _mm512_loadu_si512(pb.add(j).cast()),
      )
    };
    let (p0, p1) = madd_i8(av, bv);
    acc0 = _mm512_add_epi32(acc0, p0);
    acc1 = _mm512_add_epi32(acc1, p1);
    j += 64;
  }
  if j < n {
    let m = mask64(n - j);
    // SAFETY: masked loads access only the `n - j` in-bounds lanes;
    // masked-off lanes are zero and contribute nothing.
    let (av, bv) = unsafe {
      (
        _mm512_maskz_loadu_epi8(m, pa.add(j)),
        _mm512_maskz_loadu_epi8(m, pb.add(j)),
      )
    };
    let (p0, p1) = madd_i8(av, bv);
    acc0 = _mm512_add_epi32(acc0, p0);
    acc1 = _mm512_add_epi32(acc1, p1);
  }
  _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1))
}

/// `|a - b|` per byte, widened to `i16` in two halves (lane order does not
/// matter for a sum).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn absdiff_u8_wide(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
  let zero = _mm512_setzero_si512();
  let d = _mm512_sub_epi8(_mm512_max_epu8(a, b), _mm512_min_epu8(a, b));
  (_mm512_unpacklo_epi8(d, zero), _mm512_unpackhi_epi8(d, zero))
}

/// Loads 64 bytes of each input; the last partial block is loaded masked
/// with zeros (equal zeros add nothing to a squared distance).
///
/// # Safety
/// The CPU supports AVX-512 BW; `j < n` and both pointers are valid for
/// `n` reads.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load_u8_pair(
  pa: *const u8,
  pb: *const u8,
  j: usize,
  n: usize,
) -> (__m512i, __m512i) {
  if j + 64 <= n {
    // SAFETY: `j + 64 <= n`.
    unsafe {
      (
        _mm512_loadu_si512(pa.add(j).cast()),
        _mm512_loadu_si512(pb.add(j).cast()),
      )
    }
  } else {
    let m = mask64(n - j);
    // SAFETY: masked loads access only the `n - j` in-bounds lanes.
    unsafe {
      (
        _mm512_maskz_loadu_epi8(m, pa.add(j).cast()),
        _mm512_maskz_loadu_epi8(m, pb.add(j).cast()),
      )
    }
  }
}

/// Exact `u8` squared L2 with VNNI (`vpdpwssd` on the widened
/// differences). Wrapping `i32` lanes reinterpret as exact `u32` sums.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx2")]
fn l2sq_u8_vnni_impl(a: &[u8], b: &[u8]) -> u32 {
  let n = a.len().min(b.len());
  if n < 64 {
    // Below one zmm of data the masked tail costs more than it saves.
    return crate::avx2::l2sq_u8_impl(a, b);
  }
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = _mm512_setzero_si512();
  let mut acc1 = _mm512_setzero_si512();
  let mut j = 0;
  while j < n {
    // SAFETY: `j < n`; both pointers are valid for `n` reads.
    let (av, bv) = unsafe { load_u8_pair(pa, pb, j, n) };
    let (lo, hi) = absdiff_u8_wide(av, bv);
    acc0 = _mm512_dpwssd_epi32(acc0, lo, lo);
    acc1 = _mm512_dpwssd_epi32(acc1, hi, hi);
    j += 64;
  }
  _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1)) as u32
}

/// Exact `u8` squared L2 without VNNI (`madd` on the widened
/// differences).
#[target_feature(enable = "avx512f,avx512bw,avx2")]
fn l2sq_u8_bw_impl(a: &[u8], b: &[u8]) -> u32 {
  let n = a.len().min(b.len());
  if n < 64 {
    // Below one zmm of data the masked tail costs more than it saves.
    return crate::avx2::l2sq_u8_impl(a, b);
  }
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = _mm512_setzero_si512();
  let mut acc1 = _mm512_setzero_si512();
  let mut j = 0;
  while j < n {
    // SAFETY: `j < n`; both pointers are valid for `n` reads.
    let (av, bv) = unsafe { load_u8_pair(pa, pb, j, n) };
    let (lo, hi) = absdiff_u8_wide(av, bv);
    acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(lo, lo));
    acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(hi, hi));
    j += 64;
  }
  _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1)) as u32
}

/// XOR of eight words of each input; the last partial block is loaded
/// masked with zeros.
///
/// # Safety
/// The CPU supports AVX-512 F; `j < n` and both pointers are valid for `n`
/// reads.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_xor(
  pa: *const u64,
  pb: *const u64,
  j: usize,
  n: usize,
) -> __m512i {
  if j + 8 <= n {
    // SAFETY: `j + 8 <= n`.
    unsafe {
      _mm512_xor_si512(
        _mm512_loadu_si512(pa.add(j).cast()),
        _mm512_loadu_si512(pb.add(j).cast()),
      )
    }
  } else {
    let m = ((1u32 << (n - j)) - 1) as __mmask8;
    // SAFETY: masked loads access only the `n - j` in-bounds lanes.
    unsafe {
      _mm512_xor_si512(
        _mm512_maskz_loadu_epi64(m, pa.add(j).cast()),
        _mm512_maskz_loadu_epi64(m, pb.add(j).cast()),
      )
    }
  }
}

#[target_feature(enable = "avx512f,avx512vpopcntdq,popcnt")]
fn hamming_vpopcntdq_impl(a: &[u64], b: &[u64]) -> u32 {
  let n = a.len().min(b.len());
  if n < 8 {
    // Below one zmm of data, scalar POPCNT is faster than a masked load.
    return a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
  }
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = _mm512_setzero_si512();
  let mut acc1 = _mm512_setzero_si512();
  let mut j = 0;
  while j + 16 <= n {
    // SAFETY: `j + 8 < n`; both pointers are valid for `n` reads.
    let (x0, x1) =
      unsafe { (load_xor(pa, pb, j, n), load_xor(pa, pb, j + 8, n)) };
    acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(x0));
    acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(x1));
    j += 16;
  }
  while j < n {
    // SAFETY: `j < n`; both pointers are valid for `n` reads.
    let x = unsafe { load_xor(pa, pb, j, n) };
    acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(x));
    j += 8;
  }
  _mm512_reduce_add_epi64(_mm512_add_epi64(acc0, acc1)) as u32
}

/// Hamming distance without VPOPCNTDQ: nibble lookup table per byte, then
/// `vpsadbw` into 64-bit lanes.
#[target_feature(enable = "avx512f,avx512bw,popcnt")]
fn hamming_bw_impl(a: &[u64], b: &[u64]) -> u32 {
  let n = a.len().min(b.len());
  if n < 8 {
    // Below one zmm of data, scalar POPCNT is faster than a masked load.
    return a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
  }
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
  ));
  let low = _mm512_set1_epi8(0x0f);
  let zero = _mm512_setzero_si512();
  let mut acc = zero;
  let mut j = 0;
  while j < n {
    // SAFETY: `j < n`; both pointers are valid for `n` reads.
    let x = unsafe { load_xor(pa, pb, j, n) };
    let lo = _mm512_and_si512(x, low);
    let hi = _mm512_and_si512(_mm512_srli_epi16::<4>(x), low);
    let c = _mm512_add_epi8(
      _mm512_shuffle_epi8(lut, lo),
      _mm512_shuffle_epi8(lut, hi),
    );
    acc = _mm512_add_epi64(acc, _mm512_sad_epu8(c, zero));
    j += 8;
  }
  _mm512_reduce_add_epi64(acc) as u32
}

/// Loads sixteen binary16 values and widens them to `f32`.
///
/// # Safety
/// The CPU supports AVX-512 F; `p` is valid for 16 reads.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_f16(p: *const u16) -> __m512 {
  // SAFETY: guaranteed by the caller.
  _mm512_cvtph_ps(unsafe { _mm256_loadu_si256(p.cast()) })
}

#[target_feature(enable = "avx512f")]
fn f16_impl<const L2: bool>(a: &[u16], b: &[u16]) -> f32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc = [_mm512_setzero_ps(); 4];
  let mut j = 0;
  while j + 64 <= n {
    for u in 0..4 {
      // SAFETY: `j + 16 * u + 16 <= n`.
      let (x, y) =
        unsafe { (load_f16(pa.add(j + 16 * u)), load_f16(pb.add(j + 16 * u))) };
      acc[u] = step::<L2>(acc[u], x, y);
    }
    j += 64;
  }
  // Constant accumulator indices (not a running counter) keep the
  // accumulators in registers.
  for u in 0..3 {
    if j + 16 <= n {
      // SAFETY: `j + 16 <= n`.
      let (x, y) = unsafe { (load_f16(pa.add(j)), load_f16(pb.add(j))) };
      acc[u] = step::<L2>(acc[u], x, y);
      j += 16;
    }
  }
  if j < n {
    let (mut ta, mut tb) = ([0u16; 16], [0u16; 16]);
    ta[..n - j].copy_from_slice(&a[j..n]);
    tb[..n - j].copy_from_slice(&b[j..n]);
    // SAFETY: both buffers hold 16 values. Zero padding adds nothing.
    let (x, y) = unsafe { (load_f16(ta.as_ptr()), load_f16(tb.as_ptr())) };
    acc[3] = step::<L2>(acc[3], x, y);
  }
  reduce(acc)
}

#[target_feature(enable = "avx512f")]
fn f16_to_f32_impl(src: &[u16], dst: &mut [f32]) {
  let n = src.len().min(dst.len());
  let mut j = 0;
  while j + 16 <= n {
    // SAFETY: `j + 16 <= n` for both slices.
    unsafe {
      let v = load_f16(src.as_ptr().add(j));
      _mm512_storeu_ps(dst.as_mut_ptr().add(j), v);
    }
    j += 16;
  }
  for k in j..n {
    dst[k] = scalar::f16_to_f32(src[k]);
  }
}

#[target_feature(enable = "avx512f")]
fn f32_to_f16_impl(src: &[f32], dst: &mut [u16]) {
  let n = src.len().min(dst.len());
  let mut j = 0;
  while j + 16 <= n {
    // SAFETY: `j + 16 <= n` for both slices.
    unsafe {
      let v = _mm512_loadu_ps(src.as_ptr().add(j));
      let h =
        _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
      _mm256_storeu_si256(dst.as_mut_ptr().add(j).cast(), h);
    }
    j += 16;
  }
  for k in j..n {
    dst[k] = scalar::f32_to_f16(src[k]);
  }
}
