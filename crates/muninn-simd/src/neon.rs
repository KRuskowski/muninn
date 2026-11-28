//! NEON (Advanced SIMD) kernels for AArch64.
//!
//! NEON is part of the AArch64 baseline. The dot-product extension
//! (`sdot`/`udot`) selects faster integer kernels when present; it is
//! issued through inline assembly because its intrinsics are not stable
//! Rust yet, and so are the half-precision conversions (`fcvtl`, `fcvtn`;
//! baseline instructions, but their intrinsics need the unstable `f16`
//! type). [`kernels`] builds the table only after runtime detection.

// Index loops over small const-size arrays read better than iterator
// chains in SIMD code, and unroll to the same machine code.
#![allow(clippy::needless_range_loop)]

use core::arch::aarch64::*;
use core::arch::asm;
use std::sync::OnceLock;

use muninn_core::Metric;

use crate::dispatch::{Isa, Kernels};
use crate::{finish, scalar};

/// Rows per kernel call in `distance_gather`: as many as `distance_batch`
/// keeps in registers.
const GATHER_ROWS: usize = BATCH_ROWS;

/// How many groups of `GATHER_ROWS` ahead `distance_gather` prefetches.
/// Two measured best on SIFT-1M graph search (Emerald Rapids, 4 threads):
/// one group ahead arrives too late, and prefetching all rows up front
/// stalls on the line fill buffers.
const GATHER_AHEAD: usize = 2;

/// Rows per iteration in `distance_batch`: 4 rows x 4 accumulators use 16
/// of the 32 vector registers for sums; the query vector is loaded once
/// per group of rows.
const BATCH_ROWS: usize = 4;

/// Optional features detected on top of NEON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extras {
  pub dotprod: bool,
}

/// The optional features the running CPU has, or `None` without NEON.
pub(crate) fn detected() -> Option<Extras> {
  std::arch::is_aarch64_feature_detected!("neon").then(|| Extras {
    dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
  })
}

/// The NEON kernels, if the running CPU supports them.
pub(crate) fn kernels() -> Option<&'static Kernels> {
  static KERNELS: OnceLock<Option<Kernels>> = OnceLock::new();
  KERNELS
    .get_or_init(|| {
      // SAFETY: `detected` confirmed NEON and reports exactly the extras
      // present.
      detected().map(|extras| unsafe { build(extras) })
    })
    .as_ref()
}

/// The kernel table using NEON plus `extras`.
///
/// # Safety
/// The CPU must support NEON and every extra set in `extras`. Tests also
/// build tables for subsets of the detected extras.
pub(crate) unsafe fn build(extras: Extras) -> Kernels {
  Kernels {
    isa: Isa::Neon,
    features: if extras.dotprod {
      "neon+dotprod"
    } else {
      "neon"
    },
    dot,
    l2sq,
    distance_batch,
    distance_gather,
    dot_f32_u8,
    dot_f32_u8_gather,
    dot_i8: if extras.dotprod {
      dot_i8_dotprod
    } else {
      dot_i8_neon
    },
    l2sq_u8: if extras.dotprod {
      l2sq_u8_dotprod
    } else {
      l2sq_u8_neon
    },
    hamming,
    dot_f16,
    l2sq_f16,
    f16_to_f32: f16_to_f32_slice,
    f32_to_f16: f32_to_f16_slice,
  }
}

// Safe wrappers. Each is reachable only through a table from `build`, whose
// contract guarantees the CPU supports NEON and, for the `_dotprod`
// variants, the dot-product extension; that is the safety argument for
// every call below.

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

/// Distances to rows picked by id (see `crate::distance_gather`):
/// `GATHER_ROWS` rows per kernel call with the query loaded once, while
/// every line of the next group's rows is prefetched.
fn distance_gather(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  ids: &[u32],
  out: &mut [f32],
) {
  let dim = query.len();
  let n = ids.len().min(out.len());
  if dim == 0 {
    out[..n].fill(finish(metric, 0.0));
    return;
  }
  // Every id names a whole row (checked by the caller,
  // `crate::distance_gather`); `get` keeps a broken caller from reading
  // out of bounds.
  let row = |i: usize| -> *const f32 {
    let s = ids[i] as usize * dim;
    rows
      .get(s..s + dim)
      .map_or(core::ptr::null(), |r| r.as_ptr())
  };
  // SAFETY: CPU features confirmed (see above).
  unsafe {
    if metric == Metric::L2 {
      gather::<true>(metric, query.as_ptr(), dim, n, row, out)
    } else {
      gather::<false>(metric, query.as_ptr(), dim, n, row, out)
    }
  }
}

/// Prefetches every cache line of the row at `p` (`dim` floats). A null
/// `p` is skipped.
#[inline]
#[target_feature(enable = "neon")]
fn prefetch_row(p: *const f32, dim: usize) {
  if p.is_null() {
    return;
  }
  let mut off = 0;
  while off < dim {
    prefetch(p.wrapping_add(off));
    off += 16;
  }
  // The last line, when the row does not start on a line boundary.
  prefetch(p.wrapping_add(dim - 1));
}

/// The body of `distance_gather`; `row(i)` is the start of row `ids[i]`,
/// valid for `dim` reads, or null for an id out of range (scored as NaN).
/// Every line of the rows `GATHER_AHEAD` groups ahead is prefetched while
/// a group is scored.
#[inline]
#[target_feature(enable = "neon")]
fn gather<const L2: bool>(
  metric: Metric,
  q: *const f32,
  dim: usize,
  n: usize,
  row: impl Fn(usize) -> *const f32,
  out: &mut [f32],
) {
  let ahead = GATHER_ROWS * GATHER_AHEAD;
  for i in 0..n.min(ahead) {
    prefetch_row(row(i), dim);
  }
  let mut i = 0;
  while i + GATHER_ROWS <= n {
    let ptrs: [*const f32; GATHER_ROWS] = core::array::from_fn(|r| row(i + r));
    for j in i + ahead..n.min(i + ahead + GATHER_ROWS) {
      prefetch_row(row(j), dim);
    }
    if ptrs.iter().all(|p| !p.is_null()) {
      // SAFETY: every pointer is valid for `dim` reads.
      let d = unsafe { rows_f32::<GATHER_ROWS, L2, false>(q, ptrs, dim, 0) };
      for r in 0..GATHER_ROWS {
        out[i + r] = finish(metric, d[r]);
      }
    } else {
      for r in 0..GATHER_ROWS {
        out[i + r] = one::<L2>(metric, q, ptrs[r], dim);
      }
    }
    i += GATHER_ROWS;
  }
  while i < n {
    out[i] = one::<L2>(metric, q, row(i), dim);
    i += 1;
  }
}

/// One gathered row's distance; NaN for a null (out-of-range) row.
#[inline]
#[target_feature(enable = "neon")]
fn one<const L2: bool>(
  metric: Metric,
  q: *const f32,
  p: *const f32,
  dim: usize,
) -> f32 {
  if p.is_null() {
    return f32::NAN;
  }
  // SAFETY: a non-null row pointer is valid for `dim` reads.
  finish(metric, unsafe {
    rows_f32::<1, L2, false>(q, [p], dim, 0)[0]
  })
}

fn dot_f32_u8(a: &[f32], b: &[u8]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  let n = a.len().min(b.len());
  // SAFETY: CPU features confirmed (see above); both pointers are valid
  // for `n` reads.
  unsafe { f32_u8_impl(a.as_ptr(), b.as_ptr(), n) }
}

fn dot_f32_u8_gather(w: &[f32], codes: &[u8], ids: &[u32], out: &mut [f32]) {
  let dim = w.len();
  let n = ids.len().min(out.len());
  if dim == 0 {
    out[..n].fill(0.0);
    return;
  }
  // `get` keeps a caller with bad ids (only possible through the table;
  // the free function checks) from reading out of bounds.
  let row = |i: usize| -> Option<&[u8]> {
    let s = ids[i] as usize * dim;
    codes.get(s..s + dim)
  };
  let ahead = GATHER_ROWS * GATHER_AHEAD;
  for i in 0..n.min(ahead) {
    if let Some(r) = row(i) {
      prefetch_codes(r);
    }
  }
  for i in 0..n {
    if i + ahead < n
      && let Some(r) = row(i + ahead)
    {
      prefetch_codes(r);
    }
    out[i] = match row(i) {
      // SAFETY: CPU features confirmed (see above); both pointers are
      // valid for `dim` reads.
      Some(r) => unsafe { f32_u8_impl(w.as_ptr(), r.as_ptr(), dim) },
      None => f32::NAN,
    };
  }
}

/// Prefetches every cache line of a code row.
#[inline]
fn prefetch_codes(r: &[u8]) {
  let p = r.as_ptr();
  let mut off = 0;
  while off < r.len() {
    prefetch(p.wrapping_add(off).cast());
    off += 64;
  }
}

fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { dot_i8_neon_impl(a, b) }
}

fn dot_i8_dotprod(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed, including dotprod (see above).
  unsafe { dot_i8_dotprod_impl(a, b) }
}

fn l2sq_u8_neon(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed (see above).
  unsafe { l2sq_u8_neon_impl(a, b) }
}

fn l2sq_u8_dotprod(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  // SAFETY: CPU features confirmed, including dotprod (see above).
  unsafe { l2sq_u8_dotprod_impl(a, b) }
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
#[target_feature(enable = "neon")]
fn step<const L2: bool>(
  acc: float32x4_t,
  q: float32x4_t,
  r: float32x4_t,
) -> float32x4_t {
  if L2 {
    let d = vsubq_f32(q, r);
    vfmaq_f32(acc, d, d)
  } else {
    vfmaq_f32(acc, q, r)
  }
}

/// Sum of the four accumulators, then a horizontal sum (`faddp`, fixed
/// order).
#[inline]
#[target_feature(enable = "neon")]
fn reduce4(acc: [float32x4_t; 4]) -> f32 {
  vaddvq_f32(vaddq_f32(
    vaddq_f32(acc[0], acc[1]),
    vaddq_f32(acc[2], acc[3]),
  ))
}

/// Prefetches the cache line holding `p` into L1. `p` need not be valid.
#[inline]
fn prefetch(p: *const f32) {
  // SAFETY: `prfm` is only a hint; it never faults or reads memory
  // architecturally, whatever the address.
  unsafe {
    asm!(
      "prfm pldl1keep, [{p}]",
      p = in(reg) p,
      options(nostack, preserves_flags, readonly)
    )
  }
}

/// Copies the last `rem < 4` floats at `p` into a zero-padded vector.
///
/// # Safety
/// `p` is valid for `rem` reads.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_tail(p: *const f32, rem: usize) -> float32x4_t {
  let mut t = [0.0f32; 4];
  // SAFETY: `p` is valid for `rem <= 4` reads (caller), `t` holds 4.
  unsafe {
    core::ptr::copy_nonoverlapping(p, t.as_mut_ptr(), rem);
    vld1q_f32(t.as_ptr())
  }
}

/// The f32 kernel shared by `dot`, `l2sq` and `distance_batch`: the dot
/// product (or squared L2 distance when `L2`) of `q` with each of `R`
/// rows. Every row is accumulated in exactly the same order whatever `R`
/// is, so batched and single-row results are bit-identical. With `PF`,
/// prefetches the row data `pf` elements ahead of each row.
///
/// # Safety
/// The CPU supports NEON; `q` and every row pointer are valid for `n`
/// reads.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn rows_f32<const R: usize, const L2: bool, const PF: bool>(
  q: *const f32,
  rows: [*const f32; R],
  n: usize,
  pf: usize,
) -> [f32; R] {
  let mut acc = [[vdupq_n_f32(0.0); 4]; R];
  let mut j = 0;
  while j + 16 <= n {
    for u in 0..4 {
      // SAFETY: `j + 4 * u + 4 <= n` and every pointer is valid for `n`
      // reads.
      let qv = unsafe { vld1q_f32(q.add(j + 4 * u)) };
      for r in 0..R {
        // SAFETY: as above.
        let rv = unsafe { vld1q_f32(rows[r].add(j + 4 * u)) };
        acc[r][u] = step::<L2>(acc[r][u], qv, rv);
      }
    }
    if PF {
      // 16 floats are one 64-byte cache line.
      for r in 0..R {
        prefetch(rows[r].wrapping_add(pf + j));
      }
    }
    j += 16;
  }
  // At most three full vectors remain; give each its own accumulator.
  // Constant accumulator indices (not a running counter) keep the
  // accumulators in registers.
  for u in 0..3 {
    if j + 4 <= n {
      // SAFETY: `j + 4 <= n`.
      let qv = unsafe { vld1q_f32(q.add(j)) };
      for r in 0..R {
        // SAFETY: `j + 4 <= n`.
        let rv = unsafe { vld1q_f32(rows[r].add(j)) };
        acc[r][u] = step::<L2>(acc[r][u], qv, rv);
      }
      j += 4;
    }
  }
  if j < n {
    // SAFETY: `n - j` elements remain at each pointer.
    let qv = unsafe { load_tail(q.add(j), n - j) };
    for r in 0..R {
      // SAFETY: as above.
      let rv = unsafe { load_tail(rows[r].add(j), n - j) };
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
#[target_feature(enable = "neon")]
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

/// Sixteen codes at `c` widened to four `f32` vectors, multiplied with
/// the sixteen weights at `w` and added to the accumulators.
///
/// # Safety
/// The CPU supports NEON; `w` and `c` are valid for 16 reads.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn f32_u8_block(
  acc: &mut [float32x4_t; 4],
  w: *const f32,
  c: *const u8,
) {
  // SAFETY: guaranteed by the caller.
  let c = unsafe { vld1q_u8(c) };
  let lo = vmovl_u8(vget_low_u8(c));
  let hi = vmovl_high_u8(c);
  let y = [
    vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
    vcvtq_f32_u32(vmovl_high_u16(lo)),
    vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
    vcvtq_f32_u32(vmovl_high_u16(hi)),
  ];
  for u in 0..4 {
    // SAFETY: `w` is valid for 16 reads.
    let x = unsafe { vld1q_f32(w.add(4 * u)) };
    acc[u] = vfmaq_f32(acc[u], x, y[u]);
  }
}

/// `sum a_i * b_i` with the codes widened to `f32`: sixteen codes per
/// step into four accumulators, zero-padded tail.
///
/// # Safety
/// The CPU supports NEON; `a` and `b` are valid for `n` reads.
#[target_feature(enable = "neon")]
unsafe fn f32_u8_impl(a: *const f32, b: *const u8, n: usize) -> f32 {
  let mut acc = [vdupq_n_f32(0.0); 4];
  let mut j = 0;
  while j + 16 <= n {
    // SAFETY: `j + 16 <= n`.
    unsafe { f32_u8_block(&mut acc, a.add(j), b.add(j)) };
    j += 16;
  }
  if j < n {
    let rem = n - j;
    let (mut tw, mut tc) = ([0.0f32; 16], [0u8; 16]);
    // SAFETY: `rem < 16` values remain at both pointers; the buffers hold
    // 16. Zero padding adds nothing.
    unsafe {
      core::ptr::copy_nonoverlapping(a.add(j), tw.as_mut_ptr(), rem);
      core::ptr::copy_nonoverlapping(b.add(j), tc.as_mut_ptr(), rem);
      f32_u8_block(&mut acc, tw.as_ptr(), tc.as_ptr());
    }
  }
  reduce4(acc)
}

/// Exact `i8` dot product: `smull` into `i16` (|-128 * -128| fits), then
/// pairwise accumulate into `i32`.
#[target_feature(enable = "neon")]
fn dot_i8_neon_impl(a: &[i8], b: &[i8]) -> i32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = vdupq_n_s32(0);
  let mut acc1 = vdupq_n_s32(0);
  let mut j = 0;
  while j + 16 <= n {
    // SAFETY: `j + 16 <= n`.
    let (av, bv) = unsafe { (vld1q_s8(pa.add(j)), vld1q_s8(pb.add(j))) };
    acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(av), vget_low_s8(bv)));
    acc1 = vpadalq_s16(acc1, vmull_high_s8(av, bv));
    j += 16;
  }
  let mut s = vaddvq_s32(vaddq_s32(acc0, acc1));
  for k in j..n {
    s = s.wrapping_add(a[k] as i32 * b[k] as i32);
  }
  s
}

/// `acc + sum of 4 adjacent i8 products` per lane (`sdot`).
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
  let mut acc = acc;
  // SAFETY: register-only instruction, available because this function
  // enables `dotprod`.
  unsafe {
    asm!(
      "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
      acc = inout(vreg) acc,
      a = in(vreg) a,
      b = in(vreg) b,
      options(pure, nomem, nostack, preserves_flags)
    );
  }
  acc
}

/// `acc + sum of 4 adjacent u8 products` per lane (`udot`).
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn udot(acc: uint32x4_t, a: uint8x16_t, b: uint8x16_t) -> uint32x4_t {
  let mut acc = acc;
  // SAFETY: register-only instruction, available because this function
  // enables `dotprod`.
  unsafe {
    asm!(
      "udot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
      acc = inout(vreg) acc,
      a = in(vreg) a,
      b = in(vreg) b,
      options(pure, nomem, nostack, preserves_flags)
    );
  }
  acc
}

#[target_feature(enable = "neon,dotprod")]
fn dot_i8_dotprod_impl(a: &[i8], b: &[i8]) -> i32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = vdupq_n_s32(0);
  let mut acc1 = vdupq_n_s32(0);
  let mut j = 0;
  while j + 32 <= n {
    // SAFETY: `j + 32 <= n`.
    let (a0, b0, a1, b1) = unsafe {
      (
        vld1q_s8(pa.add(j)),
        vld1q_s8(pb.add(j)),
        vld1q_s8(pa.add(j + 16)),
        vld1q_s8(pb.add(j + 16)),
      )
    };
    acc0 = sdot(acc0, a0, b0);
    acc1 = sdot(acc1, a1, b1);
    j += 32;
  }
  if j + 16 <= n {
    // SAFETY: `j + 16 <= n`.
    let (a0, b0) = unsafe { (vld1q_s8(pa.add(j)), vld1q_s8(pb.add(j))) };
    acc0 = sdot(acc0, a0, b0);
    j += 16;
  }
  let mut s = vaddvq_s32(vaddq_s32(acc0, acc1));
  for k in j..n {
    s = s.wrapping_add(a[k] as i32 * b[k] as i32);
  }
  s
}

/// Exact `u8` squared L2: `uabd`, `umull` into `u16` (255^2 fits), then
/// pairwise accumulate into `u32`.
#[target_feature(enable = "neon")]
fn l2sq_u8_neon_impl(a: &[u8], b: &[u8]) -> u32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = vdupq_n_u32(0);
  let mut acc1 = vdupq_n_u32(0);
  let mut j = 0;
  while j + 16 <= n {
    // SAFETY: `j + 16 <= n`.
    let (av, bv) = unsafe { (vld1q_u8(pa.add(j)), vld1q_u8(pb.add(j))) };
    let d = vabdq_u8(av, bv);
    acc0 = vpadalq_u16(acc0, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
    acc1 = vpadalq_u16(acc1, vmull_high_u8(d, d));
    j += 16;
  }
  let mut s = vaddvq_u32(vaddq_u32(acc0, acc1));
  for k in j..n {
    let d = a[k] as i32 - b[k] as i32;
    s = s.wrapping_add((d * d) as u32);
  }
  s
}

#[target_feature(enable = "neon,dotprod")]
fn l2sq_u8_dotprod_impl(a: &[u8], b: &[u8]) -> u32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc0 = vdupq_n_u32(0);
  let mut acc1 = vdupq_n_u32(0);
  let mut j = 0;
  while j + 32 <= n {
    // SAFETY: `j + 32 <= n`.
    let (d0, d1) = unsafe {
      (
        vabdq_u8(vld1q_u8(pa.add(j)), vld1q_u8(pb.add(j))),
        vabdq_u8(vld1q_u8(pa.add(j + 16)), vld1q_u8(pb.add(j + 16))),
      )
    };
    acc0 = udot(acc0, d0, d0);
    acc1 = udot(acc1, d1, d1);
    j += 32;
  }
  if j + 16 <= n {
    // SAFETY: `j + 16 <= n`.
    let d0 = unsafe { vabdq_u8(vld1q_u8(pa.add(j)), vld1q_u8(pb.add(j))) };
    acc0 = udot(acc0, d0, d0);
    j += 16;
  }
  let mut s = vaddvq_u32(vaddq_u32(acc0, acc1));
  for k in j..n {
    let d = a[k] as i32 - b[k] as i32;
    s = s.wrapping_add((d * d) as u32);
  }
  s
}

/// Hamming distance: `cnt` per byte, pairwise accumulate into `u16` lanes
/// (flushed to `u32` before they can overflow).
#[target_feature(enable = "neon")]
fn hamming_impl(a: &[u64], b: &[u64]) -> u32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc32 = vdupq_n_u32(0);
  let mut j = 0;
  while j + 2 <= n {
    // Each iteration adds at most 16 to a u16 lane: 2048 iterations
    // (4096 words) stay below 2^16.
    let end = n.min(j + 4096);
    let mut acc16 = vdupq_n_u16(0);
    while j + 2 <= end {
      // SAFETY: `j + 2 <= n`.
      let x = unsafe { veorq_u64(vld1q_u64(pa.add(j)), vld1q_u64(pb.add(j))) };
      acc16 = vpadalq_u8(acc16, vcntq_u8(vreinterpretq_u8_u64(x)));
      j += 2;
    }
    acc32 = vpadalq_u16(acc32, acc16);
  }
  let mut s = vaddvq_u32(acc32);
  for k in j..n {
    s = s.wrapping_add((a[k] ^ b[k]).count_ones());
  }
  s
}

/// Widens eight binary16 values to two `f32` vectors (`fcvtl`, `fcvtl2`).
#[inline]
#[target_feature(enable = "neon")]
fn widen_f16(x: uint16x8_t) -> (float32x4_t, float32x4_t) {
  let lo: float32x4_t;
  let hi: float32x4_t;
  // SAFETY: register-only baseline AdvSIMD instructions (they may set
  // FPSR exception flags, so no `preserves_flags`). `lo` is an
  // `out` (not `lateout`) operand, so it cannot share a register with `x`,
  // which the second instruction still reads.
  unsafe {
    asm!(
      "fcvtl {lo:v}.4s, {x:v}.4h",
      "fcvtl2 {hi:v}.4s, {x:v}.8h",
      x = in(vreg) x,
      lo = out(vreg) lo,
      hi = out(vreg) hi,
      options(pure, nomem, nostack)
    );
  }
  (lo, hi)
}

/// Narrows two `f32` vectors to eight binary16 values (`fcvtn`, `fcvtn2`),
/// rounding to nearest even.
#[inline]
#[target_feature(enable = "neon")]
fn narrow_f16(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
  let out: uint16x8_t;
  // SAFETY: register-only baseline AdvSIMD instructions (they may set
  // FPSR exception flags, so no `preserves_flags`). `o` is an `out`
  // (not `lateout`) operand, so it cannot share a register with `hi`,
  // which the second instruction still reads.
  unsafe {
    asm!(
      "fcvtn {o:v}.4h, {lo:v}.4s",
      "fcvtn2 {o:v}.8h, {hi:v}.4s",
      lo = in(vreg) lo,
      hi = in(vreg) hi,
      o = out(vreg) out,
      options(pure, nomem, nostack)
    );
  }
  out
}

#[target_feature(enable = "neon")]
fn f16_impl<const L2: bool>(a: &[u16], b: &[u16]) -> f32 {
  let n = a.len().min(b.len());
  let (pa, pb) = (a.as_ptr(), b.as_ptr());
  let mut acc = [vdupq_n_f32(0.0); 4];
  let mut j = 0;
  while j + 16 <= n {
    // SAFETY: `j + 16 <= n`.
    let (a0, a1, b0, b1) = unsafe {
      (
        vld1q_u16(pa.add(j)),
        vld1q_u16(pa.add(j + 8)),
        vld1q_u16(pb.add(j)),
        vld1q_u16(pb.add(j + 8)),
      )
    };
    let ((x0, x1), (x2, x3)) = (widen_f16(a0), widen_f16(a1));
    let ((y0, y1), (y2, y3)) = (widen_f16(b0), widen_f16(b1));
    acc[0] = step::<L2>(acc[0], x0, y0);
    acc[1] = step::<L2>(acc[1], x1, y1);
    acc[2] = step::<L2>(acc[2], x2, y2);
    acc[3] = step::<L2>(acc[3], x3, y3);
    j += 16;
  }
  if j + 8 <= n {
    // SAFETY: `j + 8 <= n`.
    let (a0, b0) = unsafe { (vld1q_u16(pa.add(j)), vld1q_u16(pb.add(j))) };
    let ((x0, x1), (y0, y1)) = (widen_f16(a0), widen_f16(b0));
    acc[0] = step::<L2>(acc[0], x0, y0);
    acc[1] = step::<L2>(acc[1], x1, y1);
    j += 8;
  }
  if j < n {
    let (mut ta, mut tb) = ([0u16; 8], [0u16; 8]);
    ta[..n - j].copy_from_slice(&a[j..n]);
    tb[..n - j].copy_from_slice(&b[j..n]);
    // SAFETY: both buffers hold 8 values. Zero padding adds nothing.
    let (a0, b0) = unsafe { (vld1q_u16(ta.as_ptr()), vld1q_u16(tb.as_ptr())) };
    let ((x0, x1), (y0, y1)) = (widen_f16(a0), widen_f16(b0));
    acc[2] = step::<L2>(acc[2], x0, y0);
    acc[3] = step::<L2>(acc[3], x1, y1);
  }
  reduce4(acc)
}

#[target_feature(enable = "neon")]
fn f16_to_f32_impl(src: &[u16], dst: &mut [f32]) {
  let n = src.len().min(dst.len());
  let mut j = 0;
  while j + 8 <= n {
    // SAFETY: `j + 8 <= n` for both slices.
    unsafe {
      let (lo, hi) = widen_f16(vld1q_u16(src.as_ptr().add(j)));
      vst1q_f32(dst.as_mut_ptr().add(j), lo);
      vst1q_f32(dst.as_mut_ptr().add(j + 4), hi);
    }
    j += 8;
  }
  for k in j..n {
    dst[k] = scalar::f16_to_f32(src[k]);
  }
}

#[target_feature(enable = "neon")]
fn f32_to_f16_impl(src: &[f32], dst: &mut [u16]) {
  let n = src.len().min(dst.len());
  let mut j = 0;
  while j + 8 <= n {
    // SAFETY: `j + 8 <= n` for both slices.
    unsafe {
      let lo = vld1q_f32(src.as_ptr().add(j));
      let hi = vld1q_f32(src.as_ptr().add(j + 4));
      vst1q_u16(dst.as_mut_ptr().add(j), narrow_f16(lo, hi));
    }
    j += 8;
  }
  for k in j..n {
    dst[k] = scalar::f32_to_f16(src[k]);
  }
}
