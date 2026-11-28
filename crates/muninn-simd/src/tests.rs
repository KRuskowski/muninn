//! Every kernel of every instruction set the running CPU supports,
//! including variants with optional features switched off, against the
//! scalar reference.
//!
//! Set `MUNINN_TEST_REQUIRE_ISA=avx512` (or `avx2`, `neon`) to make the
//! suite fail when that instruction set is unavailable, so CI cannot pass
//! by silently testing only the scalar path.

use muninn_core::Metric;

use crate::dispatch::{Isa, Kernels, SCALAR, select};
use crate::{finish, scalar};

/// Every kernel table testable on this machine.
fn variants() -> Vec<Kernels> {
  #[allow(unused_mut)]
  let mut v = vec![SCALAR];
  #[cfg(target_arch = "x86_64")]
  {
    if let Some(k) = crate::avx2::kernels() {
      v.push(*k);
    }
    if let Some(ex) = crate::avx512::detected() {
      for vnni in [false, true] {
        for vpopcntdq in [false, true] {
          if (vnni && !ex.vnni) || (vpopcntdq && !ex.vpopcntdq) {
            continue;
          }
          let extras = crate::avx512::Extras { vnni, vpopcntdq };
          // SAFETY: the base features were detected and only extras the
          // CPU has are enabled.
          v.push(unsafe { crate::avx512::build(extras) });
        }
      }
    }
  }
  #[cfg(target_arch = "aarch64")]
  {
    if let Some(ex) = crate::neon::detected() {
      for dotprod in [false, true] {
        if dotprod && !ex.dotprod {
          continue;
        }
        let extras = crate::neon::Extras { dotprod };
        // SAFETY: NEON was detected and only extras the CPU has are
        // enabled.
        v.push(unsafe { crate::neon::build(extras) });
      }
    }
  }
  v
}

/// SplitMix64: tiny, deterministic, good enough for test data.
struct Rng(u64);

impl Rng {
  fn new(seed: u64) -> Self {
    Self(seed)
  }

  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
  }

  /// Uniform in `[-1, 1)`.
  fn f32(&mut self) -> f32 {
    (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
  }

  /// Random `i8`, with the extremes -128 and 127 over-represented.
  fn i8(&mut self) -> i8 {
    let r = self.next_u64();
    match r & 7 {
      0 => i8::MIN,
      1 => i8::MAX,
      _ => (r >> 8) as i8,
    }
  }

  /// Random `u8`, with the extremes 0 and 255 over-represented.
  fn u8(&mut self) -> u8 {
    let r = self.next_u64();
    match r & 7 {
      0 => 0,
      1 => 255,
      _ => (r >> 8) as u8,
    }
  }

  /// A buffer of `len + 1` values; callers slice at offset 0 or 1 so that
  /// both aligned and misaligned starts are exercised.
  fn fill<T>(
    &mut self,
    len: usize,
    mut f: impl FnMut(&mut Self) -> T,
  ) -> Vec<T> {
    (0..=len).map(|_| f(self)).collect()
  }
}

impl Rng {
  /// Exactly `len` values.
  fn exact<T>(&mut self, len: usize, f: impl FnMut(&mut Self) -> T) -> Vec<T> {
    let mut v = self.fill(len, f);
    v.truncate(len);
    v
  }
}

/// The `len`-element window of `buf` starting at `off`.
fn win<T>(buf: &[T], off: usize, len: usize) -> &[T] {
  &buf[off..off + len]
}

/// Float results may differ from the reference by reassociation error,
/// which is bounded relative to the sum of the absolute terms.
fn assert_close(got: f32, want: f32, mag: f32, what: &str) {
  let tol = 1e-5 * mag + 1e-6;
  assert!(
    (got - want).abs() <= tol,
    "{what}: got {got}, want {want} (tolerance {tol})"
  );
}

const MAX_LEN: usize = 1024;

#[test]
fn reports_tested_variants() {
  let v = variants();
  let names: Vec<&str> = v.iter().map(|k| k.features).collect();
  eprintln!("kernel variants under test: {names:?}");
  assert_eq!(v[0].isa, Isa::Scalar);
  if let Ok(req) = std::env::var("MUNINN_TEST_REQUIRE_ISA") {
    let isa = Isa::parse(&req).expect("MUNINN_TEST_REQUIRE_ISA is not an ISA");
    assert!(
      v.iter().any(|k| k.isa == isa),
      "{req} required but not supported here; tested {names:?}"
    );
  }
}

#[test]
fn dot_matches_scalar() {
  for k in variants() {
    let mut rng = Rng::new(1);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, Rng::f32);
      let b = rng.fill(len, Rng::f32);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      let mag: f32 = a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum();
      let what = format!("{} dot len {len}", k.features);
      assert_close((k.dot)(a, b), scalar::dot(a, b), mag, &what);
    }
  }
}

#[test]
fn l2sq_matches_scalar() {
  for k in variants() {
    let mut rng = Rng::new(2);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, Rng::f32);
      let b = rng.fill(len, Rng::f32);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      let want = scalar::l2sq(a, b);
      let what = format!("{} l2sq len {len}", k.features);
      assert_close((k.l2sq)(a, b), want, want, &what);
    }
  }
}

/// Reference distance and its error magnitude for one row.
fn reference(metric: Metric, q: &[f32], row: &[f32]) -> (f32, f32) {
  match metric {
    Metric::L2 => {
      let d = scalar::l2sq(q, row);
      (d, d)
    }
    Metric::Dot | Metric::Cosine => {
      let mag = q.iter().zip(row).map(|(x, y)| (x * y).abs()).sum();
      (finish(metric, scalar::dot(q, row)), mag)
    }
  }
}

const METRICS: [Metric; 3] = [Metric::L2, Metric::Dot, Metric::Cosine];

#[test]
fn distance_batch_matches_scalar() {
  for k in variants() {
    let mut rng = Rng::new(3);
    for dim in 0..=MAX_LEN {
      // 1..=9 rows: covers every remainder of the 2- and 4-row groups.
      let n = 1 + (dim * 7) % 9;
      let q = rng.fill(dim, Rng::f32);
      let rows = rng.fill(dim * n, Rng::f32);
      let (q, rows) = (win(&q, dim % 2, dim), win(&rows, 1, dim * n));
      for metric in METRICS {
        let mut out = vec![f32::NAN; n];
        (k.distance_batch)(metric, q, rows, &mut out);
        for (i, &got) in out.iter().enumerate() {
          let row = if dim == 0 {
            &[][..]
          } else {
            &rows[i * dim..(i + 1) * dim]
          };
          let (want, mag) = reference(metric, q, row);
          let what =
            format!("{} batch {metric:?} dim {dim} row {i}/{n}", k.features);
          assert_close(got, want, mag, &what);
        }
      }
    }
  }
}

/// Batched results must equal the single-row kernels bit for bit, and must
/// not depend on where a batch starts, so a scan gives identical results
/// however it is split across threads or calls.
#[test]
fn distance_batch_is_bitwise_split_invariant() {
  for k in variants() {
    let mut rng = Rng::new(4);
    for dim in [1, 3, 15, 16, 17, 31, 32, 63, 64, 65, 96, 100, 128, 383, 768] {
      for n in 0..=13 {
        let q = rng.fill(dim, Rng::f32);
        let rows = rng.fill(dim * n, Rng::f32);
        let (q, rows) = (&q[..dim], &rows[..dim * n]);
        for metric in METRICS {
          let mut all = vec![0.0; n];
          (k.distance_batch)(metric, q, rows, &mut all);
          for (i, row) in rows.chunks_exact(dim).enumerate() {
            let single = match metric {
              Metric::L2 => (k.l2sq)(q, row),
              _ => finish(metric, (k.dot)(q, row)),
            };
            assert_eq!(
              all[i].to_bits(),
              single.to_bits(),
              "{} {metric:?} dim {dim} row {i}/{n}",
              k.features
            );
          }
          for start in 1..n.min(4) {
            let mut part = vec![0.0; n - start];
            (k.distance_batch)(metric, q, &rows[start * dim..], &mut part);
            for (i, d) in part.iter().enumerate() {
              assert_eq!(d.to_bits(), all[start + i].to_bits());
            }
          }
        }
      }
    }
  }
}

/// Gathered rows score bit-identically to `distance` on each row, for
/// every id count (all group remainders) and dimension, repeated ids
/// included.
#[test]
fn distance_gather_matches_distance_bitwise() {
  for k in variants() {
    let mut rng = Rng::new(21);
    for dim in (0..=MAX_LEN).step_by(7).chain([1, 16, 64, 100, 128, 256]) {
      let rows_n = 13;
      let rows = rng.exact(dim * rows_n, Rng::f32);
      let q = rng.exact(dim, Rng::f32);
      for count in 0..=11 {
        let ids: Vec<u32> = (0..count)
          .map(|_| (rng.next_u64() % rows_n as u64) as u32)
          .collect();
        for metric in METRICS {
          let mut out = vec![0.0f32; count + 1];
          (k.distance_gather)(metric, &q, &rows, &ids, &mut out);
          for (i, &id) in ids.iter().enumerate() {
            let row = &rows[id as usize * dim..(id as usize + 1) * dim];
            let mut want = [0.0f32];
            (k.distance_batch)(metric, &q, row, &mut want);
            assert_eq!(
              out[i].to_bits(),
              want[0].to_bits(),
              "{} {metric:?} dim {dim} count {count} slot {i}",
              k.features
            );
          }
          assert_eq!(out[count], 0.0, "wrote past the ids");
        }
      }
    }
  }
}

/// Gathered code rows score bit-identically to `dot_f32_u8` on each
/// row, for every id count and dimension.
#[test]
fn dot_f32_u8_gather_matches_single_rows_bitwise() {
  for k in variants() {
    let mut rng = Rng::new(31);
    for dim in (0..=MAX_LEN).step_by(9).chain([1, 16, 64, 100, 128, 256]) {
      let rows_n = 11;
      let codes = rng.exact(dim * rows_n, Rng::u8);
      let w = rng.exact(dim, Rng::f32);
      for count in 0..=10 {
        let ids: Vec<u32> = (0..count)
          .map(|_| (rng.next_u64() % rows_n as u64) as u32)
          .collect();
        let mut out = vec![0.0f32; count + 1];
        (k.dot_f32_u8_gather)(&w, &codes, &ids, &mut out);
        for (i, &id) in ids.iter().enumerate() {
          let row = &codes[id as usize * dim..(id as usize + 1) * dim];
          let want = (k.dot_f32_u8)(&w, row);
          assert_eq!(
            out[i].to_bits(),
            want.to_bits(),
            "{} dim {dim} count {count} slot {i}",
            k.features
          );
        }
        assert_eq!(out[count], 0.0, "wrote past the ids");
      }
    }
    let mut out = [0.0f32; 3];
    (k.dot_f32_u8_gather)(&[1.0; 4], &[1u8; 8], &[1, 2, 0], &mut out);
    assert!(out[1].is_nan() && out[0] == 4.0 && out[2] == 4.0);
  }
  let r = std::panic::catch_unwind(|| {
    let mut out = [0.0f32; 1];
    crate::dot_f32_u8_gather(&[0.0; 4], &[0; 8], &[2], &mut out);
  });
  assert!(r.is_err());
}

/// Called through the table, a bad id scores NaN instead of reading out
/// of bounds; the free function panics.
#[test]
fn distance_gather_rejects_bad_ids() {
  for k in variants() {
    let rows = [1.0f32; 8];
    let q = [0.5f32; 4];
    let mut out = [0.0f32; 6];
    let ids = [0, 2, 1, 7, 0, 1];
    (k.distance_gather)(Metric::L2, &q, &rows, &ids, &mut out);
    assert!(out[1].is_nan() && out[3].is_nan(), "{}", k.features);
    assert_eq!(out[0], 1.0);
    assert_eq!(out[5], 1.0);
  }
  let r = std::panic::catch_unwind(|| {
    let mut out = [0.0f32; 1];
    crate::distance_gather(Metric::L2, &[0.0; 4], &[0.0; 8], &[2], &mut out);
  });
  assert!(r.is_err());
}

#[test]
fn distance_batch_handles_empty_inputs() {
  for k in variants() {
    for metric in METRICS {
      (k.distance_batch)(metric, &[1.0, 2.0], &[], &mut []);
      let mut out = [f32::NAN; 3];
      (k.distance_batch)(metric, &[], &[], &mut out);
      let want = finish(metric, 0.0);
      assert!(out.iter().all(|d| d.to_bits() == want.to_bits()));
    }
  }
}

#[test]
fn dot_f32_u8_matches_scalar() {
  for k in variants() {
    let mut rng = Rng::new(11);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, Rng::f32);
      let b = rng.fill(len, Rng::u8);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      let mag: f32 = a.iter().zip(b).map(|(x, &y)| (x * y as f32).abs()).sum();
      let what = format!("{} dot_f32_u8 len {len}", k.features);
      assert_close((k.dot_f32_u8)(a, b), scalar::dot_f32_u8(a, b), mag, &what);
    }
  }
}

/// Small integer weights make every product and partial sum exact in
/// `f32` (all stay below 2^24), so every instruction set must agree with
/// a plain integer sum bit for bit, extreme codes included.
#[test]
fn dot_f32_u8_is_exact_on_integer_weights() {
  for k in variants() {
    let mut rng = Rng::new(12);
    for len in [0, 1, 7, 15, 16, 17, 63, 64, 65, 100, 128, 255, 256, 1024] {
      let b = rng.exact(len, Rng::u8);
      let a: Vec<f32> = (0..len)
        .map(|i| match i % 4 {
          0 => 15.0,
          1 => -15.0,
          _ => (rng.next_u64() % 31) as f32 - 15.0,
        })
        .collect();
      let want: i64 =
        a.iter().zip(&b).map(|(&x, &y)| x as i64 * y as i64).sum();
      let got = (k.dot_f32_u8)(&a, &b);
      assert_eq!(got, want as f32, "{} len {len}", k.features);
    }
  }
}

#[test]
fn dot_i8_matches_scalar_exactly() {
  for k in variants() {
    let mut rng = Rng::new(5);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, Rng::i8);
      let b = rng.fill(len, Rng::i8);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      assert_eq!(
        (k.dot_i8)(a, b),
        scalar::dot_i8(a, b),
        "{} len {len}",
        k.features
      );
    }
  }
}

#[test]
fn dot_i8_extremes_are_exact() {
  let pairs: [(i8, i8); 4] =
    [(-128, -128), (127, 127), (-128, 127), (127, -128)];
  for k in variants() {
    for (x, y) in pairs {
      for len in 0..=MAX_LEN {
        let (a, b) = (vec![x; len], vec![y; len]);
        assert_eq!(
          (k.dot_i8)(&a, &b),
          scalar::dot_i8(&a, &b),
          "{} {x}*{y} len {len}",
          k.features
        );
      }
    }
    // Alternating extremes.
    let a: Vec<i8> = (0..MAX_LEN).map(|i| [-128, 127][i % 2]).collect();
    let b: Vec<i8> = (0..MAX_LEN).map(|i| [-128, 127, 127][i % 3]).collect();
    assert_eq!((k.dot_i8)(&a, &b), scalar::dot_i8(&a, &b), "{}", k.features);
  }
}

/// Sums close to the `i32` limit, where intermediate (biased) sums in the
/// VNNI path wrap, must still come out exact.
#[test]
fn dot_i8_large_sums_are_exact() {
  let len = 131_071;
  for k in variants() {
    for (x, y) in [(-128i8, -128i8), (127, -128), (-128, 127), (127, 127)] {
      let (a, b) = (vec![x; len], vec![y; len]);
      let want = x as i64 * y as i64 * len as i64;
      assert_eq!((k.dot_i8)(&a, &b) as i64, want, "{} {x}*{y}", k.features);
    }
  }
}

#[test]
fn l2sq_u8_matches_scalar_exactly() {
  for k in variants() {
    let mut rng = Rng::new(6);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, Rng::u8);
      let b = rng.fill(len, Rng::u8);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      assert_eq!(
        (k.l2sq_u8)(a, b),
        scalar::l2sq_u8(a, b),
        "{} len {len}",
        k.features
      );
    }
  }
}

#[test]
fn l2sq_u8_extremes_are_exact() {
  for k in variants() {
    for len in 0..=MAX_LEN {
      let (a, b) = (vec![0u8; len], vec![255u8; len]);
      let want = 255 * 255 * len as u32;
      assert_eq!((k.l2sq_u8)(&a, &b), want, "{} len {len}", k.features);
      assert_eq!((k.l2sq_u8)(&b, &a), want, "{} len {len}", k.features);
      assert_eq!((k.l2sq_u8)(&b, &b), 0, "{} len {len}", k.features);
    }
    // A total above i32::MAX (but within u32) must not be mangled.
    let len = 66_000;
    let (a, b) = (vec![0u8; len], vec![255u8; len]);
    assert_eq!(
      (k.l2sq_u8)(&a, &b),
      255 * 255 * len as u32,
      "{}",
      k.features
    );
  }
}

/// Past the exact range the true sum no longer fits; every instruction
/// set must then agree on the same wrapped (mod 2^32) value instead of
/// one panicking and another wrapping.
#[test]
fn l2sq_u8_wraps_identically_past_exact_limit() {
  // 66_051 is the longest worst-case input whose sum fits in u32.
  let len = 70_000;
  let (a, b) = (vec![0u8; len], vec![255u8; len]);
  let want = (255u64 * 255 * len as u64) as u32;
  for k in variants() {
    assert_eq!((k.l2sq_u8)(&a, &b), want, "{}", k.features);
  }
}

#[test]
fn dot_i8_wraps_identically_past_exact_limit() {
  // 131_071 * 16_384 is the largest all -128 sum that fits in i32.
  let len = 140_000;
  let a = vec![i8::MIN; len];
  let want = (16_384i64 * len as i64) as i32;
  for k in variants() {
    assert_eq!((k.dot_i8)(&a, &a), want, "{}", k.features);
  }
}

#[test]
fn hamming_matches_scalar_exactly() {
  for k in variants() {
    let mut rng = Rng::new(7);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, Rng::next_u64);
      let b = rng.fill(len, Rng::next_u64);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      assert_eq!(
        (k.hamming)(a, b),
        scalar::hamming(a, b),
        "{} len {len}",
        k.features
      );
      let (zeros, ones) = (vec![0u64; len], vec![u64::MAX; len]);
      assert_eq!((k.hamming)(&zeros, &ones), 64 * len as u32);
    }
  }
}

fn random_f16(rng: &mut Rng) -> u16 {
  scalar::f32_to_f16(rng.f32() * 4.0)
}

#[test]
fn dot_f16_matches_scalar() {
  for k in variants() {
    let mut rng = Rng::new(8);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, random_f16);
      let b = rng.fill(len, random_f16);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      let mag: f32 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (scalar::f16_to_f32(x) * scalar::f16_to_f32(y)).abs())
        .sum();
      let what = format!("{} dot_f16 len {len}", k.features);
      assert_close((k.dot_f16)(a, b), scalar::dot_f16(a, b), mag, &what);
    }
  }
}

#[test]
fn l2sq_f16_matches_scalar() {
  for k in variants() {
    let mut rng = Rng::new(9);
    for len in 0..=MAX_LEN {
      let a = rng.fill(len, random_f16);
      let b = rng.fill(len, random_f16);
      let (a, b) = (win(&a, len % 2, len), win(&b, (len / 2) % 2, len));
      let want = scalar::l2sq_f16(a, b);
      let what = format!("{} l2sq_f16 len {len}", k.features);
      assert_close((k.l2sq_f16)(a, b), want, want, &what);
    }
  }
}

/// Bitwise equality, except that any NaN matches any NaN.
fn same_f32(x: f32, y: f32) -> bool {
  x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
}

fn is_f16_nan(h: u16) -> bool {
  h & 0x7c00 == 0x7c00 && h & 0x3ff != 0
}

#[test]
fn f16_to_f32_slice_matches_scalar_for_every_pattern() {
  let all: Vec<u16> = (0..=u16::MAX).collect();
  for k in variants() {
    // Odd chunk lengths exercise every tail length.
    let mut start = 0;
    for chunk in (1..).map(|c| c % 37) {
      if start >= all.len() {
        break;
      }
      let src = &all[start..(start + chunk).min(all.len())];
      let mut dst = vec![0.0; src.len()];
      (k.f16_to_f32)(src, &mut dst);
      for (&h, &f) in src.iter().zip(&dst) {
        assert!(
          same_f32(f, scalar::f16_to_f32(h)),
          "{} {h:#06x} -> {f}",
          k.features
        );
      }
      start += src.len();
    }
  }
}

#[test]
fn f32_to_f16_slice_matches_scalar() {
  let mut rng = Rng::new(10);
  let mut src: Vec<f32> = (0..200_000)
    .map(|i| match i % 3 {
      // Any bit pattern: NaNs, infinities, subnormals, huge values.
      0 => f32::from_bits(rng.next_u64() as u32),
      // Around the binary16 range, where rounding matters.
      1 => rng.f32() * 70_000.0,
      _ => rng.f32() * 1e-4,
    })
    .collect();
  // Midpoints between consecutive binary16 values (ties to even).
  for h in (0u16..0x7bff).step_by(7) {
    let lo = scalar::f16_to_f32(h) as f64;
    let hi = scalar::f16_to_f32(h + 1) as f64;
    src.push(((lo + hi) / 2.0) as f32);
    src.push(-(((lo + hi) / 2.0) as f32));
  }
  src.extend([
    0.0,
    -0.0,
    65504.0,
    65520.0,
    f32::INFINITY,
    f32::MIN_POSITIVE,
  ]);
  for k in variants() {
    for len in [0, 1, 7, 8, 9, 15, 16, 17, 33, src.len()] {
      let s = &src[..len];
      let mut dst = vec![0u16; len];
      (k.f32_to_f16)(s, &mut dst);
      for (&x, &h) in s.iter().zip(&dst) {
        let want = scalar::f32_to_f16(x);
        if x.is_nan() {
          assert!(is_f16_nan(h), "{} {x} -> {h:#06x}", k.features);
        } else {
          assert_eq!(h, want, "{} {x:e} ({:#010x})", k.features, x.to_bits());
        }
      }
    }
  }
}

#[test]
fn isa_selection_caps_and_falls_back() {
  let only = |set: &'static [Isa]| move |isa: Isa| set.contains(&isa);
  let x86 = only(&[Isa::Avx2, Isa::Avx512]);
  assert_eq!(select(None, x86), Isa::Avx512);
  assert_eq!(select(Some(Isa::Avx512), x86), Isa::Avx512);
  assert_eq!(select(Some(Isa::Avx2), x86), Isa::Avx2);
  assert_eq!(select(Some(Isa::Scalar), x86), Isa::Scalar);
  // Not available on x86: the next one down.
  assert_eq!(select(Some(Isa::Neon), x86), Isa::Scalar);
  let avx2_only = only(&[Isa::Avx2]);
  assert_eq!(select(Some(Isa::Avx512), avx2_only), Isa::Avx2);
  assert_eq!(select(None, avx2_only), Isa::Avx2);
  let arm = only(&[Isa::Neon]);
  assert_eq!(select(None, arm), Isa::Neon);
  assert_eq!(select(Some(Isa::Avx512), arm), Isa::Neon);
  assert_eq!(select(Some(Isa::Avx2), arm), Isa::Neon);
  assert_eq!(select(Some(Isa::Scalar), arm), Isa::Scalar);
  assert_eq!(select(None, only(&[])), Isa::Scalar);
}

#[test]
fn isa_names_round_trip() {
  for isa in Isa::ALL {
    assert_eq!(Isa::parse(isa.name()), Some(isa));
    assert_eq!(Isa::parse(&isa.name().to_uppercase()), Some(isa));
  }
  assert_eq!(Isa::parse(" avx2 "), Some(Isa::Avx2));
  assert_eq!(Isa::parse("sse2"), None);
  assert!(Isa::Scalar.is_supported());
}

#[test]
fn dispatch_reports_the_truth() {
  let k = crate::kernels();
  assert_eq!(crate::active_isa(), k.isa.name());
  assert!(k.isa.is_supported());
  let requested = std::env::var("MUNINN_ISA").ok();
  let requested = requested.as_deref().and_then(Isa::parse);
  assert_eq!(k.isa, select(requested, Isa::is_supported));
  if requested.is_none() {
    assert_eq!(k.isa, Isa::best());
  }
  // `kernels_for` hands out the same table the dispatcher uses.
  assert!(std::ptr::eq(k, crate::kernels_for(k.isa).unwrap()));
  for isa in Isa::ALL {
    if let Some(t) = crate::kernels_for(isa) {
      assert_eq!(t.isa, isa);
    }
  }
}

#[test]
fn free_functions_use_the_dispatched_kernels() {
  let k = crate::kernels();
  let mut rng = Rng::new(11);
  let (a, b) = (rng.exact(300, Rng::f32), rng.exact(300, Rng::f32));
  assert_eq!(crate::dot(&a, &b).to_bits(), (k.dot)(&a, &b).to_bits());
  assert_eq!(crate::l2sq(&a, &b).to_bits(), (k.l2sq)(&a, &b).to_bits());
  for metric in METRICS {
    let mut out = [0.0; 3];
    crate::distance_batch(metric, &a[..100], &b, &mut out);
    let single = crate::distance(metric, &a[..100], &b[100..200]);
    assert_eq!(out[1].to_bits(), single.to_bits());
  }
  let (x, y) = (rng.exact(300, Rng::i8), rng.exact(300, Rng::i8));
  assert_eq!(crate::dot_i8(&x, &y), scalar::dot_i8(&x, &y));
  let (x, y) = (rng.exact(300, Rng::u8), rng.exact(300, Rng::u8));
  assert_eq!(crate::l2sq_u8(&x, &y), scalar::l2sq_u8(&x, &y));
  assert_eq!(
    crate::dot_f32_u8(&a, &x).to_bits(),
    (k.dot_f32_u8)(&a, &x).to_bits()
  );
  let (x, y) = (rng.exact(30, Rng::next_u64), rng.exact(30, Rng::next_u64));
  assert_eq!(crate::hamming(&x, &y), scalar::hamming(&x, &y));
  let (x, y) = (rng.exact(300, random_f16), rng.exact(300, random_f16));
  assert_eq!(
    crate::dot_f16(&x, &y).to_bits(),
    (k.dot_f16)(&x, &y).to_bits()
  );
  assert_eq!(
    crate::l2sq_f16(&x, &y).to_bits(),
    (k.l2sq_f16)(&x, &y).to_bits()
  );
  let mut h = vec![0u16; 300];
  crate::f32_to_f16_slice(&a, &mut h);
  let mut back = vec![0.0f32; 300];
  crate::f16_to_f32_slice(&h, &mut back);
  for ((&x, &hx), &bx) in a.iter().zip(&h).zip(&back) {
    assert_eq!(hx, crate::f32_to_f16(x));
    assert_eq!(bx.to_bits(), crate::f16_to_f32(hx).to_bits());
  }
  let mut v = a.clone();
  crate::normalize(&mut v);
  assert!((crate::norm_sq(&v) - 1.0).abs() < 1e-5);
}
