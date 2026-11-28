//! Portable reference kernels. Correct for any input; the SIMD kernels
//! must agree with these within floating-point reassociation error, and
//! exactly for the integer kernels.

use muninn_core::Metric;

/// Dot product.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  // Four independent accumulators so even the fallback pipelines.
  let mut acc = [0.0f32; 4];
  let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
  let (ra, rb) = (ca.remainder(), cb.remainder());
  for (x, y) in ca.zip(cb) {
    for i in 0..4 {
      acc[i] += x[i] * y[i];
    }
  }
  let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  for (x, y) in ra.iter().zip(rb) {
    s += x * y;
  }
  s
}

/// Squared Euclidean distance.
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  let mut acc = [0.0f32; 4];
  let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
  let (ra, rb) = (ca.remainder(), cb.remainder());
  for (x, y) in ca.zip(cb) {
    for i in 0..4 {
      let d = x[i] - y[i];
      acc[i] += d * d;
    }
  }
  let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  for (x, y) in ra.iter().zip(rb) {
    let d = x - y;
    s += d * d;
  }
  s
}

/// Distances from `query` to consecutive rows, see
/// [`crate::distance_batch`].
pub fn distance_batch(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  out: &mut [f32],
) {
  let dim = query.len();
  debug_assert_eq!(rows.len(), dim * out.len());
  if dim == 0 {
    out.fill(crate::finish(metric, 0.0));
    return;
  }
  for (o, row) in out.iter_mut().zip(rows.chunks_exact(dim)) {
    *o = match metric {
      Metric::L2 => l2sq(query, row),
      Metric::Dot | Metric::Cosine => crate::finish(metric, dot(query, row)),
    };
  }
}

/// Distances from `query` to rows picked by id, see
/// [`crate::distance_gather`].
pub fn distance_gather(
  metric: Metric,
  query: &[f32],
  rows: &[f32],
  ids: &[u32],
  out: &mut [f32],
) {
  let dim = query.len();
  for (o, &id) in out.iter_mut().zip(ids) {
    let s = id as usize * dim;
    let Some(row) = rows.get(s..s + dim) else {
      *o = f32::NAN;
      continue;
    };
    *o = match metric {
      Metric::L2 => l2sq(query, row),
      Metric::Dot | Metric::Cosine => crate::finish(metric, dot(query, row)),
    };
  }
}

/// Dot product of `f32` weights with `u8` codes widened to `f32`, see
/// [`crate::dot_f32_u8`].
pub fn dot_f32_u8(a: &[f32], b: &[u8]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  let mut acc = [0.0f32; 4];
  let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
  let (ra, rb) = (ca.remainder(), cb.remainder());
  for (x, y) in ca.zip(cb) {
    for i in 0..4 {
      acc[i] += x[i] * y[i] as f32;
    }
  }
  let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  for (x, &y) in ra.iter().zip(rb) {
    s += x * y as f32;
  }
  s
}

/// Dot products of `w` with code rows picked by id, see
/// [`crate::dot_f32_u8_gather`]. A bad id scores NaN.
pub fn dot_f32_u8_gather(
  w: &[f32],
  codes: &[u8],
  ids: &[u32],
  out: &mut [f32],
) {
  let dim = w.len();
  for (o, &id) in out.iter_mut().zip(ids) {
    let s = id as usize * dim;
    *o = match codes.get(s..s + dim) {
      Some(r) => dot_f32_u8(w, r),
      None => f32::NAN,
    };
  }
}

/// `i8` dot product in `i32`. The sum wraps (mod 2^32) instead of
/// overflowing, like every SIMD implementation, see [`crate::dot_i8`].
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  a.iter()
    .zip(b)
    .fold(0i32, |s, (&x, &y)| s.wrapping_add(x as i32 * y as i32))
}

/// `u8` squared L2 distance in `u32`. The sum wraps (mod 2^32) instead of
/// overflowing, like every SIMD implementation, see [`crate::l2sq_u8`].
pub fn l2sq_u8(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  a.iter().zip(b).fold(0u32, |s, (&x, &y)| {
    let d = x as i32 - y as i32;
    s.wrapping_add((d * d) as u32)
  })
}

/// Hamming distance over packed bits. The sum wraps (mod 2^32) instead of
/// overflowing, like every SIMD implementation, see [`crate::hamming`].
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  a.iter()
    .zip(b)
    .fold(0u32, |s, (x, y)| s.wrapping_add((x ^ y).count_ones()))
}

/// Half-precision dot product, computed in `f32`.
pub fn dot_f16(a: &[u16], b: &[u16]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  let mut acc = [0.0f32; 4];
  let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
  let (ra, rb) = (ca.remainder(), cb.remainder());
  for (x, y) in ca.zip(cb) {
    for i in 0..4 {
      acc[i] += f16_to_f32(x[i]) * f16_to_f32(y[i]);
    }
  }
  let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  for (&x, &y) in ra.iter().zip(rb) {
    s += f16_to_f32(x) * f16_to_f32(y);
  }
  s
}

/// Half-precision squared Euclidean distance, computed in `f32`.
pub fn l2sq_f16(a: &[u16], b: &[u16]) -> f32 {
  debug_assert_eq!(a.len(), b.len());
  let mut acc = [0.0f32; 4];
  let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
  let (ra, rb) = (ca.remainder(), cb.remainder());
  for (x, y) in ca.zip(cb) {
    for i in 0..4 {
      let d = f16_to_f32(x[i]) - f16_to_f32(y[i]);
      acc[i] += d * d;
    }
  }
  let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  for (&x, &y) in ra.iter().zip(rb) {
    let d = f16_to_f32(x) - f16_to_f32(y);
    s += d * d;
  }
  s
}

/// Converts a binary16 bit pattern to `f32`. Exact; NaN inputs give a
/// quiet NaN with the same sign and the payload shifted up.
pub fn f16_to_f32(h: u16) -> f32 {
  let sign = ((h & 0x8000) as u32) << 16;
  let exp = (h >> 10) & 0x1f;
  let man = (h & 0x3ff) as u32;
  match (exp, man) {
    (0, 0) => f32::from_bits(sign),
    (0, _) => {
      // Subnormal: man * 2^-24, exact in f32.
      let v = man as f32 * (1.0 / 16_777_216.0);
      f32::from_bits(v.to_bits() | sign)
    }
    (0x1f, 0) => f32::from_bits(sign | 0x7f80_0000),
    (0x1f, _) => f32::from_bits(sign | 0x7fc0_0000 | (man << 13)),
    _ => f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13)),
  }
}

/// Converts an `f32` to the nearest binary16 bit pattern, ties to even,
/// the same rounding as the F16C and AArch64 conversion instructions.
/// Overflow gives infinity; NaN gives a quiet NaN.
pub fn f32_to_f16(x: f32) -> u16 {
  let b = x.to_bits();
  let sign = ((b >> 16) & 0x8000) as u16;
  let exp = ((b >> 23) & 0xff) as i32;
  let man = b & 0x7f_ffff;
  if exp == 0xff {
    return if man == 0 {
      sign | 0x7c00
    } else {
      sign | 0x7e00 | (man >> 13) as u16
    };
  }
  // Biased binary16 exponent.
  let e = exp - 127 + 15;
  if e >= 0x1f {
    return sign | 0x7c00;
  }
  if e <= 0 {
    // Subnormal or zero result. Below half the smallest subnormal
    // everything rounds to zero.
    if e < -10 {
      return sign;
    }
    let m = man | 0x80_0000;
    let shift = (14 - e) as u32;
    let half = 1u32 << (shift - 1);
    let rem = m & ((1u32 << shift) - 1);
    let mut r = m >> shift;
    if rem > half || (rem == half && r & 1 == 1) {
      r += 1;
    }
    return sign | r as u16;
  }
  let mut r = ((e as u32) << 10) | (man >> 13);
  let rem = man & 0x1fff;
  // A carry out of the mantissa correctly bumps the exponent, up to
  // infinity.
  if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
    r += 1;
  }
  sign | r as u16
}

/// Converts binary16 bit patterns to `f32`, element by element.
pub fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
  debug_assert_eq!(src.len(), dst.len());
  for (d, &s) in dst.iter_mut().zip(src) {
    *d = f16_to_f32(s);
  }
}

/// Converts `f32` values to binary16 bit patterns, element by element.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
  debug_assert_eq!(src.len(), dst.len());
  for (d, &s) in dst.iter_mut().zip(src) {
    *d = f32_to_f16(s);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn small_known_values() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [5.0, 4.0, 3.0, 2.0, 1.0];
    assert_eq!(dot(&a, &b), 35.0);
    assert_eq!(l2sq(&a, &b), 40.0);
    assert_eq!(dot(&[], &[]), 0.0);
    assert_eq!(dot_i8(&[-128, 127], &[-128, 127]), 16384 + 16129);
    assert_eq!(l2sq_u8(&[0, 255], &[255, 0]), 2 * 255 * 255);
    assert_eq!(hamming(&[0b1011, u64::MAX], &[0b0001, 0]), 2 + 64);
  }

  #[test]
  fn f16_known_values() {
    let cases: [(u16, f32); 10] = [
      (0x0000, 0.0),
      (0x3c00, 1.0),
      (0xc000, -2.0),
      (0x3555, 1365.0 / 4096.0),
      (0x7bff, 65504.0),
      (0x0400, 1.0 / 16384.0),
      (0x0001, 1.0 / 16_777_216.0),
      (0x03ff, 1023.0 / 16_777_216.0),
      (0x7c00, f32::INFINITY),
      (0xfc00, f32::NEG_INFINITY),
    ];
    for (h, f) in cases {
      assert_eq!(f16_to_f32(h), f, "{h:#06x}");
      assert_eq!(f32_to_f16(f), h, "{f}");
    }
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert!(f16_to_f32(0x7e00).is_nan());
    assert!(f16_to_f32(0x7c01).is_nan());
    assert_eq!(f32_to_f16(f32::NAN) & 0x7e00, 0x7e00);
    // Rounding: ties to even, overflow to infinity, underflow to zero.
    assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
    assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
    assert_eq!(f32_to_f16(65520.0), 0x7c00);
    assert_eq!(f32_to_f16(65519.0), 0x7bff);
    assert_eq!(f32_to_f16(2.0f32.powi(-25)), 0x0000);
    assert_eq!(f32_to_f16(2.0f32.powi(-25) * 1.000_001), 0x0001);
    assert_eq!(f32_to_f16(-1e-30), 0x8000);
  }

  #[test]
  fn f16_round_trips_every_non_nan_pattern() {
    for h in 0..=u16::MAX {
      let f = f16_to_f32(h);
      if f.is_nan() {
        assert!(h & 0x7c00 == 0x7c00 && h & 0x3ff != 0);
        assert_eq!(f32_to_f16(f) & 0x7e00, 0x7e00);
      } else {
        assert_eq!(f32_to_f16(f), h, "{h:#06x}");
      }
    }
  }

  #[test]
  fn f32_to_f16_rounds_to_nearest() {
    // Every f32 between consecutive positive binary16 values must map to
    // the nearer one (ties to even). Check midpoints and their
    // neighbours for every finite positive binary16.
    for h in 0u16..0x7bff {
      let lo = f16_to_f32(h) as f64;
      let hi = f16_to_f32(h + 1) as f64;
      let mid = ((lo + hi) / 2.0) as f32;
      let even = if h & 1 == 0 { h } else { h + 1 };
      assert_eq!(f32_to_f16(mid), even, "tie above {h:#06x}");
      let below = f32::from_bits(mid.to_bits() - 1);
      let above = f32::from_bits(mid.to_bits() + 1);
      assert_eq!(f32_to_f16(below), h, "below mid of {h:#06x}");
      assert_eq!(f32_to_f16(above), h + 1, "above mid of {h:#06x}");
    }
  }
}
