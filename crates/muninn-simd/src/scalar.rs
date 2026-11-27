//! Portable reference kernels. Correct for any input; the SIMD kernels
//! must agree with these within floating-point reassociation error.

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

/// `i8` dot product in `i32`.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
  debug_assert_eq!(a.len(), b.len());
  a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// `u8` squared L2 distance in `u32`.
pub fn l2sq_u8(a: &[u8], b: &[u8]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  a.iter()
    .zip(b)
    .map(|(&x, &y)| {
      let d = x as i32 - y as i32;
      (d * d) as u32
    })
    .sum()
}

/// Hamming distance over packed bits.
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
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
}
