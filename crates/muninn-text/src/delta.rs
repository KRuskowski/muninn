//! Delta decoding of doc-id blocks: validation and prefix sums.
//!
//! [`check`] summarises a block's deltas (smallest and total) so the
//! caller can validate it before trusting any id; [`prefix_sum`] then
//! turns deltas into ids. With AVX2 (detected at run time) the prefix sum
//! runs eight lanes at a time; [`prefix_sum_scalar`] is the reference it
//! is tested against.

use crate::bitpack::BLOCK;

/// Smallest of `v[from..]` and the sum of all of `v`, widened.
#[inline]
pub(crate) fn check(v: &[u32], from: usize) -> (u32, u64) {
  let min = v.get(from..).unwrap_or(&[]).iter().copied().min();
  let total = v.iter().map(|&d| u64::from(d)).sum();
  (min.unwrap_or(u32::MAX), total)
}

/// `v[k] = base + v[0] + ... + v[k]` (wrapping) for every `k`.
#[inline]
pub(crate) fn prefix_sum(v: &mut [u32; BLOCK], base: u32) {
  #[cfg(target_arch = "x86_64")]
  if std::arch::is_x86_feature_detected!("avx2") {
    // SAFETY: AVX2 is available, checked just above.
    unsafe { prefix_sum_avx2(v, base) };
    return;
  }
  prefix_sum_scalar(v, base);
}

/// The scalar reference of [`prefix_sum`].
#[inline]
pub(crate) fn prefix_sum_scalar(v: &mut [u32; BLOCK], base: u32) {
  let mut acc = base;
  for d in v.iter_mut() {
    acc = acc.wrapping_add(*d);
    *d = acc;
  }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn prefix_sum_avx2(v: &mut [u32; BLOCK], base: u32) {
  use std::arch::x86_64::*;
  let mut carry = _mm256_set1_epi32(base as i32);
  let last = _mm256_set1_epi32(7);
  for chunk in v.chunks_exact_mut(8) {
    let p = chunk.as_mut_ptr().cast::<__m256i>();
    // SAFETY: `chunk` is 8 `u32`s, exactly one unaligned 256-bit load.
    let mut x = unsafe { _mm256_loadu_si256(p) };
    // Inclusive scan inside each 128-bit lane...
    x = _mm256_add_epi32(x, _mm256_slli_si256::<4>(x));
    x = _mm256_add_epi32(x, _mm256_slli_si256::<8>(x));
    // ...then add the low lane's total to the high lane.
    let low_total = _mm256_shuffle_epi32::<0xFF>(x);
    x = _mm256_add_epi32(
      x,
      _mm256_permute2x128_si256::<0x08>(low_total, low_total),
    );
    x = _mm256_add_epi32(x, carry);
    // SAFETY: as for the load.
    unsafe { _mm256_storeu_si256(p, x) };
    carry = _mm256_permutevar8x32_epi32(x, last);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn prefix_sum_matches_scalar_reference() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for round in 0..2000 {
      let mut v = [0u32; BLOCK];
      for x in v.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Small deltas, large ones and some that wrap.
        *x = match round % 3 {
          0 => (state % 7) as u32,
          1 => (state % 100_000) as u32,
          _ => state as u32,
        };
      }
      let base = (state >> 40) as u32;
      let mut a = v;
      let mut b = v;
      prefix_sum(&mut a, base);
      prefix_sum_scalar(&mut b, base);
      assert_eq!(a, b, "round {round}");
      let (mn, total) = check(&v[..77], 1);
      assert_eq!(mn, v[1..77].iter().copied().min().unwrap());
      assert_eq!(total, v[..77].iter().map(|&d| u64::from(d)).sum::<u64>());
    }
    assert_eq!(check(&[5], 1), (u32::MAX, 5));
  }
}
