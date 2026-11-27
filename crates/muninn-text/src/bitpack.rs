//! Fixed-width bit packing of up to 128 `u32` values.
//!
//! Values are laid out little-endian, back to back, `w` bits each, so a
//! block of 128 values at width `w` takes exactly `16 * w` bytes. The
//! decoder reads one unaligned `u64` per value and is monomorphised per
//! width so every shift and mask is a constant; LLVM unrolls and
//! vectorises it. It may read up to 8 bytes past the packed data, so
//! callers either guarantee that slack or go through [`unpack`], which
//! copies short inputs into a padded buffer.

/// Values per block.
pub(crate) const BLOCK: usize = 128;

/// Bytes needed to pack `n` values of width `w`.
#[inline]
pub(crate) fn packed_len(n: usize, w: u32) -> usize {
  (n * w as usize).div_ceil(8)
}

/// Bit width needed to represent `max`.
#[inline]
pub(crate) fn bits_needed(max: u32) -> u32 {
  32 - max.leading_zeros()
}

/// Appends `values` (at most [`BLOCK`]) packed at width `w`. Every value
/// must fit in `w` bits.
pub(crate) fn pack(values: &[u32], w: u32, out: &mut Vec<u8>) {
  debug_assert!(values.len() <= BLOCK && w <= 32);
  let start = out.len();
  let len = packed_len(values.len(), w);
  out.resize(start + len + 8, 0);
  if w > 0 {
    let dst = &mut out[start..];
    for (i, &v) in values.iter().enumerate() {
      debug_assert!(w == 32 || v >> w == 0, "{v} does not fit {w} bits");
      let bit = i * w as usize;
      let b = bit >> 3;
      let mut word = u64::from_le_bytes(dst[b..b + 8].try_into().unwrap());
      word |= u64::from(v) << (bit & 7);
      dst[b..b + 8].copy_from_slice(&word.to_le_bytes());
    }
  }
  out.truncate(start + len);
}

#[inline(always)]
fn unpack_w<const W: u32>(src: &[u8], out: &mut [u32; BLOCK]) {
  if W == 0 {
    out.fill(0);
    return;
  }
  // One length check up front lets LLVM drop the per-value checks: the
  // last read starts at byte (127 * W) / 8 < 16 * W.
  let src = &src[..16 * W as usize + 8];
  let mask = if W == 32 {
    u64::from(u32::MAX)
  } else {
    (1u64 << W) - 1
  };
  for (i, o) in out.iter_mut().enumerate() {
    let bit = i * W as usize;
    let b = bit >> 3;
    let word = u64::from_le_bytes(src[b..b + 8].try_into().unwrap());
    *o = ((word >> (bit & 7)) & mask) as u32;
  }
}

macro_rules! dispatch {
  ($w:expr, $src:expr, $out:expr, $($n:literal)*) => {
    match $w {
      $($n => unpack_w::<$n>($src, $out),)*
      _ => unreachable!("width checked by caller"),
    }
  };
}

/// Decodes 128 values of width `w` from `src`, which must hold at least
/// `16 * w + 8` bytes. Values past the ones actually packed are garbage.
#[inline]
fn unpack_fast(src: &[u8], w: u32, out: &mut [u32; BLOCK]) {
  dispatch!(
    w, src, out, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
    22 23 24 25 26 27 28 29 30 31 32
  )
}

/// Decodes `n` values of width `w` from the start of `src` into
/// `out[..n]` (entries past `n` are unspecified). Returns `false` if `w`
/// is invalid or `src` is too short, which only happens on corrupt data.
#[inline]
pub(crate) fn unpack(
  src: &[u8],
  w: u32,
  n: usize,
  out: &mut [u32; BLOCK],
) -> bool {
  if w > 32 || n > BLOCK {
    return false;
  }
  let need = packed_len(n, w);
  if src.len() >= 16 * w as usize + 8 {
    unpack_fast(src, w, out);
  } else if src.len() >= need {
    let mut buf = [0u8; 16 * 32 + 8];
    buf[..need].copy_from_slice(&src[..need]);
    unpack_fast(&buf, w, out);
  } else {
    return false;
  }
  true
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Bit-by-bit reference decoder.
  fn reference_unpack(src: &[u8], w: u32, n: usize) -> Vec<u32> {
    (0..n)
      .map(|i| {
        let mut v = 0u32;
        for bit in 0..w as usize {
          let at = i * w as usize + bit;
          if (src[at / 8] >> (at % 8)) & 1 == 1 {
            v |= 1 << bit;
          }
        }
        v
      })
      .collect()
  }

  #[test]
  fn fast_unpack_matches_reference_for_every_width() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    for w in 0..=32u32 {
      for n in [0usize, 1, 2, 7, 8, 63, 127, 128] {
        let mask = if w == 32 { u32::MAX } else { (1u32 << w) - 1 };
        let values: Vec<u32> = (0..n).map(|_| (rnd() as u32) & mask).collect();
        let mut packed = Vec::new();
        pack(&values, w, &mut packed);
        assert_eq!(packed.len(), packed_len(n, w));
        assert_eq!(reference_unpack(&packed, w, n), values);
        // Exact-length input exercises the padded-copy path; trailing
        // slack exercises the fast path.
        let mut out = [0u32; BLOCK];
        assert!(unpack(&packed, w, n, &mut out));
        assert_eq!(&out[..n], &values[..], "w={w} n={n}");
        let mut slack = packed.clone();
        slack.resize(packed.len() + 16 * 32 + 8, 0xAA);
        let mut out2 = [0u32; BLOCK];
        assert!(unpack(&slack, w, n, &mut out2));
        assert_eq!(&out2[..n], &values[..], "w={w} n={n} fast");
      }
    }
  }

  #[test]
  fn rejects_bad_width_and_short_input() {
    let mut out = [0u32; BLOCK];
    assert!(!unpack(&[0; 1000], 33, 1, &mut out));
    assert!(!unpack(&[0; 3], 8, 4, &mut out));
    assert!(unpack(&[], 0, 128, &mut out));
    assert_eq!(bits_needed(0), 0);
    assert_eq!(bits_needed(1), 1);
    assert_eq!(bits_needed(u32::MAX), 32);
  }
}
