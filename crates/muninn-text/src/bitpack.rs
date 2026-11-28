//! Fixed-width bit packing of up to 128 `u32` values, in two layouts.
//!
//! * **Plain** ([`pack`], [`unpack`]): values back to back, little-endian,
//!   `w` bits each; `n` values take `ceil(n * w / 8)` bytes. The decoder
//!   reads one unaligned `u64` per value, may read up to 8 bytes past the
//!   packed data (callers guarantee that slack or go through [`unpack`],
//!   which copies short inputs into a padded buffer). Used for partial
//!   blocks, and for every block of format-1 files.
//! * **Lanes** ([`pack_lanes`], [`unpack_lanes`]): exactly 128 values in
//!   four interleaved lanes. Value `i` is the `i / 4`-th value of lane
//!   `i % 4`; each lane is a little-endian bit stream of `u32` words, and
//!   word `k` of lane `l` is the `(4k + l)`-th word of the output. Also
//!   `16 * w` bytes, but decoding needs no gathers: every row of four
//!   consecutive values is one 16-byte load, constant shifts and a mask,
//!   which vectorises on any SIMD width. Used for full blocks.
//!
//! Both decoders are monomorphised per width so every shift and mask is
//! a constant.

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

/// Lanes of the interleaved layout.
const LANES: usize = 4;

/// Appends exactly [`BLOCK`] values packed at width `w` in the lanes
/// layout (`16 * w` bytes). Every value must fit in `w` bits.
pub(crate) fn pack_lanes(values: &[u32; BLOCK], w: u32, out: &mut Vec<u8>) {
  debug_assert!(w <= 32);
  let w = w as usize;
  let mut words = [0u32; BLOCK];
  for (i, &v) in values.iter().enumerate() {
    debug_assert!(w == 32 || v >> w == 0, "{v} does not fit {w} bits");
    let (lane, row) = (i % LANES, i / LANES);
    let bit = row * w;
    let (k, s) = (bit / 32, bit % 32);
    if w > 0 {
      words[k * LANES + lane] |= v << s;
      if s + w > 32 {
        words[(k + 1) * LANES + lane] |= v >> (32 - s);
      }
    }
  }
  for x in &words[..w * LANES] {
    out.extend_from_slice(&x.to_le_bytes());
  }
}

#[inline(always)]
fn unpack_lanes_w<const W: u32>(src: &[u8], out: &mut [u32; BLOCK]) {
  let w = W as usize;
  if w == 0 {
    out.fill(0);
    return;
  }
  // One length check up front lets LLVM drop the per-word checks.
  let src = &src[..16 * w];
  let word = |k: usize, lane: usize| {
    let at = (k * LANES + lane) * 4;
    u32::from_le_bytes(src[at..at + 4].try_into().unwrap())
  };
  let mask = if w == 32 { u32::MAX } else { (1u32 << w) - 1 };
  for (row, o) in out.chunks_exact_mut(LANES).enumerate() {
    let bit = row * w;
    let (k, s) = (bit / 32, bit % 32);
    for (lane, o) in o.iter_mut().enumerate() {
      let mut v = word(k, lane) >> s;
      if s + w > 32 {
        v |= word(k + 1, lane) << (32 - s);
      }
      *o = v & mask;
    }
  }
}

macro_rules! dispatch_lanes {
  ($w:expr, $src:expr, $out:expr, $($n:literal)*) => {
    match $w {
      $($n => unpack_lanes_w::<$n>($src, $out),)*
      _ => unreachable!("width checked by caller"),
    }
  };
}

/// Decodes [`BLOCK`] values of width `w` in the lanes layout from the
/// start of `src`. Returns `false` if `w` is invalid or `src` is shorter
/// than `16 * w` bytes, which only happens on corrupt data.
#[inline]
pub(crate) fn unpack_lanes(src: &[u8], w: u32, out: &mut [u32; BLOCK]) -> bool {
  if w > 32 || src.len() < 16 * w as usize {
    return false;
  }
  dispatch_lanes!(
    w, src, out, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
    22 23 24 25 26 27 28 29 30 31 32
  );
  true
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

  /// Bit-by-bit reference decoder of the lanes layout.
  fn reference_unpack_lanes(src: &[u8], w: u32) -> Vec<u32> {
    let w = w as usize;
    (0..BLOCK)
      .map(|i| {
        let (lane, row) = (i % LANES, i / LANES);
        let mut v = 0u32;
        for bit in 0..w {
          let at = row * w + bit;
          let word = (at / 32) * LANES + lane;
          let byte = word * 4 + (at % 32) / 8;
          if (src[byte] >> (at % 8)) & 1 == 1 {
            v |= 1 << bit;
          }
        }
        v
      })
      .collect()
  }

  #[test]
  fn lanes_unpack_matches_reference_for_every_width() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for w in 0..=32u32 {
      for round in 0..20 {
        let mask = if w == 32 { u32::MAX } else { (1u32 << w) - 1 };
        let mut values = [0u32; BLOCK];
        for v in values.iter_mut() {
          state ^= state << 13;
          state ^= state >> 7;
          state ^= state << 17;
          // Some rounds use only the extremes, to fill every bit.
          *v = match round % 3 {
            0 => mask,
            _ => (state as u32) & mask,
          };
        }
        let mut packed = Vec::new();
        pack_lanes(&values, w, &mut packed);
        assert_eq!(packed.len(), 16 * w as usize);
        assert_eq!(reference_unpack_lanes(&packed, w), values.to_vec());
        let mut out = [0u32; BLOCK];
        assert!(unpack_lanes(&packed, w, &mut out));
        assert_eq!(out, values, "w={w}");
        if w > 0 {
          assert!(!unpack_lanes(&packed[..packed.len() - 1], w, &mut out));
        }
      }
    }
    assert!(!unpack_lanes(&[0; 1000], 33, &mut [0; BLOCK]));
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
