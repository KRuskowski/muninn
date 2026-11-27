//! Fast, stable hashing for id lookups.
//!
//! The on-disk id-map tables (see [`crate::idmap`]) store the seed they
//! were built with and hash with exactly these functions, so they are part
//! of the file format: changing them requires a format version bump.

use std::hash::{BuildHasher, Hasher};

const K: u64 = 0x9E37_79B9_7F4A_7C15;

/// MurmurHash3's 64-bit finaliser: full avalanche in five operations.
#[inline]
pub(crate) fn fmix64(mut x: u64) -> u64 {
  x ^= x >> 33;
  x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
  x ^= x >> 33;
  x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
  x ^ (x >> 33)
}

/// Hash of a `u64` id.
#[inline]
pub(crate) fn hash_u64(x: u64, seed: u64) -> u64 {
  fmix64(x ^ seed)
}

/// Hash of a byte string, eight bytes per step.
#[inline]
pub(crate) fn hash_bytes(b: &[u8], seed: u64) -> u64 {
  let mut h = seed ^ (b.len() as u64).wrapping_mul(K);
  let mut chunks = b.chunks_exact(8);
  for c in &mut chunks {
    let w = u64::from_le_bytes(c.try_into().unwrap());
    h = (h ^ w).wrapping_mul(K).rotate_left(31);
  }
  let rest = chunks.remainder();
  if !rest.is_empty() {
    let mut tail = [0u8; 8];
    tail[..rest.len()].copy_from_slice(rest);
    h = (h ^ u64::from_le_bytes(tail))
      .wrapping_mul(K)
      .rotate_left(31);
  }
  fmix64(h)
}

/// [`BuildHasher`] for in-memory id maps: the hashes above with a seed
/// drawn from the standard library's per-process randomness, so user ids
/// cannot be chosen to collide.
#[derive(Clone, Debug)]
pub(crate) struct IdBuildHasher {
  seed: u64,
}

impl Default for IdBuildHasher {
  fn default() -> Self {
    let seed = std::collections::hash_map::RandomState::new().hash_one(K);
    Self { seed }
  }
}

impl BuildHasher for IdBuildHasher {
  type Hasher = IdHasher;

  fn build_hasher(&self) -> IdHasher {
    IdHasher(self.seed)
  }
}

/// Streaming form of [`hash_u64`] / [`hash_bytes`].
#[derive(Clone, Debug)]
pub(crate) struct IdHasher(u64);

impl Hasher for IdHasher {
  #[inline]
  fn finish(&self) -> u64 {
    self.0
  }

  #[inline]
  fn write(&mut self, bytes: &[u8]) {
    self.0 = hash_bytes(bytes, self.0);
  }

  #[inline]
  fn write_u8(&mut self, i: u8) {
    self.0 = hash_u64(i as u64, self.0);
  }

  #[inline]
  fn write_u64(&mut self, i: u64) {
    self.0 = hash_u64(i, self.0);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bytes_hash_depends_on_every_byte_and_length() {
    let base = b"document-000123".to_vec();
    let h = hash_bytes(&base, 1);
    for i in 0..base.len() {
      let mut v = base.clone();
      v[i] ^= 1;
      assert_ne!(hash_bytes(&v, 1), h);
    }
    assert_ne!(hash_bytes(b"ab", 1), hash_bytes(b"ab\0", 1));
    assert_ne!(hash_bytes(b"ab", 1), hash_bytes(b"ab", 2));
  }

  #[test]
  fn u64_hash_spreads_low_bits() {
    let mask = 1023u64;
    let mut seen = vec![0u32; 1024];
    for i in 0..1024u64 {
      seen[(hash_u64(i, 0) & mask) as usize] += 1;
    }
    // Sequential keys must not pile into a few buckets.
    assert!(seen.iter().all(|&c| c < 10));
  }
}
