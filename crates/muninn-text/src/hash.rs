//! A fast, seeded hasher for the in-memory term dictionary.
//!
//! The standard SipHash is slow for short keys. This is a folded-multiply
//! hash (the construction used by wyhash/foldhash) with a per-process
//! random seed taken from the standard library, so crafted inputs cannot
//! precompute collisions across processes.

use std::hash::{BuildHasher, Hasher};
use std::sync::OnceLock;

const K0: u64 = 0xa076_1d64_78bd_642f;
const K1: u64 = 0xe703_7ed1_a0b4_28db;

#[inline]
fn fold_mul(a: u64, b: u64) -> u64 {
  let p = u128::from(a) * u128::from(b);
  (p as u64) ^ ((p >> 64) as u64)
}

fn process_seed() -> u64 {
  static SEED: OnceLock<u64> = OnceLock::new();
  *SEED.get_or_init(|| {
    std::collections::hash_map::RandomState::new().hash_one(0x6d75_6e69u64)
  })
}

#[derive(Clone, Copy)]
pub(crate) struct FoldHasher {
  h: u64,
}

impl Hasher for FoldHasher {
  #[inline]
  fn write(&mut self, bytes: &[u8]) {
    let mut h = self.h ^ (bytes.len() as u64).wrapping_mul(K1);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
      let w = u64::from_le_bytes(c.try_into().unwrap());
      h = fold_mul(h ^ w, K0);
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
      let mut buf = [0u8; 8];
      buf[..rest.len()].copy_from_slice(rest);
      h = fold_mul(h ^ u64::from_le_bytes(buf), K1);
    }
    self.h = h;
  }

  #[inline]
  fn write_u8(&mut self, i: u8) {
    self.h = fold_mul(self.h ^ u64::from(i), K0);
  }

  #[inline]
  fn finish(&self) -> u64 {
    fold_mul(self.h, K1)
  }
}

/// Builds [`FoldHasher`]s sharing the process seed.
#[derive(Clone, Copy, Default)]
pub(crate) struct FoldBuild;

impl BuildHasher for FoldBuild {
  type Hasher = FoldHasher;

  #[inline]
  fn build_hasher(&self) -> FoldHasher {
    FoldHasher { h: process_seed() }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn distinguishes_lengths_and_content() {
    let b = FoldBuild;
    let hs: Vec<u64> = ["", "a", "a\0", "ab", "abcdefgh", "abcdefghi"]
      .iter()
      .map(|s| b.hash_one(s))
      .collect();
    for i in 0..hs.len() {
      for j in i + 1..hs.len() {
        assert_ne!(hs[i], hs[j]);
      }
    }
    assert_eq!(b.hash_one("same"), b.hash_one("same"));
  }
}
