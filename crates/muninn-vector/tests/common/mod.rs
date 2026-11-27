//! Shared helpers for the integration tests: a deterministic PRNG,
//! clustered synthetic data, recall, and a bounded thread pool.

#![allow(dead_code)]

use muninn_core::{BitSet, DocId, Hit};

/// SplitMix64: tiny, fast, and identical on every platform.
pub struct Rng(u64);

impl Rng {
  pub fn new(seed: u64) -> Self {
    Self(seed)
  }

  pub fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  /// Uniform in `[0, 1)`.
  pub fn uniform(&mut self) -> f64 {
    (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
  }

  /// Standard normal (Box-Muller).
  pub fn gauss(&mut self) -> f32 {
    let u = 1.0 - self.uniform();
    let v = self.uniform();
    ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
  }
}

/// Gaussian clusters: `clusters` centres drawn from N(0, 1) per
/// dimension, points scattered around a random centre with standard
/// deviation `spread`. Row-major, `dim` floats per row.
pub struct Clusters {
  dim: usize,
  centres: Vec<f32>,
  spread: f32,
}

impl Clusters {
  pub fn new(dim: usize, clusters: usize, spread: f32, seed: u64) -> Self {
    let mut r = Rng::new(seed);
    let centres = (0..dim * clusters).map(|_| r.gauss()).collect();
    Self {
      dim,
      centres,
      spread,
    }
  }

  /// `n` points; different `seed`s give independent samples (queries).
  pub fn sample(&self, n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    let k = self.centres.len() / self.dim;
    let mut out = Vec::with_capacity(n * self.dim);
    for _ in 0..n {
      let c = (r.next_u64() % k as u64) as usize;
      let centre = &self.centres[c * self.dim..(c + 1) * self.dim];
      out.extend(centre.iter().map(|&x| x + self.spread * r.gauss()));
    }
    out
  }
}

/// Fraction of the exact top-k ids that `got` found.
pub fn recall(got: &[Hit], truth: &[Hit]) -> f64 {
  if truth.is_empty() {
    return 1.0;
  }
  let found = truth
    .iter()
    .filter(|t| got.iter().any(|g| g.id == t.id))
    .count();
  found as f64 / truth.len() as f64
}

/// A set holding about `fraction` of `0..n`, chosen pseudo-randomly.
pub fn random_set(n: usize, fraction: f64, seed: u64) -> BitSet {
  let mut r = Rng::new(seed);
  let mut s = BitSet::new(n);
  for id in 0..n {
    if r.uniform() < fraction {
      s.insert(id as DocId);
    }
  }
  s
}

/// Caps rayon at 8 threads (or `RAYON_NUM_THREADS` if smaller): the test
/// machines are shared.
pub fn init_pool() {
  let want = std::env::var("RAYON_NUM_THREADS")
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(8)
    .clamp(1, 8);
  let _ = rayon::ThreadPoolBuilder::new()
    .num_threads(want)
    .build_global();
}
