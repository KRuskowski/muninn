//! Shared helpers for the engine's integration tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A fresh directory under cargo's per-target temp dir, removed on drop.
pub struct TestDir(PathBuf);

impl TestDir {
  pub fn new(name: &str) -> TestDir {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let p = Path::new(env!("CARGO_TARGET_TMPDIR"))
      .join(format!("muninn-{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    TestDir(p)
  }

  pub fn path(&self) -> &Path {
    &self.0
  }
}

impl Drop for TestDir {
  fn drop(&mut self) {
    let _ = std::fs::remove_dir_all(&self.0);
  }
}

/// SplitMix64: small, fast, good enough for test data.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
  pub fn new(seed: u64) -> Rng {
    Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
  }

  pub fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
  }

  /// Uniform in `0..n`.
  pub fn below(&mut self, n: u64) -> u64 {
    self.next_u64() % n
  }

  /// Uniform in `[0, 1)`.
  pub fn f32(&mut self) -> f32 {
    (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
  }

  /// Approximately standard normal (sum of uniforms).
  pub fn normal(&mut self) -> f32 {
    (0..6).map(|_| self.f32()).sum::<f32>() - 3.0
  }

  pub fn chance(&mut self, p: f64) -> bool {
    (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 <= p
  }
}

/// Copies a directory tree, skipping the database lock file.
pub fn copy_dir(from: &Path, to: &Path) {
  std::fs::create_dir_all(to).unwrap();
  for e in std::fs::read_dir(from).unwrap() {
    let e = e.unwrap();
    let name = e.file_name();
    if name == "LOCK" {
      continue;
    }
    let ty = e.file_type().unwrap();
    if ty.is_dir() {
      copy_dir(&e.path(), &to.join(&name));
    } else if ty.is_file() {
      // A file can vanish between listing and copying (gc, WAL
      // truncation); a real crash would not have it either.
      let _ = std::fs::copy(e.path(), to.join(&name));
    }
  }
}

/// Files in `dir` whose names start with `prefix`, sorted.
pub fn files_with_prefix(dir: &Path, prefix: &str) -> Vec<PathBuf> {
  let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
    .unwrap()
    .map(|e| e.unwrap().path())
    .filter(|p| {
      p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with(prefix))
    })
    .collect();
  v.sort();
  v
}
