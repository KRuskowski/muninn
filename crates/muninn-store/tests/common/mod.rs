//! Shared helpers for integration tests.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A fresh directory under cargo's per-target temp dir (never `/tmp`),
/// removed on drop.
pub struct TestDir(PathBuf);

impl TestDir {
  pub fn new(name: &str) -> TestDir {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let p = Path::new(env!("CARGO_TARGET_TMPDIR"))
      .join(format!("muninn-store-{name}-{}-{n}", std::process::id()));
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
