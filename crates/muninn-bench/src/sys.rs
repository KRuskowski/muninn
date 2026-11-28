//! Machine description and memory figures recorded with every result.
//!
//! What the process cannot find out by itself (git commit, compiler
//! flags, cloud machine type, CPU pinning) is passed in by
//! `scripts/bench_all.sh` through `BENCH_*` environment variables.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// CPU feature flags worth recording for SIMD-heavy code.
const FLAGS: &[&str] = &[
  "sse4_2",
  "avx",
  "avx2",
  "fma",
  "f16c",
  "avx512f",
  "avx512bw",
  "avx512vl",
  "avx512dq",
  "avx512_vnni",
  "avx512_bf16",
  "avx512_fp16",
  "avx_vnni",
  "amx_tile",
];

/// The machine and build a result came from.
#[derive(Clone, Debug, Serialize)]
pub struct SystemInfo {
  /// Host name.
  pub hostname: String,
  /// CPU model string from `/proc/cpuinfo`.
  pub cpu_model: String,
  /// SIMD-relevant CPU flags present on this machine.
  pub cpu_flags: Vec<String>,
  /// Logical CPUs this process may run on.
  pub available_cpus: usize,
  /// CPU list the run was pinned to (`BENCH_CPUS`), if any.
  pub cpu_pinning: Option<String>,
  /// Total RAM in bytes.
  pub mem_total_bytes: Option<u64>,
  /// Kernel release.
  pub kernel: String,
  /// Cloud machine type (`BENCH_MACHINE_TYPE`), e.g. `c4-standard-16`.
  pub machine_type: Option<String>,
  /// Commit of the muninn checkout (`BENCH_GIT_COMMIT`).
  pub git_commit: Option<String>,
  /// `RUSTFLAGS` the harness was built with (`BENCH_RUSTFLAGS`).
  pub rustflags: Option<String>,
  /// `rustc --version` (`BENCH_RUSTC`).
  pub rustc: Option<String>,
  /// Label of the run (`BENCH_LABEL`), e.g. "shared VM, noisy".
  pub label: Option<String>,
  /// SIMD kernel set muninn chose at runtime.
  pub muninn_simd_isa: String,
  /// Seconds since the Unix epoch when this was collected.
  pub timestamp: u64,
}

fn env(name: &str) -> Option<String> {
  std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn read(path: &str) -> String {
  std::fs::read_to_string(path).unwrap_or_default()
}

impl SystemInfo {
  /// Collects the description of this machine and process.
  pub fn collect() -> Self {
    let cpuinfo = read("/proc/cpuinfo");
    let field = |name: &str| {
      cpuinfo
        .lines()
        .find(|l| l.starts_with(name))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim().to_string())
    };
    let present: Vec<String> = field("flags")
      .unwrap_or_default()
      .split_whitespace()
      .map(str::to_string)
      .collect();
    let cpu_flags = FLAGS
      .iter()
      .filter(|f| present.iter().any(|p| p == *f))
      .map(|f| f.to_string())
      .collect();
    let mem_total_bytes = read("/proc/meminfo")
      .lines()
      .find(|l| l.starts_with("MemTotal:"))
      .and_then(parse_kib);
    Self {
      hostname: read("/proc/sys/kernel/hostname").trim().to_string(),
      cpu_model: field("model name").unwrap_or_else(|| "unknown".into()),
      cpu_flags,
      available_cpus: std::thread::available_parallelism()
        .map_or(1, |n| n.get()),
      cpu_pinning: env("BENCH_CPUS"),
      mem_total_bytes,
      kernel: read("/proc/sys/kernel/osrelease").trim().to_string(),
      machine_type: env("BENCH_MACHINE_TYPE"),
      git_commit: env("BENCH_GIT_COMMIT"),
      rustflags: env("BENCH_RUSTFLAGS"),
      rustc: env("BENCH_RUSTC"),
      label: env("BENCH_LABEL"),
      muninn_simd_isa: muninn_simd::active_isa().to_string(),
      timestamp: SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()),
    }
  }
}

/// Parses a `/proc` line such as `VmRSS:   1234 kB` into bytes.
fn parse_kib(line: &str) -> Option<u64> {
  let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
  Some(kib * 1024)
}

fn status_field(name: &str) -> Option<u64> {
  read("/proc/self/status")
    .lines()
    .find(|l| l.starts_with(name))
    .and_then(parse_kib)
}

/// Resident set size of this process in bytes (Linux only).
pub fn rss_bytes() -> Option<u64> {
  status_field("VmRSS:")
}

/// Peak resident set size of this process in bytes (Linux only).
pub fn peak_rss_bytes() -> Option<u64> {
  status_field("VmHWM:")
}

/// Difference of two optional RSS readings.
pub fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<i64> {
  Some(after? as i64 - before? as i64)
}

/// Total size of the regular files under `path`, recursively.
pub fn dir_size(path: &Path) -> std::io::Result<u64> {
  let meta = std::fs::metadata(path)?;
  if meta.is_file() {
    return Ok(meta.len());
  }
  let mut total = 0;
  for entry in std::fs::read_dir(path)? {
    total += dir_size(&entry?.path())?;
  }
  Ok(total)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_proc_kib_lines() {
    assert_eq!(parse_kib("VmRSS:\t  2048 kB"), Some(2 << 20));
    assert_eq!(parse_kib("garbage"), None);
    assert_eq!(rss_delta(Some(10), Some(4)), Some(-6));
    assert_eq!(rss_delta(None, Some(4)), None);
  }

  #[test]
  fn sums_directory_sizes() {
    let dir = std::env::temp_dir()
      .join(format!("muninn-bench-{}-size", std::process::id()));
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("a"), [0u8; 10]).unwrap();
    std::fs::write(dir.join("sub/b"), [0u8; 5]).unwrap();
    assert_eq!(dir_size(&dir).unwrap(), 15);
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[test]
  fn collects_without_panicking() {
    let s = SystemInfo::collect();
    assert!(s.available_cpus >= 1);
    assert!(!s.muninn_simd_isa.is_empty());
  }
}
