//! WAL append throughput per durability mode, writer count and record
//! size. Prints one table row per configuration.
//!
//! `cargo bench -p muninn-store --bench wal`. Each configuration runs for
//! `MUNINN_WAL_BENCH_SECS` seconds (default 2) or until it has written
//! 256 MiB, whichever comes first, in a fresh directory under cargo's
//! target tmp dir (set `MUNINN_WAL_BENCH_DIR` to measure another device);
//! the directory is deleted afterwards.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use muninn_store::{Durability, Wal, WalOptions};

const BYTE_BUDGET: u64 = 256 << 20;

struct Row {
  records: u64,
  bytes: u64,
  secs: f64,
  syncs: u64,
  writes: u64,
}

fn run(
  dir: &Path,
  durability: Durability,
  writers: usize,
  size: usize,
  secs: f64,
) -> Row {
  let _ = std::fs::remove_dir_all(dir);
  let wal = Arc::new(
    Wal::open(
      dir,
      WalOptions {
        durability,
        segment_size: 64 << 20,
        min_next_lsn: 1,
      },
    )
    .unwrap(),
  );
  let stop = Arc::new(AtomicBool::new(false));
  let written = Arc::new(AtomicU64::new(0));
  let barrier = Arc::new(Barrier::new(writers + 1));
  let handles: Vec<_> = (0..writers)
    .map(|t| {
      let (wal, stop, written, barrier) =
        (wal.clone(), stop.clone(), written.clone(), barrier.clone());
      std::thread::spawn(move || {
        let payload: Vec<u8> =
          (0..size).map(|i| (i * 7 + t * 13) as u8).collect();
        barrier.wait();
        let mut n = 0u64;
        while !stop.load(Ordering::Relaxed) {
          wal.append(&payload).unwrap();
          n += 1;
          let total = written.fetch_add(size as u64, Ordering::Relaxed);
          if total + size as u64 >= BYTE_BUDGET {
            stop.store(true, Ordering::Relaxed);
          }
        }
        n
      })
    })
    .collect();
  barrier.wait();
  let start = Instant::now();
  let deadline = start + Duration::from_secs_f64(secs);
  while !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
    std::thread::sleep(Duration::from_millis(1));
  }
  stop.store(true, Ordering::Relaxed);
  let records: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
  let elapsed = start.elapsed().as_secs_f64();
  let stats = wal.stats();
  drop(wal);
  let _ = std::fs::remove_dir_all(dir);
  Row {
    records,
    bytes: records * size as u64,
    secs: elapsed,
    syncs: stats.syncs,
    writes: stats.writes,
  }
}

fn main() {
  // `cargo test --benches` runs bench binaries with `--test`: keep it
  // instant there.
  let smoke = std::env::args().any(|a| a == "--test");
  let secs: f64 = std::env::var("MUNINN_WAL_BENCH_SECS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(if smoke { 0.05 } else { 2.0 });
  let base: PathBuf = std::env::var_os("MUNINN_WAL_BENCH_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
  let dir = base.join(format!("muninn-wal-bench-{}", std::process::id()));
  println!(
    "WAL append throughput, {secs} s per row, dir {}",
    dir.display()
  );
  println!(
    "| mode | writers | record | records/s | MB/s | records/write | records/fsync |"
  );
  println!("|---|---:|---:|---:|---:|---:|---:|");
  let modes = [
    ("SyncEachWrite", Durability::SyncEachWrite),
    (
      "SyncBatch(10ms)",
      Durability::SyncBatch {
        max_delay: Duration::from_millis(10),
      },
    ),
    ("NoSync", Durability::NoSync),
  ];
  let sizes: &[usize] = if smoke { &[1024] } else { &[1024, 4096] };
  let writer_counts: &[usize] = if smoke { &[4] } else { &[1, 4, 16] };
  for (name, mode) in modes {
    for &size in sizes {
      for &writers in writer_counts {
        let r = run(&dir, mode, writers, size, secs);
        let per = |n: u64| {
          if n == 0 {
            "-".to_string()
          } else {
            format!("{:.1}", r.records as f64 / n as f64)
          }
        };
        println!(
          "| {name} | {writers} | {} KB | {:.0} | {:.1} | {} | {} |",
          size / 1024,
          r.records as f64 / r.secs,
          r.bytes as f64 / r.secs / 1e6,
          per(r.writes),
          per(r.syncs),
        );
      }
    }
  }
}
