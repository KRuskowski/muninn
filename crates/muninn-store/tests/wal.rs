//! WAL behaviour: round trips, crash recovery, group commit, rotation.

mod common;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::TestDir;
use muninn_core::Error;
use muninn_store::wal::{
  DISCARDED_SUFFIX, MAX_UNSYNCED_SEGMENTS, SEGMENT_SUFFIX,
};
use muninn_store::{Durability, Lsn, Recovery, Wal, WalOptions};

type Records = Vec<(Lsn, Vec<u8>)>;

fn opts(durability: Durability) -> WalOptions {
  WalOptions {
    durability,
    ..WalOptions::default()
  }
}

fn small_segments(durability: Durability, size: u64) -> WalOptions {
  WalOptions {
    durability,
    segment_size: size,
    min_next_lsn: 1,
  }
}

fn recover_all(
  dir: &Path,
  o: WalOptions,
  from: Lsn,
) -> (Wal, Records, Recovery) {
  let mut got = Vec::new();
  let (wal, rec) = Wal::recover(dir, o, from, |lsn, p| {
    got.push((lsn, p.to_vec()));
    Ok(())
  })
  .unwrap();
  (wal, got, rec)
}

fn payload(i: u64) -> Vec<u8> {
  let len = (i * 37 % 200) as usize;
  (0..len).map(|j| (i as usize * 31 + j) as u8).collect()
}

fn segment_files(dir: &Path) -> Vec<PathBuf> {
  let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
    .unwrap()
    .map(|e| e.unwrap().path())
    .filter(|p| p.to_str().unwrap().ends_with(SEGMENT_SUFFIX))
    .collect();
  v.sort();
  v
}

#[test]
fn round_trip_and_reopen_continues_lsns() {
  let d = TestDir::new("wal-rt");
  {
    let wal = Wal::open(d.path(), opts(Durability::SyncEachWrite)).unwrap();
    assert_eq!(wal.last_lsn(), 0);
    for i in 1..=50u64 {
      assert_eq!(wal.append(&payload(i)).unwrap(), i);
    }
    let r = wal
      .append_batch(&[b"x".as_slice(), b"", b"yz".as_slice()])
      .unwrap();
    assert_eq!(r, 51..54);
    assert_eq!(wal.append_batch(&[]).unwrap(), 54..54);
    assert_eq!(wal.durable_lsn(), 53);
    let mut live = Vec::new();
    wal
      .replay(49, |l, p| {
        live.push((l, p.to_vec()));
        Ok(())
      })
      .unwrap();
    assert_eq!(live.len(), 5);
    assert_eq!(live[4], (53, b"yz".to_vec()));
    wal.close().unwrap();
  }
  let (wal, got, rec) = recover_all(d.path(), opts(Durability::NoSync), 1);
  assert_eq!(got.len(), 53);
  for (i, (l, p)) in got.iter().take(50).enumerate() {
    assert_eq!(*l, i as u64 + 1);
    assert_eq!(*p, payload(i as u64 + 1));
  }
  assert_eq!(got[51], (52, Vec::new()));
  assert_eq!(rec.records_replayed, 53);
  assert_eq!(rec.scan_start_lsn, 1);
  assert_eq!(rec.last_lsn, 53);
  assert!(rec.truncated.is_none() && rec.discarded.is_empty());
  assert_eq!(wal.append(b"next").unwrap(), 54);
  drop(wal);
  let (_, got, _) = recover_all(d.path(), opts(Durability::NoSync), 40);
  assert_eq!(got.first().unwrap().0, 40);
  assert_eq!(got.last().unwrap(), &(54, b"next".to_vec()));
}

#[test]
fn torn_tail_at_every_offset_of_last_record() {
  let d = TestDir::new("wal-torn");
  let n = 6u64;
  let last_payload: Vec<u8> = (0..41u8).collect();
  {
    let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
    for i in 1..n {
      wal.append(&payload(i)).unwrap();
    }
    wal.append(&last_payload).unwrap();
    wal.close().unwrap();
  }
  let seg = segment_files(d.path()).pop().unwrap();
  let full = std::fs::read(&seg).unwrap();
  let last_start = full.len() - 8 - last_payload.len();
  for cut in last_start..full.len() {
    std::fs::write(&seg, &full[..cut]).unwrap();
    let (wal, got, rec) = recover_all(d.path(), opts(Durability::NoSync), 1);
    assert_eq!(got.len() as u64, n - 1, "cut at {cut}");
    for (i, (l, p)) in got.iter().enumerate() {
      assert_eq!(
        (*l, p.as_slice()),
        (i as u64 + 1, &payload(i as u64 + 1)[..])
      );
    }
    // A cut exactly at a record boundary is a clean end, not a tear.
    assert_eq!(rec.truncated.is_some(), cut != last_start, "cut at {cut}");
    assert_eq!(std::fs::metadata(&seg).unwrap().len(), last_start as u64);
    // The log is usable: the lost record's LSN is reissued.
    assert_eq!(wal.append(b"again").unwrap(), n);
    drop(wal);
    let (_, got, rec) = recover_all(d.path(), opts(Durability::NoSync), n);
    assert_eq!(got, vec![(n, b"again".to_vec())]);
    assert!(rec.truncated.is_none());
  }
  // And the untouched file still replays completely.
  std::fs::write(&seg, &full).unwrap();
  let (_, got, _) = recover_all(d.path(), opts(Durability::NoSync), 1);
  assert_eq!(got.last().unwrap(), &(n, last_payload));
}

#[test]
fn corrupt_crc_in_the_middle_stops_replay_there() {
  let d = TestDir::new("wal-crc");
  {
    let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
    for i in 1..=10u64 {
      wal.append(&[i as u8; 100]).unwrap();
    }
  }
  let seg = segment_files(d.path()).pop().unwrap();
  let mut bytes = std::fs::read(&seg).unwrap();
  // Records are 108 bytes after a 32-byte header; corrupt record 5.
  let rec5 = 32 + 4 * 108;
  bytes[rec5 + 8 + 50] ^= 0xFF;
  std::fs::write(&seg, &bytes).unwrap();
  let (wal, got, rec) = recover_all(d.path(), opts(Durability::NoSync), 1);
  assert_eq!(
    got.iter().map(|r| r.0).collect::<Vec<_>>(),
    vec![1, 2, 3, 4]
  );
  let t = rec.truncated.unwrap();
  assert!(t.reason.contains("checksum"), "{}", t.reason);
  assert_eq!(t.offset, rec5 as u64);
  assert_eq!(t.dropped_bytes, 6 * 108);
  assert_eq!(wal.append(b"five").unwrap(), 5);
}

#[test]
fn corruption_in_an_earlier_segment_discards_later_ones() {
  let d = TestDir::new("wal-mid-seg");
  {
    let o = small_segments(Durability::NoSync, 600);
    let wal = Wal::open(d.path(), o).unwrap();
    for i in 1..=30u64 {
      wal.append(&[i as u8; 92]).unwrap();
    }
    assert!(wal.segment_count() >= 4);
  }
  let segs = segment_files(d.path());
  let mut b = std::fs::read(&segs[1]).unwrap();
  let first_in_seg1 = u64::from_le_bytes(b[16..24].try_into().unwrap());
  b[32 + 100 + 8 + 3] ^= 1; // second record of segment 1
  std::fs::write(&segs[1], &b).unwrap();
  let (wal, got, rec) =
    recover_all(d.path(), small_segments(Durability::NoSync, 600), 1);
  assert_eq!(got.len() as u64, first_in_seg1);
  assert_eq!(rec.discarded.len(), segs.len() - 2);
  for p in &rec.discarded {
    assert!(p.to_str().unwrap().ends_with(DISCARDED_SUFFIX));
    assert!(p.exists());
  }
  assert_eq!(segment_files(d.path()).len(), 2);
  assert_eq!(wal.append(b"n").unwrap(), first_in_seg1 + 1);
}

#[test]
fn zero_filled_tail_is_trimmed() {
  let d = TestDir::new("wal-zeros");
  {
    let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
    for i in 1..=3u64 {
      wal.append(&payload(i)).unwrap();
    }
  }
  let seg = segment_files(d.path()).pop().unwrap();
  let len = std::fs::metadata(&seg).unwrap().len();
  let mut b = std::fs::read(&seg).unwrap();
  b.extend_from_slice(&[0u8; 4096]);
  std::fs::write(&seg, &b).unwrap();
  let (wal, got, rec) = recover_all(d.path(), opts(Durability::NoSync), 1);
  assert_eq!(got.len(), 3);
  assert_eq!(rec.truncated.unwrap().dropped_bytes, 4096);
  assert_eq!(std::fs::metadata(&seg).unwrap().len(), len);
  assert_eq!(wal.append(b"4").unwrap(), 4);
}

#[test]
fn torn_segment_header_is_reinitialised() {
  let d = TestDir::new("wal-torn-header");
  {
    let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
    for i in 1..=4u64 {
      wal.append(&payload(i)).unwrap();
    }
  }
  // A crash right after creating the next segment file.
  let next = d.path().join(format!("{:020}{SEGMENT_SUFFIX}", 5));
  std::fs::write(&next, b"MUNWAL").unwrap();
  // Replay from 1 scans both segments; from MAX only the last.
  for from in [1, Lsn::MAX] {
    let (wal, got, rec) = recover_all(d.path(), opts(Durability::NoSync), from);
    assert_eq!(got.len(), if from == 1 { 4 } else { 0 });
    assert_eq!(rec.last_lsn, 4);
    assert_eq!(wal.last_lsn(), 4);
    drop(wal);
    std::fs::write(&next, b"MUNWAL").unwrap();
  }
  let (wal, _, rec) = recover_all(d.path(), opts(Durability::NoSync), 1);
  assert!(rec.truncated.is_some());
  assert_eq!(wal.append(b"5").unwrap(), 5);
  drop(wal);
  let (_, got, rec) = recover_all(d.path(), opts(Durability::NoSync), 1);
  assert_eq!(got.len(), 5);
  assert!(rec.truncated.is_none());
}

fn concurrent(dir: &Path, durability: Durability, threads: u64, per: u64) {
  let wal = Arc::new(Wal::open(dir, opts(durability)).unwrap());
  let handles: Vec<_> = (0..threads)
    .map(|t| {
      let wal = wal.clone();
      std::thread::spawn(move || {
        let mut mine = Vec::new();
        for i in 0..per {
          let p = format!("t{t}-r{i}").into_bytes();
          let lsn = if i % 5 == 4 {
            let r = wal.append_batch(&[&p, b"batch-second"]).unwrap();
            assert_eq!(r.end - r.start, 2);
            r.start
          } else {
            wal.append(&p).unwrap()
          };
          mine.push((lsn, p));
        }
        mine
      })
    })
    .collect();
  let mut all: Vec<(Lsn, Vec<u8>)> = Vec::new();
  for h in handles {
    let mine = h.join().unwrap();
    // Each thread sees its own LSNs strictly increasing.
    assert!(mine.windows(2).all(|w| w[0].0 < w[1].0));
    all.extend(mine);
  }
  let batches = threads * (per / 5);
  let total = threads * per + batches;
  let unique: HashSet<Lsn> = all.iter().map(|r| r.0).collect();
  assert_eq!(unique.len(), all.len());
  assert_eq!(wal.last_lsn(), total);
  assert_eq!(wal.stats().records, total);
  Arc::try_unwrap(wal).unwrap().close().unwrap();
  let (_, got, rec) = recover_all(dir, opts(Durability::NoSync), 1);
  assert_eq!(rec.records_replayed, total);
  assert_eq!(
    got.iter().map(|r| r.0).collect::<Vec<_>>(),
    (1..=total).collect::<Vec<_>>()
  );
  for (lsn, p) in &all {
    assert_eq!(&got[*lsn as usize - 1].1, p);
    assert_eq!(got[*lsn as usize - 1].0, *lsn);
  }
}

#[test]
fn concurrent_appends_get_unique_ordered_lsns() {
  let d = TestDir::new("wal-conc");
  concurrent(d.path(), Durability::NoSync, 16, 400);
}

#[test]
fn concurrent_durable_appends_replay_completely() {
  let d = TestDir::new("wal-conc-sync");
  concurrent(d.path(), Durability::SyncEachWrite, 8, 25);
}

#[test]
fn group_commit_shares_fsyncs() {
  let d = TestDir::new("wal-group");
  let wal =
    Arc::new(Wal::open(d.path(), opts(Durability::SyncEachWrite)).unwrap());
  let threads = 8;
  let per = 40;
  let handles: Vec<_> = (0..threads)
    .map(|t| {
      let wal = wal.clone();
      std::thread::spawn(move || {
        for i in 0..per {
          let lsn = wal.append(format!("{t}/{i}").as_bytes()).unwrap();
          // Durable on return.
          assert!(wal.durable_lsn() >= lsn);
        }
      })
    })
    .collect();
  for h in handles {
    h.join().unwrap();
  }
  let s = wal.stats();
  assert_eq!(s.records, threads * per);
  assert!(s.syncs < s.records, "no fsync sharing: {s:?}");
  assert!(s.writes < s.records, "no write batching: {s:?}");
  assert_eq!(wal.durable_lsn(), threads * per);
}

#[test]
fn sync_batch_bounds_the_unsynced_window() {
  let d = TestDir::new("wal-batch");
  let o = opts(Durability::SyncBatch {
    max_delay: Duration::from_millis(20),
  });
  let wal = Wal::open(d.path(), o).unwrap();
  for i in 0..20u64 {
    wal.append(&payload(i)).unwrap();
  }
  let start = Instant::now();
  while wal.durable_lsn() < 20 {
    assert!(start.elapsed() < Duration::from_secs(30), "never synced");
    std::thread::sleep(Duration::from_millis(5));
  }
  assert!(wal.stats().syncs >= 1);
  wal.close().unwrap();
}

#[test]
fn no_sync_never_fsyncs_unless_asked() {
  let d = TestDir::new("wal-nosync");
  let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
  for i in 0..10u64 {
    wal.append(&payload(i)).unwrap();
  }
  assert_eq!(wal.stats().syncs, 0);
  assert_eq!(wal.durable_lsn(), 0);
  wal.sync().unwrap();
  assert_eq!(wal.durable_lsn(), 10);
  assert_eq!(wal.stats().syncs, 1);
}

#[test]
fn rotation_by_size_and_on_request() {
  let d = TestDir::new("wal-rotate");
  let o = small_segments(Durability::SyncEachWrite, 1000);
  let wal = Wal::open(d.path(), o.clone()).unwrap();
  for i in 1..=100u64 {
    wal.append(&[i as u8; 92]).unwrap();
  }
  let files = segment_files(d.path());
  assert_eq!(files.len(), wal.segment_count());
  assert!(files.len() >= 10, "{} segments", files.len());
  for f in &files {
    // Header plus at most nine 100-byte records fit 1000 bytes.
    assert!(std::fs::metadata(f).unwrap().len() <= 1000);
  }
  // Each file is named after its first record's LSN.
  let mut expect_first = 1;
  for f in &files {
    let name = f.file_stem().unwrap().to_str().unwrap();
    assert_eq!(name.parse::<u64>().unwrap(), expect_first);
    let len = std::fs::metadata(f).unwrap().len();
    expect_first += (len - 32) / 100;
  }
  let before = wal.segment_count();
  wal.rotate().unwrap();
  assert_eq!(wal.segment_count(), before + 1);
  wal.rotate().unwrap(); // empty segment: no-op
  assert_eq!(wal.segment_count(), before + 1);
  assert_eq!(wal.append(b"after").unwrap(), 101);
  drop(wal);
  let (_, got, rec) = recover_all(d.path(), o, 1);
  assert_eq!(got.len(), 101);
  assert!(rec.truncated.is_none());
}

#[test]
fn truncate_before_deletes_whole_obsolete_segments() {
  let d = TestDir::new("wal-trunc");
  let o = small_segments(Durability::NoSync, 1000);
  let wal = Wal::open(d.path(), o.clone()).unwrap();
  for i in 1..=100u64 {
    wal.append(&[i as u8; 92]).unwrap();
  }
  let before = segment_files(d.path()).len();
  let deleted = wal.truncate_before(50).unwrap();
  assert!(deleted > 0);
  assert_eq!(segment_files(d.path()).len(), before - deleted);
  // Everything from 50 on is still there; the first kept record is at or
  // below 50, never above.
  let mut first = None;
  let n = wal
    .replay(0, |l, _| {
      first.get_or_insert(l);
      Ok(())
    })
    .unwrap();
  let first = first.unwrap();
  assert!(first <= 50 && first > 40, "first kept {first}");
  assert_eq!(n, 100 - first + 1);
  // The active segment always survives, so LSNs continue after reopen.
  wal.truncate_before(Lsn::MAX).unwrap();
  assert_eq!(wal.segment_count(), 1);
  drop(wal);
  let wal = Wal::open(d.path(), o).unwrap();
  assert_eq!(wal.last_lsn(), 100);
  assert_eq!(wal.append(b"x").unwrap(), 101);
}

#[test]
fn min_next_lsn_skips_ahead_and_recovery_accepts_the_jump() {
  let d = TestDir::new("wal-min");
  let mut o = opts(Durability::NoSync);
  {
    let wal = Wal::open(d.path(), o.clone()).unwrap();
    for i in 1..=3u64 {
      wal.append(&payload(i)).unwrap();
    }
  }
  o.min_next_lsn = 1000;
  {
    let wal = Wal::open(d.path(), o.clone()).unwrap();
    assert_eq!(wal.append(b"k").unwrap(), 1000);
  }
  let (_, got, rec) = recover_all(d.path(), o.clone(), 1);
  assert_eq!(
    got.iter().map(|r| r.0).collect::<Vec<_>>(),
    vec![1, 2, 3, 1000]
  );
  assert!(rec.truncated.is_none());
  let fresh = TestDir::new("wal-min-fresh");
  let (wal, _, rec) = recover_all(fresh.path(), o, 1);
  // The engine can see that LSNs 1..1000 are not in this log.
  assert_eq!(rec.scan_start_lsn, 1000);
  assert_eq!(wal.append(b"k").unwrap(), 1000);
}

#[test]
fn replay_callback_error_aborts_open() {
  let d = TestDir::new("wal-cb");
  {
    let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
    wal.append(b"a").unwrap();
  }
  let r = Wal::recover(d.path(), opts(Durability::NoSync), 1, |_, _| {
    Err(Error::invalid("engine refused"))
  });
  assert!(matches!(r, Err(Error::InvalidArgument(_))));
}

// Regression tests from the PR #10 review.

#[test]
fn replay_handles_lsn_skips() {
  let d = TestDir::new("wal-replay-skip");
  {
    let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
    for _ in 0..3 {
      wal.append(b"x").unwrap();
    }
  }
  let mut o = opts(Durability::NoSync);
  o.min_next_lsn = 1000;
  let wal = Wal::open(d.path(), o).unwrap();
  wal.append(b"y").unwrap();
  let mut got = Vec::new();
  wal
    .replay(1, |l, _| {
      got.push(l);
      Ok(())
    })
    .unwrap();
  assert_eq!(got, vec![1, 2, 3, 1000]);
}

#[test]
fn reopen_does_not_claim_durability_without_fsync() {
  let d = TestDir::new("wal-reopen-durable");
  {
    let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
    for _ in 0..5 {
      wal.append(b"x").unwrap();
    }
    assert_eq!(wal.durable_lsn(), 0);
  }
  let wal = Wal::open(d.path(), opts(Durability::SyncEachWrite)).unwrap();
  // The tail may still be only in the page cache: reporting it durable
  // requires an fsync during open.
  assert_eq!(wal.durable_lsn(), 5);
  assert!(wal.stats().syncs > 0, "{:?}", wal.stats());
}

#[test]
fn no_sync_sync_covers_rotated_segments() {
  let d = TestDir::new("wal-nosync-rotated");
  let wal =
    Wal::open(d.path(), small_segments(Durability::NoSync, 4096)).unwrap();
  for _ in 0..100 {
    wal.append(&[7u8; 200]).unwrap();
  }
  assert!(wal.segment_count() > 3);
  let before = wal.stats();
  assert_eq!(before.syncs, 0);
  wal.sync().unwrap();
  let after = wal.stats();
  // Every segment rotated out unsynced, plus the active one, plus the
  // directory entries of the new segments.
  assert_eq!(after.syncs as usize, wal.segment_count(), "{after:?}");
  assert!(after.dir_syncs > before.dir_syncs, "{after:?}");
  assert_eq!(wal.durable_lsn(), 100);
  // Nothing left to do on a second sync.
  wal.sync().unwrap();
  assert_eq!(wal.stats().syncs, after.syncs);
}

/// A log that rotates often without ever syncing must not keep one open
/// file per rotated segment until the next sync.
#[cfg(target_os = "linux")]
#[test]
fn no_sync_rotation_keeps_no_handle_per_segment() {
  // Other tests run in parallel and open files too: allow some slack,
  // far below the one handle per segment of a leak.
  let open_files = || std::fs::read_dir("/proc/self/fd").unwrap().count();
  let d = TestDir::new("wal-nosync-handles");
  let wal =
    Wal::open(d.path(), small_segments(Durability::NoSync, 4096)).unwrap();
  let before = open_files();
  for _ in 0..6000 {
    wal.append(&[7u8; 200]).unwrap();
  }
  let rotated = wal.segment_count() - 1;
  assert!(rotated >= 250, "{rotated} segments");
  let after = open_files();
  assert!(
    after < before + 100,
    "{before} open files before, {after} after {rotated} rotations"
  );
  // Every full group of rotated segments was synced and closed; nothing
  // is claimed durable before a sync.
  let eager = rotated / MAX_UNSYNCED_SEGMENTS * MAX_UNSYNCED_SEGMENTS;
  assert_eq!(wal.stats().syncs as usize, eager);
  assert_eq!(wal.durable_lsn(), 0);
  // Sync reaches the remaining rotated segments, also when some of them
  // were deleted in between, then the active one.
  wal.truncate_before(3000).unwrap();
  wal.sync().unwrap();
  assert_eq!(wal.durable_lsn(), 6000);
  assert_eq!(wal.stats().syncs as usize, rotated + 1);
  drop(wal);
  let (_, got, rec) =
    recover_all(d.path(), small_segments(Durability::NoSync, 4096), 3000);
  assert!(rec.truncated.is_none(), "{rec:?}");
  assert_eq!(got.last().unwrap().0, 6000);
}

#[test]
fn segment_named_zero_is_corrupt_not_a_panic() {
  let d = TestDir::new("wal-seg-zero");
  std::fs::write(d.path().join("00000000000000000000.wal"), b"garbage")
    .unwrap();
  let r = std::panic::catch_unwind(|| {
    Wal::open(d.path(), opts(Durability::NoSync)).map(|_| ())
  });
  let r = r.expect("panicked instead of returning an error");
  assert!(matches!(r, Err(Error::Corrupt(_))), "{r:?}");
}

#[test]
fn segment_named_max_lsn_is_rejected() {
  let d = TestDir::new("wal-seg-max");
  std::fs::write(d.path().join(format!("{}.wal", u64::MAX)), b"garbage")
    .unwrap();
  let r = std::panic::catch_unwind(|| {
    let wal = Wal::open(d.path(), opts(Durability::NoSync))?;
    wal.append(b"x").map(|_| ())
  });
  let r = r.expect("panicked instead of returning an error");
  assert!(r.is_err());
}

#[test]
fn new_wal_dir_parent_is_synced() {
  let d = TestDir::new("wal-parent-sync");
  // Existing directory: only the first segment's entry is synced.
  let wal = Wal::open(d.path(), opts(Durability::NoSync)).unwrap();
  assert_eq!(wal.stats().dir_syncs, 1);
  drop(wal);
  // New directory: its entry in the parent is synced too.
  let new = d.path().join("wal");
  let wal = Wal::open(&new, opts(Durability::NoSync)).unwrap();
  assert_eq!(wal.stats().dir_syncs, 2);
  drop(wal);
  // Two new levels: each new directory's entry in its parent.
  let nested = d.path().join("collection").join("wal");
  let wal = Wal::open(&nested, opts(Durability::NoSync)).unwrap();
  assert_eq!(wal.stats().dir_syncs, 3);
}

#[cfg(unix)]
#[test]
fn truncate_before_keeps_segments_it_could_not_delete() {
  use std::os::unix::fs::PermissionsExt;
  let d = TestDir::new("wal-trunc-fail");
  let wal =
    Wal::open(d.path(), small_segments(Durability::NoSync, 1000)).unwrap();
  for i in 1..=50u64 {
    wal.append(&[i as u8; 92]).unwrap();
  }
  let before = wal.segment_count();
  std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o555))
    .unwrap();
  let r = wal.truncate_before(Lsn::MAX);
  std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o755))
    .unwrap();
  assert!(r.is_err());
  // The log still knows every file that is still on disk.
  assert_eq!(wal.segment_count(), segment_files(d.path()).len());
  assert_eq!(wal.segment_count(), before);
  assert_eq!(wal.replay(1, |_, _| Ok(())).unwrap(), 50);
  // And a retry works.
  assert_eq!(wal.truncate_before(Lsn::MAX).unwrap(), before - 1);
}

#[test]
fn sync_batch_zero_delay_acts_as_sync_each_write() {
  let d = TestDir::new("wal-batch-zero");
  let wal = Wal::open(
    d.path(),
    opts(Durability::SyncBatch {
      max_delay: Duration::ZERO,
    }),
  )
  .unwrap();
  for i in 0..200u64 {
    let lsn = wal.append(&payload(i)).unwrap();
    assert!(wal.durable_lsn() >= lsn, "record {lsn} not durable");
  }
}

#[test]
fn open_and_recover_agree_on_mid_log_corruption() {
  let d = TestDir::new("wal-open-agree");
  let o = small_segments(Durability::NoSync, 600);
  {
    let wal = Wal::open(d.path(), o.clone()).unwrap();
    for i in 1..=30u64 {
      wal.append(&[i as u8; 92]).unwrap();
    }
  }
  let segs = segment_files(d.path());
  let mut b = std::fs::read(&segs[0]).unwrap();
  b[32 + 8 + 5] ^= 1; // first record
  std::fs::write(&segs[0], &b).unwrap();
  let lsn = {
    let wal = Wal::open(d.path(), o.clone()).unwrap();
    wal.append(b"after open").unwrap()
  };
  // Whatever open() decided, recovery must keep what was appended after.
  let (_, got, _) = recover_all(d.path(), o, 1);
  assert_eq!(got.last().unwrap(), &(lsn, b"after open".to_vec()));
}
