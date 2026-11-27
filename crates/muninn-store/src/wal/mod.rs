//! Write-ahead log with group commit.
//!
//! The WAL is a directory of segment files named after the LSN of their
//! first record (`<first_lsn>.wal`, 20 zero-padded digits). Records are
//! opaque byte strings; the engine encodes its operations. Each append
//! gets the next LSN (log sequence number), starting at 1.
//!
//! # Group commit
//!
//! Appends from many threads are copied into a shared buffer under a short
//! mutex. Whichever appender finds no I/O in progress becomes the leader:
//! it takes the whole buffer, writes it with one `write` call and, when the
//! durability mode asks for it, one `fdatasync`, while new appends keep
//! filling the next buffer. When it finishes, it wakes the threads whose
//! records it covered, and the next waiting appender leads the next round.
//! The cost of one fsync is therefore shared by every record that arrived
//! while the previous one was in flight, and throughput grows with the
//! number of concurrent writers instead of being capped at one record per
//! fsync.
//!
//! # Durability
//!
//! See [`Durability`]. In every mode an append returns only after its
//! record was handed to the OS, so a process crash never loses an
//! acknowledged record; the modes differ in what an OS crash or power loss
//! can lose.
//!
//! # Recovery
//!
//! [`Wal::recover`] reads segments in LSN order, verifying every record's
//! checksum, and stops at the first record that is incomplete or fails
//! validation. The file is truncated there and any later segments are set
//! aside (renamed to `*.wal.discarded`), so the log is always a gap-free
//! prefix of what was written (point-in-time recovery). A partially written
//! tail, the normal result of a crash, is therefore never an error.

mod format;

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use muninn_core::{Error, Result};

use crate::fs::sync_dir;
use crate::mmap::map_file;
pub use format::MAX_RECORD_LEN;
use format::{
  HEADER_LEN, RECORD_HEADER_LEN, SegmentHeader, decode_header, encode_header,
  record_header, scan_records,
};

/// Log sequence number. The first record ever appended gets 1; 0 means
/// "nothing".
pub type Lsn = u64;

/// File-name suffix of WAL segment files.
pub const SEGMENT_SUFFIX: &str = ".wal";
/// Suffix appended to segments set aside by recovery.
pub const DISCARDED_SUFFIX: &str = ".discarded";

/// Buffers above this size are freed after use rather than kept around.
const MAX_KEPT_BUFFER: usize = 8 << 20;

/// What an acknowledged append survives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
  /// An append returns after its record is on stable storage (fdatasync).
  /// Concurrent appends share fsyncs (group commit).
  SyncEachWrite,
  /// An append returns after its record reached the OS; a background
  /// thread fsyncs the log every `max_delay`, while appends continue. An OS crash or power loss loses at
  /// most the records of the last `max_delay` (plus one fsync).
  SyncBatch {
    /// Longest time a record may stay written but not fsynced.
    max_delay: Duration,
  },
  /// Never fsync unless [`Wal::sync`] is called. Survives process crashes
  /// only.
  NoSync,
}

/// WAL configuration.
#[derive(Clone, Debug)]
pub struct WalOptions {
  /// When appends are acknowledged; see [`Durability`].
  pub durability: Durability,
  /// A segment is closed and a new one started once the next write would
  /// take it past this many bytes. One write (a group of records) may
  /// exceed it.
  pub segment_size: u64,
  /// The next LSN is at least this. Pass the manifest's checkpoint + 1 so
  /// LSNs keep increasing even if the WAL lost its tail or was removed.
  pub min_next_lsn: Lsn,
}

impl Default for WalOptions {
  fn default() -> Self {
    Self {
      durability: Durability::SyncEachWrite,
      segment_size: 64 << 20,
      min_next_lsn: 1,
    }
  }
}

/// What [`Wal::recover`] found and repaired.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Recovery {
  /// Segment files read (segments wholly before the replay start are
  /// skipped without reading).
  pub segments_scanned: usize,
  /// First LSN of the first segment read. If this is above the `from`
  /// passed to [`Wal::recover`], records between them are not in the log
  /// (for example, the WAL directory was replaced).
  pub scan_start_lsn: Lsn,
  /// Records passed to the replay callback.
  pub records_replayed: u64,
  /// Last valid LSN in the log after recovery (0 if none).
  pub last_lsn: Lsn,
  /// Where the log was cut, if it was.
  pub truncated: Option<Truncation>,
  /// Later segments set aside because they followed the cut.
  pub discarded: Vec<PathBuf>,
}

/// A cut made by recovery at an incomplete or invalid record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Truncation {
  /// The segment that was cut.
  pub segment: PathBuf,
  /// Its new length.
  pub offset: u64,
  /// Bytes removed from it.
  pub dropped_bytes: u64,
  /// What was wrong at `offset`.
  pub reason: String,
}

/// Counters, for monitoring and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalStats {
  /// Records appended since open.
  pub records: u64,
  /// Bytes written, record headers included.
  pub bytes: u64,
  /// `write` calls, i.e. groups of records written together.
  pub writes: u64,
  /// `fdatasync` calls on segment files.
  pub syncs: u64,
  /// Segment files created since open.
  pub segments_created: u64,
}

#[derive(Clone, Debug)]
struct SegmentInfo {
  first_lsn: Lsn,
  path: PathBuf,
}

/// The segment currently appended to. Touched only by the I/O leader.
struct Active {
  /// Shared so the background syncer can fsync it without blocking
  /// writes.
  file: Arc<File>,
  len: u64,
}

struct State {
  next_lsn: Lsn,
  /// Encoded records with LSNs `written + 1 .. next_lsn` not yet taken by
  /// a leader.
  buf: Vec<u8>,
  spare: Vec<u8>,
  /// Every record up to this LSN was handed to the OS.
  written: Lsn,
  /// Every record up to this LSN is on stable storage.
  synced: Lsn,
  io_active: bool,
  rotate_requested: bool,
  failed: Option<String>,
  shutdown: bool,
}

#[derive(Default)]
struct Counters {
  records: AtomicU64,
  bytes: AtomicU64,
  writes: AtomicU64,
  syncs: AtomicU64,
  segments_created: AtomicU64,
}

struct Shared {
  dir: PathBuf,
  opts: WalOptions,
  state: Mutex<State>,
  /// Signalled when an I/O round finishes.
  io_done: Condvar,
  /// Wakes the background syncer for shutdown.
  syncer_wake: Condvar,
  active: Mutex<Active>,
  segments: Mutex<Vec<SegmentInfo>>,
  counters: Counters,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
  // A panic while holding one of these locks is a bug, but the protected
  // data stays consistent (every update is a plain store), so keep going.
  m.lock().unwrap_or_else(|e| e.into_inner())
}

fn segment_path(dir: &Path, first_lsn: Lsn) -> PathBuf {
  dir.join(format!("{first_lsn:020}{SEGMENT_SUFFIX}"))
}

fn parse_segment_name(name: &str) -> Option<Lsn> {
  let digits = name.strip_suffix(SEGMENT_SUFFIX)?;
  if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
    return None;
  }
  digits.parse().ok()
}

fn list_segments(dir: &Path) -> Result<Vec<SegmentInfo>> {
  let mut v = Vec::new();
  for e in std::fs::read_dir(dir)? {
    let e = e?;
    if let Some(first_lsn) = e.file_name().to_str().and_then(parse_segment_name)
    {
      v.push(SegmentInfo {
        first_lsn,
        path: e.path(),
      });
    }
  }
  v.sort_by_key(|s| s.first_lsn);
  Ok(v)
}

/// Creates a segment file holding only its header.
fn create_segment(
  dir: &Path,
  header: SegmentHeader,
  sync: bool,
) -> Result<(File, PathBuf)> {
  let path = segment_path(dir, header.first_lsn);
  let mut file = OpenOptions::new()
    .read(true)
    .write(true)
    .create_new(true)
    .open(&path)?;
  file.write_all(&encode_header(header))?;
  if sync {
    file.sync_data()?;
    sync_dir(dir)?;
  }
  Ok((file, path))
}

fn poisoned(msg: &str) -> Error {
  Error::Io(std::io::Error::other(format!(
    "WAL is unusable after an earlier I/O error: {msg}"
  )))
}

impl Shared {
  fn sync_on_rotate(&self) -> bool {
    self.opts.durability != Durability::NoSync
  }

  /// Waits until record `target` is written (or synced, with `sync`),
  /// leading I/O rounds whenever none is in progress. With `force`, leads
  /// at least one round even if `target` is already reached.
  fn drive<'a>(
    &'a self,
    mut st: MutexGuard<'a, State>,
    target: Lsn,
    sync: bool,
    mut force: bool,
  ) -> Result<()> {
    let each = self.opts.durability == Durability::SyncEachWrite;
    loop {
      let reached = if sync {
        st.synced >= target
      } else {
        st.written >= target
      };
      // Checked before `failed`: a record that made it is acknowledged
      // even if a later round failed.
      if reached && !force {
        return Ok(());
      }
      if let Some(msg) = &st.failed {
        return Err(poisoned(msg));
      }
      if st.io_active {
        st = self.io_done.wait(st).unwrap_or_else(|e| e.into_inner());
        continue;
      }
      // Lead one round: everything buffered so far.
      st.io_active = true;
      force = false;
      let spare = std::mem::take(&mut st.spare);
      let mut buf = std::mem::replace(&mut st.buf, spare);
      let first = st.written + 1;
      let last = st.next_lsn - 1;
      let rotate = std::mem::take(&mut st.rotate_requested);
      let do_sync = (sync || each) && st.synced < last;
      drop(st);

      let res = self.io(&buf, first, do_sync, rotate);

      buf.clear();
      st = lock(&self.state);
      st.io_active = false;
      if buf.capacity() <= MAX_KEPT_BUFFER
        && buf.capacity() > st.spare.capacity()
      {
        st.spare = buf;
      }
      match res {
        Ok(()) => {
          st.written = last;
          if do_sync {
            st.synced = st.synced.max(last);
          }
          self.io_done.notify_all();
        }
        Err(e) => {
          st.failed = Some(e.to_string());
          self.io_done.notify_all();
          return Err(e);
        }
      }
    }
  }

  /// One leader round: optional rotation, one write, optional sync.
  fn io(&self, buf: &[u8], first: Lsn, sync: bool, rotate: bool) -> Result<()> {
    let mut a = lock(&self.active);
    let has_records = a.len > HEADER_LEN as u64;
    let full = a.len + buf.len() as u64 > self.opts.segment_size;
    if (rotate && has_records) || (full && has_records && !buf.is_empty()) {
      self.rotate(&mut a, first)?;
    }
    if !buf.is_empty() {
      (&*a.file).write_all(buf)?;
      a.len += buf.len() as u64;
      self.counters.writes.fetch_add(1, Ordering::Relaxed);
      self
        .counters
        .bytes
        .fetch_add(buf.len() as u64, Ordering::Relaxed);
    }
    if sync {
      a.file.sync_data()?;
      self.counters.syncs.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
  }

  /// Closes the active segment and starts one whose first record is
  /// `first`. The old segment is synced first (unless `NoSync`) so that a
  /// newer segment never exists on disk while older records are not.
  fn rotate(&self, a: &mut Active, first: Lsn) -> Result<()> {
    let sync = self.sync_on_rotate();
    if sync {
      a.file.sync_data()?;
      self.counters.syncs.fetch_add(1, Ordering::Relaxed);
    }
    let header = SegmentHeader {
      first_lsn: first,
      prev_lsn: first - 1,
    };
    let (file, path) = create_segment(&self.dir, header, sync)?;
    *a = Active {
      file: Arc::new(file),
      len: HEADER_LEN as u64,
    };
    lock(&self.segments).push(SegmentInfo {
      first_lsn: first,
      path,
    });
    self
      .counters
      .segments_created
      .fetch_add(1, Ordering::Relaxed);
    Ok(())
  }

  fn syncer(self: Arc<Self>, max_delay: Duration) {
    let mut st = lock(&self.state);
    loop {
      if st.shutdown {
        return;
      }
      st = self
        .syncer_wake
        .wait_timeout(st, max_delay)
        .unwrap_or_else(|e| e.into_inner())
        .0;
      if st.shutdown || st.failed.is_some() {
        return;
      }
      // Sync what is already written, without taking the I/O leader role,
      // so appenders keep writing while the fsync runs. Every record up to
      // `target` is in the current segment or in an older one, and older
      // ones were synced when they were rotated out.
      let target = st.written;
      if st.synced >= target {
        continue;
      }
      drop(st);
      let file = lock(&self.active).file.clone();
      let res = file.sync_data();
      st = lock(&self.state);
      match res {
        Ok(()) => {
          self.counters.syncs.fetch_add(1, Ordering::Relaxed);
          st.synced = st.synced.max(target);
        }
        Err(e) => {
          // Poisons the WAL; appenders report it.
          st.failed = Some(e.to_string());
          self.io_done.notify_all();
          return;
        }
      }
    }
  }
}

/// A write-ahead log open for appending. `Send + Sync`; share it by
/// reference or `Arc` among appending threads.
pub struct Wal {
  shared: Arc<Shared>,
  syncer: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Wal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Wal")
      .field("dir", &self.shared.dir)
      .field("last_lsn", &self.last_lsn())
      .finish()
  }
}

impl Wal {
  /// Opens (creating if needed) the WAL in `dir` without replaying it.
  /// Only the last segment is read, to find where the log ends.
  pub fn open(dir: &Path, opts: WalOptions) -> Result<Wal> {
    Ok(Self::recover(dir, opts, Lsn::MAX, |_, _| Ok(()))?.0)
  }

  /// Opens (creating if needed) the WAL in `dir`, passing every valid
  /// record with LSN `>= from` to `replay` in LSN order, and repairs a
  /// torn or corrupt tail (see the module docs). An error from `replay`
  /// aborts the open and is returned.
  pub fn recover<F>(
    dir: &Path,
    opts: WalOptions,
    from: Lsn,
    mut replay: F,
  ) -> Result<(Wal, Recovery)>
  where
    F: FnMut(Lsn, &[u8]) -> Result<()>,
  {
    std::fs::create_dir_all(dir)?;
    let mut report = Recovery::default();
    let mut segs = list_segments(dir)?;
    let mut active: Option<(Active, Lsn)> = None;

    if !segs.is_empty() {
      let last = segs.len() - 1;
      let start = segs
        .iter()
        .rposition(|s| s.first_lsn <= from)
        .unwrap_or(0)
        .min(last);
      report.scan_start_lsn = segs[start].first_lsn;
      // Index, end offset and next LSN of the last good segment scanned.
      let mut good: Option<(usize, u64, Lsn)> = None;
      let mut cut_after: Option<usize> = None;
      for i in start..=last {
        let seg = segs[i].clone();
        report.segments_scanned += 1;
        let region = map_file(&seg.path)?;
        let bytes = region.bytes();
        let prev_end = good.map(|g| g.2 - 1);
        let header = decode_header(bytes).and_then(|h| {
          if h.first_lsn != seg.first_lsn {
            Err(format!("segment header says first LSN {}", h.first_lsn))
          } else {
            Ok(h)
          }
        });
        let header = match header {
          Ok(h) => h,
          Err(reason) => {
            // Unusable header: keep the file as an empty segment (its name
            // still fixes its first LSN) and drop everything after it.
            let len = bytes.len() as u64;
            drop(region);
            let prev = prev_end.unwrap_or(seg.first_lsn - 1);
            reinit_segment(&seg.path, seg.first_lsn, prev)?;
            report.truncated = Some(Truncation {
              segment: seg.path.clone(),
              offset: HEADER_LEN as u64,
              dropped_bytes: len.saturating_sub(HEADER_LEN as u64),
              reason,
            });
            good = Some((i, HEADER_LEN as u64, seg.first_lsn));
            cut_after = Some(i);
            break;
          }
        };
        if let Some(pe) = prev_end
          && header.prev_lsn != pe
        {
          // The previous segment lost records this one follows.
          report.truncated = Some(Truncation {
            segment: segs[i - 1].path.clone(),
            offset: good.unwrap().1,
            dropped_bytes: 0,
            reason: format!(
              "{} continues from LSN {}, but the log ends at {pe}",
              seg.path.display(),
              header.prev_lsn
            ),
          });
          cut_after = Some(i - 1);
          break;
        }
        let end = scan_records(bytes, seg.first_lsn, Lsn::MAX, |lsn, p| {
          if lsn >= from {
            report.records_replayed += 1;
            replay(lsn, p)
          } else {
            Ok(())
          }
        })?;
        let len = bytes.len() as u64;
        drop(region);
        good = Some((i, end.offset as u64, end.next_lsn));
        if let Some(reason) = end.stop {
          let f = OpenOptions::new().write(true).open(&seg.path)?;
          f.set_len(end.offset as u64)?;
          f.sync_all()?;
          report.truncated = Some(Truncation {
            segment: seg.path.clone(),
            offset: end.offset as u64,
            dropped_bytes: len - end.offset as u64,
            reason,
          });
          cut_after = Some(i);
          break;
        }
      }
      if let Some(c) = cut_after {
        for s in segs.drain(c + 1..) {
          let mut aside = s.path.clone().into_os_string();
          aside.push(DISCARDED_SUFFIX);
          let aside = PathBuf::from(aside);
          std::fs::rename(&s.path, &aside)?;
          report.discarded.push(aside);
        }
        sync_dir(dir)?;
      }
      let (gi, end, next) = good.expect("at least one segment scanned");
      let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&segs[gi].path)?;
      file.seek(SeekFrom::Start(end))?;
      active = Some((
        Active {
          file: Arc::new(file),
          len: end,
        },
        next,
      ));
    }

    let min_next = opts.min_next_lsn.max(1);
    let (active, next_lsn) = match active {
      Some((a, next)) if next >= min_next => (a, next),
      other => {
        // Fresh log, or one that must skip ahead: new segment.
        let prev_end = other.as_ref().map_or(min_next - 1, |(_, n)| n - 1);
        let header = SegmentHeader {
          first_lsn: min_next,
          prev_lsn: prev_end,
        };
        let (file, path) = create_segment(dir, header, true)?;
        segs.push(SegmentInfo {
          first_lsn: min_next,
          path,
        });
        let a = Active {
          file: Arc::new(file),
          len: HEADER_LEN as u64,
        };
        if report.segments_scanned == 0 {
          report.scan_start_lsn = min_next;
        }
        (a, min_next)
      }
    };
    report.last_lsn = next_lsn - 1;
    let durable = next_lsn - 1;
    let shared = Arc::new(Shared {
      dir: dir.to_path_buf(),
      opts: opts.clone(),
      state: Mutex::new(State {
        next_lsn,
        buf: Vec::new(),
        spare: Vec::new(),
        written: durable,
        synced: durable,
        io_active: false,
        rotate_requested: false,
        failed: None,
        shutdown: false,
      }),
      io_done: Condvar::new(),
      syncer_wake: Condvar::new(),
      active: Mutex::new(active),
      segments: Mutex::new(segs),
      counters: Counters::default(),
    });
    let syncer = match opts.durability {
      Durability::SyncBatch { max_delay } => {
        let s = shared.clone();
        Some(
          std::thread::Builder::new()
            .name("muninn-wal-sync".into())
            .spawn(move || s.syncer(max_delay))?,
        )
      }
      _ => None,
    };
    Ok((Wal { shared, syncer }, report))
  }

  /// Appends one record and returns its LSN, once the record is as
  /// durable as the [`Durability`] mode promises.
  pub fn append(&self, payload: &[u8]) -> Result<Lsn> {
    Ok(self.append_batch(&[payload])?.start)
  }

  /// Appends records with consecutive LSNs, written together, and returns
  /// their LSN range. An empty batch returns an empty range without I/O.
  pub fn append_batch(&self, payloads: &[&[u8]]) -> Result<Range<Lsn>> {
    if let Some(p) = payloads.iter().find(|p| p.len() > MAX_RECORD_LEN) {
      return Err(Error::invalid(format!(
        "WAL record of {} bytes exceeds the {MAX_RECORD_LEN}-byte limit",
        p.len()
      )));
    }
    // Checksums are computed before taking the lock.
    let mut one = [[0u8; RECORD_HEADER_LEN]; 1];
    let mut many = Vec::new();
    let headers: &[[u8; RECORD_HEADER_LEN]] = if payloads.len() == 1 {
      one[0] = record_header(payloads[0]);
      &one
    } else {
      many.extend(payloads.iter().map(|p| record_header(p)));
      &many
    };
    let bytes: usize =
      payloads.iter().map(|p| p.len() + RECORD_HEADER_LEN).sum();

    let sh = &*self.shared;
    let mut st = lock(&sh.state);
    if let Some(msg) = &st.failed {
      return Err(poisoned(msg));
    }
    let first = st.next_lsn;
    if payloads.is_empty() {
      return Ok(first..first);
    }
    st.buf.reserve(bytes);
    for (h, p) in headers.iter().zip(payloads) {
      st.buf.extend_from_slice(h);
      st.buf.extend_from_slice(p);
    }
    st.next_lsn += payloads.len() as u64;
    let end = st.next_lsn;
    sh.counters
      .records
      .fetch_add(payloads.len() as u64, Ordering::Relaxed);
    let sync = sh.opts.durability == Durability::SyncEachWrite;
    sh.drive(st, end - 1, sync, false)?;
    Ok(first..end)
  }

  /// Makes every record appended so far durable, whatever the mode.
  pub fn sync(&self) -> Result<()> {
    let st = lock(&self.shared.state);
    let target = st.next_lsn - 1;
    self.shared.drive(st, target, true, false)
  }

  /// Closes the current segment and starts a new one, so that
  /// [`Wal::truncate_before`] can delete everything appended so far once
  /// it is checkpointed. No-op if the current segment has no records.
  pub fn rotate(&self) -> Result<()> {
    let mut st = lock(&self.shared.state);
    st.rotate_requested = true;
    self.shared.drive(st, 0, false, true)
  }

  /// LSN of the last record appended (0 if none).
  pub fn last_lsn(&self) -> Lsn {
    lock(&self.shared.state).next_lsn - 1
  }

  /// Every record up to this LSN is on stable storage.
  pub fn durable_lsn(&self) -> Lsn {
    lock(&self.shared.state).synced
  }

  /// The WAL directory.
  pub fn dir(&self) -> &Path {
    &self.shared.dir
  }

  /// Counters since open.
  pub fn stats(&self) -> WalStats {
    let c = &self.shared.counters;
    WalStats {
      records: c.records.load(Ordering::Relaxed),
      bytes: c.bytes.load(Ordering::Relaxed),
      writes: c.writes.load(Ordering::Relaxed),
      syncs: c.syncs.load(Ordering::Relaxed),
      segments_created: c.segments_created.load(Ordering::Relaxed),
    }
  }

  /// Number of segment files in the log.
  pub fn segment_count(&self) -> usize {
    lock(&self.shared.segments).len()
  }

  /// Passes every record with LSN `>= from` appended (and written) so far
  /// to `visit`, in order, and returns how many it passed. Runs alongside
  /// appends. Unlike recovery, an invalid record here is an error: the log
  /// was verified when it was opened.
  pub fn replay<F>(&self, from: Lsn, mut visit: F) -> Result<u64>
  where
    F: FnMut(Lsn, &[u8]) -> Result<()>,
  {
    let end = lock(&self.shared.state).written;
    let segs = lock(&self.shared.segments).clone();
    let mut count = 0u64;
    for (i, seg) in segs.iter().enumerate() {
      let next_first = segs.get(i + 1).map(|s| s.first_lsn);
      if next_first.is_some_and(|n| n <= from) {
        continue;
      }
      if seg.first_lsn > end {
        break;
      }
      let expected_last = next_first.map_or(end, |n| (n - 1).min(end));
      let region = map_file(&seg.path)?;
      let bytes = region.bytes();
      decode_header(bytes)
        .map_err(|r| Error::corrupt(format!("{}: {r}", seg.path.display())))?;
      let r = scan_records(bytes, seg.first_lsn, expected_last, |lsn, p| {
        if lsn >= from {
          count += 1;
          visit(lsn, p)
        } else {
          Ok(())
        }
      })?;
      if r.next_lsn <= expected_last {
        return Err(Error::corrupt(format!(
          "{}: record {} unreadable: {}",
          seg.path.display(),
          r.next_lsn,
          r.stop.unwrap_or_else(|| "file ends early".into())
        )));
      }
    }
    Ok(count)
  }

  /// Deletes segment files holding only records with LSN `< lsn`. Call
  /// after the engine durably checkpointed everything below `lsn`. The
  /// segment being appended to is never deleted. Returns the number of
  /// files deleted.
  pub fn truncate_before(&self, lsn: Lsn) -> Result<usize> {
    let victims: Vec<SegmentInfo> = {
      let mut segs = lock(&self.shared.segments);
      let n = segs
        .windows(2)
        .take_while(|w| w[1].first_lsn <= lsn)
        .count();
      segs.drain(..n).collect()
    };
    for v in &victims {
      match std::fs::remove_file(&v.path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
      }
    }
    if !victims.is_empty() {
      sync_dir(&self.shared.dir)?;
    }
    Ok(victims.len())
  }

  /// Flushes and, unless the mode is `NoSync`, syncs outstanding records,
  /// then closes the log, reporting any error. Dropping a `Wal` does the
  /// same but ignores errors.
  pub fn close(mut self) -> Result<()> {
    self.shutdown()
  }

  fn shutdown(&mut self) -> Result<()> {
    {
      let mut st = lock(&self.shared.state);
      st.shutdown = true;
      self.shared.syncer_wake.notify_all();
    }
    if let Some(h) = self.syncer.take() {
      let _ = h.join();
    }
    if self.shared.opts.durability != Durability::NoSync {
      let st = lock(&self.shared.state);
      if st.failed.is_none() {
        let target = st.next_lsn - 1;
        return self.shared.drive(st, target, true, false);
      }
    }
    Ok(())
  }
}

impl Drop for Wal {
  fn drop(&mut self) {
    let _ = self.shutdown();
  }
}

/// Rewrites a segment as just its header.
fn reinit_segment(path: &Path, first_lsn: Lsn, prev_lsn: Lsn) -> Result<()> {
  let mut f = OpenOptions::new().write(true).open(path)?;
  f.set_len(0)?;
  f.write_all(&encode_header(SegmentHeader {
    first_lsn,
    prev_lsn,
  }))?;
  f.sync_all()?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn segment_names() {
    let p = segment_path(Path::new("w"), 42);
    let name = p.file_name().unwrap().to_str().unwrap();
    assert_eq!(name, "00000000000000000042.wal");
    assert_eq!(parse_segment_name(name), Some(42));
    assert_eq!(parse_segment_name("42.wal"), None);
    assert_eq!(
      parse_segment_name("00000000000000000042.wal.discarded"),
      None
    );
  }
}
