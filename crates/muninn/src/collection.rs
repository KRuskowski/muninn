//! A collection: write path, recovery, flush and compaction.
//!
//! # Write path
//!
//! `upsert`/`delete` validate and encode the batch as one WAL record and
//! append it. Appends run concurrently, so concurrent writers share
//! fsyncs (group commit). Records are then *applied* strictly in LSN
//! order: a writer waits on a condition variable until every lower LSN is
//! applied. The memtable therefore always equals what replaying the log
//! produces, including for conflicting writes to one id. Applying holds
//! the state mutex and the active memtable's write lock, and ends by
//! publishing a new [`Version`], so a search sees all of a batch or none
//! of it.
//!
//! An upsert first deletes the id's current version wherever it lives: the
//! active memtable's id map is authoritative for it; other sources are
//! probed newest first (frozen memtables, then segments) and a hit counts
//! only if not already deleted. At most one live doc per id exists, so the
//! first hit is the only one. No global id map is kept: it would cost
//! memory per document and an O(documents) rebuild at open, while the
//! probes cost one hash lookup per source and compaction keeps sources few.
//!
//! # Flush
//!
//! When the active memtable is full it is frozen (a new one takes writes
//! at once) together with its checkpoint: the LSN of the last record it
//! holds. A single background worker per collection writes it as a
//! segment, persists changed tombstones of every segment, commits a
//! manifest with the new segment and checkpoint, swaps the frozen
//! memtable for the segment in a new version (same doc ids, same
//! tombstones, so searches see no gap and no duplicates), and deletes WAL
//! files below the *previous* manifest's checkpoint: the manifest store
//! falls back to the previous manifest if the newest is unreadable, and
//! that one needs the log from its own checkpoint on.
//!
//! Persisted tombstones may include deletes logged after the checkpoint.
//! Replaying those records again is harmless: deleting an already deleted
//! doc changes nothing, and an upsert that finds no live version simply
//! inserts.
//!
//! # Compaction
//!
//! The worker also merges runs of neighbouring segments (so the order of
//! documents across segments stays the write order) and rewrites segments
//! with many deletes. Deletes that arrive while a merge runs are carried
//! over to the merged segment when it is published. Obsolete files are
//! removed by the manifest store's gc, only from the worker, which is the
//! only thread that creates segments.
//!
//! # Locks
//!
//! Order: `disk` → `state` → memtable `inner` → `current`. Readers take
//! only memtable `inner` (read) and then `current` (read).

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use muninn_core::{BitSet, DocId, Error, Result};
use muninn_store::wal::MAX_RECORD_LEN;
use muninn_store::{
  Lsn, Manifest, ManifestStore, Wal, WalOptions, read_tombstones,
  write_tombstones,
};
use muninn_text::DELETED;

use crate::codec::{self, Op};
use crate::memtable::{MemInner, Memtable, read, write};
use crate::options::{Durability, Failpoint, Options};
use crate::schema::Schema;
use crate::segment::{self, Segment};
use crate::types::{CollectionStats, Document, Id, SegmentStats};
use crate::version::{Source, Version, View};

/// Name of the WAL directory inside a collection directory.
pub(crate) const WAL_DIR: &str = "wal";

/// How long the worker waits before retrying after a failed job.
const RETRY_DELAY: Duration = Duration::from_millis(500);

/// A frozen memtable waiting for the worker.
#[derive(Clone, Copy, Debug)]
struct Frozen {
  mem_id: u64,
  /// LSN of the last record the memtable holds.
  checkpoint: Lsn,
  seq: u64,
}

/// Mutable collection state; see the module docs.
pub(crate) struct State {
  /// The newest version; `current` holds a copy for readers.
  version: Arc<Version>,
  applied_lsn: Lsn,
  /// Checkpoint of the committed manifest.
  checkpoint_lsn: Lsn,
  /// Checkpoint of the newest frozen memtable (or the manifest's).
  frozen_upto: Lsn,
  frozen: VecDeque<Frozen>,
  freeze_seq: u64,
  flushed_seq: u64,
  compact_req: u64,
  compact_done: u64,
  auto_compact_pending: bool,
  next_mem_id: u64,
  /// Writes are refused: an earlier write or the worker failed in a way
  /// that left memory and log out of step.
  poisoned: Option<String>,
  /// Last background error, and a counter bumped on each one.
  bg_error: Option<String>,
  err_gen: u64,
  shutdown: bool,
}

/// What only the worker touches: the manifest store and which tombstones
/// are on disk.
struct Disk {
  store: ManifestStore,
  manifest: Manifest,
  /// Per segment id: tombstone file version, and the bitset written (the
  /// very `Arc` a view held, so an unchanged view is `ptr_eq`).
  tombs: HashMap<u64, (u64, Option<Arc<BitSet>>)>,
}

/// The shared core of a collection.
pub(crate) struct Core {
  pub dir: PathBuf,
  pub name: String,
  pub schema: Schema,
  /// `schema`'s filterable fields, for checking query filters.
  pub filter_schema: muninn_filter::Schema,
  pub opts: Options,
  wal: Wal,
  state: Mutex<State>,
  /// Writers waiting for their turn or for room, and flush/compact
  /// callers waiting for the worker.
  cv: Condvar,
  worker_cv: Condvar,
  pub(crate) current: RwLock<Arc<Version>>,
  disk: Mutex<Disk>,
  dropped: AtomicBool,
}

fn poisoned_err(msg: &str) -> Error {
  Error::Io(std::io::Error::other(format!(
    "collection refuses writes after an earlier failure ({msg}); reopen \
     the database to recover"
  )))
}

fn publish(current: &RwLock<Arc<Version>>, st: &mut State, v: Version) {
  let v = Arc::new(v);
  st.version = v.clone();
  *write(current) = v;
}

/// Applies one WAL record to the memtable and publishes the result.
/// Shared by live writes and recovery.
fn apply_record(
  schema: &Schema,
  current: &RwLock<Arc<Version>>,
  st: &mut State,
  rec: &[u8],
) -> Result<()> {
  let op = codec::decode(rec, schema)?;
  let v = st.version.clone();
  let n = v.views.len();
  let active = v.active().clone();
  let mut g = write(&active.inner);
  let frozen: Vec<(usize, RwLockReadGuard<'_, MemInner>)> = (v.nseg..n - 1)
    .map(|i| (i, read(&v.views[i].memtable().expect("frozen").inner)))
    .collect();
  let mut tombs: Vec<Option<BitSet>> = vec![None; n];
  let mut kill = |id: &Id, g: &mut MemInner| {
    let mark = |i: usize, d: DocId, tombs: &mut Vec<Option<BitSet>>| {
      let t = tombs[i].get_or_insert_with(|| (*v.views[i].tomb).clone());
      t.grow(d as usize + 1);
      t.insert(d);
    };
    let dead = |i: usize, d: DocId, tombs: &Vec<Option<BitSet>>| match &tombs[i]
    {
      Some(t) => t.contains(d),
      None => v.views[i].is_deleted(d),
    };
    if let Some(d) = g.ids.remove(id.as_borrowed()) {
      mark(n - 1, d, &mut tombs);
      return;
    }
    for (i, fg) in frozen.iter().rev() {
      if let Some(d) = fg.ids.get(id.as_borrowed())
        && !dead(*i, d, &tombs)
      {
        mark(*i, d, &mut tombs);
        return;
      }
    }
    for i in (0..v.nseg).rev() {
      let seg = v.views[i].segment().expect("segment");
      if let Some(d) = seg.ids.get(id.as_borrowed())
        && !dead(i, d, &tombs)
      {
        mark(i, d, &mut tombs);
        return;
      }
    }
  };
  let res = (|| -> Result<()> {
    match &op {
      Op::Upsert(docs) => {
        for d in docs {
          kill(&d.id, &mut g);
          active.insert(&mut g, d)?;
        }
      }
      Op::Delete(ids) => {
        for id in ids {
          kill(id, &mut g);
        }
      }
    }
    Ok(())
  })();
  // Publish even a partial batch's rows only together with the error
  // path poisoning the collection; on success, everything at once.
  res?;
  active.publish(&g);
  let mut nv = (*v).clone();
  for (i, t) in tombs.into_iter().enumerate() {
    if let Some(t) = t {
      nv.views[i].deleted = t.count();
      nv.views[i].tomb = Arc::new(t);
    }
  }
  nv.views[n - 1].len = g.docs;
  drop(frozen);
  publish(current, st, nv);
  drop(g);
  Ok(())
}

/// Freezes the active memtable and starts a new one.
fn freeze(
  schema: &Schema,
  current: &RwLock<Arc<Version>>,
  st: &mut State,
) -> Result<()> {
  let m = Memtable::new(st.next_mem_id, schema)?;
  st.next_mem_id += 1;
  let old = st.version.active().id;
  let mut nv = (*st.version).clone();
  nv.views.push(View::mem(Arc::new(m)));
  st.freeze_seq += 1;
  st.frozen.push_back(Frozen {
    mem_id: old,
    checkpoint: st.applied_lsn,
    seq: st.freeze_seq,
  });
  st.frozen_upto = st.applied_lsn;
  publish(current, st, nv);
  Ok(())
}

/// Whether the active memtable reached a flush threshold.
fn is_full(opts: &Options, st: &State) -> bool {
  let a = st.version.active_view();
  if a.len as usize >= opts.memtable_max_docs {
    return true;
  }
  let bytes = read(&st.version.active().inner).bytes;
  bytes >= opts.memtable_max_bytes
}

/// Removes files of segments a crashed flush or merge wrote but never
/// committed: ids at or above the manifest's `next_segment_id`. The
/// store's gc keeps those, because in a running collection they belong to
/// work in flight; at open nothing is in flight (the directory is locked).
fn remove_uncommitted(dir: &Path, next_id: u64) -> Result<()> {
  let mut removed = false;
  for e in std::fs::read_dir(dir)? {
    let e = e?;
    let name = e.file_name();
    let Some(rest) = name.to_str().and_then(|n| n.strip_prefix("seg-")) else {
      continue;
    };
    let digits: String =
      rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() || !e.file_type()?.is_file() {
      continue;
    }
    if digits.parse::<u64>().is_ok_and(|id| id >= next_id) {
      std::fs::remove_file(e.path())?;
      removed = true;
    }
  }
  if removed {
    muninn_store::fs::sync_dir(dir)?;
  }
  Ok(())
}

/// Picks segments to merge in the background: a contiguous range of
/// `segs` (live, total docs per segment).
fn pick_merge(
  segs: &[(u64, u64)],
  opts: &Options,
) -> Option<std::ops::Range<usize>> {
  let n = segs.len();
  let base = opts.memtable_max_docs.max(1) as u64;
  let f = opts.merge_factor;
  let cap = opts.max_merge_docs as u64;
  // 1. Bound the number of segments: merge the smallest neighbours.
  if n > 3 * f + 8 {
    let best = (0..n - 1)
      .map(|i| (segs[i].0 + segs[i + 1].0, i))
      .filter(|&(s, _)| s <= cap)
      .min();
    if let Some((_, i)) = best {
      return Some(i..i + 2);
    }
  }
  // 2. Size tiers: `f` neighbours of similar size.
  let tier = |live: u64| {
    let (mut t, mut cap) = (0u32, base);
    while live > cap {
      cap = cap.saturating_mul(f as u64);
      t += 1;
    }
    t
  };
  if n >= f {
    for i in 0..=n - f {
      let w = &segs[i..i + f];
      let t = tier(w[0].0);
      let sum: u64 = w.iter().map(|s| s.0).sum();
      if w.iter().all(|s| tier(s.0) == t) && sum <= cap {
        return Some(i..i + f);
      }
    }
  }
  // 3. Reclaim space from segments with many deletes. Last, so constant
  // deletes cannot starve the merges above.
  for (i, &(live, total)) in segs.iter().enumerate() {
    if total > 0
      && (total - live) as f64 / total as f64 >= opts.compact_deleted_ratio
    {
      return Some(i..i + 1);
    }
  }
  None
}

enum Job {
  Flush(Frozen),
  Compact { full: bool, req: u64 },
}

impl Core {
  /// Creates the on-disk skeleton of a new collection at `dir` (which must
  /// not exist), atomically: built under a temporary name, then renamed.
  pub(crate) fn create_dir(dir: &Path, schema: &Schema) -> Result<()> {
    let parent = muninn_store::fs::temp_path_for(dir);
    let tmp = parent.with_extension("creating");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    let (mut store, cur) = ManifestStore::open(&tmp)?;
    debug_assert!(cur.is_none());
    let mut m = Manifest::new(schema.to_json());
    store.commit(&mut m)?;
    Wal::open(&tmp.join(WAL_DIR), WalOptions::default())?.close()?;
    std::fs::rename(&tmp, dir)?;
    muninn_store::fs::sync_dir(dir.parent().unwrap_or(Path::new(".")))?;
    Ok(())
  }

  /// Opens the collection at `dir`: loads the manifest and segments and
  /// replays the WAL.
  pub(crate) fn open(dir: &Path, name: &str, opts: &Options) -> Result<Core> {
    let (store, m) = ManifestStore::open(dir)?;
    let m = m.ok_or_else(|| {
      Error::corrupt(format!("{} has no manifest", dir.display()))
    })?;
    let schema = Schema::from_json(&m.schema)?;
    remove_uncommitted(dir, m.next_segment_id)?;
    store.gc(true, &[])?;
    let mut views = Vec::with_capacity(m.segments.len() + 1);
    let mut tombs = HashMap::new();
    for e in &m.segments {
      let seg = Arc::new(Segment::open(dir, e, &schema)?);
      let mut view = View {
        len: seg.len,
        src: Source::Seg(seg.clone()),
        tomb: Arc::new(BitSet::new(0)),
        deleted: 0,
      };
      match &e.tombstones {
        Some(name) => {
          let t = read_tombstones(&dir.join(name))?;
          if t.segment_id != e.id || t.deleted.capacity() > seg.len as usize {
            return Err(Error::corrupt(format!(
              "tombstones {name} do not belong to segment {}",
              e.id
            )));
          }
          let bits = Arc::new(t.deleted);
          view.deleted = bits.count();
          view.tomb = bits.clone();
          tombs.insert(e.id, (t.version, Some(bits)));
        }
        None => {
          tombs.insert(e.id, (0, None));
        }
      }
      views.push(view);
    }
    let nseg = views.len();
    views.push(View::mem(Arc::new(Memtable::new(1, &schema)?)));
    let version = Arc::new(Version { views, nseg });
    let current = RwLock::new(version.clone());
    let cp = m.wal_checkpoint_lsn;
    let mut st = State {
      version,
      applied_lsn: cp,
      checkpoint_lsn: cp,
      frozen_upto: cp,
      frozen: VecDeque::new(),
      freeze_seq: 0,
      flushed_seq: 0,
      compact_req: 0,
      compact_done: 0,
      auto_compact_pending: true,
      next_mem_id: 2,
      poisoned: None,
      bg_error: None,
      err_gen: 0,
      shutdown: false,
    };
    let wal_opts = WalOptions {
      durability: opts.durability,
      segment_size: opts.wal_segment_size,
      min_next_lsn: cp + 1,
    };
    let (wal, rec) =
      Wal::recover(&dir.join(WAL_DIR), wal_opts, cp + 1, |lsn, payload| {
        apply_record(&schema, &current, &mut st, payload)?;
        st.applied_lsn = lsn;
        if is_full(opts, &st) {
          freeze(&schema, &current, &mut st)?;
        }
        Ok(())
      })?;
    if rec.scan_start_lsn > cp + 1 && rec.last_lsn >= rec.scan_start_lsn {
      return Err(Error::corrupt(format!(
        "{}: the write-ahead log lacks records {} to {} after the last \
         checkpoint",
        dir.display(),
        cp + 1,
        rec.scan_start_lsn - 1
      )));
    }
    st.applied_lsn = st.applied_lsn.max(wal.last_lsn());
    Ok(Core {
      dir: dir.to_path_buf(),
      name: name.to_string(),
      filter_schema: schema.filter_schema()?,
      schema,
      opts: opts.clone(),
      wal,
      state: Mutex::new(st),
      cv: Condvar::new(),
      worker_cv: Condvar::new(),
      current,
      disk: Mutex::new(Disk {
        store,
        manifest: m,
        tombs,
      }),
      dropped: AtomicBool::new(false),
    })
  }

  fn lock(&self) -> Result<MutexGuard<'_, State>> {
    self
      .state
      .lock()
      .map_err(|_| poisoned_err("a thread panicked while writing"))
  }

  fn wait<'a>(
    &self,
    g: MutexGuard<'a, State>,
  ) -> Result<MutexGuard<'a, State>> {
    self
      .cv
      .wait(g)
      .map_err(|_| poisoned_err("a thread panicked while writing"))
  }

  /// The newest snapshot.
  pub(crate) fn snapshot(&self) -> Arc<Version> {
    read(&self.current).clone()
  }

  pub(crate) fn check_live(&self) -> Result<()> {
    if self.dropped.load(Ordering::Acquire) {
      return Err(Error::NotFound(format!(
        "collection {:?} was dropped",
        self.name
      )));
    }
    Ok(())
  }

  pub(crate) fn mark_dropped(&self) {
    self.dropped.store(true, Ordering::Release);
  }

  /// Validates, logs and applies one batch.
  fn write(&self, rec: Vec<u8>) -> Result<()> {
    self.check_live()?;
    if rec.len() > MAX_RECORD_LEN {
      return Err(Error::invalid(format!(
        "batch encodes to {} bytes, above the {MAX_RECORD_LEN}-byte limit; \
         split it",
        rec.len()
      )));
    }
    {
      let st = self.lock()?;
      if let Some(m) = &st.poisoned {
        return Err(poisoned_err(m));
      }
    }
    // A failed append consumed no LSN that a later append could follow:
    // the WAL refuses every append after an I/O failure, and an append
    // that succeeded implies every lower LSN succeeded too. So writers
    // that got an LSN always find their predecessors applied.
    let lsn = self.wal.append(&rec)?;
    let mut st = self.lock()?;
    loop {
      if let Some(m) = &st.poisoned {
        return Err(poisoned_err(m));
      }
      if st.applied_lsn + 1 >= lsn {
        break;
      }
      st = self.wait(st)?;
    }
    if st.applied_lsn + 1 != lsn {
      let msg = format!("LSN {lsn} after applied {}", st.applied_lsn);
      st.poisoned = Some(msg.clone());
      self.cv.notify_all();
      return Err(poisoned_err(&msg));
    }
    // Room for the memtable: a bounded number of pending flushes. If the
    // worker keeps failing, stop waiting (the record is logged and must
    // be applied in order); the failure surfaces through flush().
    let gen0 = st.err_gen;
    while st.frozen.len() > self.opts.max_pending_flushes
      && !st.shutdown
      && st.err_gen == gen0
    {
      if let Some(m) = &st.poisoned {
        return Err(poisoned_err(m));
      }
      st = self.wait(st)?;
    }
    if let Err(e) = apply_record(&self.schema, &self.current, &mut st, &rec) {
      st.poisoned = Some(format!("applying LSN {lsn}: {e}"));
      self.cv.notify_all();
      return Err(e);
    }
    st.applied_lsn = lsn;
    if is_full(&self.opts, &st) {
      freeze(&self.schema, &self.current, &mut st)?;
      self.worker_cv.notify_one();
    }
    self.cv.notify_all();
    Ok(())
  }

  pub(crate) fn upsert(&self, docs: &[Document]) -> Result<()> {
    if docs.is_empty() {
      return Ok(());
    }
    self.write(codec::encode_upsert(&self.schema, docs)?)
  }

  pub(crate) fn delete(&self, ids: &[Id]) -> Result<()> {
    if ids.is_empty() {
      return Ok(());
    }
    self.write(codec::encode_delete(ids)?)
  }

  /// Waits until `done(state)` or a failure.
  fn wait_for(
    &self,
    mut st: MutexGuard<'_, State>,
    done: impl Fn(&State) -> bool,
  ) -> Result<()> {
    let gen0 = st.err_gen;
    loop {
      if done(&st) {
        return Ok(());
      }
      if let Some(m) = &st.poisoned {
        return Err(poisoned_err(m));
      }
      if st.err_gen != gen0 {
        let msg = st.bg_error.clone().unwrap_or_default();
        return Err(Error::Io(std::io::Error::other(format!(
          "background flush failed: {msg}"
        ))));
      }
      if st.shutdown {
        return Err(Error::invalid("collection is closing"));
      }
      st = self.wait(st)?;
    }
  }

  /// Writes everything applied so far to segments.
  pub(crate) fn flush(&self) -> Result<()> {
    self.check_live()?;
    let mut st = self.lock()?;
    if let Some(m) = &st.poisoned {
      return Err(poisoned_err(m));
    }
    if st.applied_lsn > st.frozen_upto {
      freeze(&self.schema, &self.current, &mut st)?;
    }
    let target = st.freeze_seq;
    self.worker_cv.notify_one();
    self.wait_for(st, |s| s.flushed_seq >= target)
  }

  /// Flushes, then merges every segment into one without deleted docs.
  pub(crate) fn compact(&self) -> Result<()> {
    self.flush()?;
    let mut st = self.lock()?;
    st.compact_req += 1;
    let req = st.compact_req;
    self.worker_cv.notify_one();
    self.wait_for(st, |s| s.compact_done >= req)
  }

  /// Stops the worker after its current job.
  pub(crate) fn shutdown(&self) {
    if let Ok(mut st) = self.state.lock() {
      st.shutdown = true;
    }
    self.worker_cv.notify_all();
    self.cv.notify_all();
  }

  fn failpoint(&self, at: Failpoint) -> Result<()> {
    if self.opts.failpoint == Some(at) {
      let mut st = self.lock()?;
      st.poisoned = Some(format!("failpoint {at:?}"));
      st.shutdown = true;
      self.cv.notify_all();
      return Err(Error::Io(std::io::Error::other(format!(
        "failpoint {at:?}"
      ))));
    }
    Ok(())
  }

  /// See [`Failpoint::SlowPublish`].
  fn pause(&self) {
    if self.opts.failpoint == Some(Failpoint::SlowPublish) {
      std::thread::sleep(Duration::from_millis(20));
    }
  }

  /// The worker thread's loop.
  pub(crate) fn run_worker(self: Arc<Self>) {
    let mut retry_wait = false;
    let mut flushed_last = false;
    loop {
      let job = {
        let Ok(mut st) = self.state.lock() else {
          return;
        };
        if retry_wait && !st.shutdown {
          st = match self.worker_cv.wait_timeout(st, RETRY_DELAY) {
            Ok((g, _)) => g,
            Err(_) => return,
          };
        }
        loop {
          if st.shutdown || st.poisoned.is_some() {
            return;
          }
          let compaction = if st.compact_req > st.compact_done {
            Some(Job::Compact {
              full: true,
              req: st.compact_req,
            })
          } else if self.opts.auto_compact && st.auto_compact_pending {
            Some(Job::Compact {
              full: false,
              req: 0,
            })
          } else {
            None
          };
          // Flushes first, but under sustained writes alternate with
          // compaction so merges (and explicit compact calls) are not
          // starved; writers blocked on the flush backlog always win.
          let blocked = st.frozen.len() > self.opts.max_pending_flushes;
          match (st.frozen.front(), compaction) {
            (Some(f), Some(c)) => {
              if blocked || !flushed_last {
                break Job::Flush(*f);
              }
              break c;
            }
            (Some(f), None) => break Job::Flush(*f),
            (None, Some(c)) => break c,
            (None, None) => {}
          }
          st = match self.worker_cv.wait(st) {
            Ok(g) => g,
            Err(_) => return,
          };
        }
      };
      flushed_last = matches!(job, Job::Flush(_));
      let res = match job {
        Job::Flush(f) => self.flush_one(f),
        Job::Compact { full, req } => {
          let r = self.compact_once(full);
          if let Ok(mut st) = self.state.lock() {
            match (&r, full) {
              (Ok(_), true) => st.compact_done = st.compact_done.max(req),
              (Ok(false), false) => st.auto_compact_pending = false,
              _ => {}
            }
            self.cv.notify_all();
          }
          r.map(|_| ())
        }
      };
      retry_wait = false;
      if let Err(e) = res {
        let Ok(mut st) = self.state.lock() else {
          return;
        };
        if st.poisoned.is_some() {
          self.cv.notify_all();
          return;
        }
        st.bg_error = Some(e.to_string());
        st.err_gen += 1;
        self.cv.notify_all();
        retry_wait = true;
      }
    }
  }

  /// Writes the oldest frozen memtable as a segment and publishes it.
  fn flush_one(&self, f: Frozen) -> Result<()> {
    self.wal.rotate()?;
    if self.opts.durability == Durability::NoSync {
      // Without syncs the WAL keeps every rotated-out segment file open
      // until the next `Wal::sync`; one rotation per flush would leak a
      // file handle per flush. One fsync per flush releases them.
      self.wal.sync()?;
    }
    let mut disk = self.disk.lock().map_err(|_| poisoned_err("disk lock"))?;
    let v = self.snapshot();
    let mem = v.views[v.nseg..]
      .iter()
      .filter_map(View::memtable)
      .find(|m| m.id == f.mem_id)
      .cloned()
      .ok_or_else(|| Error::invalid("frozen memtable vanished"))?;
    drop(v);
    let seg_id = disk.manifest.next_segment_id;
    let seg = {
      let g = read(&mem.inner);
      if g.docs > 0 {
        let entry = segment::write_from_memtable(
          &self.dir,
          seg_id,
          &mem,
          &g,
          &self.schema,
          &self.opts,
        )?;
        Some(Arc::new(Segment::open(&self.dir, &entry, &self.schema)?))
      } else {
        None
      }
    };
    self.failpoint(Failpoint::AfterSegmentWrite)?;
    // Tombstones to persist: every segment's current ones, and the new
    // segment's (the frozen memtable's, same doc ids).
    let mut targets: Vec<(u64, Arc<BitSet>, usize, u32)> = Vec::new();
    {
      let st = self.lock()?;
      let v = &st.version;
      for view in &v.views[..v.nseg] {
        let s = view.segment().expect("segment");
        targets.push((s.id, view.tomb.clone(), view.deleted, s.len));
      }
      if let Some(s) = &seg {
        let view = v.views[v.nseg..]
          .iter()
          .find(|w| w.memtable().is_some_and(|m| m.id == f.mem_id))
          .ok_or_else(|| Error::invalid("frozen memtable vanished"))?;
        targets.push((s.id, view.tomb.clone(), view.deleted, s.len));
      }
    }
    self.pause();
    let mut m = disk.manifest.clone();
    let prev_checkpoint = m.wal_checkpoint_lsn;
    if let Some(s) = &seg {
      m.segments.push(s.entry.clone());
      m.next_segment_id = seg_id + 1;
    }
    let mut written = Vec::new();
    for (id, tomb, deleted, len) in targets {
      let (ver, persisted) = disk.tombs.get(&id).cloned().unwrap_or((0, None));
      let changed = match &persisted {
        Some(p) => !Arc::ptr_eq(p, &tomb),
        None => deleted > 0,
      };
      if !changed {
        continue;
      }
      let mut bits = (*tomb).clone();
      bits.grow(len as usize);
      let name = write_tombstones(&self.dir, id, ver + 1, &bits)?;
      if let Some(e) = m.segments.iter_mut().find(|e| e.id == id) {
        e.tombstones = Some(name);
      }
      written.push((id, ver + 1, tomb));
    }
    m.wal_checkpoint_lsn = f.checkpoint.max(prev_checkpoint);
    disk.store.commit(&mut m)?;
    disk.manifest = m;
    for (id, ver, t) in written {
      disk.tombs.insert(id, (ver, Some(t)));
    }
    if let Some(s) = &seg {
      disk.tombs.entry(s.id).or_insert((0, None));
    }
    self.failpoint(Failpoint::AfterManifestCommit)?;
    {
      let mut st = self.lock()?;
      let mut nv = (*st.version).clone();
      let i = nv.nseg;
      let is_f = nv.views[i].memtable().is_some_and(|m| m.id == f.mem_id);
      if !is_f {
        return Err(Error::invalid("frozen memtables out of order"));
      }
      match &seg {
        Some(s) => {
          let old = &nv.views[i];
          nv.views[i] = View {
            src: Source::Seg(s.clone()),
            len: s.len,
            tomb: old.tomb.clone(),
            deleted: old.deleted,
          };
          nv.nseg += 1;
        }
        None => {
          nv.views.remove(i);
        }
      }
      st.frozen.pop_front();
      st.flushed_seq = f.seq;
      st.checkpoint_lsn = f.checkpoint.max(st.checkpoint_lsn);
      st.auto_compact_pending = true;
      publish(&self.current, &mut st, nv);
      self.cv.notify_all();
    }
    self.wal.truncate_before(prev_checkpoint + 1)?;
    disk.store.gc(true, &[])?;
    Ok(())
  }

  /// One merge (all segments if `full`, else what the policy picks).
  /// Returns whether anything was merged.
  fn compact_once(&self, full: bool) -> Result<bool> {
    let mut disk = self.disk.lock().map_err(|_| poisoned_err("disk lock"))?;
    let v = self.snapshot();
    let segs: Vec<(Arc<Segment>, Arc<BitSet>, usize)> = v.views[..v.nseg]
      .iter()
      .map(|w| {
        (
          w.segment().expect("segment").clone(),
          w.tomb.clone(),
          w.deleted,
        )
      })
      .collect();
    drop(v);
    let range = if full {
      if segs.len() > 1 || segs.iter().any(|s| s.2 > 0) {
        0..segs.len()
      } else {
        return Ok(false);
      }
    } else {
      let sizes: Vec<(u64, u64)> = segs
        .iter()
        .map(|(s, _, d)| (u64::from(s.len) - *d as u64, u64::from(s.len)))
        .collect();
      match pick_merge(&sizes, &self.opts) {
        Some(r) => r,
        None => return Ok(false),
      }
    };
    let run = &segs[range.clone()];
    let id = disk.manifest.next_segment_id;
    let inputs: Vec<(&Segment, &BitSet)> =
      run.iter().map(|(s, t, _)| (&**s, &**t)).collect();
    let merged =
      segment::write_merged(&self.dir, id, &inputs, &self.schema, &self.opts)?;
    let merged = match merged {
      Some((entry, maps)) => Some((
        Arc::new(Segment::open(&self.dir, &entry, &self.schema)?),
        maps,
      )),
      None => None,
    };
    self.failpoint(Failpoint::AfterCompactionWrite)?;
    self.pause();
    let run_ids: HashSet<u64> = run.iter().map(|(s, _, _)| s.id).collect();
    let mut m = disk.manifest.clone();
    let pos = m
      .segments
      .iter()
      .position(|e| run_ids.contains(&e.id))
      .ok_or_else(|| Error::invalid("merged segments not in manifest"))?;
    let in_order = m.segments[pos..]
      .iter()
      .take(run.len())
      .map(|e| e.id)
      .eq(run.iter().map(|(s, _, _)| s.id));
    if !in_order {
      return Err(Error::invalid("manifest and memory disagree on segments"));
    }
    let replacement: Vec<_> =
      merged.iter().map(|(s, _)| s.entry.clone()).collect();
    m.segments.splice(pos..pos + run.len(), replacement);
    if merged.is_some() {
      m.next_segment_id = id + 1;
    }
    disk.store.commit(&mut m)?;
    disk.manifest = m;
    for sid in &run_ids {
      disk.tombs.remove(sid);
    }
    if let Some((s, _)) = &merged {
      disk.tombs.insert(s.id, (0, None));
    }
    {
      let mut st = self.lock()?;
      let mut nv = (*st.version).clone();
      let same = nv.views[range.clone()]
        .iter()
        .zip(run)
        .all(|(w, (s, _, _))| w.segment().is_some_and(|x| Arc::ptr_eq(x, s)));
      if !same {
        return Err(Error::invalid("segments changed during a merge"));
      }
      let new_view = merged.as_ref().map(|(s, maps)| {
        let mut bits = BitSet::new(s.len as usize);
        for (k, w) in nv.views[range.clone()].iter().enumerate() {
          let t0 = &run[k].1;
          for d in w.tomb.iter() {
            if !t0.contains(d) {
              let new = maps[k][d as usize];
              if new != DELETED {
                bits.insert(new);
              }
            }
          }
        }
        let deleted = bits.count();
        View {
          src: Source::Seg(s.clone()),
          len: s.len,
          tomb: Arc::new(bits),
          deleted,
        }
      });
      let removed = range.len();
      let added = usize::from(new_view.is_some());
      nv.views.splice(range.clone(), new_view);
      nv.nseg = nv.nseg + added - removed;
      publish(&self.current, &mut st, nv);
      self.cv.notify_all();
    }
    disk.store.gc(true, &[])?;
    Ok(true)
  }

  /// Sizes and counters.
  pub(crate) fn stats(&self) -> Result<CollectionStats> {
    self.check_live()?;
    let (v, last, cp, pending) = {
      let st = self.lock()?;
      (
        st.version.clone(),
        st.applied_lsn,
        st.checkpoint_lsn,
        st.frozen.len(),
      )
    };
    let mut s = CollectionStats {
      pending_flushes: pending,
      last_lsn: last,
      checkpoint_lsn: cp,
      ..Default::default()
    };
    for w in &v.views {
      let live = u64::from(w.len) - w.deleted as u64;
      s.live_docs += live;
      s.deleted_docs += w.deleted as u64;
      match &w.src {
        Source::Mem(_) => s.memtable_docs += u64::from(w.len),
        Source::Seg(seg) => s.segments.push(SegmentStats {
          id: seg.id,
          docs: u64::from(seg.len),
          deleted: w.deleted as u64,
          vector_index: seg.vectors.as_ref().map_or("none", |x| x.kind_name()),
        }),
      }
    }
    Ok(s)
  }
}

/// A collection handle. Cheap to clone; all clones share one collection.
/// Every method may be called from many threads at once.
#[derive(Clone)]
pub struct Collection {
  pub(crate) h: Arc<Handle>,
}

impl std::fmt::Debug for Collection {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Collection")
      .field("name", &self.h.core.name)
      .finish()
  }
}

/// Owns the worker thread; stops it when the last handle goes away.
pub(crate) struct Handle {
  pub core: Arc<Core>,
  worker: Mutex<Option<JoinHandle<()>>>,
  /// Keeps the database directory locked while any handle lives.
  _lock: Arc<muninn_store::DirLock>,
}

impl Handle {
  pub(crate) fn start(
    core: Core,
    lock: Arc<muninn_store::DirLock>,
  ) -> Result<Arc<Handle>> {
    let core = Arc::new(core);
    let c = core.clone();
    let t = std::thread::Builder::new()
      .name(format!("muninn-{}", core.name))
      .spawn(move || c.run_worker())?;
    Ok(Arc::new(Handle {
      core,
      worker: Mutex::new(Some(t)),
      _lock: lock,
    }))
  }

  /// Stops the worker and waits for it.
  pub(crate) fn stop(&self) {
    self.core.shutdown();
    let t = self.worker.lock().ok().and_then(|mut g| g.take());
    if let Some(t) = t {
      let _ = t.join();
    }
  }
}

impl Drop for Handle {
  fn drop(&mut self) {
    self.stop();
  }
}

impl Collection {
  /// The collection's name.
  pub fn name(&self) -> &str {
    &self.h.core.name
  }

  /// The collection's schema.
  pub fn schema(&self) -> &Schema {
    &self.h.core.schema
  }

  /// Inserts or replaces documents, atomically: after a crash the whole
  /// batch is there or none of it, and no search sees part of it. Returns
  /// once the batch is as durable as [`crate::Options::durability`]
  /// promises and visible to searches.
  ///
  /// An error means the batch may or may not have been applied (for
  /// example when the disk failed after logging it); validation errors
  /// (wrong dimension, unknown field) mean it was not.
  pub fn upsert(&self, docs: &[Document]) -> Result<()> {
    self.h.core.upsert(docs)
  }

  /// Deletes documents by id, atomically like [`Collection::upsert`].
  /// Unknown ids are ignored.
  pub fn delete(&self, ids: &[Id]) -> Result<()> {
    self.h.core.delete(ids)
  }

  /// The current version of a document, if it exists. For cosine
  /// collections the returned vector is normalised.
  pub fn get(&self, id: &Id) -> Result<Option<Document>> {
    self.h.core.check_live()?;
    crate::search::get(&self.h.core, id)
  }

  /// Searches; see [`crate::Query`].
  pub fn search(&self, q: &crate::Query) -> Result<Vec<crate::SearchHit>> {
    self.h.core.check_live()?;
    crate::search::search(&self.h.core, q)
  }

  /// Writes everything written so far into segments and shortens the
  /// write-ahead log. Not needed for durability (the log provides it);
  /// useful before a backup or to make the next open faster.
  pub fn flush(&self) -> Result<()> {
    self.h.core.flush()
  }

  /// Flushes, then merges all segments into one, dropping deleted and
  /// replaced documents for good. Searches and writes continue meanwhile.
  pub fn compact(&self) -> Result<()> {
    self.h.core.compact()
  }

  /// Sizes and counters.
  pub fn stats(&self) -> Result<CollectionStats> {
    self.h.core.stats()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn opts() -> Options {
    Options {
      memtable_max_docs: 100,
      merge_factor: 4,
      ..Options::default()
    }
  }

  #[test]
  fn policy_bounds_count_then_tiers_then_deletes() {
    let o = opts();
    assert_eq!(pick_merge(&[(100, 100), (10, 100)], &o), Some(1..2));
    // A tier window wins over a segment with many deletes.
    let s = [(10, 100), (90, 90), (100, 100), (50, 50), (80, 80)];
    assert_eq!(pick_merge(&s, &o), Some(0..4));
    assert_eq!(pick_merge(&[(100, 100); 3], &o), None);
    let s = [(1000, 1000), (90, 90), (100, 100), (50, 50), (80, 80)];
    assert_eq!(pick_merge(&s, &o), Some(1..5));
    // A different tier in the window blocks it.
    let s = [(90, 90), (1000, 1000), (100, 100), (50, 50), (80, 80)];
    assert_eq!(pick_merge(&s, &o), None);
    // Too many segments: smallest neighbouring pair.
    let mut s = vec![(1000u64, 1000u64); 3 * 4 + 9];
    for (i, x) in s.iter_mut().enumerate() {
      if i % 2 == 0 {
        *x = (5000 + i as u64, 5000 + i as u64);
      }
    }
    s[7] = (1, 1);
    let r = pick_merge(&s, &o).unwrap();
    assert!(r.contains(&7) && r.len() == 2);
  }
}
