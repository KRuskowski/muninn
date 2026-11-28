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
//! tombstones, so searches see no gap and no duplicates), commits the
//! manifest a second time, unchanged, and only then deletes WAL files
//! below the checkpoint. The manifest store keeps two generations and
//! falls back to the older one if the newest is unreadable; until the
//! second commit that older one is the state before the flush, which
//! needs the log from its own checkpoint on. Merges commit twice too, so
//! the segments they replace are deleted as soon as the merge is done.
//! Disk use is therefore bounded by the live segments, the log since the
//! last flush (at most the memtables in memory), and the output of a
//! merge in progress. (On Windows a file still mapped by a running search
//! cannot be deleted; gc skips it, and the next flush, merge or open
//! removes it.)
//!
//! While writes outpace the worker (more frozen memtables wait, or the
//! collection already ends with such segments), a flush in a collection
//! with an HNSW schema writes its segment with a flat index: its vectors
//! are about to be inserted into a larger graph by a merge anyway (see
//! [`pick_fold`] and `segment::write_merged`, which extends the graph of
//! a merge's first segment instead of rebuilding it). A bulk load then
//! puts every vector into a graph about once, instead of once per flush
//! plus once per merge level.
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
use crate::options::{Failpoint, Options};
use crate::schema::{IndexKind, Schema};
use crate::segment::{self, Segment};
use crate::types::{CollectionStats, Document, Id, SegmentStats};
use crate::version::{Source, Version, View};

/// Name of the WAL directory inside a collection directory.
pub(crate) const WAL_DIR: &str = "wal";

/// Marker file left by a recovery that found the log damaged before its
/// end; the collection refuses to open while it exists.
pub(crate) const WAL_DAMAGED: &str = "WAL-DAMAGED";

/// Shortest wait between checks for a pause in writes (see
/// [`Options::merge_idle_delay`]).
const MIN_IDLE_CHECK: Duration = Duration::from_millis(1);

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
  /// Highest full-compaction request that failed, and why.
  compact_failed: u64,
  compact_error: Option<String>,
  auto_compact_pending: bool,
  /// Segments without a graph wait at the end of the collection for a
  /// merge that builds one (see [`pick_fold`]); the applied LSN when the
  /// worker last looked. If no write arrives for
  /// [`Options::merge_idle_delay`], they are merged anyway.
  fold_waiting: Option<Lsn>,
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
  /// `None` once the collection is dropped (its files are released).
  wal: RwLock<Option<Wal>>,
  state: Mutex<State>,
  /// Writers waiting for their turn or for room, and flush/compact
  /// callers waiting for the worker.
  cv: Condvar,
  worker_cv: Condvar,
  pub(crate) current: RwLock<Arc<Version>>,
  disk: Mutex<Disk>,
  dropped: AtomicBool,
  /// Must stay the last field: fields drop in declaration order, so this
  /// runs after the WAL and the segment maps owned above are released.
  remover: DirRemover,
}

/// Marker file in a dropped collection's directory that could not be
/// renamed aside; such a directory is deleted, never opened.
pub(crate) const DROPPED_MARKER: &str = "DROPPED";

/// Deletes a dropped collection's directory. A directory carrying the
/// [`DROPPED_MARKER`] loses the marker last, so a removal that fails
/// half-way (files still in use) never leaves remains that look like a
/// collection.
pub(crate) fn remove_collection_dir(path: &Path) -> std::io::Result<()> {
  let marker = path.join(DROPPED_MARKER);
  if !marker.exists() {
    return std::fs::remove_dir_all(path);
  }
  for e in std::fs::read_dir(path)? {
    let e = e?;
    if e.file_name() == DROPPED_MARKER {
      continue;
    }
    if e.file_type()?.is_dir() {
      std::fs::remove_dir_all(e.path())?;
    } else {
      std::fs::remove_file(e.path())?;
    }
  }
  std::fs::remove_file(&marker)?;
  std::fs::remove_dir(path)
}

/// Deletes a dropped collection's directory once the collection is
/// released. On Windows a directory cannot be removed while files in it
/// are open or mapped, so deleting at drop time would fail whenever a
/// handle or a running search still uses the collection. If something
/// outside the core (a search still holding a snapshot) keeps files
/// mapped, removal fails here and `Database::open` sweeps the directory.
struct DirRemover(Mutex<Option<PathBuf>>);

impl Drop for DirRemover {
  fn drop(&mut self) {
    let path = match self.0.get_mut() {
      Ok(p) => p.take(),
      Err(e) => e.into_inner().take(),
    };
    if let Some(p) = path {
      let _ = remove_collection_dir(&p);
    }
  }
}

fn poisoned_err(msg: &str) -> Error {
  Error::Unavailable(format!(
    "collection refuses writes after an earlier failure ({msg}); reopen \
     the database to recover"
  ))
}

fn publish(current: &RwLock<Arc<Version>>, st: &mut State, v: Version) {
  let v = Arc::new(v);
  st.version = v.clone();
  *write(current) = v;
}

/// Applies one WAL record to the memtable and publishes the result.
/// Shared by live writes and recovery. Returns how many live documents
/// the record deleted or replaced.
fn apply_record(
  schema: &Schema,
  current: &RwLock<Arc<Version>>,
  st: &mut State,
  rec: &[u8],
) -> Result<usize> {
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
      return true;
    }
    for (i, fg) in frozen.iter().rev() {
      if let Some(d) = fg.ids.get(id.as_borrowed())
        && !dead(*i, d, &tombs)
      {
        mark(*i, d, &mut tombs);
        return true;
      }
    }
    for i in (0..v.nseg).rev() {
      let seg = v.views[i].segment().expect("segment");
      if let Some(d) = seg.ids.get(id.as_borrowed())
        && !dead(i, d, &tombs)
      {
        mark(i, d, &mut tombs);
        return true;
      }
    }
    false
  };
  let res = (|| -> Result<usize> {
    let mut killed = 0;
    match &op {
      Op::Upsert(docs) => {
        for d in docs {
          killed += usize::from(kill(&d.id, &mut g));
          active.insert(&mut g, d)?;
        }
      }
      Op::Delete(ids) => {
        for id in ids {
          killed += usize::from(kill(id, &mut g));
        }
      }
    }
    Ok(killed)
  })();
  // Publish even a partial batch's rows only together with the error
  // path poisoning the collection; on success, everything at once.
  let killed = res?;
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
  Ok(killed)
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
    // A file still in use (Windows refuses to delete mapped files) is
    // left for a later attempt; it is never referenced by a manifest.
    if digits.parse::<u64>().is_ok_and(|id| id >= next_id)
      && std::fs::remove_file(e.path()).is_ok()
    {
      removed = true;
    }
  }
  if removed {
    muninn_store::fs::sync_dir(dir)?;
  }
  Ok(())
}

/// Above this many segments, background merging first merges the
/// smallest neighbours, and gets turns even while writers wait for
/// flushes.
fn max_segments(opts: &Options) -> usize {
  3 * opts.merge_factor + 8
}

/// What the merge policy knows about a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SegSize {
  live: u64,
  total: u64,
  /// It has an HNSW graph.
  graph: bool,
}

/// A fold may rewrite a graph segment up to this many times the size of
/// the documents it adds to it. Rewriting is sequential I/O, cheap next
/// to the graph inserts it saves, but not free.
const FOLD_MAX_REWRITE: u64 = 16;

/// Whether `segs` ends with segments that have no graph (in a collection
/// whose schema asks for one), waiting for [`pick_fold`].
fn has_trailing_flats(segs: &[SegSize]) -> bool {
  segs.last().is_some_and(|s| !s.graph)
}

/// The merge that gives segments flushed without a graph one. Such
/// segments form runs; normally one run at the end of the collection,
/// but a merge by another rule can leave part of a run between graph
/// segments. A run is merged into the graph segment before it when that
/// is cheap (its graph is extended, see `segment::write_merged`),
/// otherwise its oldest segments are merged into a segment of their own,
/// never above `max_merge_docs` documents.
///
/// The run at the end waits until it is worth it: `merge_factor`
/// segments, or at least a memtable's worth and a `merge_factor`th of
/// the graph segment, so that bulk loads insert every vector into a graph
/// once instead of building a graph per flush and rebuilding it per
/// merge. With `quiet` (no writes for a while) it is merged whatever its
/// size, so searches do not keep scanning it exactly. A run left between
/// graph segments no longer grows, so it is merged at once.
fn pick_fold(
  segs: &[SegSize],
  opts: &Options,
  quiet: bool,
) -> Option<std::ops::Range<usize>> {
  let n = segs.len();
  let mut i = 0;
  while i < n {
    if segs[i].graph {
      i += 1;
      continue;
    }
    let end = segs[i..].iter().position(|s| s.graph).map_or(n, |e| i + e);
    if let Some(r) = fold_run(segs, i..end, end == n, opts, quiet) {
      return Some(r);
    }
    i = end;
  }
  None
}

/// [`pick_fold`] for the graphless run `run`; `tail` if it ends the
/// collection.
fn fold_run(
  segs: &[SegSize],
  run: std::ops::Range<usize>,
  tail: bool,
  opts: &Options,
  quiet: bool,
) -> Option<std::ops::Range<usize>> {
  let (start, end) = (run.start, run.end);
  let run_docs: u64 = segs[run.clone()].iter().map(|s| s.live).sum();
  let run_len = end - start;
  let base = start.checked_sub(1).map(|i| segs[i]);
  let f = opts.merge_factor as u64;
  let unit = opts.memtable_max_docs as u64;
  let cap = opts.max_merge_docs as u64;
  let urgent = !tail
    || run_len as u64 >= f
    || run_docs >= unit.max(base.map_or(0, |b| b.live / f));
  if !urgent && !quiet {
    return None;
  }
  // A stranded run is rewritten into its graph segment once; the tail
  // keeps growing, so it is folded only while that stays cheap.
  if let Some(b) = base
    && b.live == b.total
    && b.live + run_docs <= cap
    && (!tail || b.live <= FOLD_MAX_REWRITE * run_docs)
  {
    return Some(start - 1..end);
  }
  // The oldest segments of the run that fit under the cap (at least one:
  // a single segment's graph is no larger than the segment).
  let mut stop = start;
  let mut docs = 0;
  while stop < end && (stop == start || docs + segs[stop].live <= cap) {
    docs += segs[stop].live;
    stop += 1;
  }
  if docs >= opts.hnsw_min_docs as u64 || stop - start >= 2 {
    return Some(start..stop);
  }
  None
}

/// Picks segments to merge in the background: a contiguous range of
/// `segs`. `lazy` says flushes may leave graphs to merges (see
/// [`pick_fold`]); `quiet` that no write arrived for a while.
fn pick_merge(
  segs: &[SegSize],
  opts: &Options,
  lazy: bool,
  quiet: bool,
) -> Option<std::ops::Range<usize>> {
  // 0. Drop segments whose documents are all deleted: nearly free, and
  // under heavy writes the rules below could otherwise keep them around.
  if let Some(i) = segs.iter().position(|s| s.live == 0) {
    let end = segs[i..]
      .iter()
      .position(|s| s.live > 0)
      .map_or(segs.len(), |e| i + e);
    return Some(i..end);
  }
  if lazy && let Some(r) = pick_fold(segs, opts, quiet) {
    return Some(r);
  }
  let segs: Vec<(u64, u64)> = segs.iter().map(|s| (s.live, s.total)).collect();
  let n = segs.len();
  let base = opts.memtable_max_docs.max(1) as u64;
  let f = opts.merge_factor;
  let cap = opts.max_merge_docs as u64;
  // 1. Bound the number of segments: merge the smallest neighbours.
  if n > max_segments(opts) {
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

fn seg_size(s: &Segment, deleted: usize) -> SegSize {
  SegSize {
    live: u64::from(s.len) - deleted as u64,
    total: u64::from(s.len),
    graph: matches!(s.vectors, Some(segment::VecIndex::Hnsw(_))),
  }
}

/// The merge policy's view of a version's segments.
fn seg_sizes(v: &Version) -> Vec<SegSize> {
  v.views[..v.nseg]
    .iter()
    .map(|w| seg_size(w.segment().expect("segment"), w.deleted))
    .collect()
}

#[derive(Clone, Copy)]
enum Job {
  Flush(Frozen),
  /// `quiet`: no write arrived for [`Options::merge_idle_delay`].
  Compact {
    full: bool,
    req: u64,
    quiet: bool,
  },
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
    match std::fs::read_to_string(dir.join(WAL_DAMAGED)) {
      Ok(msg) => {
        return Err(Error::corrupt(format!("{}: {msg}", dir.display())));
      }
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
      Err(e) => return Err(e.into()),
    }
    let (store, m) = ManifestStore::open(dir)?;
    let m = m.ok_or_else(|| {
      Error::corrupt(format!("{} has no manifest", dir.display()))
    })?;
    let schema = Schema::from_stored(&m.schema)?;
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
      compact_failed: 0,
      compact_error: None,
      auto_compact_pending: true,
      fold_waiting: None,
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
    // A torn tail (the normal result of a crash) is cut silently. A cut
    // that set later log files aside, or that lands at or below the
    // checkpoint (records that were synced before the manifest was
    // committed), means damage in the middle of the log: acknowledged
    // writes after it would be lost. The WAL has already repaired its
    // files, so record the refusal in a marker that later opens honour.
    let below_checkpoint = rec.truncated.is_some() && rec.last_lsn < cp;
    if !rec.discarded.is_empty() || below_checkpoint {
      let msg = format!(
        "the write-ahead log is damaged before its end: {:?}; files set \
         aside: {:?}; log ends at LSN {}, checkpoint {cp}. Writes after \
         the damage are not recovered. Copy the files away for analysis; \
         delete {WAL_DAMAGED} to open the collection with the writes that \
         survived",
        rec.truncated, rec.discarded, rec.last_lsn
      );
      muninn_store::fs::write_atomic(&dir.join(WAL_DAMAGED), msg.as_bytes())?;
      return Err(Error::corrupt(format!("{}: {msg}", dir.display())));
    }
    st.applied_lsn = st.applied_lsn.max(wal.last_lsn());
    Ok(Core {
      dir: dir.to_path_buf(),
      name: name.to_string(),
      filter_schema: schema.filter_schema()?,
      schema,
      opts: opts.clone(),
      wal: RwLock::new(Some(wal)),
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
      remover: DirRemover(Mutex::new(None)),
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

  /// Runs `f` on the open WAL; `NotFound` once the collection is dropped.
  fn with_wal<T>(&self, f: impl FnOnce(&Wal) -> Result<T>) -> Result<T> {
    let g = read(&self.wal);
    match g.as_ref() {
      Some(w) => f(w),
      None => Err(Error::NotFound(format!(
        "collection {:?} was dropped",
        self.name
      ))),
    }
  }

  /// Releases the files of a dropped collection, so its directory can be
  /// deleted (on Windows open or mapped files block that): closes the WAL
  /// and replaces the snapshot with an empty one, which unmaps every
  /// segment no running search still holds. Call after the worker
  /// stopped.
  pub(crate) fn release_files(&self) {
    let wal = write(&self.wal).take();
    if let Some(w) = wal {
      let _ = w.close();
    }
    if let Ok(m) = Memtable::new(u64::MAX, &self.schema) {
      let empty = Arc::new(Version {
        views: vec![View::mem(Arc::new(m))],
        nseg: 0,
      });
      let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
      st.version = empty.clone();
      st.frozen.clear();
      *write(&self.current) = empty;
    }
  }

  /// Deletes `path` (the dropped collection's directory) when the last
  /// reference to this collection goes away.
  pub(crate) fn remove_on_release(&self, path: PathBuf) {
    *self.remover.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(path);
  }

  /// Validates, logs and applies one batch. Returns how many live
  /// documents it deleted or replaced.
  fn write(&self, rec: Vec<u8>) -> Result<usize> {
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
    let lsn = self.with_wal(|w| w.append(&rec))?;
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
    let killed = match apply_record(&self.schema, &self.current, &mut st, &rec)
    {
      Ok(k) => k,
      Err(e) => {
        st.poisoned = Some(format!("applying LSN {lsn}: {e}"));
        self.cv.notify_all();
        return Err(e);
      }
    };
    st.applied_lsn = lsn;
    if is_full(&self.opts, &st) {
      freeze(&self.schema, &self.current, &mut st)?;
      self.worker_cv.notify_one();
    }
    self.cv.notify_all();
    Ok(killed)
  }

  pub(crate) fn upsert(&self, docs: &[Document]) -> Result<()> {
    if docs.is_empty() {
      return Ok(());
    }
    self.write(codec::encode_upsert(&self.schema, docs)?)?;
    Ok(())
  }

  pub(crate) fn delete(&self, ids: &[Id]) -> Result<usize> {
    if ids.is_empty() {
      return Ok(0);
    }
    self.write(codec::encode_delete(ids)?)
  }

  /// Waits until `done(state)` or a failure; flush failures count only if
  /// `flush_errors`. Returns the state lock.
  fn wait_for<'a>(
    &'a self,
    mut st: MutexGuard<'a, State>,
    flush_errors: bool,
    done: impl Fn(&State) -> bool,
  ) -> Result<MutexGuard<'a, State>> {
    let gen0 = st.err_gen;
    loop {
      if done(&st) {
        return Ok(st);
      }
      if let Some(m) = &st.poisoned {
        return Err(poisoned_err(m));
      }
      if flush_errors && st.err_gen != gen0 {
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
    drop(self.wait_for(st, true, |s| s.flushed_seq >= target)?);
    Ok(())
  }

  /// Flushes, then merges every segment into one without deleted docs.
  pub(crate) fn compact(&self) -> Result<()> {
    self.flush()?;
    let mut st = self.lock()?;
    st.compact_req += 1;
    let req = st.compact_req;
    self.worker_cv.notify_one();
    let st = self.wait_for(st, false, |s| s.compact_done >= req)?;
    if st.compact_failed >= req {
      let msg = st.compact_error.clone().unwrap_or_default();
      return Err(Error::Io(std::io::Error::other(format!(
        "compaction failed: {msg}"
      ))));
    }
    Ok(())
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
              quiet: false,
            })
          } else if self.opts.auto_compact && st.auto_compact_pending {
            Some(Job::Compact {
              full: false,
              req: 0,
              quiet: false,
            })
          } else {
            None
          };
          // Flushes first, but under sustained writes alternate with
          // compaction so merges (and explicit compact calls) are not
          // starved; writers blocked on the flush backlog win, unless
          // segments pile up: every flush adds one, and each write and
          // flush costs a little per segment, so letting flushes win
          // then would only slow them down further.
          let blocked = st.frozen.len() > self.opts.max_pending_flushes
            && st.version.nseg <= max_segments(&self.opts);
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
          // Segments waiting for a graph: merge them once writes pause
          // (at once with a zero delay; checking again at most every
          // millisecond otherwise, so the worker never spins).
          if let Some(lsn) = st.fold_waiting {
            let delay = self.opts.merge_idle_delay;
            if delay.is_zero() {
              st.fold_waiting = None;
              break Job::Compact {
                full: false,
                req: 0,
                quiet: true,
              };
            }
            let (g, timeout) = match self
              .worker_cv
              .wait_timeout(st, delay.max(MIN_IDLE_CHECK))
            {
              Ok(r) => r,
              Err(_) => return,
            };
            st = g;
            if timeout.timed_out() && st.fold_waiting.is_some() {
              if st.applied_lsn == lsn && st.frozen.is_empty() {
                st.fold_waiting = None;
                break Job::Compact {
                  full: false,
                  req: 0,
                  quiet: true,
                };
              }
              st.fold_waiting = Some(st.applied_lsn);
            }
            continue;
          }
          st = match self.worker_cv.wait(st) {
            Ok(g) => g,
            Err(_) => return,
          };
        }
      };
      flushed_last = matches!(job, Job::Flush(_));
      // A panic in a job is a broken invariant: poison the collection and
      // wake everyone waiting on the worker instead of leaving them
      // waiting for a thread that is gone.
      let res =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
          Job::Flush(f) => self.flush_one(f).map(|_| None),
          Job::Compact { full, req, quiet } => self
            .compact_once(full, quiet)
            .map(|merged| Some((full, req, merged))),
        }));
      let res = match res {
        Ok(r) => r,
        Err(panic) => {
          let msg = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".into());
          self.poison(&format!("background worker panicked: {msg}"));
          return;
        }
      };
      retry_wait = false;
      let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
      if st.poisoned.is_some() {
        self.cv.notify_all();
        return;
      }
      match (res, job) {
        (Ok(Some((full, req, merged))), _) => {
          if full {
            st.compact_done = st.compact_done.max(req);
          } else if !merged {
            st.auto_compact_pending = false;
          }
        }
        (Ok(None), _) => {}
        (Err(e), Job::Compact { full, req, .. }) => {
          // Not retried: an explicit compact() reports the failure to its
          // caller; background merging tries again after the next flush.
          if full {
            st.compact_done = st.compact_done.max(req);
            st.compact_failed = st.compact_failed.max(req);
            st.compact_error = Some(e.to_string());
          } else {
            st.auto_compact_pending = false;
            st.compact_error = Some(e.to_string());
          }
          drop(st);
          self.remove_failed_output();
          self.cv.notify_all();
          continue;
        }
        (Err(e), Job::Flush(_)) => {
          // The frozen memtable stays (its writes are in the log) and the
          // flush is retried after a pause.
          st.bg_error = Some(e.to_string());
          st.err_gen += 1;
          retry_wait = true;
          drop(st);
          self.remove_failed_output();
          self.cv.notify_all();
          continue;
        }
      }
      drop(st);
      self.cv.notify_all();
    }
  }

  /// Refuses further writes, wakes every waiter and stops the worker.
  fn poison(&self, msg: &str) {
    let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
    st.poisoned.get_or_insert_with(|| msg.to_string());
    st.shutdown = true;
    drop(st);
    self.cv.notify_all();
    self.worker_cv.notify_all();
  }

  /// Deletes files a failed job wrote for a segment it never committed
  /// (ids at or above the manifest's `next_segment_id`). Only the worker
  /// creates segments, and it is between jobs, so nothing is in flight.
  fn remove_failed_output(&self) {
    let next = match self.disk.lock() {
      Ok(d) => d.manifest.next_segment_id,
      Err(_) => return,
    };
    // Best effort: the files are harmless, and the next open retries.
    let _ = remove_uncommitted(&self.dir, next);
  }

  /// Writes the oldest frozen memtable as a segment and publishes it.
  fn flush_one(&self, f: Frozen) -> Result<()> {
    self.with_wal(Wal::rotate)?;
    match self.opts.failpoint {
      Some(Failpoint::SlowFlush) => {
        std::thread::sleep(Duration::from_millis(300))
      }
      Some(Failpoint::PanicInFlush) => panic!("failpoint PanicInFlush"),
      _ => {}
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
    // While writes outpace the worker (more memtables wait behind this
    // one, or earlier flushes already left their graphs to a merge), the
    // segment gets no graph: a merge soon inserts its vectors into a
    // larger graph, so building one now would be work thrown away.
    let graph = !self.lazy_graph() || {
      let st = self.lock()?;
      st.frozen.len() <= 1 && !has_trailing_flats(&seg_sizes(&st.version))
    };
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
          graph,
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
    // The captured tombstones may include deletes logged after the
    // checkpoint and not yet synced. Persisting them before their log
    // records are durable would let a power cut keep a delete (for
    // example the old version of a replaced document) while losing the
    // write that caused it. One sync per flush makes them durable first.
    self.with_wal(Wal::sync)?;
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
    self.commit(&mut disk, &mut m)?;
    if self.opts.failpoint == Some(Failpoint::FailAfterCommit) {
      return Err(self.fatal("failpoint FailAfterCommit"));
    }
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
        drop(st);
        return Err(self.fatal("frozen memtables out of order"));
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
      st.checkpoint_lsn = f.checkpoint.max(st.checkpoint_lsn);
      st.auto_compact_pending = true;
      publish(&self.current, &mut st, nv);
      self.note_waiting_flats(&mut st);
      self.cv.notify_all();
    }
    self.seal(&mut disk)?;
    self.tidy(&disk);
    // Only now does `flush()` return: the log is short again.
    self.lock()?.flushed_seq = f.seq;
    self.cv.notify_all();
    Ok(())
  }

  /// Commits the current manifest once more, unchanged. The store keeps
  /// the two newest generations and falls back to the older one if the
  /// newest is unreadable, so until this second commit the generation
  /// before the change stays in charge of its files and of the log from
  /// its checkpoint on. Afterwards both retained generations describe the
  /// same state, so [`Core::tidy`] can release everything the change made
  /// obsolete: merged-away segments right after a compaction, the log
  /// right after a flush. A manifest is a few kilobytes, so this costs two
  /// small fsyncs per flush or merge.
  fn seal(&self, disk: &mut Disk) -> Result<()> {
    let mut m = disk.manifest.clone();
    self.commit(disk, &mut m)?;
    disk.manifest = m;
    Ok(())
  }

  /// Deletes log files below the oldest retained manifest's checkpoint
  /// and segment files no retained manifest names. Cleanup only: the
  /// change is already durable and visible, and a failure leaves extra
  /// files that the next flush or merge removes.
  fn tidy(&self, disk: &Disk) {
    let oldest = disk
      .store
      .retained()
      .iter()
      .map(|m| m.wal_checkpoint_lsn)
      .min()
      .unwrap_or(0);
    if let Err(e) = self.with_wal(|w| w.truncate_before(oldest + 1)) {
      self.warn(&format!("WAL truncation failed: {e}"));
    }
    if let Err(e) = disk.store.gc(true, &[]) {
      self.warn(&format!("removing obsolete files failed: {e}"));
    }
  }

  /// Commits a manifest. Any error is fatal: once the commit has started
  /// the manifest may be durable (the rename can succeed and the
  /// directory sync fail), so retrying the job could rewrite files a
  /// committed manifest names. Reopening recovers from whatever is on
  /// disk.
  fn commit(&self, disk: &mut Disk, m: &mut Manifest) -> Result<()> {
    disk
      .store
      .commit(m)
      .map_err(|e| self.fatal(&format!("manifest commit failed: {e}")))
  }

  /// Poisons the collection and returns the error to report.
  fn fatal(&self, msg: &str) -> Error {
    self.poison(msg);
    poisoned_err(msg)
  }

  /// Records a background problem that needs no action from callers.
  fn warn(&self, msg: &str) {
    let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
    st.bg_error = Some(msg.to_string());
  }

  /// Whether flushes may leave a segment's graph to a later merge: the
  /// schema asks for HNSW and background merging is on.
  fn lazy_graph(&self) -> bool {
    self.opts.auto_compact
      && self
        .schema
        .vector
        .as_ref()
        .is_some_and(|f| matches!(f.index, IndexKind::Hnsw { .. }))
  }

  /// Arms the quiet-period merge of segments waiting for a graph (see
  /// [`pick_fold`]) if there is one to do, else disarms it.
  fn note_waiting_flats(&self, st: &mut State) {
    let sizes = seg_sizes(&st.version);
    if self.lazy_graph()
      && has_trailing_flats(&sizes)
      && pick_fold(&sizes, &self.opts, true).is_some()
    {
      let lsn = st.applied_lsn;
      st.fold_waiting.get_or_insert(lsn);
    } else {
      st.fold_waiting = None;
    }
  }

  /// One merge (all segments if `full`, else what the policy picks).
  /// Returns whether anything was merged.
  fn compact_once(&self, full: bool, quiet: bool) -> Result<bool> {
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
      let sizes: Vec<SegSize> =
        segs.iter().map(|(s, _, d)| seg_size(s, *d)).collect();
      match pick_merge(&sizes, &self.opts, self.lazy_graph(), quiet) {
        Some(r) => r,
        None => {
          let mut st = self.lock()?;
          self.note_waiting_flats(&mut st);
          return Ok(false);
        }
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
    if full && self.opts.failpoint == Some(Failpoint::FailCompaction) {
      return Err(Error::Io(std::io::Error::other("failpoint FailCompaction")));
    }
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
    // Docs deleted by writes not yet durable are about to disappear from
    // disk for good: make those writes durable first.
    self.with_wal(Wal::sync)?;
    self.commit(&mut disk, &mut m)?;
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
        drop(st);
        return Err(self.fatal("segments changed during a merge"));
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
      self.note_waiting_flats(&mut st);
      self.cv.notify_all();
    }
    self.failpoint(Failpoint::AfterCompactionCommit)?;
    self.seal(&mut disk)?;
    self.tidy(&disk);
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
      durable_lsn: self.with_wal(|w| Ok(w.durable_lsn())).unwrap_or(0),
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
    self.h.core.delete(ids)?;
    Ok(())
  }

  /// Like [`Collection::delete`], and returns how many of `ids` named a
  /// document that existed (an id listed twice counts once). The count
  /// is exact even with concurrent writers: it is taken when the batch is
  /// applied, in log order.
  pub fn delete_counted(&self, ids: &[Id]) -> Result<usize> {
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

  /// Graph segments of (live, total) docs.
  fn g(s: &[(u64, u64)]) -> Vec<SegSize> {
    s.iter()
      .map(|&(live, total)| SegSize {
        live,
        total,
        graph: true,
      })
      .collect()
  }

  fn flat(live: u64) -> SegSize {
    SegSize {
      live,
      total: live,
      graph: false,
    }
  }

  fn pick(s: &[(u64, u64)], o: &Options) -> Option<std::ops::Range<usize>> {
    pick_merge(&g(s), o, false, false)
  }

  #[test]
  fn policy_bounds_count_then_tiers_then_deletes() {
    let o = opts();
    assert_eq!(pick(&[(100, 100), (10, 100)], &o), Some(1..2));
    // A tier window wins over a segment with many deletes.
    let s = [(10, 100), (90, 90), (100, 100), (50, 50), (80, 80)];
    assert_eq!(pick(&s, &o), Some(0..4));
    assert_eq!(pick(&[(100, 100); 3], &o), None);
    let s = [(1000, 1000), (90, 90), (100, 100), (50, 50), (80, 80)];
    assert_eq!(pick(&s, &o), Some(1..5));
    // A different tier in the window blocks it.
    let s = [(90, 90), (1000, 1000), (100, 100), (50, 50), (80, 80)];
    assert_eq!(pick(&s, &o), None);
    // Too many segments: smallest neighbouring pair.
    let mut s = vec![(1000u64, 1000u64); 3 * 4 + 9];
    for (i, x) in s.iter_mut().enumerate() {
      if i % 2 == 0 {
        *x = (5000 + i as u64, 5000 + i as u64);
      }
    }
    s[7] = (1, 1);
    let r = pick(&s, &o).unwrap();
    assert!(r.contains(&7) && r.len() == 2);
  }

  #[test]
  fn fold_waits_for_enough_flat_segments_then_extends_the_graph() {
    let o = Options {
      memtable_max_docs: 100,
      merge_factor: 4,
      hnsw_min_docs: 50,
      ..Options::default()
    };
    let fold = |s: &[SegSize], quiet| pick_merge(s, &o, true, quiet);
    // One memtable's worth after a small graph: fold into it.
    let mut s = g(&[(100, 100)]);
    s.push(flat(100));
    assert_eq!(fold(&s, false), Some(0..2));
    // A large graph waits for a quarter of its size or four segments...
    let mut s = g(&[(1000, 1000)]);
    s.extend([flat(100), flat(100)]);
    assert_eq!(fold(&s, false), None);
    s.push(flat(100));
    assert_eq!(fold(&s, false), Some(0..4));
    let mut s = g(&[(1000, 1000)]);
    s.extend([flat(100), flat(100)]);
    // ... or for writes to pause.
    assert_eq!(fold(&s, true), Some(0..3));
    // Without lazy graphs the old policy applies.
    assert_eq!(pick_merge(&s, &o, false, true), None);
    // A graph with deletes is not extended (it must lose them first):
    // the run gets a graph of its own.
    let mut s = g(&[(900, 1000)]);
    s.extend([flat(100); 4]);
    assert_eq!(fold(&s, false), Some(1..5));
    // Nor is one too large to rewrite for so few documents.
    let mut s = g(&[(100_000, 100_000)]);
    s.extend([flat(100); 4]);
    assert_eq!(fold(&s, false), Some(1..5));
    // A single small flat segment with nothing to fold into stays.
    assert_eq!(fold(&[flat(10)], true), None);
    assert_eq!(fold(&[flat(60)], true), Some(0..1));
    assert!(!has_trailing_flats(&g(&[(1, 1)])));
  }

  /// Background merges never produce segments above `max_merge_docs`,
  /// also when a long graphless tail cannot be folded into a graph.
  #[test]
  fn fold_respects_max_merge_docs() {
    let o = Options {
      memtable_max_docs: 1000,
      merge_factor: 4,
      hnsw_min_docs: 10,
      max_merge_docs: 2500,
      ..Options::default()
    };
    let s = vec![flat(1000); 4];
    let r = pick_merge(&s, &o, true, false);
    let docs: u64 = r.clone().map_or(0, |r| s[r].iter().map(|x| x.live).sum());
    assert!(docs <= o.max_merge_docs as u64, "{r:?} merges {docs} docs");
    // The oldest segments that fit go first.
    assert_eq!(r, Some(0..2));
  }

  /// A graphless run left between graph segments (for example by the
  /// segment-count rule merging the end of a run) is still folded.
  #[test]
  fn stranded_graphless_segments_are_folded() {
    let o = Options {
      memtable_max_docs: 100,
      merge_factor: 4,
      hnsw_min_docs: 50,
      ..Options::default()
    };
    // G, F, G': F is folded into G although G is far larger than F.
    let mut s = g(&[(10_000, 10_000)]);
    s.push(flat(30));
    s.extend(g(&[(200, 200)]));
    assert_eq!(pick_merge(&s, &o, true, false), Some(0..2));
    // G with deletes cannot be extended: a large enough F gets a graph
    // of its own.
    let mut s = g(&[(9_000, 10_000)]);
    s.push(flat(60));
    s.extend(g(&[(200, 200)]));
    assert_eq!(pick_merge(&s, &o, true, false), Some(1..2));
  }

  #[test]
  fn fully_deleted_segments_go_first() {
    let o = opts();
    let s = [(100, 100), (0, 50), (0, 80), (10, 90), (0, 5)];
    assert_eq!(pick(&s, &o), Some(1..3));
    let s = [(100, 100), (100, 100), (0, 5)];
    assert_eq!(pick(&s, &o), Some(2..3));
  }
}
