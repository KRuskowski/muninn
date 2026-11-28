//! Database options.

use std::time::Duration;

pub use muninn_store::Durability;

/// Settings for a [`crate::Database`] and every collection in it.
#[derive(Clone, Debug)]
pub struct Options {
  /// When a write is acknowledged; see [`Durability`]. Default:
  /// [`Durability::SyncEachWrite`] (on stable storage before `upsert`
  /// returns).
  pub durability: Durability,
  /// The in-memory part of a collection is written to a segment once it
  /// holds this many documents. Default 50 000.
  pub memtable_max_docs: usize,
  /// ... or this many bytes (approximate). Default 256 MiB.
  pub memtable_max_bytes: usize,
  /// Writers wait while this many full memtables are still being
  /// written to segments, so memory stays bounded when writes outpace
  /// the disk. Default 2.
  pub max_pending_flushes: usize,
  /// Segments with fewer documents than this get an exact (flat) vector
  /// index even when the schema asks for HNSW: scanning is faster than a
  /// graph at that size. Default 10 000.
  pub hnsw_min_docs: usize,
  /// Merge segments in the background. Default true.
  pub auto_compact: bool,
  /// Background merging merges this many neighbouring segments of
  /// similar size at a time. Default 4.
  pub merge_factor: usize,
  /// A segment in which at least this fraction of documents is deleted
  /// is rewritten without them. Default 0.3.
  pub compact_deleted_ratio: f64,
  /// Background merges never produce segments above this many documents
  /// (an explicit [`crate::Collection::compact`] ignores the limit).
  /// Default 4 000 000.
  pub max_merge_docs: usize,
  /// Size of one write-ahead-log file before a new one is started.
  /// Default 64 MiB.
  pub wal_segment_size: u64,
  /// Fault injection for crash tests. Not part of the stable API.
  #[doc(hidden)]
  pub failpoint: Option<Failpoint>,
}

impl Default for Options {
  fn default() -> Self {
    Self {
      durability: Durability::SyncEachWrite,
      memtable_max_docs: 50_000,
      memtable_max_bytes: 256 << 20,
      max_pending_flushes: 2,
      hnsw_min_docs: 10_000,
      auto_compact: true,
      merge_factor: 4,
      compact_deleted_ratio: 0.3,
      max_merge_docs: 4_000_000,
      wal_segment_size: 64 << 20,
      failpoint: None,
    }
  }
}

impl Options {
  /// Sets the durability mode.
  pub fn durability(mut self, d: Durability) -> Self {
    self.durability = d;
    self
  }

  /// Writes are on disk at most `max_delay` after they are acknowledged
  /// (see [`Durability::SyncBatch`]).
  pub fn sync_every(self, max_delay: Duration) -> Self {
    self.durability(Durability::SyncBatch { max_delay })
  }

  /// Sets the memtable document limit.
  pub fn memtable_max_docs(mut self, n: usize) -> Self {
    self.memtable_max_docs = n;
    self
  }

  /// Turns background merging on or off.
  pub fn auto_compact(mut self, on: bool) -> Self {
    self.auto_compact = on;
    self
  }

  pub(crate) fn validate(&self) -> muninn_core::Result<()> {
    let bad = |m: &str| Err(muninn_core::Error::invalid(m.to_string()));
    if self.memtable_max_docs == 0 || self.memtable_max_bytes == 0 {
      return bad("memtable limits must be positive");
    }
    if self.max_pending_flushes == 0 {
      return bad("max_pending_flushes must be positive");
    }
    if self.merge_factor < 2 {
      return bad("merge_factor must be at least 2");
    }
    if !(self.compact_deleted_ratio > 0.0 && self.compact_deleted_ratio <= 1.0)
    {
      return bad("compact_deleted_ratio must be in (0, 1]");
    }
    Ok(())
  }
}

/// Where a crash test makes the background worker stop dead, as if the
/// process had been killed there. The collection then refuses writes;
/// drop the database and reopen it to recover.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failpoint {
  /// After a flush wrote its segment files, before the manifest commit.
  AfterSegmentWrite,
  /// After a flush committed the manifest, before the in-memory switch
  /// and the log truncation.
  AfterManifestCommit,
  /// After a compaction wrote its segment, before the manifest commit.
  AfterCompactionWrite,
  /// Not a crash: the worker pauses 20 ms inside the windows where a
  /// flush or merge has captured state that concurrent writes may change
  /// before it publishes, so tests can hit those races reliably.
  SlowPublish,
  /// Not a crash: a flush pauses 300 ms after rotating the log, before it
  /// captures tombstones, so a test can write during the flush.
  SlowFlush,
  /// The worker panics at the start of a flush.
  PanicInFlush,
  /// A full compaction fails (an I/O-like error) after writing its
  /// segment files, before the manifest commit.
  FailCompaction,
  /// A flush reports an error right after its manifest commit, as when
  /// the directory fsync after the rename fails.
  FailAfterCommit,
}
