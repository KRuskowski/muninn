//! Durable storage.
//!
//! Scope of this crate (see `docs/ARCHITECTURE.md`): the write-ahead log,
//! memory-mapped immutable files exposed as [`muninn_core::Region`]s,
//! the user-id to [`muninn_core::DocId`] map, document payload storage,
//! tombstones, and the per-collection manifest that names live segments
//! and is replaced atomically.
//!
//! A collection directory managed with this crate looks like:
//!
//! ```text
//! <dir>/LOCK                       DirLock
//! <dir>/MANIFEST-<generation>      ManifestStore (newest valid wins)
//! <dir>/wal/<first_lsn>.wal        Wal segments
//! <dir>/seg-<id>.<ext>             segment files named by the manifest
//! <dir>/seg-<id>.<version>.tomb    tombstones
//! ```
//!
//! Portable to Unix and Windows. Platform differences: directory fsync
//! and `madvise` hints exist on Unix only and are no-ops elsewhere (see
//! [`fs`] and [`advise`]); on Windows a memory-mapped file cannot be
//! deleted until it is unmapped, so [`ManifestStore::gc`] skips files that
//! are still in use there.

pub mod fs;
mod hash;
mod idmap;
mod lock;
mod manifest;
mod mmap;
mod payload;
mod tombstone;
mod view;
pub mod wal;

pub use idmap::{
  ExternalId, ExternalIdRef, FrozenIdMap, IDMAP_KIND, IDMAP_VERSION, IdMap,
};
pub use lock::{DirLock, LOCK_FILE};
pub use manifest::{
  KEEP_MANIFESTS, MANIFEST_KIND, MANIFEST_PREFIX, MANIFEST_VERSION, Manifest,
  ManifestStore, SegmentEntry, manifest_file_name, segment_file_name,
  segment_prefix,
};
pub use mmap::{Advice, advise, map_file, map_file_with};
pub use payload::{
  PAYLOAD_KIND, PAYLOAD_VERSION, PayloadBuilder, PayloadStore, write_payloads,
};
pub use tombstone::{
  TOMBSTONE_KIND, TOMBSTONE_VERSION, Tombstones, latest_tombstones,
  read_tombstones, tombstone_file_name, write_tombstones,
};
pub use wal::{Durability, Lsn, Recovery, Wal, WalOptions, WalStats};
