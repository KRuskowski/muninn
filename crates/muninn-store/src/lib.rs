//! Durable storage.
//!
//! Scope of this crate (see `docs/ARCHITECTURE.md`): the write-ahead log,
//! memory-mapped immutable files exposed as [`muninn_core::Region`]s,
//! the user-id to [`muninn_core::DocId`] map, document payload storage,
//! tombstones, and the per-collection manifest that names live segments
//! and is replaced atomically.
