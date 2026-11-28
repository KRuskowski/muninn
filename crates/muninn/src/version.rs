//! Snapshots of a collection.
//!
//! A [`Version`] lists every source of a collection in write order:
//! segments (oldest first), then frozen memtables, then the active
//! memtable, which is always last. Each [`View`] pins how many docs of
//! the source are visible and which are deleted. Versions are immutable
//! and shared through `Arc`; a write or a flush publishes a new one.
//! Tombstones are copy-on-write: a batch that deletes docs of a source
//! clones that source's bitset once, so a search holding an older version
//! keeps seeing exactly the deletes of its snapshot.

use std::sync::Arc;

use muninn_core::{BitSet, DocId};

use crate::memtable::Memtable;
use crate::segment::Segment;

/// One source of documents.
#[derive(Clone)]
pub(crate) enum Source {
  Mem(Arc<Memtable>),
  Seg(Arc<Segment>),
}

/// A source as seen by one version.
#[derive(Clone)]
pub(crate) struct View {
  pub src: Source,
  /// Visible docs: `0..len`.
  pub len: u32,
  /// Deleted docs; ids beyond its capacity are not deleted.
  pub tomb: Arc<BitSet>,
  /// `tomb.count()`, cached.
  pub deleted: usize,
}

impl View {
  pub(crate) fn mem(m: Arc<Memtable>) -> View {
    View {
      src: Source::Mem(m),
      len: 0,
      tomb: Arc::new(BitSet::new(0)),
      deleted: 0,
    }
  }

  pub(crate) fn is_deleted(&self, d: DocId) -> bool {
    self.tomb.contains(d)
  }

  pub(crate) fn memtable(&self) -> Option<&Arc<Memtable>> {
    match &self.src {
      Source::Mem(m) => Some(m),
      Source::Seg(_) => None,
    }
  }

  pub(crate) fn segment(&self) -> Option<&Arc<Segment>> {
    match &self.src {
      Source::Seg(s) => Some(s),
      Source::Mem(_) => None,
    }
  }
}

/// An immutable snapshot; see the module docs.
#[derive(Clone)]
pub(crate) struct Version {
  pub views: Vec<View>,
  /// `views[..nseg]` are segments.
  pub nseg: usize,
}

impl Version {
  pub(crate) fn active(&self) -> &Arc<Memtable> {
    self
      .views
      .last()
      .and_then(View::memtable)
      .expect("a version always ends with the active memtable")
  }

  pub(crate) fn active_view(&self) -> &View {
    self.views.last().expect("a version is never empty")
  }
}
