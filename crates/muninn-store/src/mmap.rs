//! Memory-mapped files exposed as [`Region`]s.
//!
//! Immutable muninn files (segments, id maps, payloads) are opened by
//! mapping them read-only and handing the map to [`Region::new`], which
//! keeps it alive for as long as any clone or slice of the region exists.
//! Opening a file therefore costs one `mmap` call; pages are read lazily.
//!
//! The contract for every caller of [`map_file`]: bytes already in the
//! map must not change and the file must not be truncated while any map
//! of it is alive (truncation turns accesses into `SIGBUS`). Appending is
//! fine, since a map covers only the length the file had when mapped.
//! In muninn:
//!
//! - Segment, id-map, payload, tombstone and manifest files are written
//!   once under a temporary name, renamed into place (see [`crate::fs`])
//!   and afterwards only deleted.
//! - WAL segments are mapped by recovery and by [`crate::Wal::replay`].
//!   The active segment is appended to while replay may have it mapped,
//!   which never touches mapped bytes. Recovery truncates or rewrites a
//!   segment only after dropping its own map, before the `Wal` exists, so
//!   no replay can hold one.
//!
//! On Unix, deleting a mapped file leaves existing maps valid. Windows
//! may refuse to delete or replace a mapped file (older systems, or
//! without POSIX delete semantics); [`crate::ManifestStore::gc`] then
//! leaves it for a later run.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use muninn_core::{AlignedBuf, Error, Region, Result};

/// Access-pattern hints for mapped regions, passed to `madvise` on Unix
/// and ignored elsewhere.
///
/// Only non-destructive hints are offered, so advising any region (even
/// one backed by heap memory rather than a map) never changes its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
  /// Default kernel read-ahead.
  Normal,
  /// Random access (graph traversal, id lookups): disables read-ahead so
  /// each fault reads one page instead of a window around it.
  Random,
  /// Sequential scan: aggressive read-ahead, pages freed sooner.
  Sequential,
  /// Will be needed soon: starts asynchronous read-in of the range.
  WillNeed,
}

#[cfg(unix)]
impl Advice {
  fn to_libc(self) -> libc::c_int {
    match self {
      Self::Normal => libc::MADV_NORMAL,
      Self::Random => libc::MADV_RANDOM,
      Self::Sequential => libc::MADV_SEQUENTIAL,
      Self::WillNeed => libc::MADV_WILLNEED,
    }
  }
}

/// Owner type that lets a [`Mmap`] back a [`Region`].
struct MapOwner(Mmap);

impl AsRef<[u8]> for MapOwner {
  fn as_ref(&self) -> &[u8] {
    &self.0
  }
}

/// Maps `path` read-only and returns the whole file as a [`Region`].
///
/// A zero-length file yields an empty region (zero-length maps are not
/// allowed by the OS). The map is released when the last clone of the
/// region, or of any slice of it, is dropped.
pub fn map_file(path: &Path) -> Result<Region> {
  let file = File::open(path)?;
  let len = file.metadata()?.len();
  if len == 0 {
    return Ok(Region::new(AlignedBuf::new()));
  }
  if usize::try_from(len).is_err() {
    return Err(Error::invalid(format!(
      "{} is too large to map ({len} bytes)",
      path.display()
    )));
  }
  // SAFETY: the map is read-only, and per the module contract no muninn
  // code changes mapped bytes or truncates a file while it is mapped (the
  // active WAL segment only grows, beyond the mapped length). The kernel
  // keeps the pages valid even if the file is unlinked.
  let map = unsafe { Mmap::map(&file)? };
  Ok(Region::new(MapOwner(map)))
}

/// [`map_file`], then applies `advice` to the whole map.
pub fn map_file_with(path: &Path, advice: Advice) -> Result<Region> {
  let region = map_file(path)?;
  advise(&region, advice)?;
  Ok(region)
}

/// Gives the kernel an access-pattern hint for the pages under `region`.
///
/// Works on any sub-region (for example one container section). The range
/// is widened to whole pages, which is harmless because every hint offered
/// is non-destructive. Empty regions, and every region on non-Unix
/// platforms, are a no-op.
#[cfg(not(unix))]
pub fn advise(region: &Region, advice: Advice) -> Result<()> {
  let _ = (region, advice);
  Ok(())
}

/// Gives the kernel an access-pattern hint for the pages under `region`.
///
/// Works on any sub-region (for example one container section). The range
/// is widened to whole pages, which is harmless because every hint offered
/// is non-destructive. Empty regions, and every region on non-Unix
/// platforms, are a no-op.
#[cfg(unix)]
pub fn advise(region: &Region, advice: Advice) -> Result<()> {
  let bytes = region.bytes();
  if bytes.is_empty() {
    return Ok(());
  }
  let page = page_size();
  let start = bytes.as_ptr() as usize;
  let aligned = start & !(page - 1);
  let len = start + bytes.len() - aligned;
  // SAFETY: [aligned, aligned + len) covers exactly the pages that hold
  // `bytes`, all of which are mapped because `bytes` is a live slice. The
  // hints passed never discard or alter page contents.
  let rc = unsafe {
    libc::madvise(aligned as *mut libc::c_void, len, advice.to_libc())
  };
  if rc != 0 {
    return Err(std::io::Error::last_os_error().into());
  }
  Ok(())
}

#[cfg(unix)]
fn page_size() -> usize {
  // SAFETY: sysconf has no preconditions and only reads system state.
  let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
  if p > 0 { p as usize } else { 4096 }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn advise_accepts_heap_regions_and_subranges() {
    let mut b = AlignedBuf::new();
    b.extend_from_slice(&vec![7u8; 3 * 4096 + 5]);
    let r = Region::new(b);
    for a in [
      Advice::Normal,
      Advice::Random,
      Advice::Sequential,
      Advice::WillNeed,
    ] {
      advise(&r, a).unwrap();
      advise(&r.slice(100..5000).unwrap(), a).unwrap();
    }
    advise(&r.slice(3..3).unwrap(), Advice::Random).unwrap();
    assert!(r.bytes().iter().all(|&x| x == 7));
  }
}
