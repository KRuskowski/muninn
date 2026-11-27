//! Persisted deletes of one segment.
//!
//! A segment's deleted docs are a [`BitSet`]. Each time the engine
//! persists it, it writes a new file with a higher version
//! ([`tombstone_file_name`]) atomically; the manifest names the current
//! one, and [`latest_tombstones`] finds the newest valid version when the
//! manifest is not authoritative (for example during repair). Older
//! versions become garbage for [`crate::ManifestStore::gc`].
//!
//! # File format (`tomb`, version 1)
//!
//! | section | contents                                          |
//! |---------|---------------------------------------------------|
//! | `meta`  | `[u64; 4]`: segment id, version, capacity, count  |
//! | `words` | the bitset's `u64` words                          |
//!
//! The whole file is checksummed and verified on every read; files are
//! small (one bit per doc).

use std::path::Path;

use muninn_core::container::{Container, tag};
use muninn_core::{BitSet, Error, Result};

use crate::mmap::map_file;

/// Container kind of a tombstone file.
pub const TOMBSTONE_KIND: [u8; 8] = tag("tomb");
/// Current tombstone format version.
pub const TOMBSTONE_VERSION: u32 = 1;

/// A tombstone file's contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tombstones {
  /// The segment the deletes belong to.
  pub segment_id: u64,
  /// Monotonic per segment; a higher version supersedes a lower one.
  pub version: u64,
  /// Deleted docs.
  pub deleted: BitSet,
}

/// File name of version `version` of segment `segment_id`'s tombstones:
/// `seg-<id>.<version>.tomb`, both zero padded so names sort numerically.
pub fn tombstone_file_name(segment_id: u64, version: u64) -> String {
  format!(
    "{}.{version:010}.tomb",
    crate::manifest::segment_prefix(segment_id)
  )
}

fn parse_name(name: &str, segment_id: u64) -> Option<u64> {
  let prefix = format!("{}.", crate::manifest::segment_prefix(segment_id));
  name
    .strip_prefix(&prefix)?
    .strip_suffix(".tomb")
    .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))?
    .parse()
    .ok()
}

/// Atomically writes `deleted` as version `version` of segment
/// `segment_id`'s tombstones in `dir`. Returns the file name.
pub fn write_tombstones(
  dir: &Path,
  segment_id: u64,
  version: u64,
  deleted: &BitSet,
) -> Result<String> {
  let name = tombstone_file_name(segment_id, version);
  let meta: [u64; 4] = [
    segment_id,
    version,
    deleted.capacity() as u64,
    deleted.count() as u64,
  ];
  crate::fs::write_container_file(
    &dir.join(&name),
    TOMBSTONE_KIND,
    TOMBSTONE_VERSION,
    |w| {
      w.pod_section(tag("meta"), &meta)?;
      w.pod_section(tag("words"), deleted.words())
    },
  )?;
  Ok(name)
}

/// Reads and fully verifies a tombstone file.
pub fn read_tombstones(path: &Path) -> Result<Tombstones> {
  let c =
    Container::open_kind(map_file(path)?, TOMBSTONE_KIND, TOMBSTONE_VERSION)?;
  c.verify_checksum()?;
  let meta_r = c.section(tag("meta"))?;
  let meta = meta_r.cast::<u64>()?;
  if meta.len() != 4 {
    return Err(Error::corrupt("tombstone meta has the wrong size"));
  }
  let cap = usize::try_from(meta[2])
    .ok()
    .filter(|&c| c <= u32::MAX as usize)
    .ok_or_else(|| Error::corrupt("tombstone capacity out of range"))?;
  let words_r = c.section(tag("words"))?;
  let words = words_r.cast::<u64>()?;
  if words.len() != cap.div_ceil(64) {
    return Err(Error::corrupt("tombstone bitset has the wrong size"));
  }
  let deleted = BitSet::from_words(words.to_vec(), cap);
  if deleted.count() as u64 != meta[3] {
    return Err(Error::corrupt("tombstone count does not match its bits"));
  }
  Ok(Tombstones {
    segment_id: meta[0],
    version: meta[1],
    deleted,
  })
}

/// The newest valid tombstone file of `segment_id` in `dir`, with its file
/// name. Unreadable or corrupt versions are skipped in favour of older
/// ones; a file whose contents disagree with its name counts as corrupt.
pub fn latest_tombstones(
  dir: &Path,
  segment_id: u64,
) -> Result<Option<(String, Tombstones)>> {
  let mut found: Vec<(u64, String)> = Vec::new();
  for e in std::fs::read_dir(dir)? {
    let name = e?.file_name();
    if let Some(name) = name.to_str()
      && let Some(v) = parse_name(name, segment_id)
    {
      found.push((v, name.to_string()));
    }
  }
  found.sort_unstable_by(|a, b| b.cmp(a));
  for (v, name) in found {
    match read_tombstones(&dir.join(&name)) {
      Ok(t) if t.version == v && t.segment_id == segment_id => {
        return Ok(Some((name, t)));
      }
      _ => continue,
    }
  }
  Ok(None)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn names_sort_and_parse() {
    let a = tombstone_file_name(3, 9);
    let b = tombstone_file_name(3, 10);
    assert!(a < b);
    assert_eq!(parse_name(&a, 3), Some(9));
    assert_eq!(parse_name(&a, 4), None);
    assert_eq!(parse_name("seg-0000000003..tomb", 3), None);
  }
}
