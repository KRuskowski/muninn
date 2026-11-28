//! The per-collection manifest, and garbage collection of unreferenced
//! files.
//!
//! The manifest is the single source of truth for which segment files make
//! up a collection. It is never modified in place: each change writes a
//! new generation `MANIFEST-<generation>` atomically (temporary file,
//! fsync, rename, fsync directory). Loading picks the newest generation
//! that verifies, so a crash at any point of a commit leaves the previous
//! manifest in charge. The newest [`KEEP_MANIFESTS`] valid generations are
//! retained as fallbacks, and files they reference are never collected.
//!
//! # File format (`manifest`, version 1)
//!
//! A container of kind `manifest` with one section, `json`: the
//! [`Manifest`] as pretty-printed JSON. The container's CRC-32C is verified
//! on every load, so a torn or bit-flipped manifest is never trusted.
//!
//! # File naming
//!
//! Files belonging to segment `id` are named `seg-<id>.*` (see
//! [`segment_file_name`] and [`crate::tombstone_file_name`]). Garbage
//! collection only ever removes `seg-*` files no retained manifest names,
//! superseded `MANIFEST-*` files and `*.tmp` files; anything else in the
//! directory (the lock file, the WAL directory, user files) is left alone.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{Error, Region, Result};
use serde::{Deserialize, Serialize};

use crate::fs::{TEMP_SUFFIX, sync_dir, write_atomic};
use crate::mmap::map_file;

/// Container kind of a manifest file.
pub const MANIFEST_KIND: [u8; 8] = tag("manifest");
/// Current manifest format version.
pub const MANIFEST_VERSION: u32 = 1;
/// File-name prefix of manifest files.
pub const MANIFEST_PREFIX: &str = "MANIFEST-";
/// Number of valid manifest generations kept on disk.
pub const KEEP_MANIFESTS: usize = 2;

const SEGMENT_PREFIX: &str = "seg-";

/// `seg-<id>`, the prefix of every file of segment `id`.
pub fn segment_prefix(id: u64) -> String {
  format!("{SEGMENT_PREFIX}{id:010}")
}

/// Conventional name of a segment file: `seg-<id>.<ext>`.
pub fn segment_file_name(id: u64, ext: &str) -> String {
  format!("{}.{ext}", segment_prefix(id))
}

/// Name of the manifest file of `generation`.
pub fn manifest_file_name(generation: u64) -> String {
  format!("{MANIFEST_PREFIX}{generation:020}")
}

fn parse_manifest_name(name: &str) -> Option<u64> {
  name
    .strip_prefix(MANIFEST_PREFIX)
    .filter(|g| !g.is_empty() && g.bytes().all(|b| b.is_ascii_digit()))?
    .parse()
    .ok()
}

/// One live segment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentEntry {
  /// Unique, never reused segment id.
  pub id: u64,
  /// Number of docs (live or deleted) in the segment.
  pub doc_count: u64,
  /// The segment's files by role (e.g. `"idmap"`, `"hnsw"`), as plain file
  /// names relative to the collection directory.
  pub files: BTreeMap<String, String>,
  /// Current tombstone file, if any docs were deleted.
  #[serde(default)]
  pub tombstones: Option<String>,
  /// Engine-defined per-segment metadata, stored verbatim.
  #[serde(default)]
  pub meta: serde_json::Value,
}

/// A collection's manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
  /// Set by [`ManifestStore::commit`]; strictly increasing.
  pub generation: u64,
  /// The collection schema, opaque to the store.
  pub schema: serde_json::Value,
  /// Live segments, in the engine's order.
  pub segments: Vec<SegmentEntry>,
  /// Id for the next segment created.
  pub next_segment_id: u64,
  /// Every WAL record at or below this LSN is reflected in `segments`;
  /// replay starts after it.
  pub wal_checkpoint_lsn: u64,
}

fn check_file_name(name: &str) -> Result<()> {
  let bad = name.is_empty()
    || name == "."
    || name == ".."
    || name.contains(['/', '\\', '\0']);
  if bad {
    return Err(Error::invalid(format!(
      "manifest file name {name:?} is not a plain file name"
    )));
  }
  Ok(())
}

impl Manifest {
  /// An empty manifest for a new collection.
  pub fn new(schema: serde_json::Value) -> Self {
    Self {
      generation: 0,
      schema,
      segments: Vec::new(),
      next_segment_id: 1,
      wal_checkpoint_lsn: 0,
    }
  }

  /// Every file name the manifest references.
  pub fn referenced_files(&self) -> impl Iterator<Item = &str> + '_ {
    self.segments.iter().flat_map(|s| {
      s.files
        .values()
        .map(String::as_str)
        .chain(s.tombstones.as_deref())
    })
  }

  /// Checks invariants: unique segment ids below `next_segment_id`, and
  /// plain file names.
  pub fn validate(&self) -> Result<()> {
    let mut ids = HashSet::new();
    for s in &self.segments {
      if !ids.insert(s.id) {
        return Err(Error::invalid(format!("duplicate segment id {}", s.id)));
      }
      if s.id >= self.next_segment_id {
        return Err(Error::invalid(format!(
          "segment id {} is not below next_segment_id {}",
          s.id, self.next_segment_id
        )));
      }
    }
    self.referenced_files().try_for_each(check_file_name)
  }

  /// Serialises into the on-disk container format.
  pub fn encode(&self) -> Result<Vec<u8>> {
    let json = serde_json::to_vec_pretty(self)
      .map_err(|e| Error::invalid(format!("manifest JSON: {e}")))?;
    let mut w =
      ContainerWriter::new(Vec::new(), MANIFEST_KIND, MANIFEST_VERSION)?;
    w.section(tag("json"), &json)?;
    w.finish()
  }

  /// Parses and fully verifies the on-disk container format.
  pub fn decode(region: Region) -> Result<Self> {
    let c = Container::open_kind(region, MANIFEST_KIND, MANIFEST_VERSION)?;
    c.verify_checksum()?;
    let json = c.section(tag("json"))?;
    let m: Manifest = serde_json::from_slice(json.bytes())
      .map_err(|e| Error::corrupt(format!("manifest JSON: {e}")))?;
    m.validate()
      .map_err(|e| Error::corrupt(format!("manifest: {e}")))?;
    Ok(m)
  }
}

/// The segment id in a `seg-<id>...` file name (also temporary ones).
fn segment_id_of(name: &str) -> Option<u64> {
  let rest = name.strip_prefix(SEGMENT_PREFIX)?;
  let end = rest.find('.').unwrap_or(rest.len());
  let digits = &rest[..end];
  if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
    return None;
  }
  digits.parse().ok()
}

/// The process id in a temporary file name `<target>.<pid>.<n>.tmp`.
fn temp_pid_of(name: &str) -> Option<u32> {
  let mut parts = name.strip_suffix(TEMP_SUFFIX)?.rsplit('.');
  let _n = parts.next()?;
  parts.next()?.parse().ok()
}

/// True for errors meaning "another handle or map still uses the file",
/// which only Windows reports on delete.
fn in_use(e: &std::io::Error) -> bool {
  cfg!(windows) && e.kind() == std::io::ErrorKind::PermissionDenied
}

/// Reads one manifest file and checks it is the generation its name says.
fn read_manifest(path: &Path, generation: u64) -> Result<Manifest> {
  let m = Manifest::decode(map_file(path)?)?;
  if m.generation != generation {
    return Err(Error::corrupt(format!(
      "{} holds generation {}",
      path.display(),
      m.generation
    )));
  }
  Ok(m)
}

/// Reads and commits manifests of one collection directory.
///
/// Only one `ManifestStore` may write a directory at a time; hold a
/// [`crate::DirLock`] on it.
#[derive(Debug)]
pub struct ManifestStore {
  dir: PathBuf,
  /// Highest generation seen on disk, valid or not; commits go above it.
  max_generation: u64,
  /// Newest valid manifests, oldest first, at most [`KEEP_MANIFESTS`].
  retained: Vec<Manifest>,
}

impl ManifestStore {
  /// Scans `dir` and loads the newest valid manifest, if any.
  ///
  /// Corrupt or torn manifests are skipped in favour of older ones, and
  /// temporary files left by interrupted commits are ignored (remove them
  /// with [`ManifestStore::gc`]). Returns `None` only for a directory
  /// without manifest files.
  ///
  /// Errors: an I/O error reading a manifest is returned as is (it may be
  /// transient, and skipping the newest manifest because of it would
  /// silently roll the collection back); manifest files that are all
  /// corrupt are an [`Error::Corrupt`], never an empty collection.
  pub fn open(dir: &Path) -> Result<(ManifestStore, Option<Manifest>)> {
    let mut found: Vec<u64> = Vec::new();
    for e in std::fs::read_dir(dir)? {
      let e = e?;
      if let Some(g) = e.file_name().to_str().and_then(parse_manifest_name) {
        found.push(g);
      }
    }
    found.sort_unstable_by(|a, b| b.cmp(a));
    let has_any = !found.is_empty();
    let max_generation = found.first().copied().unwrap_or(0);
    let mut retained = Vec::new();
    for g in found {
      if retained.len() == KEEP_MANIFESTS {
        break;
      }
      match read_manifest(&dir.join(manifest_file_name(g)), g) {
        Ok(m) => retained.push(m),
        Err(Error::Corrupt(_)) => continue,
        Err(e) => return Err(e),
      }
    }
    if retained.is_empty() && has_any {
      return Err(Error::corrupt(format!(
        "{} holds manifest files but none is valid; refusing to open the \
         collection as empty",
        dir.display()
      )));
    }
    retained.reverse();
    let current = retained.last().cloned();
    let store = ManifestStore {
      dir: dir.to_path_buf(),
      max_generation,
      retained,
    };
    Ok((store, current))
  }

  /// The collection directory.
  pub fn dir(&self) -> &Path {
    &self.dir
  }

  /// The newest committed manifest, if any.
  pub fn current(&self) -> Option<&Manifest> {
    self.retained.last()
  }

  /// The retained generations, oldest first (at most [`KEEP_MANIFESTS`]).
  /// [`ManifestStore::open`] falls back to an older one if the newest is
  /// unreadable, so its files and the log from its checkpoint on must be
  /// kept.
  pub fn retained(&self) -> &[Manifest] {
    &self.retained
  }

  /// Atomically publishes `manifest` as the next generation and sets its
  /// `generation` field. When this returns, the new manifest is durable.
  /// Manifest files beyond the [`KEEP_MANIFESTS`] newest are then deleted;
  /// segment files are not touched (see [`ManifestStore::gc`]).
  pub fn commit(&mut self, manifest: &mut Manifest) -> Result<()> {
    manifest.validate()?;
    let generation = self.max_generation.checked_add(1).ok_or_else(|| {
      Error::corrupt(format!(
        "{} has a manifest with the largest possible generation; remove \
         it to commit",
        self.dir.display()
      ))
    })?;
    manifest.generation = generation;
    let bytes = manifest.encode()?;
    write_atomic(&self.dir.join(manifest_file_name(generation)), &bytes)?;
    self.max_generation = generation;
    self.retained.push(manifest.clone());
    while self.retained.len() > KEEP_MANIFESTS {
      let old = self.retained.remove(0);
      let path = self.dir.join(manifest_file_name(old.generation));
      match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
      }
    }
    Ok(())
  }

  /// Finds files in the directory that no retained manifest needs:
  /// `seg-*` files not referenced, manifests other than the retained
  /// generations, and `*.tmp` files. With `delete`, removes them and
  /// fsyncs the directory. Returns the paths found (with `delete`, the
  /// paths removed).
  ///
  /// Never collected, so gc may run while flushes and compactions write
  /// new segments:
  ///
  /// - files of segments with id `>= next_segment_id` of the current
  ///   manifest: the engine allocates new segment ids from the manifest, so
  ///   these are output not yet committed;
  /// - files of the segment ids in `pending`, for work that reuses an
  ///   existing id (for example a new tombstone version being written);
  /// - temporary files created by this process, which may still be in
  ///   use. Other temporary files are left over from a process that died
  ///   (the directory lock guarantees there is no other live writer);
  /// - every `seg-*` file when no manifest was ever committed.
  ///
  /// Refuses (with an error) when manifest files exist that this store
  /// has not loaded: everything would look unreferenced.
  ///
  /// A file the OS refuses to delete because it is in use (Windows without
  /// POSIX delete semantics, for a file still mapped by a reader) is left
  /// in place, omitted from the result, and collected by a later call.
  pub fn gc(&self, delete: bool, pending: &[u64]) -> Result<Vec<PathBuf>> {
    let next_id = self.current().map_or(0, |m| m.next_segment_id);
    let own_pid = std::process::id();
    let protected = |name: &str| {
      segment_id_of(name)
        .is_some_and(|id| id >= next_id || pending.contains(&id))
    };
    let referenced: HashSet<&str> = self
      .retained
      .iter()
      .flat_map(Manifest::referenced_files)
      .collect();
    let keep_gens: HashSet<u64> =
      self.retained.iter().map(|m| m.generation).collect();
    let mut garbage = Vec::new();
    for e in std::fs::read_dir(&self.dir)? {
      let e = e?;
      if !e.file_type()?.is_file() {
        continue;
      }
      let name = e.file_name();
      let Some(name) = name.to_str() else { continue };
      let is_garbage = if name.ends_with(TEMP_SUFFIX) {
        temp_pid_of(name) != Some(own_pid) && !protected(name)
      } else if let Some(g) = parse_manifest_name(name) {
        if self.retained.is_empty() {
          return Err(Error::invalid(format!(
            "gc refused: {} holds manifest {name}, which this store has not \
             loaded",
            self.dir.display()
          )));
        }
        !keep_gens.contains(&g)
      } else {
        name.starts_with(SEGMENT_PREFIX)
          && !referenced.contains(name)
          && !protected(name)
      };
      if is_garbage {
        garbage.push(e.path());
      }
    }
    garbage.sort();
    if delete && !garbage.is_empty() {
      let mut removed = Vec::with_capacity(garbage.len());
      for p in garbage {
        match std::fs::remove_file(&p) {
          Ok(()) => removed.push(p),
          Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
          Err(e) if in_use(&e) => {}
          Err(e) => return Err(e.into()),
        }
      }
      sync_dir(&self.dir)?;
      return Ok(removed);
    }
    Ok(garbage)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn names() {
    assert_eq!(parse_manifest_name(&manifest_file_name(42)), Some(42));
    assert_eq!(parse_manifest_name("MANIFEST-"), None);
    assert_eq!(
      parse_manifest_name("MANIFEST-00000000000000000042.1.2.tmp"),
      None
    );
    assert_eq!(segment_file_name(7, "idmap"), "seg-0000000007.idmap");
  }

  #[test]
  fn validate_rejects_bad_names_and_ids() {
    let mut m = Manifest::new(serde_json::json!({"dim": 4}));
    m.segments.push(SegmentEntry {
      id: 1,
      doc_count: 0,
      files: BTreeMap::from([("idmap".into(), "../evil".into())]),
      tombstones: None,
      meta: serde_json::Value::Null,
    });
    m.next_segment_id = 2;
    assert!(m.validate().is_err());
    m.segments[0]
      .files
      .insert("idmap".into(), "seg-1.idmap".into());
    m.validate().unwrap();
    m.next_segment_id = 1;
    assert!(m.validate().is_err());
  }

  #[test]
  fn encode_decode_round_trip() {
    let mut m = Manifest::new(serde_json::json!({"fields": ["a", "b"]}));
    m.generation = 3;
    let bytes = m.encode().unwrap();
    assert_eq!(Manifest::decode(Region::copy_from(&bytes)).unwrap(), m);
    let mut bad = bytes.clone();
    bad[70] ^= 0x20;
    assert!(Manifest::decode(Region::copy_from(&bad)).is_err());
  }
}
