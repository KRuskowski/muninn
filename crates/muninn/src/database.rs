//! The database: a directory of collections.
//!
//! ```text
//! <dir>/LOCK                         exclusive lock (one process at a time)
//! <dir>/collections/<name>/          one collection (see muninn-store)
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use muninn_core::{Error, Result};
use muninn_store::DirLock;
use muninn_store::fs::sync_dir;

use crate::collection::{
  Collection, Core, DROPPED_MARKER, Handle, remove_collection_dir,
};
use crate::options::Options;
use crate::schema::Schema;

const COLLECTIONS_DIR: &str = "collections";

/// Prefix of a dropped collection's directory, renamed aside until its
/// files are released. Never a valid collection name.
const DROPPED_PREFIX: &str = ".dropped-";

/// An open database. Dropping it (and every [`Collection`] handle taken
/// from it) stops background work and releases the directory.
///
/// Opening replays every collection's write-ahead log, so all
/// acknowledged writes are back, per the durability mode they were made
/// with. A collection that cannot be opened (damaged files) does not stop
/// the others: it is listed by [`Database::open_failures`] and asking for
/// it returns the error.
pub struct Database {
  dir: PathBuf,
  opts: Options,
  lock: Arc<DirLock>,
  collections: Mutex<BTreeMap<String, Collection>>,
  /// Collections that failed to open, with the reason.
  failures: Mutex<BTreeMap<String, String>>,
}

impl std::fmt::Debug for Database {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Database").field("dir", &self.dir).finish()
  }
}

/// Collection names: 1 to 64 ASCII letters, digits, `_` or `-`.
fn check_name(name: &str) -> Result<()> {
  let ok = !name.is_empty()
    && name.len() <= 64
    && name
      .bytes()
      .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
  if !ok {
    return Err(Error::invalid(format!(
      "collection name {name:?} must be 1 to 64 letters, digits, '_' or '-'"
    )));
  }
  Ok(())
}

impl Database {
  /// Opens (creating if needed) the database in `dir`. Fails if another
  /// process (or another `Database` in this one) has it open.
  pub fn open(dir: impl AsRef<Path>, opts: Options) -> Result<Database> {
    opts.validate()?;
    let dir = dir.as_ref().to_path_buf();
    std::fs::create_dir_all(&dir)?;
    let lock = Arc::new(DirLock::acquire(&dir)?);
    let cdir = dir.join(COLLECTIONS_DIR);
    if !cdir.exists() {
      std::fs::create_dir_all(&cdir)?;
      sync_dir(&dir)?;
    }
    let mut collections = BTreeMap::new();
    let mut failures = BTreeMap::new();
    let mut leftovers = Vec::new();
    for e in std::fs::read_dir(&cdir)? {
      let e = e?;
      if !e.file_type()?.is_dir() {
        continue;
      }
      let Some(name) = e.file_name().to_str().map(String::from) else {
        continue;
      };
      if check_name(&name).is_err() {
        // Interrupted create_collection, or a dropped collection whose
        // files were still in use when it was dropped.
        if name.starts_with(DROPPED_PREFIX)
          || name.ends_with(".creating")
          || name.ends_with(".dropped")
        {
          leftovers.push(e.path());
        }
        continue;
      }
      if e.path().join(DROPPED_MARKER).exists() {
        leftovers.push(e.path());
        continue;
      }
      match Core::open(&e.path(), &name, &opts) {
        Ok(core) => {
          let h = Handle::start(core, lock.clone())?;
          collections.insert(name, Collection { h });
        }
        // An I/O error may be transient (or the whole disk): report it
        // rather than pretend the database opened.
        Err(Error::Io(e)) => return Err(Error::Io(e)),
        Err(e) => {
          failures.insert(name, e.to_string());
        }
      }
    }
    // Best effort: a directory that cannot be removed yet (files still
    // open elsewhere) is never opened and is tried again next time.
    for p in leftovers {
      let _ = remove_collection_dir(&p);
    }
    Ok(Database {
      dir,
      opts,
      lock,
      collections: Mutex::new(collections),
      failures: Mutex::new(failures),
    })
  }

  fn map(&self) -> Result<MutexGuard<'_, BTreeMap<String, Collection>>> {
    self.collections.lock().map_err(|_| {
      Error::Io(std::io::Error::other("database lock poisoned by a panic"))
    })
  }

  /// Collections that could not be opened, with the reason. Their files
  /// are left untouched; [`Database::drop_collection`] deletes them.
  pub fn open_failures(&self) -> Vec<(String, String)> {
    self
      .failures
      .lock()
      .map(|f| f.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
      .unwrap_or_default()
  }

  fn failure(&self, name: &str) -> Option<String> {
    self.failures.lock().ok().and_then(|f| f.get(name).cloned())
  }

  /// The database directory.
  pub fn path(&self) -> &Path {
    &self.dir
  }

  /// Creates a collection. Fails with `AlreadyExists` if the name is
  /// taken.
  pub fn create_collection(
    &self,
    name: &str,
    schema: Schema,
  ) -> Result<Collection> {
    check_name(name)?;
    schema.validate()?;
    let mut map = self.map()?;
    if map.contains_key(name) {
      return Err(Error::AlreadyExists(format!("collection {name:?}")));
    }
    let path = self.dir.join(COLLECTIONS_DIR).join(name);
    if path.join(DROPPED_MARKER).exists() {
      // A dropped collection that could not be set aside; its files are
      // free by now unless a search was still running.
      let _ = remove_collection_dir(&path);
    }
    if path.exists() {
      return Err(Error::AlreadyExists(format!(
        "{} exists but is not an open collection",
        path.display()
      )));
    }
    Core::create_dir(&path, &schema)?;
    let core = Core::open(&path, name, &self.opts)?;
    let c = Collection {
      h: Handle::start(core, self.lock.clone())?,
    };
    map.insert(name.to_string(), c.clone());
    Ok(c)
  }

  /// A handle to an existing collection.
  pub fn collection(&self, name: &str) -> Result<Collection> {
    if let Some(c) = self.map()?.get(name) {
      return Ok(c.clone());
    }
    match self.failure(name) {
      Some(reason) => Err(Error::Corrupt(format!(
        "collection {name:?} could not be opened: {reason}"
      ))),
      None => Err(Error::NotFound(format!("collection {name:?}"))),
    }
  }

  /// Names of all collections, sorted.
  pub fn list_collections(&self) -> Result<Vec<String>> {
    Ok(self.map()?.keys().cloned().collect())
  }

  /// Deletes a collection and its files. The name can be reused at
  /// once; handles to the old collection still held fail with `NotFound`
  /// from then on.
  ///
  /// The directory is renamed aside at once and deleted when the last
  /// handle and running search release it (Windows cannot delete files
  /// that are open or mapped). Anything left over, for example after a
  /// crash, is deleted when the database is next opened.
  pub fn drop_collection(&self, name: &str) -> Result<()> {
    if self.failure(name).is_some() && !self.map()?.contains_key(name) {
      let path = self.dir.join(COLLECTIONS_DIR).join(name);
      let aside = self.set_aside(name, &path)?;
      let _ = remove_collection_dir(&aside);
      if let Ok(mut f) = self.failures.lock() {
        f.remove(name);
      }
      return Ok(());
    }
    let c = self
      .map()?
      .remove(name)
      .ok_or_else(|| Error::NotFound(format!("collection {name:?}")))?;
    c.h.core.mark_dropped();
    c.h.stop();
    c.h.core.release_files();
    let path = c.h.core.dir.clone();
    let doomed = match self.set_aside(name, &path) {
      Ok(aside) => aside,
      Err(_) => {
        // The rename was refused (a file in use without delete
        // sharing). Mark the directory so it is never opened again.
        muninn_store::fs::write_atomic(&path.join(DROPPED_MARKER), b"")?;
        path
      }
    };
    // Usually nothing holds the files any more; if a running search still
    // maps a segment, the directory goes when the collection is released,
    // or at the next open.
    if remove_collection_dir(&doomed).is_err() {
      c.h.core.remove_on_release(doomed);
    }
    drop(c);
    Ok(())
  }

  /// Renames a collection directory to a hidden, unique name, durably.
  fn set_aside(&self, name: &str, path: &Path) -> Result<PathBuf> {
    static N: AtomicU64 = AtomicU64::new(0);
    let cdir = self.dir.join(COLLECTIONS_DIR);
    let aside = cdir.join(format!(
      "{DROPPED_PREFIX}{name}-{}-{}",
      std::process::id(),
      N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::rename(path, &aside)?;
    sync_dir(&cdir)?;
    Ok(aside)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn names() {
    for ok in ["a", "docs_v2", "A-1", &"x".repeat(64)] {
      check_name(ok).unwrap();
    }
    for bad in ["", "a.b", "a/b", "..", "é", &"x".repeat(65)] {
      assert!(check_name(bad).is_err(), "{bad:?}");
    }
  }
}
