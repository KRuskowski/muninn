//! The database: a directory of collections.
//!
//! ```text
//! <dir>/LOCK                         exclusive lock (one process at a time)
//! <dir>/collections/<name>/          one collection (see muninn-store)
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use muninn_core::{Error, Result};
use muninn_store::DirLock;
use muninn_store::fs::{sync_dir, temp_path_for};

use crate::collection::{Collection, Core, Handle};
use crate::options::Options;
use crate::schema::Schema;

const COLLECTIONS_DIR: &str = "collections";

/// An open database. Dropping it (and every [`Collection`] handle taken
/// from it) stops background work and releases the directory.
///
/// Opening replays every collection's write-ahead log, so all
/// acknowledged writes are back, per the durability mode they were made
/// with.
pub struct Database {
  dir: PathBuf,
  opts: Options,
  lock: Arc<DirLock>,
  collections: Mutex<BTreeMap<String, Collection>>,
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
        // Interrupted create_collection or drop_collection.
        if name.ends_with(".creating") || name.ends_with(".dropped") {
          leftovers.push(e.path());
        }
        continue;
      }
      let core = Core::open(&e.path(), &name, &opts)?;
      let h = Handle::start(core, lock.clone())?;
      collections.insert(name, Collection { h });
    }
    for p in leftovers {
      std::fs::remove_dir_all(&p)?;
    }
    Ok(Database {
      dir,
      opts,
      lock,
      collections: Mutex::new(collections),
    })
  }

  fn map(&self) -> Result<MutexGuard<'_, BTreeMap<String, Collection>>> {
    self.collections.lock().map_err(|_| {
      Error::Io(std::io::Error::other("database lock poisoned by a panic"))
    })
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
    self
      .map()?
      .get(name)
      .cloned()
      .ok_or_else(|| Error::NotFound(format!("collection {name:?}")))
  }

  /// Names of all collections, sorted.
  pub fn list_collections(&self) -> Result<Vec<String>> {
    Ok(self.map()?.keys().cloned().collect())
  }

  /// Deletes a collection and its files. Handles to it still held fail
  /// with `NotFound` from then on.
  pub fn drop_collection(&self, name: &str) -> Result<()> {
    let c = self
      .map()?
      .remove(name)
      .ok_or_else(|| Error::NotFound(format!("collection {name:?}")))?;
    c.h.core.mark_dropped();
    c.h.stop();
    let path = c.h.core.dir.clone();
    drop(c);
    // Rename first so a crash never leaves half a collection under its
    // name; the leftover is removed now or at the next open.
    let aside = temp_path_for(&path).with_extension("dropped");
    std::fs::rename(&path, &aside)?;
    sync_dir(&self.dir.join(COLLECTIONS_DIR))?;
    std::fs::remove_dir_all(&aside)?;
    Ok(())
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
