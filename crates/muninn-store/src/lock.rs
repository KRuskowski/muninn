//! Exclusive per-directory lock.
//!
//! A database or collection directory is opened for writing by at most one
//! [`DirLock`] at a time, across processes and within one process. The
//! lock is the operating system's file lock on `<dir>/LOCK`
//! ([`File::try_lock`]: `flock` on Unix, `LockFileEx` on Windows); the
//! kernel releases it when the holder exits or crashes, so a stale lock
//! file never blocks a restart.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use muninn_core::Result;

/// Name of the lock file inside a locked directory.
pub const LOCK_FILE: &str = "LOCK";

/// Holds the exclusive lock on a directory until dropped.
#[derive(Debug)]
pub struct DirLock {
  _file: File,
  path: PathBuf,
}

impl DirLock {
  /// Locks `dir` (which must exist) or fails immediately.
  ///
  /// When another holder has the directory, the error is an I/O error of
  /// kind [`std::io::ErrorKind::WouldBlock`] naming the directory and,
  /// where the platform lets other processes read a locked file (Unix),
  /// the process id recorded by the holder.
  pub fn acquire(dir: &Path) -> Result<DirLock> {
    let path = dir.join(LOCK_FILE);
    let mut file = OpenOptions::new()
      .read(true)
      .write(true)
      .create(true)
      .truncate(false)
      .open(&path)?;
    match file.try_lock() {
      Ok(()) => {}
      Err(TryLockError::Error(e)) => return Err(e.into()),
      Err(TryLockError::WouldBlock) => {
        // Windows locks are mandatory, so this read fails there and the
        // message goes without the pid.
        let holder = std::fs::read_to_string(&path)
          .ok()
          .map(|s| s.trim().to_string())
          .filter(|s| !s.is_empty())
          .map(|pid| format!(" (held by process {pid})"))
          .unwrap_or_default();
        return Err(
          std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            format!(
              "{} is already open for writing by another muninn \
               instance{holder}",
              dir.display()
            ),
          )
          .into(),
        );
      }
    }
    // Diagnostic only: record our pid for the error message above.
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    writeln!(file, "{}", std::process::id())?;
    Ok(DirLock { _file: file, path })
  }

  /// Path of the lock file.
  pub fn path(&self) -> &Path {
    &self.path
  }
}
