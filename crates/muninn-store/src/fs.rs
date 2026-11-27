//! Durable file replacement.
//!
//! Every file muninn publishes under its final name goes through the same
//! sequence: write a uniquely named temporary file next to the target,
//! fsync it, rename it over the target, fsync the directory. A crash at
//! any point leaves either the old file or the complete new one under the
//! final name, never a mix; at worst a stray `*.tmp` file remains, which
//! [`crate::ManifestStore::gc`] removes.
//!
//! On Windows the rename is `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING
//! | MOVEFILE_WRITE_THROUGH` (see [`rename`]), which replaces the target
//! atomically and returns once the change is on disk; there is no way to
//! open a directory for fsync there, so [`sync_dir`] is a no-op. The file
//! itself is flushed with `sync_all` (`FlushFileBuffers`) before the
//! rename. A file that is memory mapped may not be replaceable on
//! Windows, which is why manifests and tombstones are copied out of their
//! maps and unmapped before anything replaces them.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use muninn_core::Result;
use muninn_core::container::ContainerWriter;

/// File-name suffix of temporary files written by this module.
pub const TEMP_SUFFIX: &str = ".tmp";

/// Buffer size for streaming writes. Large enough that a container writer
/// issues few system calls, small enough not to matter for memory.
const WRITE_BUF: usize = 1 << 20;

/// Fsyncs a directory so that entries created, renamed or removed in it
/// survive a crash. A no-op on non-Unix platforms (see module docs).
pub fn sync_dir(dir: &Path) -> Result<()> {
  #[cfg(unix)]
  File::open(dir)?.sync_all()?;
  #[cfg(not(unix))]
  let _ = dir;
  Ok(())
}

/// Renames `from` to `to`, replacing `to` atomically if it exists.
///
/// On Windows this is `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING` and
/// `MOVEFILE_WRITE_THROUGH`, so the call returns only once the rename is
/// on disk (there is no directory fsync there). Elsewhere it is
/// [`std::fs::rename`]; follow it with [`sync_dir`].
pub fn rename(from: &Path, to: &Path) -> Result<()> {
  #[cfg(windows)]
  {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
      MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    let wide = |p: &Path| -> Vec<u16> {
      p.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
    };
    let (f, t) = (wide(from), wide(to));
    // SAFETY: both arguments are NUL-terminated UTF-16 strings that live
    // until the call returns; MoveFileExW does not keep them.
    let ok = unsafe {
      MoveFileExW(
        f.as_ptr(),
        t.as_ptr(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
      )
    };
    if ok == 0 {
      return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
  }
  #[cfg(not(windows))]
  {
    std::fs::rename(from, to)?;
    Ok(())
  }
}

/// The directory a path lives in; `.` for a bare file name.
pub(crate) fn parent_dir(path: &Path) -> &Path {
  match path.parent() {
    Some(p) if !p.as_os_str().is_empty() => p,
    _ => Path::new("."),
  }
}

/// A fresh temporary path next to `path`: `<name>.<pid>.<n>.tmp`.
///
/// Unique per process and call, so concurrent writers of different (or
/// even the same) targets never share a temporary file.
pub fn temp_path_for(path: &Path) -> PathBuf {
  static COUNTER: AtomicU64 = AtomicU64::new(0);
  let n = COUNTER.fetch_add(1, Ordering::Relaxed);
  let mut name: OsString = path.file_name().unwrap_or_default().to_owned();
  name.push(format!(".{}.{n}{TEMP_SUFFIX}", std::process::id()));
  parent_dir(path).join(name)
}

/// Removes the temporary file unless disarmed; keeps error paths tidy.
struct TempGuard(Option<PathBuf>);

impl Drop for TempGuard {
  fn drop(&mut self) {
    if let Some(p) = self.0.take() {
      let _ = std::fs::remove_file(p);
    }
  }
}

/// Atomically replaces (or creates) `path` with what `write` produces.
///
/// `write` streams into a buffered temporary file. After it returns, the
/// file is flushed and fsynced, renamed to `path`, and the directory is
/// fsynced. Returns the number of bytes in the new file. On error the
/// temporary file is removed and `path` is untouched.
pub fn write_atomic_with<F>(path: &Path, write: F) -> Result<u64>
where
  F: FnOnce(&mut BufWriter<File>) -> Result<()>,
{
  let tmp = temp_path_for(path);
  let mut guard = TempGuard(Some(tmp.clone()));
  let file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
  let mut out = BufWriter::with_capacity(WRITE_BUF, file);
  write(&mut out)?;
  let file = out.into_inner().map_err(|e| e.into_error())?;
  file.sync_all()?;
  let len = file.metadata()?.len();
  drop(file);
  rename(&tmp, path)?;
  guard.0 = None;
  sync_dir(parent_dir(path))?;
  Ok(len)
}

/// Atomically replaces (or creates) `path` with `bytes`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
  write_atomic_with(path, |w| Ok(w.write_all(bytes)?))?;
  Ok(())
}

/// Durably writes a container file at `path` in one streaming pass.
///
/// `build` adds sections to a [`ContainerWriter`] of the given `kind` and
/// `version`; the table and footer are written after it returns, then the
/// file is published as in [`write_atomic_with`]. Returns the file size.
pub fn write_container_file<F>(
  path: &Path,
  kind: [u8; 8],
  version: u32,
  build: F,
) -> Result<u64>
where
  F: FnOnce(&mut ContainerWriter<&mut BufWriter<File>>) -> Result<()>,
{
  write_atomic_with(path, |out| {
    let mut w = ContainerWriter::new(out, kind, version)?;
    build(&mut w)?;
    w.finish()?;
    Ok(())
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn temp_paths_are_unique_and_suffixed() {
    let a = temp_path_for(Path::new("dir/MANIFEST"));
    let b = temp_path_for(Path::new("dir/MANIFEST"));
    assert_ne!(a, b);
    assert_eq!(a.parent(), Some(Path::new("dir")));
    let name = a.file_name().unwrap().to_str().unwrap();
    assert!(name.starts_with("MANIFEST.") && name.ends_with(TEMP_SUFFIX));
    assert_eq!(parent_dir(Path::new("x")), Path::new("."));
  }
}
