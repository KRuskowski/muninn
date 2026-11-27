//! Typed zero-copy views that keep their backing [`Region`] alive.

use muninn_core::{Pod, Region, Result};

/// A `&[T]` over a region, validated once at construction.
///
/// [`Region::cast`] re-checks length and alignment and goes through the
/// owner's `as_ref` (a dynamic call) on every use. Hot lookups instead keep
/// the pointer it returned once, together with the region that owns the
/// bytes, so element access is a plain slice index.
pub(crate) struct View<T: Pod> {
  _region: Region,
  ptr: *const T,
  len: usize,
}

// SAFETY: a View is an immutable `&[T]` into bytes kept alive by `_region`
// (itself Send + Sync); T: Pod is Send + Sync. Sharing it is sharing a
// shared slice.
unsafe impl<T: Pod> Send for View<T> {}
// SAFETY: see the Send impl; no interior mutability.
unsafe impl<T: Pod> Sync for View<T> {}

impl<T: Pod> View<T> {
  pub(crate) fn new(region: Region) -> Result<Self> {
    let s = region.cast::<T>()?;
    let (ptr, len) = (s.as_ptr(), s.len());
    Ok(Self {
      _region: region,
      ptr,
      len,
    })
  }

  /// An empty view, for optional sections that are absent.
  pub(crate) fn empty() -> Self {
    Self {
      _region: Region::copy_from(&[]),
      ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
      len: 0,
    }
  }

  #[inline]
  pub(crate) fn as_slice(&self) -> &[T] {
    // SAFETY: ptr/len came from a successful `Region::cast` on `_region`,
    // whose owner returns the same bytes on every call (Region contract)
    // and lives as long as `self`. For `empty`, ptr is dangling but
    // aligned and non-null with len 0, which is a valid empty slice.
    unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
  }
}

impl<T: Pod> Clone for View<T> {
  fn clone(&self) -> Self {
    Self {
      _region: self._region.clone(),
      ptr: self.ptr,
      len: self.len,
    }
  }
}
