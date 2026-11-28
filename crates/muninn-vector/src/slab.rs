//! Arrays that are either owned and growable or mapped read-only.
//!
//! Every index keeps its big arrays (vectors, graph, codes) in a [`Slab`].
//! A freshly built index owns them in a `Vec`; an index opened from a
//! [`Region`] borrows them in place, so opening costs no copy. The first
//! mutation of a mapped slab copies it into a `Vec` (copy-on-write).

use std::ptr::NonNull;

use muninn_core::{Error, Pod, Region, Result};

/// An owned or mapped array of `T`.
pub(crate) struct Slab<T: Pod> {
  repr: Repr<T>,
}

enum Repr<T: Pod> {
  Owned(Vec<T>),
  Mapped {
    /// Keeps the bytes behind `ptr` alive.
    _region: Region,
    ptr: NonNull<T>,
    len: usize,
  },
}

// SAFETY: a mapped slab is a shared, immutable view of bytes owned by a
// `Region` (whose owner is `Send + Sync`); `T: Pod` is `Send + Sync`.
// Nothing mutates the mapped bytes, so sharing across threads is sound.
unsafe impl<T: Pod> Send for Slab<T> {}
// SAFETY: see `Send`; `&Slab` only ever hands out `&[T]`.
unsafe impl<T: Pod> Sync for Slab<T> {}

impl<T: Pod> Slab<T> {
  /// An owned slab.
  pub(crate) fn owned(values: Vec<T>) -> Self {
    Self {
      repr: Repr::Owned(values),
    }
  }

  /// Views `region` as `[T]` without copying. Falls back to a copy only if
  /// the region is misaligned for `T` (for example a `Vec<u8>` that the
  /// caller did not allocate aligned). Fails if the length is ragged.
  pub(crate) fn from_region(region: Region) -> Result<Self> {
    let size = std::mem::size_of::<T>();
    if !region.len().is_multiple_of(size) {
      return Err(Error::corrupt(format!(
        "section of {} bytes is not a whole number of {size}-byte values",
        region.len()
      )));
    }
    let len = region.len() / size;
    let ptr = match region.cast::<T>() {
      Ok(s) => NonNull::from(s).cast::<T>(),
      Err(_) => {
        let v = (0..len)
          .map(|i| region.read::<T>(i * size))
          .collect::<Result<Vec<T>>>()?;
        return Ok(Self::owned(v));
      }
    };
    Ok(Self {
      repr: Repr::Mapped {
        _region: region,
        ptr,
        len,
      },
    })
  }

  /// The values.
  #[inline]
  pub(crate) fn as_slice(&self) -> &[T] {
    match &self.repr {
      Repr::Owned(v) => v,
      // SAFETY: `ptr`/`len` came from `Region::cast`, which checked
      // alignment and length; `_region` keeps the owner alive and its
      // bytes never change (a `Region` owner contract), so the slice is
      // valid for as long as `self` is borrowed.
      Repr::Mapped { ptr, len, .. } => unsafe {
        std::slice::from_raw_parts(ptr.as_ptr(), *len)
      },
    }
  }

  /// Mutable access to an owned `Vec`, copying a mapped slab first.
  pub(crate) fn to_mut(&mut self) -> &mut Vec<T> {
    if let Repr::Mapped { .. } = self.repr {
      let v = self.as_slice().to_vec();
      self.repr = Repr::Owned(v);
    }
    match &mut self.repr {
      Repr::Owned(v) => v,
      Repr::Mapped { .. } => unreachable!("converted to owned above"),
    }
  }

  /// Whether the values live in a mapped region rather than a `Vec`.
  pub(crate) fn is_mapped(&self) -> bool {
    matches!(self.repr, Repr::Mapped { .. })
  }

  #[inline]
  pub(crate) fn len(&self) -> usize {
    self.as_slice().len()
  }

  /// Heap or mapped bytes held by this slab.
  pub(crate) fn bytes(&self) -> usize {
    self.len() * std::mem::size_of::<T>()
  }
}

impl<T: Pod> Default for Slab<T> {
  fn default() -> Self {
    Self::owned(Vec::new())
  }
}

impl<T: Pod> Clone for Slab<T> {
  fn clone(&self) -> Self {
    match &self.repr {
      Repr::Owned(v) => Self::owned(v.clone()),
      Repr::Mapped { _region, ptr, len } => Self {
        repr: Repr::Mapped {
          _region: _region.clone(),
          ptr: *ptr,
          len: *len,
        },
      },
    }
  }
}

impl<T: Pod> std::fmt::Debug for Slab<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Slab")
      .field("len", &self.len())
      .field("mapped", &self.is_mapped())
      .finish()
  }
}

#[cfg(test)]
mod tests {
  use muninn_core::AlignedBuf;

  use super::*;

  #[test]
  fn mapped_is_zero_copy_and_copy_on_write() {
    let mut b = AlignedBuf::new();
    b.extend_from_pod(&[1.0f32, 2.0, 3.0]);
    let r = Region::new(b);
    let base = r.bytes().as_ptr();
    let mut s = Slab::<f32>::from_region(r).unwrap();
    assert!(s.is_mapped());
    assert_eq!(s.as_slice().as_ptr().cast::<u8>(), base);
    assert_eq!(s.as_slice(), &[1.0, 2.0, 3.0]);
    let c = s.clone();
    s.to_mut().push(4.0);
    assert!(!s.is_mapped());
    assert_eq!(s.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0]);
  }

  #[test]
  fn misaligned_region_is_copied_and_ragged_rejected() {
    let mut b = AlignedBuf::new();
    b.extend_from_slice(&[0]);
    b.extend_from_pod(&[7u32, 8]);
    let r = Region::new(b);
    let s = Slab::<u32>::from_region(r.slice(1..9).unwrap()).unwrap();
    assert!(!s.is_mapped());
    assert_eq!(s.as_slice(), &[7, 8]);
    assert!(Slab::<u32>::from_region(r.slice(1..8).unwrap()).is_err());
  }
}
