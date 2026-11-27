//! Immutable byte regions that index files are opened from.
//!
//! A [`Region`] is a cheap, clonable view into bytes owned by something
//! else: a memory map, an [`AlignedBuf`], a `Vec<u8>`. Indexes are opened
//! from a `Region` and read their arrays in place, so opening a segment
//! from disk costs a few header reads instead of a deserialisation pass.
//!
//! On-disk layout rule for every muninn file: arrays start at offsets that
//! are multiples of [`ALIGN`], so a page-aligned memory map yields
//! correctly aligned `&[f32]`, `&[u32]` and SIMD loads.

use std::ops::Range;
use std::sync::Arc;

use crate::pod::{Pod, as_bytes};
use crate::{Error, Result};

/// Alignment, in bytes, of every array inside a muninn file and of every
/// [`AlignedBuf`]. One cache line; also covers AVX-512 loads.
pub const ALIGN: usize = 64;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u8; ALIGN]);

/// A growable byte buffer whose start is aligned to [`ALIGN`].
///
/// Used to build index files in memory and to hold regions that were
/// copied rather than mapped, so casts to `&[f32]` always succeed.
#[derive(Clone, Default)]
pub struct AlignedBuf {
  lines: Vec<Line>,
  len: usize,
}

impl AlignedBuf {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn with_capacity(bytes: usize) -> Self {
    Self {
      lines: Vec::with_capacity(bytes.div_ceil(ALIGN)),
      len: 0,
    }
  }

  /// A buffer of `len` zero bytes.
  pub fn zeroed(len: usize) -> Self {
    Self {
      lines: vec![Line([0; ALIGN]); len.div_ceil(ALIGN)],
      len,
    }
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.len
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.len == 0
  }

  #[inline]
  pub fn as_slice(&self) -> &[u8] {
    // SAFETY: `lines` owns at least `len` initialised bytes, contiguous
    // because `Line` is a repr(C) byte array with no padding.
    unsafe {
      std::slice::from_raw_parts(self.lines.as_ptr().cast::<u8>(), self.len)
    }
  }

  #[inline]
  pub fn as_mut_slice(&mut self) -> &mut [u8] {
    // SAFETY: as in `as_slice`, and we hold `&mut self`.
    unsafe {
      std::slice::from_raw_parts_mut(
        self.lines.as_mut_ptr().cast::<u8>(),
        self.len,
      )
    }
  }

  /// Resizes to `len` bytes, zero-filling any new bytes.
  pub fn resize(&mut self, len: usize) {
    self.lines.resize(len.div_ceil(ALIGN), Line([0; ALIGN]));
    if len > self.len {
      // Bytes past the old length may hold stale data from a shrink.
      let old = self.len;
      self.len = len;
      self.as_mut_slice()[old..].fill(0);
    } else {
      self.len = len;
    }
  }

  pub fn extend_from_slice(&mut self, bytes: &[u8]) {
    let old = self.len;
    self.resize(old + bytes.len());
    self.as_mut_slice()[old..].copy_from_slice(bytes);
  }

  /// Appends `Pod` values in native byte order.
  pub fn extend_from_pod<T: Pod>(&mut self, values: &[T]) {
    self.extend_from_slice(as_bytes(values));
  }

  /// Zero-pads so the length is a multiple of `align` (a power of two).
  /// Returns the new length, which is where the next array starts.
  pub fn pad_to(&mut self, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    let target = self.len.next_multiple_of(align);
    self.resize(target);
    target
  }
}

impl AsRef<[u8]> for AlignedBuf {
  fn as_ref(&self) -> &[u8] {
    self.as_slice()
  }
}

impl std::io::Write for AlignedBuf {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    self.extend_from_slice(buf);
    Ok(buf.len())
  }

  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

impl std::fmt::Debug for AlignedBuf {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("AlignedBuf")
      .field("len", &self.len)
      .finish()
  }
}

type Owner = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// A cheap, clonable, immutable view of bytes owned elsewhere.
#[derive(Clone)]
pub struct Region {
  owner: Owner,
  start: usize,
  len: usize,
}

impl Region {
  /// Wraps an owner, typically a memory map or an [`AlignedBuf`]. The
  /// owner's `as_ref()` must return the same bytes every time.
  pub fn new<T: AsRef<[u8]> + Send + Sync + 'static>(owner: T) -> Self {
    let len = owner.as_ref().len();
    Self {
      owner: Arc::new(owner),
      start: 0,
      len,
    }
  }

  /// Copies bytes into a fresh aligned buffer.
  pub fn copy_from(bytes: &[u8]) -> Self {
    let mut buf = AlignedBuf::with_capacity(bytes.len());
    buf.extend_from_slice(bytes);
    Self::new(buf)
  }

  #[inline]
  pub fn bytes(&self) -> &[u8] {
    &(*self.owner).as_ref()[self.start..self.start + self.len]
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.len
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// A sub-region sharing the same owner. Fails if out of bounds.
  pub fn slice(&self, range: Range<usize>) -> Result<Region> {
    if range.start > range.end || range.end > self.len {
      return Err(Error::corrupt(format!(
        "range {range:?} outside region of {} bytes",
        self.len
      )));
    }
    Ok(Region {
      owner: self.owner.clone(),
      start: self.start + range.start,
      len: range.end - range.start,
    })
  }

  /// Views the whole region as `&[T]` without copying. Fails if the
  /// region is misaligned for `T` or its length is not a multiple of
  /// `size_of::<T>()`; both mean the file was not written by muninn.
  pub fn cast<T: Pod>(&self) -> Result<&[T]> {
    let bytes = self.bytes();
    let size = std::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(size) {
      return Err(Error::corrupt(format!(
        "{} bytes is not a whole number of {size}-byte values",
        bytes.len()
      )));
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
      return Err(Error::corrupt("misaligned array in region"));
    }
    // SAFETY: length and alignment were checked above, T is Pod so any
    // bit pattern is valid, and the bytes live as long as `self`.
    Ok(unsafe {
      std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size)
    })
  }

  /// Reads one little-endian-encoded `Pod` value at `offset`, unaligned.
  pub fn read<T: Pod>(&self, offset: usize) -> Result<T> {
    let size = std::mem::size_of::<T>();
    let bytes = self.bytes();
    let end = offset
      .checked_add(size)
      .filter(|&e| e <= bytes.len())
      .ok_or_else(|| {
        Error::corrupt(format!("read of {size} bytes at {offset} overruns"))
      })?;
    // SAFETY: bounds checked above; read_unaligned has no alignment needs
    // and T is Pod so every bit pattern is valid.
    Ok(unsafe { bytes[offset..end].as_ptr().cast::<T>().read_unaligned() })
  }
}

impl AsRef<[u8]> for Region {
  fn as_ref(&self) -> &[u8] {
    self.bytes()
  }
}

impl std::fmt::Debug for Region {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Region").field("len", &self.len).finish()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn aligned_buf_is_aligned_and_pads() {
    let mut b = AlignedBuf::new();
    b.extend_from_slice(&[1, 2, 3]);
    assert_eq!(b.as_slice().as_ptr() as usize % ALIGN, 0);
    assert_eq!(b.pad_to(ALIGN), 64);
    b.extend_from_pod(&[1.5f32, 2.5]);
    assert_eq!(b.len(), 72);
    assert_eq!(&b.as_slice()[..3], &[1, 2, 3]);
    assert!(b.as_slice()[3..64].iter().all(|&x| x == 0));
  }

  #[test]
  fn shrink_then_grow_zero_fills() {
    let mut b = AlignedBuf::new();
    b.extend_from_slice(&[9; 10]);
    b.resize(2);
    b.resize(10);
    assert_eq!(&b.as_slice()[2..], &[0; 8]);
  }

  #[test]
  fn region_cast_and_slice() {
    let mut b = AlignedBuf::new();
    b.extend_from_pod(&[7u32, 8, 9, 10]);
    let r = Region::new(b);
    assert_eq!(r.cast::<u32>().unwrap(), &[7, 8, 9, 10]);
    let tail = r.slice(8..16).unwrap();
    assert_eq!(tail.cast::<u32>().unwrap(), &[9, 10]);
    assert_eq!(r.read::<u32>(4).unwrap(), 8);
    assert!(r.slice(8..17).is_err());
    assert!(r.read::<u64>(12).is_err());
    // Misaligned and ragged views are rejected, not UB.
    assert!(r.slice(1..5).unwrap().cast::<u32>().is_err());
    assert!(r.slice(0..6).unwrap().cast::<u32>().is_err());
  }
}
