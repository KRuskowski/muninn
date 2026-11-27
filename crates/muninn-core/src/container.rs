//! The one file format every muninn index file uses.
//!
//! ```text
//! offset 0     header (64 bytes)
//!              magic "MUNINN\0\1", kind [u8; 8], version u32, zero pad
//! offset 64    section 0 bytes, then zero pad to a multiple of 64
//!              section 1 bytes, ...
//! table        per section: tag [u8; 8], offset u64, len u64 (24 bytes)
//! footer       last 64 bytes: table_offset u64, section_count u32,
//!              crc32c u32 of every byte before the footer,
//!              magic "MUNINN\0\1" at bytes 56..64, zero pad between
//! ```
//!
//! All integers are little-endian. Every section starts at a multiple of
//! [`ALIGN`], so a memory-mapped file yields aligned arrays. The table and
//! footer sit at the end so the file can be written in one streaming pass
//! with plain [`std::io::Write`] (no seeking), which lets index builders
//! write gigabyte sections without buffering them.
//!
//! Opening checks the structure only. The checksum covers the whole file,
//! so it is verified on demand ([`Container::verify_checksum`]) rather
//! than on every open of a large memory map.

use std::io::Write;

use crate::crc32c::Crc32c;
use crate::pod::{Pod, as_bytes};
use crate::{ALIGN, Error, Region, Result};

/// File magic, at the start of the header and the end of the footer.
pub const MAGIC: [u8; 8] = *b"MUNINN\0\x01";
const HEADER_LEN: usize = 64;
const FOOTER_LEN: usize = 64;
const ENTRY_LEN: usize = 24;

/// Converts a short ASCII name into an 8-byte tag, zero padded.
/// Panics (at compile time when used in a const) if longer than 8 bytes.
pub const fn tag(name: &str) -> [u8; 8] {
  let b = name.as_bytes();
  assert!(b.len() <= 8, "tags are at most 8 bytes");
  let mut t = [0u8; 8];
  let mut i = 0;
  while i < b.len() {
    t[i] = b[i];
    i += 1;
  }
  t
}

fn tag_str(t: &[u8; 8]) -> String {
  String::from_utf8_lossy(t)
    .trim_end_matches('\0')
    .to_string()
}

/// Streams sections into a container file.
pub struct ContainerWriter<W: Write> {
  out: W,
  pos: u64,
  crc: Crc32c,
  table: Vec<([u8; 8], u64, u64)>,
  open: Option<([u8; 8], u64)>,
}

impl<W: Write> ContainerWriter<W> {
  /// Writes the header. `kind` names the file type (e.g. `tag("hnsw")`),
  /// `version` its format version; readers reject other kinds/versions.
  pub fn new(out: W, kind: [u8; 8], version: u32) -> Result<Self> {
    let mut w = Self {
      out,
      pos: 0,
      crc: Crc32c::new(),
      table: Vec::new(),
      open: None,
    };
    let mut h = [0u8; HEADER_LEN];
    h[0..8].copy_from_slice(&MAGIC);
    h[8..16].copy_from_slice(&kind);
    h[16..20].copy_from_slice(&version.to_le_bytes());
    w.raw(&h)?;
    Ok(w)
  }

  fn raw(&mut self, bytes: &[u8]) -> Result<()> {
    self.out.write_all(bytes)?;
    self.crc.update(bytes);
    self.pos += bytes.len() as u64;
    Ok(())
  }

  fn pad(&mut self) -> Result<()> {
    let rem = (self.pos % ALIGN as u64) as usize;
    if rem != 0 {
      self.raw(&[0u8; ALIGN][..ALIGN - rem])?;
    }
    Ok(())
  }

  /// Starts a section; follow with [`Self::write`] calls and
  /// [`Self::end_section`]. Tags must be unique within a file.
  pub fn begin_section(&mut self, tag: [u8; 8]) -> Result<()> {
    if self.open.is_some() {
      return Err(Error::invalid("previous section still open"));
    }
    if self.table.iter().any(|e| e.0 == tag) {
      return Err(Error::invalid(format!("duplicate tag {}", tag_str(&tag))));
    }
    self.pad()?;
    self.open = Some((tag, self.pos));
    Ok(())
  }

  /// Appends bytes to the open section.
  pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
    if self.open.is_none() {
      return Err(Error::invalid("no open section"));
    }
    self.raw(bytes)
  }

  /// Appends `Pod` values (little-endian targets only; see crate docs).
  pub fn write_pod<T: Pod>(&mut self, values: &[T]) -> Result<()> {
    self.write(as_bytes(values))
  }

  pub fn end_section(&mut self) -> Result<()> {
    let (tag, start) = self
      .open
      .take()
      .ok_or_else(|| Error::invalid("no open section"))?;
    self.table.push((tag, start, self.pos - start));
    Ok(())
  }

  /// Writes a whole section in one call.
  pub fn section(&mut self, tag: [u8; 8], bytes: &[u8]) -> Result<()> {
    self.begin_section(tag)?;
    self.write(bytes)?;
    self.end_section()
  }

  /// Writes a whole section of `Pod` values in one call.
  pub fn pod_section<T: Pod>(
    &mut self,
    tag: [u8; 8],
    values: &[T],
  ) -> Result<()> {
    self.section(tag, as_bytes(values))
  }

  /// Writes the table and footer and returns the underlying writer.
  /// The caller still owns flushing/fsync.
  pub fn finish(mut self) -> Result<W> {
    if self.open.is_some() {
      return Err(Error::invalid("section still open at finish"));
    }
    self.pad()?;
    let table_offset = self.pos;
    let table = std::mem::take(&mut self.table);
    let mut entries = Vec::with_capacity(table.len() * ENTRY_LEN);
    for (tag, off, len) in &table {
      entries.extend_from_slice(tag);
      entries.extend_from_slice(&off.to_le_bytes());
      entries.extend_from_slice(&len.to_le_bytes());
    }
    self.raw(&entries)?;
    let mut f = [0u8; FOOTER_LEN];
    f[0..8].copy_from_slice(&table_offset.to_le_bytes());
    f[8..12].copy_from_slice(&(table.len() as u32).to_le_bytes());
    f[12..16].copy_from_slice(&self.crc.finish().to_le_bytes());
    f[56..64].copy_from_slice(&MAGIC);
    self.out.write_all(&f)?;
    Ok(self.out)
  }
}

/// A container opened from a [`Region`] (usually a memory map).
#[derive(Clone, Debug)]
pub struct Container {
  region: Region,
  kind: [u8; 8],
  version: u32,
  sections: Vec<([u8; 8], Region)>,
  crc: u32,
}

impl Container {
  /// Parses header, table and footer. Does not verify the checksum.
  pub fn open(region: Region) -> Result<Self> {
    let n = region.len();
    if n < HEADER_LEN + FOOTER_LEN {
      return Err(Error::corrupt("file too short for a muninn container"));
    }
    let b = region.bytes();
    if b[0..8] != MAGIC || b[n - 8..n] != MAGIC {
      return Err(Error::corrupt("bad magic: not a muninn file"));
    }
    let kind: [u8; 8] = b[8..16].try_into().unwrap();
    let version = region.read::<u32>(16)?.to_le();
    let foot = n - FOOTER_LEN;
    if b[foot + 16..n - 8].iter().any(|&x| x != 0) {
      return Err(Error::corrupt("footer padding is not zero"));
    }
    let table_offset = u64::from_le(region.read::<u64>(foot)?) as usize;
    let count = u32::from_le(region.read::<u32>(foot + 8)?) as usize;
    let crc = u32::from_le(region.read::<u32>(foot + 12)?);
    let table_end = count
      .checked_mul(ENTRY_LEN)
      .and_then(|l| l.checked_add(table_offset))
      .filter(|&e| e == foot)
      .ok_or_else(|| Error::corrupt("section table does not fit"))?;
    let mut sections = Vec::with_capacity(count);
    let mut at = table_offset;
    while at < table_end {
      let tag: [u8; 8] = b[at..at + 8].try_into().unwrap();
      let off = u64::from_le(region.read::<u64>(at + 8)?) as usize;
      let len = u64::from_le(region.read::<u64>(at + 16)?) as usize;
      let end = off
        .checked_add(len)
        .filter(|&e| off >= HEADER_LEN && e <= table_offset)
        .ok_or_else(|| {
          Error::corrupt(format!("section {} out of bounds", tag_str(&tag)))
        })?;
      if !off.is_multiple_of(ALIGN) {
        return Err(Error::corrupt("misaligned section"));
      }
      sections.push((tag, region.slice(off..end)?));
      at += ENTRY_LEN;
    }
    Ok(Self {
      region,
      kind,
      version,
      sections,
      crc,
    })
  }

  /// Opens and checks the file is of `kind` at exactly `version`.
  pub fn open_kind(
    region: Region,
    kind: [u8; 8],
    version: u32,
  ) -> Result<Self> {
    let c = Self::open(region)?;
    if c.kind != kind {
      return Err(Error::corrupt(format!(
        "expected a {} file, found {}",
        tag_str(&kind),
        tag_str(&c.kind)
      )));
    }
    if c.version != version {
      return Err(Error::corrupt(format!(
        "{} format version {} is not supported (expected {version})",
        tag_str(&kind),
        c.version
      )));
    }
    Ok(c)
  }

  pub fn kind(&self) -> [u8; 8] {
    self.kind
  }

  pub fn version(&self) -> u32 {
    self.version
  }

  /// The bytes of section `tag`.
  pub fn section(&self, tag: [u8; 8]) -> Result<Region> {
    self.try_section(tag).ok_or_else(|| {
      Error::corrupt(format!("missing section {}", tag_str(&tag)))
    })
  }

  /// The bytes of section `tag`, if present.
  pub fn try_section(&self, tag: [u8; 8]) -> Option<Region> {
    self
      .sections
      .iter()
      .find(|s| s.0 == tag)
      .map(|s| s.1.clone())
  }

  /// Section tags in file order.
  pub fn tags(&self) -> impl Iterator<Item = [u8; 8]> + '_ {
    self.sections.iter().map(|s| s.0)
  }

  /// Reads the whole file and checks the CRC-32C in the footer.
  pub fn verify_checksum(&self) -> Result<()> {
    let n = self.region.len();
    let mut c = Crc32c::new();
    c.update(&self.region.bytes()[..n - FOOTER_LEN]);
    if c.finish() != self.crc {
      return Err(Error::corrupt("checksum mismatch"));
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::AlignedBuf;

  const KIND: [u8; 8] = tag("test");

  fn sample() -> AlignedBuf {
    let mut w = ContainerWriter::new(AlignedBuf::new(), KIND, 3).unwrap();
    w.section(tag("meta"), b"hello").unwrap();
    w.begin_section(tag("floats")).unwrap();
    w.write_pod(&[1.0f32, 2.0]).unwrap();
    w.write_pod(&[3.0f32]).unwrap();
    w.end_section().unwrap();
    w.pod_section(tag("empty"), &[] as &[u32]).unwrap();
    w.finish().unwrap()
  }

  #[test]
  fn round_trip() {
    let c = Container::open_kind(Region::new(sample()), KIND, 3).unwrap();
    assert_eq!(c.section(tag("meta")).unwrap().bytes(), b"hello");
    let f = c.section(tag("floats")).unwrap();
    assert_eq!(f.cast::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
    assert!(c.section(tag("empty")).unwrap().is_empty());
    assert!(c.try_section(tag("nope")).is_none());
    assert_eq!(c.tags().count(), 3);
    c.verify_checksum().unwrap();
  }

  #[test]
  fn rejects_wrong_kind_version_and_corruption() {
    let buf = sample();
    let r = Region::new(buf.clone());
    assert!(Container::open_kind(r.clone(), tag("other"), 3).is_err());
    assert!(Container::open_kind(r, KIND, 4).is_err());
    let mut bad = buf.clone();
    bad.as_mut_slice()[70] ^= 1;
    let c = Container::open(Region::new(bad)).unwrap();
    assert!(c.verify_checksum().is_err());
    // The footer is outside the checksum, so its padding is checked on
    // open instead.
    let mut pad = buf.clone();
    let at = pad.len() - 30;
    pad.as_mut_slice()[at] = 1;
    assert!(Container::open(Region::new(pad)).is_err());
    let truncated = Region::copy_from(&buf.as_slice()[..buf.len() - 1]);
    assert!(Container::open(truncated).is_err());
  }

  #[test]
  fn writer_misuse_is_an_error() {
    let mut w = ContainerWriter::new(Vec::new(), KIND, 1).unwrap();
    assert!(w.write(b"x").is_err());
    w.section(tag("a"), b"1").unwrap();
    assert!(w.section(tag("a"), b"2").is_err());
    w.begin_section(tag("b")).unwrap();
    assert!(w.begin_section(tag("c")).is_err());
    assert!(w.finish().is_err());
  }
}
