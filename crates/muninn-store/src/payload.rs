//! Stored document payloads (JSON bytes per [`DocId`]).
//!
//! The store treats payloads as opaque bytes; the engine serialises JSON.
//! An empty payload means "none stored". [`PayloadBuilder`] is the mutable
//! memtable form; [`PayloadStore`] reads a frozen container of kind
//! `payload` in place.
//!
//! # File format (`payload`, version 1)
//!
//! | section   | contents                                           |
//! |-----------|----------------------------------------------------|
//! | `data`    | every payload, concatenated in doc order           |
//! | `offsets` | `[u64; docs + 1]`, doc `d` is `data[off[d]..off[d+1]]` |
//!
//! `data` comes first so a writer can stream payloads without buffering
//! them, keeping only the offsets in memory.
//!
//! Payloads are not compressed. Typical JSON payloads are a few hundred
//! bytes, too small for per-payload compression to gain much, and reads
//! are zero-copy only while stored bytes are the payload bytes.

use std::io::Write;
use std::path::Path;

use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{DocId, Error, Region, Result};

use crate::view::View;

/// Container kind of a payload file.
pub const PAYLOAD_KIND: [u8; 8] = tag("payload");
/// Current payload format version.
pub const PAYLOAD_VERSION: u32 = 1;

/// Streams payloads, in doc order, into a `payload` container.
pub fn write_payloads<'a, W, I>(out: W, payloads: I) -> Result<W>
where
  W: Write,
  I: IntoIterator<Item = &'a [u8]>,
{
  let mut w = ContainerWriter::new(out, PAYLOAD_KIND, PAYLOAD_VERSION)?;
  let mut offsets = vec![0u64];
  let mut at = 0u64;
  w.begin_section(tag("data"))?;
  for p in payloads {
    w.write(p)?;
    at += p.len() as u64;
    offsets.push(at);
  }
  w.end_section()?;
  if offsets.len() > DocId::MAX as usize {
    return Err(Error::invalid("too many payloads for one segment"));
  }
  w.pod_section(tag("offsets"), &offsets)?;
  w.finish()
}

/// Mutable payload store for the memtable: append in doc order.
#[derive(Clone, Debug)]
pub struct PayloadBuilder {
  offsets: Vec<usize>,
  data: Vec<u8>,
}

impl Default for PayloadBuilder {
  fn default() -> Self {
    Self::new()
  }
}

impl PayloadBuilder {
  /// An empty builder.
  pub fn new() -> Self {
    Self {
      offsets: vec![0],
      data: Vec::new(),
    }
  }

  /// Number of docs.
  pub fn len(&self) -> usize {
    self.offsets.len() - 1
  }

  /// True if no doc was added.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Total payload bytes held.
  pub fn data_len(&self) -> usize {
    self.data.len()
  }

  /// Appends the payload of the next doc and returns that doc's id.
  pub fn push(&mut self, payload: &[u8]) -> DocId {
    let doc = self.len() as DocId;
    self.data.extend_from_slice(payload);
    self.offsets.push(self.data.len());
    doc
  }

  /// The payload of `doc` (empty if none was stored).
  pub fn get(&self, doc: DocId) -> Option<&[u8]> {
    let d = doc as usize;
    let (a, b) = (*self.offsets.get(d)?, *self.offsets.get(d + 1)?);
    Some(&self.data[a..b])
  }

  /// Writes the frozen form as a `payload` container into `out`.
  pub fn write_to<W: Write>(&self, out: W) -> Result<W> {
    write_payloads(out, self.offsets.windows(2).map(|w| &self.data[w[0]..w[1]]))
  }

  /// Durably writes the frozen form to `path` (see [`crate::fs`]).
  pub fn write_file(&self, path: &Path) -> Result<u64> {
    crate::fs::write_atomic_with(path, |out| {
      self.write_to(out)?;
      Ok(())
    })
  }
}

/// Immutable payloads read in place from a `payload` container.
#[derive(Clone)]
pub struct PayloadStore {
  offsets: View<u64>,
  data: View<u8>,
}

impl std::fmt::Debug for PayloadStore {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("PayloadStore")
      .field("len", &self.len())
      .finish()
  }
}

impl PayloadStore {
  /// Opens a store from the bytes of a `payload` container. Checks
  /// structure, not the checksum; reads nothing proportional to size.
  pub fn open(region: Region) -> Result<Self> {
    let c = Container::open_kind(region, PAYLOAD_KIND, PAYLOAD_VERSION)?;
    let offsets = View::<u64>::new(c.section(tag("offsets"))?)?;
    let data = View::<u8>::new(c.section(tag("data"))?)?;
    let off = offsets.as_slice();
    if off.is_empty() || off[0] != 0 {
      return Err(Error::corrupt("payload offsets must start at 0"));
    }
    if off[off.len() - 1] != data.as_slice().len() as u64 {
      return Err(Error::corrupt("payload offsets do not cover the data"));
    }
    Ok(Self { offsets, data })
  }

  /// Number of docs.
  pub fn len(&self) -> usize {
    self.offsets.as_slice().len() - 1
  }

  /// True if the store holds no docs.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// The payload of `doc`, borrowed from the mapped file (empty if none
  /// was stored). Fails for docs outside the segment or corrupt offsets.
  #[inline]
  pub fn get(&self, doc: DocId) -> Result<&[u8]> {
    let off = self.offsets.as_slice();
    let d = doc as usize;
    if d + 1 >= off.len() {
      return Err(Error::invalid(format!(
        "doc {doc} outside payload store of {} docs",
        off.len() - 1
      )));
    }
    let (a, b) = (off[d], off[d + 1]);
    usize::try_from(a)
      .ok()
      .zip(usize::try_from(b).ok())
      .and_then(|(a, b)| self.data.as_slice().get(a..b))
      .ok_or_else(|| Error::corrupt(format!("payload {doc} out of range")))
  }
}
