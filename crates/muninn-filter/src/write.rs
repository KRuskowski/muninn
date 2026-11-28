//! Writes the frozen payload index format from a [`MemIndex`] or from a
//! merge of frozen indexes.
//!
//! [`MemIndex`]: crate::MemIndex

use std::borrow::Cow;
use std::io::Write;

use muninn_core::container::{ContainerWriter, tag};
use muninn_core::{DocId, Error, Result};

use crate::value::key_f64;
use crate::{FieldType, Schema};

/// Container kind of a frozen payload index.
pub(crate) const KIND: [u8; 8] = tag("pindex");
/// Format version; bump on any change.
pub(crate) const VERSION: u32 = 1;
/// `u64`s per field record in the `fields` section.
pub(crate) const RECORD: usize = 12;
/// Entries per chunk when streaming numeric columns.
const CHUNK: usize = 4096;

/// Tag of section `kind` for field `i`: the kind letter and four hex
/// digits, e.g. `P002a`.
pub(crate) fn field_tag(kind: u8, i: usize) -> [u8; 8] {
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut t = [0u8; 8];
  t[0] = kind;
  for (k, b) in t[1..5].iter_mut().enumerate() {
    *b = HEX[(i >> (12 - 4 * k)) & 15];
  }
  t
}

/// Where a posting lives in a field's `P` section, in `u32` units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Posting {
  /// A bitset of `2 * ceil(doc_count / 64)` `u32` words rather than a
  /// sorted list of `count` ids.
  pub(crate) bits: bool,
  pub(crate) off: usize,
  /// Number of documents.
  pub(crate) count: usize,
}

const BITS_FLAG: u64 = 1 << 63;

impl Posting {
  pub(crate) fn encode(self) -> [u64; 2] {
    let flag = if self.bits { BITS_FLAG } else { 0 };
    [self.off as u64 | flag, self.count as u64]
  }
}

/// Receives each term and its postings, in term order.
pub(crate) type TermVisitor<'a> = dyn FnMut(&[u8], &[DocId]) -> Result<()> + 'a;

/// What the writer needs from its input. Postings are sorted and unique,
/// terms come in byte order with non-empty postings, numeric entries are
/// sorted by `(key, doc)` and unique (float keys via `f64_key`).
pub(crate) trait WriteSource {
  fn schema(&self) -> &Schema;
  fn doc_count(&self) -> usize;
  fn exists(&self, field: usize) -> Result<Cow<'_, [DocId]>>;
  fn bools(&self, field: usize, value: bool) -> Result<Cow<'_, [DocId]>>;
  fn terms(&self, field: usize, visit: &mut TermVisitor<'_>) -> Result<()>;
  fn numbers(&self, field: usize) -> Result<Vec<(i64, DocId)>>;
}

/// Appends postings to an open `P` section, choosing the encoding.
struct PostingWriter {
  pos: usize,
  doc_count: usize,
  scratch: Vec<u32>,
}

impl PostingWriter {
  fn put<W: Write>(
    &mut self,
    w: &mut ContainerWriter<W>,
    docs: &[DocId],
  ) -> Result<Posting> {
    if docs.last().is_some_and(|&d| d as usize >= self.doc_count) {
      return Err(Error::invalid("posting id beyond the document count"));
    }
    let count = docs.len();
    // A bitset costs doc_count / 8 bytes, a list 4 bytes per id: take
    // whichever is smaller. Dense terms then also evaluate as a
    // sequential OR of words instead of scattered bit writes.
    if count * 32 > self.doc_count {
      if self.pos % 2 == 1 {
        w.write_pod(&[0u32])?;
        self.pos += 1;
      }
      let words = 2 * self.doc_count.div_ceil(64);
      self.scratch.clear();
      self.scratch.resize(words, 0);
      for &d in docs {
        self.scratch[(d >> 5) as usize] |= 1 << (d & 31);
      }
      w.write_pod(&self.scratch)?;
      let p = Posting {
        bits: true,
        off: self.pos,
        count,
      };
      self.pos += words;
      Ok(p)
    } else {
      w.write_pod(docs)?;
      let p = Posting {
        bits: false,
        off: self.pos,
        count,
      };
      self.pos += count;
      Ok(p)
    }
  }
}

/// Writes the frozen index of `src`. Layout, per field `i` (hex):
///
/// - `Pxxxx` `u32` postings: the exists set first, then each keyword
///   term's or each bool value's postings
/// - keyword: `Txxxx` term bytes, `Oxxxx` `u64` term offsets (count + 1),
///   `Dxxxx` `u64` pairs: posting descriptor per term
/// - integer / float: `Vxxxx` `i64` / `f64` values ascending,
///   `Ixxxx` `u32` doc of each value
///
/// then `fields` (`RECORD` `u64`s per field: type code, name offset and
/// length, exists descriptor, two more descriptors for bool false/true,
/// term or value count), `names` (UTF-8) and `meta` (`u64` doc count,
/// field count, two reserved zeros).
pub(crate) fn write_index<W: Write>(
  src: &impl WriteSource,
  out: W,
) -> Result<W> {
  let schema = src.schema();
  let doc_count = src.doc_count();
  if doc_count > u32::MAX as usize {
    return Err(Error::invalid("too many documents for one index"));
  }
  let mut w = ContainerWriter::new(out, KIND, VERSION)?;
  let mut table: Vec<u64> = Vec::with_capacity(schema.len() * RECORD);
  let mut names: Vec<u8> = Vec::new();
  for (i, def) in schema.fields().iter().enumerate() {
    let ty = def.field_type();
    let mut rec = [0u64; RECORD];
    rec[0] = ty.code();
    rec[1] = names.len() as u64;
    rec[2] = def.name().len() as u64;
    names.extend_from_slice(def.name().as_bytes());
    let mut pw = PostingWriter {
      pos: 0,
      doc_count,
      scratch: Vec::new(),
    };
    w.begin_section(field_tag(b'P', i))?;
    let exists = src.exists(i)?;
    rec[3..5].copy_from_slice(&pw.put(&mut w, &exists)?.encode());
    drop(exists);
    match ty {
      FieldType::Keyword => {
        let mut bytes: Vec<u8> = Vec::new();
        let mut offs: Vec<u64> = vec![0];
        let mut descs: Vec<u64> = Vec::new();
        src.terms(i, &mut |term, docs| {
          let p = pw.put(&mut w, docs)?;
          bytes.extend_from_slice(term);
          offs.push(bytes.len() as u64);
          descs.extend_from_slice(&p.encode());
          Ok(())
        })?;
        w.end_section()?;
        w.section(field_tag(b'T', i), &bytes)?;
        w.pod_section(field_tag(b'O', i), &offs)?;
        w.pod_section(field_tag(b'D', i), &descs)?;
        rec[9] = (offs.len() - 1) as u64;
      }
      FieldType::Bool => {
        for (b, slot) in [(false, 5), (true, 7)] {
          let docs = src.bools(i, b)?;
          rec[slot..slot + 2].copy_from_slice(&pw.put(&mut w, &docs)?.encode());
        }
        w.end_section()?;
      }
      FieldType::Integer | FieldType::Float => {
        w.end_section()?;
        let nums = src.numbers(i)?;
        w.begin_section(field_tag(b'V', i))?;
        for chunk in nums.chunks(CHUNK) {
          if ty == FieldType::Integer {
            let vals: Vec<i64> = chunk.iter().map(|e| e.0).collect();
            w.write_pod(&vals)?;
          } else {
            let vals: Vec<f64> = chunk.iter().map(|e| key_f64(e.0)).collect();
            w.write_pod(&vals)?;
          }
        }
        w.end_section()?;
        w.begin_section(field_tag(b'I', i))?;
        for chunk in nums.chunks(CHUNK) {
          let docs: Vec<u32> = chunk.iter().map(|e| e.1).collect();
          w.write_pod(&docs)?;
        }
        w.end_section()?;
        rec[9] = nums.len() as u64;
      }
    }
    table.extend_from_slice(&rec);
  }
  w.pod_section(tag("fields"), &table)?;
  w.section(tag("names"), &names)?;
  w.pod_section(tag("meta"), &[doc_count as u64, schema.len() as u64, 0, 0])?;
  w.finish()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tags_are_distinct_hex() {
    assert_eq!(&field_tag(b'P', 0x2a)[..], b"P002a\0\0\0");
    assert_eq!(&field_tag(b'V', 0xffff)[..], b"Vffff\0\0\0");
    assert_ne!(field_tag(b'P', 1), field_tag(b'P', 16));
  }
}
