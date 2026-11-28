//! Binary encoding of write-ahead-log records and stored documents.
//!
//! One `upsert` or `delete` call is one WAL record, so a batch is
//! recovered entirely or not at all. The engine applies a batch by
//! decoding the very bytes it logged, on the live path and during
//! recovery alike, so the two cannot disagree.
//!
//! # WAL record, format 1
//!
//! All integers are LEB128 varints unless noted; ids and floats are
//! little-endian.
//!
//! ```text
//! record  = format:u8(=1) op:u8 count docs-or-ids
//! op      = 1 (upsert, followed by `count` docs)
//!         | 2 (delete, followed by `count` ids)
//! id      = 0:u8 u64:8 bytes | 1:u8 len bytes(UTF-8)
//! doc     = id dim f32[dim] stored        (dim 0: no vector)
//! stored  = nfields { field len bytes(UTF-8) } plen payload(JSON)
//! ```
//!
//! `field` is the index of the text field in the schema. `stored` is
//! kept verbatim (behind a one-byte format tag) as the document's stored
//! record, so `get` and payload retrieval decode the same bytes.

use muninn_core::{Error, Result};

use crate::schema::Schema;
use crate::types::{Document, Id};

/// Format byte of every WAL record.
pub(crate) const WAL_FORMAT: u8 = 1;
/// Format byte in front of every stored record.
pub(crate) const STORED_FORMAT: u8 = 1;

const OP_UPSERT: u8 = 1;
const OP_DELETE: u8 = 2;

const ID_U64: u8 = 0;
const ID_STR: u8 = 1;

/// Largest string id, in bytes.
pub(crate) const MAX_ID_BYTES: usize = 64 << 10;

/// Deepest payload nesting accepted. serde_json writes any depth but
/// refuses to read past 128 levels, so a deeper payload could be logged
/// yet never read back; this keeps a safe margin.
pub const MAX_PAYLOAD_DEPTH: usize = 100;

/// Largest squared norm of a vector. With `|a|^2, |b|^2 <= M`, every
/// squared L2 distance is at most `4 M` and every dot product at most `M`,
/// so no distance between accepted vectors overflows `f32`.
const MAX_NORM_SQ: f64 = f32::MAX as f64 / 4.0;

/// Nesting depth of a JSON value (scalars are 0), without recursion.
fn json_depth(v: &serde_json::Value) -> usize {
  use serde_json::Value;
  let mut max = 0;
  let mut stack: Vec<(&Value, usize)> = vec![(v, 0)];
  while let Some((v, d)) = stack.pop() {
    match v {
      Value::Array(a) => {
        max = max.max(d + 1);
        stack.extend(a.iter().map(|x| (x, d + 1)));
      }
      Value::Object(o) => {
        max = max.max(d + 1);
        stack.extend(o.values().map(|x| (x, d + 1)));
      }
      _ => {}
    }
  }
  max
}

/// Checks a vector (document or query) is usable: finite components and
/// a squared norm that keeps every distance finite.
pub(crate) fn check_vector(v: &[f32], what: &str) -> Result<()> {
  if let Some(x) = v.iter().find(|x| !x.is_finite()) {
    return Err(Error::invalid(format!(
      "{what} has a non-finite vector component {x}"
    )));
  }
  let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
  if norm_sq > MAX_NORM_SQ {
    return Err(Error::invalid(format!(
      "{what} has a vector of squared length {norm_sq:e}, above the \
       {MAX_NORM_SQ:e} at which distances overflow"
    )));
  }
  Ok(())
}

pub(crate) fn put_varint(out: &mut Vec<u8>, mut v: u64) {
  while v >= 0x80 {
    out.push(v as u8 | 0x80);
    v >>= 7;
  }
  out.push(v as u8);
}

fn put_id(out: &mut Vec<u8>, id: &Id) {
  match id {
    Id::U64(v) => {
      out.push(ID_U64);
      out.extend_from_slice(&v.to_le_bytes());
    }
    Id::Str(s) => {
      out.push(ID_STR);
      put_varint(out, s.len() as u64);
      out.extend_from_slice(s.as_bytes());
    }
  }
}

fn check_id(id: &Id) -> Result<()> {
  if let Id::Str(s) = id
    && s.len() > MAX_ID_BYTES
  {
    return Err(Error::invalid(format!(
      "string id of {} bytes exceeds the {MAX_ID_BYTES}-byte limit",
      s.len()
    )));
  }
  Ok(())
}

/// Validates `docs` against `schema` and encodes them as one upsert
/// record. Nothing that passes here can fail to apply.
pub(crate) fn encode_upsert(
  schema: &Schema,
  docs: &[Document],
) -> Result<Vec<u8>> {
  let mut out = Vec::with_capacity(16 + docs.len() * 64);
  out.push(WAL_FORMAT);
  out.push(OP_UPSERT);
  put_varint(&mut out, docs.len() as u64);
  let mut payload = Vec::new();
  for d in docs {
    check_id(&d.id)?;
    match (&schema.vector, &d.vector) {
      (Some(f), Some(v)) => {
        if v.len() != f.dim {
          return Err(Error::DimensionMismatch {
            expected: f.dim,
            got: v.len(),
          });
        }
        check_vector(v, &format!("document {:?}", d.id))?;
      }
      (Some(_), None) => {
        return Err(Error::invalid(format!(
          "document {:?} has no vector, but the collection requires one",
          d.id
        )));
      }
      (None, Some(_)) => {
        return Err(Error::invalid(format!(
          "document {:?} has a vector, but the collection has no vector \
           field",
          d.id
        )));
      }
      (None, None) => {}
    }
    put_id(&mut out, &d.id);
    match &d.vector {
      Some(v) => {
        put_varint(&mut out, v.len() as u64);
        out.extend_from_slice(muninn_core::as_bytes(v.as_slice()));
      }
      None => put_varint(&mut out, 0),
    }
    put_varint(&mut out, d.text.len() as u64);
    for (name, text) in &d.text {
      let f = schema.text_field(name).ok_or_else(|| {
        Error::invalid(format!(
          "document {:?} has text field {name:?}, which the schema does \
           not declare",
          d.id
        ))
      })?;
      put_varint(&mut out, f as u64);
      put_varint(&mut out, text.len() as u64);
      out.extend_from_slice(text.as_bytes());
    }
    payload.clear();
    if json_depth(&d.payload) > MAX_PAYLOAD_DEPTH {
      return Err(Error::invalid(format!(
        "document {:?} has a payload nested deeper than \
         {MAX_PAYLOAD_DEPTH} levels",
        d.id
      )));
    }
    if !d.payload.is_null() {
      serde_json::to_writer(&mut payload, &d.payload)
        .map_err(|e| Error::invalid(format!("payload JSON: {e}")))?;
    }
    put_varint(&mut out, payload.len() as u64);
    out.extend_from_slice(&payload);
  }
  Ok(out)
}

/// Encodes a delete record.
pub(crate) fn encode_delete(ids: &[Id]) -> Result<Vec<u8>> {
  let mut out = Vec::with_capacity(8 + ids.len() * 12);
  out.push(WAL_FORMAT);
  out.push(OP_DELETE);
  put_varint(&mut out, ids.len() as u64);
  for id in ids {
    check_id(id)?;
    put_id(&mut out, id);
  }
  Ok(out)
}

/// A cursor over encoded bytes. Every read is bounds-checked.
pub(crate) struct Reader<'a> {
  buf: &'a [u8],
  pos: usize,
}

fn short() -> Error {
  Error::corrupt("record ends early")
}

impl<'a> Reader<'a> {
  pub(crate) fn new(buf: &'a [u8]) -> Self {
    Self { buf, pos: 0 }
  }

  pub(crate) fn is_empty(&self) -> bool {
    self.pos == self.buf.len()
  }

  fn u8(&mut self) -> Result<u8> {
    let b = *self.buf.get(self.pos).ok_or_else(short)?;
    self.pos += 1;
    Ok(b)
  }

  pub(crate) fn varint(&mut self) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
      let b = self.u8()?;
      v |= u64::from(b & 0x7f) << shift;
      if b < 0x80 {
        return Ok(v);
      }
    }
    Err(Error::corrupt("varint too long"))
  }

  fn len(&mut self) -> Result<usize> {
    let n = self.varint()?;
    usize::try_from(n)
      .ok()
      .filter(|&n| n <= self.buf.len() - self.pos)
      .ok_or_else(short)
  }

  pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
    let end = self.pos.checked_add(n).ok_or_else(short)?;
    let b = self.buf.get(self.pos..end).ok_or_else(short)?;
    self.pos = end;
    Ok(b)
  }

  fn str(&mut self) -> Result<&'a str> {
    let n = self.len()?;
    std::str::from_utf8(self.bytes(n)?)
      .map_err(|_| Error::corrupt("text is not UTF-8"))
  }

  fn id(&mut self) -> Result<Id> {
    match self.u8()? {
      ID_U64 => {
        let b = self.bytes(8)?;
        Ok(Id::U64(u64::from_le_bytes(b.try_into().unwrap())))
      }
      ID_STR => Ok(Id::Str(self.str()?.to_string())),
      t => Err(Error::corrupt(format!("unknown id tag {t}"))),
    }
  }

  /// The rest of the buffer.
  fn rest(&mut self) -> &'a [u8] {
    let r = &self.buf[self.pos..];
    self.pos = self.buf.len();
    r
  }
}

/// One decoded document of an upsert record, borrowing the record.
pub(crate) struct DocRef<'a> {
  pub id: Id,
  /// Little-endian `f32`s; empty without a vector.
  pub vector: &'a [u8],
  /// The `stored` part of the record (text fields and payload).
  pub stored: &'a [u8],
}

impl DocRef<'_> {
  /// Decodes the vector into `out` (cleared first).
  pub fn vector_into(&self, out: &mut Vec<f32>) {
    out.clear();
    out.extend(
      self
        .vector
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
    );
  }
}

/// A decoded WAL record.
pub(crate) enum Op<'a> {
  Upsert(Vec<DocRef<'a>>),
  Delete(Vec<Id>),
}

/// Decodes a WAL record, checking it against `schema` so that applying
/// it cannot fail on bad input.
pub(crate) fn decode<'a>(rec: &'a [u8], schema: &Schema) -> Result<Op<'a>> {
  let mut r = Reader::new(rec);
  let format = r.u8()?;
  if format != WAL_FORMAT {
    return Err(Error::corrupt(format!(
      "unsupported WAL record format {format}"
    )));
  }
  let op = r.u8()?;
  let count = r.varint()?;
  // Every entry takes at least 2 bytes; refuses absurd counts before
  // allocating.
  if count > rec.len() as u64 {
    return Err(short());
  }
  let dim = schema.vector.as_ref().map_or(0, |v| v.dim);
  let mut seen = vec![u64::MAX; schema.text_fields.len()];
  let out = match op {
    OP_UPSERT => {
      let mut docs = Vec::with_capacity(count as usize);
      for doc in 0..count {
        let id = r.id()?;
        let d = r.varint()?;
        if d as usize != dim {
          return Err(Error::corrupt(format!(
            "WAL vector has {d} dimensions, schema has {dim}"
          )));
        }
        let vector = r.bytes(dim * 4)?;
        let start = r.pos;
        let nfields = r.varint()?;
        for _ in 0..nfields {
          let f = r.varint()?;
          let slot = seen
            .get_mut(f as usize)
            .ok_or_else(|| Error::corrupt(format!("unknown text field {f}")))?;
          if *slot == doc {
            return Err(Error::corrupt(format!("text field {f} repeated")));
          }
          *slot = doc;
          r.str()?;
        }
        let plen = r.len()?;
        r.bytes(plen)?;
        docs.push(DocRef {
          id,
          vector,
          stored: &rec[start..r.pos],
        });
      }
      Op::Upsert(docs)
    }
    OP_DELETE => {
      let mut ids = Vec::with_capacity(count as usize);
      for _ in 0..count {
        ids.push(r.id()?);
      }
      Op::Delete(ids)
    }
    op => return Err(Error::corrupt(format!("unknown WAL op {op}"))),
  };
  if !r.is_empty() {
    return Err(Error::corrupt("trailing bytes after WAL record"));
  }
  Ok(out)
}

/// A stored record: text fields and payload of one document.
pub(crate) struct Stored<'a> {
  pub text: Vec<(usize, &'a str)>,
  pub payload: &'a [u8],
}

/// Decodes a stored record (format byte included).
pub(crate) fn decode_stored(b: &[u8]) -> Result<Stored<'_>> {
  let mut r = Reader::new(b);
  if r.u8()? != STORED_FORMAT {
    return Err(Error::corrupt("unknown stored record format"));
  }
  let n = r.varint()?;
  let mut text = Vec::new();
  for _ in 0..n {
    let f = r.varint()? as usize;
    text.push((f, r.str()?));
  }
  let plen = r.len()?;
  let payload = r.bytes(plen)?;
  if !r.rest().is_empty() {
    return Err(Error::corrupt("trailing bytes after stored record"));
  }
  Ok(Stored { text, payload })
}

/// Parses a stored payload; empty means `Null`.
pub(crate) fn parse_payload(b: &[u8]) -> Result<serde_json::Value> {
  if b.is_empty() {
    return Ok(serde_json::Value::Null);
  }
  serde_json::from_slice(b)
    .map_err(|e| Error::corrupt(format!("stored payload: {e}")))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::schema::VectorField;
  use muninn_core::Metric;
  use serde_json::json;

  fn schema() -> Schema {
    Schema::vector(VectorField::flat(3, Metric::L2))
      .with_text_fields(["title", "body"])
  }

  #[test]
  fn upsert_round_trip() {
    let s = schema();
    let docs = vec![
      Document::new(7u64)
        .vector(vec![1.0, -2.5, 3.0])
        .text("body", "hello world")
        .payload(json!({"a": 1})),
      Document::new("x").vector(vec![0.0; 3]),
    ];
    let rec = encode_upsert(&s, &docs).unwrap();
    let Op::Upsert(out) = decode(&rec, &s).unwrap() else {
      panic!("not an upsert")
    };
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].id, Id::U64(7));
    let mut v = Vec::new();
    out[0].vector_into(&mut v);
    assert_eq!(v, vec![1.0, -2.5, 3.0]);
    let mut st = vec![STORED_FORMAT];
    st.extend_from_slice(out[0].stored);
    let d = decode_stored(&st).unwrap();
    assert_eq!(d.text, vec![(1, "hello world")]);
    assert_eq!(parse_payload(d.payload).unwrap(), json!({"a": 1}));
    assert_eq!(out[1].id, Id::Str("x".into()));
  }

  #[test]
  fn delete_round_trip_and_rejects_damage() {
    let s = schema();
    let rec = encode_delete(&[Id::U64(1), Id::Str("é".into())]).unwrap();
    let Op::Delete(ids) = decode(&rec, &s).unwrap() else {
      panic!("not a delete")
    };
    assert_eq!(ids, vec![Id::U64(1), Id::Str("é".into())]);
    for cut in 0..rec.len() {
      assert!(decode(&rec[..cut], &s).is_err(), "cut at {cut}");
    }
    let mut extra = rec.clone();
    extra.push(0);
    assert!(decode(&extra, &s).is_err());
  }

  #[test]
  fn validation_rejects_bad_documents() {
    let s = schema();
    let bad = [
      Document::new(1u64),
      Document::new(1u64).vector(vec![1.0; 2]),
      Document::new(1u64).vector(vec![f32::NAN, 0.0, 0.0]),
      Document::new(1u64).vector(vec![0.0; 3]).text("nope", "x"),
    ];
    for d in bad {
      assert!(encode_upsert(&s, &[d]).is_err());
    }
    let t = Schema::text(["a"]);
    assert!(
      encode_upsert(&t, &[Document::new(1u64).vector(vec![1.0])]).is_err()
    );
  }

  #[test]
  fn depth_and_norm_limits() {
    let mut v = serde_json::json!(1);
    for _ in 0..MAX_PAYLOAD_DEPTH {
      v = serde_json::json!({"a": [v]}).get("a").cloned().unwrap();
    }
    assert_eq!(json_depth(&v), MAX_PAYLOAD_DEPTH);
    assert_eq!(json_depth(&serde_json::json!({"a": {"b": [1]}})), 3);
    assert_eq!(json_depth(&serde_json::json!(3)), 0);
    check_vector(&[1e18; 4], "v").unwrap();
    assert!(check_vector(&[1e19; 4], "v").is_err());
    assert!(check_vector(&[f32::INFINITY], "v").is_err());
  }

  #[test]
  fn varints() {
    for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
      let mut b = Vec::new();
      put_varint(&mut b, v);
      assert_eq!(Reader::new(&b).varint().unwrap(), v);
    }
  }
}
