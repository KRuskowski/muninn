//! The immutable payload index of a segment, read in place from a
//! [`Region`].

use std::borrow::Cow;
use std::cmp::Ordering;
use std::io::Write;

use muninn_core::container::{Container, tag};
use muninn_core::{BitSet, DocId, Error, Region, Result};

use crate::bits::Bits;
use crate::plan::{self, Pred, Source};
use crate::value::f64_key;
use crate::write::{
  KIND, Posting, RECORD, TermVisitor, VERSION, WriteSource, field_tag,
  write_index,
};
use crate::{FieldType, Filter, Schema};

/// Marks a document as deleted in a merge remap.
pub const DELETED: DocId = DocId::MAX;

#[derive(Clone, Debug)]
enum Column {
  Keyword {
    /// Term bytes, concatenated in sorted order.
    terms: Region,
    /// `u64` offsets into `terms`, one more than the term count.
    offs: Region,
    /// `u64` posting descriptor pairs, one per term.
    descs: Region,
  },
  Integer {
    vals: Region,
    docs: Region,
  },
  Float {
    vals: Region,
    docs: Region,
  },
  Bool([Posting; 2]),
}

#[derive(Clone, Debug)]
struct FrozenField {
  post: Region,
  exists: Posting,
  column: Column,
}

/// Payload index of a segment: written once (from a [`crate::MemIndex`]
/// or by [`FrozenIndex::merge`]) and opened zero-copy from a [`Region`].
///
/// Keyword postings are sorted `u32` id lists, or bitsets when that is
/// smaller (more than one document in 32). Numeric fields store their
/// values sorted, next to the document of each value, so a range is two
/// binary searches and one pass over the matching ids.
#[derive(Clone, Debug)]
pub struct FrozenIndex {
  container: Container,
  schema: Schema,
  doc_count: usize,
  fields: Vec<FrozenField>,
}

/// A slice cast that was validated when the index was opened.
fn u32s(r: &Region) -> &[u32] {
  r.cast().unwrap_or(&[])
}

fn u64s(r: &Region) -> &[u64] {
  r.cast().unwrap_or(&[])
}

fn corrupt(i: usize, what: &str) -> Error {
  Error::corrupt(format!("payload index field {i}: {what}"))
}

impl FrozenIndex {
  /// Opens an index written by [`crate::MemIndex::write_to`] or
  /// [`FrozenIndex::merge`]. Checks the structure and every offset, so
  /// later reads cannot go out of bounds; call
  /// [`FrozenIndex::verify_checksum`] to also detect flipped bits.
  pub fn open(region: Region) -> Result<Self> {
    let container = Container::open_kind(region, KIND, VERSION)?;
    let meta = container.section(tag("meta"))?;
    let meta = meta.cast::<u64>()?;
    if meta.len() < 2 {
      return Err(Error::corrupt("payload index meta too short"));
    }
    let doc_count = usize::try_from(meta[0])
      .ok()
      .filter(|&n| n <= u32::MAX as usize)
      .ok_or_else(|| Error::corrupt("payload index doc count too large"))?;
    let nfields = meta[1] as usize;
    let table = container.section(tag("fields"))?;
    let table = table.cast::<u64>()?;
    if nfields.checked_mul(RECORD) != Some(table.len()) {
      return Err(Error::corrupt("payload index field table size mismatch"));
    }
    let names = container.section(tag("names"))?;
    let bits_len = 2 * doc_count.div_ceil(64);
    let mut schema = Schema::new();
    let mut fields = Vec::with_capacity(nfields);
    for (i, rec) in table.chunks_exact(RECORD).enumerate() {
      let ty = FieldType::from_code(rec[0])
        .ok_or_else(|| corrupt(i, "unknown field type"))?;
      let name = usize::try_from(rec[1])
        .ok()
        .zip(usize::try_from(rec[2]).ok())
        .and_then(|(o, l)| names.bytes().get(o..o.checked_add(l)?))
        .and_then(|b| std::str::from_utf8(b).ok())
        .ok_or_else(|| corrupt(i, "bad field name"))?;
      schema
        .add(name, ty)
        .map_err(|e| corrupt(i, &e.to_string()))?;
      let post = container.section(field_tag(b'P', i))?;
      let plen = post.cast::<u32>()?.len();
      let posting = |a: u64, count: u64| -> Result<Posting> {
        let bits = a & (1 << 63) != 0;
        let off = (a & !(1 << 63)) as usize;
        let count = count as usize;
        let len = if bits { bits_len } else { count };
        let fits = off.checked_add(len).is_some_and(|e| e <= plen);
        if !fits || count > doc_count || (bits && off % 2 == 1) {
          return Err(corrupt(i, "posting out of bounds"));
        }
        Ok(Posting { bits, off, count })
      };
      let exists = posting(rec[3], rec[4])?;
      let column = match ty {
        FieldType::Keyword => {
          let terms = container.section(field_tag(b'T', i))?;
          let offs = container.section(field_tag(b'O', i))?;
          let descs = container.section(field_tag(b'D', i))?;
          let n = rec[9] as usize;
          let o = offs.cast::<u64>()?;
          let d = descs.cast::<u64>()?;
          if o.len() != n.saturating_add(1) || d.len() != 2 * n {
            return Err(corrupt(i, "term table size mismatch"));
          }
          if o[0] != 0 || o[n] != terms.len() as u64 {
            return Err(corrupt(i, "term offsets out of bounds"));
          }
          let bytes = terms.bytes();
          let mut prev: Option<&[u8]> = None;
          for w in o.windows(2) {
            let t = usize::try_from(w[0])
              .ok()
              .zip(usize::try_from(w[1]).ok())
              .and_then(|(a, b)| bytes.get(a..b))
              .ok_or_else(|| corrupt(i, "term offsets out of order"))?;
            if prev.is_some_and(|p| p >= t) {
              return Err(corrupt(i, "terms not sorted"));
            }
            prev = Some(t);
          }
          for pair in d.chunks_exact(2) {
            posting(pair[0], pair[1])?;
          }
          Column::Keyword { terms, offs, descs }
        }
        FieldType::Integer | FieldType::Float => {
          let vals = container.section(field_tag(b'V', i))?;
          let docs = container.section(field_tag(b'I', i))?;
          let n = rec[9] as usize;
          let nv = if ty == FieldType::Integer {
            vals.cast::<i64>()?.len()
          } else {
            vals.cast::<f64>()?.len()
          };
          if nv != n || docs.cast::<u32>()?.len() != n {
            return Err(corrupt(i, "value column size mismatch"));
          }
          if ty == FieldType::Integer {
            Column::Integer { vals, docs }
          } else {
            Column::Float { vals, docs }
          }
        }
        FieldType::Bool => {
          Column::Bool([posting(rec[5], rec[6])?, posting(rec[7], rec[8])?])
        }
      };
      fields.push(FrozenField {
        post,
        exists,
        column,
      });
    }
    Ok(Self {
      container,
      schema,
      doc_count,
      fields,
    })
  }

  /// Reads the whole file and checks its CRC-32C.
  pub fn verify_checksum(&self) -> Result<()> {
    self.container.verify_checksum()
  }

  /// The declared fields, as stored in the file.
  pub fn schema(&self) -> &Schema {
    &self.schema
  }

  /// One past the largest document id the index holds values for.
  pub fn doc_count(&self) -> usize {
    self.doc_count
  }

  /// Ids `0..num_docs` matching `filter`. Fails with `InvalidArgument` if
  /// the filter uses an undeclared field or a value of the wrong type.
  pub fn evaluate(&self, filter: &Filter, num_docs: usize) -> Result<BitSet> {
    plan::evaluate(self, filter, num_docs)
  }

  /// Estimated fraction of `0..num_docs` matching `filter`, from posting
  /// lengths and range counts, without evaluating it. Treats clauses as
  /// independent. Returns 1.0 for a filter `evaluate` would reject.
  pub fn estimate(&self, filter: &Filter, num_docs: usize) -> f32 {
    plan::estimate(self, filter, num_docs)
  }

  /// Merges several frozen indexes into one, written in one streaming
  /// pass to `out`.
  ///
  /// Each input comes with a remap: `remap[old_id]` is the document's id
  /// in the merged index, or [`DELETED`] to drop it. A remap must cover
  /// every id of its input (`len() >= doc_count()`). The merged index
  /// declares `schema`'s fields; an input lacking a field contributes no
  /// values to it, and a field declared with a different type is an
  /// error. The merged document count is one past the largest new id.
  pub fn merge<W: Write>(
    schema: &Schema,
    inputs: &[(&FrozenIndex, &[DocId])],
    out: W,
  ) -> Result<W> {
    let mut maps = Vec::with_capacity(inputs.len());
    let mut doc_count = 0usize;
    for (k, (input, remap)) in inputs.iter().enumerate() {
      if remap.len() < input.doc_count {
        return Err(Error::invalid(format!(
          "merge input {k}: remap covers {} ids, index has {}",
          remap.len(),
          input.doc_count
        )));
      }
      if let Some(max) = remap.iter().copied().filter(|&d| d != DELETED).max() {
        doc_count = doc_count.max(max as usize + 1);
      }
      let mut map = Vec::with_capacity(schema.len());
      for f in schema.fields() {
        map.push(match input.schema.index_of(f.name()) {
          None => None,
          Some(j) => {
            let have = input.schema.fields()[j].field_type();
            if have != f.field_type() {
              return Err(Error::invalid(format!(
                "merge input {k}: field {:?} is {}, expected {}",
                f.name(),
                have.name(),
                f.field_type().name()
              )));
            }
            Some(j)
          }
        });
      }
      maps.push(map);
    }
    let src = Merge {
      schema,
      inputs,
      maps,
      doc_count,
    };
    write_index(&src, out)
  }

  fn list(&self, field: usize, p: Posting) -> &[u32] {
    let post = u32s(&self.fields[field].post);
    let len = if p.bits {
      2 * self.doc_count.div_ceil(64)
    } else {
      p.count
    };
    post.get(p.off..p.off + len).unwrap_or(&[])
  }

  fn fill_posting(&self, field: usize, p: Posting, bits: &mut Bits) {
    let data = self.list(field, p);
    if p.bits {
      bits.or_words32(data);
    } else {
      bits.set_sorted(data);
    }
  }

  fn for_each_doc(&self, field: usize, p: Posting, mut f: impl FnMut(DocId)) {
    let data = self.list(field, p);
    if p.bits {
      for (wi, &w) in data.iter().enumerate() {
        let mut w = w;
        while w != 0 {
          f(wi as u32 * 32 + w.trailing_zeros());
          w &= w - 1;
        }
      }
    } else {
      data.iter().copied().for_each(f);
    }
  }

  /// Number of terms of a keyword field (0 for other types).
  fn term_count(&self, field: usize) -> usize {
    match &self.fields[field].column {
      Column::Keyword { offs, .. } => u64s(offs).len().saturating_sub(1),
      _ => 0,
    }
  }

  fn term(&self, field: usize, t: usize) -> &[u8] {
    match &self.fields[field].column {
      Column::Keyword { terms, offs, .. } => {
        let o = u64s(offs);
        terms
          .bytes()
          .get(o[t] as usize..o[t + 1] as usize)
          .unwrap_or(&[])
      }
      _ => &[],
    }
  }

  fn term_posting(&self, field: usize, t: usize) -> Posting {
    match &self.fields[field].column {
      Column::Keyword { descs, .. } => {
        let d = u64s(descs);
        decode(d[2 * t], d[2 * t + 1])
      }
      _ => Posting {
        bits: false,
        off: 0,
        count: 0,
      },
    }
  }

  fn find_term(&self, field: usize, term: &[u8]) -> Option<Posting> {
    let (mut lo, mut hi) = (0, self.term_count(field));
    while lo < hi {
      let mid = lo + (hi - lo) / 2;
      match self.term(field, mid).cmp(term) {
        Ordering::Less => lo = mid + 1,
        Ordering::Greater => hi = mid,
        Ordering::Equal => return Some(self.term_posting(field, mid)),
      }
    }
    None
  }

  /// The doc ids of entries with `lo <= value <= hi`.
  fn int_docs(&self, field: usize, lo: i64, hi: i64) -> &[u32] {
    match &self.fields[field].column {
      Column::Integer { vals, docs } => {
        let v = vals.cast::<i64>().unwrap_or(&[]);
        let s = v.partition_point(|&x| x < lo);
        let e = v.partition_point(|&x| x <= hi);
        u32s(docs).get(s..e.max(s)).unwrap_or(&[])
      }
      _ => &[],
    }
  }

  fn float_docs(&self, field: usize, lo: f64, hi: f64) -> &[u32] {
    match &self.fields[field].column {
      Column::Float { vals, docs } => {
        let v = vals.cast::<f64>().unwrap_or(&[]);
        let s = v.partition_point(|&x| x < lo);
        let e = v.partition_point(|&x| x <= hi);
        u32s(docs).get(s..e.max(s)).unwrap_or(&[])
      }
      _ => &[],
    }
  }

  /// Calls `f` with every (value key, doc) entry of a numeric field.
  fn for_each_number(&self, field: usize, mut f: impl FnMut(i64, DocId)) {
    match &self.fields[field].column {
      Column::Integer { vals, docs } => {
        let v = vals.cast::<i64>().unwrap_or(&[]);
        for (&x, &d) in v.iter().zip(u32s(docs)) {
          f(x, d);
        }
      }
      Column::Float { vals, docs } => {
        let v = vals.cast::<f64>().unwrap_or(&[]);
        for (&x, &d) in v.iter().zip(u32s(docs)) {
          f(f64_key(x), d);
        }
      }
      _ => {}
    }
  }
}

fn decode(a: u64, count: u64) -> Posting {
  Posting {
    bits: a & (1 << 63) != 0,
    off: (a & !(1 << 63)) as usize,
    count: count as usize,
  }
}

impl Source for FrozenIndex {
  fn schema(&self) -> &Schema {
    &self.schema
  }

  fn count(&self, field: usize, pred: &Pred) -> u64 {
    let f = &self.fields[field];
    let n = match (pred, &f.column) {
      (Pred::Exists, _) => f.exists.count,
      (Pred::Terms(ts), Column::Keyword { .. }) => ts
        .iter()
        .filter_map(|t| self.find_term(field, t.as_bytes()))
        .map(|p| p.count)
        .sum(),
      (Pred::Ints(vs), _) => {
        vs.iter().map(|&v| self.int_docs(field, v, v).len()).sum()
      }
      (Pred::IntRange(lo, hi), _) => self.int_docs(field, *lo, *hi).len(),
      (Pred::Floats(vs), _) => {
        vs.iter().map(|&v| self.float_docs(field, v, v).len()).sum()
      }
      (Pred::FloatRange(lo, hi), _) => self.float_docs(field, *lo, *hi).len(),
      (Pred::Bools(w), Column::Bool(ps)) => {
        (0..2).filter(|&b| w[b]).map(|b| ps[b].count).sum()
      }
      _ => 0,
    };
    n as u64
  }

  fn fill(&self, field: usize, pred: &Pred, bits: &mut Bits) {
    let f = &self.fields[field];
    match (pred, &f.column) {
      (Pred::Exists, _) => self.fill_posting(field, f.exists, bits),
      (Pred::Terms(ts), Column::Keyword { .. }) => {
        for t in ts {
          if let Some(p) = self.find_term(field, t.as_bytes()) {
            self.fill_posting(field, p, bits);
          }
        }
      }
      (Pred::Ints(vs), _) => {
        for &v in vs {
          for &d in self.int_docs(field, v, v) {
            bits.set(d);
          }
        }
      }
      (Pred::IntRange(lo, hi), _) => {
        for &d in self.int_docs(field, *lo, *hi) {
          bits.set(d);
        }
      }
      (Pred::Floats(vs), _) => {
        for &v in vs {
          for &d in self.float_docs(field, v, v) {
            bits.set(d);
          }
        }
      }
      (Pred::FloatRange(lo, hi), _) => {
        for &d in self.float_docs(field, *lo, *hi) {
          bits.set(d);
        }
      }
      (Pred::Bools(w), Column::Bool(ps)) => {
        for b in 0..2 {
          if w[b] {
            self.fill_posting(field, ps[b], bits);
          }
        }
      }
      _ => {}
    }
  }
}

/// A merge of frozen indexes, as a source for the writer.
struct Merge<'a> {
  schema: &'a Schema,
  inputs: &'a [(&'a FrozenIndex, &'a [DocId])],
  /// Per input, per output field: the input's field index.
  maps: Vec<Vec<Option<usize>>>,
  doc_count: usize,
}

#[inline]
fn remap(map: &[DocId], d: DocId) -> Option<DocId> {
  map.get(d as usize).copied().filter(|&m| m != DELETED)
}

/// Sorts (only if needed) and dedups gathered ids.
fn finish(mut v: Vec<DocId>) -> Vec<DocId> {
  if !v.is_sorted() {
    v.sort_unstable();
  }
  v.dedup();
  v
}

impl Merge<'_> {
  /// The remapped docs of one posting per input, merged.
  fn gather(
    &self,
    field: usize,
    pick: impl Fn(&FrozenIndex, usize) -> Option<Posting>,
  ) -> Vec<DocId> {
    let mut out = Vec::new();
    for ((input, map), fmap) in self.inputs.iter().zip(&self.maps) {
      if let Some(j) = fmap[field]
        && let Some(p) = pick(input, j)
      {
        input.for_each_doc(j, p, |d| out.extend(remap(map, d)));
      }
    }
    finish(out)
  }
}

impl WriteSource for Merge<'_> {
  fn schema(&self) -> &Schema {
    self.schema
  }

  fn doc_count(&self) -> usize {
    self.doc_count
  }

  fn exists(&self, field: usize) -> Result<Cow<'_, [DocId]>> {
    Ok(Cow::Owned(
      self.gather(field, |ix, j| Some(ix.fields[j].exists)),
    ))
  }

  fn bools(&self, field: usize, value: bool) -> Result<Cow<'_, [DocId]>> {
    Ok(Cow::Owned(self.gather(
      field,
      |ix, j| match &ix.fields[j].column {
        Column::Bool(ps) => Some(ps[usize::from(value)]),
        _ => None,
      },
    )))
  }

  fn terms(&self, field: usize, visit: &mut TermVisitor<'_>) -> Result<()> {
    // One cursor per input that has the field; repeatedly take the
    // smallest current term across inputs.
    let mut cursors: Vec<(usize, usize, usize)> = Vec::new();
    for (k, fmap) in self.maps.iter().enumerate() {
      if let Some(j) = fmap[field] {
        cursors.push((k, j, 0));
      }
    }
    let mut docs = Vec::new();
    loop {
      let mut min: Option<&[u8]> = None;
      for &(k, j, t) in &cursors {
        let ix = self.inputs[k].0;
        if t < ix.term_count(j) {
          let term = ix.term(j, t);
          if min.is_none_or(|m| term < m) {
            min = Some(term);
          }
        }
      }
      let Some(term) = min else {
        return Ok(());
      };
      docs.clear();
      for (k, j, t) in &mut cursors {
        let (ix, map) = self.inputs[*k];
        if *t < ix.term_count(*j) && ix.term(*j, *t) == term {
          ix.for_each_doc(*j, ix.term_posting(*j, *t), |d| {
            docs.extend(remap(map, d))
          });
          *t += 1;
        }
      }
      let merged = finish(std::mem::take(&mut docs));
      if !merged.is_empty() {
        visit(term, &merged)?;
      }
      docs = merged;
    }
  }

  fn numbers(&self, field: usize) -> Result<Vec<(i64, DocId)>> {
    let mut out = Vec::new();
    for ((input, map), fmap) in self.inputs.iter().zip(&self.maps) {
      if let Some(j) = fmap[field] {
        input.for_each_number(j, |k, d| {
          if let Some(nd) = remap(map, d) {
            out.push((k, nd));
          }
        });
      }
    }
    if !out.is_sorted() {
      out.sort_unstable();
    }
    out.dedup();
    Ok(out)
  }
}
