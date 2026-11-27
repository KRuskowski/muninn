//! The mutable payload index of a memtable.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Write;

use muninn_core::{BitSet, DocId, Error, Result};
use serde_json::Value as Json;

use crate::bits::Bits;
use crate::plan::{self, Pred, Source};
use crate::schema::{Scalar, extract};
use crate::value::f64_key;
use crate::write::{TermVisitor, WriteSource, write_index};
use crate::{FieldType, Filter, Schema};

/// Unsorted entries collected before they are sorted into a run.
const TAIL: usize = 1024;

/// `(key, doc)` entries of one numeric field kept as a few sorted runs
/// plus a short unsorted tail (the logarithmic method): an insert costs
/// amortised O(log n), and a range query costs one binary search per run
/// (at most ~log2(n / 1024) runs) plus a scan of the tail.
#[derive(Debug, Default)]
pub(crate) struct Runs {
  runs: Vec<Vec<(i64, DocId)>>,
  tail: Vec<(i64, DocId)>,
}

impl Runs {
  fn push(&mut self, key: i64, doc: DocId) {
    self.tail.push((key, doc));
    if self.tail.len() >= TAIL {
      let mut run = std::mem::take(&mut self.tail);
      run.sort_unstable();
      self.runs.push(run);
      // Merge while the newest run is at least as large as the one
      // before it, so run sizes stay strictly decreasing.
      while self.runs.len() >= 2 {
        let n = self.runs.len();
        if self.runs[n - 2].len() > self.runs[n - 1].len() {
          break;
        }
        let b = self.runs.pop().unwrap();
        let a = self.runs.pop().unwrap();
        self.runs.push(merge_sorted(&a, &b));
      }
    }
  }

  /// Entries with `lo <= key <= hi`, run by run.
  fn for_each_slice(&self, lo: i64, hi: i64, mut f: impl FnMut(&[(i64, u32)])) {
    for run in &self.runs {
      let s = run.partition_point(|e| e.0 < lo);
      let e = run.partition_point(|e| e.0 <= hi);
      if s < e {
        f(&run[s..e]);
      }
    }
  }

  fn count(&self, lo: i64, hi: i64) -> u64 {
    let mut n = 0;
    self.for_each_slice(lo, hi, |s| n += s.len() as u64);
    n + self
      .tail
      .iter()
      .filter(|e| (lo..=hi).contains(&e.0))
      .count() as u64
  }

  fn fill(&self, lo: i64, hi: i64, bits: &mut Bits) {
    self.for_each_slice(lo, hi, |s| {
      for &(_, d) in s {
        bits.set(d);
      }
    });
    for &(k, d) in &self.tail {
      if (lo..=hi).contains(&k) {
        bits.set(d);
      }
    }
  }

  /// Entries whose key is in `keys` (sorted, unique): per run either one
  /// merge pass (many keys) or a narrowing binary search per key (few
  /// keys), then one pass over the tail. Never one tail scan per key.
  fn for_each_in_set(&self, keys: &[i64], mut f: impl FnMut(DocId)) {
    for run in &self.runs {
      if keys.len().saturating_mul(8) >= run.len() {
        let (mut i, mut j) = (0, 0);
        while i < run.len() && j < keys.len() {
          match run[i].0.cmp(&keys[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
              f(run[i].1);
              i += 1;
            }
          }
        }
      } else {
        let mut lo = 0;
        for &k in keys {
          let rest = &run[lo..];
          let s = rest.partition_point(|e| e.0 < k);
          let e = rest.partition_point(|e| e.0 <= k);
          for &(_, d) in &rest[s..e] {
            f(d);
          }
          lo += e;
        }
      }
    }
    for &(k, d) in &self.tail {
      if keys.binary_search(&k).is_ok() {
        f(d);
      }
    }
  }

  fn count_in_set(&self, keys: &[i64]) -> u64 {
    let mut n = 0;
    self.for_each_in_set(keys, |_| n += 1);
    n
  }

  fn fill_in_set(&self, keys: &[i64], bits: &mut Bits) {
    self.for_each_in_set(keys, |d| bits.set(d));
  }

  /// Every entry, sorted by `(key, doc)`.
  fn sorted(&self) -> Vec<(i64, DocId)> {
    let mut all: Vec<(i64, DocId)> = Vec::with_capacity(
      self.runs.iter().map(Vec::len).sum::<usize>() + self.tail.len(),
    );
    for run in &self.runs {
      all.extend_from_slice(run);
    }
    all.extend_from_slice(&self.tail);
    all.sort_unstable();
    all
  }
}

/// Keys of floats sorted in total order (as `Pred::Floats` is), which
/// come out sorted too.
fn float_keys(vs: &[f64]) -> Vec<i64> {
  vs.iter().map(|&v| f64_key(v)).collect()
}

fn merge_sorted(a: &[(i64, DocId)], b: &[(i64, DocId)]) -> Vec<(i64, DocId)> {
  let mut out = Vec::with_capacity(a.len() + b.len());
  let (mut i, mut j) = (0, 0);
  while i < a.len() && j < b.len() {
    if a[i] <= b[j] {
      out.push(a[i]);
      i += 1;
    } else {
      out.push(b[j]);
      j += 1;
    }
  }
  out.extend_from_slice(&a[i..]);
  out.extend_from_slice(&b[j..]);
  out
}

/// Postings shorter than this always stay lists.
const MIN_BITSET: usize = 1024;

/// The docs of one term, bool value or exists set. Starts as a sorted id
/// list and turns into a growable bitset once it holds more than one id
/// in 32, so dense postings evaluate as a word-wise OR.
#[derive(Debug)]
enum Posting {
  List(Vec<DocId>),
  Bits { words: Vec<u64>, count: usize },
}

impl Default for Posting {
  fn default() -> Self {
    Self::List(Vec::new())
  }
}

impl Posting {
  /// Adds `doc`, which is at least every id added before.
  fn push(&mut self, doc: DocId) {
    let i = doc as usize;
    match self {
      Self::List(v) => {
        if v.last() == Some(&doc) {
          return;
        }
        v.push(doc);
        if v.len() >= MIN_BITSET && v.len() * 32 > i + 1 {
          let mut words = vec![0u64; (i >> 6) + 1];
          for &d in v.iter() {
            words[(d >> 6) as usize] |= 1 << (d & 63);
          }
          *self = Self::Bits {
            words,
            count: v.len(),
          };
        }
      }
      Self::Bits { words, count } => {
        if words.len() <= i >> 6 {
          words.resize((i >> 6) + 1, 0);
        }
        let bit = 1u64 << (i & 63);
        if words[i >> 6] & bit == 0 {
          words[i >> 6] |= bit;
          *count += 1;
        }
      }
    }
  }

  fn len(&self) -> usize {
    match self {
      Self::List(v) => v.len(),
      Self::Bits { count, .. } => *count,
    }
  }

  fn fill(&self, bits: &mut Bits) {
    match self {
      Self::List(v) => bits.set_sorted(v),
      Self::Bits { words, .. } => bits.or_words64(words),
    }
  }

  /// The ids, sorted.
  fn docs(&self) -> Cow<'_, [DocId]> {
    match self {
      Self::List(v) => Cow::Borrowed(v),
      Self::Bits { words, count } => {
        let mut out = Vec::with_capacity(*count);
        for (wi, &w) in words.iter().enumerate() {
          let mut w = w;
          while w != 0 {
            out.push(wi as u32 * 64 + w.trailing_zeros());
            w &= w - 1;
          }
        }
        Cow::Owned(out)
      }
    }
  }
}

#[derive(Debug)]
enum Column {
  /// Term to its docs.
  Keyword(HashMap<Box<str>, Posting>),
  /// Integer values, or floats mapped through `f64_key`.
  Number(Runs),
  /// Docs holding `false` (0) and `true` (1).
  Bool([Posting; 2]),
}

#[derive(Debug)]
struct MemField {
  /// Docs holding at least one value of the field's type.
  exists: Posting,
  column: Column,
}

/// Payload index of a memtable: documents are added one at a time, with
/// increasing ids, and are filterable immediately.
///
/// Write it out as a frozen index with [`MemIndex::write_to`] when the
/// memtable is flushed.
#[derive(Debug)]
pub struct MemIndex {
  schema: Schema,
  fields: Vec<MemField>,
  num_docs: usize,
  scratch: Vec<i64>,
}

impl MemIndex {
  /// An empty index over `schema`'s fields.
  pub fn new(schema: Schema) -> Self {
    let fields = schema
      .fields()
      .iter()
      .map(|f| MemField {
        exists: Posting::default(),
        column: match f.field_type() {
          FieldType::Keyword => Column::Keyword(HashMap::new()),
          FieldType::Integer | FieldType::Float => {
            Column::Number(Runs::default())
          }
          FieldType::Bool => Column::Bool(Default::default()),
        },
      })
      .collect();
    Self {
      schema,
      fields,
      num_docs: 0,
      scratch: Vec::new(),
    }
  }

  /// The declared fields.
  pub fn schema(&self) -> &Schema {
    &self.schema
  }

  /// One past the largest id added so far (0 when empty).
  pub fn num_docs(&self) -> usize {
    self.num_docs
  }

  /// Indexes the filterable fields of `payload` for document `doc`.
  ///
  /// Ids must be strictly increasing; gaps are fine (a skipped id simply
  /// has no values). Values of the wrong type are ignored. `DocId::MAX`
  /// is reserved (it marks deleted documents in a merge remap).
  pub fn add(&mut self, doc: DocId, payload: &Json) -> Result<()> {
    if doc == DocId::MAX {
      return Err(Error::invalid("doc id u32::MAX is reserved"));
    }
    if (doc as usize) < self.num_docs {
      return Err(Error::invalid(format!(
        "doc ids must increase: got {doc} after {}",
        self.num_docs - 1
      )));
    }
    let Self {
      schema,
      fields,
      scratch,
      ..
    } = self;
    for (def, field) in schema.fields().iter().zip(fields.iter_mut()) {
      let mut any = false;
      match &mut field.column {
        Column::Keyword(map) => extract(def, payload, &mut |s| {
          if let Scalar::Str(t) = s {
            any = true;
            match map.get_mut(t) {
              Some(posting) => posting.push(doc),
              None => {
                map.insert(t.into(), Posting::List(vec![doc]));
              }
            }
          }
        }),
        Column::Number(runs) => {
          scratch.clear();
          extract(def, payload, &mut |s| match s {
            Scalar::Int(i) => scratch.push(i),
            Scalar::Float(f) => scratch.push(f64_key(f)),
            _ => {}
          });
          scratch.sort_unstable();
          scratch.dedup();
          any = !scratch.is_empty();
          for &k in scratch.iter() {
            runs.push(k, doc);
          }
        }
        Column::Bool(lists) => {
          let mut seen = [false; 2];
          extract(def, payload, &mut |s| {
            if let Scalar::Bool(b) = s {
              seen[usize::from(b)] = true;
            }
          });
          for (list, seen) in lists.iter_mut().zip(seen) {
            if seen {
              list.push(doc);
              any = true;
            }
          }
        }
      }
      if any {
        field.exists.push(doc);
      }
    }
    self.num_docs = doc as usize + 1;
    Ok(())
  }

  /// Ids `0..num_docs` matching `filter`. Ids added at or beyond
  /// `num_docs` are left out. Fails with `InvalidArgument` if the filter
  /// uses an undeclared field or a value of the wrong type.
  pub fn evaluate(&self, filter: &Filter, num_docs: usize) -> Result<BitSet> {
    plan::evaluate(self, filter, num_docs)
  }

  /// Estimated fraction of `0..num_docs` matching `filter`, from posting
  /// lengths and range counts, without evaluating it. Treats clauses as
  /// independent. Returns 1.0 for a filter `evaluate` would reject.
  pub fn estimate(&self, filter: &Filter, num_docs: usize) -> f32 {
    plan::estimate(self, filter, num_docs)
  }

  /// Writes the frozen form (see [`crate::FrozenIndex`]) in one
  /// streaming pass and returns the writer. Flushing and fsync are the
  /// caller's.
  pub fn write_to<W: Write>(&self, out: W) -> Result<W> {
    write_index(self, out)
  }

  fn keyword(&self, field: usize) -> Option<&HashMap<Box<str>, Posting>> {
    match &self.fields[field].column {
      Column::Keyword(m) => Some(m),
      _ => None,
    }
  }
}

impl Source for MemIndex {
  fn schema(&self) -> &Schema {
    &self.schema
  }

  fn count(&self, field: usize, pred: &Pred) -> u64 {
    let f = &self.fields[field];
    match (pred, &f.column) {
      (Pred::Exists, _) => f.exists.len() as u64,
      (Pred::Terms(ts), Column::Keyword(m)) => ts
        .iter()
        .map(|t| m.get(*t).map_or(0, |l| l.len() as u64))
        .sum(),
      (Pred::Ints(vs), Column::Number(r)) => r.count_in_set(vs),
      (Pred::Floats(vs), Column::Number(r)) => r.count_in_set(&float_keys(vs)),
      (Pred::IntRange(lo, hi), Column::Number(r)) => r.count(*lo, *hi),
      (Pred::FloatRange(lo, hi), Column::Number(r)) => {
        r.count(f64_key(*lo), f64_key(*hi))
      }
      (Pred::Bools(w), Column::Bool(l)) => {
        (0..2).filter(|&b| w[b]).map(|b| l[b].len() as u64).sum()
      }
      _ => 0,
    }
  }

  fn fill(&self, field: usize, pred: &Pred, bits: &mut Bits) {
    let f = &self.fields[field];
    match (pred, &f.column) {
      (Pred::Exists, _) => f.exists.fill(bits),
      (Pred::Terms(ts), Column::Keyword(m)) => {
        for t in ts {
          if let Some(posting) = m.get(*t) {
            posting.fill(bits);
          }
        }
      }
      (Pred::Ints(vs), Column::Number(r)) => r.fill_in_set(vs, bits),
      (Pred::Floats(vs), Column::Number(r)) => {
        r.fill_in_set(&float_keys(vs), bits)
      }
      (Pred::IntRange(lo, hi), Column::Number(r)) => r.fill(*lo, *hi, bits),
      (Pred::FloatRange(lo, hi), Column::Number(r)) => {
        r.fill(f64_key(*lo), f64_key(*hi), bits)
      }
      (Pred::Bools(w), Column::Bool(l)) => {
        for b in 0..2 {
          if w[b] {
            l[b].fill(bits);
          }
        }
      }
      _ => {}
    }
  }
}

impl WriteSource for MemIndex {
  fn schema(&self) -> &Schema {
    &self.schema
  }

  fn doc_count(&self) -> usize {
    self.num_docs
  }

  fn exists(&self, field: usize) -> Result<Cow<'_, [DocId]>> {
    Ok(self.fields[field].exists.docs())
  }

  fn bools(&self, field: usize, value: bool) -> Result<Cow<'_, [DocId]>> {
    Ok(match &self.fields[field].column {
      Column::Bool(l) => l[usize::from(value)].docs(),
      _ => Cow::Borrowed(&[]),
    })
  }

  fn terms(&self, field: usize, visit: &mut TermVisitor<'_>) -> Result<()> {
    let Some(map) = self.keyword(field) else {
      return Ok(());
    };
    let mut terms: Vec<(&str, &Posting)> =
      map.iter().map(|(t, p)| (&**t, p)).collect();
    terms.sort_unstable_by_key(|e| e.0);
    for (t, p) in terms {
      visit(t.as_bytes(), &p.docs())?;
    }
    Ok(())
  }

  fn numbers(&self, field: usize) -> Result<Vec<(i64, DocId)>> {
    Ok(match &self.fields[field].column {
      Column::Number(r) => r.sorted(),
      _ => Vec::new(),
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;

  #[test]
  fn runs_match_a_sorted_vec() {
    let mut r = Runs::default();
    let mut all = Vec::new();
    let mut x = 12345u64;
    for d in 0..10_000u32 {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      let k = (x % 2001) as i64 - 1000;
      r.push(k, d);
      all.push((k, d));
    }
    assert!(r.runs.len() <= 4, "{} runs", r.runs.len());
    all.sort_unstable();
    assert_eq!(r.sorted(), all);
    for keys in [
      vec![],
      vec![-1000, 3, 999],
      (-1000..1000).step_by(3).collect(),
    ] {
      let want: Vec<(i64, u32)> = all
        .iter()
        .copied()
        .filter(|e| keys.contains(&e.0))
        .collect();
      assert_eq!(r.count_in_set(&keys), want.len() as u64);
      let mut b = Bits::new(10_000);
      r.fill_in_set(&keys, &mut b);
      let mut docs: Vec<u32> = want.iter().map(|e| e.1).collect();
      docs.sort_unstable();
      docs.dedup();
      assert_eq!(b.into_bitset().iter().collect::<Vec<_>>(), docs);
    }
    for (lo, hi) in [(-1000, 1000), (0, 0), (-5, 17), (999, 2000)] {
      let want = all.iter().filter(|e| e.0 >= lo && e.0 <= hi).count();
      assert_eq!(r.count(lo, hi), want as u64);
      let mut b = Bits::new(10_000);
      r.fill(lo, hi, &mut b);
      let mut docs: Vec<u32> = all
        .iter()
        .filter(|e| e.0 >= lo && e.0 <= hi)
        .map(|e| e.1)
        .collect();
      docs.sort_unstable();
      docs.dedup();
      assert_eq!(b.into_bitset().iter().collect::<Vec<_>>(), docs);
    }
  }

  #[test]
  fn postings_turn_dense_into_bitsets() {
    let mut p = Posting::default();
    let mut want = Vec::new();
    for d in (0..200_000u32).filter(|d| d % 7 == 0 || d % 1000 == 1) {
      p.push(d);
      p.push(d);
      want.push(d);
    }
    assert!(matches!(p, Posting::Bits { .. }));
    assert_eq!(p.len(), want.len());
    assert_eq!(&*p.docs(), want.as_slice());
    let mut b = Bits::new(1000);
    p.fill(&mut b);
    let got: Vec<u32> = b.into_bitset().iter().collect();
    let cut: Vec<u32> = want.iter().copied().filter(|&d| d < 1000).collect();
    assert_eq!(got, cut);
    // Sparse postings stay lists.
    let mut q = Posting::default();
    for d in (0..2_000_000u32).step_by(100) {
      q.push(d);
    }
    assert!(matches!(q, Posting::List(_)));
  }

  #[test]
  fn ids_must_increase() {
    let schema = Schema::new().with("k", FieldType::Keyword).unwrap();
    let mut m = MemIndex::new(schema);
    m.add(3, &json!({"k": "a"})).unwrap();
    assert!(m.add(3, &json!({"k": "a"})).is_err());
    assert!(m.add(1, &json!({})).is_err());
    assert!(m.add(DocId::MAX, &json!({})).is_err());
    m.add(10, &json!({"k": ["a", "a", "b"]})).unwrap();
    assert_eq!(m.num_docs(), 11);
    let got = m.evaluate(&Filter::eq("k", "a"), 11).unwrap();
    assert_eq!(got.iter().collect::<Vec<_>>(), [3, 10]);
    let got = m.evaluate(&Filter::eq("k", "a"), 5).unwrap();
    assert_eq!(got.iter().collect::<Vec<_>>(), [3]);
    assert_eq!(got.capacity(), 5);
  }
}
