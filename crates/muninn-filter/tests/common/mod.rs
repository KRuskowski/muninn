//! Shared test helpers: a seeded RNG, random payloads and filters, and a
//! brute-force evaluator over raw JSON that the indexes are checked
//! against. The evaluator is written from the documented semantics, not
//! from the index code.

#![allow(dead_code)]

use std::ops::Bound;

use muninn_filter::{FieldType, Filter, Schema, Value};
use serde_json::{Value as Json, json};

pub struct Rng(u64);

impl Rng {
  pub fn new(seed: u64) -> Self {
    Self(seed)
  }

  pub fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  pub fn below(&mut self, n: u64) -> u64 {
    self.next() % n
  }

  pub fn chance(&mut self, p: f64) -> bool {
    (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
  }

  pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
    &xs[self.below(xs.len() as u64) as usize]
  }

  pub fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
    lo + self.below((hi - lo + 1) as u64) as i64
  }
}

/// Keywords, skewed so some terms are dense (bitset postings) and some
/// rare (list postings).
pub const VOCAB: [&str; 8] = ["a", "a", "a", "b", "b", "c", "dd", "été"];
/// Rare keywords: each in well under 1/32 of documents, so they are
/// stored as sorted id lists rather than bitsets.
pub const RARE: [&str; 12] = [
  "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "rx", "ry",
];

pub fn keyword(r: &mut Rng) -> &'static str {
  let words: &[&'static str] = if r.chance(0.15) { &RARE } else { &VOCAB };
  words[r.below(words.len() as u64) as usize]
}

pub const FLOATS: [f64; 9] =
  [-1.5, -0.0, 0.0, 0.5, 1.0, 2.0, 2.5, 1e-300, 7.25];

pub fn schema() -> Schema {
  Schema::new()
    .with("kw", FieldType::Keyword)
    .and_then(|s| s.with("tags", FieldType::Keyword))
    .and_then(|s| s.with("n", FieldType::Integer))
    .and_then(|s| s.with("x", FieldType::Float))
    .and_then(|s| s.with("b", FieldType::Bool))
    .and_then(|s| s.with("o.k", FieldType::Keyword))
    .and_then(|s| s.with("o.v", FieldType::Integer))
    .unwrap()
}

fn word(r: &mut Rng) -> Json {
  json!(keyword(r))
}

fn int_like(r: &mut Rng) -> Json {
  match r.below(10) {
    0 => json!(*r.pick(&[2.0, 2.5, -0.0, 1000.0, 3e0])),
    1 => json!("3"),
    2 => json!(null),
    3 => json!([r.range_i64(-5, 5), r.range_i64(-5, 5)]),
    4 => json!(u64::MAX),
    _ => json!(r.range_i64(-5, 5)),
  }
}

fn float_like(r: &mut Rng) -> Json {
  match r.below(10) {
    0 => json!(r.range_i64(-3, 3)),
    1 => json!(true),
    2 => json!([*r.pick(&FLOATS), *r.pick(&FLOATS)]),
    3 => json!([]),
    _ => json!(*r.pick(&FLOATS)),
  }
}

/// A random payload exercising arrays, nesting, missing fields and type
/// mismatches.
pub fn payload(r: &mut Rng) -> Json {
  let mut m = serde_json::Map::new();
  match r.below(20) {
    0..=13 => {
      m.insert("kw".into(), word(r));
    }
    14..=15 => {
      let n = r.below(4);
      m.insert("kw".into(), (0..n).map(|_| word(r)).collect());
    }
    16 => {
      m.insert("kw".into(), json!(3));
    }
    _ => {}
  }
  match r.below(6) {
    0 => {}
    1 => {
      m.insert("tags".into(), word(r));
    }
    2 => {
      m.insert("tags".into(), json!([[word(r)], word(r)]));
    }
    _ => {
      let n = r.below(5);
      m.insert("tags".into(), (0..n).map(|_| word(r)).collect());
    }
  }
  if !r.chance(0.15) {
    m.insert("n".into(), int_like(r));
  }
  if !r.chance(0.15) {
    m.insert("x".into(), float_like(r));
  }
  match r.below(8) {
    0 => {}
    1 => {
      m.insert("b".into(), json!("true"));
    }
    2 => {
      m.insert("b".into(), json!([true, false]));
    }
    _ => {
      m.insert("b".into(), json!(r.chance(0.3)));
    }
  }
  let obj = |r: &mut Rng| json!({"k": word(r), "v": r.range_i64(-3, 3)});
  match r.below(6) {
    0 => {}
    1 => {
      m.insert("o".into(), json!("flat"));
    }
    2 => {
      let a = obj(r);
      let b = obj(r);
      m.insert("o".into(), json!([a, b, 1]));
    }
    _ => {
      m.insert("o".into(), obj(r));
    }
  }
  Json::Object(m)
}

const FIELDS: [(&str, FieldType); 7] = [
  ("kw", FieldType::Keyword),
  ("tags", FieldType::Keyword),
  ("n", FieldType::Integer),
  ("x", FieldType::Float),
  ("b", FieldType::Bool),
  ("o.k", FieldType::Keyword),
  ("o.v", FieldType::Integer),
];

/// A random filter value of the right type for `ty`. `finite` keeps
/// infinities out (for JSON round trips).
fn value(r: &mut Rng, ty: FieldType, finite: bool) -> Value {
  match ty {
    FieldType::Keyword => {
      if r.chance(0.1) {
        Value::Str("zzz".into())
      } else {
        Value::Str(keyword(r).to_string())
      }
    }
    FieldType::Integer | FieldType::Float => match r.below(6) {
      0 => Value::Float(*r.pick(&FLOATS)),
      1 if !finite => {
        Value::Float(*r.pick(&[f64::INFINITY, f64::NEG_INFINITY, 2.0]))
      }
      2 => Value::Float(*r.pick(&[2.0, 2.5, -0.0, 1000.0])),
      _ => Value::Int(r.range_i64(-6, 6)),
    },
    FieldType::Bool => Value::Bool(r.chance(0.5)),
  }
}

fn bound(r: &mut Rng, ty: FieldType, finite: bool) -> Bound<Value> {
  match r.below(3) {
    0 => Bound::Unbounded,
    1 => Bound::Included(value(r, ty, finite)),
    _ => Bound::Excluded(value(r, ty, finite)),
  }
}

/// A random, well-typed filter over [`schema`] of at most `depth` levels.
pub fn filter(r: &mut Rng, depth: u32, finite: bool) -> Filter {
  if depth > 1 && r.chance(0.45) {
    let n = r.below(4) as usize;
    return match r.below(3) {
      0 => Filter::And((0..n).map(|_| filter(r, depth - 1, finite)).collect()),
      1 => Filter::Or((0..n).map(|_| filter(r, depth - 1, finite)).collect()),
      _ => !filter(r, depth - 1, finite),
    };
  }
  let &(name, ty) = r.pick(&FIELDS);
  let numeric = matches!(ty, FieldType::Integer | FieldType::Float);
  match r.below(if numeric { 5 } else { 4 }) {
    0 | 1 => Filter::Eq(name.into(), value(r, ty, finite)),
    2 => {
      let n = r.below(4);
      Filter::In(name.into(), (0..n).map(|_| value(r, ty, finite)).collect())
    }
    3 => Filter::Exists(name.into()),
    _ => Filter::Range {
      field: name.into(),
      lower: bound(r, ty, finite),
      upper: bound(r, ty, finite),
    },
  }
}

// ---- brute-force oracle ----

/// A payload value of a field's type.
#[derive(Clone, Copy, Debug)]
enum Lit<'a> {
  S(&'a str),
  I(i64),
  F(f64),
  B(bool),
}

fn collect<'a>(v: &'a Json, path: &[&str], out: &mut Vec<&'a Json>) {
  if let Json::Array(items) = v {
    for item in items {
      collect(item, path, out);
    }
    return;
  }
  if path.is_empty() {
    out.push(v);
  } else if let Some(child) = v.get(path[0]) {
    collect(child, &path[1..], out);
  }
}

fn typed(ty: FieldType, v: &Json) -> Option<Lit<'_>> {
  match (ty, v) {
    (FieldType::Keyword, Json::String(s)) => Some(Lit::S(s)),
    (FieldType::Bool, Json::Bool(b)) => Some(Lit::B(*b)),
    (FieldType::Float, Json::Number(n)) => Some(Lit::F(n.as_f64()?)),
    (FieldType::Integer, Json::Number(n)) => {
      if n.is_i64() {
        n.as_i64().map(Lit::I)
      } else if n.is_u64() {
        None
      } else {
        let f = n.as_f64()?;
        (f == f.trunc() && f.abs() < 9e18).then_some(Lit::I(f as i64))
      }
    }
    _ => None,
  }
}

fn field_values<'a>(
  schema: &Schema,
  name: &str,
  doc: &'a Json,
) -> Vec<Lit<'a>> {
  let ty = schema.field(name).unwrap().field_type();
  let path: Vec<&str> = name.split('.').collect();
  let mut raw = Vec::new();
  collect(doc, &path, &mut raw);
  raw.into_iter().filter_map(|v| typed(ty, v)).collect()
}

fn num(v: &Value) -> f64 {
  match *v {
    Value::Int(i) => i as f64,
    Value::Float(f) => f,
    _ => panic!("not a number: {v:?}"),
  }
}

fn lit_eq(x: Lit, v: &Value) -> bool {
  match (x, v) {
    (Lit::S(s), Value::Str(t)) => s == t,
    (Lit::B(a), Value::Bool(b)) => a == *b,
    (Lit::I(i), Value::Int(j)) => i == *j,
    // Small test numbers convert exactly, so f64 equality is the
    // documented "exact whole number" rule.
    (Lit::I(i), Value::Float(f)) => (i as f64) == *f,
    (Lit::F(a), v) => a == num(v),
    _ => panic!("ill-typed comparison {x:?} vs {v:?}"),
  }
}

fn lit_num(x: Lit) -> f64 {
  match x {
    Lit::I(i) => i as f64,
    Lit::F(f) => f,
    _ => panic!("range on non-number {x:?}"),
  }
}

fn in_range(x: Lit, lower: &Bound<Value>, upper: &Bound<Value>) -> bool {
  let x = lit_num(x);
  let lo = match lower {
    Bound::Unbounded => true,
    Bound::Included(v) => x >= num(v),
    Bound::Excluded(v) => x > num(v),
  };
  let hi = match upper {
    Bound::Unbounded => true,
    Bound::Included(v) => x <= num(v),
    Bound::Excluded(v) => x < num(v),
  };
  lo && hi
}

/// Whether `doc` matches `f`, straight from the documented semantics.
pub fn matches(schema: &Schema, f: &Filter, doc: &Json) -> bool {
  match f {
    Filter::Eq(name, v) => field_values(schema, name, doc)
      .iter()
      .any(|&x| lit_eq(x, v)),
    Filter::In(name, vs) => field_values(schema, name, doc)
      .iter()
      .any(|&x| vs.iter().any(|v| lit_eq(x, v))),
    Filter::Range {
      field,
      lower,
      upper,
    } => field_values(schema, field, doc)
      .iter()
      .any(|&x| in_range(x, lower, upper)),
    Filter::Exists(name) => !field_values(schema, name, doc).is_empty(),
    Filter::And(c) => c.iter().all(|f| matches(schema, f, doc)),
    Filter::Or(c) => c.iter().any(|f| matches(schema, f, doc)),
    Filter::Not(c) => !matches(schema, c, doc),
  }
}

/// Ids `0..num_docs` matching `f`; `docs` holds `(id, payload)` and ids
/// without a payload have no fields.
pub fn brute(
  schema: &Schema,
  f: &Filter,
  docs: &[(u32, Json)],
  num_docs: usize,
) -> Vec<u32> {
  let empty = Json::Object(Default::default());
  let mut by_id = vec![&empty; num_docs];
  for (id, p) in docs {
    if (*id as usize) < num_docs {
      by_id[*id as usize] = p;
    }
  }
  (0..num_docs as u32)
    .filter(|&id| matches(schema, f, by_id[id as usize]))
    .collect()
}
