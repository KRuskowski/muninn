//! Declared filterable fields and how their values are read from JSON.

use std::collections::HashMap;

use muninn_core::{Error, Result};
use serde_json::Value as Json;

use crate::value::{json_f64, json_i64};

/// Most fields one schema may declare.
pub const MAX_FIELDS: usize = 1 << 16;

/// The type of a filterable field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldType {
  /// Exact-match strings (`eq`, `in`, `exists`).
  Keyword,
  /// Whole numbers that fit in an `i64` (`eq`, `in`, `range`, `exists`).
  Integer,
  /// Finite `f64` numbers (`eq`, `in`, `range`, `exists`).
  Float,
  /// `true` / `false` (`eq`, `in`, `exists`).
  Bool,
}

impl FieldType {
  /// The name used in the JSON schema form: `keyword`, `integer`,
  /// `float` or `bool`.
  pub fn name(self) -> &'static str {
    match self {
      Self::Keyword => "keyword",
      Self::Integer => "integer",
      Self::Float => "float",
      Self::Bool => "bool",
    }
  }

  /// Parses [`FieldType::name`].
  pub fn from_name(name: &str) -> Option<Self> {
    Some(match name {
      "keyword" => Self::Keyword,
      "integer" => Self::Integer,
      "float" => Self::Float,
      "bool" => Self::Bool,
      _ => return None,
    })
  }

  pub(crate) fn code(self) -> u64 {
    match self {
      Self::Keyword => 1,
      Self::Integer => 2,
      Self::Float => 3,
      Self::Bool => 4,
    }
  }

  pub(crate) fn from_code(code: u64) -> Option<Self> {
    Some(match code {
      1 => Self::Keyword,
      2 => Self::Integer,
      3 => Self::Float,
      4 => Self::Bool,
      _ => return None,
    })
  }
}

/// One declared field: a dot-separated path into the payload and a type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
  name: String,
  ty: FieldType,
  path: Vec<String>,
}

impl Field {
  /// The field path as declared, e.g. `"meta.lang"`.
  pub fn name(&self) -> &str {
    &self.name
  }

  /// The declared type.
  pub fn field_type(&self) -> FieldType {
    self.ty
  }
}

/// The filterable fields of a collection.
///
/// A field name is a dot-separated path into the JSON payload:
/// `"meta.lang"` reads `{"meta": {"lang": "en"}}`. Arrays met anywhere
/// along the path are flattened, so `"authors.name"` reads every name in
/// `{"authors": [{"name": "a"}, {"name": "b"}]}` and the field is
/// multi-valued. Object keys that contain a dot cannot be addressed.
/// Values of the wrong type are ignored for that document (a string in an
/// integer field is as if the field were missing), never an error.
#[derive(Clone, Debug, Default)]
pub struct Schema {
  fields: Vec<Field>,
  by_name: HashMap<String, usize>,
}

/// Schemas are equal when they declare the same fields with the same
/// types, in any order.
impl PartialEq for Schema {
  fn eq(&self, other: &Self) -> bool {
    self.fields.len() == other.fields.len()
      && self
        .fields
        .iter()
        .all(|f| other.field(&f.name).is_some_and(|g| g.ty == f.ty))
  }
}

impl Schema {
  /// A schema with no fields.
  pub fn new() -> Self {
    Self::default()
  }

  /// Declares a field and returns its index. Fails on an empty path
  /// segment, a duplicate name or too many fields.
  pub fn add(&mut self, name: &str, ty: FieldType) -> Result<usize> {
    if name.is_empty() || name.split('.').any(str::is_empty) {
      return Err(Error::invalid(format!(
        "field name {name:?} must be non-empty dot-separated keys"
      )));
    }
    if self.by_name.contains_key(name) {
      return Err(Error::invalid(format!("field {name:?} declared twice")));
    }
    if self.fields.len() >= MAX_FIELDS {
      return Err(Error::invalid(format!(
        "a schema holds at most {MAX_FIELDS} fields"
      )));
    }
    let i = self.fields.len();
    self.fields.push(Field {
      name: name.to_string(),
      ty,
      path: name.split('.').map(str::to_string).collect(),
    });
    self.by_name.insert(name.to_string(), i);
    Ok(i)
  }

  /// Builder form of [`Schema::add`].
  pub fn with(mut self, name: &str, ty: FieldType) -> Result<Self> {
    self.add(name, ty)?;
    Ok(self)
  }

  /// Number of fields.
  pub fn len(&self) -> usize {
    self.fields.len()
  }

  /// True if no field is declared.
  pub fn is_empty(&self) -> bool {
    self.fields.is_empty()
  }

  /// Fields in declaration order.
  pub fn fields(&self) -> &[Field] {
    &self.fields
  }

  /// Index of the field named `name`.
  pub fn index_of(&self, name: &str) -> Option<usize> {
    self.by_name.get(name).copied()
  }

  /// The field named `name`.
  pub fn field(&self, name: &str) -> Option<&Field> {
    self.index_of(name).map(|i| &self.fields[i])
  }

  /// Checks that `filter` only uses declared fields with values of the
  /// right types, as [`crate::MemIndex::evaluate`] would.
  pub fn check(&self, filter: &crate::Filter) -> Result<()> {
    crate::plan::compile(self, filter).map(|_| ())
  }
}

/// A typed value read from a payload.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Scalar<'a> {
  Str(&'a str),
  Int(i64),
  Float(f64),
  Bool(bool),
}

/// Calls `f` with every JSON value at `path` under `v`, flattening arrays.
fn walk<'a>(v: &'a Json, path: &[String], f: &mut impl FnMut(&'a Json)) {
  match v {
    Json::Array(items) => {
      for item in items {
        walk(item, path, f);
      }
    }
    _ => match path.split_first() {
      None => f(v),
      Some((key, rest)) => {
        if let Some(child) = v.as_object().and_then(|m| m.get(key.as_str())) {
          walk(child, rest, f);
        }
      }
    },
  }
}

/// Calls `f` with every value of `field` in `payload` that has the
/// field's type; values of other types are skipped.
pub(crate) fn extract<'a>(
  field: &Field,
  payload: &'a Json,
  f: &mut impl FnMut(Scalar<'a>),
) {
  match field.ty {
    FieldType::Keyword => walk(payload, &field.path, &mut |v| {
      if let Json::String(s) = v {
        f(Scalar::Str(s));
      }
    }),
    FieldType::Integer => walk(payload, &field.path, &mut |v| {
      if let Some(i) = v.as_number().and_then(json_i64) {
        f(Scalar::Int(i));
      }
    }),
    FieldType::Float => walk(payload, &field.path, &mut |v| {
      if let Some(x) = v.as_number().and_then(json_f64) {
        f(Scalar::Float(x));
      }
    }),
    FieldType::Bool => walk(payload, &field.path, &mut |v| {
      if let Json::Bool(b) = v {
        f(Scalar::Bool(*b));
      }
    }),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use serde_json::json;

  fn values(schema: &Schema, name: &str, doc: &Json) -> Vec<String> {
    let mut out = Vec::new();
    extract(schema.field(name).unwrap(), doc, &mut |s| {
      out.push(format!("{s:?}"))
    });
    out
  }

  #[test]
  fn paths_flatten_arrays_and_skip_mismatches() {
    let s = Schema::new()
      .with("a.b", FieldType::Integer)
      .unwrap()
      .with("t", FieldType::Keyword)
      .unwrap()
      .with("x", FieldType::Float)
      .unwrap()
      .with("f", FieldType::Bool)
      .unwrap();
    let doc = json!({
      "a": [{"b": 1}, {"b": [2, 3.0, 3.5, "4"]}, {"c": 5}, 7],
      "t": ["x", ["y"], 1, null],
      "x": [-0.0, 2, "1.5", 1e300],
      "f": [true, "false", 0],
    });
    assert_eq!(values(&s, "a.b", &doc), ["Int(1)", "Int(2)", "Int(3)"]);
    assert_eq!(values(&s, "t", &doc), [r#"Str("x")"#, r#"Str("y")"#]);
    assert_eq!(
      values(&s, "x", &doc),
      ["Float(0.0)", "Float(2.0)", "Float(1e300)"]
    );
    assert_eq!(values(&s, "f", &doc), ["Bool(true)"]);
    assert!(values(&s, "t", &json!("x")).is_empty());
    assert!(values(&s, "a.b", &json!({"a": {"b": {}}})).is_empty());
  }

  #[test]
  fn add_validates_names() {
    let mut s = Schema::new();
    assert_eq!(s.add("a", FieldType::Keyword).unwrap(), 0);
    assert!(s.add("a", FieldType::Integer).is_err());
    assert!(s.add("", FieldType::Integer).is_err());
    assert!(s.add("a..b", FieldType::Integer).is_err());
    assert!(s.add(".a", FieldType::Integer).is_err());
    assert_eq!(s.index_of("a"), Some(0));
    assert_eq!(s.len(), 1);
    assert!(!s.is_empty());
    for ty in [
      FieldType::Keyword,
      FieldType::Integer,
      FieldType::Float,
      FieldType::Bool,
    ] {
      assert_eq!(FieldType::from_name(ty.name()), Some(ty));
      assert_eq!(FieldType::from_code(ty.code()), Some(ty));
    }
    assert_eq!(FieldType::from_code(0), None);
    assert_eq!(FieldType::from_name("text"), None);
  }
}
