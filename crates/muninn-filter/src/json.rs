//! JSON forms of [`Filter`] and [`Schema`], as the HTTP server accepts
//! them.
//!
//! ```text
//! filter := {"eq":     {FIELD: SCALAR}}
//!         | {"in":     {FIELD: [SCALAR, ...]}}
//!         | {"range":  {FIELD: {"gt"|"gte"|"lt"|"lte": NUMBER, ...}}}
//!         | {"exists": FIELD}
//!         | {"and":    [filter, ...]}
//!         | {"or":     [filter, ...]}
//!         | {"not":    filter}
//! SCALAR := string | number | true | false
//! schema := {FIELD: "keyword" | "integer" | "float" | "bool", ...}
//! ```
//!
//! Numbers written without a fraction or exponent that fit in an `i64`
//! become [`Value::Int`], every other number [`Value::Float`].

use std::ops::Bound;
use std::str::FromStr;

use muninn_core::{Error, Result};
use serde::de::Error as _;
use serde::ser::Error as _;
use serde_json::{Map, Number, Value as Json};

use crate::plan::MAX_DEPTH;
use crate::{FieldType, Filter, Schema, Value};

const OPS: &str = "eq, in, range, exists, and, or, not";

fn err(at: &str, msg: impl std::fmt::Display) -> Error {
  Error::invalid(format!("filter at {at}: {msg}"))
}

/// Short description of a JSON value's type, for error messages.
fn kind(v: &Json) -> &'static str {
  match v {
    Json::Null => "null",
    Json::Bool(_) => "a boolean",
    Json::Number(_) => "a number",
    Json::String(_) => "a string",
    Json::Array(_) => "an array",
    Json::Object(_) => "an object",
  }
}

/// The single `(field, value)` entry of `{"field": value}`.
fn single<'a>(at: &str, op: &str, v: &'a Json) -> Result<(&'a str, &'a Json)> {
  let obj = v.as_object().ok_or_else(|| {
    err(
      at,
      format!("{op} expects {{\"<field>\": ...}}, got {}", kind(v)),
    )
  })?;
  let mut it = obj.iter();
  match (it.next(), it.next()) {
    (Some((k, v)), None) => {
      if k.is_empty() {
        Err(err(at, "field name is empty"))
      } else {
        Ok((k.as_str(), v))
      }
    }
    (None, _) => Err(err(at, format!("{op} needs a field"))),
    _ => Err(err(
      at,
      format!("{op} takes exactly one field; combine several with \"and\""),
    )),
  }
}

fn number(n: &Number) -> Value {
  match n.as_i64() {
    Some(i) => Value::Int(i),
    None => Value::Float(n.as_f64().unwrap_or(f64::NAN)),
  }
}

fn scalar(at: &str, v: &Json) -> Result<Value> {
  Ok(match v {
    Json::String(s) => Value::Str(s.clone()),
    Json::Number(n) => number(n),
    Json::Bool(b) => Value::Bool(*b),
    _ => {
      return Err(err(
        at,
        format!("expected a string, number or boolean, got {}", kind(v)),
      ));
    }
  })
}

fn parse(v: &Json, at: &str, depth: usize) -> Result<Filter> {
  if depth > MAX_DEPTH {
    return Err(err(at, format!("nested deeper than {MAX_DEPTH} levels")));
  }
  let obj = v.as_object().ok_or_else(|| {
    err(
      at,
      format!("expected an object with one of {OPS}, got {}", kind(v)),
    )
  })?;
  let mut it = obj.iter();
  let (op, arg) = match (it.next(), it.next()) {
    (Some(e), None) => e,
    (None, _) => return Err(err(at, format!("empty object; expected {OPS}"))),
    _ => {
      return Err(err(
        at,
        "an object holds exactly one operator; combine several with \"and\"",
      ));
    }
  };
  let here = format!("{at}.{op}");
  Ok(match op.as_str() {
    "eq" => {
      let (field, value) = single(&here, op, arg)?;
      Filter::Eq(
        field.to_string(),
        scalar(&format!("{here}.{field}"), value)?,
      )
    }
    "in" => {
      let (field, values) = single(&here, op, arg)?;
      let at = format!("{here}.{field}");
      let arr = values.as_array().ok_or_else(|| {
        err(&at, format!("in expects an array, got {}", kind(values)))
      })?;
      let vals = arr
        .iter()
        .enumerate()
        .map(|(i, v)| scalar(&format!("{at}[{i}]"), v))
        .collect::<Result<_>>()?;
      Filter::In(field.to_string(), vals)
    }
    "range" => {
      let (field, bounds) = single(&here, op, arg)?;
      let at = format!("{here}.{field}");
      let bounds = bounds.as_object().ok_or_else(|| {
        err(
          &at,
          format!(
            "range expects {{\"gte\": ..., \"lt\": ...}}, got {}",
            kind(bounds)
          ),
        )
      })?;
      let (mut lower, mut upper) = (Bound::Unbounded, Bound::Unbounded);
      for (k, v) in bounds {
        let n = match v {
          Json::Number(n) => number(n),
          _ => {
            return Err(err(
              &format!("{at}.{k}"),
              format!("range bound must be a number, got {}", kind(v)),
            ));
          }
        };
        let (slot, b) = match k.as_str() {
          "gt" => (&mut lower, Bound::Excluded(n)),
          "gte" => (&mut lower, Bound::Included(n)),
          "lt" => (&mut upper, Bound::Excluded(n)),
          "lte" => (&mut upper, Bound::Included(n)),
          _ => {
            return Err(err(
              &at,
              format!("unknown range bound {k:?}; use gt, gte, lt, lte"),
            ));
          }
        };
        if !matches!(slot, Bound::Unbounded) {
          return Err(err(
            &at,
            "give at most one lower (gt/gte) and one upper (lt/lte) bound",
          ));
        }
        *slot = b;
      }
      Filter::Range {
        field: field.to_string(),
        lower,
        upper,
      }
    }
    "exists" => match arg {
      Json::String(s) if !s.is_empty() => Filter::Exists(s.clone()),
      _ => {
        return Err(err(
          &here,
          format!("exists expects a field name string, got {}", kind(arg)),
        ));
      }
    },
    "and" | "or" => {
      let arr = arg.as_array().ok_or_else(|| {
        err(&here, format!("{op} expects an array, got {}", kind(arg)))
      })?;
      let children = arr
        .iter()
        .enumerate()
        .map(|(i, c)| parse(c, &format!("{here}[{i}]"), depth + 1))
        .collect::<Result<Vec<_>>>()?;
      if op == "and" {
        Filter::And(children)
      } else {
        Filter::Or(children)
      }
    }
    "not" => Filter::Not(Box::new(parse(arg, &here, depth + 1)?)),
    _ => {
      return Err(err(at, format!("unknown operator {op:?}; expected {OPS}")));
    }
  })
}

fn value_json(v: &Value) -> Result<Json> {
  Ok(match v {
    Value::Str(s) => Json::String(s.clone()),
    Value::Int(i) => Json::Number((*i).into()),
    Value::Bool(b) => Json::Bool(*b),
    Value::Float(f) => Json::Number(Number::from_f64(*f).ok_or_else(|| {
      Error::invalid(format!("{f} cannot be written as JSON"))
    })?),
  })
}

fn field_obj(field: &str, v: Json) -> Json {
  let mut m = Map::new();
  m.insert(field.to_string(), v);
  Json::Object(m)
}

fn op_obj(op: &str, v: Json) -> Json {
  field_obj(op, v)
}

impl Filter {
  /// Parses the JSON form (see the grammar in the crate's `json` module
  /// docs). Errors are `InvalidArgument` and name the offending location,
  /// e.g. `filter at $.and[1].range.year: range bound must be a number`.
  pub fn from_json(v: &Json) -> Result<Filter> {
    parse(v, "$", 1)
  }

  /// Parses JSON text.
  pub fn parse(text: &str) -> Result<Filter> {
    let v: Json = serde_json::from_str(text)
      .map_err(|e| Error::invalid(format!("filter is not valid JSON: {e}")))?;
    Self::from_json(&v)
  }

  /// The JSON form. Fails only for NaN or infinite float values, which
  /// JSON cannot represent.
  pub fn to_json(&self) -> Result<Json> {
    Ok(match self {
      Filter::Eq(f, v) => op_obj("eq", field_obj(f, value_json(v)?)),
      Filter::In(f, vs) => op_obj(
        "in",
        field_obj(
          f,
          Json::Array(vs.iter().map(value_json).collect::<Result<_>>()?),
        ),
      ),
      Filter::Range {
        field,
        lower,
        upper,
      } => {
        let mut m = Map::new();
        match lower {
          Bound::Included(v) => m.insert("gte".into(), value_json(v)?),
          Bound::Excluded(v) => m.insert("gt".into(), value_json(v)?),
          Bound::Unbounded => None,
        };
        match upper {
          Bound::Included(v) => m.insert("lte".into(), value_json(v)?),
          Bound::Excluded(v) => m.insert("lt".into(), value_json(v)?),
          Bound::Unbounded => None,
        };
        op_obj("range", field_obj(field, Json::Object(m)))
      }
      Filter::Exists(f) => op_obj("exists", Json::String(f.clone())),
      Filter::And(c) => op_obj(
        "and",
        Json::Array(c.iter().map(Filter::to_json).collect::<Result<_>>()?),
      ),
      Filter::Or(c) => op_obj(
        "or",
        Json::Array(c.iter().map(Filter::to_json).collect::<Result<_>>()?),
      ),
      Filter::Not(c) => op_obj("not", c.to_json()?),
    })
  }
}

impl FromStr for Filter {
  type Err = Error;

  fn from_str(s: &str) -> Result<Self> {
    Self::parse(s)
  }
}

impl serde::Serialize for Filter {
  fn serialize<S: serde::Serializer>(
    &self,
    s: S,
  ) -> std::result::Result<S::Ok, S::Error> {
    self.to_json().map_err(S::Error::custom)?.serialize(s)
  }
}

impl<'de> serde::Deserialize<'de> for Filter {
  fn deserialize<D: serde::Deserializer<'de>>(
    d: D,
  ) -> std::result::Result<Self, D::Error> {
    let v = Json::deserialize(d)?;
    Filter::from_json(&v).map_err(D::Error::custom)
  }
}

impl Schema {
  /// Parses `{"field": "keyword" | "integer" | "float" | "bool", ...}`.
  pub fn from_json(v: &Json) -> Result<Schema> {
    let obj = v.as_object().ok_or_else(|| {
      Error::invalid(format!(
        "schema must be an object of field types, got {}",
        kind(v)
      ))
    })?;
    let mut s = Schema::new();
    for (name, ty) in obj {
      let ty = ty.as_str().and_then(FieldType::from_name).ok_or_else(|| {
        Error::invalid(format!(
          "field {name:?}: type must be \"keyword\", \"integer\", \"float\" \
           or \"bool\", got {ty}"
        ))
      })?;
      s.add(name, ty)?;
    }
    Ok(s)
  }

  /// The JSON form read by [`Schema::from_json`].
  pub fn to_json(&self) -> Json {
    let mut m = Map::new();
    for f in self.fields() {
      m.insert(
        f.name().to_string(),
        Json::String(f.field_type().name().into()),
      );
    }
    Json::Object(m)
  }
}

impl serde::Serialize for Schema {
  fn serialize<S: serde::Serializer>(
    &self,
    s: S,
  ) -> std::result::Result<S::Ok, S::Error> {
    self.to_json().serialize(s)
  }
}

impl<'de> serde::Deserialize<'de> for Schema {
  fn deserialize<D: serde::Deserializer<'de>>(
    d: D,
  ) -> std::result::Result<Self, D::Error> {
    let v = Json::deserialize(d)?;
    Schema::from_json(&v).map_err(D::Error::custom)
  }
}
