//! Turns a [`Filter`] into a typed plan against a [`Schema`], and
//! evaluates or estimates plans against any index that implements
//! [`Source`].

use std::ops::Bound;

use muninn_core::{BitSet, Error, Result};

use crate::bits::Bits;
use crate::value::norm_f64;
use crate::value::{TWO_POW_63, exact_i64, f64_key, next_down, next_up};
use crate::{FieldType, Filter, Schema, Value};

/// Deepest filter tree accepted, so evaluation cannot exhaust the stack.
pub const MAX_DEPTH: usize = 256;

/// A predicate on one field, already checked against its type and
/// normalised to the field's domain.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Pred<'f> {
  /// Keyword is one of these terms (sorted, unique).
  Terms(Vec<&'f str>),
  /// Integer is one of these (sorted, unique).
  Ints(Vec<i64>),
  /// Integer in `lo..=hi`, `lo <= hi`.
  IntRange(i64, i64),
  /// Float is one of these (normalised, sorted, unique; may be infinite).
  Floats(Vec<f64>),
  /// Float in `lo..=hi`, `lo <= hi`, both normalised.
  FloatRange(f64, f64),
  /// Bool is `false` (index 0) and/or `true` (index 1).
  Bools([bool; 2]),
  /// Field has any value.
  Exists,
}

/// A compiled filter: field names resolved, constants folded.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Plan<'f> {
  All,
  Empty,
  Leaf(usize, Pred<'f>),
  And(Vec<Plan<'f>>),
  Or(Vec<Plan<'f>>),
  Not(Box<Plan<'f>>),
}

/// What evaluation needs from an index.
pub(crate) trait Source {
  fn schema(&self) -> &Schema;

  /// Number of (value, doc) entries matching `pred`: an upper bound on
  /// matching documents, used for estimates. Must be cheap.
  fn count(&self, field: usize, pred: &Pred) -> u64;

  /// Sets the bit of every document matching `pred`.
  fn fill(&self, field: usize, pred: &Pred, bits: &mut Bits);
}

pub(crate) fn evaluate<S: Source>(
  src: &S,
  filter: &Filter,
  num_docs: usize,
) -> Result<BitSet> {
  let plan = compile(src.schema(), filter)?;
  let est = annotate(src, &plan, num_docs);
  Ok(eval(src, &plan, &est, num_docs).into_bitset())
}

pub(crate) fn estimate<S: Source>(
  src: &S,
  filter: &Filter,
  num_docs: usize,
) -> f32 {
  match compile(src.schema(), filter) {
    Ok(plan) => annotate(src, &plan, num_docs).value,
    Err(_) => 1.0,
  }
}

/// Estimated selectivity of every plan node, mirroring the plan's shape
/// (`kids[i]` belongs to child `i`; `Not` has one kid). Computed once per
/// evaluation, so each leaf is counted once however deep it sits.
struct Est {
  value: f32,
  kids: Vec<Est>,
}

fn annotate<S: Source>(src: &S, plan: &Plan, n: usize) -> Est {
  let leaf = |value| Est {
    value,
    kids: Vec::new(),
  };
  match plan {
    Plan::All => leaf(1.0),
    Plan::Empty => leaf(0.0),
    Plan::Leaf(f, p) => leaf(if n == 0 {
      0.0
    } else {
      (src.count(*f, p) as f64 / n as f64).min(1.0) as f32
    }),
    Plan::And(c) => {
      let kids: Vec<Est> = c.iter().map(|p| annotate(src, p, n)).collect();
      Est {
        value: kids.iter().map(|k| k.value).product(),
        kids,
      }
    }
    Plan::Or(c) => {
      let kids: Vec<Est> = c.iter().map(|p| annotate(src, p, n)).collect();
      Est {
        value: 1.0 - kids.iter().map(|k| 1.0 - k.value).product::<f32>(),
        kids,
      }
    }
    Plan::Not(c) => {
      let kid = annotate(src, c, n);
      Est {
        value: 1.0 - kid.value,
        kids: vec![kid],
      }
    }
  }
}

fn eval<S: Source>(src: &S, plan: &Plan, est: &Est, n: usize) -> Bits {
  match plan {
    Plan::All => Bits::full(n),
    Plan::Empty => Bits::new(n),
    Plan::Leaf(f, p) => {
      let mut b = Bits::new(n);
      src.fill(*f, p, &mut b);
      b
    }
    Plan::Not(c) => {
      let mut b = eval(src, c, &est.kids[0], n);
      b.negate();
      b
    }
    Plan::And(children) => {
      // Most selective first, so the running set shrinks fastest and an
      // empty intersection stops the work early.
      let mut order: Vec<usize> = (0..children.len()).collect();
      order.sort_by(|&a, &b| est.kids[a].value.total_cmp(&est.kids[b].value));
      let first = order[0];
      let mut acc = eval(src, &children[first], &est.kids[first], n);
      for &i in &order[1..] {
        if acc.is_empty() {
          break;
        }
        match &children[i] {
          Plan::Not(inner) => {
            acc.and_not(&eval(src, inner, &est.kids[i].kids[0], n))
          }
          child => acc.and_with(&eval(src, child, &est.kids[i], n)),
        }
      }
      acc
    }
    Plan::Or(children) => {
      // Least selective first, so a full set stops the work early.
      let mut order: Vec<usize> = (0..children.len()).collect();
      order.sort_by(|&a, &b| est.kids[b].value.total_cmp(&est.kids[a].value));
      let mut acc = Bits::new(n);
      for &i in &order {
        match &children[i] {
          Plan::Leaf(f, p) => src.fill(*f, p, &mut acc),
          child => acc.or_with(&eval(src, child, &est.kids[i], n)),
        }
        if acc.is_full() {
          break;
        }
      }
      acc
    }
  }
}

/// Resolves fields, checks types and folds constants.
pub(crate) fn compile<'f>(
  schema: &Schema,
  filter: &'f Filter,
) -> Result<Plan<'f>> {
  compile_at(schema, filter, 1)
}

fn compile_at<'f>(
  schema: &Schema,
  filter: &'f Filter,
  depth: usize,
) -> Result<Plan<'f>> {
  if depth > MAX_DEPTH {
    return Err(Error::invalid(format!(
      "filter is nested deeper than {MAX_DEPTH} levels"
    )));
  }
  Ok(match filter {
    Filter::And(children) => {
      let mut out = Vec::with_capacity(children.len());
      let mut empty = false;
      // Every child is compiled, even after an empty one, so type errors
      // are reported regardless of what they are combined with.
      for c in children {
        match compile_at(schema, c, depth + 1)? {
          Plan::All => {}
          Plan::Empty => empty = true,
          Plan::And(inner) => out.extend(inner),
          p => out.push(p),
        }
      }
      if empty {
        return Ok(Plan::Empty);
      }
      match out.len() {
        0 => Plan::All,
        1 => out.pop().unwrap(),
        _ => Plan::And(out),
      }
    }
    Filter::Or(children) => {
      let mut out = Vec::with_capacity(children.len());
      let mut all = false;
      for c in children {
        match compile_at(schema, c, depth + 1)? {
          Plan::Empty => {}
          Plan::All => all = true,
          Plan::Or(inner) => out.extend(inner),
          p => out.push(p),
        }
      }
      if all {
        return Ok(Plan::All);
      }
      match out.len() {
        0 => Plan::Empty,
        1 => out.pop().unwrap(),
        _ => Plan::Or(out),
      }
    }
    Filter::Not(child) => match compile_at(schema, child, depth + 1)? {
      Plan::All => Plan::Empty,
      Plan::Empty => Plan::All,
      Plan::Not(inner) => *inner,
      p => Plan::Not(Box::new(p)),
    },
    Filter::Exists(name) => Plan::Leaf(field(schema, name)?.0, Pred::Exists),
    Filter::Eq(name, v) => {
      let (i, ty) = field(schema, name)?;
      leaf(i, set_pred(name, ty, "eq", std::slice::from_ref(v))?)
    }
    Filter::In(name, vs) => {
      let (i, ty) = field(schema, name)?;
      leaf(i, set_pred(name, ty, "in", vs)?)
    }
    Filter::Range {
      field: name,
      lower,
      upper,
    } => {
      let (i, ty) = field(schema, name)?;
      leaf(i, range_pred(name, ty, lower, upper)?)
    }
  })
}

fn leaf(field: usize, pred: Option<Pred>) -> Plan {
  match pred {
    Some(p) => Plan::Leaf(field, p),
    None => Plan::Empty,
  }
}

fn field(schema: &Schema, name: &str) -> Result<(usize, FieldType)> {
  let i = schema.index_of(name).ok_or_else(|| {
    Error::invalid(format!("field {name:?} is not declared filterable"))
  })?;
  Ok((i, schema.fields()[i].field_type()))
}

fn type_error(name: &str, ty: FieldType, op: &str, v: &Value) -> Error {
  let want = match ty {
    FieldType::Keyword => "a string",
    FieldType::Integer | FieldType::Float => "a number",
    FieldType::Bool => "a boolean",
  };
  Error::invalid(format!(
    "field {name:?} is a {} field; {op} needs {want}, got {} {v}",
    ty.name(),
    v.type_name()
  ))
}

fn nan_error(name: &str) -> Error {
  Error::invalid(format!("NaN is not a valid filter value (field {name:?})"))
}

/// `eq` / `in` on a field of type `ty`. `None` means "matches nothing".
fn set_pred<'f>(
  name: &str,
  ty: FieldType,
  op: &str,
  values: &'f [Value],
) -> Result<Option<Pred<'f>>> {
  let pred = match ty {
    FieldType::Keyword => {
      let mut terms = Vec::with_capacity(values.len());
      for v in values {
        match v {
          Value::Str(s) => terms.push(s.as_str()),
          _ => return Err(type_error(name, ty, op, v)),
        }
      }
      terms.sort_unstable();
      terms.dedup();
      (!terms.is_empty()).then_some(Pred::Terms(terms))
    }
    FieldType::Integer => {
      let mut ints = Vec::with_capacity(values.len());
      for v in values {
        match *v {
          Value::Int(i) => ints.push(i),
          // A float equals an integer only if it is that whole number.
          Value::Float(f) if f.is_nan() => return Err(nan_error(name)),
          Value::Float(f) => ints.extend(exact_i64(f)),
          _ => return Err(type_error(name, ty, op, v)),
        }
      }
      ints.sort_unstable();
      ints.dedup();
      (!ints.is_empty()).then_some(Pred::Ints(ints))
    }
    FieldType::Float => {
      let mut floats = Vec::with_capacity(values.len());
      for v in values {
        match *v {
          Value::Int(i) => floats.push(i as f64),
          Value::Float(f) if f.is_nan() => return Err(nan_error(name)),
          Value::Float(f) => floats.push(norm_f64(f)),
          _ => return Err(type_error(name, ty, op, v)),
        }
      }
      floats.sort_unstable_by_key(|&f| f64_key(f));
      floats.dedup();
      (!floats.is_empty()).then_some(Pred::Floats(floats))
    }
    FieldType::Bool => {
      let mut which = [false; 2];
      for v in values {
        match *v {
          Value::Bool(b) => which[usize::from(b)] = true,
          _ => return Err(type_error(name, ty, op, v)),
        }
      }
      (which != [false; 2]).then_some(Pred::Bools(which))
    }
  };
  Ok(pred)
}

/// A range bound as an `f64`, rejecting non-numbers and NaN.
fn bound_f64(name: &str, v: &Value) -> Result<f64> {
  match *v {
    Value::Int(i) => Ok(i as f64),
    Value::Float(f) if f.is_nan() => Err(nan_error(name)),
    Value::Float(f) => Ok(f),
    _ => Err(Error::invalid(format!(
      "range bound on {name:?} must be a number, got {} {v}",
      v.type_name()
    ))),
  }
}

/// Smallest integer satisfying `x > f` (`excl`) or `x >= f`, if any.
fn int_lower_of_f64(f: f64, excl: bool) -> Option<i64> {
  let c = f.ceil();
  if c >= TWO_POW_63 {
    return None;
  }
  // -2^63 as a float is below every integer (see `exact_i64`).
  if c <= -TWO_POW_63 {
    return Some(i64::MIN);
  }
  let i = c as i64;
  if excl && c == f {
    i.checked_add(1)
  } else {
    Some(i)
  }
}

/// Largest integer satisfying `x < f` (`excl`) or `x <= f`, if any.
fn int_upper_of_f64(f: f64, excl: bool) -> Option<i64> {
  let c = f.floor();
  // -2^63 as a float is below every integer (see `exact_i64`).
  if c <= -TWO_POW_63 {
    return None;
  }
  if c >= TWO_POW_63 {
    return Some(i64::MAX);
  }
  let i = c as i64;
  if excl && c == f {
    i.checked_sub(1)
  } else {
    Some(i)
  }
}

fn int_lower(name: &str, b: &Bound<Value>) -> Result<Option<i64>> {
  Ok(match b {
    Bound::Unbounded => Some(i64::MIN),
    Bound::Included(Value::Int(i)) => Some(*i),
    Bound::Excluded(Value::Int(i)) => i.checked_add(1),
    Bound::Included(v) => int_lower_of_f64(bound_f64(name, v)?, false),
    Bound::Excluded(v) => int_lower_of_f64(bound_f64(name, v)?, true),
  })
}

fn int_upper(name: &str, b: &Bound<Value>) -> Result<Option<i64>> {
  Ok(match b {
    Bound::Unbounded => Some(i64::MAX),
    Bound::Included(Value::Int(i)) => Some(*i),
    Bound::Excluded(Value::Int(i)) => i.checked_sub(1),
    Bound::Included(v) => int_upper_of_f64(bound_f64(name, v)?, false),
    Bound::Excluded(v) => int_upper_of_f64(bound_f64(name, v)?, true),
  })
}

fn float_lower(name: &str, b: &Bound<Value>) -> Result<Option<f64>> {
  Ok(
    match b {
      Bound::Unbounded => Some(f64::NEG_INFINITY),
      Bound::Included(v) => Some(bound_f64(name, v)?),
      Bound::Excluded(v) => {
        let f = bound_f64(name, v)?;
        (f != f64::INFINITY).then(|| next_up(f))
      }
    }
    .map(norm_f64),
  )
}

fn float_upper(name: &str, b: &Bound<Value>) -> Result<Option<f64>> {
  Ok(
    match b {
      Bound::Unbounded => Some(f64::INFINITY),
      Bound::Included(v) => Some(bound_f64(name, v)?),
      Bound::Excluded(v) => {
        let f = bound_f64(name, v)?;
        (f != f64::NEG_INFINITY).then(|| next_down(f))
      }
    }
    .map(norm_f64),
  )
}

fn range_pred<'f>(
  name: &str,
  ty: FieldType,
  lower: &Bound<Value>,
  upper: &Bound<Value>,
) -> Result<Option<Pred<'f>>> {
  match ty {
    FieldType::Integer => {
      let lo = int_lower(name, lower)?;
      let hi = int_upper(name, upper)?;
      Ok(match (lo, hi) {
        (Some(i64::MIN), Some(i64::MAX)) => Some(Pred::Exists),
        (Some(lo), Some(hi)) if lo <= hi => Some(Pred::IntRange(lo, hi)),
        _ => None,
      })
    }
    FieldType::Float => {
      let lo = float_lower(name, lower)?;
      let hi = float_upper(name, upper)?;
      Ok(match (lo, hi) {
        (Some(lo), Some(hi))
          if lo == f64::NEG_INFINITY && hi == f64::INFINITY =>
        {
          Some(Pred::Exists)
        }
        (Some(lo), Some(hi)) if lo <= hi => Some(Pred::FloatRange(lo, hi)),
        _ => None,
      })
    }
    FieldType::Keyword | FieldType::Bool => Err(Error::invalid(format!(
      "range needs an integer or float field; {name:?} is a {} field",
      ty.name()
    ))),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn schema() -> Schema {
    Schema::new()
      .with("k", FieldType::Keyword)
      .unwrap()
      .with("i", FieldType::Integer)
      .unwrap()
      .with("f", FieldType::Float)
      .unwrap()
      .with("b", FieldType::Bool)
      .unwrap()
  }

  fn plan(f: &Filter) -> Plan<'_> {
    compile(&schema(), f).unwrap()
  }

  /// Compiles and prints, so the plan can outlive the filter.
  fn r(f: Filter) -> String {
    format!("{:?}", plan(&f))
  }

  fn p(plan: Plan) -> String {
    format!("{plan:?}")
  }

  fn err(f: &Filter) -> String {
    compile(&schema(), f).unwrap_err().to_string()
  }

  #[test]
  fn integer_bounds_from_floats() {
    assert_eq!(
      r(Filter::range("i", 1.5..=4.0)),
      p(Plan::Leaf(1, Pred::IntRange(2, 4)))
    );
    assert_eq!(
      r(Filter::range("i", 1.0..4.0)),
      p(Plan::Leaf(1, Pred::IntRange(1, 3)))
    );
    assert_eq!(r(Filter::gt("i", 1.0)), r(Filter::gte("i", 2)));
    assert_eq!(r(Filter::gt("i", 1.5)), r(Filter::gte("i", 2)));
    assert_eq!(r(Filter::lt("i", -1.5)), r(Filter::lte("i", -2)));
    assert_eq!(r(Filter::gt("i", i64::MAX)), p(Plan::Empty));
    assert_eq!(r(Filter::lt("i", i64::MIN)), p(Plan::Empty));
    assert_eq!(r(Filter::gte("i", 1e19)), p(Plan::Empty));
    assert_eq!(r(Filter::lte("i", -1e19)), p(Plan::Empty));
    assert_eq!(r(Filter::lte("i", 1e19)), p(Plan::Leaf(1, Pred::Exists)));
    assert_eq!(
      r(Filter::gt("i", f64::NEG_INFINITY)),
      r(Filter::exists("i"))
    );
    assert_eq!(r(Filter::lt("i", f64::INFINITY)), r(Filter::exists("i")));
    assert_eq!(r(Filter::gt("i", f64::INFINITY)), p(Plan::Empty));
    assert_eq!(r(Filter::range("i", 5..5)), p(Plan::Empty));
    assert_eq!(r(Filter::eq("i", 2.5)), p(Plan::Empty));
    assert_eq!(
      r(Filter::eq("i", 2.0)),
      p(Plan::Leaf(1, Pred::Ints(vec![2])))
    );
  }

  #[test]
  fn float_bounds() {
    assert_eq!(
      r(Filter::range("f", 1..=2)),
      p(Plan::Leaf(2, Pred::FloatRange(1.0, 2.0)))
    );
    assert_eq!(
      r(Filter::gt("f", 0.0)),
      p(Plan::Leaf(
        2,
        Pred::FloatRange(f64::from_bits(1), f64::INFINITY)
      ))
    );
    assert_eq!(
      r(Filter::gte("f", -0.0)),
      p(Plan::Leaf(2, Pred::FloatRange(0.0, f64::INFINITY)))
    );
    assert_eq!(r(Filter::gt("f", f64::INFINITY)), p(Plan::Empty));
    assert_eq!(r(Filter::lt("f", f64::NEG_INFINITY)), p(Plan::Empty));
    assert_eq!(r(Filter::range("f", 2.0..1.0)), p(Plan::Empty));
    assert_eq!(
      r(Filter::range::<i64>("f", ..)),
      p(Plan::Leaf(2, Pred::Exists))
    );
    assert_eq!(
      r(Filter::is_in(
        "f",
        [Value::Float(-0.0), Value::Int(0), 1.into()]
      )),
      p(Plan::Leaf(2, Pred::Floats(vec![0.0, 1.0])))
    );
  }

  #[test]
  fn folding() {
    let none = Filter::is_in("k", Vec::<&str>::new());
    assert_eq!(plan(&none), Plan::Empty);
    assert_eq!(plan(&!none.clone()), Plan::All);
    assert_eq!(plan(&Filter::and([])), Plan::All);
    assert_eq!(plan(&Filter::or([])), Plan::Empty);
    assert_eq!(plan(&(Filter::exists("k") & none.clone())), Plan::Empty);
    assert_eq!(
      plan(&(Filter::exists("k") | none.clone())),
      Plan::Leaf(0, Pred::Exists)
    );
    assert_eq!(plan(&!!Filter::exists("k")), Plan::Leaf(0, Pred::Exists));
    assert_eq!(
      plan(&Filter::and([Filter::exists("k"), Filter::and([])])),
      Plan::Leaf(0, Pred::Exists)
    );
    assert_eq!(
      plan(&Filter::is_in("k", ["b", "a", "b"])),
      Plan::Leaf(0, Pred::Terms(vec!["a", "b"]))
    );
    assert_eq!(
      plan(&Filter::is_in("b", [true, true])),
      Plan::Leaf(3, Pred::Bools([false, true]))
    );
  }

  #[test]
  fn errors_are_clear() {
    assert!(err(&Filter::eq("nope", 1)).contains("\"nope\" is not declared"));
    assert!(err(&Filter::eq("k", 1)).contains("keyword field; eq needs a"));
    assert!(err(&Filter::eq("i", "x")).contains("needs a number"));
    assert!(err(&Filter::is_in("b", [1])).contains("in needs a boolean"));
    assert!(err(&Filter::range("k", 1..)).contains("integer or float"));
    assert!(err(&Filter::range("b", 1..)).contains("bool field"));
    assert!(err(&Filter::gte("i", "x")).contains("must be a number"));
    assert!(err(&Filter::eq("f", f64::NAN)).contains("NaN"));
    assert!(err(&Filter::eq("i", f64::NAN)).contains("NaN"));
    assert!(err(&Filter::lt("f", f64::NAN)).contains("NaN"));
    // Errors are found even behind a constant-false sibling.
    let none = Filter::is_in("k", Vec::<&str>::new());
    assert!(compile(&schema(), &(none & Filter::eq("x", 1))).is_err());
    let mut deep = Filter::exists("k");
    for _ in 0..MAX_DEPTH {
      deep = !deep;
    }
    assert!(err(&deep).contains("deeper"));
  }
}
