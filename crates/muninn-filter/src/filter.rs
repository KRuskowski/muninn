//! The filter language: a tree of predicates on payload fields.

use std::ops::{BitAnd, BitOr, Bound, Not, RangeBounds};

use crate::Value;

/// A boolean predicate over a document's payload fields.
///
/// Build filters with the helpers and operators, or parse the JSON form
/// with [`Filter::from_json`] / [`Filter::parse`]:
///
/// ```
/// use muninn_filter::Filter;
///
/// let f = Filter::eq("lang", "en")
///   & Filter::range("year", 2020..)
///   & !Filter::is_in("tag", ["a", "b"]);
/// assert_eq!(
///   f.to_json().unwrap().to_string(),
///   r#"{"and":[{"eq":{"lang":"en"}},{"range":{"year":{"gte":2020}}},"#
///     .to_string()
///     + r#"{"not":{"in":{"tag":["a","b"]}}}]}"#
/// );
/// ```
///
/// A field holding an array matches `Eq`, `In` and `Range` when any one
/// element matches; `Not` is the complement of the matching set, so
/// `!Filter::eq("tag", "a")` keeps documents with no `"a"` element,
/// including documents without the field.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
  /// The field holds this value (any element, for arrays).
  Eq(String, Value),
  /// The field holds one of these values. An empty list matches nothing.
  In(String, Vec<Value>),
  /// The field holds a number within the bounds. Both bounds unbounded
  /// matches every document with a number in the field.
  Range {
    /// Field path.
    field: String,
    /// `Included` is `gte`, `Excluded` is `gt`.
    lower: Bound<Value>,
    /// `Included` is `lte`, `Excluded` is `lt`.
    upper: Bound<Value>,
  },
  /// The field holds at least one value of its declared type.
  Exists(String),
  /// Every child matches. An empty list matches every document.
  And(Vec<Filter>),
  /// At least one child matches. An empty list matches nothing.
  Or(Vec<Filter>),
  /// The child does not match.
  Not(Box<Filter>),
}

impl Filter {
  /// `field == value`.
  pub fn eq(field: impl Into<String>, value: impl Into<Value>) -> Self {
    Self::Eq(field.into(), value.into())
  }

  /// `field` is one of `values`.
  pub fn is_in<V: Into<Value>>(
    field: impl Into<String>,
    values: impl IntoIterator<Item = V>,
  ) -> Self {
    Self::In(field.into(), values.into_iter().map(Into::into).collect())
  }

  /// `field` within a Rust range: `range("year", 2020..2030)` is
  /// `gte 2020, lt 2030`; `range("price", ..=9.5)` is `lte 9.5`. A
  /// `(Bound, Bound)` pair expresses exclusive lower bounds.
  pub fn range<V: Into<Value> + Clone>(
    field: impl Into<String>,
    range: impl RangeBounds<V>,
  ) -> Self {
    Self::Range {
      field: field.into(),
      lower: range.start_bound().map(|v| v.clone().into()),
      upper: range.end_bound().map(|v| v.clone().into()),
    }
  }

  /// `field > value`.
  pub fn gt(field: impl Into<String>, value: impl Into<Value>) -> Self {
    Self::bounds(field, Bound::Excluded(value.into()), Bound::Unbounded)
  }

  /// `field >= value`.
  pub fn gte(field: impl Into<String>, value: impl Into<Value>) -> Self {
    Self::bounds(field, Bound::Included(value.into()), Bound::Unbounded)
  }

  /// `field < value`.
  pub fn lt(field: impl Into<String>, value: impl Into<Value>) -> Self {
    Self::bounds(field, Bound::Unbounded, Bound::Excluded(value.into()))
  }

  /// `field <= value`.
  pub fn lte(field: impl Into<String>, value: impl Into<Value>) -> Self {
    Self::bounds(field, Bound::Unbounded, Bound::Included(value.into()))
  }

  fn bounds(
    field: impl Into<String>,
    lower: Bound<Value>,
    upper: Bound<Value>,
  ) -> Self {
    Self::Range {
      field: field.into(),
      lower,
      upper,
    }
  }

  /// `field` holds a value of its declared type.
  pub fn exists(field: impl Into<String>) -> Self {
    Self::Exists(field.into())
  }

  /// All of `filters`.
  pub fn and(filters: impl IntoIterator<Item = Filter>) -> Self {
    Self::And(filters.into_iter().collect())
  }

  /// Any of `filters`.
  pub fn or(filters: impl IntoIterator<Item = Filter>) -> Self {
    Self::Or(filters.into_iter().collect())
  }

  /// Depth of the tree: 1 for a leaf.
  pub fn depth(&self) -> usize {
    match self {
      Self::And(c) | Self::Or(c) => {
        1 + c.iter().map(Filter::depth).max().unwrap_or(0)
      }
      Self::Not(c) => 1 + c.depth(),
      _ => 1,
    }
  }
}

impl BitAnd for Filter {
  type Output = Filter;

  /// `a & b`, flattening nested `And`s.
  fn bitand(self, rhs: Filter) -> Filter {
    let mut v = match self {
      Self::And(v) => v,
      f => vec![f],
    };
    match rhs {
      Self::And(r) => v.extend(r),
      f => v.push(f),
    }
    Self::And(v)
  }
}

impl BitOr for Filter {
  type Output = Filter;

  /// `a | b`, flattening nested `Or`s.
  fn bitor(self, rhs: Filter) -> Filter {
    let mut v = match self {
      Self::Or(v) => v,
      f => vec![f],
    };
    match rhs {
      Self::Or(r) => v.extend(r),
      f => v.push(f),
    }
    Self::Or(v)
  }
}

impl Not for Filter {
  type Output = Filter;

  /// `!a`.
  fn not(self) -> Filter {
    Self::Not(Box::new(self))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn builders_and_operators() {
    let f = Filter::eq("a", 1) & Filter::eq("b", "x") & Filter::exists("c");
    assert!(matches!(&f, Filter::And(v) if v.len() == 3));
    let g = (Filter::eq("a", 1) | Filter::eq("a", 2)) | Filter::eq("a", 3);
    assert!(matches!(&g, Filter::Or(v) if v.len() == 3));
    assert_eq!(g.depth(), 2);
    assert_eq!((!f).depth(), 3);
    assert_eq!(
      Filter::range("y", 1..=5),
      Filter::Range {
        field: "y".into(),
        lower: Bound::Included(Value::Int(1)),
        upper: Bound::Included(Value::Int(5)),
      }
    );
    assert_eq!(Filter::range("p", ..2.5), Filter::lt("p", 2.5),);
    assert_eq!(
      Filter::range("p", (Bound::Excluded(1), Bound::Unbounded)),
      Filter::gt("p", 1)
    );
    assert_eq!(Filter::gte("p", 1), Filter::range("p", 1..));
    assert_eq!(Filter::lte("p", 1), Filter::range("p", ..=1));
    assert_eq!(
      Filter::is_in("t", ["a", "b"]),
      Filter::In("t".into(), vec!["a".into(), "b".into()])
    );
    assert_eq!(Filter::and([]), Filter::And(vec![]));
    assert_eq!(Filter::or([Filter::exists("x")]).depth(), 2);
  }
}
