//! Scalar values compared by filters, and the number conversions that
//! give integer and float fields their exact semantics.

use std::fmt;

/// A scalar on the right-hand side of `eq`, `in` and `range`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
  /// A string. Compared with keyword fields, byte for byte.
  Str(String),
  /// A signed integer. Compared with integer and float fields.
  Int(i64),
  /// A float. Compared with float fields, and with integer fields when it
  /// is a whole number (see the crate docs).
  Float(f64),
  /// A boolean. Compared with bool fields.
  Bool(bool),
}

impl Value {
  /// A short name of the value's type, for error messages.
  pub fn type_name(&self) -> &'static str {
    match self {
      Self::Str(_) => "string",
      Self::Int(_) => "integer",
      Self::Float(_) => "float",
      Self::Bool(_) => "boolean",
    }
  }

  /// The value as an `f64`, if it is a number.
  pub fn as_f64(&self) -> Option<f64> {
    match *self {
      Self::Int(i) => Some(i as f64),
      Self::Float(f) => Some(f),
      _ => None,
    }
  }
}

impl fmt::Display for Value {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Str(s) => write!(f, "{s:?}"),
      Self::Int(i) => write!(f, "{i}"),
      Self::Float(x) => write!(f, "{x:?}"),
      Self::Bool(b) => write!(f, "{b}"),
    }
  }
}

impl From<&str> for Value {
  fn from(s: &str) -> Self {
    Self::Str(s.to_string())
  }
}

impl From<String> for Value {
  fn from(s: String) -> Self {
    Self::Str(s)
  }
}

impl From<&String> for Value {
  fn from(s: &String) -> Self {
    Self::Str(s.clone())
  }
}

impl From<bool> for Value {
  fn from(b: bool) -> Self {
    Self::Bool(b)
  }
}

impl From<f64> for Value {
  fn from(f: f64) -> Self {
    Self::Float(f)
  }
}

impl From<f32> for Value {
  fn from(f: f32) -> Self {
    Self::Float(f64::from(f))
  }
}

macro_rules! from_int {
  ($($t:ty),*) => {
    $(impl From<$t> for Value {
      fn from(i: $t) -> Self {
        Self::Int(i64::from(i))
      }
    })*
  };
}

from_int!(i8, i16, i32, i64, u8, u16, u32);

/// 2^63 as an `f64`: the first float above every `i64`.
pub(crate) const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

/// Folds `-0.0` into `0.0`, so equality and ordering of stored floats
/// agree and plain `<` matches `f64::total_cmp` (NaN never reaches here).
#[inline]
pub(crate) fn norm_f64(f: f64) -> f64 {
  if f == 0.0 { 0.0 } else { f }
}

/// The whole number `f` as an `i64`, if `f` is one and lies strictly
/// between -2^63 and 2^63.
///
/// -2^63 itself is excluded although it equals `i64::MIN`: JSON integers
/// below `i64::MIN` are parsed as floats and round to exactly -2^63, so
/// as a float it stands for "below every integer", just as 2^63 stands
/// for "above every integer". `i64::MIN` written as a JSON integer is
/// parsed as an integer and is unaffected.
#[inline]
pub(crate) fn exact_i64(f: f64) -> Option<i64> {
  (f.fract() == 0.0 && f > -TWO_POW_63 && f < TWO_POW_63).then_some(f as i64)
}

/// The integer value of a JSON number, if it is a whole number that fits
/// in an `i64` (`3`, `3.0` and `3e0` all qualify; `3.5` does not).
#[inline]
pub(crate) fn json_i64(n: &serde_json::Number) -> Option<i64> {
  if let Some(i) = n.as_i64() {
    return Some(i);
  }
  if n.is_u64() {
    return None;
  }
  exact_i64(n.as_f64()?)
}

/// The float value of a JSON number. serde_json never holds NaN or
/// infinities, so the result is finite.
#[inline]
pub(crate) fn json_f64(n: &serde_json::Number) -> Option<f64> {
  n.as_f64().filter(|f| f.is_finite()).map(norm_f64)
}

/// The smallest float greater than `f` (`f` itself for NaN and +inf).
pub(crate) fn next_up(f: f64) -> f64 {
  if f.is_nan() || f == f64::INFINITY {
    return f;
  }
  if f == 0.0 {
    return f64::from_bits(1);
  }
  let b = f.to_bits();
  if f > 0.0 {
    f64::from_bits(b + 1)
  } else {
    f64::from_bits(b - 1)
  }
}

/// The largest float less than `f` (`f` itself for NaN and -inf).
pub(crate) fn next_down(f: f64) -> f64 {
  -next_up(-f)
}

/// Maps a float to an `i64` whose signed order is the float's total
/// order. Its own inverse via [`key_f64`].
#[inline]
pub(crate) fn f64_key(f: f64) -> i64 {
  let b = f.to_bits() as i64;
  b ^ (((b >> 63) as u64) >> 1) as i64
}

/// Inverse of [`f64_key`].
#[inline]
pub(crate) fn key_f64(k: i64) -> f64 {
  f64::from_bits((k ^ (((k >> 63) as u64) >> 1) as i64) as u64)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn float_key_preserves_order_and_round_trips() {
    let xs = [
      f64::NEG_INFINITY,
      -f64::MAX,
      -1e10,
      -2.0,
      -1.0,
      -f64::MIN_POSITIVE,
      -f64::from_bits(1),
      0.0,
      f64::from_bits(1),
      f64::MIN_POSITIVE,
      1.0,
      2.5,
      1e300,
      f64::MAX,
      f64::INFINITY,
    ];
    for w in xs.windows(2) {
      assert!(f64_key(w[0]) < f64_key(w[1]), "{} vs {}", w[0], w[1]);
    }
    for &x in &xs {
      assert_eq!(key_f64(f64_key(x)).to_bits(), x.to_bits());
    }
  }

  #[test]
  fn next_up_down() {
    assert_eq!(next_up(0.0), f64::from_bits(1));
    assert_eq!(next_up(-0.0), f64::from_bits(1));
    assert_eq!(next_down(0.0), -f64::from_bits(1));
    assert_eq!(next_up(1.0), 1.0 + f64::EPSILON);
    assert_eq!(next_up(-f64::from_bits(1)), 0.0);
    assert_eq!(next_up(f64::MAX), f64::INFINITY);
    assert_eq!(next_up(f64::NEG_INFINITY), -f64::MAX);
    assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
    assert_eq!(next_down(f64::NEG_INFINITY), f64::NEG_INFINITY);
  }

  #[test]
  fn exact_integers() {
    assert_eq!(exact_i64(3.0), Some(3));
    assert_eq!(exact_i64(-0.0), Some(0));
    assert_eq!(exact_i64(3.5), None);
    assert_eq!(exact_i64(TWO_POW_63), None);
    assert_eq!(exact_i64(-TWO_POW_63), None);
    assert_eq!(
      exact_i64(-9_223_372_036_854_774_784.0),
      Some(i64::MIN + 1024)
    );
    let n: serde_json::Number =
      serde_json::from_str("-9223372036854775809").unwrap();
    assert_eq!(json_i64(&n), None);
    let n: serde_json::Number =
      serde_json::from_str("-9223372036854775808").unwrap();
    assert_eq!(json_i64(&n), Some(i64::MIN));
    assert_eq!(exact_i64(f64::INFINITY), None);
    assert_eq!(exact_i64(f64::NAN), None);
    let n: serde_json::Number = serde_json::from_str("3e0").unwrap();
    assert_eq!(json_i64(&n), Some(3));
    let n: serde_json::Number =
      serde_json::from_str("18446744073709551615").unwrap();
    assert_eq!(json_i64(&n), None);
  }
}
