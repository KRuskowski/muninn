/// How two vectors are compared.
///
/// Internally every index works with a *distance* where lower is closer,
/// so a single top-k structure serves all metrics:
///
/// | metric   | distance            | public score            |
/// |----------|---------------------|-------------------------|
/// | `L2`     | squared L2          | `-distance`             |
/// | `Dot`    | `-dot(a, b)`        | `dot(a, b)`             |
/// | `Cosine` | `1 - dot(a, b)`     | `1 - distance` (cosine) |
///
/// Cosine vectors are normalised once at insert time, so cosine search
/// costs exactly one dot product per candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Metric {
  L2 = 0,
  Dot = 1,
  Cosine = 2,
}

impl Metric {
  /// Converts an internal distance into the public "higher is better"
  /// score carried by [`crate::Hit`].
  #[inline]
  pub fn score(self, distance: f32) -> f32 {
    match self {
      Self::L2 | Self::Dot => -distance,
      Self::Cosine => 1.0 - distance,
    }
  }

  /// Whether vectors must be normalised before they are stored.
  #[inline]
  pub fn normalizes(self) -> bool {
    matches!(self, Self::Cosine)
  }

  /// Stable on-disk tag.
  pub fn to_u8(self) -> u8 {
    self as u8
  }

  /// Inverse of [`Metric::to_u8`].
  pub fn from_u8(tag: u8) -> Option<Self> {
    match tag {
      0 => Some(Self::L2),
      1 => Some(Self::Dot),
      2 => Some(Self::Cosine),
      _ => None,
    }
  }

  /// Parses `"l2"`, `"dot"` or `"cosine"` (case-insensitive).
  pub fn parse(name: &str) -> Option<Self> {
    match name.to_ascii_lowercase().as_str() {
      "l2" | "euclidean" => Some(Self::L2),
      "dot" | "ip" | "inner_product" => Some(Self::Dot),
      "cosine" | "cos" => Some(Self::Cosine),
      _ => None,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tags_round_trip() {
    for m in [Metric::L2, Metric::Dot, Metric::Cosine] {
      assert_eq!(Metric::from_u8(m.to_u8()), Some(m));
    }
    assert_eq!(Metric::from_u8(9), None);
  }

  #[test]
  fn scores_are_higher_is_better() {
    assert!(Metric::L2.score(1.0) > Metric::L2.score(2.0));
    assert!(Metric::Dot.score(-3.0) > Metric::Dot.score(-1.0));
    assert_eq!(Metric::Cosine.score(0.0), 1.0);
  }
}
