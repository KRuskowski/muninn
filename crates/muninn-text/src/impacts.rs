//! Per-block score bounds that stay valid under any corpus statistics.
//!
//! BM25's per-term contribution `w * tf / (tf + k1 * (1 - b + b * len /
//! avgdl))` grows with `tf` and shrinks with `len`, for every
//! non-negative weight `w` and every positive `avgdl`. So instead of a
//! precomputed maximum score (which would bake in one segment's IDF and
//! average length), each block stores the Pareto frontier of its `(tf, norm)` pairs: no stored pair
//! has both a higher `tf` and a shorter document than another. The block
//! bound for any query-time statistics is the maximum contribution over
//! those few pairs.
//!
//! The frontier is capped at [`IMPACTS`] points. Two neighbours are
//! merged into `(larger tf, smaller norm)`, which dominates both, so the
//! cap loosens the bound but never invalidates it. A `tf` of
//! [`TF_INF`] stands for "65535 or more" and bounds the contribution by
//! its supremum `w`.

use crate::bitpack::BLOCK;

/// Frontier points stored per block.
pub(crate) const IMPACTS: usize = 6;

/// Saturated term frequency meaning "at least this much".
pub(crate) const TF_INF: u32 = 0xFFFF;

/// Up to [`IMPACTS`] packed points `tf | norm << 16`; zero means unused
/// (real term frequencies are at least 1).
pub(crate) type Impacts = [u32; IMPACTS];

#[inline]
pub(crate) fn impact_tf(i: u32) -> u32 {
  i & 0xFFFF
}

#[inline]
pub(crate) fn impact_norm(i: u32) -> u8 {
  (i >> 16) as u8
}

/// The capped Pareto frontier of `(tf, norm)` points.
pub(crate) fn frontier(points: &[(u32, u8)]) -> Impacts {
  debug_assert!(points.len() <= BLOCK);
  let mut pts: Vec<(u32, u8)> =
    points.iter().map(|&(tf, n)| (tf.min(TF_INF), n)).collect();
  // Highest tf first; among equal tf the shortest document first.
  pts.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
  let mut front: Vec<(u32, u8)> = Vec::with_capacity(IMPACTS * 2);
  for p in pts {
    // Keep p only if it is shorter than every point with a higher tf.
    if front.last().is_none_or(|l| p.1 < l.1) {
      front.push(p);
    }
  }
  // `front` now has strictly decreasing tf and strictly decreasing norm.
  while front.len() > IMPACTS {
    let mut best = 0;
    let mut best_gap = u8::MAX;
    for j in 0..front.len() - 1 {
      let gap = front[j].1 - front[j + 1].1;
      if gap < best_gap {
        best_gap = gap;
        best = j;
      }
    }
    front[best].1 = front[best + 1].1;
    front.remove(best + 1);
  }
  let mut out = [0u32; IMPACTS];
  for (o, &(tf, n)) in out.iter_mut().zip(&front) {
    *o = tf | (u32::from(n) << 16);
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  fn unpack(imp: &Impacts) -> Vec<(u32, u8)> {
    imp
      .iter()
      .filter(|&&i| i != 0)
      .map(|&i| (impact_tf(i), impact_norm(i)))
      .collect()
  }

  fn dominated(p: (u32, u8), front: &[(u32, u8)]) -> bool {
    front
      .iter()
      .any(|&(tf, n)| tf >= p.0.min(TF_INF) && n <= p.1)
  }

  #[test]
  fn exact_frontier_when_small() {
    let pts = [(1, 3), (2, 5), (1, 9), (3, 20), (2, 4), (1, 3)];
    let f = unpack(&frontier(&pts));
    assert_eq!(f, vec![(3, 20), (2, 4), (1, 3)]);
  }

  #[test]
  fn capped_frontier_still_dominates_every_point() {
    let mut state = 12345u64;
    for round in 0..200 {
      let n = 1 + round % 128;
      let pts: Vec<(u32, u8)> = (0..n)
        .map(|_| {
          state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
          let tf = 1 + ((state >> 33) % 70_000) as u32;
          // Correlate tf with length so frontiers are long.
          let norm = ((tf / 300) as u8).wrapping_add((state >> 20) as u8 % 7);
          (tf, norm)
        })
        .collect();
      let f = unpack(&frontier(&pts));
      assert!(f.len() <= IMPACTS && !f.is_empty());
      for &p in &pts {
        assert!(dominated(p, &f), "{p:?} not dominated by {f:?}");
      }
    }
  }
}
