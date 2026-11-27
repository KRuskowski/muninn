//! BM25 scoring, exactly as every muninn text index computes it.
//!
//! For a query with distinct analysed terms `t_1..t_n` (duplicates in the
//! query count once), the score of document `d` is
//!
//! ```text
//! score(d) = sum over i with tf_i(d) > 0 of
//!              w_i * tf_i / (tf_i + k1 * (1 - b + b * len(d) / avgdl))
//! w_i      = idf(df_i) * (k1 + 1)
//! idf(df)  = ln(1 + (N - df + 0.5) / (df + 0.5)),  df clamped to <= N
//! ```
//!
//! with `k1 = 1.2`, `b = 0.75`, `N` the number of documents, `avgdl` the
//! average exact document length (1 when there are no tokens), and
//! `len(d)` the document length after the one-byte quantisation in
//! [`crate::norm`]. Terms are summed in query order in `f32`, so equal
//! inputs give bit-identical scores and ties are broken by smaller id.
//!
//! `N`, `avgdl` and `df` come from the index itself, or from
//! [`crate::CorpusStats`] supplied by the caller so that scores from
//! several segments are comparable.

use crate::impacts::{IMPACTS, Impacts, TF_INF, impact_norm, impact_tf};
use crate::norm::decode_doc_len;

/// Term frequency saturation.
pub const K1: f32 = 1.2;
/// Length normalisation strength.
pub const B: f32 = 0.75;

/// Inverse document frequency of a term in `doc_freq` of `doc_count`
/// documents. Always positive.
pub fn idf(doc_freq: u64, doc_count: u64) -> f32 {
  let n = doc_count as f64;
  let df = doc_freq.min(doc_count) as f64;
  (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32
}

/// Average document length; 1 when there are no documents or tokens.
pub fn avg_doc_len(total_doc_len: u64, doc_count: u64) -> f32 {
  if doc_count == 0 || total_doc_len == 0 {
    1.0
  } else {
    (total_doc_len as f64 / doc_count as f64) as f32
  }
}

/// Query-time constants: the length-normalisation term for every
/// one-byte document length.
pub(crate) struct LenNorms {
  pub kc: [f32; 256],
}

impl LenNorms {
  pub(crate) fn new(avgdl: f32) -> Self {
    let mut kc = [0f32; 256];
    for (n, k) in kc.iter_mut().enumerate() {
      let len = decode_doc_len(n as u8) as f32;
      *k = K1 * (1.0 - B + B * (len / avgdl));
    }
    Self { kc }
  }
}

/// One term's contribution. `kc` is `LenNorms::kc[norm]`.
#[inline(always)]
pub(crate) fn contrib(w: f32, tf: u32, kc: f32) -> f32 {
  let tf = tf as f32;
  w * tf / (tf + kc)
}

/// Upper bound of [`contrib`] over a block with these impacts.
#[inline]
pub(crate) fn bound(imp: &Impacts, w: f32, norms: &LenNorms) -> f32 {
  let mut best = 0f32;
  for &i in imp.iter().take(IMPACTS) {
    if i == 0 {
      break;
    }
    let tf = impact_tf(i);
    let v = if tf == TF_INF {
      w
    } else {
      contrib(w, tf, norms.kc[usize::from(impact_norm(i))])
    };
    best = best.max(v);
  }
  best
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn idf_is_positive_and_decreasing() {
    let mut prev = f32::INFINITY;
    for df in 0..=100 {
      let v = idf(df, 100);
      assert!(v > 0.0 && v < prev);
      prev = v;
    }
    assert_eq!(idf(500, 100), idf(100, 100));
    assert!(idf(0, 0) > 0.0);
    assert_eq!(avg_doc_len(0, 0), 1.0);
    assert_eq!(avg_doc_len(30, 10), 3.0);
  }

  #[test]
  fn contribution_monotonic_in_tf_and_len() {
    let norms = LenNorms::new(7.3);
    let w = 3.1;
    for n in 0..255usize {
      assert!(norms.kc[n] <= norms.kc[n + 1]);
      for tf in 1..300 {
        assert!(contrib(w, tf, norms.kc[n]) <= contrib(w, tf + 1, norms.kc[n]));
        assert!(contrib(w, tf, norms.kc[n + 1]) <= contrib(w, tf, norms.kc[n]));
      }
    }
    assert!(contrib(w, u32::MAX, norms.kc[0]) <= w);
  }
}
