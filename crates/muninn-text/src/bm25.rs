//! BM25 scoring, exactly as every muninn text index computes it.
//!
//! For a query with distinct analysed terms `t_1..t_n` (duplicates in the
//! query count once), the score of document `d` is
//!
//! ```text
//! score(d) = sum over i with tf_i(d) > 0 of  w_i / (1 + K(d) / tf_i)
//! K(d)     = k1 * (1 - b + b * len(d) / avgdl)
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
//! `w / (1 + K / tf)` equals the textbook `w * tf / (tf + K)`, but unlike
//! it is monotonic under `f32` rounding: every operation is monotonic in
//! its operand, so the contribution never decreases with `tf`, never
//! increases with `K` (document length), and never exceeds `w`. The
//! search's block bounds depend on exactly that.
//!
//! `N`, `avgdl` and `df` come from the index itself, or from
//! [`crate::CorpusStats`] supplied by the caller so that scores from
//! several segments are comparable.

use std::sync::{Arc, Mutex};

use crate::impacts::{Impacts, TF_INF, impact_norm, impact_tf};
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

/// Term frequencies up to this have their quotients `kc / tf` in a
/// per-query table ([`LenNorms::quot`]).
pub(crate) const TF_TABLE: usize = 16;

/// Query-time constants: the length-normalisation term for every
/// one-byte document length, and the quotients `kc / tf` for small tf.
pub(crate) struct LenNorms {
  avgdl: f32,
  pub kc: [f32; 256],
  /// `q[tf][norm] = kc[norm] / tf`, the same rounded value [`contrib`]
  /// computes, for `tf <= TF_TABLE`. Row 0 is infinity (`kc / 0`), so
  /// unused impact slots need no branch.
  q: [[f32; 256]; TF_TABLE + 1],
}

impl LenNorms {
  pub(crate) fn new(avgdl: f32) -> Self {
    let mut kc = [0f32; 256];
    for (n, k) in kc.iter_mut().enumerate() {
      let len = decode_doc_len(n as u8) as f32;
      *k = K1 * (1.0 - B + B * (len / avgdl));
    }
    let mut q = [[0f32; 256]; TF_TABLE + 1];
    for (tf, row) in q.iter_mut().enumerate() {
      for (v, &k) in row.iter_mut().zip(&kc) {
        *v = k / tf as f32;
      }
    }
    Self { avgdl, kc, q }
  }

  /// `kc[norm] / tf`, rounded exactly as in [`contrib`].
  #[inline(always)]
  pub(crate) fn quot(&self, tf: u32, norm: u8) -> f32 {
    match self.q.get(tf as usize) {
      Some(row) => row[usize::from(norm)],
      None => self.kc[usize::from(norm)] / tf as f32,
    }
  }

  /// `contrib(w, tf, kc[norm])`, bit for bit, with one division fewer
  /// for small tf.
  #[inline(always)]
  pub(crate) fn score(&self, w: f32, tf: u32, norm: u8) -> f32 {
    w / (1.0 + self.quot(tf, norm))
  }

  /// Smallest quotient `kc / tf` over the frontier points of `imp`
  /// (0 for a saturated tf, infinity if there are none). The block's
  /// bound for weight `w` is `w / (1 + qmin)`.
  #[inline]
  pub(crate) fn qmin(&self, imp: &Impacts) -> f32 {
    let mut m = f32::INFINITY;
    for &i in imp {
      let tf = impact_tf(i);
      let n = usize::from(impact_norm(i));
      // Unused slots are 0, which row 0 maps to infinity.
      let q = if (tf as usize) <= TF_TABLE {
        self.q[tf as usize][n]
      } else if tf == TF_INF {
        0.0
      } else {
        self.kc[n] / tf as f32
      };
      m = fmin(m, q);
    }
    m
  }
}

/// The smaller of two non-NaN floats (one instruction, unlike
/// `f32::min`, which also handles NaN).
#[inline(always)]
pub(crate) fn fmin(a: f32, b: f32) -> f32 {
  if b < a { b } else { a }
}

/// The last [`LenNorms`] an index built, reused while the average
/// document length stays the same (a frozen index searched with its own
/// or a collection's unchanging statistics), which saves building the
/// table on every query.
#[derive(Default)]
pub(crate) struct NormsCache(Mutex<Option<Arc<LenNorms>>>);

impl NormsCache {
  /// The table for `avgdl`, built if the cached one is for another.
  pub(crate) fn get(&self, avgdl: f32) -> Arc<LenNorms> {
    let lock = || self.0.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(n) = &*lock()
      && n.avgdl.to_bits() == avgdl.to_bits()
    {
      return n.clone();
    }
    let n = Arc::new(LenNorms::new(avgdl));
    *lock() = Some(n.clone());
    n
  }
}

impl Clone for NormsCache {
  fn clone(&self) -> Self {
    Self::default()
  }
}

impl std::fmt::Debug for NormsCache {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("NormsCache")
  }
}

/// The bound `w / (1 + qmin)` for a smallest quotient from
/// [`LenNorms::qmin`].
#[inline]
pub(crate) fn bound_from_qmin(w: f32, qmin: f32) -> f32 {
  if qmin == f32::INFINITY {
    0.0
  } else {
    w / (1.0 + qmin)
  }
}

/// One term's contribution. `kc` is `LenNorms::kc[norm]`.
#[inline(always)]
pub(crate) fn contrib(w: f32, tf: u32, kc: f32) -> f32 {
  // Not `w * tf / (tf + kc)`: that is not monotonic in tf after
  // rounding (see the module docs).
  w / (1.0 + kc / tf as f32)
}

/// Upper bound of [`contrib`] over a block with these impacts: exactly
/// the largest `contrib` over the frontier points (with `w` for a
/// saturated tf).
///
/// Every rounded operation in `contrib` is monotonic, so the point with
/// the smallest rounded quotient `kc / tf` has the largest contribution;
/// this takes that minimum and divides once, bit-identical to
/// evaluating `contrib` at every point. For the same reason the largest
/// bound over several blocks is `w / (1 + smallest qmin)`.
#[inline]
pub(crate) fn bound(imp: &Impacts, w: f32, norms: &LenNorms) -> f32 {
  bound_from_qmin(w, norms.qmin(imp))
}

/// [`bound`] by evaluating [`contrib`] at every frontier point: the
/// reference the fast form is tested against.
#[cfg(test)]
pub(crate) fn bound_reference(imp: &Impacts, w: f32, norms: &LenNorms) -> f32 {
  let mut best = 0f32;
  for &i in imp {
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
  use crate::impacts::IMPACTS;

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
  fn fast_bound_equals_reference_bit_for_bit() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    for avgdl in [0.3f32, 1.0, 7.3, 56.0, 1.0e4, 3.0e9] {
      let norms = LenNorms::new(avgdl);
      for _ in 0..20_000 {
        let w = (rnd() % 10_000) as f32 / 97.0;
        let n = (rnd() % (IMPACTS as u64 + 1)) as usize;
        let mut imp = [0u32; IMPACTS];
        for slot in imp.iter_mut().take(n) {
          let tf = match rnd() % 4 {
            0 => TF_INF,
            1 => 1 + (rnd() % 3) as u32,
            _ => 1 + (rnd() % 65_534) as u32,
          };
          *slot = tf | (((rnd() % 256) as u32) << 16);
        }
        let fast = bound(&imp, w, &norms);
        let slow = bound_reference(&imp, w, &norms);
        assert_eq!(fast.to_bits(), slow.to_bits(), "{imp:?} w {w} {avgdl}");
      }
    }
  }

  #[test]
  fn table_score_equals_contrib_bit_for_bit() {
    for avgdl in [0.3f32, 1.0, 7.3, 56.0, 1.0e4, 3.0e9] {
      let norms = LenNorms::new(avgdl);
      for w in [0.0123f32, 1.0, 3.1, 47.9] {
        for n in 0..=255u8 {
          for tf in (0..=40u32).chain([65_535, u32::MAX]) {
            let a = norms.score(w, tf, n);
            let b = contrib(w, tf, norms.kc[usize::from(n)]);
            assert_eq!(a.to_bits(), b.to_bits(), "tf {tf} n {n} w {w}");
          }
        }
      }
    }
  }

  #[test]
  fn contribution_monotonic_in_tf_and_len() {
    // Block bounds rely on exact (not approximate) monotonicity: a
    // document with lower tf or a longer length must never score above
    // the frontier point that dominates it, after f32 rounding, for any
    // supplied statistics. And no contribution may exceed `w`, which is
    // the bound used for saturated tf.
    let mut row = vec![0f32; 65536];
    for avgdl in [0.3f32, 1.0, 7.3, 120.0, 1.0e4, 1.0e7, 3.0e9] {
      let norms = LenNorms::new(avgdl);
      for w in [0.0123f32, 3.1, 47.9] {
        for n in 0..256usize {
          if n > 0 {
            assert!(norms.kc[n - 1] <= norms.kc[n]);
          }
          let kc = norms.kc[n];
          let mut prev = 0f32;
          for tf in 1..=65535u32 {
            let c = contrib(w, tf, kc);
            assert!(c >= prev, "tf {tf} avgdl {avgdl} norm {n} w {w}");
            assert!(c <= w, "above w: tf {tf} avgdl {avgdl} norm {n}");
            if n > 0 {
              assert!(c <= row[tf as usize], "len: tf {tf} avgdl {avgdl}");
            }
            row[tf as usize] = c;
            prev = c;
          }
          assert!(contrib(w, u32::MAX, kc) <= w);
        }
      }
    }
  }
}
