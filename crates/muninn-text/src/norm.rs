//! One-byte document length encoding.
//!
//! Document lengths are stored as one byte per document, Lucene-style
//! (`SmallFloat.intToByte4`): lengths below 24 are exact, larger ones
//! keep four significant bits and are rounded *down*. BM25 always scores
//! with the decoded length, in every index type, so in-memory and on-disk
//! indexes rank identically. Exact lengths are still stored on disk for
//! corpus statistics and merging.

/// Values below this are stored verbatim.
const FREE: u32 = 24;

fn long_to_int4(i: u32) -> u32 {
  let num_bits = 32 - i.leading_zeros();
  if num_bits < 4 {
    return i;
  }
  let shift = num_bits - 4;
  ((i >> shift) & 0x07) | ((shift + 1) << 3)
}

fn int4_to_long(i: u32) -> u64 {
  let bits = u64::from(i & 0x07);
  let shift = (i >> 3) as i32 - 1;
  if shift == -1 {
    bits
  } else {
    (bits | 0x08) << shift
  }
}

/// Encodes a document length (in tokens) into one byte. Monotonic;
/// lengths above `i32::MAX` saturate.
pub fn encode_doc_len(len: u32) -> u8 {
  let i = len.min(i32::MAX as u32);
  if i < FREE {
    i as u8
  } else {
    (FREE + long_to_int4(i - FREE)) as u8
  }
}

/// Decodes [`encode_doc_len`]: the largest representable length that is
/// not above the original.
pub fn decode_doc_len(norm: u8) -> u32 {
  let i = u32::from(norm);
  if i < FREE {
    i
  } else {
    FREE + int4_to_long(i - FREE) as u32
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn small_lengths_are_exact() {
    for len in 0..32 {
      assert_eq!(decode_doc_len(encode_doc_len(len)), len);
    }
  }

  #[test]
  fn monotonic_and_rounds_down() {
    let mut prev = 0u8;
    let mut len = 0u32;
    while len < 100_000_000 {
      let n = encode_doc_len(len);
      assert!(n >= prev, "not monotonic at {len}");
      assert!(decode_doc_len(n) <= len);
      // The decoded value encodes to itself.
      assert_eq!(encode_doc_len(decode_doc_len(n)), n);
      prev = n;
      len += 1 + len / 50;
    }
    assert_eq!(encode_doc_len(u32::MAX), 255);
    let mut last = 0;
    for n in 0..=255u8 {
      let d = decode_doc_len(n);
      assert!(n == 0 || d > last, "decode not strictly increasing");
      last = d;
    }
  }
}
