//! LEB128 variable-length integers.

/// Appends `v` as LEB128 (7 bits per byte, low bits first).
#[inline]
pub(crate) fn put(out: &mut Vec<u8>, mut v: u64) {
  while v >= 0x80 {
    out.push((v as u8) | 0x80);
    v >>= 7;
  }
  out.push(v as u8);
}

/// Reads a LEB128 value at `*pos` and advances it. `None` if the input
/// ends early or the value does not fit in 64 bits.
#[inline]
pub(crate) fn get(src: &[u8], pos: &mut usize) -> Option<u64> {
  let mut v = 0u64;
  let mut shift = 0u32;
  loop {
    let b = *src.get(*pos)?;
    *pos += 1;
    if shift == 63 && b > 1 {
      return None;
    }
    v |= u64::from(b & 0x7f) << shift;
    if b < 0x80 {
      return Some(v);
    }
    shift += 7;
    if shift > 63 {
      return None;
    }
  }
}

/// Reads a LEB128 value that must fit in a `u32`.
#[inline]
pub(crate) fn get_u32(src: &[u8], pos: &mut usize) -> Option<u32> {
  get(src, pos).and_then(|v| u32::try_from(v).ok())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn round_trips_edges() {
    let values = [0, 1, 127, 128, 300, u32::MAX as u64, u64::MAX];
    let mut buf = Vec::new();
    for &v in &values {
      put(&mut buf, v);
    }
    let mut pos = 0;
    for &v in &values {
      assert_eq!(get(&buf, &mut pos), Some(v));
    }
    assert_eq!(pos, buf.len());
    assert_eq!(get(&buf, &mut pos), None);
  }

  #[test]
  fn rejects_truncated_and_overlong() {
    assert_eq!(get(&[0x80], &mut 0), None);
    assert_eq!(get(&[0xff; 11], &mut 0), None);
    let mut buf = Vec::new();
    put(&mut buf, u32::MAX as u64 + 1);
    assert_eq!(get_u32(&buf, &mut 0), None);
  }
}
