//! On-disk format of WAL segment files.
//!
//! ```text
//! segment header (32 bytes)
//!   0  magic "MUNWAL\0\1"
//!   8  version u32
//!  12  crc32c u32 of bytes 0..12 and 16..32
//!  16  first_lsn u64: LSN of the first record in this file
//!  24  prev_lsn u64: last LSN of the log when this file was created
//! records, back to back
//!   0  len u32: payload length
//!   4  crc u32: masked crc32c of the len bytes followed by the payload
//!   8  payload (len bytes)
//! ```
//!
//! All integers are little-endian. A record's LSN is implicit: the file's
//! `first_lsn` plus its index in the file. The record CRC is masked
//! (rotated and offset, as in LevelDB) so that a run of zero bytes, which
//! is what a crash can leave past the end of an extended file, never parses
//! as a valid empty record. `prev_lsn` is `first_lsn - 1` except when the
//! log deliberately skips LSNs (see `WalOptions::min_next_lsn`), which lets
//! recovery tell a deliberate jump from a lost file.

use muninn_core::crc32c::Crc32c;

use super::Lsn;

pub(crate) const SEGMENT_MAGIC: [u8; 8] = *b"MUNWAL\0\x01";
pub(crate) const SEGMENT_VERSION: u32 = 1;
pub(crate) const HEADER_LEN: usize = 32;
pub(crate) const RECORD_HEADER_LEN: usize = 8;

/// Largest payload a record may carry.
pub const MAX_RECORD_LEN: usize = 1 << 30;

const MASK_DELTA: u32 = 0xA282_EAD8;

#[inline]
fn mask(crc: u32) -> u32 {
  crc.rotate_right(15).wrapping_add(MASK_DELTA)
}

/// The 8-byte header of a record carrying `payload`.
#[inline]
pub(crate) fn record_header(payload: &[u8]) -> [u8; RECORD_HEADER_LEN] {
  let len = (payload.len() as u32).to_le_bytes();
  let mut c = Crc32c::new();
  c.update(&len);
  c.update(payload);
  let mut h = [0u8; RECORD_HEADER_LEN];
  h[..4].copy_from_slice(&len);
  h[4..].copy_from_slice(&mask(c.finish()).to_le_bytes());
  h
}

/// Header fields of a segment file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SegmentHeader {
  pub first_lsn: Lsn,
  pub prev_lsn: Lsn,
}

fn header_crc(h: &[u8]) -> u32 {
  let mut c = Crc32c::new();
  c.update(&h[0..12]);
  c.update(&h[16..32]);
  c.finish()
}

pub(crate) fn encode_header(h: SegmentHeader) -> [u8; HEADER_LEN] {
  let mut b = [0u8; HEADER_LEN];
  b[0..8].copy_from_slice(&SEGMENT_MAGIC);
  b[8..12].copy_from_slice(&SEGMENT_VERSION.to_le_bytes());
  b[16..24].copy_from_slice(&h.first_lsn.to_le_bytes());
  b[24..32].copy_from_slice(&h.prev_lsn.to_le_bytes());
  let crc = header_crc(&b);
  b[12..16].copy_from_slice(&crc.to_le_bytes());
  b
}

pub(crate) fn decode_header(b: &[u8]) -> Result<SegmentHeader, String> {
  if b.len() < HEADER_LEN {
    return Err(format!("segment header incomplete ({} bytes)", b.len()));
  }
  if b[0..8] != SEGMENT_MAGIC {
    return Err("bad segment magic".into());
  }
  let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
  let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
  if u32_at(12) != header_crc(b) {
    return Err("segment header checksum mismatch".into());
  }
  if u32_at(8) != SEGMENT_VERSION {
    return Err(format!("unsupported WAL version {}", u32_at(8)));
  }
  Ok(SegmentHeader {
    first_lsn: u64_at(16),
    prev_lsn: u64_at(24),
  })
}

/// Where a scan of a segment's records ended.
#[derive(Debug)]
pub(crate) struct ScanEnd {
  /// Offset just past the last valid record.
  pub offset: usize,
  /// LSN the next record would have.
  pub next_lsn: Lsn,
  /// Why the scan stopped before the end of the bytes, if it did.
  pub stop: Option<String>,
}

/// Visits valid records in `bytes` (a whole segment file) starting at
/// `HEADER_LEN`, the first having `first_lsn`, until the bytes end, a
/// record with LSN above `max_lsn` would start, or a record fails
/// validation. `visit` errors abort the scan.
pub(crate) fn scan_records<E>(
  bytes: &[u8],
  first_lsn: Lsn,
  max_lsn: Lsn,
  mut visit: impl FnMut(Lsn, &[u8]) -> Result<(), E>,
) -> Result<ScanEnd, E> {
  let mut off = HEADER_LEN.min(bytes.len());
  let mut lsn = first_lsn;
  let mut stop = None;
  while off < bytes.len() && lsn <= max_lsn {
    let rest = &bytes[off..];
    if rest.len() < RECORD_HEADER_LEN {
      stop = Some(format!("incomplete record header at offset {off}"));
      break;
    }
    let len = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
    let crc = u32::from_le_bytes(rest[4..8].try_into().unwrap());
    if len > MAX_RECORD_LEN {
      stop = Some(format!("record length {len} at offset {off} is invalid"));
      break;
    }
    if rest.len() - RECORD_HEADER_LEN < len {
      stop = Some(format!("incomplete record at offset {off}"));
      break;
    }
    let payload = &rest[RECORD_HEADER_LEN..RECORD_HEADER_LEN + len];
    let mut c = Crc32c::new();
    c.update(&rest[0..4]);
    c.update(payload);
    if mask(c.finish()) != crc {
      stop = Some(format!("record checksum mismatch at offset {off}"));
      break;
    }
    visit(lsn, payload)?;
    off += RECORD_HEADER_LEN + len;
    lsn += 1;
  }
  Ok(ScanEnd {
    offset: off,
    next_lsn: lsn,
    stop,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use muninn_core::crc32c::crc32c;

  #[test]
  fn zero_bytes_are_not_a_record() {
    let mut seg = encode_header(SegmentHeader {
      first_lsn: 1,
      prev_lsn: 0,
    })
    .to_vec();
    seg.extend_from_slice(&[0u8; 64]);
    let end =
      scan_records::<()>(&seg, 1, u64::MAX, |_, _| unreachable!()).unwrap();
    assert_eq!(end.offset, HEADER_LEN);
    assert!(end.stop.is_some());
    assert_ne!(mask(crc32c(&[0, 0, 0, 0])), 0);
  }

  #[test]
  fn header_round_trip_and_rejection() {
    let h = SegmentHeader {
      first_lsn: 77,
      prev_lsn: 70,
    };
    let b = encode_header(h);
    assert_eq!(decode_header(&b), Ok(h));
    for i in 0..HEADER_LEN {
      let mut bad = b;
      bad[i] ^= 4;
      assert!(decode_header(&bad).is_err(), "flip at {i} accepted");
    }
    assert!(decode_header(&b[..31]).is_err());
  }

  #[test]
  fn records_scan_and_stop_at_max_lsn() {
    let mut seg = encode_header(SegmentHeader {
      first_lsn: 10,
      prev_lsn: 9,
    })
    .to_vec();
    for p in [&b"a"[..], b"", b"hello"] {
      seg.extend_from_slice(&record_header(p));
      seg.extend_from_slice(p);
    }
    let mut got = Vec::new();
    let end = scan_records::<()>(&seg, 10, u64::MAX, |l, p| {
      got.push((l, p.to_vec()));
      Ok(())
    })
    .unwrap();
    assert_eq!(end.offset, seg.len());
    assert_eq!(end.next_lsn, 13);
    assert!(end.stop.is_none());
    assert_eq!(got[2], (12, b"hello".to_vec()));
    let end = scan_records::<()>(&seg, 10, 10, |_, _| Ok(())).unwrap();
    assert_eq!(end.next_lsn, 11);
    assert_eq!(end.offset, HEADER_LEN + 9);
  }
}
