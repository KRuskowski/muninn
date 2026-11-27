//! CRC-32C (Castagnoli), slice-by-8 software implementation.

const POLY: u32 = 0x82F6_3B78;

const fn make_tables() -> [[u32; 256]; 8] {
  let mut t = [[0u32; 256]; 8];
  let mut i = 0;
  while i < 256 {
    let mut c = i as u32;
    let mut k = 0;
    while k < 8 {
      c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
      k += 1;
    }
    t[0][i] = c;
    i += 1;
  }
  let mut s = 1;
  while s < 8 {
    let mut i = 0;
    while i < 256 {
      let prev = t[s - 1][i];
      t[s][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
      i += 1;
    }
    s += 1;
  }
  t
}

static TABLES: [[u32; 256]; 8] = make_tables();

/// Incremental CRC-32C.
#[derive(Clone, Copy, Debug)]
pub struct Crc32c(u32);

impl Default for Crc32c {
  fn default() -> Self {
    Self::new()
  }
}

impl Crc32c {
  pub const fn new() -> Self {
    Self(!0)
  }

  pub fn update(&mut self, bytes: &[u8]) {
    let t = &TABLES;
    let mut c = self.0;
    let mut chunks = bytes.chunks_exact(8);
    for ch in &mut chunks {
      let lo = c ^ u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
      let hi = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]);
      c = t[7][(lo & 0xFF) as usize]
        ^ t[6][((lo >> 8) & 0xFF) as usize]
        ^ t[5][((lo >> 16) & 0xFF) as usize]
        ^ t[4][(lo >> 24) as usize]
        ^ t[3][(hi & 0xFF) as usize]
        ^ t[2][((hi >> 8) & 0xFF) as usize]
        ^ t[1][((hi >> 16) & 0xFF) as usize]
        ^ t[0][(hi >> 24) as usize];
    }
    for &b in chunks.remainder() {
      c = (c >> 8) ^ t[0][((c ^ b as u32) & 0xFF) as usize];
    }
    self.0 = c;
  }

  pub fn finish(self) -> u32 {
    !self.0
  }
}

/// One-shot CRC-32C of `bytes`.
pub fn crc32c(bytes: &[u8]) -> u32 {
  let mut c = Crc32c::new();
  c.update(bytes);
  c.finish()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn known_vectors() {
    // RFC 3720 test vectors.
    assert_eq!(crc32c(b""), 0);
    assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
    assert_eq!(crc32c(&[0xFFu8; 32]), 0x62A8_AB43);
  }

  #[test]
  fn incremental_matches_one_shot() {
    let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
    let mut c = Crc32c::new();
    for part in data.chunks(7) {
      c.update(part);
    }
    assert_eq!(c.finish(), crc32c(&data));
  }
}
