//! Decoding helpers for the small `meta` section every index file has:
//! a little array of `u64` fields, validated before use.

use muninn_core::{Error, Metric, Region, Result};

use crate::quant::Quantization;

/// Reads at least `fields` `u64`s from a meta section (copied; it is
/// tiny). Extra trailing fields are ignored.
pub(crate) fn read(r: Region, fields: usize) -> Result<Vec<u64>> {
  if r.len() < fields * 8 {
    return Err(Error::corrupt(format!(
      "meta section of {} bytes is shorter than {fields} fields",
      r.len()
    )));
  }
  (0..fields).map(|i| r.read::<u64>(i * 8)).collect()
}

/// A count or size that must fit in `usize` and in a `u32` id space.
pub(crate) fn usize_field(v: u64, what: &str) -> Result<usize> {
  if v > u32::MAX as u64 {
    return Err(Error::corrupt(format!("{what} {v} is out of range")));
  }
  Ok(v as usize)
}

pub(crate) fn metric(v: u64) -> Result<Metric> {
  u8::try_from(v)
    .ok()
    .and_then(Metric::from_u8)
    .ok_or_else(|| Error::corrupt(format!("unknown metric tag {v}")))
}

pub(crate) fn quantization(v: u64) -> Result<Quantization> {
  u8::try_from(v)
    .ok()
    .and_then(Quantization::from_u8)
    .ok_or_else(|| Error::corrupt(format!("unknown quantisation tag {v}")))
}
