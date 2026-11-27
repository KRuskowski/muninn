//! Shared vocabulary for every muninn crate.
//!
//! Nothing in here allocates on a hot path or depends on another crate.
//! Index crates code against these types so they can be composed by the
//! engine without adapters.
//!
//! muninn files store numbers in native byte order and are read in place,
//! so muninn supports little-endian targets only (x86-64, AArch64).

#[cfg(not(target_endian = "little"))]
compile_error!("muninn supports little-endian targets only");

pub mod container;
pub mod crc32c;
mod error;
mod filter;
mod metric;
mod pod;
mod region;
mod topk;

pub use error::{Error, Result};
pub use filter::{BitSet, DocFilter};
pub use metric::Metric;
pub use pod::{Pod, as_bytes};
pub use region::{ALIGN, AlignedBuf, Region};
pub use topk::TopK;

/// Dense, segment-local document id.
///
/// Ids are assigned densely from 0 within one segment so they can index
/// arrays and bitsets directly. Mapping to user-facing ids is the store's
/// job, never an index's.
pub type DocId = u32;

/// One search result.
///
/// `score` is always "higher is better", whatever produced it: BM25,
/// cosine similarity, a dot product, or a negated squared L2 distance
/// (see [`Metric::score`]). Fusion code relies on that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
  pub id: DocId,
  pub score: f32,
}

impl Hit {
  pub const fn new(id: DocId, score: f32) -> Self {
    Self { id, score }
  }
}
