//! Append-only vector storage that readers scan while it grows.
//!
//! The memtable's vectors must be searchable without blocking the writer
//! and without the writer blocking searches. A `Vec<f32>` cannot do that:
//! growing it moves the data under any reader. [`VecArena`] stores rows
//! in chunks that are allocated once and never move; chunk `c` holds
//! `BASE_ROWS << c` rows, so 24 chunk slots cover every `u32` row id and
//! a small memtable allocates only a few kilobytes.
//!
//! # Protocol
//!
//! - Exactly one [`ArenaWriter`] exists per arena (they are created
//!   together), and pushing needs `&mut ArenaWriter`, so there is a single
//!   writer at a time.
//! - The writer copies new rows into slots at and above the published
//!   length, then [`VecArena::publish`]es the new length with `Release`.
//! - Readers load the published length with `Acquire` and only ever form
//!   `&[f32]` over rows below it. Those rows are never written again.
//!
//! Readers and the writer therefore never touch the same floats, and the
//! release/acquire pair makes the writer's copies visible before a reader
//! can see the length that covers them.

use std::ptr::NonNull;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Rows in chunk 0.
const BASE_ROWS: usize = 1024;
/// `BASE_ROWS * (2^24 - 1)` rows exceed `u32::MAX`.
const MAX_CHUNKS: usize = 24;

/// One fixed allocation of `floats` zeroed `f32`s.
struct Chunk {
  ptr: NonNull<f32>,
  floats: usize,
}

impl Chunk {
  fn zeroed(floats: usize) -> Self {
    // `vec![0.0; n]` asks the allocator for zeroed memory, so untouched
    // pages of a large chunk are not committed.
    let b: Box<[f32]> = vec![0f32; floats].into_boxed_slice();
    let raw = Box::into_raw(b) as *mut f32;
    Self {
      ptr: NonNull::new(raw).expect("Box pointers are non-null"),
      floats,
    }
  }
}

impl Drop for Chunk {
  fn drop(&mut self) {
    let slice =
      std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.floats);
    // SAFETY: `ptr`/`floats` came from `Box::into_raw` of a `Box<[f32]>`
    // of exactly this length in `zeroed`, and are freed only here, once.
    drop(unsafe { Box::from_raw(slice) });
  }
}

/// Chunk index and row offset within the chunk of row `r`.
#[inline]
fn locate(r: usize) -> (usize, usize) {
  let q = r / BASE_ROWS + 1;
  let c = (usize::BITS - 1 - q.leading_zeros()) as usize;
  let start = BASE_ROWS * ((1usize << c) - 1);
  (c, r - start)
}

/// Rows in chunk `c`.
#[inline]
fn chunk_rows(c: usize) -> usize {
  BASE_ROWS << c
}

/// Append-only rows of `dim` floats; see the module docs.
pub(crate) struct VecArena {
  id: u64,
  dim: usize,
  chunks: [OnceLock<Chunk>; MAX_CHUNKS],
  published: AtomicUsize,
}

// SAFETY: chunks are plain heap allocations owned by the arena. Shared
// access follows the module protocol: readers only form shared slices
// over published rows, which are immutable from then on, and the single
// writer (proved by `&mut ArenaWriter`) only writes unpublished rows
// through raw pointers. No `&mut` to arena memory is ever created.
unsafe impl Sync for VecArena {}
// SAFETY: the arena owns its chunks outright; moving it to another
// thread moves that ownership, like a `Vec<Box<[f32]>>`.
unsafe impl Send for VecArena {}

/// The single writer of a [`VecArena`].
pub(crate) struct ArenaWriter {
  arena_id: u64,
  len: usize,
}

impl ArenaWriter {
  /// Rows pushed, published or not.
  pub(crate) fn len(&self) -> usize {
    self.len
  }
}

impl VecArena {
  /// An empty arena of `dim`-float rows and its only writer.
  pub(crate) fn new(dim: usize) -> (VecArena, ArenaWriter) {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    assert!(dim > 0, "arena rows need at least one dimension");
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let arena = VecArena {
      id,
      dim,
      chunks: std::array::from_fn(|_| OnceLock::new()),
      published: AtomicUsize::new(0),
    };
    (
      arena,
      ArenaWriter {
        arena_id: id,
        len: 0,
      },
    )
  }

  /// Appends one row (not visible until [`VecArena::publish`]).
  pub(crate) fn push(&self, w: &mut ArenaWriter, row: &[f32]) {
    assert_eq!(w.arena_id, self.id, "writer belongs to another arena");
    assert_eq!(row.len(), self.dim, "row has the wrong dimension");
    let (c, off) = locate(w.len);
    assert!(c < MAX_CHUNKS, "arena is full");
    let chunk =
      self.chunks[c].get_or_init(|| Chunk::zeroed(chunk_rows(c) * self.dim));
    debug_assert!((off + 1) * self.dim <= chunk.floats);
    // SAFETY: row `w.len` lies inside chunk `c` (`locate` and the chunk
    // size agree), and it is at or above the published length, so no
    // reader has a reference to these floats (module protocol). `&mut
    // ArenaWriter` makes this the only writer. `row` is a separate
    // allocation, so the ranges do not overlap.
    unsafe {
      std::ptr::copy_nonoverlapping(
        row.as_ptr(),
        chunk.ptr.as_ptr().add(off * self.dim),
        self.dim,
      );
    }
    w.len += 1;
  }

  /// Makes every pushed row visible to readers.
  pub(crate) fn publish(&self, w: &ArenaWriter) {
    assert_eq!(w.arena_id, self.id, "writer belongs to another arena");
    self.published.store(w.len, Ordering::Release);
  }

  /// Number of published rows.
  #[inline]
  pub(crate) fn len(&self) -> usize {
    self.published.load(Ordering::Acquire)
  }

  /// Floats per row.
  pub(crate) fn dim(&self) -> usize {
    self.dim
  }

  /// Published row `r`. Panics if `r` is not published.
  pub(crate) fn row(&self, r: usize) -> &[f32] {
    assert!(r < self.len(), "row {r} is not published");
    let (c, off) = locate(r);
    let chunk = self.chunks[c].get().expect("published rows are allocated");
    // SAFETY: row `r` is published (checked above), so it was fully
    // written before the `Release` store our `Acquire` load observed and
    // is never written again; the range lies inside the chunk.
    unsafe {
      std::slice::from_raw_parts(
        chunk.ptr.as_ptr().add(off * self.dim),
        self.dim,
      )
    }
  }

  /// The published rows `0..n` (clamped to the published length) as
  /// contiguous runs: `(first row, row-major floats)`, in row order.
  pub(crate) fn runs(&self, n: usize) -> Vec<(usize, &[f32])> {
    let n = n.min(self.len());
    let mut out = Vec::new();
    let mut r = 0;
    while r < n {
      let (c, off) = locate(r);
      debug_assert_eq!(off, 0);
      let rows = chunk_rows(c).min(n - r);
      let chunk = self.chunks[c].get().expect("published rows are allocated");
      // SAFETY: rows `r..r + rows` are published (below `n`, which is
      // clamped to the acquired length) and lie at the start of chunk `c`;
      // published rows are immutable (module protocol).
      let s = unsafe {
        std::slice::from_raw_parts(chunk.ptr.as_ptr(), rows * self.dim)
      };
      out.push((r, s));
      r += rows;
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn locate_is_contiguous() {
    let mut expect = (0usize, 0usize);
    for r in 0..BASE_ROWS * 40 {
      assert_eq!(locate(r), expect, "row {r}");
      expect.1 += 1;
      if expect.1 == chunk_rows(expect.0) {
        expect = (expect.0 + 1, 0);
      }
    }
    let (c, _) = locate(u32::MAX as usize);
    assert!(c < MAX_CHUNKS);
  }

  #[test]
  fn rows_round_trip_across_chunks() {
    let (a, mut w) = VecArena::new(3);
    let n = BASE_ROWS * 3 + 17;
    for i in 0..n {
      a.push(&mut w, &[i as f32, -(i as f32), 0.5]);
    }
    assert_eq!(a.len(), 0);
    a.publish(&w);
    assert_eq!(a.len(), n);
    assert_eq!(a.row(n - 1), &[(n - 1) as f32, -((n - 1) as f32), 0.5]);
    let runs = a.runs(usize::MAX);
    assert_eq!(runs.len(), 3);
    let mut seen = 0;
    for (start, floats) in runs {
      assert_eq!(start, seen);
      for (j, row) in floats.chunks_exact(3).enumerate() {
        assert_eq!(row[0], (start + j) as f32);
      }
      seen += floats.len() / 3;
    }
    assert_eq!(seen, n);
    assert_eq!(a.runs(5)[0].1.len(), 15);
  }

  #[test]
  fn readers_see_only_published_rows_while_writing() {
    let (a, mut w) = VecArena::new(4);
    std::thread::scope(|s| {
      let a = &a;
      s.spawn(move || {
        for i in 0..20_000usize {
          a.push(&mut w, &[i as f32; 4]);
          if i % 7 == 0 {
            a.publish(&w);
          }
        }
        a.publish(&w);
      });
      for _ in 0..4 {
        s.spawn(move || {
          let mut last = 0;
          while last < 20_000 {
            let runs = a.runs(usize::MAX);
            let mut n = 0;
            for (start, floats) in runs {
              for (j, row) in floats.chunks_exact(4).enumerate() {
                assert!(row.iter().all(|&x| x == (start + j) as f32));
              }
              n = start + floats.len() / 4;
            }
            assert!(n >= last);
            last = n;
          }
        });
      }
    });
  }
}
