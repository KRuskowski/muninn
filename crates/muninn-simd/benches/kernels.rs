//! Every kernel at typical embedding dimensions, for every instruction set
//! the running CPU supports (benchmark ids are `<kernel>/<isa>/<dim>`),
//! plus a 10k-row scan comparing `distance_batch` with a per-row loop.
//!
//! Run with `cargo bench -p muninn-simd`; filter with e.g.
//! `cargo bench -p muninn-simd -- 'dot/avx512'`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{
  BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use muninn_core::Metric;
use muninn_simd::{Isa, Kernels, kernels_for};

const DIMS: [usize; 6] = [32, 96, 128, 384, 768, 1536];
const SCAN_ROWS: usize = 10_000;

/// xorshift64*: deterministic data without a `rand` dependency.
struct Rng(u64);

impl Rng {
  fn next_u64(&mut self) -> u64 {
    self.0 ^= self.0 >> 12;
    self.0 ^= self.0 << 25;
    self.0 ^= self.0 >> 27;
    self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
  }

  fn f32(&mut self) -> f32 {
    (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
  }

  fn vec<T>(&mut self, n: usize, mut f: impl FnMut(&mut Self) -> T) -> Vec<T> {
    (0..n).map(|_| f(self)).collect()
  }
}

/// A two-slice kernel, as stored in [`Kernels`].
type Kernel<T, R> = fn(&[T], &[T]) -> R;

fn tables() -> Vec<&'static Kernels> {
  Isa::ALL.into_iter().filter_map(kernels_for).collect()
}

/// Benchmarks `pick(k)(a, b)` for every kernel table and dimension.
/// `len(dim)` is the slice length for a vector of `dim` elements.
fn pair<T, R>(
  c: &mut Criterion,
  group: &str,
  gen_one: fn(&mut Rng) -> T,
  len: fn(usize) -> usize,
  pick: fn(&Kernels) -> Kernel<T, R>,
) {
  let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
  let mut g = c.benchmark_group(group);
  for dim in DIMS {
    let a = rng.vec(len(dim), gen_one);
    let b = rng.vec(len(dim), gen_one);
    g.throughput(Throughput::Elements(dim as u64));
    for k in tables() {
      let f = pick(k);
      let id = BenchmarkId::new(k.isa.name(), dim);
      g.bench_function(id, |bch| bch.iter(|| f(black_box(&a), black_box(&b))));
    }
  }
  g.finish();
}

fn same(dim: usize) -> usize {
  dim
}

fn words(bits: usize) -> usize {
  bits.div_ceil(64)
}

fn f16(rng: &mut Rng) -> u16 {
  muninn_simd::f32_to_f16(rng.f32())
}

/// `dot_f32_u8`: weights and codes of the same dimension.
fn mixed(c: &mut Criterion) {
  let mut rng = Rng(0x6a09_e667_f3bc_c908);
  let mut g = c.benchmark_group("dot_f32_u8");
  for dim in DIMS {
    let a = rng.vec(dim, Rng::f32);
    let b = rng.vec(dim, |r| r.next_u64() as u8);
    g.throughput(Throughput::Elements(dim as u64));
    for k in tables() {
      let f = k.dot_f32_u8;
      let id = BenchmarkId::new(k.isa.name(), dim);
      g.bench_function(id, |bch| bch.iter(|| f(black_box(&a), black_box(&b))));
    }
  }
  g.finish();
}

fn kernels(c: &mut Criterion) {
  for k in tables() {
    eprintln!("benchmarking {} ({})", k.isa.name(), k.features);
  }
  pair(c, "dot", Rng::f32, same, |k| k.dot);
  pair(c, "l2sq", Rng::f32, same, |k| k.l2sq);
  pair(c, "dot_i8", |r| r.next_u64() as i8, same, |k| k.dot_i8);
  pair(c, "l2sq_u8", |r| r.next_u64() as u8, same, |k| k.l2sq_u8);
  // Dimension in bits, packed into u64 words.
  pair(c, "hamming", Rng::next_u64, words, |k| k.hamming);
  pair(c, "dot_f16", f16, same, |k| k.dot_f16);
  pair(c, "l2sq_f16", f16, same, |k| k.l2sq_f16);
}

/// One query against 10k rows: the batched kernel, and the same scan as a
/// loop of single-row calls, to show what batching buys.
fn scan(c: &mut Criterion) {
  let mut rng = Rng(0x2545_f491_4f6c_dd1d);
  for (name, batched) in [("scan10k_batch", true), ("scan10k_loop", false)] {
    let mut g = c.benchmark_group(name);
    for dim in DIMS {
      let q = rng.vec(dim, Rng::f32);
      let rows = rng.vec(dim * SCAN_ROWS, Rng::f32);
      let mut out = vec![0.0f32; SCAN_ROWS];
      g.throughput(Throughput::Elements(SCAN_ROWS as u64));
      for k in tables() {
        let id = BenchmarkId::new(k.isa.name(), dim);
        if batched {
          let f = k.distance_batch;
          g.bench_function(id, |bch| {
            bch
              .iter(|| f(Metric::L2, black_box(&q), black_box(&rows), &mut out))
          });
        } else {
          let f = k.l2sq;
          g.bench_function(id, |bch| {
            bch.iter(|| {
              for (o, row) in out.iter_mut().zip(rows.chunks_exact(dim)) {
                *o = f(black_box(&q), row);
              }
            })
          });
        }
        black_box(&out);
      }
    }
    g.finish();
  }
}

criterion_group! {
  name = benches;
  config = Criterion::default()
    .warm_up_time(Duration::from_millis(500))
    .measurement_time(Duration::from_secs(2))
    .sample_size(30);
  targets = kernels, mixed, scan
}
criterion_main!(benches);
