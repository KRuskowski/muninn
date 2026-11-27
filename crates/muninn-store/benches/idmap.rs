//! Id-map lookup latency with one million ids, frozen (memory-resident
//! container) and in-memory, against a sorted-array binary search on the
//! same keys as the alternative design.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use muninn_core::{AlignedBuf, Region};
use muninn_store::{ExternalId, FrozenIdMap, IdMap};

const N: u64 = 1_000_000;

/// Spreads sequential numbers over the u64 range (a bijection).
fn key(i: u64) -> u64 {
  i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5555
}

fn str_key(i: u64) -> String {
  format!("user-{i:08}")
}

/// Every id in a random order, so lookups miss the cache like real ones.
fn order() -> Vec<u64> {
  let mut v: Vec<u64> = (0..N).collect();
  let mut s = 0x2545_F491_4F6C_DD1Du64;
  for i in (1..v.len()).rev() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    v.swap(i, (s % (i as u64 + 1)) as usize);
  }
  v
}

fn bench(c: &mut Criterion) {
  let order = order();
  let mut nums = IdMap::new();
  let mut strs = IdMap::new();
  for i in 0..N {
    nums.insert(ExternalId::U64(key(i))).unwrap();
    strs.insert(ExternalId::Str(str_key(i))).unwrap();
  }
  let fnum =
    FrozenIdMap::open(Region::new(nums.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  let fstr =
    FrozenIdMap::open(Region::new(strs.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  let mut sorted: Vec<(u64, u32)> =
    (0..N).map(|i| (key(i), i as u32)).collect();
  sorted.sort_unstable();
  let qnum: Vec<u64> = order.iter().map(|&i| key(i)).collect();
  let qstr: Vec<String> = order.iter().map(|&i| str_key(i)).collect();

  let mut g = c.benchmark_group("idmap_1m");
  g.throughput(Throughput::Elements(1));
  let mut at = 0usize;
  let mut next = move || {
    at = (at + 1) % N as usize;
    at
  };
  g.bench_function("frozen_u64_hit", |b| {
    b.iter(|| black_box(fnum.get_u64(black_box(qnum[next()]))))
  });
  g.bench_function("frozen_u64_miss", |b| {
    b.iter(|| black_box(fnum.get_u64(black_box(!qnum[next()]))))
  });
  g.bench_function("sorted_u64_binary_search", |b| {
    b.iter(|| {
      let k = black_box(qnum[next()]);
      black_box(
        sorted
          .binary_search_by_key(&k, |e| e.0)
          .ok()
          .map(|i| sorted[i].1),
      )
    })
  });
  g.bench_function("memtable_u64_hit", |b| {
    b.iter(|| {
      black_box(
        nums.get(muninn_store::ExternalIdRef::U64(black_box(qnum[next()]))),
      )
    })
  });
  g.bench_function("frozen_str_hit", |b| {
    b.iter(|| black_box(fstr.get_str(black_box(&qstr[next()]))))
  });
  g.bench_function("memtable_str_hit", |b| {
    b.iter(|| {
      black_box(
        strs.get(muninn_store::ExternalIdRef::Str(black_box(&qstr[next()]))),
      )
    })
  });
  g.bench_function("frozen_external_id", |b| {
    b.iter(|| black_box(fstr.external_id(black_box(order[next()] as u32)).ok()))
  });
  g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
