//! Payload index benchmarks on 1M synthetic documents: build time,
//! frozen size, open time, and evaluate()/estimate() latency on the
//! frozen and the mutable index.
//!
//! Payload fields: `lang` keyword (10 values, Zipf-like, "en" ~50%),
//! `tags` 1-3 keywords from 1000 (Zipf), `year` integer 1990..2030,
//! `price` float 0..1000, `score` integer 0..1M, `flag` bool (~30%).

use std::hint::black_box;
use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use muninn_core::{AlignedBuf, Region};
use muninn_filter::Schema;
use muninn_filter::{FieldType, Filter, FilterIndex, FrozenIndex, MemIndex};
use serde_json::{Value, json};

const N: usize = 1_000_000;

struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    // splitmix64
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  fn unit(&mut self) -> f64 {
    (self.next() >> 11) as f64 / (1u64 << 53) as f64
  }

  /// Approximately Zipf(1) over `0..n`.
  fn zipf(&mut self, n: usize) -> usize {
    let x = (n as f64 + 1.0).powf(self.unit()) - 1.0;
    (x as usize).min(n - 1)
  }
}

fn schema() -> Schema {
  Schema::new()
    .with("lang", FieldType::Keyword)
    .and_then(|s| s.with("tags", FieldType::Keyword))
    .and_then(|s| s.with("year", FieldType::Integer))
    .and_then(|s| s.with("price", FieldType::Float))
    .and_then(|s| s.with("score", FieldType::Integer))
    .and_then(|s| s.with("flag", FieldType::Bool))
    .unwrap()
}

const LANGS: [&str; 10] =
  ["en", "de", "fr", "es", "it", "nl", "pl", "sv", "fi", "da"];

fn payloads() -> Vec<Value> {
  let mut r = Rng(42);
  (0..N)
    .map(|_| {
      let lang = if r.unit() < 0.5 {
        "en"
      } else {
        LANGS[1 + r.zipf(9)]
      };
      let ntags = 1 + (r.next() % 3) as usize;
      let tags: Vec<String> = (0..ntags)
        .map(|_| format!("t{:04}", r.zipf(1000)))
        .collect();
      json!({
        "lang": lang,
        "tags": tags,
        "year": 1990 + (r.next() % 40) as i64,
        "price": (r.unit() * 1000.0 * 100.0).round() / 100.0,
        "score": (r.next() % 1_000_000) as i64,
        "flag": r.unit() < 0.3,
      })
    })
    .collect()
}

fn build(payloads: &[Value]) -> MemIndex {
  let mut m = MemIndex::new(schema());
  for (i, p) in payloads.iter().enumerate() {
    m.add(i as u32, p).unwrap();
  }
  m
}

fn queries() -> Vec<(&'static str, Filter)> {
  vec![
    ("eq_common_keyword", Filter::eq("lang", "en")),
    ("eq_rare_keyword", Filter::eq("tags", "t0900")),
    ("range_10pct_float", Filter::range("price", 0.0..100.0)),
    ("range_1pct_int", Filter::range("score", 0..10_000)),
    (
      "and_3_clauses",
      Filter::eq("lang", "en")
        & Filter::gte("year", 2020)
        & Filter::eq("flag", true),
    ),
    (
      "or_3_clauses",
      Filter::eq("lang", "fr")
        | Filter::eq("tags", "t0005")
        | Filter::lt("price", 10.0),
    ),
  ]
}

fn bench(c: &mut Criterion) {
  let t = Instant::now();
  let payloads = payloads();
  eprintln!("generated {N} payloads in {:.2?}", t.elapsed());

  let t = Instant::now();
  let mem = build(&payloads);
  let add_time = t.elapsed();
  let t = Instant::now();
  let buf = mem.write_to(AlignedBuf::new()).unwrap();
  let write_time = t.elapsed();
  let size = buf.len();
  let region = Region::new(buf);
  let t = Instant::now();
  let frozen = FrozenIndex::open(region.clone()).unwrap();
  let open_time = t.elapsed();
  eprintln!(
    "1M docs: add {add_time:.2?}, write {write_time:.2?}, open \
     {open_time:.2?}, frozen size {size} bytes ({:.1} bytes/doc)",
    size as f64 / N as f64
  );
  for (name, f) in queries() {
    let got = frozen.evaluate(&f, N).unwrap();
    assert_eq!(got, mem.evaluate(&f, N).unwrap(), "{name}");
    eprintln!(
      "{name}: {} matches ({:.3}%), estimate {:.3}%",
      got.count(),
      got.count() as f64 * 100.0 / N as f64,
      frozen.estimate(&f, N) * 100.0
    );
  }

  let mut g = c.benchmark_group("build_1m");
  g.sample_size(10);
  g.bench_function("mem_add", |b| b.iter_with_large_drop(|| build(&payloads)));
  g.bench_function("write_frozen", |b| {
    b.iter_with_large_drop(|| {
      mem.write_to(AlignedBuf::with_capacity(size)).unwrap()
    })
  });
  g.bench_function("open_frozen", |b| {
    b.iter(|| FrozenIndex::open(region.clone()).unwrap())
  });
  g.finish();

  for (kind, ix) in [
    ("frozen", &frozen as &dyn FilterIndex),
    ("mem", &mem as &dyn FilterIndex),
  ] {
    let mut g = c.benchmark_group(format!("evaluate_1m_{kind}"));
    for (name, f) in queries() {
      g.bench_function(name, |b| {
        b.iter(|| ix.evaluate(black_box(&f), N).unwrap())
      });
    }
    g.finish();
  }

  let mut g = c.benchmark_group("estimate_1m_frozen");
  for (name, f) in queries() {
    g.bench_function(name, |b| b.iter(|| frozen.estimate(black_box(&f), N)));
  }
  g.finish();
  // Peak resident memory, where the OS reports it (Linux).
  if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
    for line in status.lines().filter(|l| l.starts_with("VmHWM")) {
      eprintln!("peak memory {line}");
    }
  }
}

criterion_group!(benches, bench);
criterion_main!(benches);
