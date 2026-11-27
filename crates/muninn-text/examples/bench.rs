//! Indexing and query benchmark on a synthetic Zipfian corpus.
//!
//! ```text
//! cargo run --release -p muninn-text --example bench -- \
//!   [--docs 1000000] [--vocab 200000] [--queries 1000] [--k 10]
//! ```
//!
//! Reports indexing throughput, index size, and query latency (p50/p99)
//! for 1-, 2- and 5-term queries, pruned vs exhaustive, on the frozen and
//! the in-memory index. Every pruned result is checked against the
//! exhaustive one.

use std::hint::black_box;
use std::time::{Duration, Instant};

use muninn_core::{AlignedBuf, Region};
use muninn_text::{Analyzer, MemTextIndex, TextIndex, TextSearcher};

struct Rng(u64);

impl Rng {
  fn next_u64(&mut self) -> u64 {
    self.0 ^= self.0 >> 12;
    self.0 ^= self.0 << 25;
    self.0 ^= self.0 >> 27;
    self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
  }

  fn unit(&mut self) -> f64 {
    (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
  }
}

struct Zipf(Vec<f64>);

impl Zipf {
  fn new(n: usize, s: f64) -> Self {
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for r in 0..n {
      acc += 1.0 / ((r + 1) as f64).powf(s);
      cdf.push(acc);
    }
    cdf.iter_mut().for_each(|c| *c /= acc);
    Self(cdf)
  }

  fn sample(&self, rng: &mut Rng) -> usize {
    let u = rng.unit();
    self.0.partition_point(|&c| c < u).min(self.0.len() - 1)
  }
}

fn word(rank: usize, out: &mut String) {
  const SYL: [&str; 16] = [
    "ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "ba", "de", "fo", "gu",
    "ha", "je", "pe", "zo",
  ];
  let mut r = rank + 1;
  while r > 0 {
    out.push_str(SYL[r % 16]);
    r /= 16;
  }
}

fn arg(name: &str, default: usize) -> usize {
  let args: Vec<String> = std::env::args().collect();
  args
    .iter()
    .position(|a| a == name)
    .and_then(|i| args.get(i + 1))
    .map(|v| v.parse().expect("numeric argument"))
    .unwrap_or(default)
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
  let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
  sorted[i].as_secs_f64() * 1e6
}

fn main() {
  let docs = arg("--docs", 1_000_000);
  let vocab = arg("--vocab", 200_000);
  let nq = arg("--queries", 1000);
  let k = arg("--k", 10);
  let zipf = Zipf::new(vocab, 1.07);
  let mut rng = Rng(0x5eed_1234_5678_9abc);

  let mut texts = Vec::with_capacity(docs);
  let mut tokens = 0usize;
  for _ in 0..docs {
    let len = 5 + (rng.next_u64() % 26) as usize;
    tokens += len;
    let mut t = String::with_capacity(len * 7);
    for i in 0..len {
      if i > 0 {
        t.push(' ');
      }
      word(zipf.sample(&mut rng), &mut t);
    }
    texts.push(t);
  }
  let text_bytes: usize = texts.iter().map(String::len).sum();
  println!(
    "corpus: {docs} docs, {tokens} tokens ({:.1}/doc), {:.1} MB text, \
     vocabulary {vocab} (Zipf s=1.07), k={k}",
    tokens as f64 / docs as f64,
    text_bytes as f64 / 1e6
  );

  let mut mem = MemTextIndex::new(Analyzer::default());
  let t0 = Instant::now();
  for (i, t) in texts.iter().enumerate() {
    mem.add(i as u32, t).unwrap();
  }
  let index_time = t0.elapsed();
  drop(texts);
  println!(
    "index (memory): {:.2} s, {:.0} docs/s, {:.1} M tokens/s, {} terms, \
     ~{:.1} MB heap",
    index_time.as_secs_f64(),
    docs as f64 / index_time.as_secs_f64(),
    tokens as f64 / index_time.as_secs_f64() / 1e6,
    mem.num_terms(),
    mem.heap_bytes() as f64 / 1e6
  );

  let t0 = Instant::now();
  let buf = mem.write_to(AlignedBuf::new()).unwrap();
  let write_time = t0.elapsed();
  let size = buf.len();
  let t0 = Instant::now();
  let frozen = TextIndex::open(Region::new(buf)).unwrap();
  let open_time = t0.elapsed();
  println!(
    "freeze: write {:.2} s, open {:.1} us, file {:.1} MB ({:.2} bytes/token)",
    write_time.as_secs_f64(),
    open_time.as_secs_f64() * 1e6,
    size as f64 / 1e6,
    size as f64 / tokens as f64
  );

  println!();
  println!("single-term latency by term rank (frozen, pruned, mean of 200):");
  println!("| rank | doc freq | mean us |");
  println!("|---|---|---|");
  for rank in [0usize, 10, 100, 1000, 10_000, 100_000, usize::MAX] {
    let mut q = String::new();
    if rank == usize::MAX {
      q.push_str("unknownterm");
    } else {
      word(rank.min(vocab - 1), &mut q);
    }
    let df = frozen.doc_freq(&q).unwrap();
    let t = Instant::now();
    for _ in 0..200 {
      black_box(frozen.search(&q, k, None).unwrap());
    }
    let us = t.elapsed().as_secs_f64() / 200.0 * 1e6;
    let label = if rank == usize::MAX {
      "unknown".to_string()
    } else {
      rank.to_string()
    };
    println!("| {label} | {df} | {us:.2} |");
  }

  println!();
  println!(
    "| index  | terms | mode       | p50 us | p99 us | mean us | \
     speedup (mean) |"
  );
  println!("|---|---|---|---|---|---|---|");
  for nterms in [1usize, 2, 5] {
    let queries: Vec<String> = (0..nq)
      .map(|_| {
        let mut q = String::new();
        for i in 0..nterms {
          if i > 0 {
            q.push(' ');
          }
          word(zipf.sample(&mut rng), &mut q);
        }
        q
      })
      .collect();
    for (name, idx) in [
      ("frozen", &frozen as &dyn TextSearcher),
      ("memory", &mem as &dyn TextSearcher),
    ] {
      let exhaustive = |q: &str| match name {
        "frozen" => frozen.search_exhaustive(q, k, None, None).unwrap(),
        _ => mem.search_exhaustive(q, k, None, None).unwrap(),
      };
      // Warm up and check pruned == exhaustive on every query.
      for q in &queries {
        let a = idx.search(q, k, None).unwrap();
        let b = exhaustive(q);
        assert_eq!(a, b, "pruned and exhaustive differ for {q:?}");
      }
      // Alternate modes per query so both see the same cache state.
      let mut lat: [Vec<Duration>; 2] =
        [Vec::with_capacity(nq), Vec::with_capacity(nq)];
      for q in &queries {
        let t = Instant::now();
        black_box(idx.search(q, k, None).unwrap());
        lat[0].push(t.elapsed());
        let t = Instant::now();
        black_box(exhaustive(q));
        lat[1].push(t.elapsed());
      }
      let mut means = [0f64; 2];
      for (mi, mode) in ["pruned", "exhaustive"].into_iter().enumerate() {
        let l = &mut lat[mi];
        l.sort_unstable();
        means[mi] =
          l.iter().map(Duration::as_secs_f64).sum::<f64>() / nq as f64 * 1e6;
        let speedup = if mi == 1 {
          format!("{:.1}x", means[1] / means[0])
        } else {
          String::new()
        };
        println!(
          "| {name} | {nterms} | {mode} | {:.1} | {:.1} | {:.1} | {speedup} |",
          pct(l, 0.5),
          pct(l, 0.99),
          means[mi]
        );
      }
    }
  }
}
