//! What a text search through `Collection::search` costs on top of
//! searching the segment's text index directly.
//!
//! ```text
//! cargo run --release -p muninn --example text_overhead -- \
//!   <collection.tsv> <queries.tsv> [docs] [default|english]
//! ```
//!
//! Loads the first `docs` passages (default 1 000 000) of an MS MARCO
//! style `id<TAB>text` file into a collection with one text field,
//! compacts it into one segment, then alternates rounds of the same
//! queries through the engine and through the segment's text index file
//! opened on its own, on one thread, and prints the best throughput of
//! each. `ENGINE_ONLY=1` runs only engine rounds (for profiling).

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use muninn::store::map_file;
use muninn::text::{TextIndex, TextSearcher};
use muninn::{
  AnalyzerConfig, Database, Document, Durability, Options, Query, Schema,
};

fn read_tsv(path: &Path, limit: usize) -> Vec<String> {
  let f = std::fs::File::open(path).expect("open tsv");
  std::io::BufReader::new(f)
    .lines()
    .take(limit)
    .map(|l| {
      let l = l.expect("read tsv");
      l.split_once('\t').map(|x| x.1.to_string()).unwrap_or(l)
    })
    .collect()
}

/// One way to run a query.
type Search<'a> = &'a dyn Fn(&str);

fn qps(queries: &[String], f: Search<'_>) -> f64 {
  let t = Instant::now();
  queries.iter().for_each(|q| f(q));
  queries.len() as f64 / t.elapsed().as_secs_f64()
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  let coll = PathBuf::from(args.get(1).expect("collection.tsv"));
  let qfile = PathBuf::from(args.get(2).expect("queries.tsv"));
  let n: usize = args.get(3).map_or(1_000_000, |s| s.parse().unwrap());
  let config = match args.get(4).map_or("default", String::as_str) {
    "english" => AnalyzerConfig::english(),
    _ => AnalyzerConfig::default(),
  };
  let docs = read_tsv(&coll, n);
  let queries = read_tsv(&qfile, usize::MAX);
  let dir = PathBuf::from("target/text-overhead");
  let _ = std::fs::remove_dir_all(&dir);
  let db =
    Database::open(&dir, Options::default().durability(Durability::NoSync))
      .unwrap();
  let c = db
    .create_collection("c", Schema::text(["text"]).with_analyzer(config))
    .unwrap();
  let t = Instant::now();
  for (b, chunk) in docs.chunks(1000).enumerate() {
    let batch: Vec<Document> = chunk
      .iter()
      .enumerate()
      .map(|(i, d)| Document::new((b * 1000 + i) as u64).text("text", d))
      .collect();
    c.upsert(&batch).unwrap();
  }
  c.compact().unwrap();
  println!(
    "{} docs loaded and compacted in {:.1} s",
    docs.len(),
    t.elapsed().as_secs_f64()
  );
  let coll_dir = dir.join("collections").join("c");
  let text_file = std::fs::read_dir(&coll_dir)
    .unwrap()
    .map(|e| e.unwrap().path())
    .find(|p| p.extension().is_some_and(|e| e == "t0"))
    .expect("segment text file");
  let raw = TextIndex::open(map_file(&text_file).unwrap()).unwrap();
  let engine = |q: &str| {
    std::hint::black_box(c.search(&Query::text(q).k(10)).unwrap());
  };
  let direct = |q: &str| {
    std::hint::black_box(raw.search(q, 10, None).unwrap());
  };
  if std::env::var_os("ENGINE_ONLY").is_some() {
    for _ in 0..5 {
      println!("engine {:.0} q/s", qps(&queries, &engine));
    }
    return;
  }
  let mut best = [0f64; 2];
  // Alternate the order so neither side always runs on a warmer cache.
  for round in 0..6 {
    let sides: [(usize, Search<'_>); 2] = if round % 2 == 0 {
      [(0, &direct), (1, &engine)]
    } else {
      [(1, &engine), (0, &direct)]
    };
    for (i, f) in sides {
      best[i] = best[i].max(qps(&queries, f));
    }
  }
  println!(
    "1 thread: index {:.0} q/s, engine {:.0} q/s ({:+.1}%, {:.1} us per \
     query)",
    best[0],
    best[1],
    (best[1] / best[0] - 1.0) * 100.0,
    (1.0 / best[1] - 1.0 / best[0]) * 1e6
  );
  drop(c);
  drop(db);
  let _ = std::fs::remove_dir_all(&dir);
}
