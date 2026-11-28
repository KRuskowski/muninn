//! `muninn-bench`: benchmark muninn on standard datasets.
//!
//! See `docs/BENCHMARKS.md`; `scripts/bench_all.sh` runs everything.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use muninn_bench::engine::{
  EngineTextRun, EngineVectorRun, run_engine_text, run_engine_vector,
};
use muninn_bench::run::{Pools, write_json};
use muninn_bench::sys::SystemInfo;
use muninn_bench::text::{AnalyzerKind, TextRun, run_text};
use muninn_bench::vector::{
  FlatRun, HnswRun, parse_quantization, run_flat, run_hnsw, verify_ground_truth,
};

/// Default `ef` sweep, shared with `scripts/run_ann.py`.
const DEFAULT_EFS: &str = "10,16,24,32,48,64,96,128,192,256,384,512";

#[derive(Parser)]
#[command(version, about = "Benchmark muninn against other engines")]
struct Cli {
  #[command(subcommand)]
  cmd: Cmd,
}

#[derive(Args)]
struct Common {
  /// Threads for builds and all-thread figures (0: one per CPU).
  #[arg(long, default_value_t = 0)]
  threads: usize,
  /// Measurement repetitions per configuration; the best is kept.
  #[arg(long, default_value_t = 1)]
  runs: usize,
  /// Results per query.
  #[arg(long, default_value_t = 10)]
  k: usize,
  /// Directory for temporary index files.
  #[arg(long, default_value = "bench-work")]
  work_dir: PathBuf,
  /// Directory for result JSON files.
  #[arg(long, default_value = "bench-results")]
  out_dir: PathBuf,
}

#[derive(Subcommand)]
enum Cmd {
  /// HNSW build and ef sweep (one graph; one result per traversal kind).
  Hnsw {
    /// Dataset directory (base.fbin, query.fbin, gt.ibin, meta.json).
    #[arg(long)]
    data: PathBuf,
    /// HNSW M (level 0 keeps 2*M neighbours).
    #[arg(long, default_value_t = 16)]
    m: usize,
    /// HNSW ef_construction.
    #[arg(long, default_value_t = 200)]
    efc: usize,
    /// Comma-separated search ef values.
    #[arg(long, default_value = DEFAULT_EFS)]
    ef: String,
    /// Comma-separated traversal kinds: none, sq8, binary.
    #[arg(long, default_value = "none")]
    quant: String,
    /// Use only the first N queries.
    #[arg(long)]
    queries: Option<usize>,
    /// Keep the built full-precision index at this path, and if a
    /// readable index is already there, search it instead of building
    /// (build time is then reported as 0). For search tuning runs.
    #[arg(long)]
    reuse_index: Option<PathBuf>,
    /// Search an in-memory copy of the index file instead of the mapped
    /// file.
    #[arg(long)]
    in_memory: bool,
    #[command(flatten)]
    common: Common,
  },
  /// Exact search with FlatIndex.
  Flat {
    /// Dataset directory.
    #[arg(long)]
    data: PathBuf,
    /// Use only the first N queries.
    #[arg(long, default_value_t = 1000)]
    queries: usize,
    #[command(flatten)]
    common: Common,
  },
  /// Check a dataset's ground truth with muninn's exact search.
  VerifyGt {
    /// Dataset directory.
    #[arg(long)]
    data: PathBuf,
    /// Number of queries to check, evenly spaced.
    #[arg(long, default_value_t = 1000)]
    sample: usize,
    #[command(flatten)]
    common: Common,
  },
  /// BM25 top-k on MS MARCO passages: muninn and tantivy.
  Text {
    /// collection.tsv (pid<TAB>passage).
    #[arg(long)]
    collection: PathBuf,
    /// Queries TSV (qid<TAB>query), e.g. queries.dev.small.tsv.
    #[arg(long)]
    queries: PathBuf,
    /// Index only the first N passages.
    #[arg(long)]
    limit: Option<usize>,
    /// Use only the first N queries.
    #[arg(long)]
    query_limit: Option<usize>,
    /// Analysis: default or english.
    #[arg(long, default_value = "default")]
    analyzer: String,
    /// Comma-separated tantivy indexing thread counts; the first builds
    /// the searched index. Empty skips tantivy.
    #[arg(long, default_value = "1")]
    tantivy_index_threads: String,
    #[command(flatten)]
    common: Common,
  },
  /// End to end through the engine: ingest a vector dataset into a
  /// collection, compact it, sweep ef.
  EngineVector {
    /// Dataset directory.
    #[arg(long)]
    data: PathBuf,
    /// HNSW M.
    #[arg(long, default_value_t = 16)]
    m: usize,
    /// HNSW ef_construction.
    #[arg(long, default_value_t = 200)]
    efc: usize,
    /// Comma-separated search ef values.
    #[arg(long, default_value = DEFAULT_EFS)]
    ef: String,
    /// Use only the first N queries.
    #[arg(long)]
    queries: Option<usize>,
    /// Documents per upsert call.
    #[arg(long, default_value_t = 1000)]
    batch: usize,
    /// Write durability: each, batch or none.
    #[arg(long, default_value = "each")]
    durability: String,
    #[command(flatten)]
    common: Common,
  },
  /// End to end through the engine: ingest MS MARCO passages, compact,
  /// BM25 top-k, compared with the component-level index.
  EngineText {
    /// collection.tsv (pid<TAB>passage).
    #[arg(long)]
    collection: PathBuf,
    /// Queries TSV (qid<TAB>query).
    #[arg(long)]
    queries: PathBuf,
    /// Index only the first N passages.
    #[arg(long)]
    limit: Option<usize>,
    /// Use only the first N queries.
    #[arg(long)]
    query_limit: Option<usize>,
    /// Analysis: default or english.
    #[arg(long, default_value = "default")]
    analyzer: String,
    /// Documents per upsert call.
    #[arg(long, default_value_t = 1000)]
    batch: usize,
    /// Write durability: each, batch or none.
    #[arg(long, default_value = "each")]
    durability: String,
    #[command(flatten)]
    common: Common,
  },
  /// Print this machine's description as JSON.
  Sysinfo,
}

fn list<T, E>(
  s: &str,
  parse: impl Fn(&str) -> Result<T, E>,
) -> Result<Vec<T>, E> {
  s.split(',')
    .map(str::trim)
    .filter(|v| !v.is_empty())
    .map(parse)
    .collect()
}

fn main() -> muninn_bench::Result<()> {
  match Cli::parse().cmd {
    Cmd::Hnsw {
      data,
      m,
      efc,
      ef,
      quant,
      queries,
      reuse_index,
      in_memory,
      common,
    } => {
      let pools = Pools::new(common.threads)?;
      let cfg = HnswRun {
        reuse_index,
        in_memory,
        data,
        m,
        ef_construction: efc,
        efs: list(&ef, str::parse::<usize>)?,
        quant: list(&quant, parse_quantization)?,
        k: common.k,
        query_limit: queries,
        runs: common.runs,
        work_dir: common.work_dir,
        out_dir: common.out_dir,
      };
      for f in run_hnsw(&cfg, &pools)? {
        println!("{}", f.display());
      }
    }
    Cmd::Flat {
      data,
      queries,
      common,
    } => {
      let pools = Pools::new(common.threads)?;
      let cfg = FlatRun {
        data,
        k: common.k,
        query_limit: Some(queries),
        runs: common.runs,
        work_dir: common.work_dir,
        out_dir: common.out_dir,
      };
      println!("{}", run_flat(&cfg, &pools)?.display());
    }
    Cmd::VerifyGt {
      data,
      sample,
      common,
    } => {
      let pools = Pools::new(common.threads)?;
      let c = verify_ground_truth(&data, sample, common.k, &pools)?;
      eprintln!(
        "{}: {} queries, id sets equal {:.4}, equal up to ties {:.4}, \
         recall {:.4}",
        c.dataset, c.sampled, c.id_set_match, c.match_with_ties, c.recall
      );
      let out = common.out_dir.join(format!("{}__gt-check.json", c.dataset));
      write_json(&out, &c)?;
      println!("{}", out.display());
      if c.match_with_ties < 0.99 {
        return Err(
          format!(
            "ground truth disagrees with exact search on {:.1}% of \
             sampled queries",
            (1.0 - c.match_with_ties) * 100.0
          )
          .into(),
        );
      }
    }
    Cmd::Text {
      collection,
      queries,
      limit,
      query_limit,
      analyzer,
      tantivy_index_threads,
      common,
    } => {
      let pools = Pools::new(common.threads)?;
      let cfg = TextRun {
        collection,
        queries,
        limit,
        query_limit,
        analyzer: AnalyzerKind::parse(&analyzer)?,
        k: common.k,
        runs: common.runs,
        tantivy_index_threads: list(
          &tantivy_index_threads,
          str::parse::<usize>,
        )?,
        work_dir: common.work_dir,
        out_dir: common.out_dir,
      };
      println!("{}", run_text(&cfg, &pools)?.display());
    }
    Cmd::EngineVector {
      data,
      m,
      efc,
      ef,
      queries,
      batch,
      durability,
      common,
    } => {
      let pools = Pools::new(common.threads)?;
      let cfg = EngineVectorRun {
        data,
        m,
        ef_construction: efc,
        efs: list(&ef, str::parse::<usize>)?,
        k: common.k,
        query_limit: queries,
        runs: common.runs,
        batch,
        durability,
        work_dir: common.work_dir,
        out_dir: common.out_dir,
      };
      println!("{}", run_engine_vector(&cfg, &pools)?.display());
    }
    Cmd::EngineText {
      collection,
      queries,
      limit,
      query_limit,
      analyzer,
      batch,
      durability,
      common,
    } => {
      let pools = Pools::new(common.threads)?;
      let cfg = EngineTextRun {
        text: TextRun {
          collection,
          queries,
          limit,
          query_limit,
          analyzer: AnalyzerKind::parse(&analyzer)?,
          k: common.k,
          runs: common.runs,
          tantivy_index_threads: Vec::new(),
          work_dir: common.work_dir,
          out_dir: common.out_dir,
        },
        batch,
        durability,
      };
      println!("{}", run_engine_text(&cfg, &pools)?.display());
    }
    Cmd::Sysinfo => {
      println!("{}", serde_json::to_string_pretty(&SystemInfo::collect())?);
    }
  }
  Ok(())
}
