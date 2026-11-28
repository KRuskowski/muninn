//! The `muninn` command-line tool: run the server, or work on a data
//! directory directly (import, search, stats, compact).
//!
//! The direct commands open the database themselves, so they cannot run
//! while a server has the same directory open; they fail with an
//! "already open" error instead.

use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use muninn::{
  Collection, Database, Document, Durability, Error, Options, Result,
};
use serde_json::{Value, json};

use crate::model::{self, SearchRequest, SearchResponse};
use crate::server::{AppState, ServerConfig};

/// muninn: embedded vector and full-text search database.
#[derive(Debug, Parser)]
#[command(name = "muninn", version, about, long_about = None)]
pub struct Cli {
  /// What to do.
  #[command(subcommand)]
  pub command: Command,
}

/// Subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
  /// Run the HTTP/JSON server.
  Serve(ServeArgs),
  /// Import newline-delimited JSON documents into a collection.
  Import(ImportArgs),
  /// Search a collection and print the hits as JSON.
  Search(SearchArgs),
  /// Print statistics of every collection as JSON.
  Stats(DataArgs),
  /// Merge a collection's on-disk parts into one, dropping deleted data.
  Compact(CompactArgs),
}

/// When writes are acknowledged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DurabilityArg {
  /// On disk before the write returns (safest).
  Each,
  /// On disk within --sync-interval-ms of returning.
  Batch,
  /// Left to the operating system (survives process crashes only).
  None,
}

/// The data directory and engine settings.
#[derive(Clone, Debug, Args)]
pub struct DataArgs {
  /// Data directory (created if missing).
  #[arg(long, value_name = "DIR")]
  pub data: PathBuf,
  /// When writes are acknowledged.
  #[arg(long, value_enum, default_value_t = DurabilityArg::Each)]
  pub durability: DurabilityArg,
  /// Longest delay before a write is on disk with --durability batch.
  #[arg(long, value_name = "MS", default_value_t = 10)]
  pub sync_interval_ms: u64,
  /// Documents kept in memory per collection before they are written to
  /// an on-disk segment (default 50000).
  #[arg(long, value_name = "N")]
  pub memtable_max_docs: Option<usize>,
  /// Do not merge on-disk segments in the background.
  #[arg(long)]
  pub no_auto_compact: bool,
}

impl DataArgs {
  /// The engine options these flags describe.
  pub fn options(&self) -> Options {
    let mut o = Options::default().durability(match self.durability {
      DurabilityArg::Each => Durability::SyncEachWrite,
      DurabilityArg::Batch => Durability::SyncBatch {
        max_delay: Duration::from_millis(self.sync_interval_ms.max(1)),
      },
      DurabilityArg::None => Durability::NoSync,
    });
    if let Some(n) = self.memtable_max_docs {
      o = o.memtable_max_docs(n);
    }
    o.auto_compact(!self.no_auto_compact)
  }

  /// Opens the database.
  pub fn open(&self) -> Result<Database> {
    Database::open(&self.data, self.options())
  }
}

/// Log verbosity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum LogLevel {
  /// Errors only.
  Error,
  /// Errors and warnings.
  Warn,
  /// Startup, shutdown and errors.
  Info,
  /// Also every request.
  Debug,
}

/// `muninn serve`.
#[derive(Clone, Debug, Args)]
pub struct ServeArgs {
  /// Data directory and engine settings.
  #[command(flatten)]
  pub data: DataArgs,
  /// Address to listen on. Use 127.0.0.1 unless the port is protected
  /// by a firewall: the API has no authentication.
  #[arg(long, default_value = "127.0.0.1:7700")]
  pub addr: String,
  /// Largest request body, e.g. 1048576, 512KiB, 64MiB, 1GiB.
  #[arg(long, value_name = "SIZE", default_value = "64MiB",
        value_parser = parse_size)]
  pub body_limit: usize,
  /// Documents per write batch of the bulk import endpoint.
  #[arg(long, value_name = "N", default_value_t = 1000)]
  pub bulk_batch: usize,
  /// Worker threads for network I/O (default: one per CPU).
  #[arg(long, value_name = "N")]
  pub threads: Option<usize>,
  /// Log verbosity (logs go to standard error).
  #[arg(long, value_enum, default_value_t = LogLevel::Info)]
  pub log_level: LogLevel,
}

/// `muninn import`.
#[derive(Clone, Debug, Args)]
pub struct ImportArgs {
  /// Data directory and engine settings.
  #[command(flatten)]
  pub data: DataArgs,
  /// Collection to import into.
  #[arg(long)]
  pub collection: String,
  /// Create the collection with this schema (JSON file) if it does not
  /// exist. If it exists, its schema must be the same.
  #[arg(long, value_name = "FILE")]
  pub create_schema: Option<PathBuf>,
  /// Documents per write batch (each batch is stored all-or-nothing).
  #[arg(long, value_name = "N", default_value_t = 1000)]
  pub batch: usize,
  /// Do not write the imported documents to segments at the end.
  #[arg(long)]
  pub no_flush: bool,
  /// No progress output.
  #[arg(long, short)]
  pub quiet: bool,
  /// JSONL file, one document per line; `-` reads standard input.
  pub file: PathBuf,
}

/// `muninn search`.
#[derive(Clone, Debug, Args)]
pub struct SearchArgs {
  /// Data directory and engine settings.
  #[command(flatten)]
  pub data: DataArgs,
  /// Collection to search.
  #[arg(long)]
  pub collection: String,
  /// Query text.
  #[arg(long)]
  pub text: Option<String>,
  /// JSON file holding the query vector: `[0.1, ...]` or
  /// `{"vector": [0.1, ...]}`.
  #[arg(long, value_name = "FILE")]
  pub vector_file: Option<PathBuf>,
  /// Payload filter in the JSON filter language, e.g.
  /// '{"eq":{"lang":"en"}}'.
  #[arg(long, value_name = "JSON")]
  pub filter: Option<String>,
  /// Number of results.
  #[arg(short, long, default_value_t = 10)]
  pub k: usize,
  /// HNSW candidate-list size.
  #[arg(long)]
  pub ef: Option<usize>,
  /// Text fields to search, comma separated (default: all).
  #[arg(long, value_delimiter = ',')]
  pub fields: Option<Vec<String>>,
  /// Hybrid fusion as JSON, e.g. '{"weighted":{"vector":0.7,"text":0.3}}'.
  #[arg(long, value_name = "JSON")]
  pub fusion: Option<String>,
  /// Include each hit's payload.
  #[arg(long)]
  pub with_payload: bool,
}

/// `muninn compact`.
#[derive(Clone, Debug, Args)]
pub struct CompactArgs {
  /// Data directory and engine settings.
  #[command(flatten)]
  pub data: DataArgs,
  /// Collection to compact.
  #[arg(long)]
  pub collection: String,
}

/// Parses a byte size: a number with an optional unit (`B`, `K`/`KB`/
/// `KiB`, `M`/`MB`/`MiB`, `G`/`GB`/`GiB`; all units are powers of 1024).
pub fn parse_size(s: &str) -> std::result::Result<usize, String> {
  let t = s.trim();
  let split = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
  let (num, unit) = t.split_at(split);
  let n: usize = num
    .parse()
    .map_err(|_| format!("{s:?} is not a size like 1048576 or 64MiB"))?;
  let mult: usize = match unit.trim().to_ascii_lowercase().as_str() {
    "" | "b" => 1,
    "k" | "kb" | "kib" => 1 << 10,
    "m" | "mb" | "mib" => 1 << 20,
    "g" | "gb" | "gib" => 1 << 30,
    u => return Err(format!("unknown size unit {u:?} in {s:?}")),
  };
  match n.checked_mul(mult) {
    Some(0) => Err("size must be positive".into()),
    Some(v) => Ok(v),
    None => Err(format!("size {s:?} is too large")),
  }
}

/// Runs the tool with the process's arguments.
pub fn main() -> ExitCode {
  let cli = Cli::parse();
  let r = match cli.command {
    Command::Serve(a) => serve(a),
    Command::Import(a) => import(a),
    Command::Search(a) => search(a),
    Command::Stats(a) => stats(a),
    Command::Compact(a) => compact(a),
  };
  match r {
    Ok(()) => ExitCode::SUCCESS,
    Err(e) => {
      eprintln!("error: {e}");
      ExitCode::FAILURE
    }
  }
}

type CliResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn init_logging(level: LogLevel) {
  let level = match level {
    LogLevel::Error => tracing::Level::ERROR,
    LogLevel::Warn => tracing::Level::WARN,
    LogLevel::Info => tracing::Level::INFO,
    LogLevel::Debug => tracing::Level::DEBUG,
  };
  let _ = tracing_subscriber::fmt()
    .with_max_level(level)
    .with_writer(std::io::stderr)
    .with_ansi(false)
    .with_target(false)
    .try_init();
}

/// `muninn serve`: runs until SIGINT/SIGTERM, then flushes and closes.
pub fn serve(a: ServeArgs) -> CliResult {
  init_logging(a.log_level);
  if a.bulk_batch == 0 || a.bulk_batch > crate::MAX_BULK_BATCH {
    return Err(
      format!(
        "--bulk-batch must be between 1 and {}",
        crate::MAX_BULK_BATCH
      )
      .into(),
    );
  }
  let mut rt = tokio::runtime::Builder::new_multi_thread();
  rt.enable_all().thread_name("muninn-http");
  if let Some(n) = a.threads {
    rt.worker_threads(n.max(1));
  }
  let rt = rt.build()?;
  tracing::info!("opening {}", a.data.data.display());
  let db = Arc::new(a.data.open()?);
  let cfg = ServerConfig {
    body_limit: a.body_limit,
    bulk_batch: a.bulk_batch,
  };
  let state = AppState::new(db.clone(), cfg);
  rt.block_on(async {
    let listener = tokio::net::TcpListener::bind(&a.addr).await?;
    tracing::info!("listening on http://{}", listener.local_addr()?);
    crate::serve(listener, state, crate::shutdown_signal()).await
  })?;
  drop(rt);
  tracing::info!("flushing and closing the database");
  crate::close(db, Duration::from_secs(30))?;
  Ok(())
}

/// Counts and timings of an import.
#[derive(Clone, Copy, Debug)]
pub struct ImportReport {
  /// Documents stored.
  pub docs: u64,
  /// Time to read, parse and store them.
  pub elapsed: Duration,
}

/// Bytes of complete lines read per chunk when a batch is not yet full.
const IMPORT_CHUNK_BYTES: usize = 16 << 20;

/// Imports newline-delimited JSON documents from `input` into `c`, in
/// atomic batches of `batch` documents, keeping the order of the input
/// (a later line with the same id wins). Reading, parsing (on all cores)
/// and writing overlap. `progress` is called after every batch with the
/// documents stored so far.
pub fn import_jsonl(
  c: &Collection,
  input: impl BufRead + Send,
  batch: usize,
  mut progress: impl FnMut(u64),
) -> Result<ImportReport> {
  let batch = batch.max(1);
  let start = Instant::now();
  // (first line number, number of lines, complete lines / documents)
  let (raw_tx, raw_rx) = sync_channel::<(u64, u64, Vec<u8>)>(2);
  let (doc_tx, doc_rx) = sync_channel::<(u64, u64, Vec<Document>)>(2);
  let mut docs = 0u64;
  std::thread::scope(|s| {
    let reader = s.spawn(move || -> Result<()> {
      let mut input = input;
      let mut line = 1u64;
      loop {
        let first = line;
        let mut buf = Vec::new();
        // Lines read, and non-blank ones (documents).
        let mut n = 0u64;
        let mut docs = 0;
        while docs < batch && buf.len() < IMPORT_CHUNK_BYTES {
          let start = buf.len();
          let read = input.read_until(b'\n', &mut buf)?;
          if read == 0 {
            break;
          }
          if !buf.ends_with(b"\n") {
            buf.push(b'\n');
          }
          n += 1;
          if !buf[start..].iter().all(u8::is_ascii_whitespace) {
            docs += 1;
          }
        }
        if n == 0 {
          return Ok(());
        }
        line += n;
        if raw_tx.send((first, n, buf)).is_err() {
          return Ok(());
        }
      }
    });
    let parser = s.spawn(move || -> Result<()> {
      for (first, n, buf) in raw_rx {
        let docs = model::parse_ndjson(&buf, first)?;
        if doc_tx.send((first, n, docs)).is_err() {
          break;
        }
      }
      Ok(())
    });
    let mut write_err = None;
    for (first, n, batch) in doc_rx {
      if batch.is_empty() {
        continue;
      }
      if let Err(e) = c.upsert(&batch) {
        // Validation errors get the lines of the rejected batch.
        write_err = Some(match e {
          Error::InvalidArgument(m) => Error::InvalidArgument(format!(
            "lines {first}-{}: {m}",
            first + n - 1
          )),
          e @ Error::DimensionMismatch { .. } => Error::InvalidArgument(
            format!("lines {first}-{}: {e}", first + n - 1),
          ),
          e => e,
        });
        break;
      }
      docs += batch.len() as u64;
      progress(docs);
    }
    // Dropping the receiver (end of loop) stops the other stages.
    let join = |r: std::thread::Result<Result<()>>| {
      r.unwrap_or_else(|_| {
        Err(Error::Io(std::io::Error::other("import thread panicked")))
      })
    };
    let parse_r = join(parser.join());
    let read_r = join(reader.join());
    match write_err {
      Some(e) => Err(e),
      None => parse_r.and(read_r),
    }
  })?;
  Ok(ImportReport {
    docs,
    elapsed: start.elapsed(),
  })
}

fn read_json_file(path: &Path) -> std::result::Result<Value, String> {
  let text = std::fs::read(path)
    .map_err(|e| format!("reading {}: {e}", path.display()))?;
  serde_json::from_slice(&text)
    .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))
}

/// `muninn import`.
pub fn import(a: ImportArgs) -> CliResult {
  let schema = match &a.create_schema {
    Some(p) => Some(model::schema_from_json(&read_json_file(p)?)?),
    None => None,
  };
  let db = a.data.open()?;
  let c = match (db.collection(&a.collection), schema) {
    (Ok(c), Some(s)) => {
      if *c.schema() != s {
        return Err(
          format!(
            "collection {:?} exists with a different schema: {}",
            a.collection,
            model::schema_to_json(c.schema())
          )
          .into(),
        );
      }
      c
    }
    (Ok(c), None) => c,
    (Err(Error::NotFound(_)), Some(s)) => {
      db.create_collection(&a.collection, s)?
    }
    (Err(Error::NotFound(_)), None) => {
      return Err(
        format!(
          "collection {:?} does not exist; pass --create-schema FILE to \
           create it",
          a.collection
        )
        .into(),
      );
    }
    (Err(e), _) => return Err(e.into()),
  };
  let input: Box<dyn BufRead + Send> = if a.file.as_os_str() == "-" {
    Box::new(BufReader::with_capacity(1 << 20, std::io::stdin()))
  } else {
    let f = std::fs::File::open(&a.file)
      .map_err(|e| format!("opening {}: {e}", a.file.display()))?;
    Box::new(BufReader::with_capacity(1 << 20, f))
  };
  let start = Instant::now();
  let mut last = Instant::now();
  let tty = std::io::stderr().is_terminal();
  let quiet = a.quiet;
  let report = import_jsonl(&c, input, a.batch, |n| {
    if quiet || last.elapsed() < Duration::from_secs(1) {
      return;
    }
    last = Instant::now();
    let rate = n as f64 / start.elapsed().as_secs_f64();
    if tty {
      eprint!("\rimported {n} documents ({rate:.0} docs/s)   ");
    } else {
      eprintln!("imported {n} documents ({rate:.0} docs/s)");
    }
  })?;
  if tty && !quiet {
    eprintln!();
  }
  let secs = report.elapsed.as_secs_f64();
  let rate = if secs > 0.0 {
    report.docs as f64 / secs
  } else {
    0.0
  };
  let mut line = format!(
    "imported {} documents into {:?} in {secs:.2} s ({rate:.0} docs/s)",
    report.docs, a.collection
  );
  if !a.no_flush {
    let f = Instant::now();
    c.flush()?;
    line += &format!("; written to disk in {:.2} s", f.elapsed().as_secs_f64());
  }
  println!("{line}");
  Ok(())
}

/// `muninn search`.
pub fn search(a: SearchArgs) -> CliResult {
  let vector = match &a.vector_file {
    None => None,
    Some(p) => {
      let v = read_json_file(p)?;
      let v = match v {
        Value::Object(mut o) => o
          .remove("vector")
          .ok_or_else(|| format!("{} has no \"vector\" key", p.display()))?,
        v => v,
      };
      Some(serde_json::from_value::<Vec<f32>>(v).map_err(|e| {
        format!(
          "{}: the vector must be an array of numbers: {e}",
          p.display()
        )
      })?)
    }
  };
  let parse = |what: &str,
               s: &Option<String>|
   -> std::result::Result<Option<Value>, String> {
    s.as_deref()
      .map(|s| {
        serde_json::from_str(s)
          .map_err(|e| format!("--{what} is not valid JSON: {e}"))
      })
      .transpose()
  };
  let req = SearchRequest {
    vector,
    text: a.text.clone(),
    k: Some(a.k),
    ef: a.ef,
    filter: parse("filter", &a.filter)?,
    fusion: parse("fusion", &a.fusion)?,
    with_payload: a.with_payload,
    fields: a.fields.clone(),
    ..SearchRequest::default()
  };
  let db = a.data.open()?;
  let c = db.collection(&a.collection)?;
  let start = Instant::now();
  let q = req.into_query()?;
  let hits = c.search(&q)?;
  let resp = SearchResponse {
    hits: &hits,
    took_us: crate::server::micros(start.elapsed()),
  };
  let mut out = std::io::stdout().lock();
  serde_json::to_writer_pretty(&mut out, &resp)?;
  writeln!(out)?;
  Ok(())
}

/// `muninn stats`.
pub fn stats(a: DataArgs) -> CliResult {
  let db = a.open()?;
  let mut out = serde_json::Map::new();
  for name in db.list_collections()? {
    let c = db.collection(&name)?;
    out.insert(
      name,
      json!({
        "schema": model::schema_to_json(c.schema()),
        "stats": model::stats_to_json(&c.stats()?),
      }),
    );
  }
  let mut stdout = std::io::stdout().lock();
  serde_json::to_writer_pretty(&mut stdout, &json!({ "collections": out }))?;
  writeln!(stdout)?;
  Ok(())
}

/// `muninn compact`.
pub fn compact(a: CompactArgs) -> CliResult {
  let db = a.data.open()?;
  let c = db.collection(&a.collection)?;
  let start = Instant::now();
  c.compact()?;
  let s = c.stats()?;
  println!(
    "compacted {:?} in {:.2} s: {} documents in {} segment(s)",
    a.collection,
    start.elapsed().as_secs_f64(),
    s.live_docs,
    s.segments.len()
  );
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sizes() {
    assert_eq!(parse_size("1048576"), Ok(1 << 20));
    assert_eq!(parse_size("64MiB"), Ok(64 << 20));
    assert_eq!(parse_size("512k"), Ok(512 << 10));
    assert_eq!(parse_size("1 GB"), Ok(1 << 30));
    assert!(parse_size("0").is_err());
    assert!(parse_size("12XB").is_err());
    assert!(parse_size("MiB").is_err());
  }

  #[test]
  fn cli_definition_is_consistent() {
    use clap::CommandFactory;
    Cli::command().debug_assert();
  }
}
