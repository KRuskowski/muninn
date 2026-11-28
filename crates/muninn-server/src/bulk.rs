//! `POST /v1/collections/{name}/documents:bulk`: newline-delimited JSON
//! documents, streamed.
//!
//! The body is read frame by frame and cut into chunks of whole lines
//! (`batch` documents, or fewer once a chunk reaches [`MAX_CHUNK_BYTES`]).
//! A blocking worker, started once the first chunk is complete, parses
//! each chunk on all cores and upserts it as one atomic batch while the
//! next chunk is being received. The total body size is unlimited; a line
//! may not exceed [`crate::ServerConfig::bulk_line_limit`], and the client
//! may not pause longer than [`crate::ServerConfig::bulk_idle_timeout`].
//!
//! Memory per request is bounded: the buffer (one chunk plus one line),
//! two queued chunks and the one being parsed. That amount is reserved
//! from the in-flight body budget for the whole request, and at most
//! [`crate::ServerConfig::max_bulk`] bulk requests run at once.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use axum::body::Body as RawBody;
use axum::extract::{RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use http_body_util::BodyExt;
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::{ApiError, ErrorCode};
use crate::guard;
use crate::model;
use crate::server::{
  AppState, MAX_BULK_BATCH, Params, json_response, micros, query_param,
};

/// A chunk is handed to the worker once its complete lines reach this
/// many bytes, even if it has fewer than `batch` documents.
const MAX_CHUNK_BYTES: usize = 4 << 20;

/// Complete lines and the number of the first one.
struct Chunk {
  first_line: u64,
  bytes: Vec<u8>,
}

impl Chunk {
  /// "line N" or "lines N-M", for error messages.
  fn lines(&self) -> String {
    let body = self.bytes.strip_suffix(b"\n").unwrap_or(&self.bytes);
    let n = body.iter().filter(|&&b| b == b'\n').count() as u64;
    match n {
      0 => format!("line {}", self.first_line),
      n => format!("lines {}-{}", self.first_line, self.first_line + n),
    }
  }
}

/// What the worker did before it stopped.
struct Outcome {
  docs: u64,
  batches: u64,
  error: Option<ApiError>,
}

impl Outcome {
  fn empty() -> Self {
    Outcome {
      docs: 0,
      batches: 0,
      error: None,
    }
  }
}

pub(crate) async fn bulk(
  State(st): State<AppState>,
  Params(name): Params<String>,
  RawQuery(query): RawQuery,
  headers: HeaderMap,
  body: RawBody,
) -> Result<Response, ApiError> {
  let start = Instant::now();
  guard::check_content_type(
    &headers,
    &["application/x-ndjson", "application/json"],
  )?;
  let batch = match query_param(&query, "batch") {
    None => st.cfg.bulk_batch,
    Some(s) => s
      .parse::<usize>()
      .ok()
      .filter(|n| (1..=MAX_BULK_BATCH).contains(n))
      .ok_or_else(|| {
        ApiError::invalid(format!(
          "batch must be an integer between 1 and {MAX_BULK_BATCH}, got {s:?}"
        ))
      })?,
  };
  let c = st.collection(&name).await?;
  let Ok(_slot) = st.limits.bulk.clone().try_acquire_owned() else {
    return Err(ApiError::new(
      ErrorCode::Unavailable,
      format!(
        "{} bulk imports are already running; retry shortly",
        st.cfg.max_bulk
      ),
    ));
  };
  let line_limit = st.cfg.bulk_line_limit;
  let _memory =
    guard::reserve_body_bytes(&st, 4 * MAX_CHUNK_BYTES + 2 * line_limit)?;
  let mut reader =
    Reader::new(body, batch, line_limit, st.cfg.bulk_idle_timeout);

  // No worker thread until there is something to write.
  let mut read_err = None;
  let first = match reader.next_chunk().await {
    Ok(Some(c)) => Some(c),
    Ok(None) => None,
    Err(e) => {
      read_err = Some(e);
      None
    }
  };
  let out = match first {
    None => Outcome::empty(),
    Some(first) => {
      // Two chunks queued while one is parsed and written.
      let (tx, mut rx) = mpsc::channel::<Chunk>(2);
      let worker = tokio::task::spawn_blocking(move || {
        let mut out = Outcome::empty();
        while let Some(chunk) = rx.blocking_recv() {
          // Parse errors name their line; write errors get the chunk's
          // lines.
          let r = model::parse_ndjson(&chunk.bytes, chunk.first_line)
            .map_err(ApiError::from)
            .and_then(|docs| {
              c.upsert(&docs).map(|()| docs.len()).map_err(|e| {
                let e = ApiError::from(e);
                ApiError::new(
                  e.code,
                  format!("{}: {}", chunk.lines(), e.message),
                )
              })
            });
          match r {
            Ok(0) => {}
            Ok(n) => {
              out.docs += n as u64;
              out.batches += 1;
            }
            Err(e) => {
              out.error = Some(e);
              break;
            }
          }
        }
        out
      });
      if tx.send(first).await.is_ok() {
        loop {
          match reader.next_chunk().await {
            Ok(Some(c)) => {
              if tx.send(c).await.is_err() {
                break; // the worker stopped on an error
              }
            }
            Ok(None) => break,
            Err(e) => {
              read_err = Some(e);
              break;
            }
          }
        }
      }
      drop(tx);
      worker
        .await
        .map_err(|e| ApiError::internal(format!("bulk worker failed: {e}")))?
    }
  };
  // A worker error stops the reader (its sends fail), so it comes first.
  if let Some(e) = out.error.or(read_err) {
    return Err(ApiError::new(
      e.code,
      format!("{}; {}", e.message, model::stored_before(out.docs)),
    ));
  }
  json_response(
    StatusCode::OK,
    &json!({
      "count": out.docs,
      "batches": out.batches,
      "took_us": micros(start.elapsed()),
    }),
  )
}

/// Cuts a streamed body into chunks of complete lines.
struct Reader {
  body: RawBody,
  batch: usize,
  line_limit: usize,
  idle: Duration,
  /// Received bytes not yet handed out: complete lines up to `done`
  /// (`docs` of them non-empty, `lines` in all), then a partial line.
  buf: Vec<u8>,
  done: usize,
  lines: u64,
  docs: usize,
  /// Number of the first line in `buf`.
  next_line: u64,
  ready: VecDeque<Chunk>,
  eof: bool,
}

impl Reader {
  fn new(
    body: RawBody,
    batch: usize,
    line_limit: usize,
    idle: Duration,
  ) -> Self {
    Reader {
      body,
      batch,
      line_limit,
      idle,
      buf: Vec::new(),
      done: 0,
      lines: 0,
      docs: 0,
      next_line: 1,
      ready: VecDeque::new(),
      eof: false,
    }
  }

  fn too_long(&self, line: u64) -> ApiError {
    ApiError::new(
      ErrorCode::PayloadTooLarge,
      format!("line {line} exceeds the limit of {} bytes", self.line_limit),
    )
  }

  /// The next chunk, or `None` at the end of the body.
  async fn next_chunk(&mut self) -> Result<Option<Chunk>, ApiError> {
    loop {
      if let Some(c) = self.ready.pop_front() {
        return Ok(Some(c));
      }
      if self.eof {
        return Ok(None);
      }
      let frame = tokio::time::timeout(self.idle, self.body.frame())
        .await
        .map_err(|_| {
          ApiError::new(
            ErrorCode::RequestTimeout,
            format!("no data received for {:?}", self.idle),
          )
        })?;
      match frame {
        None => self.finish(),
        Some(f) => {
          let f = f.map_err(|e| {
            ApiError::invalid(format!("reading request body: {e}"))
          })?;
          if let Ok(data) = f.into_data() {
            self.push(&data)?;
          }
        }
      }
    }
  }

  /// Takes in one frame: finds its line ends, queues every chunk that is
  /// complete, and keeps the rest. Each byte is copied into `buf` once and
  /// into its chunk once; what follows the last chunk moves down once per
  /// frame.
  fn push(&mut self, data: &[u8]) -> Result<(), ApiError> {
    let mut scan = self.buf.len();
    self.buf.extend_from_slice(data);
    // Start of the complete lines not yet in a chunk.
    let mut start = 0;
    while let Some(off) = self.buf[scan..].iter().position(|&b| b == b'\n') {
      let nl = scan + off;
      let line = &self.buf[self.done..nl];
      if line.len() > self.line_limit {
        return Err(self.too_long(self.next_line + self.lines));
      }
      self.lines += 1;
      if !line.is_empty() && line != b"\r" {
        self.docs += 1;
      }
      self.done = nl + 1;
      scan = self.done;
      if self.docs == self.batch || self.done - start >= MAX_CHUNK_BYTES {
        // The whole buffer is the chunk: hand it over without a copy.
        let whole = start == 0 && self.done == self.buf.len();
        let bytes = if whole {
          std::mem::take(&mut self.buf)
        } else {
          self.buf[start..self.done].to_vec()
        };
        self.ready.push_back(Chunk {
          first_line: self.next_line,
          bytes,
        });
        self.next_line += self.lines;
        self.lines = 0;
        self.docs = 0;
        if whole {
          self.done = 0;
          scan = 0;
        } else {
          start = self.done;
        }
      }
    }
    if start > 0 {
      self.buf.drain(..start);
      self.done -= start;
    }
    if self.buf.len() - self.done > self.line_limit {
      return Err(self.too_long(self.next_line + self.lines));
    }
    Ok(())
  }

  /// End of body: whatever is left (with or without a final newline) is
  /// the last chunk.
  fn finish(&mut self) {
    self.eof = true;
    let rest = std::mem::take(&mut self.buf);
    if !rest.iter().all(u8::is_ascii_whitespace) {
      self.ready.push_back(Chunk {
        first_line: self.next_line,
        bytes: rest,
      });
    }
  }
}
