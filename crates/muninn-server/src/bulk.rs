//! `POST /v1/collections/{name}/documents:bulk`: newline-delimited JSON
//! documents, streamed.
//!
//! The body is read frame by frame and cut into chunks of whole lines
//! (`batch` lines, or fewer once a chunk reaches [`MAX_CHUNK_BYTES`]). A
//! blocking worker parses each chunk on all cores and upserts it as one
//! atomic batch while the next chunk is being received. The total body
//! size is unlimited; one line may not exceed the body limit.

use std::time::Instant;

use axum::body::Body as RawBody;
use axum::extract::{RawQuery, State};
use axum::http::StatusCode;
use axum::response::Response;
use http_body_util::BodyExt;
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::{ApiError, ErrorCode};
use crate::model;
use crate::server::{
  AppState, MAX_BULK_BATCH, Params, json_response, micros, query_param,
};

/// A chunk is handed to the worker once its complete lines reach this
/// many bytes, even if it has fewer than `batch` documents.
const MAX_CHUNK_BYTES: usize = 16 << 20;

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

pub(crate) async fn bulk(
  State(st): State<AppState>,
  Params(name): Params<String>,
  RawQuery(query): RawQuery,
  body: RawBody,
) -> Result<Response, ApiError> {
  let start = Instant::now();
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
  let c = st.collection(&name)?;
  let line_limit = st.cfg.body_limit;

  // Two chunks in flight: one being written, one parsed or queued.
  let (tx, mut rx) = mpsc::channel::<Chunk>(2);
  let worker = tokio::task::spawn_blocking(move || {
    let mut out = Outcome {
      docs: 0,
      batches: 0,
      error: None,
    };
    while let Some(chunk) = rx.blocking_recv() {
      // Parse errors name their line; write errors get the chunk's lines.
      let r = model::parse_ndjson(&chunk.bytes, chunk.first_line)
        .map_err(ApiError::from)
        .and_then(|docs| {
          c.upsert(&docs).map(|()| docs.len()).map_err(|e| {
            let e = ApiError::from(e);
            ApiError::new(e.code, format!("{}: {}", chunk.lines(), e.message))
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

  let read_err = read_chunks(body, batch, line_limit, &tx).await.err();
  drop(tx);
  let out = worker
    .await
    .map_err(|e| ApiError::internal(format!("bulk worker failed: {e}")))?;
  // A worker error stops the reader (its sends fail), so it comes first.
  if let Some(e) = out.error.or(read_err) {
    return Err(ApiError::new(
      e.code,
      format!(
        "{}; the {} documents before it were stored",
        e.message, out.docs
      ),
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

/// Reads `body`, sending chunks of complete lines to `tx`. Returns early
/// (with `Ok`) when the worker has stopped.
async fn read_chunks(
  mut body: RawBody,
  batch: usize,
  line_limit: usize,
  tx: &mpsc::Sender<Chunk>,
) -> Result<(), ApiError> {
  let mut buf: Vec<u8> = Vec::new();
  // In `buf`: complete lines, of which `docs` are not empty, ending at
  // `done`; then the start of the next line.
  let mut lines = 0u64;
  let mut docs = 0usize;
  let mut done = 0usize;
  let mut next_line = 1u64;
  let too_long = |line: u64| {
    ApiError::new(
      ErrorCode::PayloadTooLarge,
      format!("line {line} exceeds the limit of {line_limit} bytes"),
    )
  };
  while let Some(frame) = body.frame().await {
    let frame = frame
      .map_err(|e| ApiError::invalid(format!("reading request body: {e}")))?;
    let Ok(data) = frame.into_data() else {
      continue; // trailers
    };
    let mut scan = buf.len();
    buf.extend_from_slice(&data);
    while let Some(off) = buf[scan..].iter().position(|&b| b == b'\n') {
      let nl = scan + off;
      let line = &buf[done..nl];
      if line.len() > line_limit {
        return Err(too_long(next_line + lines));
      }
      lines += 1;
      if !line.is_empty() && line != b"\r" {
        docs += 1;
      }
      done = nl + 1;
      scan = done;
      if docs == batch || done >= MAX_CHUNK_BYTES {
        let rest = buf.split_off(done);
        let bytes = std::mem::replace(&mut buf, rest);
        let first_line = next_line;
        next_line += lines;
        if tx.send(Chunk { first_line, bytes }).await.is_err() {
          return Ok(());
        }
        lines = 0;
        docs = 0;
        done = 0;
        scan = 0;
      }
    }
    if buf.len() - done > line_limit {
      return Err(too_long(next_line + lines));
    }
  }
  if !buf.iter().all(u8::is_ascii_whitespace) {
    let _ = tx
      .send(Chunk {
        first_line: next_line,
        bytes: buf,
      })
      .await;
  }
  Ok(())
}
