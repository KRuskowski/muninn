//! Helpers shared by the server's integration tests.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use muninn::{Database, Options};
use muninn_server::{AppState, ServerConfig};
use serde_json::Value;
use tower::ServiceExt;

/// A fresh directory under cargo's per-target temp dir, removed on drop.
pub struct TestDir(PathBuf);

impl TestDir {
  pub fn new(name: &str) -> TestDir {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let p = Path::new(env!("CARGO_TARGET_TMPDIR"))
      .join(format!("muninn-server-{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    TestDir(p)
  }

  pub fn path(&self) -> &Path {
    &self.0
  }
}

impl Drop for TestDir {
  fn drop(&mut self) {
    let _ = std::fs::remove_dir_all(&self.0);
  }
}

/// A router over a database in `dir`. Background merging is off so that
/// tests see segments exactly as their flushes made them.
pub fn app(dir: &Path, cfg: ServerConfig) -> Router {
  let db = Database::open(dir, Options::default().auto_compact(false)).unwrap();
  muninn_server::router(AppState::new(Arc::new(db), cfg))
}

/// A router over a database in `dir` opened with `opts`.
pub fn app_with(dir: &Path, cfg: ServerConfig, opts: Options) -> Router {
  let db = Database::open(dir, opts).unwrap();
  muninn_server::router(AppState::new(Arc::new(db), cfg))
}

/// Sends one request through `app` in process.
pub async fn call(
  app: &Router,
  method: &str,
  uri: &str,
  body: impl Into<Body>,
) -> (StatusCode, Value) {
  let req = Request::builder()
    .method(method)
    .uri(uri)
    .header("content-type", "application/json")
    .body(body.into())
    .unwrap();
  let resp = app.clone().oneshot(req).await.unwrap();
  let status = resp.status();
  let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
    .await
    .unwrap();
  let v = if bytes.is_empty() {
    Value::Null
  } else {
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
      panic!("{method} {uri}: body is not JSON ({e}): {bytes:?}")
    })
  };
  (status, v)
}

/// Asserts an error response with `code`, returning its message.
pub fn assert_error(
  r: &(StatusCode, Value),
  status: u16,
  code: &str,
) -> String {
  assert_eq!(r.0.as_u16(), status, "{}", r.1);
  assert_eq!(r.1["error"]["code"], code, "{}", r.1);
  r.1["error"]["message"].as_str().unwrap().to_string()
}

/// One HTTP/1.1 request over a new TCP connection (blocking, std only).
/// Returns the status and the JSON body.
pub fn http(
  addr: SocketAddr,
  method: &str,
  path: &str,
  body: Option<&str>,
) -> (u16, Value) {
  let mut s = TcpStream::connect(addr).unwrap();
  let body = body.unwrap_or("");
  write!(
    s,
    "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\
     Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
    body.len()
  )
  .unwrap();
  let mut resp = Vec::new();
  s.read_to_end(&mut resp).unwrap();
  let text = String::from_utf8(resp).unwrap();
  let (head, body) = text.split_once("\r\n\r\n").unwrap();
  let status = head
    .split(' ')
    .nth(1)
    .and_then(|s| s.parse().ok())
    .unwrap_or_else(|| panic!("bad status line: {head}"));
  let v = if body.is_empty() {
    Value::Null
  } else {
    serde_json::from_str(body).unwrap()
  };
  (status, v)
}
