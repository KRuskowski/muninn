//! Timeouts and limits against a real server on 127.0.0.1: slow or idle
//! clients, too many connections or requests, too much body data in
//! flight, stuck bulk imports, and a shutdown that does not wait forever.

mod common;

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{TestDir, http};
use muninn::{Database, Options};
use muninn_server::{AppState, ServerConfig};
use serde_json::Value;

struct Server {
  addr: SocketAddr,
  db: Option<Arc<Database>>,
  rt: Option<tokio::runtime::Runtime>,
  stop: Option<tokio::sync::oneshot::Sender<()>>,
  task: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
}

impl Server {
  fn start(dir: &TestDir, cfg: ServerConfig) -> Server {
    Self::start_with_threads(dir, cfg, 512)
  }

  /// With at most `blocking` threads in tokio's blocking pool.
  fn start_with_threads(
    dir: &TestDir,
    cfg: ServerConfig,
    blocking: usize,
  ) -> Server {
    let opts = Options::default().auto_compact(false);
    let db = Arc::new(Database::open(dir.path(), opts).unwrap());
    let rt = tokio::runtime::Builder::new_multi_thread()
      .worker_threads(2)
      .max_blocking_threads(blocking)
      .enable_all()
      .build()
      .unwrap();
    let listener = rt
      .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
      .unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let state = AppState::new(db.clone(), cfg);
    let task = rt.spawn(muninn_server::serve(
      listener,
      state,
      async {
        let _ = stop_rx.await;
      },
      std::future::pending(),
    ));
    Server {
      addr,
      db: Some(db),
      rt: Some(rt),
      stop: Some(stop),
      task: Some(task),
    }
  }

  /// Signals shutdown and returns how long `serve` took to return.
  fn shutdown(&mut self) -> Duration {
    let t = Instant::now();
    if let Some(s) = self.stop.take() {
      let _ = s.send(());
    }
    if let (Some(rt), Some(task)) = (&self.rt, self.task.take()) {
      rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(20), task)
          .await
          .expect("serve did not return within 20 s")
          .unwrap()
          .unwrap();
      });
    }
    t.elapsed()
  }
}

impl Drop for Server {
  fn drop(&mut self) {
    if !std::thread::panicking() {
      self.shutdown();
    }
    if let Some(rt) = self.rt.take() {
      rt.shutdown_timeout(Duration::from_secs(1));
    }
    drop(self.db.take());
  }
}

fn connect(addr: SocketAddr) -> TcpStream {
  let s = TcpStream::connect(addr).unwrap();
  s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
  s
}

/// Reads until the server closes the connection or 5 s pass; returns what
/// arrived and whether the server closed it.
fn read_until_closed(s: &mut TcpStream) -> (String, bool) {
  let mut out = Vec::new();
  let mut buf = [0u8; 4096];
  loop {
    match s.read(&mut buf) {
      Ok(0) => return (String::from_utf8_lossy(&out).into(), true),
      Ok(n) => out.extend_from_slice(&buf[..n]),
      Err(e) if e.kind() == ErrorKind::ConnectionReset => {
        return (String::from_utf8_lossy(&out).into(), true);
      }
      Err(_) => return (String::from_utf8_lossy(&out).into(), false),
    }
  }
}

/// Status and JSON body of a raw response.
fn parse(resp: &str) -> (u16, Value) {
  let (head, body) = resp.split_once("\r\n\r\n").unwrap_or((resp, ""));
  let status = head
    .split(' ')
    .nth(1)
    .and_then(|s| s.parse().ok())
    .unwrap_or(0);
  (status, serde_json::from_str(body).unwrap_or(Value::Null))
}

/// Starts a request whose body never completes: headers announcing
/// `length` bytes, then only the first few. Returns once the server has
/// had time to start handling it.
fn stalled_request(
  addr: SocketAddr,
  path: &str,
  ct: &str,
  length: usize,
) -> TcpStream {
  let mut s = connect(addr);
  write!(
    s,
    "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: {ct}\r\n\
     Content-Length: {length}\r\n\r\n{{\"te"
  )
  .unwrap();
  // Let the server take it in (and its permits) before anything else.
  std::thread::sleep(Duration::from_millis(200));
  s
}

fn create(addr: SocketAddr, name: &str) {
  let r = http(
    addr,
    "PUT",
    &format!("/v1/collections/{name}"),
    Some(r#"{"text_fields": ["t"]}"#),
  );
  assert_eq!(r.0, 201, "{}", r.1);
}

/// Retries `f` for up to 3 s until it returns true.
fn eventually(mut f: impl FnMut() -> bool) -> bool {
  let end = Instant::now() + Duration::from_secs(3);
  while Instant::now() < end {
    if f() {
      return true;
    }
    std::thread::sleep(Duration::from_millis(20));
  }
  false
}

#[test]
fn slow_headers_and_idle_connections_are_cut_off() {
  let dir = TestDir::new("headers");
  let _s = Server::start(
    &dir,
    ServerConfig {
      header_timeout: Duration::from_millis(300),
      ..ServerConfig::default()
    },
  );
  let addr = _s.addr;
  // Slowloris: headers that never end.
  let mut c = connect(addr);
  write!(c, "GET /v1/health HTTP/1.1\r\nHost: 127").unwrap();
  let t = Instant::now();
  let (_, closed) = read_until_closed(&mut c);
  assert!(
    closed,
    "the server kept a connection with unfinished headers"
  );
  assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
  // A silent connection, and an idle keep-alive connection.
  let mut c = connect(addr);
  let (_, closed) = read_until_closed(&mut c);
  assert!(closed, "the server kept a silent connection");
  let mut c = connect(addr);
  write!(c, "GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
  let (resp, closed) = read_until_closed(&mut c);
  assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
  assert!(closed, "the server kept an idle keep-alive connection");
}

#[test]
fn connections_beyond_the_limit_are_closed() {
  let dir = TestDir::new("conns");
  let s = Server::start(
    &dir,
    ServerConfig {
      max_connections: 4,
      ..ServerConfig::default()
    },
  );
  let held: Vec<_> = (0..4).map(|_| connect(s.addr)).collect();
  // Accepted in order, so these four hold every slot.
  std::thread::sleep(Duration::from_millis(200));
  let mut extra = connect(s.addr);
  let t = Instant::now();
  let (_, closed) = read_until_closed(&mut extra);
  assert!(closed, "a fifth connection was kept open");
  assert!(t.elapsed() < Duration::from_secs(2));
  drop(held);
  assert!(eventually(|| {
    let mut c = connect(s.addr);
    let _ = write!(
      c,
      "GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    );
    read_until_closed(&mut c).0.starts_with("HTTP/1.1 200")
  }));
}

#[test]
fn requests_beyond_the_limit_get_503() {
  let dir = TestDir::new("requests");
  let s = Server::start(
    &dir,
    ServerConfig {
      max_requests: 1,
      ..ServerConfig::default()
    },
  );
  create(s.addr, "c");
  let stalled = stalled_request(
    s.addr,
    "/v1/collections/c/search",
    "application/json",
    100,
  );
  assert!(eventually(
    || http(s.addr, "GET", "/v1/health", None).0 == 503
  ));
  let r = http(s.addr, "GET", "/v1/health", None);
  assert_eq!(r.1["error"]["code"], "unavailable", "{}", r.1);
  drop(stalled);
  assert!(eventually(
    || http(s.addr, "GET", "/v1/health", None).0 == 200
  ));
}

#[test]
fn slow_bodies_time_out() {
  let dir = TestDir::new("slowbody");
  let s = Server::start(
    &dir,
    ServerConfig {
      body_timeout: Duration::from_millis(300),
      ..ServerConfig::default()
    },
  );
  create(s.addr, "c");
  let mut c = stalled_request(
    s.addr,
    "/v1/collections/c/search",
    "application/json",
    100,
  );
  let t = Instant::now();
  let (resp, _) = read_until_closed(&mut c);
  let (status, v) = parse(&resp);
  assert_eq!(status, 408, "{resp}");
  assert_eq!(v["error"]["code"], "request_timeout");
  assert!(t.elapsed() < Duration::from_secs(3));
}

#[test]
fn body_bytes_in_flight_are_capped() {
  let dir = TestDir::new("budget");
  let s = Server::start(
    &dir,
    ServerConfig {
      body_limit: 800,
      max_body_bytes_in_flight: 1000,
      ..ServerConfig::default()
    },
  );
  create(s.addr, "c");
  let stalled = stalled_request(
    s.addr,
    "/v1/collections/c/search",
    "application/json",
    700,
  );
  let body = format!(r#"{{"text": "{}"}}"#, "x".repeat(680));
  assert!(eventually(|| {
    let r = http(s.addr, "POST", "/v1/collections/c/search", Some(&body));
    r.0 == 503 && r.1["error"]["code"] == "unavailable"
  }));
  drop(stalled);
  assert!(eventually(|| {
    http(s.addr, "POST", "/v1/collections/c/search", Some(&body)).0 == 200
  }));
}

#[test]
fn stuck_bulk_imports_are_capped_and_time_out() {
  let dir = TestDir::new("bulkcap");
  let s = Server::start(
    &dir,
    ServerConfig {
      max_bulk: 1,
      bulk_idle_timeout: Duration::from_millis(1500),
      ..ServerConfig::default()
    },
  );
  create(s.addr, "c");
  let path = "/v1/collections/c/documents:bulk";
  let mut stalled = stalled_request(s.addr, path, "application/x-ndjson", 1000);
  let line = "{\"id\": 1, \"text\": {\"t\": \"x\"}}\n";
  assert!(eventually(|| {
    let r = http(s.addr, "POST", path, Some(line));
    r.0 == 503 && r.1["error"]["code"] == "unavailable"
  }));
  let (resp, _) = read_until_closed(&mut stalled);
  let (status, v) = parse(&resp);
  assert_eq!(status, 408, "{resp}");
  assert!(
    v["error"]["message"].as_str().unwrap().contains("no data"),
    "{v}"
  );
  assert!(eventually(
    || http(s.addr, "POST", path, Some(line)).0 == 200
  ));
}

/// Bulk requests that sent only their headers hold no thread: with a
/// blocking pool of 4 and 12 such requests open, a real import still
/// runs.
#[test]
fn bulk_requests_without_data_hold_no_threads() {
  let dir = TestDir::new("bulkthreads");
  let s = Server::start_with_threads(
    &dir,
    ServerConfig {
      max_bulk: 64,
      max_body_bytes_in_flight: 1 << 30,
      ..ServerConfig::default()
    },
    4,
  );
  create(s.addr, "c");
  let path = "/v1/collections/c/documents:bulk";
  let idle: Vec<_> = (0..12)
    .map(|_| stalled_request(s.addr, path, "application/x-ndjson", 1 << 20))
    .collect();
  std::thread::sleep(Duration::from_millis(300));
  let lines: String = (0..10)
    .map(|i| format!("{{\"id\": {i}, \"text\": {{\"t\": \"x\"}}}}\n"))
    .collect();
  let (tx, rx) = std::sync::mpsc::channel();
  let addr = s.addr;
  std::thread::spawn(move || {
    let _ = tx.send(http(addr, "POST", path, Some(&lines)));
  });
  let r = rx
    .recv_timeout(Duration::from_secs(5))
    .expect("a bulk import starved behind idle bulk requests");
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["count"], 10);
  drop(idle);
}

/// A connection stuck in a request does not hold up shutdown past the
/// deadline, and the database still closes cleanly afterwards.
#[test]
fn shutdown_is_bounded_and_the_database_still_closes() {
  let dir = TestDir::new("shutdown");
  let mut s = Server::start(
    &dir,
    ServerConfig {
      shutdown_timeout: Duration::from_millis(500),
      ..ServerConfig::default()
    },
  );
  create(s.addr, "c");
  let r = http(
    s.addr,
    "POST",
    "/v1/collections/c/documents",
    Some(r#"{"documents": [{"id": 1, "text": {"t": "kept"}}]}"#),
  );
  assert_eq!(r.0, 200);
  let _stalled = stalled_request(
    s.addr,
    "/v1/collections/c/search",
    "application/json",
    100,
  );
  std::thread::sleep(Duration::from_millis(200));
  let took = s.shutdown();
  assert!(took < Duration::from_secs(3), "shutdown took {took:?}");
  let db = s.db.take().unwrap();
  muninn_server::close(db, Duration::from_secs(5)).unwrap();
  // Closed for real: the directory opens again, with the data flushed.
  let db = Database::open(dir.path(), Options::default()).unwrap();
  let st = db.collection("c").unwrap().stats().unwrap();
  assert_eq!((st.live_docs, st.memtable_docs), (1, 0));
}
