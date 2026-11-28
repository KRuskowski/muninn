//! Load generator: search QPS and latency (vector, text, hybrid) and
//! upsert throughput over HTTP on 127.0.0.1, next to the same work done
//! through the embedded API in the same process, so the difference is the
//! server's overhead (HTTP, JSON, thread hand-off).
//!
//! ```text
//! cargo run --release -p muninn-server --example server_bench -- [dir]
//! ```
//!
//! Environment: `N` documents (default 100 000), `DIM` (128), `REQUESTS`
//! per measurement (4000), `CONC` concurrency levels (`1,8,32`),
//! `UPSERT_DOCS` (50 000), `UPSERT_BATCH` (100), `UPSERT_CONC` (8),
//! `SERVER_THREADS` async workers of the server (2), `CLIENT_THREADS`
//! of the load generator (2), `LABEL` for the machine.
//!
//! Client, server and engine share the machine, so the HTTP numbers
//! include the load generator's own CPU use.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::client::conn::http1::{self, SendRequest};
use hyper_util::rt::TokioIo;
use muninn::{
  Collection, Database, Document, FieldSpec, Metric, Options, Query, Schema,
  VectorField,
};
use muninn_server::{AppState, ServerConfig};
use serde_json::json;

fn env(name: &str, default: usize) -> usize {
  std::env::var(name)
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(default)
}

struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
  }
  fn f32(&mut self) -> f32 {
    (self.next() >> 40) as f32 / (1u64 << 24) as f32
  }
  fn normal(&mut self) -> f32 {
    (0..6).map(|_| self.f32()).sum::<f32>() - 3.0
  }
  fn below(&mut self, n: usize) -> usize {
    (self.next() % n as u64) as usize
  }
  /// Zipf-like word index: small indexes are much more frequent.
  fn word(&mut self, vocab: usize) -> usize {
    let u = self.f32().max(1e-6);
    ((vocab as f32).powf(u) as usize - 1).min(vocab - 1)
  }
}

const VOCAB: usize = 5000;
const CLUSTERS: usize = 100;

/// Clustered vectors, 8 to 12 words of Zipf-like text, a small payload.
struct Gen {
  rng: Rng,
  dim: usize,
  centroids: Vec<Vec<f32>>,
}

impl Gen {
  fn new(dim: usize) -> Gen {
    let mut rng = Rng(42);
    let centroids = (0..CLUSTERS)
      .map(|_| (0..dim).map(|_| rng.normal()).collect())
      .collect();
    Gen {
      rng,
      dim,
      centroids,
    }
  }

  fn vector(&mut self) -> Vec<f32> {
    let c = self.rng.below(CLUSTERS);
    (0..self.dim)
      .map(|d| self.centroids[c][d] + 0.3 * self.rng.normal())
      .collect()
  }

  fn text(&mut self, words: usize) -> String {
    (0..words)
      .map(|_| format!("w{}", self.rng.word(VOCAB)))
      .collect::<Vec<_>>()
      .join(" ")
  }

  fn doc(&mut self, id: u64) -> Document {
    let words = 8 + self.rng.below(5);
    let lang = ["en", "de", "fr", "es"][self.rng.below(4)];
    Document::new(id)
      .vector(self.vector())
      .text("body", self.text(words))
      .payload(json!({"lang": lang, "n": id}))
  }
}

fn doc_json(d: &Document) -> serde_json::Value {
  let id = match &d.id {
    muninn::Id::U64(n) => json!(n),
    muninn::Id::Str(s) => json!(s),
  };
  json!({"id": id, "vector": d.vector, "text": d.text, "payload": d.payload})
}

#[derive(Clone, Copy)]
enum Mode {
  Vector,
  Text,
  Hybrid,
}

impl Mode {
  fn name(self) -> &'static str {
    match self {
      Mode::Vector => "vector",
      Mode::Text => "text",
      Mode::Hybrid => "hybrid",
    }
  }
}

/// Throughput and latency of one measurement.
struct Stat {
  per_sec: f64,
  p50_us: f64,
  p99_us: f64,
}

fn stat(units: usize, wall: Duration, mut lat_ns: Vec<u64>) -> Stat {
  lat_ns.sort_unstable();
  let pct = |p: f64| {
    let i = ((lat_ns.len() as f64 * p) as usize).min(lat_ns.len() - 1);
    lat_ns[i] as f64 / 1000.0
  };
  Stat {
    per_sec: units as f64 / wall.as_secs_f64(),
    p50_us: pct(0.50),
    p99_us: pct(0.99),
  }
}

/// Runs `n` calls of `f(i)` on `conc` threads.
fn embedded(
  conc: usize,
  n: usize,
  f: impl Fn(usize) + Sync,
) -> (Duration, Vec<u64>) {
  let start = Instant::now();
  let lat = std::thread::scope(|s| {
    let hs: Vec<_> = (0..conc)
      .map(|t| {
        let f = &f;
        s.spawn(move || {
          let mut lat = Vec::new();
          let mut i = t;
          while i < n {
            let t0 = Instant::now();
            f(i);
            lat.push(t0.elapsed().as_nanos() as u64);
            i += conc;
          }
          lat
        })
      })
      .collect();
    hs.into_iter()
      .flat_map(|h| h.join().unwrap())
      .collect::<Vec<_>>()
  });
  (start.elapsed(), lat)
}

async fn connect(addr: SocketAddr) -> SendRequest<Full<Bytes>> {
  let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
  tcp.set_nodelay(true).unwrap();
  let (send, conn) = http1::handshake(TokioIo::new(tcp)).await.unwrap();
  tokio::spawn(async move {
    let _ = conn.await;
  });
  send
}

async fn post(
  send: &mut SendRequest<Full<Bytes>>,
  path: &str,
  body: Bytes,
) -> Bytes {
  send.ready().await.unwrap();
  let req = hyper::Request::post(path)
    .header("host", "127.0.0.1")
    .header("content-type", "application/json")
    .body(Full::new(body))
    .unwrap();
  let resp = send.send_request(req).await.unwrap();
  let status = resp.status();
  let bytes = resp.into_body().collect().await.unwrap().to_bytes();
  assert!(status.is_success(), "{status}: {bytes:?}");
  bytes
}

/// Sends `bodies[i]` for `i` in `0..n` to `path` over `conc` keep-alive
/// connections.
async fn http_load(
  addr: SocketAddr,
  path: &'static str,
  conc: usize,
  n: usize,
  bodies: Arc<Vec<Bytes>>,
) -> (Duration, Vec<u64>) {
  let mut conns = Vec::new();
  for _ in 0..conc {
    conns.push(connect(addr).await);
  }
  let start = Instant::now();
  let tasks: Vec<_> = conns
    .into_iter()
    .enumerate()
    .map(|(t, mut send)| {
      let bodies = bodies.clone();
      tokio::spawn(async move {
        let mut lat = Vec::new();
        let mut i = t;
        while i < n {
          let body = bodies[i % bodies.len()].clone();
          let t0 = Instant::now();
          post(&mut send, path, body).await;
          lat.push(t0.elapsed().as_nanos() as u64);
          i += conc;
        }
        lat
      })
    })
    .collect();
  let mut lat = Vec::new();
  for t in tasks {
    lat.extend(t.await.unwrap());
  }
  (start.elapsed(), lat)
}

fn schema(dim: usize) -> Schema {
  Schema::vector(VectorField::hnsw(dim, Metric::Cosine))
    .with_text_fields(["body"])
    .with_filterable(FieldSpec::keyword("lang"))
    .with_filterable(FieldSpec::integer("n"))
}

fn main() {
  let n = env("N", 100_000);
  let dim = env("DIM", 128);
  let requests = env("REQUESTS", 4000);
  let concs: Vec<usize> = std::env::var("CONC")
    .unwrap_or_else(|_| "1,8,32".into())
    .split(',')
    .filter_map(|s| s.trim().parse().ok())
    .collect();
  let up_docs = env("UPSERT_DOCS", 50_000);
  let up_batch = env("UPSERT_BATCH", 100);
  let up_conc = env("UPSERT_CONC", 8);
  let label = std::env::var("LABEL").unwrap_or_else(|_| "unlabelled".into());

  let dir = std::env::args()
    .nth(1)
    .map(PathBuf::from)
    .unwrap_or_else(|| {
      std::env::temp_dir()
        .join(format!("muninn-server-bench-{}", std::process::id()))
    });
  let _ = std::fs::remove_dir_all(&dir);
  let opts = Options::default().sync_every(Duration::from_millis(10));
  let db = Arc::new(Database::open(&dir, opts).unwrap());

  // Load and compact the search collection through the embedded API.
  eprintln!("loading {n} documents, dim {dim}");
  let c = db.create_collection("bench", schema(dim)).unwrap();
  let mut g = Gen::new(dim);
  let t = Instant::now();
  let mut id = 0u64;
  while (id as usize) < n {
    let b: Vec<Document> = (0..1000.min(n - id as usize))
      .map(|_| {
        id += 1;
        g.doc(id - 1)
      })
      .collect();
    c.upsert(&b).unwrap();
  }
  let load = t.elapsed();
  let t = Instant::now();
  c.compact().unwrap();
  eprintln!(
    "loaded in {:.1} s, compacted in {:.1} s",
    load.as_secs_f64(),
    t.elapsed().as_secs_f64()
  );
  let index = c.stats().unwrap().segments[0].vector_index;

  // Queries: one set, used by both sides.
  let nq = 1000;
  let mut qg = Gen {
    rng: Rng(7),
    dim,
    centroids: g.centroids.clone(),
  };
  let qv: Vec<Vec<f32>> = (0..nq).map(|_| qg.vector()).collect();
  let qt: Vec<String> = (0..nq).map(|_| qg.text(3)).collect();
  let query = |m: Mode, i: usize| -> Query {
    let i = i % nq;
    match m {
      Mode::Vector => Query::vector(qv[i].clone()),
      Mode::Text => Query::text(qt[i].clone()),
      Mode::Hybrid => Query::hybrid(qv[i].clone(), qt[i].clone()),
    }
    .k(10)
  };
  let body = |m: Mode, i: usize| -> Bytes {
    let v = match m {
      Mode::Vector => json!({"vector": qv[i], "k": 10}),
      Mode::Text => json!({"text": qt[i], "k": 10}),
      Mode::Hybrid => json!({"vector": qv[i], "text": qt[i], "k": 10}),
    };
    Bytes::from(serde_json::to_vec(&v).unwrap())
  };

  // The server, on its own runtime.
  let server_rt = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(env("SERVER_THREADS", 2))
    .thread_name("bench-server")
    .enable_all()
    .build()
    .unwrap();
  let listener = server_rt
    .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
    .unwrap();
  let addr = listener.local_addr().unwrap();
  let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
  let state = AppState::new(db.clone(), ServerConfig::default());
  let server = server_rt.spawn(muninn_server::serve(listener, state, async {
    let _ = stop_rx.await;
  }));
  let client_rt = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(env("CLIENT_THREADS", 2))
    .thread_name("bench-client")
    .enable_all()
    .build()
    .unwrap();

  println!(
    "## muninn-server overhead ({label}; shared VM, pinned cores)\n\n\
     {n} docs, dim {dim}, one {index} segment, k = 10, {requests} \
     requests per row.\n\n\
     | search | conc | embedded QPS | HTTP QPS | HTTP/embedded | \
     embedded p50 us | HTTP p50 us | embedded p99 us | HTTP p99 us |\n\
     |---|---|---|---|---|---|---|---|---|"
  );
  for m in [Mode::Vector, Mode::Text, Mode::Hybrid] {
    let queries: Vec<Query> = (0..nq).map(|i| query(m, i)).collect();
    let bodies = Arc::new((0..nq).map(|i| body(m, i)).collect::<Vec<_>>());
    // Warm-up of both paths.
    embedded(4, nq, |i| {
      c.search(&queries[i % nq]).unwrap();
    });
    client_rt.block_on(http_load(
      addr,
      "/v1/collections/bench/search",
      4,
      nq,
      bodies.clone(),
    ));
    for &conc in &concs {
      let (w, l) = embedded(conc, requests, |i| {
        let h = c.search(&queries[i % nq]).unwrap();
        assert!(!h.is_empty());
      });
      let e = stat(requests, w, l);
      let (w, l) = client_rt.block_on(http_load(
        addr,
        "/v1/collections/bench/search",
        conc,
        requests,
        bodies.clone(),
      ));
      let h = stat(requests, w, l);
      println!(
        "| {} | {conc} | {:.0} | {:.0} | {:.2} | {:.0} | {:.0} | {:.0} | {:.0} |",
        m.name(),
        e.per_sec,
        h.per_sec,
        h.per_sec / e.per_sec,
        e.p50_us,
        h.p50_us,
        e.p99_us,
        h.p99_us
      );
    }
  }

  // Upserts: the same batches embedded, as JSON requests, and as one
  // NDJSON bulk stream, each into a fresh collection.
  let batches: Vec<Vec<Document>> = (0..up_docs.div_ceil(up_batch))
    .map(|b| {
      (0..up_batch.min(up_docs - b * up_batch))
        .map(|i| g.doc((b * up_batch + i) as u64))
        .collect()
    })
    .collect();
  let up_bodies: Vec<Bytes> = batches
    .iter()
    .map(|b| {
      let docs: Vec<_> = b.iter().map(doc_json).collect();
      Bytes::from(serde_json::to_vec(&json!({ "documents": docs })).unwrap())
    })
    .collect();
  let ndjson: Vec<u8> = batches
    .iter()
    .flatten()
    .flat_map(|d| {
      let mut l = serde_json::to_vec(&doc_json(d)).unwrap();
      l.push(b'\n');
      l
    })
    .collect();
  let fresh = |name: &str| -> Collection {
    db.create_collection(name, schema(dim)).unwrap()
  };
  println!(
    "\n| upsert ({up_docs} docs) | conc | docs/s | p50 us/batch | p99 us/batch |\n\
     |---|---|---|---|---|"
  );
  let ce = fresh("up_embedded");
  let (w, l) = embedded(up_conc, batches.len(), |i| {
    ce.upsert(&batches[i]).unwrap();
  });
  let s = stat(up_docs, w, l);
  println!(
    "| embedded, batches of {up_batch} | {up_conc} | {:.0} | {:.0} | {:.0} |",
    s.per_sec, s.p50_us, s.p99_us
  );
  // Untimed: the segment build (HNSW, on all cores) of one phase must not
  // run during the next.
  ce.flush().unwrap();
  let ch = fresh("up_http");
  let (w, l) = client_rt.block_on(http_load(
    addr,
    "/v1/collections/up_http/documents",
    up_conc,
    up_bodies.len(),
    Arc::new(up_bodies),
  ));
  let s = stat(up_docs, w, l);
  println!(
    "| HTTP JSON, batches of {up_batch} | {up_conc} | {:.0} | {:.0} | {:.0} |",
    s.per_sec, s.p50_us, s.p99_us
  );
  ch.flush().unwrap();
  drop(ch);
  fresh("up_bulk");
  let ndjson = Bytes::from(ndjson);
  let (w, _) = client_rt.block_on(async {
    let mut send = connect(addr).await;
    let t = Instant::now();
    let r =
      post(&mut send, "/v1/collections/up_bulk/documents:bulk", ndjson).await;
    let v: serde_json::Value = serde_json::from_slice(&r).unwrap();
    assert_eq!(v["count"], up_docs);
    (t.elapsed(), ())
  });
  println!(
    "| HTTP NDJSON bulk (batches of 1000) | 1 | {:.0} | - | - |",
    up_docs as f64 / w.as_secs_f64()
  );
  for name in ["up_embedded", "up_http", "up_bulk"] {
    assert_eq!(
      db.collection(name).unwrap().stats().unwrap().live_docs,
      up_docs as u64
    );
  }

  stop_tx.send(()).unwrap();
  server_rt.block_on(server).unwrap().unwrap();
  drop(client_rt);
  drop(server_rt);
  drop(c);
  drop(ce);
  muninn_server::close(db, Duration::from_secs(30)).unwrap();
  let _ = std::fs::remove_dir_all(&dir);
}
