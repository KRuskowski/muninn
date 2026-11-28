//! Many clients upserting and searching at once over real TCP
//! connections to a server on 127.0.0.1.

mod common;

use std::net::SocketAddr;
use std::sync::Arc;

use common::{TestDir, http};
use muninn::{Database, Options};
use muninn_server::{AppState, ServerConfig};
use serde_json::json;

const CLIENTS: u64 = 16;
const BATCHES: u64 = 25;
const BATCH: u64 = 8;
const DIM: usize = 8;

/// A deterministic vector for document `id`.
fn vector(id: u64) -> Vec<f32> {
  (0..DIM)
    .map(|d| ((id * 31 + d as u64 * 17) % 97) as f32 / 97.0 + 0.01)
    .collect()
}

#[test]
fn concurrent_clients_upsert_and_search() {
  let dir = TestDir::new("concurrency");
  // Small memtables, so flushes and background merges run during the
  // test.
  let opts = Options::default()
    .memtable_max_docs(500)
    .sync_every(std::time::Duration::from_millis(5));
  let db = Arc::new(Database::open(dir.path(), opts).unwrap());
  let state = AppState::new(db.clone(), ServerConfig::default());
  let rt = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(2)
    .enable_all()
    .build()
    .unwrap();
  let listener = rt
    .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
    .unwrap();
  let addr: SocketAddr = listener.local_addr().unwrap();
  let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
  let server = rt.spawn(muninn_server::serve(listener, state, async {
    let _ = stop_rx.await;
  }));

  let r = http(
    addr,
    "PUT",
    "/v1/collections/c",
    Some(
      &json!({
        "vector": {"dim": DIM, "metric": "l2"},
        "text_fields": ["t"],
        "filterable": {"client": "integer"}
      })
      .to_string(),
    ),
  );
  assert_eq!(r.0, 201, "{}", r.1);

  std::thread::scope(|s| {
    for client in 0..CLIENTS {
      s.spawn(move || {
        for b in 0..BATCHES {
          let docs: Vec<_> = (0..BATCH)
            .map(|i| {
              let id = (client * BATCHES + b) * BATCH + i;
              json!({
                "id": id,
                "vector": vector(id),
                "text": {"t": format!("client{client} batch{b} shared")},
                "payload": {"client": client}
              })
            })
            .collect();
          let body = json!({ "documents": docs }).to_string();
          let r =
            http(addr, "POST", "/v1/collections/c/documents", Some(&body));
          assert_eq!(r.0, 200, "{}", r.1);
          assert_eq!(r.1["count"], BATCH);

          // Everything this client wrote so far is visible: its own
          // latest document is its own nearest neighbour.
          let last = (client * BATCHES + b) * BATCH + BATCH - 1;
          let q = json!({
            "vector": vector(last), "k": 1,
            "filter": {"eq": {"client": client}}
          });
          let r = http(
            addr,
            "POST",
            "/v1/collections/c/search",
            Some(&q.to_string()),
          );
          assert_eq!(r.0, 200, "{}", r.1);
          assert_eq!(r.1["hits"][0]["vector_score"], 0.0, "{}", r.1);

          let q = json!({
            "vector": vector(last), "text": format!("client{client}"),
            "k": 5, "with_payload": true
          });
          let r = http(
            addr,
            "POST",
            "/v1/collections/c/search",
            Some(&q.to_string()),
          );
          assert_eq!(r.0, 200, "{}", r.1);
          let hits = r.1["hits"].as_array().unwrap();
          assert!(!hits.is_empty() && hits.len() <= 5);
          assert!(hits.iter().all(|h| h["payload"]["client"].is_u64()));
          let r = http(
            addr,
            "GET",
            &format!("/v1/collections/c/documents/{last}"),
            None,
          );
          assert_eq!(r.0, 200, "{}", r.1);
        }
      });
    }
  });

  let total = CLIENTS * BATCHES * BATCH;
  let r = http(addr, "GET", "/v1/collections/c", None);
  assert_eq!(r.1["stats"]["live_docs"], total, "{}", r.1);
  for client in 0..CLIENTS {
    let q = json!({
      "text": format!("client{client}"), "k": 1000,
      "filter": {"eq": {"client": client}}
    });
    let r = http(
      addr,
      "POST",
      "/v1/collections/c/search",
      Some(&q.to_string()),
    );
    assert_eq!(
      r.1["hits"].as_array().unwrap().len() as u64,
      BATCHES * BATCH
    );
  }

  stop_tx.send(()).unwrap();
  rt.block_on(server).unwrap().unwrap();
  drop(rt);
  muninn_server::close(db, std::time::Duration::from_secs(10)).unwrap();
  // Closed for real: the directory can be opened again.
  let db = Database::open(dir.path(), Options::default()).unwrap();
  let s = db.collection("c").unwrap().stats().unwrap();
  assert_eq!(s.live_docs, total);
  assert_eq!(s.memtable_docs, 0, "close() flushes");
}
