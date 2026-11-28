//! Every endpoint of the HTTP API, in process (tower `oneshot`).

mod common;

use common::{TestDir, app, app_with, assert_error, call};
use muninn::{Durability, Failpoint, Options};
use muninn_server::ServerConfig;
use serde_json::{Value, json};

const SCHEMA: &str = r#"{
  "vector": {"dim": 3, "metric": "cosine", "index": "flat"},
  "text_fields": ["title", "body"],
  "filterable": {"lang": "keyword", "year": "integer", "tags": "keyword"}
}"#;

fn docs() -> Value {
  json!({"documents": [
    {"id": 1, "vector": [1.0, 0.0, 0.0],
     "text": {"title": "rust database engine", "body": "fast and small"},
     "payload": {"lang": "en", "year": 2021, "tags": ["db", "rust"]}},
    {"id": 2, "vector": [0.9, 0.1, 0.0],
     "text": {"title": "a cooking book", "body": "soups and bread"},
     "payload": {"lang": "en", "year": 2015, "tags": ["food"]}},
    {"id": "three", "vector": [0.0, 1.0, 0.0],
     "text": {"title": "eine Datenbank", "body": "database in german"},
     "payload": {"lang": "de", "year": 2023}},
    {"id": "12", "vector": [0.0, 0.0, 1.0],
     "text": {"title": "string id that looks numeric"},
     "payload": {"lang": "fr"}}
  ]})
}

async fn setup(dir: &TestDir) -> axum::Router {
  let a = app(dir.path(), ServerConfig::default());
  let r = call(&a, "PUT", "/v1/collections/docs", SCHEMA).await;
  assert_eq!(r.0, 201, "{}", r.1);
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/documents",
    docs().to_string(),
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["count"], 4);
  a
}

fn ids(v: &Value) -> Vec<Value> {
  v["hits"]
    .as_array()
    .unwrap()
    .iter()
    .map(|h| h["id"].clone())
    .collect()
}

#[tokio::test]
async fn health_unknown_routes_and_methods() {
  let dir = TestDir::new("health");
  let a = app(dir.path(), ServerConfig::default());
  let r = call(&a, "GET", "/v1/health", "").await;
  assert_eq!(r.0, 200);
  assert_eq!(r.1["status"], "ok");
  assert!(r.1["version"].is_string());
  assert_error(&call(&a, "GET", "/v1/nope", "").await, 404, "not_found");
  assert_error(&call(&a, "GET", "/", "").await, 404, "not_found");
  assert_error(
    &call(&a, "POST", "/v1/health", "").await,
    405,
    "method_not_allowed",
  );
  assert_error(
    &call(&a, "PATCH", "/v1/collections/x", "").await,
    405,
    "method_not_allowed",
  );
  let r = call(&a, "GET", "/v1/collections", "").await;
  assert_eq!(r.1, json!({"collections": [], "unavailable": {}}));
}

#[tokio::test]
async fn collection_lifecycle() {
  let dir = TestDir::new("lifecycle");
  let a = app(dir.path(), ServerConfig::default());
  let r = call(&a, "PUT", "/v1/collections/docs", SCHEMA).await;
  assert_eq!(r.0, 201, "{}", r.1);
  assert_eq!(r.1["name"], "docs");
  assert_eq!(
    r.1["schema"]["vector"],
    json!({"dim": 3, "metric": "cosine", "index": {"kind": "flat"},
           "quantization": "none"})
  );
  assert_eq!(r.1["schema"]["text_fields"], json!(["title", "body"]));
  assert_eq!(r.1["schema"]["filterable"].as_array().unwrap().len(), 3);
  assert_eq!(r.1["stats"]["live_docs"], 0);

  let m = assert_error(
    &call(&a, "PUT", "/v1/collections/docs", SCHEMA).await,
    409,
    "already_exists",
  );
  assert!(m.contains("docs"), "{m}");

  // A text-only collection with an explicit HNSW vector-less schema.
  let r = call(
    &a,
    "PUT",
    "/v1/collections/notes",
    r#"{"text_fields": ["body"], "analyzer": "english"}"#,
  )
  .await;
  assert_eq!(r.0, 201, "{}", r.1);
  assert_eq!(r.1["schema"]["vector"], Value::Null);
  assert_eq!(r.1["schema"]["analyzer"]["stemmer"], "porter_english");

  let r = call(&a, "GET", "/v1/collections", "").await;
  assert_eq!(
    r.1,
    json!({"collections": ["docs", "notes"], "unavailable": {}})
  );
  let r = call(&a, "GET", "/v1/collections/notes", "").await;
  assert_eq!(r.0, 200);
  assert_eq!(r.1["name"], "notes");

  // Schema and name validation.
  let m = assert_error(
    &call(&a, "PUT", "/v1/collections/x", r#"{"vector": {"dim": 0}}"#).await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("dimension"), "{m}");
  let m = assert_error(
    &call(&a, "PUT", "/v1/collections/x", r#"{"vectr": {}}"#).await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("unknown key \"vectr\""), "{m}");
  let m = assert_error(
    &call(
      &a,
      "PUT",
      "/v1/collections/x",
      r#"{"vector": {"dim": 3, "metric": "manhattan"}}"#,
    )
    .await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("schema.vector.metric"), "{m}");
  let m = assert_error(
    &call(&a, "PUT", "/v1/collections/bad.name", SCHEMA).await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("collection name"), "{m}");
  assert_error(
    &call(&a, "PUT", "/v1/collections/x", "{not json").await,
    400,
    "invalid_argument",
  );

  let r = call(&a, "DELETE", "/v1/collections/notes", "").await;
  assert_eq!(r.0, 200);
  assert_eq!(r.1, json!({"name": "notes", "dropped": true}));
  assert_error(
    &call(&a, "GET", "/v1/collections/notes", "").await,
    404,
    "not_found",
  );
  assert_error(
    &call(&a, "DELETE", "/v1/collections/notes", "").await,
    404,
    "not_found",
  );
  let r = call(&a, "GET", "/v1/collections", "").await;
  assert_eq!(r.1, json!({"collections": ["docs"], "unavailable": {}}));
}

#[tokio::test]
async fn upsert_then_search_get_and_delete() {
  let dir = TestDir::new("crud");
  let a = setup(&dir).await;

  // Vector search: nearest first, with both scores reported.
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"vector": [1, 0, 0], "k": 2}"#,
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(ids(&r.1), vec![json!(1), json!(2)]);
  let h = &r.1["hits"][0];
  assert!((h["score"].as_f64().unwrap() - 1.0).abs() < 1e-5);
  assert_eq!(h["vector_score"], h["score"]);
  assert_eq!(h["text_score"], Value::Null);
  assert!(h.get("payload").is_none(), "no payload unless asked: {h}");
  assert!(r.1["took_us"].is_u64());

  // Text search over all fields, then one field.
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"text": "database", "with_payload": true}"#,
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  let got = ids(&r.1);
  assert_eq!(got.len(), 2, "{}", r.1);
  assert!(got.contains(&json!(1)) && got.contains(&json!("three")));
  assert!(r.1["hits"][0]["payload"]["lang"].is_string());
  assert_eq!(r.1["hits"][0]["vector_score"], Value::Null);
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"text": "database", "fields": ["title"]}"#,
  )
  .await;
  assert_eq!(ids(&r.1), vec![json!(1)]);

  // Get by numeric id, string id, and a numeric-looking string id.
  let r = call(&a, "GET", "/v1/collections/docs/documents/1", "").await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["id"], 1);
  assert_eq!(r.1["text"]["title"], "rust database engine");
  assert_eq!(r.1["payload"]["year"], 2021);
  assert_eq!(r.1["vector"].as_array().unwrap().len(), 3);
  let r = call(&a, "GET", "/v1/collections/docs/documents/three", "").await;
  assert_eq!(r.1["id"], "three");
  assert_error(
    &call(&a, "GET", "/v1/collections/docs/documents/12", "").await,
    404,
    "not_found",
  );
  let r = call(
    &a,
    "GET",
    "/v1/collections/docs/documents/12?id_type=string",
    "",
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["id"], "12");
  assert_error(
    &call(
      &a,
      "GET",
      "/v1/collections/docs/documents/x?id_type=int",
      "",
    )
    .await,
    400,
    "invalid_argument",
  );

  // Upsert replaces.
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/documents",
    r#"{"documents": [{"id": 2, "vector": [0, 1, 0],
        "text": {"title": "replaced"}}]}"#,
  )
  .await;
  assert_eq!(r.0, 200);
  let r = call(&a, "GET", "/v1/collections/docs/documents/2", "").await;
  assert_eq!(r.1["text"], json!({"title": "replaced"}));
  assert!(r.1.get("payload").is_none());

  // Delete; unknown ids are ignored.
  let r = call(
    &a,
    "DELETE",
    "/v1/collections/docs/documents",
    r#"{"ids": [1, "nope"]}"#,
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["count"], 2);
  let m = assert_error(
    &call(&a, "GET", "/v1/collections/docs/documents/1", "").await,
    404,
    "not_found",
  );
  assert!(m.contains("document 1"), "{m}");
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"vector": [1, 0, 0], "k": 10}"#,
  )
  .await;
  assert!(!ids(&r.1).contains(&json!(1)));
  assert_eq!(ids(&r.1).len(), 3);
}

#[tokio::test]
async fn hybrid_search_and_fusion() {
  let dir = TestDir::new("hybrid");
  let a = setup(&dir).await;
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"vector": [0, 1, 0], "text": "database", "k": 4}"#,
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  // "three" is best on both sides, so first under RRF.
  assert_eq!(r.1["hits"][0]["id"], "three");
  let h = &r.1["hits"][0];
  assert!(
    h["vector_score"].is_f64() && h["text_score"].is_f64(),
    "{h}"
  );
  // Rank 1 on the vector side: 1 / (60 + 1), plus its text rank's share.
  let s = h["score"].as_f64().unwrap();
  assert!(s > 1.0 / 61.0 && s <= 2.0 / 61.0 + 1e-6, "{h}");

  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"vector": [1, 0, 0], "text": "database", "k": 4,
        "fusion": {"weighted": {"vector": 1.0, "text": 0.0}}}"#,
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["hits"][0]["id"], 1);
  assert!((r.1["hits"][0]["score"].as_f64().unwrap() - 1.0).abs() < 1e-6);

  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"vector": [1, 0, 0], "text": "database", "fusion": {"rrf": {"k": 1}}}"#,
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["hits"][0]["id"], 1);
  // 1 / (1 + 1) from the vector side plus at least 1 / (1 + 2) from text.
  let s = r.1["hits"][0]["score"].as_f64().unwrap();
  assert!(s > 0.8 && s <= 1.0 + 1e-6, "{}", r.1);

  let m = assert_error(
    &call(
      &a,
      "POST",
      "/v1/collections/docs/search",
      r#"{"vector": [1, 0, 0], "text": "x", "fusion": {"max": {}}}"#,
    )
    .await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("fusion"), "{m}");
}

#[tokio::test]
async fn filters_in_json() {
  let dir = TestDir::new("filters");
  let a = setup(&dir).await;
  let search = |body: &'static str| {
    let a = a.clone();
    async move { call(&a, "POST", "/v1/collections/docs/search", body).await }
  };
  let sorted = |v: &Value| {
    let mut x: Vec<String> = ids(v).iter().map(|i| i.to_string()).collect();
    x.sort();
    x
  };
  let r = search(
    r#"{"vector": [1, 0, 0], "k": 10, "filter": {"eq": {"lang": "en"}}}"#,
  )
  .await;
  assert_eq!(sorted(&r.1), ["1", "2"], "{}", r.1);
  let r = search(
    r#"{"vector": [1, 0, 0], "k": 10,
        "filter": {"range": {"year": {"gte": 2020}}}}"#,
  )
  .await;
  assert_eq!(sorted(&r.1), ["\"three\"", "1"]);
  let r = search(
    r#"{"text": "database", "k": 10,
        "filter": {"and": [{"in": {"lang": ["en", "fr"]}},
                           {"not": {"eq": {"tags": "food"}}}]}}"#,
  )
  .await;
  assert_eq!(sorted(&r.1), ["1"]);
  let r = search(
    r#"{"vector": [0, 0, 1], "k": 10,
        "filter": {"or": [{"eq": {"tags": "rust"}},
                          {"not": {"exists": "year"}}]}}"#,
  )
  .await;
  assert_eq!(sorted(&r.1), ["\"12\"", "1"]);
  let r = search(
    r#"{"vector": [0, 1, 0], "text": "database", "k": 10,
        "filter": {"eq": {"lang": "de"}}}"#,
  )
  .await;
  assert_eq!(sorted(&r.1), ["\"three\""]);
  // Restricting by id, together with a filter.
  let r = search(
    r#"{"vector": [1, 0, 0], "k": 10, "ids": [2, "three", 99],
        "filter": {"eq": {"lang": "en"}}}"#,
  )
  .await;
  assert_eq!(sorted(&r.1), ["2"]);

  let m = assert_error(
    &search(r#"{"text": "x", "filter": {"range": {"year": {"gte": "a"}}}}"#)
      .await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("filter at $.range.year"), "{m}");
  let m = assert_error(
    &search(r#"{"text": "x", "filter": {"eq": {"author": "me"}}}"#).await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("author"), "{m}");
  assert_error(
    &search(r#"{"text": "x", "filter": {"near": {}}}"#).await,
    400,
    "invalid_argument",
  );
}

#[tokio::test]
async fn invalid_requests_are_rejected_clearly() {
  let dir = TestDir::new("invalid");
  let a = setup(&dir).await;
  let up = "/v1/collections/docs/documents";
  let cases: &[(&str, &str, &str, &str)] = &[
    ("POST", up, "{\"documents\": [", "request body"),
    ("POST", up, "[]", "request body"),
    ("POST", up, r#"{"docs": []}"#, "unknown field `docs`"),
    (
      "POST",
      up,
      r#"{"documents": [{"id": 9, "vector": [1, 0]}]}"#,
      "vector has 2 dimensions, collection expects 3",
    ),
    (
      "POST",
      up,
      r#"{"documents": [{"id": 9}]}"#,
      "has no vector, but the collection requires one",
    ),
    (
      "POST",
      up,
      r#"{"documents": [{"id": 9, "vector": [1, 0, 0],
          "text": {"summary": "x"}}]}"#,
      "summary",
    ),
    (
      "POST",
      up,
      r#"{"documents": [{"id": -4, "vector": [1, 0, 0]}]}"#,
      "negative",
    ),
    (
      "POST",
      up,
      r#"{"documents": [{"id": [1], "vector": [1, 0, 0]}]}"#,
      "document id",
    ),
    (
      "POST",
      up,
      r#"{"documents": [{"id": 1, "vector": [1, "a", 0]}]}"#,
      "vector: expected a number, got \"a\"",
    ),
    ("DELETE", up, r#"{"ids": [1.5]}"#, "document id"),
    ("DELETE", up, "{}", "missing field `ids`"),
    (
      "POST",
      "/v1/collections/docs/search",
      "{}",
      "needs \"vector\", \"text\" or both",
    ),
    (
      "POST",
      "/v1/collections/docs/search",
      r#"{"text": "a", "k": 0}"#,
      "k must be between 1 and 10000",
    ),
    (
      "POST",
      "/v1/collections/docs/search",
      r#"{"vector": [1, 2]}"#,
      "vector has 2 dimensions",
    ),
    (
      "POST",
      "/v1/collections/docs/search",
      r#"{"text": "a", "limit": 3}"#,
      "unknown field `limit`",
    ),
    (
      "POST",
      "/v1/collections/docs/search",
      r#"{"text": "a", "fields": ["summary"]}"#,
      "summary",
    ),
  ];
  for &(method, uri, body, want) in cases {
    let r = call(&a, method, uri, body).await;
    let m = assert_error(&r, 400, "invalid_argument");
    assert!(m.contains(want), "{body}: {m:?} lacks {want:?}");
  }
  // A rejected batch stores nothing.
  let r = call(
    &a,
    "POST",
    up,
    r#"{"documents": [{"id": 50, "vector": [1, 0, 0]},
                      {"id": 51, "vector": [1, 0]}]}"#,
  )
  .await;
  assert_error(&r, 400, "invalid_argument");
  assert_error(
    &call(&a, "GET", "/v1/collections/docs/documents/50", "").await,
    404,
    "not_found",
  );
}

#[tokio::test]
async fn unknown_collection_is_not_found_everywhere() {
  let dir = TestDir::new("unknown");
  let a = app(dir.path(), ServerConfig::default());
  let body = r#"{"documents": []}"#;
  for (method, uri, body) in [
    ("GET", "/v1/collections/ghost", ""),
    ("DELETE", "/v1/collections/ghost", ""),
    ("POST", "/v1/collections/ghost/documents", body),
    (
      "DELETE",
      "/v1/collections/ghost/documents",
      r#"{"ids": [1]}"#,
    ),
    (
      "POST",
      "/v1/collections/ghost/documents:bulk",
      "{\"id\": 1}\n",
    ),
    ("GET", "/v1/collections/ghost/documents/1", ""),
    ("POST", "/v1/collections/ghost/search", r#"{"text": "a"}"#),
    ("POST", "/v1/collections/ghost/flush", ""),
    ("POST", "/v1/collections/ghost/compact", ""),
  ] {
    let m = assert_error(&call(&a, method, uri, body).await, 404, "not_found");
    assert_eq!(m, "collection \"ghost\" not found", "{method} {uri}");
  }
}

#[tokio::test]
async fn oversized_bodies_are_rejected() {
  let dir = TestDir::new("oversized");
  let a = app(
    dir.path(),
    ServerConfig {
      body_limit: 1024,
      ..ServerConfig::default()
    },
  );
  let r =
    call(&a, "PUT", "/v1/collections/t", r#"{"text_fields": ["b"]}"#).await;
  assert_eq!(r.0, 201);
  let big = json!({"documents": [
    {"id": 1, "text": {"b": "x".repeat(2000)}}
  ]})
  .to_string();
  let m = assert_error(
    &call(&a, "POST", "/v1/collections/t/documents", big).await,
    413,
    "payload_too_large",
  );
  assert!(m.contains("1024"), "{m}");
  // Bulk bodies may be larger in total, but not per line.
  let line = format!(
    "{{\"id\": 1, \"text\": {{\"b\": \"{}\"}}}}\n",
    "x".repeat(600)
  );
  let r = call(
    &a,
    "POST",
    "/v1/collections/t/documents:bulk",
    line.repeat(4),
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["count"], 4);
  let long = format!(
    "{{\"id\": 2, \"text\": {{\"b\": \"{}\"}}}}\n",
    "y".repeat(3000)
  );
  let m = assert_error(
    &call(&a, "POST", "/v1/collections/t/documents:bulk", long).await,
    413,
    "payload_too_large",
  );
  assert!(m.contains("line 1 exceeds"), "{m}");
  // Small requests still work.
  let r =
    call(&a, "POST", "/v1/collections/t/search", r#"{"text": "x"}"#).await;
  assert_eq!(r.0, 200);
}

#[tokio::test]
async fn flush_compact_and_stats() {
  let dir = TestDir::new("compact");
  let a = setup(&dir).await;
  let r = call(
    &a,
    "PUT",
    "/v1/collections/other",
    r#"{"text_fields": ["t"]}"#,
  )
  .await;
  assert_eq!(r.0, 201);

  let r = call(&a, "POST", "/v1/collections/docs/flush", "").await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert!(r.1["took_us"].is_u64());
  let r = call(&a, "GET", "/v1/collections/docs", "").await;
  let s = &r.1["stats"];
  assert_eq!(s["memtable_docs"], 0, "{s}");
  assert_eq!(s["segments"].as_array().unwrap().len(), 1);
  assert_eq!(s["live_docs"], 4);

  call(
    &a,
    "DELETE",
    "/v1/collections/docs/documents",
    r#"{"ids": [1, 2]}"#,
  )
  .await;
  call(
    &a,
    "POST",
    "/v1/collections/docs/documents",
    r#"{"documents": [{"id": 5, "vector": [1, 1, 0]}]}"#,
  )
  .await;
  call(&a, "POST", "/v1/collections/docs/flush", "").await;
  let r = call(&a, "GET", "/v1/stats", "").await;
  assert_eq!(r.0, 200);
  let s = &r.1["collections"]["docs"];
  assert_eq!(s["live_docs"], 3, "{s}");
  assert_eq!(s["deleted_docs"], 2, "{s}");
  assert_eq!(r.1["collections"]["other"]["live_docs"], 0);

  let r = call(&a, "POST", "/v1/collections/docs/compact", "").await;
  assert_eq!(r.0, 200, "{}", r.1);
  let r = call(&a, "GET", "/v1/collections/docs", "").await;
  let s = &r.1["stats"];
  assert_eq!(s["deleted_docs"], 0, "{s}");
  assert_eq!(s["live_docs"], 3);
  assert_eq!(s["segments"].as_array().unwrap().len(), 1);
  let seg = &s["segments"][0];
  assert_eq!(seg["docs"], 3);
  assert_eq!(seg["vector_index"], "flat");

  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"vector": [1, 1, 0], "k": 10}"#,
  )
  .await;
  assert_eq!(r.1["hits"][0]["id"], 5, "{}", r.1);
  assert_eq!(ids(&r.1).len(), 3);
}

#[tokio::test]
async fn data_survives_a_restart() {
  let dir = TestDir::new("restart");
  {
    let a = setup(&dir).await;
    // Half flushed to a segment, half only in the write-ahead log.
    call(&a, "POST", "/v1/collections/docs/flush", "").await;
    let r = call(
      &a,
      "POST",
      "/v1/collections/docs/documents",
      r#"{"documents": [{"id": 7, "vector": [0, 1, 1],
          "text": {"body": "written after the flush"}}]}"#,
    )
    .await;
    assert_eq!(r.0, 200);
    call(
      &a,
      "DELETE",
      "/v1/collections/docs/documents",
      r#"{"ids": ["three"]}"#,
    )
    .await;
  }
  // The router (and with it the database) is gone: open a new server.
  let a = app(dir.path(), ServerConfig::default());
  let r = call(&a, "GET", "/v1/collections", "").await;
  assert_eq!(r.1, json!({"collections": ["docs"], "unavailable": {}}));
  let r = call(&a, "GET", "/v1/collections/docs", "").await;
  assert_eq!(r.1["stats"]["live_docs"], 4, "{}", r.1);
  assert_eq!(r.1["schema"]["text_fields"], json!(["title", "body"]));
  let r = call(&a, "GET", "/v1/collections/docs/documents/7", "").await;
  assert_eq!(r.0, 200);
  assert_eq!(r.1["text"]["body"], "written after the flush");
  assert_error(
    &call(&a, "GET", "/v1/collections/docs/documents/three", "").await,
    404,
    "not_found",
  );
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"text": "flush", "filter": {"exists": "lang"}}"#,
  )
  .await;
  assert_eq!(r.0, 200);
  assert_eq!(ids(&r.1), Vec::<Value>::new());
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/search",
    r#"{"text": "flush"}"#,
  )
  .await;
  assert_eq!(ids(&r.1), vec![json!(7)]);
}

#[tokio::test]
async fn bulk_ndjson_import() {
  let dir = TestDir::new("bulk");
  let a = app(dir.path(), ServerConfig::default());
  call(
    &a,
    "PUT",
    "/v1/collections/b",
    r#"{"vector": {"dim": 2, "index": "flat"}, "text_fields": ["t"],
        "filterable": {"n": "integer"}}"#,
  )
  .await;
  let mut body = String::new();
  for i in 0..2500u64 {
    let x = i as f32 / 2500.0;
    body += &json!({
      "id": i, "vector": [x, 1.0 - x],
      "text": {"t": format!("word{} common", i % 10)},
      "payload": {"n": i}
    })
    .to_string();
    body += if i % 3 == 0 { "\r\n\n" } else { "\n" };
  }
  body.pop(); // last line without a newline
  let r = call(
    &a,
    "POST",
    "/v1/collections/b/documents:bulk?batch=1000",
    body,
  )
  .await;
  assert_eq!(r.0, 200, "{}", r.1);
  assert_eq!(r.1["count"], 2500);
  assert_eq!(r.1["batches"], 3);
  let r = call(&a, "GET", "/v1/collections/b", "").await;
  assert_eq!(r.1["stats"]["live_docs"], 2500);
  let r = call(
    &a,
    "POST",
    "/v1/collections/b/search",
    r#"{"text": "word7", "k": 1000, "filter": {"range": {"n": {"lt": 100}}}}"#,
  )
  .await;
  assert_eq!(ids(&r.1).len(), 10, "{}", r.1);

  // An error names the line and how much was stored before it.
  let body = "{\"id\": \"a\", \"vector\": [1, 0]}\n\
              {\"id\": \"b\", \"vector\": [1, 0]}\n\
              {\"id\": \"c\", \"vector\": [1]}\n\
              {\"id\": \"d\", \"vector\": [1, 0]}\n";
  let m = assert_error(
    &call(&a, "POST", "/v1/collections/b/documents:bulk?batch=1", body).await,
    400,
    "invalid_argument",
  );
  assert!(m.starts_with("line 3: "), "{m}");
  assert!(m.ends_with("the 2 documents before it were stored"), "{m}");
  let r = call(&a, "GET", "/v1/collections/b/documents/b", "").await;
  assert_eq!(r.0, 200);
  assert_error(
    &call(&a, "GET", "/v1/collections/b/documents/d", "").await,
    404,
    "not_found",
  );
  let m = assert_error(
    &call(
      &a,
      "POST",
      "/v1/collections/b/documents:bulk",
      "{\"id\": 1,\n",
    )
    .await,
    400,
    "invalid_argument",
  );
  assert!(m.starts_with("line 1: "), "{m}");
  for q in ["batch=0", "batch=x", "batch=100001"] {
    let m = assert_error(
      &call(
        &a,
        "POST",
        &format!("/v1/collections/b/documents:bulk?{q}"),
        "",
      )
      .await,
      400,
      "invalid_argument",
    );
    assert!(m.contains("batch must be"), "{m}");
  }
  // An empty body stores nothing and succeeds.
  let r = call(&a, "POST", "/v1/collections/b/documents:bulk", "\n\n").await;
  assert_eq!(r.1["count"], 0);
}

/// Nested JSON deeper than the engine stores, and vectors whose distances
/// would overflow, are invalid arguments.
#[tokio::test]
async fn engine_limits_are_invalid_arguments() {
  let dir = TestDir::new("limits");
  let a = setup(&dir).await;
  let mut deep = json!("leaf");
  for _ in 0..110 {
    deep = json!([deep]);
  }
  let body = json!({"documents": [
    {"id": 70, "vector": [1, 0, 0], "payload": {"deep": deep}}
  ]});
  let m = assert_error(
    &call(
      &a,
      "POST",
      "/v1/collections/docs/documents",
      body.to_string(),
    )
    .await,
    400,
    "invalid_argument",
  );
  assert!(m.contains("deep") || m.contains("nest"), "{m}");
  let m = assert_error(
    &call(
      &a,
      "POST",
      "/v1/collections/docs/documents",
      r#"{"documents": [{"id": 71, "vector": [1e30, 1e30, 1e30]}]}"#,
    )
    .await,
    400,
    "invalid_argument",
  );
  assert!(!m.is_empty());
  assert_error(
    &call(
      &a,
      "POST",
      "/v1/collections/docs/search",
      r#"{"vector": [1e30, 1e30, 1e30]}"#,
    )
    .await,
    400,
    "invalid_argument",
  );
}

/// A collection whose files are damaged does not keep the server from
/// starting: it is listed as unavailable, its endpoints report why, and it
/// can be dropped.
#[tokio::test]
async fn damaged_collection_is_reported_and_can_be_dropped() {
  let dir = TestDir::new("damaged");
  {
    let a = setup(&dir).await;
    let r =
      call(&a, "PUT", "/v1/collections/ok", r#"{"text_fields": ["t"]}"#).await;
    assert_eq!(r.0, 201);
  }
  let cdir = dir.path().join("collections").join("docs");
  let mut damaged = 0;
  for e in std::fs::read_dir(&cdir).unwrap() {
    let p = e.unwrap().path();
    let name = p.file_name().unwrap().to_str().unwrap().to_string();
    if name.starts_with("MANIFEST-") {
      let mut bytes = std::fs::read(&p).unwrap();
      let mid = bytes.len() / 2;
      bytes[mid] ^= 0xff;
      std::fs::write(&p, bytes).unwrap();
      damaged += 1;
    }
  }
  assert!(damaged > 0);

  let a = app(dir.path(), ServerConfig::default());
  let r = call(&a, "GET", "/v1/collections", "").await;
  assert_eq!(r.1["collections"], json!(["ok"]));
  let reason = r.1["unavailable"]["docs"].as_str().unwrap().to_string();
  assert!(!reason.is_empty());
  let r = call(&a, "GET", "/v1/stats", "").await;
  assert_eq!(r.1["unavailable"]["docs"], reason.as_str());
  assert!(r.1["collections"]["ok"].is_object());
  for (method, uri, body) in [
    ("GET", "/v1/collections/docs", ""),
    ("POST", "/v1/collections/docs/search", r#"{"text": "x"}"#),
    (
      "POST",
      "/v1/collections/docs/documents",
      r#"{"documents": []}"#,
    ),
    ("GET", "/v1/collections/docs/documents/1", ""),
  ] {
    let m = assert_error(&call(&a, method, uri, body).await, 500, "internal");
    assert!(m.contains("could not be opened"), "{method} {uri}: {m}");
  }
  assert_error(
    &call(&a, "PUT", "/v1/collections/docs", SCHEMA).await,
    409,
    "already_exists",
  );
  let r = call(&a, "DELETE", "/v1/collections/docs", "").await;
  assert_eq!(r.0, 200, "{}", r.1);
  let r = call(&a, "GET", "/v1/collections", "").await;
  assert_eq!(r.1, json!({"collections": ["ok"], "unavailable": {}}));
  let r = call(&a, "PUT", "/v1/collections/docs", SCHEMA).await;
  assert_eq!(r.0, 201, "{}", r.1);
}

/// After a background failure the engine cannot recover from, writes are
/// refused with a 500 that says how to recover.
#[tokio::test]
async fn poisoned_collection_refuses_writes_with_internal_error() {
  let dir = TestDir::new("poisoned");
  let opts = Options {
    durability: Durability::NoSync,
    failpoint: Some(Failpoint::FailAfterCommit),
    auto_compact: false,
    ..Options::default()
  };
  let a = app_with(dir.path(), ServerConfig::default(), opts);
  let r = call(&a, "PUT", "/v1/collections/docs", SCHEMA).await;
  assert_eq!(r.0, 201);
  let r = call(
    &a,
    "POST",
    "/v1/collections/docs/documents",
    docs().to_string(),
  )
  .await;
  assert_eq!(r.0, 200);
  assert_error(
    &call(&a, "POST", "/v1/collections/docs/flush", "").await,
    500,
    "internal",
  );
  let m = assert_error(
    &call(
      &a,
      "POST",
      "/v1/collections/docs/documents",
      r#"{"documents": [{"id": 9, "vector": [1, 0, 0]}]}"#,
    )
    .await,
    500,
    "internal",
  );
  assert!(m.contains("refuses writes"), "{m}");
  drop(a);
  // A restart recovers every acknowledged write.
  let a = app(dir.path(), ServerConfig::default());
  let r = call(&a, "GET", "/v1/collections/docs", "").await;
  assert_eq!(r.1["stats"]["live_docs"], 4, "{}", r.1);
}
