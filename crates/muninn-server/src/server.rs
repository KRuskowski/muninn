//! The router and its handlers.
//!
//! Every engine call (and the JSON parsing and encoding around it) runs
//! as blocking work (see `blocking`), so the async runtime keeps moving
//! bytes for other connections and is never held up by a search, an
//! fsync or a large body.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::{FromRequestParts, Path, RawQuery, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::request::Parts;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use muninn::{Collection, Database};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use tokio::sync::Semaphore;

use crate::error::{ApiError, ErrorCode};
use crate::guard::{self, Limits};
use crate::model::{self, DocOut, SearchRequest, SearchResponse};

/// Server settings. The defaults suit a server on 127.0.0.1.
#[derive(Clone, Debug)]
pub struct ServerConfig {
  /// Largest accepted request body in bytes. Default 16 MiB.
  pub body_limit: usize,
  /// Longest line of an NDJSON bulk body. Default 1 MiB.
  pub bulk_line_limit: usize,
  /// Documents per write batch of the NDJSON bulk endpoint, unless the
  /// request sets `?batch=`. Default 1000.
  pub bulk_batch: usize,
  /// Request bodies held in memory at once, in bytes, over all requests
  /// (a parsed body takes several times its size). Requests beyond it get
  /// 503. Default 64 MiB.
  pub max_body_bytes_in_flight: usize,
  /// Open connections; further ones are closed at once. Default 512.
  pub max_connections: usize,
  /// Requests in flight; further ones get 503. Default 256.
  pub max_requests: usize,
  /// NDJSON bulk requests in flight; further ones get 503. Each reserves
  /// about 18 MiB of the in-flight body budget. Default 2.
  pub max_bulk: usize,
  /// Time for a client to send a request's headers, and the longest a
  /// keep-alive connection may sit idle. Default 10 s.
  pub header_timeout: Duration,
  /// Time for a client to send a whole (non-bulk) request body. Default
  /// 60 s.
  pub body_timeout: Duration,
  /// Longest pause between two parts of a bulk body. Default 30 s.
  pub bulk_idle_timeout: Duration,
  /// After a shutdown signal, how long open connections may take to
  /// finish before they are closed. Default 30 s.
  pub shutdown_timeout: Duration,
  /// Host names (lowercase, IPv6 without brackets) accepted in the `Host`
  /// and `Origin` headers; `"*"` accepts any. Default `localhost`,
  /// `127.0.0.1` and `::1`.
  pub allowed_hosts: Vec<String>,
  /// When set, every request except `GET /v1/health` must carry
  /// `Authorization: Bearer <api_key>`.
  pub api_key: Option<String>,
}

impl Default for ServerConfig {
  fn default() -> Self {
    Self {
      body_limit: 16 << 20,
      bulk_line_limit: 1 << 20,
      bulk_batch: 1000,
      max_body_bytes_in_flight: 64 << 20,
      max_connections: 512,
      max_requests: 256,
      max_bulk: 2,
      header_timeout: Duration::from_secs(10),
      body_timeout: Duration::from_secs(60),
      bulk_idle_timeout: Duration::from_secs(30),
      shutdown_timeout: Duration::from_secs(30),
      allowed_hosts: ["localhost", "127.0.0.1", "::1"]
        .map(String::from)
        .to_vec(),
      api_key: None,
    }
  }
}

impl ServerConfig {
  /// Also accepts `host` (a name or address, optionally with a port) in
  /// `Host` and `Origin` headers.
  pub fn allow_host(&mut self, host: &str) {
    let h = if host.trim() == "*" {
      "*".to_string()
    } else {
      guard::host_of(host)
    };
    if !h.is_empty() && !self.allowed_hosts.contains(&h) {
      self.allowed_hosts.push(h);
    }
  }
}

/// Largest `?batch=` of the bulk endpoint.
pub const MAX_BULK_BATCH: usize = 100_000;

/// Collection handles by name, so that requests find their collection
/// without taking the database's lock (which creating a collection holds
/// during disk I/O).
#[derive(Default)]
struct Handles {
  map: RwLock<HashMap<String, Collection>>,
  /// Bumped by every drop, so a lookup racing a drop does not cache the
  /// dropped handle.
  generation: AtomicU64,
}

/// Shared state of all handlers: the database, the settings, collection
/// handles and limits. Cheap to clone.
#[derive(Clone)]
pub struct AppState {
  pub(crate) db: Arc<Database>,
  pub(crate) cfg: Arc<ServerConfig>,
  pub(crate) limits: Arc<Limits>,
  handles: Arc<Handles>,
}

impl AppState {
  /// State serving `db` with `cfg`. Takes a handle to every collection.
  pub fn new(db: Arc<Database>, cfg: ServerConfig) -> Self {
    let handles = Handles::default();
    if let Ok(names) = db.list_collections() {
      let mut map = handles.map.write().unwrap_or_else(|e| e.into_inner());
      for n in names {
        if let Ok(c) = db.collection(&n) {
          map.insert(n, c);
        }
      }
    }
    let limits = Limits {
      requests: Arc::new(Semaphore::new(cfg.max_requests.max(1))),
      body_bytes: Arc::new(Semaphore::new(
        cfg
          .max_body_bytes_in_flight
          .clamp(1, Semaphore::MAX_PERMITS),
      )),
      bulk: Arc::new(Semaphore::new(cfg.max_bulk.max(1))),
    };
    Self {
      db,
      cfg: Arc::new(cfg),
      limits: Arc::new(limits),
      handles: Arc::new(handles),
    }
  }

  /// The database.
  pub fn database(&self) -> &Arc<Database> {
    &self.db
  }

  /// The settings.
  pub fn config(&self) -> &ServerConfig {
    &self.cfg
  }

  /// The collection `name`: from the handle cache, or else looked up in
  /// the database as blocking work (which reports unknown and unavailable
  /// collections).
  pub(crate) async fn collection(
    &self,
    name: &str,
  ) -> Result<Collection, ApiError> {
    let cached = self
      .handles
      .map
      .read()
      .unwrap_or_else(|e| e.into_inner())
      .get(name)
      .cloned();
    if let Some(c) = cached {
      return Ok(c);
    }
    let generation = self.handles.generation.load(Ordering::Acquire);
    let db = self.db.clone();
    let n = name.to_string();
    let c = blocking(move || Ok(db.collection(&n)?)).await?;
    let mut map = self.handles.map.write().unwrap_or_else(|e| e.into_inner());
    if self.handles.generation.load(Ordering::Acquire) == generation {
      map.insert(name.to_string(), c.clone());
    }
    Ok(c)
  }

  fn remember(&self, c: &Collection) {
    self
      .handles
      .map
      .write()
      .unwrap_or_else(|e| e.into_inner())
      .insert(c.name().to_string(), c.clone());
  }

  fn forget(&self, name: &str) {
    self.handles.generation.fetch_add(1, Ordering::AcqRel);
    self
      .handles
      .map
      .write()
      .unwrap_or_else(|e| e.into_inner())
      .remove(name);
  }
}

/// The HTTP API under `/v1` (see `docs/API.md`), with its guards: Host
/// and Origin check, API key, and the cap on requests in flight.
pub fn router(state: AppState) -> Router {
  Router::new()
    .route("/v1/health", get(health))
    .route("/v1/stats", get(stats))
    .route("/v1/collections", get(list_collections))
    .route(
      "/v1/collections/{name}",
      put(create_collection)
        .get(describe_collection)
        .delete(drop_collection),
    )
    .route(
      "/v1/collections/{name}/documents",
      post(upsert).delete(delete_documents),
    )
    .route(
      "/v1/collections/{name}/documents:bulk",
      post(crate::bulk::bulk),
    )
    .route("/v1/collections/{name}/documents/{id}", get(get_document))
    .route("/v1/collections/{name}/search", post(search))
    .route("/v1/collections/{name}/flush", post(flush))
    .route("/v1/collections/{name}/compact", post(compact))
    .fallback(unknown_route)
    .method_not_allowed_fallback(wrong_method)
    // Outermost last: log, then host check, key, request cap.
    .layer(middleware::from_fn_with_state(
      state.clone(),
      guard::limit_requests,
    ))
    .layer(middleware::from_fn_with_state(
      state.clone(),
      guard::check_api_key,
    ))
    .layer(middleware::from_fn_with_state(
      state.clone(),
      guard::check_host,
    ))
    .layer(middleware::from_fn(log_request))
    .with_state(state)
}

// ---------------------------------------------------------------------
// Extractors and helpers
// ---------------------------------------------------------------------

/// Path parameters, with rejections in the API's error format.
pub(crate) struct Params<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned + Send> FromRequestParts<S>
  for Params<T>
{
  type Rejection = ApiError;

  async fn from_request_parts(
    parts: &mut Parts,
    state: &S,
  ) -> Result<Self, ApiError> {
    Path::<T>::from_request_parts(parts, state)
      .await
      .map(|Path(t)| Params(t))
      .map_err(|e| ApiError::invalid(e.body_text()))
  }
}

/// Runs blocking `f` without stalling other tasks. On a multi-threaded
/// runtime it runs in place: tokio hands this worker's other tasks to
/// another thread first ([`tokio::task::block_in_place`]), which saves a
/// cross-thread wake-up per request (about 25 us per search measured with
/// `examples/server_bench.rs`). Elsewhere it goes to the blocking pool.
pub(crate) async fn blocking<T: Send + 'static>(
  f: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
  use tokio::runtime::{Handle, RuntimeFlavor};
  if Handle::current().runtime_flavor() == RuntimeFlavor::MultiThread {
    return tokio::task::block_in_place(f);
  }
  tokio::task::spawn_blocking(f)
    .await
    .map_err(|e| ApiError::internal(format!("request worker failed: {e}")))?
}

/// A JSON response from already encoded bytes.
pub(crate) fn json_bytes(status: StatusCode, body: Vec<u8>) -> Response {
  (
    status,
    [(CONTENT_TYPE, HeaderValue::from_static("application/json"))],
    body,
  )
    .into_response()
}

/// A JSON response.
pub(crate) fn json_response(
  status: StatusCode,
  v: &impl Serialize,
) -> Result<Response, ApiError> {
  let body = serde_json::to_vec(v)
    .map_err(|e| ApiError::internal(format!("encoding response: {e}")))?;
  Ok(json_bytes(status, body))
}

pub(crate) fn micros(d: Duration) -> u64 {
  u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// The value of `key` in a raw query string (no percent-decoding needed
/// for the parameters the API uses).
pub(crate) fn query_param<'a>(
  query: &'a Option<String>,
  key: &str,
) -> Option<&'a str> {
  query.as_deref()?.split('&').find_map(|kv| {
    let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
    (k == key).then_some(v)
  })
}

async fn log_request(req: Request, next: Next) -> Response {
  if !tracing::enabled!(tracing::Level::DEBUG) {
    return next.run(req).await;
  }
  let start = Instant::now();
  let method = req.method().clone();
  let path = req.uri().path().to_owned();
  let resp = next.run(req).await;
  tracing::debug!(
    "{method} {path} -> {} in {} us",
    resp.status().as_u16(),
    micros(start.elapsed())
  );
  resp
}

async fn unknown_route() -> ApiError {
  ApiError::not_found("no such route; see docs/API.md for the /v1 endpoints")
}

async fn wrong_method() -> ApiError {
  ApiError::new(
    ErrorCode::MethodNotAllowed,
    "method not allowed on this route",
  )
}

// ---------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------

async fn health() -> Result<Response, ApiError> {
  json_response(
    StatusCode::OK,
    &json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")}),
  )
}

fn describe(c: &Collection) -> Result<Value, ApiError> {
  Ok(json!({
    "name": c.name(),
    "schema": model::schema_to_json(c.schema()),
    "stats": model::stats_to_json(&c.stats()?),
  }))
}

/// Statistics of `entries`, skipping collections dropped meanwhile and
/// listing those that fail under `unavailable` instead of failing.
fn collect_stats(
  entries: impl IntoIterator<Item = (String, muninn::Result<Collection>)>,
  unavailable: &mut Map<String, Value>,
) -> Map<String, Value> {
  let mut out = Map::new();
  for (name, c) in entries {
    match c.and_then(|c| c.stats()) {
      Ok(s) => {
        out.insert(name, model::stats_to_json(&s));
      }
      Err(muninn::Error::NotFound(_)) => {} // dropped since the listing
      Err(e) => {
        unavailable.insert(name, Value::String(ApiError::from(e).message));
      }
    }
  }
  out
}

/// `{"collections": {name: stats}, "unavailable": {name: reason}}`.
pub(crate) fn stats_json(db: &Database) -> Result<Value, ApiError> {
  let mut unavailable = match model::unavailable_json(db) {
    Value::Object(m) => m,
    _ => Map::new(),
  };
  let names = db.list_collections()?;
  let entries = names.into_iter().map(|n| {
    let c = db.collection(&n);
    (n, c)
  });
  let collections = collect_stats(entries, &mut unavailable);
  Ok(json!({"collections": collections, "unavailable": unavailable}))
}

async fn stats(State(st): State<AppState>) -> Result<Response, ApiError> {
  blocking(move || json_response(StatusCode::OK, &stats_json(&st.db)?)).await
}

async fn list_collections(
  State(st): State<AppState>,
) -> Result<Response, ApiError> {
  blocking(move || {
    let names = st.db.list_collections()?;
    json_response(
      StatusCode::OK,
      &json!({
        "collections": names,
        "unavailable": model::unavailable_json(&st.db),
      }),
    )
  })
  .await
}

async fn create_collection(
  State(st): State<AppState>,
  Params(name): Params<String>,
  headers: HeaderMap,
  body: Body,
) -> Result<Response, ApiError> {
  let body = guard::read_json(&st, &headers, body).await?;
  blocking(move || {
    let v: Value = serde_json::from_slice(&body.bytes)
      .map_err(|e| model::json_error("request body", e))?;
    let schema = model::schema_from_json(&v)?;
    let c = st.db.create_collection(&name, schema)?;
    st.remember(&c);
    json_response(StatusCode::CREATED, &describe(&c)?)
  })
  .await
}

async fn describe_collection(
  State(st): State<AppState>,
  Params(name): Params<String>,
) -> Result<Response, ApiError> {
  let c = st.collection(&name).await?;
  blocking(move || json_response(StatusCode::OK, &describe(&c)?)).await
}

async fn drop_collection(
  State(st): State<AppState>,
  Params(name): Params<String>,
) -> Result<Response, ApiError> {
  blocking(move || {
    st.forget(&name);
    let r = st.db.drop_collection(&name);
    // A lookup between the two may have cached the old handle.
    st.forget(&name);
    r?;
    json_response(StatusCode::OK, &json!({"name": name, "dropped": true}))
  })
  .await
}

async fn upsert(
  State(st): State<AppState>,
  Params(name): Params<String>,
  headers: HeaderMap,
  body: Body,
) -> Result<Response, ApiError> {
  guard::check_content_type(&headers, &["application/json"])?;
  let c = st.collection(&name).await?;
  let body = guard::read_json(&st, &headers, body).await?;
  blocking(move || {
    let start = Instant::now();
    let docs = model::parse_upsert(&body.bytes)?;
    drop(body);
    c.upsert(&docs)?;
    json_response(
      StatusCode::OK,
      &json!({"count": docs.len(), "took_us": micros(start.elapsed())}),
    )
  })
  .await
}

async fn delete_documents(
  State(st): State<AppState>,
  Params(name): Params<String>,
  headers: HeaderMap,
  body: Body,
) -> Result<Response, ApiError> {
  guard::check_content_type(&headers, &["application/json"])?;
  let c = st.collection(&name).await?;
  let body = guard::read_json(&st, &headers, body).await?;
  blocking(move || {
    let start = Instant::now();
    let ids = model::parse_delete(&body.bytes)?;
    c.delete(&ids)?;
    json_response(
      StatusCode::OK,
      &json!({"count": ids.len(), "took_us": micros(start.elapsed())}),
    )
  })
  .await
}

async fn get_document(
  State(st): State<AppState>,
  Params((name, id)): Params<(String, String)>,
  RawQuery(query): RawQuery,
) -> Result<Response, ApiError> {
  let id = model::parse_path_id(&id, query_param(&query, "id_type"))?;
  let c = st.collection(&name).await?;
  blocking(move || match c.get(&id)? {
    Some(d) => json_response(StatusCode::OK, &DocOut(&d)),
    None => Err(ApiError::not_found(format!(
      "document {} not found in collection {name:?}",
      match &id {
        muninn::Id::U64(n) => n.to_string(),
        muninn::Id::Str(s) => format!("{s:?}"),
      }
    ))),
  })
  .await
}

async fn search(
  State(st): State<AppState>,
  Params(name): Params<String>,
  headers: HeaderMap,
  body: Body,
) -> Result<Response, ApiError> {
  guard::check_content_type(&headers, &["application/json"])?;
  let c = st.collection(&name).await?;
  let body = guard::read_json(&st, &headers, body).await?;
  blocking(move || {
    let start = Instant::now();
    let q = SearchRequest::parse(&body.bytes)?.into_query()?;
    drop(body);
    let hits = c.search(&q)?;
    let resp = SearchResponse {
      hits: &hits,
      took_us: micros(start.elapsed()),
    };
    let mut out = Vec::with_capacity(model::response_capacity(&hits));
    serde_json::to_writer(&mut out, &resp)
      .map_err(|e| ApiError::internal(format!("encoding response: {e}")))?;
    Ok(json_bytes(StatusCode::OK, out))
  })
  .await
}

async fn flush(
  State(st): State<AppState>,
  Params(name): Params<String>,
) -> Result<Response, ApiError> {
  let c = st.collection(&name).await?;
  blocking(move || {
    let start = Instant::now();
    c.flush()?;
    json_response(StatusCode::OK, &json!({"took_us": micros(start.elapsed())}))
  })
  .await
}

async fn compact(
  State(st): State<AppState>,
  Params(name): Params<String>,
) -> Result<Response, ApiError> {
  let c = st.collection(&name).await?;
  blocking(move || {
    let start = Instant::now();
    c.compact()?;
    json_response(StatusCode::OK, &json!({"took_us": micros(start.elapsed())}))
  })
  .await
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn stats_survive_dropped_and_failing_collections() {
    let dir = std::env::temp_dir()
      .join(format!("muninn-server-stats-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(&dir, muninn::Options::default()).unwrap();
    let schema = muninn::Schema::text(["t"]);
    let a = db.create_collection("a", schema.clone()).unwrap();
    let b = db.create_collection("b", schema).unwrap();
    // `b` is dropped between the listing and its statistics.
    db.drop_collection("b").unwrap();
    let mut unavailable = Map::new();
    let out = collect_stats(
      [
        ("a".to_string(), Ok(a)),
        ("b".to_string(), Ok(b)),
        (
          "c".to_string(),
          Err(muninn::Error::Corrupt("bad manifest".into())),
        ),
      ],
      &mut unavailable,
    );
    assert_eq!(out.keys().collect::<Vec<_>>(), ["a"]);
    assert_eq!(unavailable.keys().collect::<Vec<_>>(), ["c"]);
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
  }
}
