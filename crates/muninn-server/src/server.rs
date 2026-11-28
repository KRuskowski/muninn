//! The router and its handlers.
//!
//! Every engine call (and the JSON parsing and encoding around it) runs
//! as blocking work (see `blocking`), so the async runtime keeps moving
//! bytes for other connections and is never held up by a search, an
//! fsync or a large body.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{
  DefaultBodyLimit, FromRequest, FromRequestParts, Path, RawQuery, Request,
  State,
};
use axum::http::header::CONTENT_TYPE;
use axum::http::request::Parts;
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use muninn::{Collection, Database};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::error::{ApiError, ErrorCode};
use crate::model::{self, DocOut, SearchRequest, SearchResponse};

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
  /// Largest accepted request body in bytes, and the longest line of an
  /// NDJSON bulk body. Default 64 MiB.
  pub body_limit: usize,
  /// Documents per write batch of the NDJSON bulk endpoint, unless the
  /// request sets `?batch=`. Default 1000.
  pub bulk_batch: usize,
}

impl Default for ServerConfig {
  fn default() -> Self {
    Self {
      body_limit: 64 << 20,
      bulk_batch: 1000,
    }
  }
}

/// Largest `?batch=` of the bulk endpoint.
pub const MAX_BULK_BATCH: usize = 100_000;

/// Shared state of all handlers: the database and the settings. Cheap to
/// clone.
#[derive(Clone)]
pub struct AppState {
  pub(crate) db: Arc<Database>,
  pub(crate) cfg: Arc<ServerConfig>,
}

impl AppState {
  /// State serving `db` with `cfg`.
  pub fn new(db: Arc<Database>, cfg: ServerConfig) -> Self {
    Self {
      db,
      cfg: Arc::new(cfg),
    }
  }

  /// The database.
  pub fn database(&self) -> &Arc<Database> {
    &self.db
  }

  pub(crate) fn collection(&self, name: &str) -> Result<Collection, ApiError> {
    Ok(self.db.collection(name)?)
  }
}

/// The HTTP API under `/v1` (see `docs/API.md`).
pub fn router(state: AppState) -> Router {
  let limit = state.cfg.body_limit;
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
    .layer(DefaultBodyLimit::max(limit))
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

/// The whole request body, limited to [`ServerConfig::body_limit`]. Its
/// content type is not checked: every body is JSON.
pub(crate) struct Body(pub Bytes);

impl FromRequest<AppState> for Body {
  type Rejection = ApiError;

  async fn from_request(
    req: Request,
    state: &AppState,
  ) -> Result<Self, ApiError> {
    Bytes::from_request(req, state)
      .await
      .map(Body)
      .map_err(|e| {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
          ApiError::new(
            ErrorCode::PayloadTooLarge,
            format!(
              "request body exceeds the limit of {} bytes",
              state.cfg.body_limit
            ),
          )
        } else {
          ApiError::invalid(format!("reading request body: {}", e.body_text()))
        }
      })
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

async fn stats(State(st): State<AppState>) -> Result<Response, ApiError> {
  blocking(move || {
    let mut out = serde_json::Map::new();
    for name in st.db.list_collections()? {
      // Skip collections dropped since the listing.
      let c = match st.db.collection(&name) {
        Ok(c) => c,
        Err(muninn::Error::NotFound(_)) => continue,
        Err(e) => return Err(e.into()),
      };
      out.insert(name, model::stats_to_json(&c.stats()?));
    }
    json_response(
      StatusCode::OK,
      &json!({
        "collections": out,
        "unavailable": model::unavailable_json(&st.db),
      }),
    )
  })
  .await
}

async fn list_collections(
  State(st): State<AppState>,
) -> Result<Response, ApiError> {
  let names = st.db.list_collections()?;
  json_response(
    StatusCode::OK,
    &json!({
      "collections": names,
      "unavailable": model::unavailable_json(&st.db),
    }),
  )
}

async fn create_collection(
  State(st): State<AppState>,
  Params(name): Params<String>,
  Body(body): Body,
) -> Result<Response, ApiError> {
  blocking(move || {
    let v: Value = serde_json::from_slice(&body)
      .map_err(|e| model::json_error("request body", e))?;
    let schema = model::schema_from_json(&v)?;
    let c = st.db.create_collection(&name, schema)?;
    json_response(StatusCode::CREATED, &describe(&c)?)
  })
  .await
}

async fn describe_collection(
  State(st): State<AppState>,
  Params(name): Params<String>,
) -> Result<Response, ApiError> {
  let c = st.collection(&name)?;
  blocking(move || json_response(StatusCode::OK, &describe(&c)?)).await
}

async fn drop_collection(
  State(st): State<AppState>,
  Params(name): Params<String>,
) -> Result<Response, ApiError> {
  blocking(move || {
    st.db.drop_collection(&name)?;
    json_response(StatusCode::OK, &json!({"name": name, "dropped": true}))
  })
  .await
}

async fn upsert(
  State(st): State<AppState>,
  Params(name): Params<String>,
  Body(body): Body,
) -> Result<Response, ApiError> {
  let c = st.collection(&name)?;
  blocking(move || {
    let start = Instant::now();
    let docs = model::parse_upsert(&body)?;
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
  Body(body): Body,
) -> Result<Response, ApiError> {
  let c = st.collection(&name)?;
  blocking(move || {
    let start = Instant::now();
    let ids = model::parse_delete(&body)?;
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
  let c = st.collection(&name)?;
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
  Body(body): Body,
) -> Result<Response, ApiError> {
  let c = st.collection(&name)?;
  blocking(move || {
    let start = Instant::now();
    let q = SearchRequest::parse(&body)?.into_query()?;
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
  let c = st.collection(&name)?;
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
  let c = st.collection(&name)?;
  blocking(move || {
    let start = Instant::now();
    c.compact()?;
    json_response(StatusCode::OK, &json!({"took_us": micros(start.elapsed())}))
  })
  .await
}
