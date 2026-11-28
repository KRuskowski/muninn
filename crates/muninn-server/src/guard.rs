//! Request guards: which hosts may call the API, the optional API key,
//! the cap on requests in flight, and reading request bodies within the
//! size, memory and time limits.
//!
//! Browsers are the threat these checks address. Any web page can send a
//! "simple" cross-origin POST (`text/plain`, no preflight) to a server on
//! 127.0.0.1, and a DNS-rebinding page can even read the answers under its
//! own host name. Requiring `application/json` bodies makes every write a
//! non-simple request that the browser preflights (and this server never
//! approves); checking `Host` and `Origin` refuses rebinding pages and
//! cross-origin requests outright.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HOST};
use axum::http::{HeaderMap, Method};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body_util::{BodyExt, LengthLimitError, Limited};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::{ApiError, ErrorCode};
use crate::server::AppState;

/// The host part of `authority` (`host`, `host:port`, `[v6]:port`),
/// lowercased, without brackets or a trailing dot.
pub(crate) fn host_of(authority: &str) -> String {
  let a = authority.trim();
  let host = if let Some(rest) = a.strip_prefix('[') {
    rest.split(']').next().unwrap_or("")
  } else {
    match a.rsplit_once(':') {
      Some((h, port)) if port.bytes().all(|b| b.is_ascii_digit()) => h,
      _ => a,
    }
  };
  host.trim_end_matches('.').to_ascii_lowercase()
}

/// The host of an `Origin` header (`scheme://host[:port]`), if it has one.
fn origin_host(origin: &str) -> Option<String> {
  let (_, rest) = origin.split_once("://")?;
  Some(host_of(rest.split('/').next().unwrap_or("")))
}

fn host_allowed(st: &AppState, host: &str) -> bool {
  let hosts = &st.cfg.allowed_hosts;
  hosts.iter().any(|h| h == "*" || h == host)
}

/// Refuses requests whose `Host` (or absolute URI) or `Origin` names a
/// host outside [`crate::ServerConfig::allowed_hosts`].
pub(crate) async fn check_host(
  State(st): State<AppState>,
  req: Request,
  next: Next,
) -> Response {
  let authority = req
    .uri()
    .authority()
    .map(|a| a.as_str().to_string())
    .or_else(|| {
      req
        .headers()
        .get(HOST)
        .and_then(|h| h.to_str().ok())
        .map(String::from)
    });
  let any = st.cfg.allowed_hosts.iter().any(|h| h == "*");
  if !any {
    let Some(authority) = authority else {
      return ApiError::new(ErrorCode::Forbidden, "the Host header is missing")
        .into_response();
    };
    let host = host_of(&authority);
    if !host_allowed(&st, &host) {
      return ApiError::new(
        ErrorCode::Forbidden,
        format!(
          "host {host:?} is not allowed; start the server with \
           --allowed-host {host} to accept it"
        ),
      )
      .into_response();
    }
  }
  if let Some(origin) = req.headers().get("origin") {
    let host = origin.to_str().ok().and_then(origin_host);
    let ok = match &host {
      Some(h) => any || host_allowed(&st, h),
      None => false, // "null" or garbage: sandboxed or file:// pages
    };
    if !ok {
      return ApiError::new(
        ErrorCode::Forbidden,
        "cross-origin requests from web pages are not allowed",
      )
      .into_response();
    }
  }
  next.run(req).await
}

/// Compares two byte strings in time that depends only on their lengths.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
  let mut diff = u8::from(a.len() != b.len());
  for i in 0..a.len().max(b.len()) {
    let x = a.get(i).copied().unwrap_or(0);
    let y = b.get(i).copied().unwrap_or(0);
    diff |= x ^ y;
  }
  diff == 0
}

/// Requires `Authorization: Bearer <key>` when an API key is configured,
/// on every route except `GET /v1/health`.
pub(crate) async fn check_api_key(
  State(st): State<AppState>,
  req: Request,
  next: Next,
) -> Response {
  if let Some(key) = &st.cfg.api_key {
    let health =
      req.method() == Method::GET && req.uri().path() == "/v1/health";
    let given = req
      .headers()
      .get(AUTHORIZATION)
      .and_then(|h| h.to_str().ok())
      .and_then(|h| h.strip_prefix("Bearer "))
      .map(str::trim);
    let ok =
      given.is_some_and(|g| constant_time_eq(g.as_bytes(), key.as_bytes()));
    if !health && !ok {
      return ApiError::new(
        ErrorCode::Unauthenticated,
        "missing or wrong API key; send \"Authorization: Bearer <key>\"",
      )
      .into_response();
    }
  }
  next.run(req).await
}

/// Refuses requests beyond [`crate::ServerConfig::max_requests`] in
/// flight with 503, instead of queueing them without bound.
pub(crate) async fn limit_requests(
  State(st): State<AppState>,
  req: Request,
  next: Next,
) -> Response {
  let Ok(_permit) = st.limits.requests.clone().try_acquire_owned() else {
    return ApiError::new(
      ErrorCode::Unavailable,
      "too many requests in flight; retry shortly",
    )
    .into_response();
  };
  next.run(req).await
}

/// Shared limits of one server.
pub(crate) struct Limits {
  /// One permit per request in flight.
  pub requests: Arc<Semaphore>,
  /// One permit per byte of request bodies held in memory.
  pub body_bytes: Arc<Semaphore>,
  /// One permit per bulk request in flight.
  pub bulk: Arc<Semaphore>,
}

/// Fails unless the `Content-Type` is one of `allowed` (parameters such as
/// `charset` are ignored).
pub(crate) fn check_content_type(
  headers: &HeaderMap,
  allowed: &[&str],
) -> Result<(), ApiError> {
  let ct = headers
    .get(CONTENT_TYPE)
    .and_then(|v| v.to_str().ok())
    .map(|v| {
      v.split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
    });
  match ct {
    Some(ct) if allowed.contains(&ct.as_str()) => Ok(()),
    Some(ct) => Err(ApiError::new(
      ErrorCode::UnsupportedMediaType,
      format!(
        "Content-Type {ct:?} is not accepted here; send {}",
        allowed.join(" or ")
      ),
    )),
    None => Err(ApiError::new(
      ErrorCode::UnsupportedMediaType,
      format!("the Content-Type header is missing; send {}", allowed[0]),
    )),
  }
}

/// Reserves `bytes` of the in-flight body budget, or fails with 503.
pub(crate) fn reserve_body_bytes(
  st: &AppState,
  bytes: usize,
) -> Result<OwnedSemaphorePermit, ApiError> {
  let n = u32::try_from(bytes.max(1)).unwrap_or(u32::MAX);
  st.limits
    .body_bytes
    .clone()
    .try_acquire_many_owned(n)
    .map_err(|_| {
      ApiError::new(
        ErrorCode::Unavailable,
        "too much request data in flight; retry shortly",
      )
    })
}

/// A request body read into memory, and its share of the in-flight body
/// budget (released when this is dropped).
pub(crate) struct JsonBody {
  pub bytes: Bytes,
  _permit: OwnedSemaphorePermit,
}

/// Reads a JSON request body: checks the content type, refuses bodies
/// over the limit by their `Content-Length` before reading, reserves the
/// body's bytes of the in-flight budget, and reads within the body
/// timeout.
pub(crate) async fn read_json(
  st: &AppState,
  headers: &HeaderMap,
  body: Body,
) -> Result<JsonBody, ApiError> {
  check_content_type(headers, &["application/json"])?;
  let limit = st.cfg.body_limit;
  let too_large = || {
    ApiError::new(
      ErrorCode::PayloadTooLarge,
      format!("request body exceeds the limit of {limit} bytes"),
    )
  };
  let declared = headers
    .get(CONTENT_LENGTH)
    .and_then(|v| v.to_str().ok())
    .and_then(|v| v.parse::<u64>().ok());
  if declared.is_some_and(|n| n > limit as u64) {
    return Err(too_large());
  }
  // Without a length (chunked), assume the worst.
  let permit = reserve_body_bytes(st, declared.map_or(limit, |n| n as usize))?;
  let read = Limited::new(body, limit).collect();
  let collected = tokio::time::timeout(st.cfg.body_timeout, read)
    .await
    .map_err(|_| {
      ApiError::new(
        ErrorCode::RequestTimeout,
        format!(
          "the request body did not arrive within {:?}",
          st.cfg.body_timeout
        ),
      )
    })?
    .map_err(|e| {
      if e.downcast_ref::<LengthLimitError>().is_some() {
        too_large()
      } else {
        ApiError::invalid(format!("reading request body: {e}"))
      }
    })?;
  Ok(JsonBody {
    bytes: collected.to_bytes(),
    _permit: permit,
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn hosts_and_origins() {
    assert_eq!(host_of("127.0.0.1:7700"), "127.0.0.1");
    assert_eq!(host_of("LocalHost."), "localhost");
    assert_eq!(host_of("[::1]:7700"), "::1");
    assert_eq!(host_of("[::1]"), "::1");
    assert_eq!(host_of("evil.example:80"), "evil.example");
    assert_eq!(
      origin_host("http://evil.example:8080/x").as_deref(),
      Some("evil.example")
    );
    assert_eq!(origin_host("null"), None);
  }

  #[test]
  fn constant_time_comparison() {
    assert!(constant_time_eq(b"secret", b"secret"));
    assert!(!constant_time_eq(b"secret", b"secreT"));
    assert!(!constant_time_eq(b"secret", b"secret2"));
    assert!(!constant_time_eq(b"", b"x"));
    assert!(constant_time_eq(b"", b""));
  }
}
