//! The error body every endpoint returns on failure.
//!
//! ```json
//! {"error": {"code": "not_found", "message": "collection \"docs\" not found"}}
//! ```

use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};

/// Machine-readable error kind, with its HTTP status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
  /// 400: the request is malformed or does not fit the collection.
  InvalidArgument,
  /// 404: unknown collection, document or route.
  NotFound,
  /// 405: the route exists but not with this method.
  MethodNotAllowed,
  /// 409: the collection already exists.
  AlreadyExists,
  /// 413: the request body exceeds the configured limit.
  PayloadTooLarge,
  /// 500: I/O failure, corrupt data or a server bug.
  Internal,
}

impl ErrorCode {
  /// The `code` string in the error body.
  pub fn as_str(self) -> &'static str {
    match self {
      Self::InvalidArgument => "invalid_argument",
      Self::NotFound => "not_found",
      Self::MethodNotAllowed => "method_not_allowed",
      Self::AlreadyExists => "already_exists",
      Self::PayloadTooLarge => "payload_too_large",
      Self::Internal => "internal",
    }
  }

  /// The HTTP status code.
  pub fn status(self) -> StatusCode {
    match self {
      Self::InvalidArgument => StatusCode::BAD_REQUEST,
      Self::NotFound => StatusCode::NOT_FOUND,
      Self::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
      Self::AlreadyExists => StatusCode::CONFLICT,
      Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
      Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
  }
}

/// A failed request: a code and a human-readable message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
  /// The error kind.
  pub code: ErrorCode,
  /// What went wrong, for people.
  pub message: String,
}

impl ApiError {
  /// An error of kind `code`.
  pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
    Self {
      code,
      message: message.into(),
    }
  }

  /// A 400 `invalid_argument` error.
  pub fn invalid(message: impl Into<String>) -> Self {
    Self::new(ErrorCode::InvalidArgument, message)
  }

  /// A 404 `not_found` error.
  pub fn not_found(message: impl Into<String>) -> Self {
    Self::new(ErrorCode::NotFound, message)
  }

  /// A 500 `internal` error.
  pub fn internal(message: impl Into<String>) -> Self {
    Self::new(ErrorCode::Internal, message)
  }
}

impl std::fmt::Display for ApiError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}: {}", self.code.as_str(), self.message)
  }
}

impl std::error::Error for ApiError {}

impl From<muninn::Error> for ApiError {
  fn from(e: muninn::Error) -> Self {
    use muninn::Error as E;
    match e {
      E::InvalidArgument(m) => Self::invalid(m),
      e @ E::DimensionMismatch { .. } => Self::invalid(e.to_string()),
      E::NotFound(m) => Self::not_found(format!("{m} not found")),
      E::AlreadyExists(m) => {
        Self::new(ErrorCode::AlreadyExists, format!("{m} already exists"))
      }
      e @ (E::Io(_) | E::Corrupt(_)) => Self::internal(e.to_string()),
    }
  }
}

impl IntoResponse for ApiError {
  fn into_response(self) -> Response {
    if self.code == ErrorCode::Internal {
      tracing::error!("{}", self.message);
    }
    let body = serde_json::json!({
      "error": {"code": self.code.as_str(), "message": self.message}
    });
    (
      self.code.status(),
      [(CONTENT_TYPE, "application/json")],
      body.to_string(),
    )
      .into_response()
  }
}
