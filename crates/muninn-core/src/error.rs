use std::fmt;

/// Result alias used across muninn.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every error muninn reports.
#[derive(Debug)]
pub enum Error {
  /// An operating system I/O error.
  Io(std::io::Error),
  /// On-disk data failed validation: bad magic, checksum or bounds.
  Corrupt(String),
  /// A caller passed something unusable.
  InvalidArgument(String),
  /// A vector had the wrong number of dimensions.
  DimensionMismatch { expected: usize, got: usize },
  /// A named thing (collection, document, field) does not exist.
  NotFound(String),
  /// A named thing already exists.
  AlreadyExists(String),
}

impl Error {
  pub fn corrupt(msg: impl Into<String>) -> Self {
    Self::Corrupt(msg.into())
  }

  pub fn invalid(msg: impl Into<String>) -> Self {
    Self::InvalidArgument(msg.into())
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Io(e) => write!(f, "I/O error: {e}"),
      Self::Corrupt(m) => write!(f, "corrupt data: {m}"),
      Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
      Self::DimensionMismatch { expected, got } => write!(
        f,
        "vector has {got} dimensions, collection expects {expected}"
      ),
      Self::NotFound(m) => write!(f, "not found: {m}"),
      Self::AlreadyExists(m) => write!(f, "already exists: {m}"),
    }
  }
}

impl std::error::Error for Error {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      Self::Io(e) => Some(e),
      _ => None,
    }
  }
}

impl From<std::io::Error> for Error {
  fn from(e: std::io::Error) -> Self {
    Self::Io(e)
  }
}
