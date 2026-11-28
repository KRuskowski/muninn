//! HTTP/JSON server and command-line tool for the muninn database.
//!
//! The server is an [`axum`] router over one [`muninn::Database`]; build
//! it with [`router`] and run it with [`serve`], or use the `muninn`
//! binary (`muninn serve --data DIR`). `docs/API.md` describes every
//! endpoint.
//!
//! ```no_run
//! use std::sync::Arc;
//! use muninn::{Database, Options};
//! use muninn_server::{AppState, ServerConfig};
//!
//! # async fn run() -> std::io::Result<()> {
//! let db = Arc::new(Database::open("data", Options::default()).unwrap());
//! let state = AppState::new(db.clone(), ServerConfig::default());
//! let listener = tokio::net::TcpListener::bind("127.0.0.1:7700").await?;
//! muninn_server::serve(listener, state, muninn_server::shutdown_signal())
//!   .await?;
//! muninn_server::close(db, std::time::Duration::from_secs(30));
//! # Ok(())
//! # }
//! ```

mod bulk;
pub mod cli;
mod error;
pub mod model;
mod server;

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use muninn::Database;

pub use error::{ApiError, ErrorCode};
pub use server::{AppState, MAX_BULK_BATCH, ServerConfig, router};

/// Serves the API on `listener` until `shutdown` completes, then stops
/// accepting connections and waits for requests in flight to finish.
/// Accepted connections get `TCP_NODELAY`.
pub async fn serve(
  listener: tokio::net::TcpListener,
  state: AppState,
  shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
  use axum::serve::ListenerExt;
  // Responses go out as soon as they are written, not after a delayed
  // acknowledgement (Nagle).
  let listener = listener.tap_io(|tcp| {
    if let Err(e) = tcp.set_nodelay(true) {
      tracing::debug!("cannot set TCP_NODELAY: {e}");
    }
  });
  axum::serve(listener, router(state))
    .with_graceful_shutdown(shutdown)
    .await
}

/// Completes on Ctrl-C (SIGINT) or, on Unix, SIGTERM.
pub async fn shutdown_signal() {
  let ctrl_c = async {
    if let Err(e) = tokio::signal::ctrl_c().await {
      tracing::error!("cannot listen for Ctrl-C: {e}");
      std::future::pending::<()>().await;
    }
  };
  #[cfg(unix)]
  let term = async {
    use tokio::signal::unix::{SignalKind, signal};
    match signal(SignalKind::terminate()) {
      Ok(mut s) => {
        s.recv().await;
      }
      Err(e) => {
        tracing::error!("cannot listen for SIGTERM: {e}");
        std::future::pending::<()>().await;
      }
    }
  };
  #[cfg(not(unix))]
  let term = std::future::pending::<()>();
  tokio::select! {
    () = ctrl_c => {}
    () = term => {}
  }
  tracing::info!("shutdown requested");
}

/// Flushes every collection to segments, then closes the database once
/// no other handle to it is left, waiting up to `wait` for blocking
/// requests still finishing. Returns the first flush error; the flushes
/// of the other collections are still attempted. Blocking: call it from
/// a thread that may block (not an async worker).
pub fn close(db: Arc<Database>, wait: Duration) -> muninn::Result<()> {
  let mut first_err = None;
  for name in db.list_collections()? {
    let r = db.collection(&name).and_then(|c| c.flush());
    match r {
      Ok(()) => tracing::info!("flushed collection {name:?}"),
      Err(e) => {
        tracing::error!("flushing collection {name:?}: {e}");
        first_err.get_or_insert(e);
      }
    }
  }
  let deadline = Instant::now() + wait;
  let mut db = db;
  loop {
    match Arc::try_unwrap(db) {
      Ok(d) => {
        drop(d);
        tracing::info!("database closed");
        break;
      }
      Err(still_shared) => {
        if Instant::now() >= deadline {
          tracing::warn!(
            "database still in use after {wait:?}; it closes when the last \
             request finishes"
          );
          break;
        }
        db = still_shared;
        std::thread::sleep(Duration::from_millis(10));
      }
    }
  }
  match first_err {
    Some(e) => Err(e),
    None => Ok(()),
  }
}
