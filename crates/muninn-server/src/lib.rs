//! HTTP/JSON server and command-line tool for the muninn database.
//!
//! The server is an [`axum`] router over one [`muninn::Database`]; build
//! it with [`router`] and run it with [`serve`], or use the `muninn`
//! binary (`muninn serve --data DIR`). `docs/API.md` describes every
//! endpoint and the security model.
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
//! let signals = muninn_server::Signals::listen()?;
//! muninn_server::serve(listener, state, signals.nth(1), signals.nth(2))
//!   .await?;
//! muninn_server::close(db, std::time::Duration::from_secs(30)).unwrap();
//! # Ok(())
//! # }
//! ```

mod bulk;
pub mod cli;
mod error;
mod guard;
pub mod model;
mod server;

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use muninn::Database;
use tokio::sync::{Semaphore, watch};
use tokio::task::JoinSet;

pub use error::{ApiError, ErrorCode};
pub use server::{AppState, MAX_BULK_BATCH, ServerConfig, router};

/// Serves the API on `listener` until `shutdown` completes. Then it stops
/// accepting connections and lets open ones finish their requests for at
/// most [`ServerConfig::shutdown_timeout`], or until `force` completes,
/// and closes whatever is still open.
///
/// Each connection gets `TCP_NODELAY`, must send each request's headers
/// within [`ServerConfig::header_timeout`] (which also ends idle
/// keep-alive connections), and beyond [`ServerConfig::max_connections`]
/// new connections are closed at once.
pub async fn serve(
  listener: tokio::net::TcpListener,
  state: AppState,
  shutdown: impl Future<Output = ()>,
  force: impl Future<Output = ()>,
) -> std::io::Result<()> {
  let cfg = state.config().clone();
  let service = TowerToHyperService::new(router(state));
  let slots = Arc::new(Semaphore::new(cfg.max_connections.max(1)));
  let mut http = http1::Builder::new();
  http
    .timer(TokioTimer::new())
    .header_read_timeout(cfg.header_timeout)
    .keep_alive(true);
  let graceful = GracefulShutdown::new();
  let mut conns = JoinSet::new();
  tokio::pin!(shutdown);
  loop {
    while conns.try_join_next().is_some() {}
    let (tcp, peer) = tokio::select! {
      r = listener.accept() => match r {
        Ok(x) => x,
        Err(e) => {
          // Out of file descriptors and the like: back off, retry.
          tracing::warn!("accepting a connection: {e}");
          tokio::time::sleep(Duration::from_millis(50)).await;
          continue;
        }
      },
      () = &mut shutdown => break,
    };
    let Ok(slot) = slots.clone().try_acquire_owned() else {
      tracing::warn!(
        "{} connections open; closing the one from {peer}",
        cfg.max_connections
      );
      drop(tcp);
      continue;
    };
    if let Err(e) = tcp.set_nodelay(true) {
      tracing::debug!("cannot set TCP_NODELAY: {e}");
    }
    let conn =
      graceful.watch(http.serve_connection(TokioIo::new(tcp), service.clone()));
    conns.spawn(async move {
      if let Err(e) = conn.await {
        tracing::debug!("connection from {peer}: {e}");
      }
      drop(slot);
    });
  }
  drop(listener);
  let open = conns.len();
  if open > 0 {
    tracing::info!(
      "waiting up to {:?} for {open} open connection(s)",
      cfg.shutdown_timeout
    );
  }
  tokio::select! {
    () = graceful.shutdown() => {}
    () = tokio::time::sleep(cfg.shutdown_timeout) => {
      tracing::warn!(
        "closing {} connection(s) still open after {:?}",
        conns.len(),
        cfg.shutdown_timeout
      );
    }
    () = force => {
      tracing::warn!("second signal: closing open connections now");
    }
  }
  // Not awaited: a request running in place (a long compaction, say)
  // cannot be interrupted, and close() waits for its handle anyway.
  conns.abort_all();
  drop(conns);
  Ok(())
}

/// Counts SIGINT (Ctrl-C) and, on Unix, SIGTERM signals for the life of
/// the runtime it was created in.
#[derive(Clone)]
pub struct Signals {
  count: watch::Receiver<u32>,
}

impl Signals {
  /// Starts listening. Must be called inside a tokio runtime.
  pub fn listen() -> std::io::Result<Signals> {
    let (tx, rx) = watch::channel(0u32);
    #[cfg(unix)]
    {
      use tokio::signal::unix::{SignalKind, signal};
      let mut int = signal(SignalKind::interrupt())?;
      let mut term = signal(SignalKind::terminate())?;
      tokio::spawn(async move {
        loop {
          tokio::select! {
            Some(()) = int.recv() => {}
            Some(()) = term.recv() => {}
            else => return,
          }
          tx.send_modify(|n| *n += 1);
          tracing::info!("shutdown signal {}", *tx.borrow());
        }
      });
    }
    #[cfg(not(unix))]
    tokio::spawn(async move {
      while tokio::signal::ctrl_c().await.is_ok() {
        tx.send_modify(|n| *n += 1);
        tracing::info!("shutdown signal {}", *tx.borrow());
      }
    });
    Ok(Signals { count: rx })
  }

  /// Signals received so far.
  pub fn count(&self) -> u32 {
    *self.count.borrow()
  }

  /// Completes once `n` signals have arrived.
  pub fn nth(&self, n: u32) -> impl Future<Output = ()> + Send + 'static {
    let mut rx = self.count.clone();
    async move {
      if rx.wait_for(|c| *c >= n).await.is_err() {
        std::future::pending::<()>().await;
      }
    }
  }
}

/// Flushes every collection to segments, then closes the database once
/// no other handle to it is left, waiting up to `wait` for requests still
/// finishing. Returns the first flush error; the flushes of the other
/// collections are still attempted. Blocking: call it from a thread that
/// may block (not an async worker).
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
