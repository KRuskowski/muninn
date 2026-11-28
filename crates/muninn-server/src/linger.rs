//! Lingering close for client connections.
//!
//! When the server answers a request before reading its body (a body over
//! the size limit, a full in-flight budget), the client may still be
//! sending that body as the server closes. A socket closed with unread
//! data, or receiving data after it closed, makes the kernel answer with
//! a reset, and on some systems (macOS among them) the reset destroys the
//! response before the client reads it. So, like other HTTP servers, the
//! server shuts down its sending side first and then reads and discards
//! what the client still sends, for a bounded time and number of bytes,
//! before closing the socket (RFC 9112, section 9.6).

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll, ready};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio::time::Sleep;

/// Longest time a closing connection keeps discarding client data.
const LINGER_TIME: Duration = Duration::from_secs(2);

/// Most client data a closing connection discards before giving up.
const LINGER_BYTES: usize = 1 << 20;

enum State {
  Open,
  Draining {
    deadline: Pin<Box<Sleep>>,
    left: usize,
  },
  Closed,
}

/// A TCP stream whose shutdown lingers: see the module documentation.
pub(crate) struct LingerStream {
  tcp: TcpStream,
  state: State,
}

impl LingerStream {
  pub(crate) fn new(tcp: TcpStream) -> Self {
    Self {
      tcp,
      state: State::Open,
    }
  }
}

impl AsyncRead for LingerStream {
  fn poll_read(
    mut self: Pin<&mut Self>,
    cx: &mut Context<'_>,
    buf: &mut ReadBuf<'_>,
  ) -> Poll<io::Result<()>> {
    Pin::new(&mut self.tcp).poll_read(cx, buf)
  }
}

impl AsyncWrite for LingerStream {
  fn poll_write(
    mut self: Pin<&mut Self>,
    cx: &mut Context<'_>,
    buf: &[u8],
  ) -> Poll<io::Result<usize>> {
    Pin::new(&mut self.tcp).poll_write(cx, buf)
  }

  fn poll_write_vectored(
    mut self: Pin<&mut Self>,
    cx: &mut Context<'_>,
    bufs: &[io::IoSlice<'_>],
  ) -> Poll<io::Result<usize>> {
    Pin::new(&mut self.tcp).poll_write_vectored(cx, bufs)
  }

  fn is_write_vectored(&self) -> bool {
    self.tcp.is_write_vectored()
  }

  fn poll_flush(
    mut self: Pin<&mut Self>,
    cx: &mut Context<'_>,
  ) -> Poll<io::Result<()>> {
    Pin::new(&mut self.tcp).poll_flush(cx)
  }

  /// Sends FIN, then discards incoming data until the client closes too,
  /// [`LINGER_TIME`] passes or [`LINGER_BYTES`] arrived. Never fails once
  /// the FIN is sent: errors while discarding just end the wait.
  fn poll_shutdown(
    mut self: Pin<&mut Self>,
    cx: &mut Context<'_>,
  ) -> Poll<io::Result<()>> {
    let this = &mut *self;
    loop {
      match &mut this.state {
        State::Open => {
          ready!(Pin::new(&mut this.tcp).poll_shutdown(cx))?;
          this.state = State::Draining {
            deadline: Box::pin(tokio::time::sleep(LINGER_TIME)),
            left: LINGER_BYTES,
          };
        }
        State::Draining { deadline, left } => {
          if deadline.as_mut().poll(cx).is_ready() {
            this.state = State::Closed;
            continue;
          }
          let mut scratch = [0u8; 8192];
          let mut buf = ReadBuf::new(&mut scratch);
          match Pin::new(&mut this.tcp).poll_read(cx, &mut buf) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Ok(())) if !buf.filled().is_empty() => {
              *left = left.saturating_sub(buf.filled().len());
              if *left == 0 {
                this.state = State::Closed;
              }
            }
            // End of stream or an error: the client is done.
            Poll::Ready(_) => this.state = State::Closed,
          }
        }
        State::Closed => return Poll::Ready(Ok(())),
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use std::io::{Read, Write};

  use super::*;
  use tokio::io::AsyncWriteExt;

  /// The shutdown waits for the client's data, then returns once the
  /// client closes, and the client reads everything the server sent.
  #[tokio::test]
  async fn shutdown_discards_late_client_data_until_the_client_closes() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = std::thread::spawn(move || {
      let mut c = std::net::TcpStream::connect(addr).unwrap();
      std::thread::sleep(Duration::from_millis(100));
      c.write_all(&vec![b'x'; 100_000]).unwrap();
      let mut got = Vec::new();
      c.read_to_end(&mut got).unwrap();
      got
    });
    let (tcp, _) = listener.accept().await.unwrap();
    let mut s = LingerStream::new(tcp);
    s.write_all(b"reply").await.unwrap();
    let t = std::time::Instant::now();
    s.shutdown().await.unwrap();
    assert!(t.elapsed() < LINGER_TIME, "{:?}", t.elapsed());
    // Closing with the client's data unread would reset the connection.
    drop(s);
    assert_eq!(client.join().unwrap(), b"reply");
  }

  /// A client that never closes holds the shutdown for at most
  /// [`LINGER_TIME`].
  #[tokio::test]
  async fn shutdown_gives_up_after_the_linger_time() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let c = std::net::TcpStream::connect(addr).unwrap();
    let (tcp, _) = listener.accept().await.unwrap();
    let mut s = LingerStream::new(tcp);
    let t = std::time::Instant::now();
    s.shutdown().await.unwrap();
    let took = t.elapsed();
    assert!(took >= LINGER_TIME && took < LINGER_TIME * 3, "{took:?}");
    drop(c);
  }
}
