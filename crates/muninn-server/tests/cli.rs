//! The `muninn` binary end to end: import, search, stats, compact, and
//! serve with a graceful shutdown.

mod common;

use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use common::TestDir;
use serde_json::{Value, json};

/// Runs a direct (non-serve) command. Background merging is off so that
/// the counts of deleted documents are deterministic.
fn muninn(args: &[&str]) -> Output {
  Command::new(env!("CARGO_BIN_EXE_muninn"))
    .args(args)
    .arg("--no-auto-compact")
    .output()
    .unwrap()
}

fn ok(args: &[&str]) -> String {
  let o = muninn(args);
  assert!(
    o.status.success(),
    "muninn {args:?} failed: {}",
    String::from_utf8_lossy(&o.stderr)
  );
  String::from_utf8(o.stdout).unwrap()
}

fn fails(args: &[&str]) -> String {
  let o = muninn(args);
  assert!(!o.status.success(), "muninn {args:?} should fail");
  String::from_utf8(o.stderr).unwrap()
}

fn write(path: &Path, text: &str) {
  std::fs::write(path, text).unwrap();
}

fn vector(i: u64) -> [f32; 4] {
  let x = i as f32;
  [x.sin(), x.cos(), (x * 0.3).sin(), 1.0]
}

#[test]
fn import_search_stats_compact() {
  let dir = TestDir::new("cli");
  let p = |name: &str| dir.path().join(name).to_str().unwrap().to_string();
  let data = p("data");
  write(
    &dir.path().join("schema.json"),
    r#"{"vector": {"dim": 4, "metric": "cosine"}, "text_fields": ["title"],
        "filterable": {"lang": "keyword"}}"#,
  );
  let mut jsonl = String::new();
  for i in 0..300u64 {
    let lang = if i % 3 == 0 { "de" } else { "en" };
    let title = format!(
      "document number {i} about {}",
      ["cats", "dogs"][i as usize % 2]
    );
    jsonl += &json!({
      "id": i, "vector": vector(i), "text": {"title": title},
      "payload": {"lang": lang}
    })
    .to_string();
    jsonl += "\n";
    if i == 10 {
      jsonl += "\n"; // blank lines are skipped
    }
  }
  write(&dir.path().join("docs.jsonl"), &jsonl);

  let out = ok(&[
    "import",
    "--data",
    &data,
    "--collection",
    "docs",
    "--create-schema",
    &p("schema.json"),
    "--batch",
    "64",
    "--quiet",
    &p("docs.jsonl"),
  ]);
  assert!(
    out.starts_with("imported 300 documents into \"docs\""),
    "{out}"
  );
  assert!(
    out.contains("docs/s") && out.contains("written to disk"),
    "{out}"
  );

  // Text search.
  let out = ok(&[
    "search",
    "--data",
    &data,
    "--collection",
    "docs",
    "--text",
    "cats",
    "-k",
    "3",
  ]);
  let v: Value = serde_json::from_str(&out).unwrap();
  let hits = v["hits"].as_array().unwrap();
  assert_eq!(hits.len(), 3);
  assert!(
    hits.iter().all(|h| h["id"].as_u64().unwrap() % 2 == 0),
    "{v}"
  );
  assert!(v["took_us"].is_u64());

  // Vector search with a filter, from a vector file.
  write(
    &dir.path().join("q.json"),
    &json!({"vector": vector(7)}).to_string(),
  );
  let out = ok(&[
    "search",
    "--data",
    &data,
    "--collection",
    "docs",
    "--vector-file",
    &p("q.json"),
    "--filter",
    r#"{"eq":{"lang":"en"}}"#,
    "--with-payload",
  ]);
  let v: Value = serde_json::from_str(&out).unwrap();
  let hits = v["hits"].as_array().unwrap();
  assert_eq!(hits.len(), 10);
  assert_eq!(hits[0]["id"], 7, "{v}");
  assert!(hits.iter().all(|h| h["payload"]["lang"] == "en"), "{v}");

  // Hybrid search with weighted fusion and a plain array vector file.
  write(&dir.path().join("q2.json"), &json!(vector(9)).to_string());
  let out = ok(&[
    "search",
    "--data",
    &data,
    "--collection",
    "docs",
    "--vector-file",
    &p("q2.json"),
    "--text",
    "dogs",
    "--fusion",
    r#"{"weighted":{"vector":0.5,"text":0.5}}"#,
    "--fields",
    "title",
    "-k",
    "5",
  ]);
  let v: Value = serde_json::from_str(&out).unwrap();
  assert_eq!(v["hits"][0]["id"], 9, "{v}");
  assert!(v["hits"][0]["text_score"].is_f64());

  // Stats: everything flushed by the import.
  let out = ok(&["stats", "--data", &data]);
  let v: Value = serde_json::from_str(&out).unwrap();
  let s = &v["collections"]["docs"]["stats"];
  assert_eq!(s["live_docs"], 300, "{v}");
  assert_eq!(s["memtable_docs"], 0, "{v}");
  assert_eq!(v["collections"]["docs"]["schema"]["vector"]["dim"], 4);

  // Re-importing replaces documents (same schema accepted); stdin works.
  let o = Command::new(env!("CARGO_BIN_EXE_muninn"))
    .args([
      "import",
      "--data",
      &data,
      "--collection",
      "docs",
      "--create-schema",
      &p("schema.json"),
      "--quiet",
      "--no-auto-compact",
      "-",
    ])
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .and_then(|mut c| {
      c.stdin.take().unwrap().write_all(jsonl.as_bytes())?;
      c.wait_with_output()
    })
    .unwrap();
  assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
  let out = ok(&["stats", "--data", &data]);
  let v: Value = serde_json::from_str(&out).unwrap();
  let s = &v["collections"]["docs"]["stats"];
  assert_eq!(s["live_docs"], 300, "{v}");
  assert_eq!(s["deleted_docs"], 300, "{v}");

  let out = ok(&["compact", "--data", &data, "--collection", "docs"]);
  assert!(out.starts_with("compacted \"docs\""), "{out}");
  assert!(out.contains("300 documents in 1 segment(s)"), "{out}");
  let out = ok(&["stats", "--data", &data]);
  let v: Value = serde_json::from_str(&out).unwrap();
  assert_eq!(v["collections"]["docs"]["stats"]["deleted_docs"], 0);

  // Errors.
  write(
    &dir.path().join("other.json"),
    r#"{"text_fields": ["title"]}"#,
  );
  let e = fails(&[
    "import",
    "--data",
    &data,
    "--collection",
    "docs",
    "--create-schema",
    &p("other.json"),
    &p("docs.jsonl"),
  ]);
  assert!(e.contains("exists with a different schema"), "{e}");
  let e = fails(&[
    "import",
    "--data",
    &data,
    "--collection",
    "new",
    &p("docs.jsonl"),
  ]);
  assert!(e.contains("--create-schema"), "{e}");
  // A failed import says how much of it was stored.
  write(
    &dir.path().join("bad.jsonl"),
    "{\"id\": 1001, \"vector\": [1,0,0,0]}\n\n\
     {\"id\": 1002, \"vector\": [1,0,0,0]\n",
  );
  let e = fails(&[
    "import",
    "--data",
    &data,
    "--collection",
    "docs",
    "--batch",
    "1",
    &p("bad.jsonl"),
  ]);
  assert!(e.contains("line 3:"), "{e}");
  assert!(e.contains("; the 1 document before it was stored"), "{e}");
  let out = ok(&["stats", "--data", &data]);
  let v: Value = serde_json::from_str(&out).unwrap();
  assert_eq!(v["collections"]["docs"]["stats"]["live_docs"], 301, "{v}");
  let e = fails(&[
    "import",
    "--data",
    &data,
    "--collection",
    "docs",
    "--batch",
    "0",
    &p("docs.jsonl"),
  ]);
  assert!(e.contains("--batch"), "{e}");
  write(
    &dir.path().join("dim.jsonl"),
    "{\"id\": 1, \"vector\": [1,0,0,0]}\n{\"id\": 2, \"vector\": [1,0]}\n",
  );
  let e = fails(&[
    "import",
    "--data",
    &data,
    "--collection",
    "docs",
    &p("dim.jsonl"),
  ]);
  assert!(e.contains("lines 1-2: vector has 2 dimensions"), "{e}");
  assert!(e.contains("; nothing was stored"), "{e}");
  let e = fails(&[
    "search",
    "--data",
    &data,
    "--collection",
    "docs",
    "--filter",
    "{",
  ]);
  assert!(e.contains("--filter is not valid JSON"), "{e}");
  let e = fails(&[
    "search",
    "--data",
    &data,
    "--collection",
    "nope",
    "--text",
    "x",
  ]);
  assert!(e.contains("collection \"nope\""), "{e}");
}

/// `muninn serve` answers on 127.0.0.1, holds the directory lock, and on
/// SIGTERM finishes, flushes every collection and exits cleanly.
#[cfg(unix)]
#[test]
fn serve_shuts_down_gracefully_on_sigterm() {
  use std::io::{BufRead, BufReader};
  use std::net::SocketAddr;

  let dir = TestDir::new("serve");
  let data = dir.path().join("data");
  let data = data.to_str().unwrap();
  let mut child = Command::new(env!("CARGO_BIN_EXE_muninn"))
    .args([
      "serve",
      "--data",
      data,
      "--addr",
      "127.0.0.1:0",
      "--durability",
      "none",
      "--log-level",
      "debug",
    ])
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
  let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
  let addr: SocketAddr = loop {
    let line = lines.next().expect("server exited early").unwrap();
    if let Some((_, a)) = line.split_once("listening on http://") {
      break a.trim().parse().unwrap();
    }
  };
  let log = std::thread::spawn(move || {
    lines.map(|l| l.unwrap()).collect::<Vec<_>>().join("\n")
  });

  let r = common::http(addr, "GET", "/v1/health", None);
  assert_eq!(r.0, 200);
  let r = common::http(
    addr,
    "PUT",
    "/v1/collections/s",
    Some(r#"{"text_fields": ["t"]}"#),
  );
  assert_eq!(r.0, 201, "{}", r.1);
  let r = common::http(
    addr,
    "POST",
    "/v1/collections/s/documents",
    Some(r#"{"documents": [{"id": 1, "text": {"t": "hello"}}]}"#),
  );
  assert_eq!(r.0, 200, "{}", r.1);

  // The directory is locked while the server runs.
  let e = fails(&["stats", "--data", data]);
  assert!(e.contains("already open"), "{e}");

  let status = Command::new("kill")
    .args(["-TERM", &child.id().to_string()])
    .status()
    .unwrap();
  assert!(status.success());
  let exit = child.wait().unwrap();
  let log = log.join().unwrap();
  assert!(exit.success(), "{exit:?}\n{log}");
  assert!(
    log.contains("POST /v1/collections/s/documents -> 200"),
    "{log}"
  );
  assert!(log.contains("flushed collection \"s\""), "{log}");
  assert!(log.contains("database closed"), "{log}");

  let out = ok(&["stats", "--data", data]);
  let v: Value = serde_json::from_str(&out).unwrap();
  let s = &v["collections"]["s"]["stats"];
  assert_eq!(s["live_docs"], 1, "{v}");
  assert_eq!(s["memtable_docs"], 0, "{v}");
}

/// With a request stuck, the first SIGTERM waits (up to the shutdown
/// timeout); a second one stops waiting, and the database is still
/// flushed and closed before the process exits.
#[cfg(unix)]
#[test]
fn second_sigterm_stops_waiting_but_still_closes() {
  use std::io::{BufRead, BufReader};
  use std::net::{SocketAddr, TcpStream};
  use std::time::{Duration, Instant};

  let dir = TestDir::new("serve2");
  let data = dir.path().join("data");
  let data = data.to_str().unwrap();
  let mut child = Command::new(env!("CARGO_BIN_EXE_muninn"))
    .args([
      "serve",
      "--data",
      data,
      "--addr",
      "127.0.0.1:0",
      "--shutdown-timeout",
      "60",
    ])
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
  let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
  let addr: SocketAddr = loop {
    let line = lines.next().expect("server exited early").unwrap();
    if let Some((_, a)) = line.split_once("listening on http://") {
      break a.trim().parse().unwrap();
    }
  };
  let log = std::thread::spawn(move || {
    lines.map(|l| l.unwrap()).collect::<Vec<_>>().join("\n")
  });
  let r = common::http(
    addr,
    "PUT",
    "/v1/collections/s",
    Some(r#"{"text_fields": ["t"]}"#),
  );
  assert_eq!(r.0, 201, "{}", r.1);
  let r = common::http(
    addr,
    "POST",
    "/v1/collections/s/documents",
    Some(r#"{"documents": [{"id": 1, "text": {"t": "hello"}}]}"#),
  );
  assert_eq!(r.0, 200, "{}", r.1);
  // A request whose body never completes.
  let mut stuck = TcpStream::connect(addr).unwrap();
  write!(
    stuck,
    "POST /v1/collections/s/search HTTP/1.1\r\nHost: 127.0.0.1\r\n\
     Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{{\"te"
  )
  .unwrap();
  std::thread::sleep(Duration::from_millis(200));

  let pid = child.id().to_string();
  let term = || {
    let ok = Command::new("kill")
      .args(["-TERM", &pid])
      .status()
      .unwrap()
      .success();
    assert!(ok);
  };
  term();
  std::thread::sleep(Duration::from_millis(1000));
  assert!(
    child.try_wait().unwrap().is_none(),
    "the first signal should wait for the open request"
  );
  term();
  let t = Instant::now();
  let exit = loop {
    if let Some(s) = child.try_wait().unwrap() {
      break s;
    }
    if t.elapsed() > Duration::from_secs(10) {
      let _ = child.kill();
      panic!("the second signal did not stop the server");
    }
    std::thread::sleep(Duration::from_millis(20));
  };
  let log = log.join().unwrap();
  assert!(exit.success(), "{exit:?}\n{log}");
  assert!(log.contains("second signal"), "{log}");
  assert!(log.contains("flushed collection \"s\""), "{log}");
  drop(stuck);
  let out = ok(&["stats", "--data", data]);
  let v: Value = serde_json::from_str(&out).unwrap();
  assert_eq!(v["collections"]["s"]["stats"]["memtable_docs"], 0, "{v}");
  assert_eq!(v["collections"]["s"]["stats"]["live_docs"], 1, "{v}");
}
