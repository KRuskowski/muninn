//! Regression tests for the findings of the independent engine review.
//! Each was seen to fail before its fix.

mod common;

use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use common::{TestDir, files_with_prefix};
use muninn::{
  Database, Document, Durability, Error, Failpoint, FieldSpec, Filter, Id,
  Metric, Options, Query, Schema, VectorField,
};
use serde_json::{Value, json};

const DIM: usize = 4;

fn schema() -> Schema {
  Schema::vector(VectorField::flat(DIM, Metric::L2))
    .with_text_fields(["t"])
    .with_filterable(FieldSpec::keyword("tag"))
}

fn doc(id: u64, v: u64) -> Document {
  Document::new(id)
    .vector(vec![id as f32, 0.0, 0.0, v as f32])
    .text("t", format!("doc {id}"))
    .payload(json!({"v": v, "tag": "x"}))
}

fn version(c: &muninn::Collection, id: u64) -> Option<u64> {
  c.get(&Id::U64(id))
    .unwrap()
    .map(|d| d.payload["v"].as_u64().unwrap())
}

fn nested(depth: usize) -> Value {
  let mut v = json!(1);
  for _ in 0..depth {
    v = json!([v]);
  }
  v
}

fn coll_dir(db: &Path, name: &str) -> std::path::PathBuf {
  db.join("collections").join(name)
}

/// Review item 1: a payload deeper than JSON parsers accept used to be
/// logged and then fail to apply, poisoning the collection and making the
/// database unopenable.
#[test]
fn deeply_nested_payload_is_rejected_before_logging() {
  let dir = TestDir::new("deep");
  let opts = Options::default().durability(Durability::NoSync);
  {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    let with = db.create_collection("f", schema()).unwrap();
    let without = db
      .create_collection(
        "p",
        Schema::vector(VectorField::flat(DIM, Metric::L2)),
      )
      .unwrap();
    for c in [&with, &without] {
      let deep = Document::new(1u64)
        .vector(vec![0.0; DIM])
        .payload(json!({"tag": "x", "deep": nested(200)}));
      match c.upsert(&[deep]) {
        Err(Error::InvalidArgument(_)) => {}
        other => {
          panic!("deep payload: expected InvalidArgument, got {other:?}")
        }
      }
      // Nesting within the limit works end to end.
      let ok = Document::new(2u64)
        .vector(vec![0.0; DIM])
        .payload(json!({"tag": "x", "deep": nested(60)}));
      c.upsert(&[ok]).unwrap();
      let hits = c
        .search(&Query::vector(vec![0.0; DIM]).with_payload(true))
        .unwrap();
      assert_eq!(hits.len(), 1);
      assert_eq!(hits[0].payload.as_ref().unwrap()["deep"], nested(60));
    }
  }
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.collection("f").unwrap();
  let q = Query::vector(vec![0.0; DIM]).filter(Filter::eq("tag", "x"));
  assert_eq!(c.search(&q).unwrap().len(), 1);
  assert!(c.get(&Id::U64(2)).unwrap().is_some());
  c.upsert(&[doc(3, 1)]).unwrap();
}

/// Review item 1 (second part): one unreadable collection must not keep
/// the database, and every other collection, from opening.
#[test]
fn one_broken_collection_does_not_block_the_others() {
  let dir = TestDir::new("isolation");
  let opts = Options::default().durability(Durability::NoSync);
  {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    db.create_collection("a", schema())
      .unwrap()
      .upsert(&[doc(1, 1)])
      .unwrap();
    db.create_collection("b", schema())
      .unwrap()
      .upsert(&[doc(1, 1)])
      .unwrap();
  }
  for m in files_with_prefix(&coll_dir(dir.path(), "b"), "MANIFEST-") {
    let mut bytes = std::fs::read(&m).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&m, bytes).unwrap();
  }
  let db = Database::open(dir.path(), opts).unwrap();
  assert_eq!(version(&db.collection("a").unwrap(), 1), Some(1));
  assert!(db.collection("b").is_err());
  let failures = db.open_failures();
  assert_eq!(failures.len(), 1, "{failures:?}");
  assert_eq!(failures[0].0, "b");
  assert_eq!(db.list_collections().unwrap(), vec!["a"]);
  assert!(db.create_collection("b", schema()).is_err());
}

/// Simulates a power cut: every WAL record above `durable` is lost.
fn lose_unsynced_wal(wal_dir: &Path, durable: u64) {
  for seg in files_with_prefix(wal_dir, "0") {
    let name = seg.file_name().unwrap().to_str().unwrap().to_string();
    if !name.ends_with(".wal") {
      continue;
    }
    let bytes = std::fs::read(&seg).unwrap();
    let first = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let (mut off, mut lsn) = (32usize, first);
    while off + 8 <= bytes.len() && lsn <= durable {
      let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
      off += 8 + len as usize;
      lsn += 1;
    }
    let f = std::fs::OpenOptions::new().write(true).open(&seg).unwrap();
    f.set_len(off.min(bytes.len()) as u64).unwrap();
  }
}

/// Review item 2: a flush persisted tombstones of writes the log had not
/// yet made durable. After a power cut the write was gone but its
/// tombstone stayed: the document vanished entirely.
#[test]
fn tombstones_are_never_more_durable_than_the_log() {
  let dir = TestDir::new("durable-tombs");
  let opts = Options {
    durability: Durability::NoSync,
    auto_compact: false,
    failpoint: Some(Failpoint::SlowFlush),
    ..Options::default()
  };
  let durable;
  {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    let c = db.create_collection("c", schema()).unwrap();
    c.upsert(&[doc(1, 1)]).unwrap();
    c.flush().unwrap(); // version 1 of doc 1 is in a segment
    c.upsert(&[doc(2, 1)]).unwrap();
    let flusher = {
      let c = c.clone();
      std::thread::spawn(move || c.flush())
    };
    // While the flush is paused (after rotating the log, before it
    // captures tombstones): replace doc 1. The record is written, not
    // synced; version 1 is now deleted in its segment.
    std::thread::sleep(Duration::from_millis(100));
    c.upsert(&[doc(1, 2)]).unwrap();
    flusher.join().unwrap().unwrap();
    durable = c.stats().unwrap().durable_lsn;
  }
  lose_unsynced_wal(&coll_dir(dir.path(), "c").join("wal"), durable);
  let opts = Options {
    failpoint: None,
    ..opts
  };
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.collection("c").unwrap();
  let v = version(&c, 1);
  assert!(
    v.is_some(),
    "doc 1 vanished: its replacement was lost but its tombstone kept"
  );
}

/// Review item 3: a failed full compaction was retried every 500 ms
/// forever, failing unrelated flushes, and left its files behind.
#[test]
fn failed_compaction_is_not_retried_and_leaves_no_files() {
  let dir = TestDir::new("compact-fail");
  let opts = Options {
    durability: Durability::NoSync,
    auto_compact: false,
    failpoint: Some(Failpoint::FailCompaction),
    ..Options::default()
  };
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.create_collection("c", schema()).unwrap();
  for i in 0..3 {
    c.upsert(&[doc(i, 1)]).unwrap();
    c.flush().unwrap();
  }
  assert!(c.compact().is_err());
  let live: std::collections::HashSet<u64> =
    c.stats().unwrap().segments.iter().map(|s| s.id).collect();
  for f in files_with_prefix(&coll_dir(dir.path(), "c"), "seg-") {
    let name = f.file_name().unwrap().to_str().unwrap();
    let id: u64 = name[4..14].parse().unwrap();
    assert!(live.contains(&id), "leftover file {name} of a failed merge");
  }
  // No retries in the background: flushes keep succeeding.
  let until = Instant::now() + Duration::from_millis(1500);
  let mut n = 100;
  while Instant::now() < until {
    c.upsert(&[doc(n, 1)]).unwrap();
    c.flush().unwrap();
    n += 1;
  }
}

/// Review item 4: a panic in the worker left flush() and writers waiting
/// forever.
#[test]
fn worker_panic_fails_waiters_instead_of_hanging() {
  let dir = TestDir::new("panic");
  let opts = Options {
    durability: Durability::NoSync,
    failpoint: Some(Failpoint::PanicInFlush),
    ..Options::default()
  };
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.create_collection("c", schema()).unwrap();
  c.upsert(&[doc(1, 1)]).unwrap();
  let (tx, rx) = mpsc::channel();
  let c2 = c.clone();
  std::thread::spawn(move || {
    let _ = tx.send(c2.flush().is_err());
  });
  match rx.recv_timeout(Duration::from_secs(20)) {
    Ok(failed) => assert!(failed, "flush must report the worker's panic"),
    Err(_) => panic!("flush() hung after the worker panicked"),
  }
  assert!(
    c.upsert(&[doc(2, 1)]).is_err(),
    "writes refused after a panic"
  );
}

/// Review item 5: an error at or after the manifest commit was retried as
/// if nothing had been committed.
#[test]
fn error_after_manifest_commit_stops_writes_and_recovers() {
  let dir = TestDir::new("after-commit");
  let opts = Options {
    durability: Durability::NoSync,
    failpoint: Some(Failpoint::FailAfterCommit),
    ..Options::default()
  };
  {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    let c = db.create_collection("c", schema()).unwrap();
    c.upsert(&[doc(1, 1), doc(2, 1)]).unwrap();
    assert!(c.flush().is_err());
    assert!(
      c.upsert(&[doc(3, 1)]).is_err(),
      "state after a failed commit is unknown: writes must stop"
    );
  }
  let opts = Options {
    failpoint: None,
    ..opts
  };
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.collection("c").unwrap();
  assert_eq!((version(&c, 1), version(&c, 2)), (Some(1), Some(1)));
  assert_eq!(c.stats().unwrap().live_docs, 2);
}

/// Review item 6: finite vectors so large that every distance overflows
/// to infinity could never be found.
#[test]
fn vectors_that_overflow_distances_are_rejected() {
  let dir = TestDir::new("huge");
  let db = Database::open(dir.path(), Options::default()).unwrap();
  let c = db.create_collection("c", schema()).unwrap();
  let huge = Document::new(1u64).vector(vec![1e30; DIM]);
  match c.upsert(&[huge]) {
    Err(Error::InvalidArgument(_)) => {}
    other => panic!("expected InvalidArgument, got {other:?}"),
  }
  assert!(c.search(&Query::vector(vec![1e30; DIM])).is_err());
  c.upsert(&[Document::new(2u64).vector(vec![1e15; DIM])])
    .unwrap();
  let hits = c.search(&Query::vector(vec![1e15; DIM])).unwrap();
  assert_eq!(hits.len(), 1);
}

/// Review item 7: recovery that cut the log in the middle (setting later
/// files aside) silently dropped acknowledged writes.
#[test]
fn damaged_log_refuses_to_open_instead_of_losing_writes() {
  let dir = TestDir::new("wal-damage");
  let opts = Options {
    durability: Durability::SyncEachWrite,
    wal_segment_size: 4096,
    ..Options::default()
  };
  {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    let c = db.create_collection("c", schema()).unwrap();
    db.create_collection("ok", schema())
      .unwrap()
      .upsert(&[doc(1, 1)])
      .unwrap();
    for i in 0..200 {
      c.upsert(&[doc(i, 1)]).unwrap();
    }
  }
  let wal = coll_dir(dir.path(), "c").join("wal");
  let segs = files_with_prefix(&wal, "0");
  assert!(segs.len() > 3, "test needs several log files: {segs:?}");
  // Flip a byte inside the first record of the second file.
  let mut bytes = std::fs::read(&segs[1]).unwrap();
  bytes[32 + 12] ^= 0xff;
  std::fs::write(&segs[1], bytes).unwrap();
  for _ in 0..2 {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    assert!(
      db.collection("c").is_err(),
      "damaged collection must not open"
    );
    assert_eq!(version(&db.collection("ok").unwrap(), 1), Some(1));
  }
}
