//! Crash recovery: every acknowledged write survives (per durability
//! mode) and nothing half-applied ever appears.
//!
//! Crashes are simulated by dropping the database without flushing, by
//! cutting the write-ahead log's tail, by fault injection at the exact
//! points between writing a segment, committing the manifest and
//! truncating the log, by corrupting the newest manifest, and by copying
//! the data directory while writers are running (the files exactly as a
//! killed process leaves them).

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use common::{Rng, TestDir, copy_dir, files_with_prefix};
use muninn::{
  Collection, Database, Document, Durability, Failpoint, Id, Metric, Options,
  Query, Schema, VectorField,
};
use serde_json::json;

const DIM: usize = 8;

fn schema() -> Schema {
  Schema::vector(VectorField::flat(DIM, Metric::L2)).with_text_fields(["body"])
}

fn doc(id: u64, version: u64) -> Document {
  let v: Vec<f32> = (0..DIM)
    .map(|i| ((id * 7 + i as u64) % 13) as f32)
    .collect();
  Document::new(id)
    .vector(v)
    .text("body", format!("doc{id} version{version}"))
    .payload(json!({"v": version}))
}

/// Expected state: id -> version.
type Expect = BTreeMap<u64, u64>;

fn version_of(c: &Collection, id: u64) -> Option<u64> {
  c.get(&Id::U64(id))
    .unwrap()
    .map(|d| d.payload["v"].as_u64().unwrap())
}

/// Checks `c` holds exactly `want`, by `get`, by stats and by a full
/// search (no duplicates).
fn verify(c: &Collection, want: &Expect, ctx: &str) {
  for (id, v) in want {
    assert_eq!(version_of(c, *id), Some(*v), "{ctx}: doc {id}");
  }
  let st = c.stats().unwrap();
  assert_eq!(st.live_docs, want.len() as u64, "{ctx}: live count");
  let hits = c
    .search(
      &Query::vector(vec![0.0; DIM])
        .k(want.len() + 10)
        .with_payload(true),
    )
    .unwrap();
  assert_eq!(hits.len(), want.len(), "{ctx}: search count");
  for h in &hits {
    let Id::U64(id) = h.id else {
      panic!("string id")
    };
    assert_eq!(
      h.payload.as_ref().unwrap()["v"].as_u64(),
      want.get(&id).copied(),
      "{ctx}: hit {id}"
    );
  }
}

fn coll_dir(db: &std::path::Path) -> std::path::PathBuf {
  db.join("collections").join("c")
}

#[test]
fn acknowledged_writes_survive_reopen_in_every_durability_mode() {
  let modes = [
    Durability::SyncEachWrite,
    Durability::SyncBatch {
      max_delay: Duration::from_millis(5),
    },
    Durability::NoSync,
  ];
  for (mi, mode) in modes.into_iter().enumerate() {
    for max_docs in [37, 1_000_000] {
      let dir = TestDir::new("durability");
      let opts = Options {
        durability: mode,
        memtable_max_docs: max_docs,
        merge_factor: 2,
        ..Options::default()
      };
      let mut want = Expect::new();
      let mut rng = Rng::new(mi as u64 * 31 + max_docs as u64);
      {
        let db = Database::open(dir.path(), opts.clone()).unwrap();
        let c = db.create_collection("c", schema()).unwrap();
        for v in 1..=60u64 {
          let batch: Vec<Document> =
            (0..8).map(|_| doc(rng.below(150), v)).collect();
          c.upsert(&batch).unwrap();
          for d in &batch {
            let Id::U64(id) = d.id else { unreachable!() };
            want.insert(id, v);
          }
          if v % 7 == 0 {
            let ids: Vec<Id> =
              (0..3).map(|_| Id::U64(rng.below(150))).collect();
            c.delete(&ids).unwrap();
            for id in &ids {
              let Id::U64(id) = id else { unreachable!() };
              want.remove(id);
            }
          }
        }
        verify(&c, &want, "before reopen");
        // Dropped without flush or close.
      }
      let db = Database::open(dir.path(), opts.clone()).unwrap();
      let c = db.collection("c").unwrap();
      verify(&c, &want, &format!("{mode:?} max_docs {max_docs}"));
      c.flush().unwrap();
      drop(c);
      drop(db);
      let db = Database::open(dir.path(), opts).unwrap();
      verify(
        &db.collection("c").unwrap(),
        &want,
        "after flush and reopen",
      );
    }
  }
}

#[test]
fn torn_wal_tail_loses_only_the_unfinished_batch() {
  for cut in [1u64, 7, 40] {
    let dir = TestDir::new("torn");
    let opts = Options::default().durability(Durability::NoSync);
    let mut want = Expect::new();
    let last: Vec<u64>;
    {
      let db = Database::open(dir.path(), opts.clone()).unwrap();
      let c = db.create_collection("c", schema()).unwrap();
      for b in 0..20u64 {
        let batch: Vec<Document> =
          (0..10).map(|j| doc(b * 10 + j, b)).collect();
        c.upsert(&batch).unwrap();
        if b < 19 {
          for j in 0..10 {
            want.insert(b * 10 + j, b);
          }
        }
      }
      // The last batch also replaces documents of batch 0: after the cut
      // none of the replacements may show.
      let upd: Vec<Document> = (0..5).map(|j| doc(j, 99)).collect();
      c.upsert(&upd).unwrap();
      last = (190..200).collect();
    }
    // Cut into the final record (the 5-document update).
    let wal = files_with_prefix(&coll_dir(dir.path()).join("wal"), "0");
    let seg = wal.last().unwrap();
    let len = std::fs::metadata(seg).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(seg).unwrap();
    f.set_len(len - cut).unwrap();
    drop(f);
    for id in &last {
      want.insert(*id, 19);
    }
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    let c = db.collection("c").unwrap();
    verify(&c, &want, &format!("cut {cut}"));
    // The log is usable after the repair.
    c.upsert(&[doc(500, 1)]).unwrap();
    want.insert(500, 1);
    drop(c);
    drop(db);
    let db = Database::open(dir.path(), opts).unwrap();
    verify(&db.collection("c").unwrap(), &want, "after repair");
  }
}

/// Writes 120 docs (with replacements and deletes), returns the state.
fn fill(c: &Collection, base: u64) -> Expect {
  let mut want = Expect::new();
  for b in 0..12u64 {
    let batch: Vec<Document> =
      (0..10).map(|j| doc(base + b * 10 + j, b)).collect();
    c.upsert(&batch).unwrap();
    for j in 0..10 {
      want.insert(base + b * 10 + j, b);
    }
  }
  c.upsert(&[doc(base, 50), doc(base + 1, 50)]).unwrap();
  want.insert(base, 50);
  want.insert(base + 1, 50);
  c.delete(&[Id::U64(base + 2), Id::U64(base + 3)]).unwrap();
  want.remove(&(base + 2));
  want.remove(&(base + 3));
  want
}

fn with_failpoint(f: Failpoint) -> Options {
  Options {
    failpoint: Some(f),
    durability: Durability::NoSync,
    ..Options::default()
  }
}

#[test]
fn crash_between_segment_write_and_manifest_commit() {
  let dir = TestDir::new("fp-seg");
  let want;
  {
    let db =
      Database::open(dir.path(), with_failpoint(Failpoint::AfterSegmentWrite))
        .unwrap();
    let c = db.create_collection("c", schema()).unwrap();
    want = fill(&c, 0);
    assert!(c.flush().is_err(), "the failpoint must stop the flush");
    assert!(
      c.upsert(&[doc(9999, 1)]).is_err(),
      "writes refused after it"
    );
    verify(&c, &want, "after the failed flush");
  }
  let cdir = coll_dir(dir.path());
  assert!(
    !files_with_prefix(&cdir, "seg-").is_empty(),
    "the crash leaves orphaned segment files"
  );
  let opts = Options::default().durability(Durability::NoSync);
  let db = Database::open(dir.path(), opts.clone()).unwrap();
  let c = db.collection("c").unwrap();
  assert!(
    files_with_prefix(&cdir, "seg-").is_empty(),
    "orphans are collected at open"
  );
  verify(&c, &want, "after recovery");
  c.flush().unwrap();
  verify(&c, &want, "after flush");
  drop(c);
  drop(db);
  let db = Database::open(dir.path(), opts).unwrap();
  verify(&db.collection("c").unwrap(), &want, "after reopen");
}

#[test]
fn crash_between_manifest_commit_and_wal_truncation() {
  let dir = TestDir::new("fp-manifest");
  let want;
  {
    let db = Database::open(
      dir.path(),
      with_failpoint(Failpoint::AfterManifestCommit),
    )
    .unwrap();
    let c = db.create_collection("c", schema()).unwrap();
    want = fill(&c, 0);
    assert!(c.flush().is_err());
  }
  let opts = Options::default().durability(Durability::NoSync);
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.collection("c").unwrap();
  // The segment is committed; the log holds the same writes but they are
  // below the checkpoint and must not be applied twice.
  let st = c.stats().unwrap();
  assert_eq!(st.segments.len(), 1, "{st:?}");
  assert_eq!(st.memtable_docs, 0, "{st:?}");
  verify(&c, &want, "after recovery");
}

#[test]
fn crash_during_compaction_keeps_the_old_segments() {
  let dir = TestDir::new("fp-compact");
  let mut want;
  {
    let db = Database::open(
      dir.path(),
      with_failpoint(Failpoint::AfterCompactionWrite),
    )
    .unwrap();
    let c = db.create_collection("c", schema()).unwrap();
    want = fill(&c, 0);
    c.flush().unwrap();
    want.extend(fill(&c, 1000));
    c.flush().unwrap();
    assert!(c.compact().is_err());
    verify(&c, &want, "after the failed compaction");
  }
  let opts = Options::default().durability(Durability::NoSync);
  let db = Database::open(dir.path(), opts.clone()).unwrap();
  let c = db.collection("c").unwrap();
  assert_eq!(c.stats().unwrap().segments.len(), 2);
  verify(&c, &want, "after recovery");
  c.compact().unwrap();
  let st = c.stats().unwrap();
  assert_eq!((st.segments.len(), st.deleted_docs), (1, 0), "{st:?}");
  verify(&c, &want, "after compaction");
  drop(c);
  drop(db);
  let db = Database::open(dir.path(), opts).unwrap();
  verify(&db.collection("c").unwrap(), &want, "after reopen");
}

#[test]
fn corrupt_newest_manifest_falls_back_to_the_previous_one() {
  let dir = TestDir::new("manifest-corrupt");
  let opts = Options::default().durability(Durability::NoSync);
  let mut want;
  {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    let c = db.create_collection("c", schema()).unwrap();
    want = fill(&c, 0);
    c.flush().unwrap();
    want.extend(fill(&c, 1000));
    c.delete(&[Id::U64(5), Id::U64(6)]).unwrap();
    want.remove(&5);
    want.remove(&6);
    c.upsert(&[doc(7, 77)]).unwrap();
    want.insert(7, 77);
    c.flush().unwrap();
    // Unflushed writes after the newest checkpoint, too.
    c.upsert(&[doc(8, 88), doc(3000, 1)]).unwrap();
    want.insert(8, 88);
    want.insert(3000, 1);
  }
  let manifests = files_with_prefix(&coll_dir(dir.path()), "MANIFEST-");
  assert_eq!(manifests.len(), 2, "two generations are kept");
  let newest = manifests.last().unwrap();
  let mut bytes = std::fs::read(newest).unwrap();
  let mid = bytes.len() / 2;
  bytes[mid] ^= 0x55;
  std::fs::write(newest, bytes).unwrap();
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.collection("c").unwrap();
  assert_eq!(c.stats().unwrap().segments.len(), 1, "previous manifest");
  verify(&c, &want, "after falling back");
}

/// Copies the data directory while writers run: the copy is what a
/// killed process leaves behind. Every batch acknowledged before the copy
/// started must be in it, and every group of documents must be at one
/// version (a batch rewrites a whole group).
#[test]
fn snapshot_of_a_running_database_recovers_consistently() {
  const WRITERS: u64 = 3;
  const GROUPS: u64 = 6;
  const SIZE: u64 = 8;
  let dir = TestDir::new("live-copy");
  let opts = Options {
    durability: Durability::NoSync,
    memtable_max_docs: 10_000_000,
    ..Options::default()
  };
  let db = Database::open(dir.path(), opts.clone()).unwrap();
  let c = db.create_collection("c", schema()).unwrap();
  let acked: Arc<Vec<AtomicU64>> =
    Arc::new((0..WRITERS * GROUPS).map(|_| AtomicU64::new(0)).collect());
  let stop = Arc::new(AtomicBool::new(false));
  let mut threads = Vec::new();
  for w in 0..WRITERS {
    let (c, acked, stop) = (c.clone(), acked.clone(), stop.clone());
    threads.push(std::thread::spawn(move || {
      let mut rng = Rng::new(w);
      let mut v = 0;
      while !stop.load(Ordering::Relaxed) {
        v += 1;
        let g = rng.below(GROUPS);
        let base = (w * GROUPS + g) * SIZE;
        let batch: Vec<Document> =
          (0..SIZE).map(|j| doc(base + j, v)).collect();
        c.upsert(&batch).unwrap();
        acked[(w * GROUPS + g) as usize].store(v, Ordering::Release);
      }
    }));
  }
  for snap in 0..3 {
    std::thread::sleep(Duration::from_millis(30));
    let floor: Vec<u64> =
      acked.iter().map(|a| a.load(Ordering::Acquire)).collect();
    let copy = TestDir::new(&format!("live-copy-{snap}"));
    copy_dir(dir.path(), copy.path());
    let db2 = Database::open(copy.path(), opts.clone()).unwrap();
    let c2 = db2.collection("c").unwrap();
    for (gi, &min_v) in floor.iter().enumerate() {
      let base = gi as u64 * SIZE;
      let vs: Vec<Option<u64>> =
        (0..SIZE).map(|j| version_of(&c2, base + j)).collect();
      assert!(
        vs.iter().all(|x| *x == vs[0]),
        "snapshot {snap} group {gi}: half a batch {vs:?}"
      );
      assert!(
        vs[0].unwrap_or(0) >= min_v,
        "snapshot {snap} group {gi}: acknowledged version {min_v} lost, \
         found {vs:?}"
      );
    }
    let st = c2.stats().unwrap();
    let hits = c2.search(&Query::vector(vec![0.0; DIM]).k(10_000)).unwrap();
    assert_eq!(hits.len() as u64, st.live_docs, "snapshot {snap}");
  }
  stop.store(true, Ordering::Relaxed);
  for t in threads {
    t.join().unwrap();
  }
}
