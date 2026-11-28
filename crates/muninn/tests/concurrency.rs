//! Readers searching while writers upsert and delete and flushes and
//! compactions run: no panics, no duplicates, and every result a
//! consistent snapshot.
//!
//! Each writer owns groups of documents and always rewrites (or deletes)
//! a whole group in one batch, stamping the batch number in the payload,
//! the text and the vector. A consistent snapshot therefore shows every
//! group either entirely absent or entirely at one batch number; a torn
//! batch, a lost delete or a document seen twice breaks that.

mod common;

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use common::{Rng, TestDir};
use muninn::{
  Collection, Database, Document, Durability, Failpoint, Id, IdFilter, Metric,
  Options, Query, Schema, SearchHit, VectorField,
};
use serde_json::json;

const DIM: usize = 8;
const WRITERS: u64 = 3;
const GROUPS: u64 = 4;
const SIZE: u64 = 16;

fn group_ids(w: u64, g: u64) -> Vec<u64> {
  let base = (w * GROUPS + g) * SIZE;
  (base..base + SIZE).collect()
}

fn doc(id: u64, w: u64, v: u64) -> Document {
  let mut vec = vec![(v % 97) as f32; DIM];
  vec[0] = id as f32;
  Document::new(id)
    .vector(vec)
    .text("body", format!("writer{w} batch{v} common"))
    .payload(json!({"v": v}))
}

/// Checks hits restricted to one group: all or nothing, one batch, no
/// duplicates.
fn check_group(hits: &[SearchHit], what: &str) {
  let ids: HashSet<&Id> = hits.iter().map(|h| &h.id).collect();
  assert_eq!(ids.len(), hits.len(), "{what}: duplicate ids {hits:?}");
  assert!(
    hits.is_empty() || hits.len() == SIZE as usize,
    "{what}: torn group of {} docs",
    hits.len()
  );
  let vs: HashSet<u64> = hits
    .iter()
    .map(|h| h.payload.as_ref().unwrap()["v"].as_u64().unwrap())
    .collect();
  assert!(vs.len() <= 1, "{what}: mixed batches {vs:?}");
}

fn reader(c: &Collection, rng: &mut Rng, doomed: &Mutex<Vec<u64>>) {
  let w = rng.below(WRITERS);
  let g = rng.below(GROUPS);
  let ids = group_ids(w, g);
  let f = IdFilter::ids(ids.iter().copied());
  let k = SIZE as usize * 2;
  let qv: Vec<f32> = (0..DIM).map(|_| rng.f32() * 50.0).collect();
  let queries = [
    ("vector", Query::vector(qv.clone())),
    ("text", Query::text(format!("writer{w} common"))),
    ("hybrid", Query::hybrid(qv, "common")),
  ];
  for (what, q) in queries {
    let hits = c
      .search(&q.k(k).id_filter(f.clone()).with_payload(true))
      .unwrap();
    check_group(&hits, what);
  }
  // Whole-collection queries: no duplicates, and every group seen is
  // whole within the result's own snapshot.
  let hits = c
    .search(
      &Query::vector(vec![0.0; DIM])
        .k((WRITERS * GROUPS * SIZE) as usize + 5)
        .id_filter(IdFilter::predicate(|id| !is_loose(id)))
        .with_payload(true),
    )
    .unwrap();
  let ids: HashSet<&Id> = hits.iter().map(|h| &h.id).collect();
  assert_eq!(ids.len(), hits.len(), "full scan duplicates");
  assert_eq!(hits.len() % SIZE as usize, 0, "full scan torn group");
  let id = Id::U64(rng.below(WRITERS * GROUPS * SIZE));
  let _ = c.get(&id).unwrap();
  // Loose documents: each id at most once. A delete lost while a flush
  // or merge swaps sources would resurrect an old version next to the
  // new one.
  let loose = c
    .search(
      &Query::vector(vec![0.0; DIM])
        .k(LOOSE as usize + 10)
        .id_filter(IdFilter::predicate(is_loose)),
    )
    .unwrap();
  let ids: HashSet<&Id> = loose.iter().map(|h| &h.id).collect();
  assert_eq!(ids.len(), loose.len(), "loose duplicates");
  // Documents whose delete was acknowledged never come back, whatever
  // flushes and merges did meanwhile.
  let gone: Vec<u64> = doomed.lock().unwrap().clone();
  if !gone.is_empty() {
    let back = c
      .search(
        &Query::vector(vec![0.0; DIM])
          .k(gone.len() + 1)
          .id_filter(IdFilter::ids(gone.iter().copied())),
      )
      .unwrap();
    assert!(back.is_empty(), "deleted documents resurrected: {back:?}");
  }
}

/// Ids of documents inserted once and then deleted for good.
const DOOMED_BASE: u64 = 2_000_000;

/// Ids of the loose documents (outside every group).
const LOOSE_BASE: u64 = 1_000_000;
const LOOSE: u64 = 2000;

fn is_loose(id: muninn::IdRef<'_>) -> bool {
  matches!(id, muninn::IdRef::U64(v) if v >= LOOSE_BASE)
}

#[test]
fn readers_writers_flushes_and_merges_run_together() {
  let dir = TestDir::new("concurrency");
  let opts = Options {
    durability: Durability::NoSync,
    memtable_max_docs: 400,
    merge_factor: 2,
    hnsw_min_docs: 64,
    // Widens the windows in which a flush or merge races with deletes.
    failpoint: Some(Failpoint::SlowPublish),
    // Room for writes to continue while flushes run (otherwise
    // backpressure stops every writer during a flush and the races above
    // cannot happen).
    max_pending_flushes: 16,
    ..Options::default()
  };
  let db = Database::open(dir.path(), opts.clone()).unwrap();
  let schema = Schema::vector(VectorField::hnsw(DIM, Metric::L2))
    .with_text_fields(["body"]);
  let c = db.create_collection("c", schema).unwrap();
  let stop = Arc::new(AtomicBool::new(false));
  let reads = Arc::new(AtomicU64::new(0));
  let maint_runs = Arc::new(AtomicU64::new(0));
  let mut threads = Vec::new();
  for w in 0..WRITERS {
    let (c, reads, maint_runs) = (c.clone(), reads.clone(), maint_runs.clone());
    threads.push(std::thread::spawn(move || {
      let mut rng = Rng::new(100 + w);
      let mut v = 0u64;
      // At least 200 batches each, and until flushes, merges and reads
      // have interleaved with the writes for a while.
      while v < 200
        || maint_runs.load(Ordering::Relaxed) < 8
        || reads.load(Ordering::Relaxed) < 150
      {
        v += 1;
        let g = rng.below(GROUPS);
        let ids = group_ids(w, g);
        if rng.chance(0.3) {
          let batch: Vec<Document> = (0..20)
            .map(|_| doc(LOOSE_BASE + rng.below(LOOSE), w, v))
            .collect();
          c.upsert(&batch).unwrap();
        }
        if rng.chance(0.15) {
          let del: Vec<Id> = ids.iter().map(|&i| Id::U64(i)).collect();
          c.delete(&del).unwrap();
        } else {
          let batch: Vec<Document> =
            ids.iter().map(|&i| doc(i, w, v)).collect();
          c.upsert(&batch).unwrap();
        }
      }
    }));
  }
  let deleter = {
    let (c, stop) = (c.clone(), stop.clone());
    std::thread::spawn(move || {
      let mut rng = Rng::new(55);
      while !stop.load(Ordering::Relaxed) {
        let del: Vec<Id> = (0..10)
          .map(|_| Id::U64(LOOSE_BASE + rng.below(LOOSE)))
          .collect();
        c.delete(&del).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
      }
    })
  };
  let doomed = Arc::new(Mutex::new(Vec::<u64>::new()));
  let doomers: Vec<_> = (0..2u64)
    .map(|t| {
      let (c, stop, doomed) = (c.clone(), stop.clone(), doomed.clone());
      std::thread::spawn(move || {
        let mut next = DOOMED_BASE + t * 100_000_000;
        let mut rng = Rng::new(66 + t);
        while !stop.load(Ordering::Relaxed) {
          let ids: Vec<u64> = (next..next + 20).collect();
          next += 20;
          let docs: Vec<Document> = ids.iter().map(|&i| doc(i, 9, 1)).collect();
          c.upsert(&docs).unwrap();
          // Let a flush freeze them, then delete them while it runs.
          std::thread::sleep(std::time::Duration::from_millis(rng.below(15)));
          for chunk in ids.chunks(5) {
            let del: Vec<Id> = chunk.iter().map(|&i| Id::U64(i)).collect();
            c.delete(&del).unwrap();
            doomed.lock().unwrap().extend_from_slice(chunk);
            std::thread::sleep(std::time::Duration::from_millis(rng.below(4)));
          }
        }
      })
    })
    .collect();
  let mut readers = Vec::new();
  for r in 0..3 {
    let (c, stop, reads, doomed) =
      (c.clone(), stop.clone(), reads.clone(), doomed.clone());
    readers.push(std::thread::spawn(move || {
      let mut rng = Rng::new(900 + r);
      while !stop.load(Ordering::Relaxed) {
        reader(&c, &mut rng, &doomed);
        reads.fetch_add(1, Ordering::Relaxed);
      }
    }));
  }
  let maint = {
    let (c, stop, runs) = (c.clone(), stop.clone(), maint_runs.clone());
    std::thread::spawn(move || {
      let mut n = 0;
      while !stop.load(Ordering::Relaxed) {
        if n % 3 == 2 {
          c.compact().unwrap();
        } else {
          c.flush().unwrap();
        }
        n += 1;
        runs.store(n, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(40));
      }
      n
    })
  };
  for t in threads {
    t.join().unwrap();
  }
  stop.store(true, Ordering::Relaxed);
  for t in readers {
    t.join().unwrap();
  }
  deleter.join().unwrap();
  for t in doomers {
    t.join().unwrap();
  }
  let maintenance = maint.join().unwrap();
  assert!(
    reads.load(Ordering::Relaxed) >= 150,
    "readers made progress"
  );
  assert!(maintenance >= 8, "maintenance ran");

  // Final state survives a reopen unchanged.
  let mut rng = Rng::new(7);
  let before: Vec<Vec<SearchHit>> = (0..WRITERS)
    .flat_map(|w| (0..GROUPS).map(move |g| (w, g)))
    .map(|(w, g)| {
      let f = IdFilter::ids(group_ids(w, g));
      let hits = c
        .search(
          &Query::vector(vec![0.0; DIM])
            .k(100)
            .id_filter(f)
            .with_payload(true),
        )
        .unwrap();
      check_group(&hits, "final");
      hits
    })
    .collect();
  reader(&c, &mut rng, &doomed);
  drop(c);
  drop(db);
  let db = Database::open(dir.path(), opts).unwrap();
  let c = db.collection("c").unwrap();
  reader(&c, &mut rng, &doomed);
  let after: Vec<Vec<SearchHit>> = (0..WRITERS)
    .flat_map(|w| (0..GROUPS).map(move |g| (w, g)))
    .map(|(w, g)| {
      let f = IdFilter::ids(group_ids(w, g));
      c.search(
        &Query::vector(vec![0.0; DIM])
          .k(100)
          .id_filter(f)
          .with_payload(true),
      )
      .unwrap()
    })
    .collect();
  let ids = |v: &Vec<Vec<SearchHit>>| -> Vec<Vec<(Id, serde_json::Value)>> {
    v.iter()
      .map(|g| {
        let mut x: Vec<_> = g
          .iter()
          .map(|h| (h.id.clone(), h.payload.clone().unwrap()))
          .collect();
        x.sort_by(|a, b| a.0.cmp(&b.0));
        x
      })
      .collect()
  };
  assert_eq!(ids(&before), ids(&after));
}
