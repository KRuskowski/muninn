//! Database-level behaviour: collections are created, listed, dropped
//! and reopened with their schemas; the directory is locked; leftovers of
//! interrupted operations are cleaned up.

mod common;

use common::TestDir;
use muninn::{
  AnalyzerConfig, Database, Document, Error, FieldSpec, Id, Metric, Options,
  Quantization, Query, Schema, VectorField,
};
use serde_json::json;

fn schema() -> Schema {
  Schema::vector(
    VectorField::hnsw(4, Metric::Cosine).with_quantization(Quantization::Sq8),
  )
  .with_text_fields(["title"])
  .with_analyzer(AnalyzerConfig::english())
  .with_filterable(FieldSpec::keyword("lang"))
}

#[test]
fn collections_are_created_listed_dropped_and_reopened() {
  let dir = TestDir::new("db");
  {
    let db = Database::open(dir.path(), Options::default()).unwrap();
    let a = db.create_collection("a", schema()).unwrap();
    db.create_collection("b", Schema::text(["body"])).unwrap();
    assert!(matches!(
      db.create_collection("a", schema()),
      Err(Error::AlreadyExists(_))
    ));
    assert!(db.create_collection("bad name", schema()).is_err());
    assert_eq!(db.list_collections().unwrap(), vec!["a", "b"]);
    a.upsert(&[Document::new("x")
      .vector(vec![1.0, 0.0, 0.0, 0.0])
      .text("title", "Running databases")
      .payload(json!({"lang": "en"}))])
      .unwrap();
    // Dropped collections disappear, also for handles still held.
    let b = db.collection("b").unwrap();
    db.drop_collection("b").unwrap();
    assert!(matches!(b.get(&Id::U64(1)), Err(Error::NotFound(_))));
    assert!(matches!(db.collection("b"), Err(Error::NotFound(_))));
    // The name can be reused.
    db.create_collection("b", Schema::text(["other"])).unwrap();
    // A second open of the same directory is refused.
    assert!(Database::open(dir.path(), Options::default()).is_err());
  }
  // Leftovers of an interrupted create are removed at open.
  let leftover = dir.path().join("collections").join("c.123.0.creating");
  std::fs::create_dir_all(&leftover).unwrap();
  let db = Database::open(dir.path(), Options::default()).unwrap();
  assert!(!leftover.exists());
  assert_eq!(db.list_collections().unwrap(), vec!["a", "b"]);
  let a = db.collection("a").unwrap();
  assert_eq!(a.schema(), &schema());
  assert_eq!(db.collection("b").unwrap().schema().text_fields, ["other"]);
  // The stored analyzer still stems: "run" finds "Running".
  let hits = a
    .search(&Query::text("run").filter(muninn::Filter::eq("lang", "en")))
    .unwrap();
  assert_eq!(hits.len(), 1);
  let d = a.get(&Id::from("x")).unwrap().unwrap();
  assert_eq!(d.payload, json!({"lang": "en"}));
}

#[test]
fn invalid_requests_are_rejected() {
  let dir = TestDir::new("db-invalid");
  let db = Database::open(dir.path(), Options::default()).unwrap();
  let c = db.create_collection("c", schema()).unwrap();
  let bad = [
    Document::new(1u64),
    Document::new(1u64).vector(vec![0.0; 3]),
    Document::new(1u64).vector(vec![0.0; 4]).text("nope", "x"),
  ];
  for d in bad {
    assert!(c.upsert(&[d]).is_err());
  }
  assert!(c.search(&Query::vector(vec![0.0; 3])).is_err());
  assert!(c.search(&Query::text("x").fields(["nope"])).is_err());
  assert!(c.search(&Query::vector(vec![f32::NAN; 4])).is_err());
  // Nothing was written by the rejected calls.
  assert_eq!(c.stats().unwrap().last_lsn, 0);
  assert!(
    c.search(&Query::vector(vec![0.0; 4]).k(0))
      .unwrap()
      .is_empty()
  );
}

/// Directories under `collections/` other than open collections.
fn leftovers(db: &std::path::Path, open: &[&str]) -> Vec<String> {
  let mut v: Vec<String> = std::fs::read_dir(db.join("collections"))
    .unwrap()
    .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
    .filter(|n| !open.contains(&n.as_str()))
    .collect();
  v.sort();
  v
}

/// Dropping must work while the collection is in use (a held handle,
/// searches running on segments and memtable): on Windows its files
/// cannot be deleted while open or mapped, so the directory is set aside
/// and removed once released.
#[test]
fn dropping_a_collection_in_use_frees_its_name_and_files() {
  let dir = TestDir::new("drop-in-use");
  let opts = Options::default().durability(muninn::Durability::NoSync);
  {
    let db = Database::open(dir.path(), opts.clone()).unwrap();
    let c = db.create_collection("x", schema()).unwrap();
    let docs: Vec<Document> = (0..200u64)
      .map(|i| {
        Document::new(i)
          .vector(vec![i as f32, 1.0, 0.0, 0.0])
          .text("title", format!("doc {i}"))
          .payload(json!({"lang": "en"}))
      })
      .collect();
    c.upsert(&docs[..150]).unwrap();
    c.flush().unwrap(); // a mapped segment
    c.upsert(&docs[150..]).unwrap(); // memtable and WAL
    let held = db.collection("x").unwrap();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let searcher = {
      let (h, stop) = (held.clone(), stop.clone());
      std::thread::spawn(move || {
        let mut ok = 0;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
          match h.search(&Query::hybrid(vec![1.0, 1.0, 0.0, 0.0], "doc").k(5)) {
            Ok(hits) => {
              assert!(!hits.is_empty());
              ok += 1;
            }
            Err(Error::NotFound(_)) => break,
            Err(e) => panic!("unexpected error {e:?}"),
          }
        }
        ok
      })
    };
    std::thread::sleep(std::time::Duration::from_millis(50));
    db.drop_collection("x").unwrap();
    assert!(matches!(db.collection("x"), Err(Error::NotFound(_))));
    assert!(matches!(held.get(&Id::U64(1)), Err(Error::NotFound(_))));
    assert!(held.upsert(&docs[..1]).is_err());
    // The name is free at once, for an unrelated new collection.
    let fresh = db.create_collection("x", Schema::text(["body"])).unwrap();
    assert_eq!(fresh.stats().unwrap().live_docs, 0);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(searcher.join().unwrap() > 0, "searches ran before the drop");
    drop(held);
    drop(c);
    assert_eq!(
      leftovers(dir.path(), &["x"]),
      Vec::<String>::new(),
      "the dropped collection's files are deleted once released"
    );
  }
  let db = Database::open(dir.path(), opts).unwrap();
  assert_eq!(db.list_collections().unwrap(), vec!["x"]);
  assert_eq!(db.collection("x").unwrap().schema().text_fields, ["body"]);
  assert_eq!(leftovers(dir.path(), &["x"]), Vec::<String>::new());
}

/// Leftovers of drops whose files were still in use (renamed aside, or
/// marked in place) are deleted when the database opens.
#[test]
fn leftovers_of_drops_are_swept_at_open() {
  let dir = TestDir::new("sweep");
  {
    let db = Database::open(dir.path(), Options::default()).unwrap();
    db.create_collection("keep", schema()).unwrap();
    db.create_collection("marked", schema()).unwrap();
  }
  let cdir = dir.path().join("collections");
  let aside = cdir.join(".dropped-old-1-2");
  std::fs::create_dir_all(aside.join("wal")).unwrap();
  std::fs::write(aside.join("wal").join("x.wal"), b"junk").unwrap();
  std::fs::write(cdir.join("marked").join("DROPPED"), b"").unwrap();
  let db = Database::open(dir.path(), Options::default()).unwrap();
  assert_eq!(db.list_collections().unwrap(), vec!["keep"]);
  assert_eq!(leftovers(dir.path(), &["keep"]), Vec::<String>::new());
  // The marked name is free again.
  db.create_collection("marked", schema()).unwrap();
}

#[test]
fn delete_counted_reports_documents_that_existed() {
  let dir = TestDir::new("delete-count");
  let db = Database::open(dir.path(), Options::default()).unwrap();
  let c = db.create_collection("c", Schema::text(["body"])).unwrap();
  let doc = |id: u64| Document::new(id).text("body", format!("doc {id}"));
  c.upsert(&[doc(1), doc(2), doc(3)]).unwrap();
  c.flush().unwrap();
  c.upsert(&[doc(4), doc(1)]).unwrap();
  let ids = |v: &[u64]| v.iter().map(|&i| Id::U64(i)).collect::<Vec<_>>();
  // 1 (replaced, in memory), 2 (in a segment), 9 missing, 1 again.
  assert_eq!(c.delete_counted(&ids(&[1, 2, 9, 1])).unwrap(), 2);
  assert_eq!(c.delete_counted(&ids(&[1, 2])).unwrap(), 0);
  assert_eq!(c.delete_counted(&[]).unwrap(), 0);
  c.delete(&ids(&[3])).unwrap();
  assert_eq!(c.stats().unwrap().live_docs, 1);
}

#[test]
fn schema_json_round_trips_through_the_public_api() {
  let s = schema();
  assert_eq!(Schema::from_json(&s.to_json()).unwrap(), s);
  assert!(matches!(
    Schema::from_json(&json!({"text_fields": "body"})),
    Err(Error::InvalidArgument(_))
  ));
}
