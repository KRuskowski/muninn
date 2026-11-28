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
