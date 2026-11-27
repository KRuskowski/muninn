//! Frozen file format: round trips, empty indexes, merge argument
//! checks, and rejection of corrupt files without panicking.

mod common;

use common::{Rng, filter, payload, schema};
use muninn_core::container::{ContainerWriter, tag};
use muninn_core::{AlignedBuf, Error, Region};
use muninn_filter::{
  DELETED, FieldType, Filter, FrozenIndex, MemIndex, Schema,
};
use serde_json::json;

fn sample() -> (MemIndex, AlignedBuf) {
  let mut r = Rng::new(9);
  let mut m = MemIndex::new(schema());
  for id in 0..500 {
    m.add(id, &payload(&mut r)).unwrap();
  }
  let buf = m.write_to(AlignedBuf::new()).unwrap();
  (m, buf)
}

#[test]
fn round_trip_preserves_schema_and_results() {
  let (m, buf) = sample();
  let f = FrozenIndex::open(Region::new(buf.clone())).unwrap();
  f.verify_checksum().unwrap();
  assert_eq!(f.schema(), m.schema());
  let names: Vec<&str> = f.schema().fields().iter().map(|x| x.name()).collect();
  assert_eq!(names, ["kw", "tags", "n", "x", "b", "o.k", "o.v"]);
  assert_eq!(f.doc_count(), 500);
  let mut r = Rng::new(10);
  for _ in 0..200 {
    let q = filter(&mut r, 3, false);
    assert_eq!(f.evaluate(&q, 500).unwrap(), m.evaluate(&q, 500).unwrap());
    assert_eq!(f.estimate(&q, 500), m.estimate(&q, 500), "{q:?}");
  }
  // Writing is deterministic.
  assert!(m.write_to(AlignedBuf::new()).unwrap().as_slice() == buf.as_slice());
}

#[test]
fn empty_indexes() {
  // No fields at all.
  let m = MemIndex::new(Schema::new());
  let f =
    FrozenIndex::open(Region::new(m.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  assert_eq!(f.doc_count(), 0);
  assert!(f.schema().is_empty());
  assert_eq!(f.evaluate(&Filter::and([]), 5).unwrap().count(), 5);
  assert_eq!(f.evaluate(&Filter::or([]), 5).unwrap().count(), 0);
  assert!(f.evaluate(&Filter::exists("x"), 5).is_err());
  // Fields but no documents.
  let m = MemIndex::new(schema());
  let f =
    FrozenIndex::open(Region::new(m.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  for q in [
    Filter::eq("kw", "a"),
    Filter::range("n", 0..),
    Filter::exists("x"),
    Filter::eq("b", true),
  ] {
    assert_eq!(f.evaluate(&q, 0).unwrap().capacity(), 0);
    assert_eq!(f.evaluate(&q, 10).unwrap().count(), 0);
    assert_eq!(m.evaluate(&q, 10).unwrap().count(), 0);
    assert_eq!(f.evaluate(&!q.clone(), 10).unwrap().count(), 10);
    assert_eq!(f.estimate(&q, 10), 0.0);
    assert_eq!(f.estimate(&q, 0), 0.0);
  }
  // Docs with no filterable values still count.
  let mut m = MemIndex::new(schema());
  m.add(0, &json!({})).unwrap();
  m.add(9, &json!(null)).unwrap();
  let f =
    FrozenIndex::open(Region::new(m.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  assert_eq!(f.doc_count(), 10);
  assert_eq!(f.evaluate(&!Filter::exists("kw"), 10).unwrap().count(), 10);
}

#[test]
fn rejects_truncated_files() {
  let (_, buf) = sample();
  let bytes = buf.as_slice();
  for len in [0, 10, 64, 127, bytes.len() / 2, bytes.len() - 1] {
    assert!(
      FrozenIndex::open(Region::copy_from(&bytes[..len])).is_err(),
      "len {len}"
    );
  }
}

#[test]
fn rejects_other_kinds_and_versions() {
  let mut w = ContainerWriter::new(AlignedBuf::new(), tag("hnsw"), 1).unwrap();
  w.section(tag("meta"), &[0u8; 32]).unwrap();
  let other = w.finish().unwrap();
  let e = FrozenIndex::open(Region::new(other)).unwrap_err();
  assert!(matches!(e, Error::Corrupt(_)), "{e}");
  let w = ContainerWriter::new(AlignedBuf::new(), tag("pindex"), 99).unwrap();
  let e = FrozenIndex::open(Region::new(w.finish().unwrap())).unwrap_err();
  assert!(e.to_string().contains("version"), "{e}");
  // Right kind, missing sections.
  let w = ContainerWriter::new(AlignedBuf::new(), tag("pindex"), 1).unwrap();
  assert!(FrozenIndex::open(Region::new(w.finish().unwrap())).is_err());
}

/// Flips bytes across the file: every damaged file must either fail to
/// open or fail its checksum, and evaluating whatever opens must not
/// panic.
#[test]
fn corruption_is_detected_and_never_panics() {
  let (_, buf) = sample();
  let mut r = Rng::new(11);
  let queries: Vec<Filter> =
    (0..20).map(|_| filter(&mut r, 3, false)).collect();
  // The footer's zero padding is outside the checksum by design of the
  // container format, so damage is applied to everything before it.
  let len = buf.len() - 64;
  let mut opened = 0;
  let step = (len / 1500).max(1);
  for pos in (0..len).step_by(step) {
    for flip in [0x01u8, 0x80, 0xff] {
      let mut bad = buf.clone();
      bad.as_mut_slice()[pos] ^= flip;
      let Ok(ix) = FrozenIndex::open(Region::new(bad)) else {
        continue;
      };
      opened += 1;
      assert!(ix.verify_checksum().is_err(), "flip at {pos} undetected");
      for q in &queries {
        let _ = ix.evaluate(q, 500);
        let _ = ix.evaluate(q, 2000);
        let _ = ix.estimate(q, 500);
      }
    }
  }
  assert!(
    opened > 0,
    "no corrupt file passed open; test covers nothing"
  );
}

/// Damage aimed at the structures open() validates: each must be
/// rejected by open() itself, not only by the checksum.
#[test]
fn open_validates_tables() {
  let schema = Schema::new()
    .with("k", FieldType::Keyword)
    .unwrap()
    .with("i", FieldType::Integer)
    .unwrap();
  let mut m = MemIndex::new(schema);
  for id in 0..100u32 {
    let p = json!({"k": format!("t{}", id % 7), "i": id as i64});
    m.add(id, &p).unwrap();
  }
  let buf = m.write_to(AlignedBuf::new()).unwrap();
  let region = Region::new(buf.clone());
  let c = muninn_core::container::Container::open(region.clone()).unwrap();
  let start = |t: [u8; 8]| {
    let s = c.section(t).unwrap();
    s.bytes().as_ptr() as usize - region.bytes().as_ptr() as usize
  };
  let patch = |off: usize, v: u64| {
    let mut b = buf.clone();
    b.as_mut_slice()[off..off + 8].copy_from_slice(&v.to_le_bytes());
    FrozenIndex::open(Region::new(b))
  };
  let fields = start(tag("fields"));
  let meta = start(tag("meta"));
  let offs = start(*b"O0000\0\0\0");
  let descs = start(*b"D0000\0\0\0");
  let cases = [
    ("bad type code", fields, 77),
    ("name out of bounds", fields + 8, 1 << 40),
    ("exists posting out of bounds", fields + 24, 1 << 30),
    ("term count mismatch", fields + 72, 3),
    ("value count mismatch", fields + 96 + 72, 99),
    ("field count mismatch", meta + 8, 3),
    ("doc count too large", meta, 1 << 33),
    ("term offsets out of order", offs + 8, 1 << 20),
    ("term posting out of bounds", descs + 8, 1 << 31),
  ];
  for (what, off, v) in cases {
    assert!(patch(off, v).is_err(), "{what} not rejected");
  }
  // Terms "t0", "t1", ...: make the second equal the first.
  let mut b = buf.clone();
  let terms = start(*b"T0000\0\0\0");
  b.as_mut_slice()[terms + 3] = b'0';
  assert!(FrozenIndex::open(Region::new(b)).is_err(), "unsorted terms");
  // Two fields with one name.
  let mut b = buf.clone();
  let names = start(tag("names"));
  b.as_mut_slice()[names] = b'i';
  assert!(FrozenIndex::open(Region::new(b)).is_err(), "duplicate name");
}

#[test]
fn merge_checks_arguments() {
  let (_, buf) = sample();
  let f = FrozenIndex::open(Region::new(buf)).unwrap();
  let short = vec![0u32; 10];
  let e = FrozenIndex::merge(&schema(), &[(&f, short.as_slice())], Vec::new());
  assert!(e.unwrap_err().to_string().contains("remap covers 10 ids"));
  let retyped = Schema::new().with("kw", FieldType::Integer).unwrap();
  let all: Vec<u32> = (0..500).collect();
  let e = FrozenIndex::merge(&retyped, &[(&f, all.as_slice())], Vec::new());
  assert!(e.unwrap_err().to_string().contains("\"kw\" is keyword"));
  // A field the input lacks merges as empty; dropped fields vanish.
  let other = Schema::new()
    .with("kw", FieldType::Keyword)
    .unwrap()
    .with("new", FieldType::Bool)
    .unwrap();
  let out =
    FrozenIndex::merge(&other, &[(&f, all.as_slice())], AlignedBuf::new())
      .unwrap();
  let g = FrozenIndex::open(Region::new(out)).unwrap();
  assert_eq!(g.schema(), &other);
  assert_eq!(g.evaluate(&Filter::exists("new"), 500).unwrap().count(), 0);
  assert_eq!(
    g.evaluate(&Filter::eq("kw", "a"), 500).unwrap(),
    f.evaluate(&Filter::eq("kw", "a"), 500).unwrap()
  );
  assert!(g.evaluate(&Filter::exists("n"), 500).is_err());
  // Everything deleted.
  let gone = vec![DELETED; 500];
  let out =
    FrozenIndex::merge(&schema(), &[(&f, gone.as_slice())], AlignedBuf::new())
      .unwrap();
  let g = FrozenIndex::open(Region::new(out)).unwrap();
  assert_eq!(g.doc_count(), 0);
  assert_eq!(g.evaluate(&Filter::exists("kw"), 3).unwrap().count(), 0);
  // No inputs at all.
  let out = FrozenIndex::merge(&schema(), &[], AlignedBuf::new()).unwrap();
  assert_eq!(FrozenIndex::open(Region::new(out)).unwrap().doc_count(), 0);
}

#[test]
fn bitset_and_list_postings_are_both_used() {
  // 1000 docs: "dense" in every doc (bitset), "rare" in 3 (list).
  let schema = Schema::new().with("k", FieldType::Keyword).unwrap();
  let mut m = MemIndex::new(schema);
  for id in 0..1000u32 {
    let mut tags = vec!["dense"];
    if id % 400 == 7 {
      tags.push("rare");
    }
    m.add(id, &json!({ "k": tags })).unwrap();
  }
  let buf = m.write_to(AlignedBuf::new()).unwrap();
  // Bitset: 1000 bits = 16 words = 128 bytes; list for 1000 would be 4000.
  let c =
    muninn_core::container::Container::open(Region::new(buf.clone())).unwrap();
  let post = c.section(*b"P0000\0\0\0").unwrap().len();
  // exists (bitset 128) + dense (bitset 128) + rare (list 12).
  assert_eq!(post, 128 + 128 + 12);
  let f = FrozenIndex::open(Region::new(buf)).unwrap();
  assert_eq!(
    f.evaluate(&Filter::eq("k", "dense"), 1000).unwrap().count(),
    1000
  );
  let rare: Vec<u32> = f
    .evaluate(&Filter::eq("k", "rare"), 1000)
    .unwrap()
    .iter()
    .collect();
  assert_eq!(rare, [7, 407, 807]);
  // Truncation applies to both encodings.
  assert_eq!(
    f.evaluate(&Filter::eq("k", "dense"), 70).unwrap().count(),
    70
  );
  assert_eq!(
    f.evaluate(&Filter::eq("k", "rare"), 408).unwrap().count(),
    2
  );
}
