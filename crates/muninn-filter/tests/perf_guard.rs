//! Coarse guards against accidental quadratic behaviour. Bounds are
//! generous (tens of times the expected cost) so slow CI runners pass.

use std::time::{Duration, Instant};

use muninn_core::{AlignedBuf, Region};
use muninn_filter::{FieldType, Filter, FrozenIndex, MemIndex, Schema};
use serde_json::json;

/// A large `in` on a numeric memtable field must cost about one pass
/// over the field, not one tail scan per listed value; and nesting must
/// not repeat the work per ancestor.
#[test]
fn large_numeric_in_on_memtable_is_not_quadratic() {
  let schema = Schema::new()
    .with("i", FieldType::Integer)
    .unwrap()
    .with("f", FieldType::Float)
    .unwrap();
  let mut m = MemIndex::new(schema);
  // 200_000 = 195 * 1024 + 320: several sorted runs plus a busy tail.
  let n = 200_000u32 + 700;
  for d in 0..n {
    let v = (d as i64 * 7919) % 400_000;
    m.add(d, &json!({"i": v, "f": v as f64 * 0.5})).unwrap();
  }
  let ints: Vec<i64> = (0..1_000_000).map(|k| k * 2).collect();
  let floats: Vec<f64> = (0..1_000_000).map(|k| k as f64 * 0.5).collect();
  let leaf_i = Filter::is_in("i", ints);
  let leaf_f = Filter::is_in("f", floats);
  let nested = !!(Filter::and([
    Filter::or([leaf_i.clone(), Filter::eq("i", -1)]),
    !!(leaf_f.clone() | Filter::eq("f", -1.0)),
  ]));
  let frozen =
    FrozenIndex::open(Region::new(m.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  for q in [&leaf_i, &leaf_f, &nested] {
    let t = Instant::now();
    let got = m.evaluate(q, n as usize).unwrap();
    let est = m.estimate(q, n as usize);
    let took = t.elapsed();
    let want = frozen.evaluate(q, n as usize).unwrap();
    assert_eq!(got, want);
    assert!((0.0..=1.0).contains(&est));
    eprintln!("{} values: {took:?}", got.count());
    assert!(took < Duration::from_secs(1), "took {took:?}");
  }
}

/// The field limit keeps per-field container sections bounded; a schema
/// at the limit writes and opens quickly.
#[test]
fn field_count_is_capped_and_fast_at_the_cap() {
  use muninn_filter::MAX_FIELDS;
  assert_eq!(MAX_FIELDS, 4096);
  let mut schema = Schema::new();
  for i in 0..MAX_FIELDS {
    let ty = [FieldType::Keyword, FieldType::Integer, FieldType::Bool][i % 3];
    schema.add(&format!("f{i}"), ty).unwrap();
  }
  let e = schema.add("one_more", FieldType::Keyword).unwrap_err();
  assert!(e.to_string().contains("at most 4096 fields"), "{e}");
  let start = Instant::now();
  let mut m = MemIndex::new(schema);
  for d in 0..50u32 {
    m.add(
      d,
      &json!({"f0": "x", "f4095": "y", "f4093": d, "f4094": true}),
    )
    .unwrap();
  }
  let buf = m.write_to(AlignedBuf::new()).unwrap();
  let f = FrozenIndex::open(Region::new(buf)).unwrap();
  assert_eq!(
    f.evaluate(&Filter::exists("f4095"), 50).unwrap().count(),
    50
  );
  let took = start.elapsed();
  assert!(took < Duration::from_secs(2), "took {took:?}");
}
