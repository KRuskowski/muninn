//! Documented semantics, checked on hand-written documents against both
//! the mutable and the frozen index.

use muninn_core::{AlignedBuf, Error, Region};
use muninn_filter::{FieldType, Filter, FilterIndex, FrozenIndex, MemIndex};
use muninn_filter::{Schema, Value};
use serde_json::{Value as Json, json};

fn schema() -> Schema {
  Schema::new()
    .with("i", FieldType::Integer)
    .and_then(|s| s.with("f", FieldType::Float))
    .and_then(|s| s.with("k", FieldType::Keyword))
    .and_then(|s| s.with("b", FieldType::Bool))
    .unwrap()
}

/// Both indexes over `docs` (ids 0, 1, ...).
fn indexes(docs: &[Json]) -> (MemIndex, FrozenIndex) {
  let mut m = MemIndex::new(schema());
  for (id, d) in docs.iter().enumerate() {
    m.add(id as u32, d).unwrap();
  }
  let f =
    FrozenIndex::open(Region::new(m.write_to(AlignedBuf::new()).unwrap()))
      .unwrap();
  (m, f)
}

/// Matching ids, asserting both indexes agree.
fn hits(ix: &(MemIndex, FrozenIndex), f: &Filter) -> Vec<u32> {
  let n = ix.0.num_docs();
  let a: Vec<u32> = ix.0.evaluate(f, n).unwrap().iter().collect();
  let b: Vec<u32> = ix.1.evaluate(f, n).unwrap().iter().collect();
  assert_eq!(a, b, "mem vs frozen for {f:?}");
  a
}

fn invalid(ix: &dyn FilterIndex, f: &Filter) -> String {
  match ix.evaluate(f, 10) {
    Err(Error::InvalidArgument(m)) => m,
    other => panic!("expected InvalidArgument for {f:?}, got {other:?}"),
  }
}

#[test]
fn integer_fields_take_whole_numbers_only() {
  let ix = indexes(&[
    json!({"i": 3}),
    json!({"i": 3.0}),
    json!({"i": 3.5}),
    json!({"i": "3"}),
    json!({"i": 18446744073709551615u64}),
    json!({"i": -9223372036854775808i64}),
    json!({"i": 4e0}),
  ]);
  assert_eq!(hits(&ix, &Filter::exists("i")), [0, 1, 5, 6]);
  assert_eq!(hits(&ix, &Filter::eq("i", 3)), [0, 1]);
  // Float filter values compare exactly with integers.
  assert_eq!(hits(&ix, &Filter::eq("i", 3.0)), [0, 1]);
  assert!(hits(&ix, &Filter::eq("i", 3.5)).is_empty());
  assert!(hits(&ix, &Filter::is_in("i", [3.5, 2.9])).is_empty());
  assert_eq!(hits(&ix, &Filter::gt("i", 2.5)), [0, 1, 6]);
  assert_eq!(hits(&ix, &Filter::gt("i", 3.0)), [6]);
  assert_eq!(hits(&ix, &Filter::lt("i", 3.5)), [0, 1, 5]);
  assert_eq!(hits(&ix, &Filter::range("i", 3.5..4.5)), [6]);
  assert_eq!(hits(&ix, &Filter::eq("i", i64::MIN)), [5]);
  assert_eq!(hits(&ix, &Filter::lte("i", -1e300)), Vec::<u32>::new());
  assert_eq!(hits(&ix, &Filter::lt("i", f64::INFINITY)), [0, 1, 5, 6]);
  assert_eq!(hits(&ix, &Filter::gt("i", f64::NEG_INFINITY)), [0, 1, 5, 6]);
  assert!(hits(&ix, &Filter::gt("i", f64::INFINITY)).is_empty());
  assert!(hits(&ix, &Filter::eq("i", f64::INFINITY)).is_empty());
}

#[test]
fn float_fields_compare_as_f64() {
  let ix = indexes(&[
    json!({"f": 0.0}),
    json!({"f": -0.0}),
    json!({"f": 2}),
    json!({"f": 2.5}),
    json!({"f": 0.1}),
    json!({"f": "2"}),
    json!({"f": 1e-310}),
  ]);
  // -0.0 and 0.0 are the same value.
  assert_eq!(hits(&ix, &Filter::eq("f", 0.0)), [0, 1]);
  assert_eq!(hits(&ix, &Filter::eq("f", -0.0)), [0, 1]);
  assert_eq!(hits(&ix, &Filter::eq("f", 0)), [0, 1]);
  // Integers in payload and filter convert to f64.
  assert_eq!(hits(&ix, &Filter::eq("f", 2)), [2]);
  assert_eq!(hits(&ix, &Filter::eq("f", 2.0)), [2]);
  // Exact equality: 0.1 matches its own f64, not a nearby sum.
  assert_eq!(hits(&ix, &Filter::eq("f", 0.1)), [4]);
  assert!(hits(&ix, &Filter::eq("f", 0.1 + 0.2 - 0.2)).is_empty());
  assert_eq!(hits(&ix, &Filter::gt("f", 0.0)), [2, 3, 4, 6]);
  assert_eq!(hits(&ix, &Filter::gt("f", -0.0)), [2, 3, 4, 6]);
  assert_eq!(hits(&ix, &Filter::lt("f", 1e-320)), [0, 1]);
  assert_eq!(hits(&ix, &Filter::range("f", 2..3)), [2, 3]);
  assert_eq!(hits(&ix, &Filter::range("f", 2.0..2.5)), [2]);
  assert_eq!(
    hits(&ix, &Filter::range("f", f64::NEG_INFINITY..=f64::INFINITY)).len(),
    6
  );
  assert!(hits(&ix, &Filter::eq("f", f64::INFINITY)).is_empty());
  assert!(hits(&ix, &Filter::lt("f", f64::NEG_INFINITY)).is_empty());
}

#[test]
fn nan_in_a_filter_is_an_error() {
  let ix = indexes(&[json!({"f": 1.0, "i": 1})]);
  for f in [
    Filter::eq("f", f64::NAN),
    Filter::is_in("f", [1.0, f64::NAN]),
    Filter::gt("f", f64::NAN),
    Filter::lt("i", f64::NAN),
    Filter::eq("i", f64::NAN),
  ] {
    for i in [&ix.0 as &dyn FilterIndex, &ix.1] {
      assert!(invalid(i, &f).contains("NaN"), "{f:?}");
      assert_eq!(i.estimate(&f, 1), 1.0);
    }
  }
}

#[test]
fn arrays_match_by_any_element() {
  let ix = indexes(&[
    json!({"k": ["a", "b"], "i": [1, 10]}),
    json!({"k": "a", "i": 5}),
    json!({"k": [], "i": []}),
    json!({}),
    json!({"k": ["c", ["a"]], "i": [[5]]}),
  ]);
  assert_eq!(hits(&ix, &Filter::eq("k", "a")), [0, 1, 4]);
  assert_eq!(hits(&ix, &Filter::is_in("k", ["b", "c"])), [0, 4]);
  // One element must satisfy the whole range: [1, 10] is not in 4..6.
  assert_eq!(hits(&ix, &Filter::range("i", 4..6)), [1, 4]);
  assert_eq!(
    hits(&ix, &(Filter::gte("i", 4) & Filter::lt("i", 6))),
    [0, 1, 4]
  );
  // Empty arrays do not count as present.
  assert_eq!(hits(&ix, &Filter::exists("k")), [0, 1, 4]);
  // Not is the complement, missing fields included.
  assert_eq!(hits(&ix, &!Filter::eq("k", "a")), [2, 3]);
  assert_eq!(hits(&ix, &!Filter::eq("k", "b")), [1, 2, 3, 4]);
}

#[test]
fn bools() {
  let ix = indexes(&[
    json!({"b": true}),
    json!({"b": false}),
    json!({"b": [true, false]}),
    json!({"b": "true"}),
    json!({"b": 1}),
  ]);
  assert_eq!(hits(&ix, &Filter::eq("b", true)), [0, 2]);
  assert_eq!(hits(&ix, &Filter::eq("b", false)), [1, 2]);
  assert_eq!(hits(&ix, &Filter::is_in("b", [true, false])), [0, 1, 2]);
  assert_eq!(hits(&ix, &Filter::exists("b")), [0, 1, 2]);
  assert_eq!(hits(&ix, &!Filter::eq("b", true)), [1, 3, 4]);
}

#[test]
fn undeclared_fields_and_mistyped_values_are_errors() {
  let ix = indexes(&[json!({"k": "a", "x": 1})]);
  for i in [&ix.0 as &dyn FilterIndex, &ix.1] {
    let m = invalid(i, &Filter::eq("x", 1));
    assert_eq!(m, "field \"x\" is not declared filterable");
    let m = invalid(i, &(Filter::exists("k") & !Filter::exists("nope")));
    assert!(m.contains("\"nope\""), "{m}");
    let m = invalid(i, &Filter::eq("k", 1));
    assert_eq!(
      m,
      "field \"k\" is a keyword field; eq needs a string, got integer 1"
    );
    let m = invalid(i, &Filter::is_in("i", [Value::from("1")]));
    assert!(m.contains("integer field; in needs a number"), "{m}");
    let m = invalid(i, &Filter::range("k", "a".."b"));
    assert!(m.contains("range needs an integer or float field"), "{m}");
    let m = invalid(i, &Filter::eq("b", 1));
    assert!(m.contains("needs a boolean"), "{m}");
    assert_eq!(i.estimate(&Filter::eq("x", 1), 10), 1.0);
  }
  assert!(schema().check(&Filter::eq("x", 1)).is_err());
  assert!(schema().check(&Filter::eq("k", "a")).is_ok());
}

#[test]
fn num_docs_bounds_the_result() {
  let ix = indexes(&[json!({"k": "a"}), json!({"k": "b"}), json!({"k": "a"})]);
  for i in [&ix.0 as &dyn FilterIndex, &ix.1] {
    let b = i.evaluate(&Filter::eq("k", "a"), 2).unwrap();
    assert_eq!((b.capacity(), b.iter().collect::<Vec<_>>()), (2, vec![0]));
    // Documents past the index have no values: Not includes them.
    let b = i.evaluate(&!Filter::eq("k", "a"), 6).unwrap();
    assert_eq!(b.iter().collect::<Vec<_>>(), [1, 3, 4, 5]);
    assert_eq!(i.evaluate(&Filter::and([]), 6).unwrap().count(), 6);
    assert_eq!(i.evaluate(&Filter::or([]), 6).unwrap().count(), 0);
    assert_eq!(i.evaluate(&Filter::eq("k", "a"), 0).unwrap().capacity(), 0);
  }
}

#[test]
fn estimates_follow_counts() {
  // k: "a" in 1 of 4 docs; i: 0..100 uniformly.
  let docs: Vec<Json> = (0..400)
    .map(|i| json!({"k": if i % 4 == 0 { "a" } else { "b" }, "i": i % 100}))
    .collect();
  let ix = indexes(&docs);
  for i in [&ix.0 as &dyn FilterIndex, &ix.1] {
    let a = Filter::eq("k", "a");
    let r = Filter::range("i", 0..10);
    assert_eq!(i.estimate(&a, 400), 0.25);
    assert_eq!(i.estimate(&r, 400), 0.1);
    assert!((i.estimate(&(a.clone() & r.clone()), 400) - 0.025).abs() < 1e-6);
    let or = 1.0 - 0.75 * 0.9;
    assert!((i.estimate(&(a.clone() | r.clone()), 400) - or).abs() < 1e-6);
    assert_eq!(i.estimate(&!a.clone(), 400), 0.75);
    assert_eq!(i.estimate(&Filter::eq("k", "zzz"), 400), 0.0);
    assert_eq!(i.estimate(&Filter::is_in("k", ["a", "b"]), 400), 1.0);
    assert_eq!(i.estimate(&Filter::and([]), 400), 1.0);
    assert_eq!(i.estimate(&a, 0), 0.0);
    // i % 4 == 0 and i % 100 < 10: three per hundred.
    assert_eq!(i.evaluate(&(a.clone() & r), 400).unwrap().count(), 12);
  }
}

#[test]
fn and_short_circuits_and_is_ordered_by_selectivity() {
  // Evaluation order is internal; this checks results stay exact when a
  // cheap empty clause comes last in the written filter.
  let docs: Vec<Json> = (0..1000).map(|i| json!({"i": i, "k": "a"})).collect();
  let ix = indexes(&docs);
  let f = Filter::exists("i") & Filter::eq("k", "a") & Filter::eq("k", "none");
  assert!(hits(&ix, &f).is_empty());
  let f = Filter::exists("i") & !Filter::lt("i", 990) & Filter::eq("k", "a");
  assert_eq!(hits(&ix, &f), (990..1000).collect::<Vec<u32>>());
  let f = Filter::eq("k", "a") | Filter::eq("i", 5);
  assert_eq!(hits(&ix, &f).len(), 1000);
}

#[test]
fn numbers_below_i64_min_are_out_of_range() {
  // serde_json parses integers below i64::MIN as floats, which round to
  // exactly -2^63; they must not alias i64::MIN.
  let docs: Vec<Json> = [
    r#"{"i": -9223372036854775808}"#,
    r#"{"i": -9223372036854775809}"#,
    r#"{"i": -9223372036854776000}"#,
    r#"{"i": -9223372036854775807}"#,
  ]
  .iter()
  .map(|t| serde_json::from_str(t).unwrap())
  .collect();
  let ix = indexes(&docs);
  assert_eq!(hits(&ix, &Filter::exists("i")), [0, 3]);
  assert_eq!(hits(&ix, &Filter::eq("i", i64::MIN)), [0]);
  let q = |t: &str| Filter::parse(t).unwrap();
  let below = "-9223372036854775809";
  assert!(hits(&ix, &q(&format!(r#"{{"eq":{{"i":{below}}}}}"#))).is_empty());
  assert!(
    hits(
      &ix,
      &q(&format!(r#"{{"range":{{"i":{{"lte":{below}}}}}}}"#))
    )
    .is_empty()
  );
  assert!(
    hits(&ix, &q(&format!(r#"{{"range":{{"i":{{"lt":{below}}}}}}}"#)))
      .is_empty()
  );
  for op in ["gt", "gte"] {
    let t = format!(r#"{{"range":{{"i":{{"{op}":{below}}}}}}}"#);
    assert_eq!(hits(&ix, &q(&t)), [0, 3], "{t}");
  }
}
