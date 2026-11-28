//! JSON form of filters and schemas: round trips and error messages.

mod common;

use std::ops::Bound;

use common::{Rng, filter};
use muninn_filter::{FieldType, Filter, Schema, Value};
use serde_json::json;

#[test]
fn example_from_the_issue_parses() {
  let text = r#"{"and":[{"eq":{"lang":"en"}},{"range":{"year":{"gte":2020}}},
    {"not":{"in":{"tag":["a","b"]}}}]}"#;
  let f = Filter::parse(text).unwrap();
  assert_eq!(
    f,
    Filter::eq("lang", "en")
      & Filter::gte("year", 2020)
      & !Filter::is_in("tag", ["a", "b"])
  );
  let back = f.to_json().unwrap();
  assert_eq!(
    back,
    serde_json::from_str::<serde_json::Value>(text).unwrap()
  );
}

#[test]
fn every_form_round_trips() {
  let cases = [
    json!({"eq": {"a": "x"}}),
    json!({"eq": {"a": 1}}),
    json!({"eq": {"a": -1.5}}),
    json!({"eq": {"a": true}}),
    json!({"in": {"a": []}}),
    json!({"in": {"a": ["x", 2, 2.5, false]}}),
    json!({"range": {"a": {}}}),
    json!({"range": {"a": {"gt": 1, "lte": 2.5}}}),
    json!({"range": {"a": {"gte": -3}}}),
    json!({"range": {"a": {"lt": 0}}}),
    json!({"exists": "a.b.c"}),
    json!({"and": []}),
    json!({"or": [{"exists": "a"}, {"not": {"exists": "b"}}]}),
  ];
  for v in cases {
    let f = Filter::from_json(&v).unwrap();
    assert_eq!(f.to_json().unwrap(), v);
  }
  let f =
    Filter::from_json(&json!({"range": {"a": {"gt": 1, "lte": 2.5}}})).unwrap();
  assert_eq!(
    f,
    Filter::Range {
      field: "a".into(),
      lower: Bound::Excluded(Value::Int(1)),
      upper: Bound::Included(Value::Float(2.5)),
    }
  );
}

#[test]
fn numbers_keep_their_kind() {
  let f = Filter::parse(r#"{"in":{"a":[2, 2.0, 2e0, 18446744073709551615]}}"#)
    .unwrap();
  assert_eq!(
    f,
    Filter::In(
      "a".into(),
      vec![
        Value::Int(2),
        Value::Float(2.0),
        Value::Float(2.0),
        Value::Float(u64::MAX as f64),
      ]
    )
  );
  // A float that happens to be whole stays a float through text.
  let text = Filter::eq("a", 2.0).to_json().unwrap().to_string();
  assert_eq!(text, r#"{"eq":{"a":2.0}}"#);
  assert_eq!(Filter::parse(&text).unwrap(), Filter::eq("a", 2.0));
}

#[test]
fn random_filters_round_trip_through_text_and_serde() {
  let mut r = Rng::new(3);
  for _ in 0..2000 {
    let f = filter(&mut r, 5, true);
    let v = f.to_json().unwrap();
    assert_eq!(Filter::from_json(&v).unwrap(), f);
    let text = v.to_string();
    assert_eq!(text.parse::<Filter>().unwrap(), f);
    let via_serde: Filter =
      serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
    assert_eq!(via_serde, f);
  }
}

#[test]
fn non_finite_values_cannot_be_written() {
  for f in [
    Filter::eq("a", f64::NAN),
    Filter::lt("a", f64::INFINITY),
    Filter::is_in("a", [1.0, f64::NEG_INFINITY]),
    !Filter::eq("a", f64::INFINITY),
  ] {
    assert!(f.to_json().is_err(), "{f:?}");
    assert!(serde_json::to_string(&f).is_err(), "{f:?}");
  }
}

#[test]
fn errors_name_the_location() {
  let cases = [
    ("[]", "filter at $: expected an object with one of eq, in"),
    ("{}", "filter at $: empty object"),
    (r#"{"eq":{"a":1},"in":{"b":[]}}"#, "exactly one operator"),
    (r#"{"like":{"a":"x"}}"#, "unknown operator \"like\""),
    (
      r#"{"eq":"a"}"#,
      "at $.eq: eq expects {\"<field>\": ...}, got a string",
    ),
    (r#"{"eq":{}}"#, "eq needs a field"),
    (
      r#"{"eq":{"a":1,"b":2}}"#,
      "exactly one field; combine several",
    ),
    (r#"{"eq":{"":1}}"#, "field name is empty"),
    (
      r#"{"eq":{"a":null}}"#,
      "at $.eq.a: expected a string, number",
    ),
    (r#"{"eq":{"a":[1]}}"#, "got an array"),
    (r#"{"in":{"a":1}}"#, "at $.in.a: in expects an array"),
    (r#"{"in":{"a":[1,{}]}}"#, "at $.in.a[1]: expected a string"),
    (r#"{"range":{"a":5}}"#, "range expects {\"gte\""),
    (
      r#"{"range":{"a":{"gte":"x"}}}"#,
      "at $.range.a.gte: range bound must",
    ),
    (
      r#"{"range":{"a":{"from":1}}}"#,
      "unknown range bound \"from\"",
    ),
    (r#"{"range":{"a":{"gt":1,"gte":2}}}"#, "at most one lower"),
    (r#"{"range":{"a":{"lt":1,"lte":2}}}"#, "one upper"),
    (r#"{"exists":5}"#, "exists expects a field name string"),
    (r#"{"exists":""}"#, "exists expects a field name string"),
    (r#"{"and":{}}"#, "at $.and: and expects an array"),
    (
      r#"{"or":[{"exists":"a"},7]}"#,
      "at $.or[1]: expected an object",
    ),
    (r#"{"not":[]}"#, "at $.not: expected an object"),
    (
      r#"{"and":[{"not":{"eq":{"a":{}}}}]}"#,
      "at $.and[0].not.eq.a",
    ),
    ("{not json", "filter is not valid JSON"),
  ];
  for (text, want) in cases {
    let e = Filter::parse(text).unwrap_err();
    assert!(
      matches!(e, muninn_core::Error::InvalidArgument(_)),
      "{text}: {e:?}"
    );
    assert!(e.to_string().contains(want), "{text}: got {e}, want {want}");
  }
  // serde surfaces the same message.
  let e = serde_json::from_str::<Filter>(r#"{"eq":{}}"#).unwrap_err();
  assert!(e.to_string().contains("eq needs a field"), "{e}");
}

#[test]
fn deep_nesting_is_rejected_not_overflowed() {
  let mut v = json!({"exists": "a"});
  for _ in 0..300 {
    v = json!({ "not": v });
  }
  let e = Filter::from_json(&v).unwrap_err();
  assert!(e.to_string().contains("nested deeper than 32"), "{e}");
}

/// The documented depth error is what users of the JSON text form see,
/// for every shape of nesting, not serde_json's recursion limit.
#[test]
fn depth_limit_is_reachable_from_text() {
  use muninn_filter::MAX_DEPTH;
  let chain = |depth: usize, open: &str, close: &str| {
    let mut t = r#"{"exists":"a"}"#.to_string();
    for _ in 1..depth {
      t = format!("{open}{t}{close}");
    }
    t
  };
  for (open, close) in [(r#"{"not":"#, "}"), (r#"{"and":["#, "]}")] {
    let ok = chain(MAX_DEPTH, open, close);
    assert_eq!(Filter::parse(&ok).unwrap().depth(), MAX_DEPTH);
    for depth in [MAX_DEPTH + 1, 100, 1000] {
      let e = Filter::parse(&chain(depth, open, close)).unwrap_err();
      let msg = e.to_string();
      assert!(msg.contains("nested deeper than 32 levels"), "{msg}");
    }
  }
}

#[test]
fn schema_json() {
  let v = json!({"lang": "keyword", "year": "integer", "p": "float",
    "meta.ok": "bool"});
  let s = Schema::from_json(&v).unwrap();
  assert_eq!(s.field("year").unwrap().field_type(), FieldType::Integer);
  assert_eq!(s.field("meta.ok").unwrap().field_type(), FieldType::Bool);
  assert_eq!(s.to_json(), v);
  let s2: Schema = serde_json::from_value(v.clone()).unwrap();
  assert_eq!(s2, s);
  assert_eq!(serde_json::to_value(&s).unwrap(), v);
  for (bad, want) in [
    (json!([]), "schema must be an object"),
    (json!({"a": "text"}), "type must be \"keyword\""),
    (json!({"a": 1}), "type must be"),
    (json!({"a..b": "bool"}), "non-empty dot-separated"),
  ] {
    let e = Schema::from_json(&bad).unwrap_err().to_string();
    assert!(e.contains(want), "{e}");
  }
}

#[test]
fn floats_round_trip_exactly_through_text() {
  // Random bit patterns cover every exponent; the default serde_json
  // parser is not correctly rounded and misses ~30% of these.
  let mut r = Rng::new(5);
  let mut checked = 0;
  while checked < 20_000 {
    let x = f64::from_bits(r.next());
    if !x.is_finite() {
      continue;
    }
    checked += 1;
    let text = Filter::eq("f", x).to_json().unwrap().to_string();
    match Filter::parse(&text).unwrap() {
      Filter::Eq(_, Value::Float(y)) => {
        assert_eq!(y.to_bits(), x.to_bits(), "{text}")
      }
      other => panic!("{text} parsed as {other:?}"),
    }
  }
}

#[test]
fn float_eq_survives_text_round_trips_end_to_end() {
  use muninn_core::{AlignedBuf, Region};
  use muninn_filter::{FrozenIndex, MemIndex};
  let schema = Schema::new().with("f", FieldType::Float).unwrap();
  let mut r = Rng::new(6);
  let mut xs = Vec::new();
  while xs.len() < 2000 {
    let x = f64::from_bits(r.next());
    if x.is_finite() {
      xs.push(x);
    }
  }
  // The engine's two mixes: payload parsed from text (HTTP) with a filter
  // built in Rust, and a payload built in Rust with a filter parsed from
  // text. Both must agree on every bit of the value.
  let mut from_text = MemIndex::new(schema.clone());
  let mut in_memory = MemIndex::new(schema);
  for (i, x) in xs.iter().enumerate() {
    let built = json!({ "f": x });
    let parsed: serde_json::Value =
      serde_json::from_str(&built.to_string()).unwrap();
    from_text.add(i as u32, &parsed).unwrap();
    in_memory.add(i as u32, &built).unwrap();
  }
  let freeze = |m: &MemIndex| {
    FrozenIndex::open(Region::new(m.write_to(AlignedBuf::new()).unwrap()))
      .unwrap()
  };
  let (ft, fm) = (freeze(&from_text), freeze(&in_memory));
  let n = xs.len();
  for (i, x) in xs.iter().enumerate() {
    let rust = Filter::eq("f", *x);
    let text = Filter::parse(&rust.to_json().unwrap().to_string()).unwrap();
    for hits in [
      from_text.evaluate(&rust, n),
      ft.evaluate(&rust, n),
      in_memory.evaluate(&text, n),
      fm.evaluate(&text, n),
    ] {
      assert!(hits.unwrap().contains(i as u32), "{x:e} not found");
    }
  }
}
