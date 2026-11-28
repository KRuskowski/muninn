//! Model-based randomized test: random upserts, deletes, flushes,
//! compactions and reopens against a simple in-memory model with exact
//! vector search, brute-force BM25 and filters. After every step the
//! engine's results must equal the model's: same ids in the same order,
//! scores within tolerance.
//!
//! Vectors have small integer components, so L2 distances are exact in
//! `f32` and ties are frequent; text uses a 12-word vocabulary, so BM25
//! ties are frequent too. Ties must break by write order (older first).
//!
//! BM25 statistics follow Lucene: a deleted or replaced document still
//! counts until a compaction removes it. The model tracks that by keeping
//! dead documents until `compact()`, which (flush + full merge) removes
//! every one of them. Background merging is off in the exact variant;
//! the second variant turns it on and checks everything that does not
//! depend on when merges happen.

mod common;

use std::collections::{BTreeMap, HashSet};
use std::ops::Bound;

use common::{Rng, TestDir};
use muninn::filter::{Filter as PFilter, Value as PValue};
use muninn::text::{Analyzer, AnalyzerConfig, Tokens, bm25, norm};
use muninn::{
  Collection, Database, Document, Durability, FieldKind, FieldSpec, Filter,
  Fusion, Id, IdFilter, Metric, Options, Query, Schema, SearchHit, VectorField,
};
use serde_json::{Value, json};

const DIM: usize = 4;
const FIELDS: [&str; 2] = ["title", "body"];
const WORDS: [&str; 12] = [
  "apple", "banana", "cherry", "date", "elder", "fig", "grape", "honey",
  "iris", "jade", "kiwi", "lemon",
];

#[derive(Clone, Debug)]
struct MDoc {
  id: Id,
  vec: Vec<f32>,
  text: BTreeMap<String, String>,
  /// Analysed tokens per field index, for present fields.
  toks: Vec<Option<Vec<String>>>,
  payload: Value,
  seq: u64,
  live: bool,
}

struct Model {
  /// Physically present docs (live or dead), in write order.
  docs: Vec<MDoc>,
  seq: u64,
  analyzer: Analyzer,
}

/// The model's form of an id filter.
type IdPred = Box<dyn Fn(&Id) -> bool>;

/// The model's form of a query's filters: id and payload.
type MFilter = Box<dyn Fn(&Id, &Value) -> bool>;

impl Model {
  fn new() -> Model {
    Model {
      docs: Vec::new(),
      seq: 0,
      analyzer: Analyzer::new(AnalyzerConfig::default()).unwrap(),
    }
  }

  fn tokens(&self, text: &str) -> Vec<String> {
    let mut t = Tokens::new();
    self.analyzer.analyze(text, &mut t);
    t.iter().map(String::from).collect()
  }

  fn kill(&mut self, id: &Id) {
    if let Some(m) = self.docs.iter_mut().find(|m| m.live && m.id == *id) {
      m.live = false;
    }
  }

  fn upsert(&mut self, batch: &[Document]) {
    for d in batch {
      self.kill(&d.id);
      let toks = FIELDS
        .iter()
        .map(|f| d.text.get(*f).map(|t| self.tokens(t)))
        .collect();
      self.seq += 1;
      self.docs.push(MDoc {
        id: d.id.clone(),
        vec: d.vector.clone().unwrap(),
        text: d.text.clone(),
        toks,
        payload: d.payload.clone(),
        seq: self.seq,
        live: true,
      });
    }
  }

  fn delete(&mut self, ids: &[Id]) {
    for id in ids {
      self.kill(id);
    }
  }

  fn compact(&mut self) {
    self.docs.retain(|m| m.live);
  }

  fn live(&self) -> impl Iterator<Item = &MDoc> {
    self.docs.iter().filter(|m| m.live)
  }

  fn get(&self, id: &Id) -> Option<&MDoc> {
    self.live().find(|m| m.id == *id)
  }

  fn top(mut v: Vec<(f32, u64, Id)>, k: usize) -> Vec<(Id, f32, u64)> {
    v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    v.truncate(k);
    v.into_iter().map(|(s, q, id)| (id, s, q)).collect()
  }

  fn vector_search(
    &self,
    q: &[f32],
    k: usize,
    f: &MFilter,
  ) -> Vec<(Id, f32, u64)> {
    let v = self
      .live()
      .filter(|m| f(&m.id, &m.payload))
      .map(|m| {
        let d: f32 = m.vec.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
        (-d, m.seq, m.id.clone())
      })
      .collect();
    Self::top(v, k)
  }

  /// Brute-force BM25, exactly as `muninn_text::bm25` documents it, with
  /// a document scoring its best field.
  fn text_search(
    &self,
    q: &str,
    fields: &[usize],
    k: usize,
    f: &MFilter,
  ) -> Vec<(Id, f32, u64)> {
    let mut terms: Vec<String> = Vec::new();
    for t in self.tokens(q) {
      if !terms.contains(&t) {
        terms.push(t);
      }
    }
    let mut best: BTreeMap<u64, (f32, Id)> = BTreeMap::new();
    for &fi in fields {
      let corpus: Vec<&MDoc> =
        self.docs.iter().filter(|m| m.toks[fi].is_some()).collect();
      let n = corpus.len() as u64;
      let total: u64 = corpus
        .iter()
        .map(|m| m.toks[fi].as_ref().unwrap().len() as u64)
        .sum();
      let avgdl = bm25::avg_doc_len(total, n);
      let w: Vec<f32> = terms
        .iter()
        .map(|t| {
          let df = corpus
            .iter()
            .filter(|m| m.toks[fi].as_ref().unwrap().contains(t))
            .count() as u64;
          bm25::idf(df, n) * (bm25::K1 + 1.0)
        })
        .collect();
      for m in corpus.iter().filter(|m| m.live && f(&m.id, &m.payload)) {
        let toks = m.toks[fi].as_ref().unwrap();
        let len =
          norm::decode_doc_len(norm::encode_doc_len(toks.len() as u32)) as f32;
        let kc = bm25::K1 * (1.0 - bm25::B + bm25::B * (len / avgdl));
        let mut s = 0f32;
        let mut hit = false;
        for (t, &wt) in terms.iter().zip(&w) {
          let tf = toks.iter().filter(|x| *x == t).count();
          if tf > 0 {
            s += wt / (1.0 + kc / tf as f32);
            hit = true;
          }
        }
        if hit {
          let e = best.entry(m.seq).or_insert((s, m.id.clone()));
          if s > e.0 {
            e.0 = s;
          }
        }
      }
    }
    let v = best
      .into_iter()
      .map(|(seq, (s, id))| (s, seq, id))
      .collect();
    Self::top(v, k)
  }

  fn hybrid_rrf(
    &self,
    v: &[f32],
    t: &str,
    k: usize,
    rrf_k: u32,
    f: &MFilter,
  ) -> Vec<(Id, f32, u64)> {
    let side = (4 * k).max(40);
    let vl = self.vector_search(v, side, f);
    let tl = self.text_search(t, &[0, 1], side, f);
    let mut fused: BTreeMap<u64, (f64, Id)> = BTreeMap::new();
    for list in [&vl, &tl] {
      for (r, (id, _, seq)) in list.iter().enumerate() {
        let e = fused.entry(*seq).or_insert((0.0, id.clone()));
        e.0 += 1.0 / (rrf_k as f64 + (r + 1) as f64);
      }
    }
    let v = fused
      .into_iter()
      .map(|(seq, (s, id))| (s as f32, seq, id))
      .collect();
    Self::top(v, k)
  }
}

fn random_id(rng: &mut Rng) -> Id {
  if rng.chance(0.75) {
    Id::U64(rng.below(30))
  } else {
    Id::Str(format!("s{}", rng.below(10)))
  }
}

fn random_vec(rng: &mut Rng) -> Vec<f32> {
  (0..DIM).map(|_| rng.below(4) as f32).collect()
}

fn random_words(rng: &mut Rng, max: u64) -> String {
  let n = rng.below(max + 1);
  (0..n)
    .map(|_| WORDS[rng.below(WORDS.len() as u64) as usize])
    .collect::<Vec<_>>()
    .join(" ")
}

fn random_doc(rng: &mut Rng, seq: u64) -> Document {
  let mut d = Document::new(random_id(rng)).vector(random_vec(rng));
  for f in FIELDS {
    if rng.chance(0.7) {
      let t = random_words(rng, 6);
      // Mixed case and punctuation exercise the analyzer on both sides.
      let t = if rng.chance(0.2) {
        t.to_uppercase() + "!"
      } else {
        t
      };
      d = d.text(f, t);
    }
  }
  if rng.chance(0.8) {
    d = d.payload(random_payload(rng, seq));
  }
  d
}

/// A random id filter and the model's equivalent.
fn random_id_filter(rng: &mut Rng) -> (Option<IdFilter>, IdPred) {
  match rng.below(6) {
    0 | 1 => (None, Box::new(|_| true)),
    2 => {
      let ids: HashSet<Id> =
        (0..rng.below(12)).map(|_| random_id(rng)).collect();
      let f = IdFilter::ids(ids.clone());
      (Some(f), Box::new(move |id| ids.contains(id)))
    }
    3 => (
      Some(IdFilter::predicate(|id| match id {
        muninn::IdRef::U64(v) => v % 2 == 0,
        muninn::IdRef::Str(s) => s.ends_with('1'),
      })),
      Box::new(|id| match id {
        Id::U64(v) => v % 2 == 0,
        Id::Str(s) => s.ends_with('1'),
      }),
    ),
    4 => {
      let ids: HashSet<Id> =
        (0..rng.below(20)).map(|_| random_id(rng)).collect();
      let f = IdFilter::ids(ids.clone()).not();
      (Some(f), Box::new(move |id| !ids.contains(id)))
    }
    _ => {
      let ids: HashSet<Id> =
        (0..rng.below(25)).map(|_| random_id(rng)).collect();
      let f = IdFilter::ids(ids.clone()).and(IdFilter::predicate(
        |id| !matches!(id, muninn::IdRef::U64(v) if v % 3 == 0),
      ));
      (
        Some(f),
        Box::new(move |id| {
          ids.contains(id) && !matches!(id, Id::U64(v) if v % 3 == 0)
        }),
      )
    }
  }
}

/// Random id and payload filters, and the model's combined equivalent.
/// Payload filters go through their JSON form half the time, as the HTTP
/// server will send them.
fn random_filter(
  rng: &mut Rng,
) -> ((Option<IdFilter>, Option<Filter>), MFilter) {
  let (idf, mid) = random_id_filter(rng);
  if rng.chance(0.4) {
    return ((idf, None), Box::new(move |id, _| mid(id)));
  }
  let pf = random_pfilter(rng, 3);
  let engine_pf = if rng.chance(0.5) {
    Filter::from_json(&pf.to_json().unwrap()).unwrap()
  } else {
    pf.clone()
  };
  (
    (idf, Some(engine_pf)),
    Box::new(move |id, p| mid(id) && pmatches(&pf, p)),
  )
}

// ---- payload filters: generation and brute-force oracle ----

/// Filterable payload fields of the model schema.
const PFIELDS: [(&str, FieldKind); 5] = [
  ("tag", FieldKind::Keyword),
  ("n", FieldKind::Integer),
  ("price", FieldKind::Float),
  ("flag", FieldKind::Bool),
  ("meta.lang", FieldKind::Keyword),
];
const TAGS: [&str; 4] = ["red", "green", "blue", "été"];
const PRICES: [f64; 6] = [-1.5, 0.0, 0.5, 2.0, 2.5, 9.75];

/// A payload with arrays, nesting, missing fields and wrongly typed
/// values (which filters must ignore).
fn random_payload(rng: &mut Rng, seq: u64) -> Value {
  let mut m = serde_json::Map::new();
  m.insert("seq".into(), json!(seq));
  let tag = |rng: &mut Rng| json!(TAGS[rng.below(TAGS.len() as u64) as usize]);
  match rng.below(6) {
    0 => {}
    1 => {
      let n = rng.below(3);
      m.insert("tag".into(), (0..n).map(|_| tag(rng)).collect());
    }
    2 => {
      m.insert("tag".into(), json!(7));
    }
    _ => {
      m.insert("tag".into(), tag(rng));
    }
  }
  match rng.below(6) {
    0 => {}
    1 => {
      m.insert("n".into(), json!("3"));
    }
    2 => {
      m.insert("n".into(), json!(2.0));
    }
    3 => {
      m.insert("n".into(), json!([rng.below(5), rng.below(5)]));
    }
    _ => {
      m.insert("n".into(), json!(rng.below(6) as i64 - 1));
    }
  }
  match rng.below(5) {
    0 => {}
    1 => {
      m.insert("price".into(), json!(rng.below(4)));
    }
    _ => {
      let p = PRICES[rng.below(PRICES.len() as u64) as usize];
      m.insert("price".into(), json!(p));
    }
  }
  match rng.below(5) {
    0 => {}
    1 => {
      m.insert("flag".into(), json!("true"));
    }
    _ => {
      m.insert("flag".into(), json!(rng.chance(0.5)));
    }
  }
  match rng.below(4) {
    0 => {}
    1 => {
      m.insert("meta".into(), json!([{"lang": "en"}, {"lang": tag(rng)}]));
    }
    _ => {
      let l = if rng.chance(0.5) { "en" } else { "de" };
      m.insert("meta".into(), json!({"lang": l}));
    }
  }
  Value::Object(m)
}

fn pvalue(rng: &mut Rng, kind: FieldKind) -> PValue {
  match kind {
    FieldKind::Keyword => match rng.below(5) {
      0 => PValue::Str("en".into()),
      1 => PValue::Str("zzz".into()),
      _ => PValue::Str(TAGS[rng.below(TAGS.len() as u64) as usize].into()),
    },
    FieldKind::Integer => match rng.below(4) {
      0 => PValue::Float(2.0),
      1 => PValue::Float(2.5),
      _ => PValue::Int(rng.below(6) as i64 - 1),
    },
    FieldKind::Float => match rng.below(3) {
      0 => PValue::Int(rng.below(4) as i64),
      _ => PValue::Float(PRICES[rng.below(PRICES.len() as u64) as usize]),
    },
    FieldKind::Bool => PValue::Bool(rng.chance(0.5)),
  }
}

fn pbound(rng: &mut Rng, kind: FieldKind) -> Bound<PValue> {
  match rng.below(3) {
    0 => Bound::Unbounded,
    1 => Bound::Included(pvalue(rng, kind)),
    _ => Bound::Excluded(pvalue(rng, kind)),
  }
}

/// A random, well-typed payload filter.
fn random_pfilter(rng: &mut Rng, depth: u32) -> PFilter {
  if depth > 1 && rng.chance(0.35) {
    let n = rng.below(3) as usize;
    return match rng.below(3) {
      0 => {
        PFilter::And((0..n).map(|_| random_pfilter(rng, depth - 1)).collect())
      }
      1 => {
        PFilter::Or((0..n).map(|_| random_pfilter(rng, depth - 1)).collect())
      }
      _ => !random_pfilter(rng, depth - 1),
    };
  }
  let (name, kind) = PFIELDS[rng.below(PFIELDS.len() as u64) as usize];
  let numeric = matches!(kind, FieldKind::Integer | FieldKind::Float);
  match rng.below(if numeric { 5 } else { 4 }) {
    0 | 1 => PFilter::Eq(name.into(), pvalue(rng, kind)),
    2 => PFilter::In(
      name.into(),
      (0..rng.below(3)).map(|_| pvalue(rng, kind)).collect(),
    ),
    3 => PFilter::Exists(name.into()),
    _ => PFilter::Range {
      field: name.into(),
      lower: pbound(rng, kind),
      upper: pbound(rng, kind),
    },
  }
}

#[derive(Clone, Copy, Debug)]
enum Lit<'a> {
  S(&'a str),
  I(i64),
  F(f64),
  B(bool),
}

fn collect_path<'a>(v: &'a Value, path: &[&str], out: &mut Vec<&'a Value>) {
  if let Value::Array(items) = v {
    for item in items {
      collect_path(item, path, out);
    }
    return;
  }
  if path.is_empty() {
    out.push(v);
  } else if let Some(child) = v.get(path[0]) {
    collect_path(child, &path[1..], out);
  }
}

fn typed(kind: FieldKind, v: &Value) -> Option<Lit<'_>> {
  match (kind, v) {
    (FieldKind::Keyword, Value::String(s)) => Some(Lit::S(s)),
    (FieldKind::Bool, Value::Bool(b)) => Some(Lit::B(*b)),
    (FieldKind::Float, Value::Number(n)) => Some(Lit::F(n.as_f64()?)),
    (FieldKind::Integer, Value::Number(n)) => {
      if let Some(i) = n.as_i64() {
        Some(Lit::I(i))
      } else {
        let f = n.as_f64()?;
        (f == f.trunc() && f.abs() < 9e18).then_some(Lit::I(f as i64))
      }
    }
    _ => None,
  }
}

fn values_of<'a>(name: &str, doc: &'a Value) -> Vec<Lit<'a>> {
  let kind = PFIELDS.iter().find(|f| f.0 == name).unwrap().1;
  let path: Vec<&str> = name.split('.').collect();
  let mut raw = Vec::new();
  collect_path(doc, &path, &mut raw);
  raw.into_iter().filter_map(|v| typed(kind, v)).collect()
}

fn pnum(v: &PValue) -> f64 {
  match *v {
    PValue::Int(i) => i as f64,
    PValue::Float(f) => f,
    _ => panic!("not a number {v:?}"),
  }
}

fn lit_eq(x: Lit, v: &PValue) -> bool {
  match (x, v) {
    (Lit::S(s), PValue::Str(t)) => s == t,
    (Lit::B(a), PValue::Bool(b)) => a == *b,
    (Lit::I(i), PValue::Int(j)) => i == *j,
    (Lit::I(i), PValue::Float(f)) => (i as f64) == *f,
    (Lit::F(a), v) => a == pnum(v),
    _ => panic!("ill-typed comparison {x:?} {v:?}"),
  }
}

fn in_range(x: Lit, lo: &Bound<PValue>, hi: &Bound<PValue>) -> bool {
  let x = match x {
    Lit::I(i) => i as f64,
    Lit::F(f) => f,
    _ => panic!("range on non-number"),
  };
  let l = match lo {
    Bound::Unbounded => true,
    Bound::Included(v) => x >= pnum(v),
    Bound::Excluded(v) => x > pnum(v),
  };
  let h = match hi {
    Bound::Unbounded => true,
    Bound::Included(v) => x <= pnum(v),
    Bound::Excluded(v) => x < pnum(v),
  };
  l && h
}

/// Whether payload `doc` matches `f`, from the documented semantics.
fn pmatches(f: &PFilter, doc: &Value) -> bool {
  match f {
    PFilter::Eq(n, v) => values_of(n, doc).iter().any(|&x| lit_eq(x, v)),
    PFilter::In(n, vs) => values_of(n, doc)
      .iter()
      .any(|&x| vs.iter().any(|v| lit_eq(x, v))),
    PFilter::Range {
      field,
      lower,
      upper,
    } => values_of(field, doc)
      .iter()
      .any(|&x| in_range(x, lower, upper)),
    PFilter::Exists(n) => !values_of(n, doc).is_empty(),
    PFilter::And(c) => c.iter().all(|f| pmatches(f, doc)),
    PFilter::Or(c) => c.iter().any(|f| pmatches(f, doc)),
    PFilter::Not(c) => !pmatches(c, doc),
  }
}

fn assert_same(got: &[SearchHit], want: &[(Id, f32, u64)], ctx: &str) {
  let g: Vec<&Id> = got.iter().map(|h| &h.id).collect();
  let w: Vec<&Id> = want.iter().map(|h| &h.0).collect();
  assert_eq!(g, w, "{ctx}: ids differ\n got {got:?}\nwant {want:?}");
  for (h, (_, s, _)) in got.iter().zip(want) {
    let tol = 1e-5 * s.abs().max(1.0);
    assert!(
      (h.score - s).abs() <= tol,
      "{ctx}: score {} vs model {s}",
      h.score
    );
  }
}

fn opts(rng: &mut Rng, auto_compact: bool) -> Options {
  Options {
    durability: Durability::NoSync,
    memtable_max_docs: 3 + rng.below(10) as usize,
    auto_compact,
    merge_factor: 2,
    ..Options::default()
  }
}

fn schema() -> Schema {
  let mut s =
    Schema::vector(VectorField::flat(DIM, Metric::L2)).with_text_fields(FIELDS);
  for (name, kind) in PFIELDS {
    s = s.with_filterable(FieldSpec {
      name: name.into(),
      kind,
    });
  }
  s
}

/// Checks the engine against the model. `exact_text` is false when
/// background merges make BM25 statistics unpredictable; text results are
/// then compared as sets.
fn check(
  c: &Collection,
  m: &Model,
  rng: &mut Rng,
  exact_text: bool,
  ctx: &str,
) {
  let st = c.stats().unwrap();
  assert_eq!(st.live_docs, m.live().count() as u64, "{ctx}: live docs");
  for _ in 0..3 {
    let k = 1 + rng.below(12) as usize;
    let (f, mf) = random_filter(rng);
    let with = |q: Query| {
      let q = match &f.0 {
        Some(f) => q.id_filter(f.clone()),
        None => q,
      };
      match &f.1 {
        Some(f) => q.filter(f.clone()),
        None => q,
      }
    };
    let qv = random_vec(rng);
    let got = c.search(&with(Query::vector(qv.clone()).k(k))).unwrap();
    assert_same(
      &got,
      &m.vector_search(&qv, k, &mf),
      &format!("{ctx} vector"),
    );

    let qt = random_words(rng, 3);
    let fields: Vec<usize> = match rng.below(3) {
      0 => vec![0],
      1 => vec![1],
      _ => vec![0, 1],
    };
    let names: Vec<&str> = fields.iter().map(|&i| FIELDS[i]).collect();
    if exact_text {
      let got = c
        .search(&with(Query::text(qt.clone()).k(k).fields(names)))
        .unwrap();
      let want = m.text_search(&qt, &fields, k, &mf);
      assert_same(&got, &want, &format!("{ctx} text {qt:?} {fields:?}"));
      let rk = 1 + rng.below(80) as u32;
      let got = c
        .search(&with(
          Query::hybrid(qv.clone(), qt.clone())
            .k(k)
            .fusion(Fusion::Rrf { k: rk }),
        ))
        .unwrap();
      let want = m.hybrid_rrf(&qv, &qt, k, rk, &mf);
      assert_same(&got, &want, &format!("{ctx} hybrid {qt:?}"));
    } else {
      let got = c
        .search(&with(Query::text(qt.clone()).k(1000).fields(names)))
        .unwrap();
      let want = m.text_search(&qt, &fields, 1000, &mf);
      let g: HashSet<&Id> = got.iter().map(|h| &h.id).collect();
      let w: HashSet<&Id> = want.iter().map(|h| &h.0).collect();
      assert_eq!(g, w, "{ctx} text set {qt:?}");
      assert_eq!(g.len(), got.len(), "{ctx}: duplicate text hits");
    }
  }
  for _ in 0..2 {
    let id = random_id(rng);
    let got = c.get(&id).unwrap();
    match m.get(&id) {
      None => assert!(got.is_none(), "{ctx}: get {id:?} should be none"),
      Some(md) => {
        let d = got.unwrap_or_else(|| panic!("{ctx}: get {id:?} missing"));
        assert_eq!(d.vector.as_deref(), Some(md.vec.as_slice()), "{ctx}");
        assert_eq!(d.text, md.text, "{ctx}");
        assert_eq!(d.payload, md.payload, "{ctx}");
      }
    }
  }
}

fn run(seed: u64, steps: usize, auto_compact: bool) {
  let dir = TestDir::new(&format!("model-{seed}"));
  let mut rng = Rng::new(seed);
  let o = opts(&mut rng, auto_compact);
  let mut db = Some(Database::open(dir.path(), o.clone()).unwrap());
  let mut c = db
    .as_ref()
    .unwrap()
    .create_collection("m", schema())
    .unwrap();
  let mut m = Model::new();
  let mut seq = 0;
  for step in 0..steps {
    let ctx = format!("seed {seed} step {step}");
    match rng.below(100) {
      0..45 => {
        let batch: Vec<Document> = (0..1 + rng.below(5))
          .map(|_| {
            seq += 1;
            random_doc(&mut rng, seq)
          })
          .collect();
        c.upsert(&batch).unwrap();
        m.upsert(&batch);
      }
      45..60 => {
        let ids: Vec<Id> =
          (0..1 + rng.below(3)).map(|_| random_id(&mut rng)).collect();
        c.delete(&ids).unwrap();
        m.delete(&ids);
      }
      60..68 => {
        c.flush().unwrap();
        let st = c.stats().unwrap();
        assert_eq!((st.memtable_docs, st.pending_flushes), (0, 0), "{ctx}");
      }
      68..73 => {
        c.compact().unwrap();
        m.compact();
        let st = c.stats().unwrap();
        assert!(st.segments.len() <= 1, "{ctx}: {st:?}");
        assert_eq!(st.deleted_docs, 0, "{ctx}: {st:?}");
      }
      73..78 => {
        drop(c);
        drop(db.take());
        db = Some(Database::open(dir.path(), o.clone()).unwrap());
        c = db.as_ref().unwrap().collection("m").unwrap();
      }
      _ => {}
    }
    check(&c, &m, &mut rng, !auto_compact, &ctx);
  }
  drop(c);
  drop(db);
}

#[test]
fn engine_matches_model_exactly() {
  for seed in 1..=8 {
    run(seed, 250, false);
  }
}

#[test]
fn engine_matches_model_with_background_merges() {
  for seed in 100..=104 {
    run(seed, 250, true);
  }
}

#[test]
fn payload_filters_are_checked_against_the_schema() {
  let dir = TestDir::new("filter-check");
  let db = Database::open(dir.path(), Options::default()).unwrap();
  let c = db.create_collection("m", schema()).unwrap();
  c.upsert(&[Document::new(1u64)
    .vector(vec![0.0; DIM])
    .payload(json!({"tag": "red"}))])
    .unwrap();
  let q = |f: Filter| c.search(&Query::vector(vec![0.0; DIM]).filter(f));
  for bad in [
    Filter::eq("undeclared", "x"),
    Filter::eq("n", "not a number"),
    Filter::range("tag", 1..2),
  ] {
    match q(bad.clone()) {
      Err(muninn::Error::InvalidArgument(_)) => {}
      other => panic!("{bad:?}: expected InvalidArgument, got {other:?}"),
    }
  }
  assert_eq!(q(Filter::eq("tag", "red")).unwrap().len(), 1);
  let j =
    Filter::parse(r#"{"or":[{"eq":{"tag":"blue"}},{"not":{"exists":"tag"}}]}"#)
      .unwrap();
  assert!(q(j).unwrap().is_empty());
  // A collection without filterable fields accepts only field-free
  // filters.
  let t = db
    .create_collection("t", Schema::vector(VectorField::flat(DIM, Metric::L2)))
    .unwrap();
  t.upsert(&[Document::new(1u64).vector(vec![0.0; DIM])])
    .unwrap();
  let tq = |f: Filter| t.search(&Query::vector(vec![0.0; DIM]).filter(f));
  assert!(tq(Filter::eq("tag", "red")).is_err());
  assert_eq!(tq(Filter::and([])).unwrap().len(), 1);
  assert!(tq(Filter::or([])).unwrap().is_empty());
}
