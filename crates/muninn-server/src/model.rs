//! JSON forms of schemas, documents, search requests and results, shared
//! by the HTTP API and the command-line tool. `docs/API.md` describes
//! them for users.
//!
//! Hot paths avoid intermediate trees: documents deserialize straight
//! into [`muninn::Document`]s and search hits serialize straight from
//! [`muninn::SearchHit`]s.

use std::collections::BTreeMap;
use std::fmt;

use muninn::text::StemmerChoice;
use muninn::{
  AnalyzerConfig, CollectionStats, Document, Error, FieldKind, FieldSpec,
  Filter, Fusion, Id, IdFilter, IndexKind, Metric, Quantization, Query, Result,
  Schema, SearchHit, VectorField,
};
use rayon::prelude::*;
use serde::de::{self, Deserializer, Visitor};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// Largest `k` a search may ask for.
pub const MAX_K: usize = 10_000;

/// Turns a serde_json error about `what` into `InvalidArgument`.
pub fn json_error(what: &str, e: serde_json::Error) -> Error {
  Error::invalid(format!("{what}: {e}"))
}

fn kind_of(v: &Value) -> &'static str {
  match v {
    Value::Null => "null",
    Value::Bool(_) => "a boolean",
    Value::Number(_) => "a number",
    Value::String(_) => "a string",
    Value::Array(_) => "an array",
    Value::Object(_) => "an object",
  }
}

fn invalid(msg: impl Into<String>) -> Error {
  Error::invalid(msg)
}

/// Rejects keys of `obj` not in `allowed`.
fn check_keys(
  at: &str,
  obj: &Map<String, Value>,
  allowed: &[&str],
) -> Result<()> {
  for k in obj.keys() {
    if !allowed.contains(&k.as_str()) {
      return Err(invalid(format!(
        "{at}: unknown key {k:?}; expected one of {}",
        allowed.join(", ")
      )));
    }
  }
  Ok(())
}

fn as_object<'a>(at: &str, v: &'a Value) -> Result<&'a Map<String, Value>> {
  v.as_object().ok_or_else(|| {
    invalid(format!("{at} must be an object, got {}", kind_of(v)))
  })
}

fn as_usize(at: &str, v: &Value) -> Result<usize> {
  v.as_u64()
    .and_then(|n| usize::try_from(n).ok())
    .ok_or_else(|| {
      invalid(format!("{at} must be a non-negative integer, got {v}"))
    })
}

fn as_str<'a>(at: &str, v: &'a Value) -> Result<&'a str> {
  v.as_str().ok_or_else(|| {
    invalid(format!("{at} must be a string, got {}", kind_of(v)))
  })
}

fn as_bool(at: &str, v: &Value) -> Result<bool> {
  v.as_bool().ok_or_else(|| {
    invalid(format!("{at} must be true or false, got {}", kind_of(v)))
  })
}

// ---------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------

fn metric_name(m: Metric) -> &'static str {
  match m {
    Metric::L2 => "l2",
    Metric::Dot => "dot",
    Metric::Cosine => "cosine",
  }
}

fn quant_name(q: Quantization) -> &'static str {
  match q {
    Quantization::None => "none",
    Quantization::Sq8 => "sq8",
    Quantization::Binary => "binary",
  }
}

fn kind_name(k: FieldKind) -> &'static str {
  match k {
    FieldKind::Keyword => "keyword",
    FieldKind::Integer => "integer",
    FieldKind::Float => "float",
    FieldKind::Bool => "bool",
  }
}

fn parse_kind(at: &str, v: &Value) -> Result<FieldKind> {
  match as_str(at, v)? {
    "keyword" => Ok(FieldKind::Keyword),
    "integer" => Ok(FieldKind::Integer),
    "float" => Ok(FieldKind::Float),
    "bool" => Ok(FieldKind::Bool),
    s => Err(invalid(format!(
      "{at}: unknown field kind {s:?}; expected keyword, integer, float or \
       bool"
    ))),
  }
}

fn parse_vector_field(v: &Value) -> Result<VectorField> {
  let at = "schema.vector";
  let o = as_object(at, v)?;
  check_keys(at, o, &["dim", "metric", "index", "quantization"])?;
  let dim = as_usize(
    "schema.vector.dim",
    o.get("dim")
      .ok_or_else(|| invalid("schema.vector.dim is required"))?,
  )?;
  let metric = match o.get("metric") {
    None => Metric::Cosine,
    Some(m) => {
      let s = as_str("schema.vector.metric", m)?;
      Metric::parse(s).ok_or_else(|| {
        invalid(format!(
          "schema.vector.metric: unknown metric {s:?}; expected cosine, dot \
           or l2"
        ))
      })?
    }
  };
  let mut field = VectorField::hnsw(dim, metric);
  match o.get("index") {
    None => {}
    Some(Value::String(s)) => match s.as_str() {
      "hnsw" => {}
      "flat" => field.index = IndexKind::Flat,
      s => {
        return Err(invalid(format!(
          "schema.vector.index: unknown index {s:?}; expected hnsw or flat"
        )));
      }
    },
    Some(ix) => {
      let at = "schema.vector.index";
      let o = as_object(at, ix)?;
      check_keys(at, o, &["kind", "m", "ef_construction"])?;
      let kind = as_str(
        "schema.vector.index.kind",
        o.get("kind")
          .ok_or_else(|| invalid("schema.vector.index.kind is required"))?,
      )?;
      match kind {
        "flat" => {
          if o.len() > 1 {
            return Err(invalid(
              "schema.vector.index: a flat index takes no parameters",
            ));
          }
          field.index = IndexKind::Flat;
        }
        "hnsw" => {
          let IndexKind::Hnsw {
            mut m,
            mut ef_construction,
          } = field.index
          else {
            unreachable!("VectorField::hnsw builds an HNSW field")
          };
          if let Some(x) = o.get("m") {
            m = as_usize("schema.vector.index.m", x)?;
          }
          if let Some(x) = o.get("ef_construction") {
            ef_construction =
              as_usize("schema.vector.index.ef_construction", x)?;
          }
          field.index = IndexKind::Hnsw { m, ef_construction };
        }
        s => {
          return Err(invalid(format!(
            "schema.vector.index.kind: unknown index {s:?}; expected hnsw \
             or flat"
          )));
        }
      }
    }
  }
  if let Some(q) = o.get("quantization") {
    field.quantization = match as_str("schema.vector.quantization", q)? {
      "none" => Quantization::None,
      "sq8" => Quantization::Sq8,
      "binary" => Quantization::Binary,
      s => {
        return Err(invalid(format!(
          "schema.vector.quantization: unknown quantization {s:?}; \
           expected none, sq8 or binary"
        )));
      }
    };
  }
  Ok(field)
}

fn parse_analyzer(v: &Value) -> Result<AnalyzerConfig> {
  let at = "schema.analyzer";
  if let Value::String(s) = v {
    return match s.as_str() {
      "default" => Ok(AnalyzerConfig::default()),
      "english" => Ok(AnalyzerConfig::english()),
      s => Err(invalid(format!(
        "{at}: unknown analyzer {s:?}; expected default, english or an \
         object"
      ))),
    };
  }
  let o = as_object(at, v)?;
  check_keys(
    at,
    o,
    &[
      "fold_diacritics",
      "english_stop_words",
      "stemmer",
      "max_token_bytes",
    ],
  )?;
  let mut a = AnalyzerConfig::default();
  if let Some(x) = o.get("fold_diacritics") {
    a.fold_diacritics = as_bool("schema.analyzer.fold_diacritics", x)?;
  }
  if let Some(x) = o.get("english_stop_words") {
    a.english_stop_words = as_bool("schema.analyzer.english_stop_words", x)?;
  }
  if let Some(x) = o.get("stemmer") {
    a.stemmer = match as_str("schema.analyzer.stemmer", x)? {
      "none" => StemmerChoice::None,
      "porter_english" => StemmerChoice::PorterEnglish,
      s => {
        return Err(invalid(format!(
          "schema.analyzer.stemmer: unknown stemmer {s:?}; expected none or \
           porter_english"
        )));
      }
    };
  }
  if let Some(x) = o.get("max_token_bytes") {
    let n = as_usize("schema.analyzer.max_token_bytes", x)?;
    a.max_token_bytes = u16::try_from(n).map_err(|_| {
      invalid(format!(
        "schema.analyzer.max_token_bytes must be at most 65535, got {n}"
      ))
    })?;
  }
  Ok(a)
}

fn parse_filterable(v: &Value) -> Result<Vec<FieldSpec>> {
  match v {
    // Shorthand: {"lang": "keyword", "year": "integer"}.
    Value::Object(o) => o
      .iter()
      .map(|(name, kind)| {
        Ok(FieldSpec {
          name: name.clone(),
          kind: parse_kind(&format!("schema.filterable.{name}"), kind)?,
        })
      })
      .collect(),
    Value::Array(a) => a
      .iter()
      .enumerate()
      .map(|(i, f)| {
        let at = format!("schema.filterable[{i}]");
        let o = as_object(&at, f)?;
        check_keys(&at, o, &["name", "kind"])?;
        let name = o
          .get("name")
          .ok_or_else(|| invalid(format!("{at}.name is required")))?;
        let kind = o
          .get("kind")
          .ok_or_else(|| invalid(format!("{at}.kind is required")))?;
        Ok(FieldSpec {
          name: as_str(&format!("{at}.name"), name)?.to_string(),
          kind: parse_kind(&format!("{at}.kind"), kind)?,
        })
      })
      .collect(),
    v => Err(invalid(format!(
      "schema.filterable must be an array or an object, got {}",
      kind_of(v)
    ))),
  }
}

/// Parses a schema from its JSON form (see `docs/API.md`) and validates
/// it.
///
/// ```
/// let s = muninn_server::model::schema_from_json(&serde_json::json!({
///   "vector": {"dim": 3, "metric": "cosine"},
///   "text_fields": ["title"],
///   "filterable": {"lang": "keyword"}
/// }))
/// .unwrap();
/// assert_eq!(s.vector.unwrap().dim, 3);
/// ```
pub fn schema_from_json(v: &Value) -> Result<Schema> {
  let o = as_object("schema", v)?;
  check_keys(
    "schema",
    o,
    &["vector", "text_fields", "analyzer", "filterable"],
  )?;
  let vector = match o.get("vector") {
    None | Some(Value::Null) => None,
    Some(f) => Some(parse_vector_field(f)?),
  };
  let text_fields = match o.get("text_fields") {
    None | Some(Value::Null) => Vec::new(),
    Some(Value::Array(a)) => a
      .iter()
      .enumerate()
      .map(|(i, f)| Ok(as_str(&format!("schema.text_fields[{i}]"), f)?.into()))
      .collect::<Result<_>>()?,
    Some(v) => {
      return Err(invalid(format!(
        "schema.text_fields must be an array of strings, got {}",
        kind_of(v)
      )));
    }
  };
  let analyzer = match o.get("analyzer") {
    None | Some(Value::Null) => AnalyzerConfig::default(),
    Some(a) => parse_analyzer(a)?,
  };
  let filterable = match o.get("filterable") {
    None | Some(Value::Null) => Vec::new(),
    Some(f) => parse_filterable(f)?,
  };
  let s = Schema {
    vector,
    text_fields,
    analyzer,
    filterable,
  };
  s.validate()?;
  Ok(s)
}

/// The JSON form of a schema, with every setting spelled out. Parsing it
/// with [`schema_from_json`] gives the same schema back.
pub fn schema_to_json(s: &Schema) -> Value {
  let vector = s.vector.as_ref().map(|v| {
    let index = match v.index {
      IndexKind::Flat => json!({"kind": "flat"}),
      IndexKind::Hnsw { m, ef_construction } => {
        json!({"kind": "hnsw", "m": m, "ef_construction": ef_construction})
      }
    };
    json!({
      "dim": v.dim,
      "metric": metric_name(v.metric),
      "index": index,
      "quantization": quant_name(v.quantization),
    })
  });
  let a = &s.analyzer;
  let stemmer = match &a.stemmer {
    StemmerChoice::None => "none".to_string(),
    StemmerChoice::PorterEnglish => "porter_english".to_string(),
    StemmerChoice::Custom(name) => name.clone(),
  };
  json!({
    "vector": vector,
    "text_fields": s.text_fields,
    "analyzer": {
      "fold_diacritics": a.fold_diacritics,
      "english_stop_words": a.english_stop_words,
      "stemmer": stemmer,
      "max_token_bytes": a.max_token_bytes,
    },
    "filterable": s.filterable.iter().map(|f| json!({
      "name": f.name, "kind": kind_name(f.kind)
    })).collect::<Vec<_>>(),
  })
}

/// The JSON form of collection statistics.
pub fn stats_to_json(s: &CollectionStats) -> Value {
  json!({
    "live_docs": s.live_docs,
    "deleted_docs": s.deleted_docs,
    "memtable_docs": s.memtable_docs,
    "pending_flushes": s.pending_flushes,
    "segments": s.segments.iter().map(|g| json!({
      "id": g.id,
      "docs": g.docs,
      "deleted": g.deleted,
      "vector_index": g.vector_index,
    })).collect::<Vec<_>>(),
    "last_lsn": s.last_lsn,
    "checkpoint_lsn": s.checkpoint_lsn,
  })
}

// ---------------------------------------------------------------------
// Ids and documents
// ---------------------------------------------------------------------

/// A document id as JSON: a string or a non-negative integer.
struct IdIn(Id);

struct IdVisitor;

impl<'de> Visitor<'de> for IdVisitor {
  type Value = IdIn;

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("a document id (a string or a non-negative integer)")
  }

  fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<IdIn, E> {
    Ok(IdIn(Id::U64(v)))
  }

  fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<IdIn, E> {
    u64::try_from(v).map(|v| IdIn(Id::U64(v))).map_err(|_| {
      E::custom(format!(
        "document id {v} is negative; ids are strings or non-negative \
         integers"
      ))
    })
  }

  fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<IdIn, E> {
    Ok(IdIn(Id::Str(v.to_owned())))
  }

  fn visit_string<E: de::Error>(
    self,
    v: String,
  ) -> std::result::Result<IdIn, E> {
    Ok(IdIn(Id::Str(v)))
  }
}

impl<'de> Deserialize<'de> for IdIn {
  fn deserialize<D: Deserializer<'de>>(
    d: D,
  ) -> std::result::Result<Self, D::Error> {
    d.deserialize_any(IdVisitor)
  }
}

fn de_ids<'de, D: Deserializer<'de>>(
  d: D,
) -> std::result::Result<Vec<Id>, D::Error> {
  let v = Vec::<IdIn>::deserialize(d)?;
  Ok(v.into_iter().map(|i| i.0).collect())
}

fn de_opt_ids<'de, D: Deserializer<'de>>(
  d: D,
) -> std::result::Result<Option<Vec<Id>>, D::Error> {
  let v = Option::<Vec<IdIn>>::deserialize(d)?;
  Ok(v.map(|v| v.into_iter().map(|i| i.0).collect()))
}

/// Serializes an id as a JSON number or string.
struct IdOut<'a>(&'a Id);

impl Serialize for IdOut<'_> {
  fn serialize<S: Serializer>(
    &self,
    s: S,
  ) -> std::result::Result<S::Ok, S::Error> {
    match self.0 {
      Id::U64(v) => s.serialize_u64(*v),
      Id::Str(v) => s.serialize_str(v),
    }
  }
}

/// The fields of a document as JSON; moved into a [`Document`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDoc {
  id: IdIn,
  #[serde(default)]
  vector: Option<Vec<f32>>,
  #[serde(default)]
  text: BTreeMap<String, String>,
  #[serde(default)]
  payload: Value,
}

/// A [`Document`] deserialized from JSON. Same layout as `Document`, so a
/// `Vec<DocIn>` turns into a `Vec<Document>` in place.
#[derive(Deserialize)]
#[serde(from = "RawDoc")]
#[repr(transparent)]
struct DocIn(Document);

impl From<RawDoc> for DocIn {
  fn from(r: RawDoc) -> Self {
    DocIn(Document {
      id: r.id.0,
      vector: r.vector,
      text: r.text,
      payload: r.payload,
    })
  }
}

fn into_docs(v: Vec<DocIn>) -> Vec<Document> {
  v.into_iter().map(|d| d.0).collect()
}

/// Parses one document: `{"id": .., "vector": [..], "text": {..},
/// "payload": ..}`.
pub fn parse_document(bytes: &[u8]) -> Result<Document> {
  serde_json::from_slice::<DocIn>(bytes)
    .map(|d| d.0)
    .map_err(|e| json_error("document", e))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpsertBody {
  documents: Vec<DocIn>,
}

/// Parses an upsert body: `{"documents": [document, ...]}`.
pub fn parse_upsert(bytes: &[u8]) -> Result<Vec<Document>> {
  serde_json::from_slice::<UpsertBody>(bytes)
    .map(|b| into_docs(b.documents))
    .map_err(|e| json_error("request body", e))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeleteBody {
  #[serde(deserialize_with = "de_ids")]
  ids: Vec<Id>,
}

/// Parses a delete body: `{"ids": [id, ...]}`.
pub fn parse_delete(bytes: &[u8]) -> Result<Vec<Id>> {
  serde_json::from_slice::<DeleteBody>(bytes)
    .map(|b| b.ids)
    .map_err(|e| json_error("request body", e))
}

/// Lines of NDJSON parsed in one task when parsing in parallel.
const PAR_LINES: usize = 64;

/// Parses newline-delimited documents (one JSON document per line; blank
/// lines are skipped). `first_line` is the 1-based number of the first
/// line in `chunk`, for error messages. Large chunks are parsed on all
/// cores; the order of the documents is kept.
pub fn parse_ndjson(chunk: &[u8], first_line: u64) -> Result<Vec<Document>> {
  let lines: Vec<(u64, &[u8])> = chunk
    .split(|&b| b == b'\n')
    .enumerate()
    .map(|(i, l)| (first_line + i as u64, l))
    .filter(|(_, l)| !l.iter().all(u8::is_ascii_whitespace))
    .collect();
  let parse = |&(n, l): &(u64, &[u8])| {
    serde_json::from_slice::<DocIn>(l)
      .map(|d| d.0)
      .map_err(|e| Error::invalid(format!("line {n}: {e}")))
  };
  if lines.len() >= 2 * PAR_LINES {
    lines
      .par_iter()
      .with_min_len(PAR_LINES)
      .map(parse)
      .collect()
  } else {
    lines.iter().map(parse).collect()
  }
}

/// Parses a document id from a URL path segment. With `id_type` `None`, a
/// segment of decimal digits that fits a `u64` is a numeric id and
/// anything else a string id; `"string"` or `"u64"` force the type.
pub fn parse_path_id(segment: &str, id_type: Option<&str>) -> Result<Id> {
  let num = || {
    segment.parse::<u64>().map(Id::U64).map_err(|_| {
      invalid(format!("id {segment:?} is not a non-negative integer"))
    })
  };
  match id_type {
    None => Ok(match segment.parse::<u64>() {
      Ok(n) if segment.bytes().all(|b| b.is_ascii_digit()) => Id::U64(n),
      _ => Id::Str(segment.to_string()),
    }),
    Some("string") => Ok(Id::Str(segment.to_string())),
    Some("u64") => num(),
    Some(t) => Err(invalid(format!(
      "id_type must be \"string\" or \"u64\", got {t:?}"
    ))),
  }
}

/// Serializes a stored document: `{"id", "vector"?, "text", "payload"?}`.
pub struct DocOut<'a>(pub &'a Document);

impl Serialize for DocOut<'_> {
  fn serialize<S: Serializer>(
    &self,
    s: S,
  ) -> std::result::Result<S::Ok, S::Error> {
    let d = self.0;
    let mut m = s.serialize_map(None)?;
    m.serialize_entry("id", &IdOut(&d.id))?;
    if let Some(v) = &d.vector {
      m.serialize_entry("vector", v)?;
    }
    m.serialize_entry("text", &d.text)?;
    if !d.payload.is_null() {
      m.serialize_entry("payload", &d.payload)?;
    }
    m.end()
  }
}

// ---------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------

/// A search request (see `docs/API.md`). Vector-only, text-only or hybrid
/// depending on which of `vector` and `text` are present.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
  /// Query vector.
  pub vector: Option<Vec<f32>>,
  /// Query text.
  pub text: Option<String>,
  /// Number of results, 1 to [`MAX_K`]; default 10.
  pub k: Option<usize>,
  /// HNSW candidate-list size.
  pub ef: Option<usize>,
  /// Quantised candidates re-scored exactly.
  pub rerank: Option<usize>,
  /// Candidates per side for hybrid queries.
  pub candidates: Option<usize>,
  /// Payload filter, in the JSON filter language.
  pub filter: Option<Value>,
  /// Only these document ids.
  #[serde(default, deserialize_with = "de_opt_ids")]
  pub ids: Option<Vec<Id>>,
  /// Fusion for hybrid queries: `"rrf"`, `{"rrf": {"k": 60}}` or
  /// `{"weighted": {"vector": 0.7, "text": 0.3}}`.
  pub fusion: Option<Value>,
  /// Return each hit's payload. Default false.
  #[serde(default)]
  pub with_payload: bool,
  /// Text fields to search (default: all).
  pub fields: Option<Vec<String>>,
}

/// Parses the `fusion` value of a search request.
pub fn parse_fusion(v: &Value) -> Result<Fusion> {
  let bad = || {
    invalid(
      "fusion must be \"rrf\", {\"rrf\": {\"k\": N}} or {\"weighted\": \
       {\"vector\": W, \"text\": W}}",
    )
  };
  if v.as_str() == Some("rrf") {
    return Ok(Fusion::default());
  }
  let o = v.as_object().ok_or_else(bad)?;
  if o.len() != 1 {
    return Err(bad());
  }
  let (name, p) = o.iter().next().ok_or_else(bad)?;
  match name.as_str() {
    "rrf" => {
      let p = as_object("fusion.rrf", p)?;
      check_keys("fusion.rrf", p, &["k"])?;
      let k = match p.get("k") {
        None => 60,
        Some(k) => u32::try_from(as_usize("fusion.rrf.k", k)?)
          .map_err(|_| invalid("fusion.rrf.k is too large"))?,
      };
      Ok(Fusion::Rrf { k })
    }
    "weighted" => {
      let p = as_object("fusion.weighted", p)?;
      check_keys("fusion.weighted", p, &["vector", "text"])?;
      let w = |key: &str| -> Result<f32> {
        let at = format!("fusion.weighted.{key}");
        let x = p
          .get(key)
          .ok_or_else(|| invalid(format!("{at} is required")))?
          .as_f64()
          .ok_or_else(|| invalid(format!("{at} must be a number")))?
          as f32;
        if !x.is_finite() || x < 0.0 {
          return Err(invalid(format!(
            "{at} must be a finite non-negative number"
          )));
        }
        Ok(x)
      };
      Ok(Fusion::Weighted {
        vector: w("vector")?,
        text: w("text")?,
      })
    }
    _ => Err(bad()),
  }
}

impl SearchRequest {
  /// Parses a request body.
  pub fn parse(bytes: &[u8]) -> Result<SearchRequest> {
    serde_json::from_slice(bytes).map_err(|e| json_error("request body", e))
  }

  /// Validates the request and builds the engine query.
  pub fn into_query(self) -> Result<Query> {
    let k = self.k.unwrap_or(10);
    if !(1..=MAX_K).contains(&k) {
      return Err(invalid(format!("k must be between 1 and {MAX_K}, got {k}")));
    }
    let mut q = match (self.vector, self.text) {
      (Some(v), Some(t)) => Query::hybrid(v, t),
      (Some(v), None) => Query::vector(v),
      (None, Some(t)) => Query::text(t),
      (None, None) => {
        return Err(invalid("a search needs \"vector\", \"text\" or both"));
      }
    };
    q = q.k(k).with_payload(self.with_payload);
    if let Some(ef) = self.ef {
      q = q.ef(ef);
    }
    if let Some(n) = self.rerank {
      q = q.rerank(n);
    }
    if let Some(n) = self.candidates {
      q = q.candidates(n);
    }
    if let Some(f) = &self.filter {
      q = q.filter(Filter::from_json(f)?);
    }
    if let Some(ids) = self.ids {
      q = q.id_filter(IdFilter::ids(ids));
    }
    if let Some(f) = &self.fusion {
      q = q.fusion(parse_fusion(f)?);
    }
    if let Some(fields) = self.fields {
      q = q.fields(fields);
    }
    Ok(q)
  }
}

/// One hit as JSON.
struct HitOut<'a>(&'a SearchHit);

impl Serialize for HitOut<'_> {
  fn serialize<S: Serializer>(
    &self,
    s: S,
  ) -> std::result::Result<S::Ok, S::Error> {
    let h = self.0;
    let mut m = s.serialize_map(None)?;
    m.serialize_entry("id", &IdOut(&h.id))?;
    m.serialize_entry("score", &h.score)?;
    m.serialize_entry("vector_score", &h.vector_score)?;
    m.serialize_entry("text_score", &h.text_score)?;
    if let Some(p) = &h.payload {
      m.serialize_entry("payload", p)?;
    }
    m.end()
  }
}

struct HitsOut<'a>(&'a [SearchHit]);

impl Serialize for HitsOut<'_> {
  fn serialize<S: Serializer>(
    &self,
    s: S,
  ) -> std::result::Result<S::Ok, S::Error> {
    s.collect_seq(self.0.iter().map(HitOut))
  }
}

/// A search response: `{"hits": [...], "took_us": N}`, serialized
/// directly from the engine's hits.
pub struct SearchResponse<'a> {
  /// The hits, best first.
  pub hits: &'a [SearchHit],
  /// Server-side time to parse the request and search, in microseconds.
  pub took_us: u64,
}

impl Serialize for SearchResponse<'_> {
  fn serialize<S: Serializer>(
    &self,
    s: S,
  ) -> std::result::Result<S::Ok, S::Error> {
    let mut m = s.serialize_map(Some(2))?;
    m.serialize_entry("hits", &HitsOut(self.hits))?;
    m.serialize_entry("took_us", &self.took_us)?;
    m.end()
  }
}

/// Serialized size estimate of a search response, to size its buffer.
pub fn response_capacity(hits: &[SearchHit]) -> usize {
  let payload = hits
    .first()
    .map_or(0, |h| h.payload.as_ref().map_or(0, |_| 128));
  32 + hits.len() * (96 + payload)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn schema_round_trip() {
    let s = schema_from_json(&json!({
      "vector": {"dim": 4, "metric": "dot",
                 "index": {"kind": "hnsw", "m": 8, "ef_construction": 100},
                 "quantization": "sq8"},
      "text_fields": ["title", "body"],
      "analyzer": "english",
      "filterable": [{"name": "lang", "kind": "keyword"},
                     {"name": "year", "kind": "integer"}]
    }))
    .unwrap();
    assert_eq!(schema_from_json(&schema_to_json(&s)).unwrap(), s);
    let t = schema_from_json(&json!({
      "text_fields": ["a"],
      "filterable": {"x": "bool", "y": "float"}
    }))
    .unwrap();
    assert_eq!(t.filterable.len(), 2);
    assert_eq!(schema_from_json(&schema_to_json(&t)).unwrap(), t);
  }

  #[test]
  fn schema_errors_name_the_place() {
    let e = |v: Value| schema_from_json(&v).unwrap_err().to_string();
    assert!(e(json!({"vector": {"dim": 3, "metrc": "l2"}})).contains("metrc"));
    assert!(e(json!({"vector": {"dim": "3"}})).contains("schema.vector.dim"));
    assert!(e(json!({"text_fields": [1]})).contains("text_fields[0]"));
    assert!(e(json!({"vector": {"dim": 0}})).contains("dimension"));
    assert!(e(json!({})).contains("vector field or at least one text"));
    assert!(e(json!([])).contains("must be an object"));
  }

  #[test]
  fn documents_and_ids() {
    let d = parse_document(
      br#"{"id": 7, "vector": [1, 2.5], "text": {"t": "x"},
           "payload": {"a": 1}}"#,
    )
    .unwrap();
    assert_eq!(d.id, Id::U64(7));
    assert_eq!(d.vector, Some(vec![1.0, 2.5]));
    assert_eq!(d.payload, json!({"a": 1}));
    assert_eq!(parse_document(br#"{"id": "x"}"#).unwrap().id, "x".into());
    assert!(parse_document(br#"{"id": -1}"#).is_err());
    assert!(parse_document(br#"{"id": 1.5}"#).is_err());
    assert!(parse_document(br#"{"id": 1, "extra": 0}"#).is_err());
    assert_eq!(parse_path_id("12", None).unwrap(), Id::U64(12));
    assert_eq!(parse_path_id("+12", None).unwrap(), Id::Str("+12".into()));
    assert_eq!(parse_path_id("12", Some("string")).unwrap(), "12".into());
    assert!(parse_path_id("ab", Some("u64")).is_err());
    let docs = parse_ndjson(b"{\"id\":1}\n\n{\"id\":2}\r\n", 1).unwrap();
    assert_eq!(docs.len(), 2);
    let e = parse_ndjson(b"{\"id\":1}\n\n{\"id\":}\n", 10).unwrap_err();
    assert!(e.to_string().contains("line 12"), "{e}");
    let many: String = (0..1000).map(|i| format!("{{\"id\":{i}}}\n")).collect();
    let docs = parse_ndjson(many.as_bytes(), 1).unwrap();
    assert!(
      docs
        .iter()
        .enumerate()
        .all(|(i, d)| d.id == Id::U64(i as u64))
    );
  }

  #[test]
  fn doc_out_round_trips() {
    let d = Document::new("a")
      .vector(vec![0.5])
      .text("t", "hello")
      .payload(json!({"x": [1, 2]}));
    let s = serde_json::to_vec(&DocOut(&d)).unwrap();
    assert_eq!(parse_document(&s).unwrap(), d);
  }

  #[test]
  fn fusion_forms() {
    assert_eq!(parse_fusion(&json!("rrf")).unwrap(), Fusion::Rrf { k: 60 });
    assert_eq!(
      parse_fusion(&json!({"rrf": {"k": 10}})).unwrap(),
      Fusion::Rrf { k: 10 }
    );
    assert_eq!(
      parse_fusion(&json!({"weighted": {"vector": 0.7, "text": 0.3}})).unwrap(),
      Fusion::Weighted {
        vector: 0.7,
        text: 0.3
      }
    );
    assert!(
      parse_fusion(&json!({"weighted": {"vector": -1, "text": 0}})).is_err()
    );
    assert!(parse_fusion(&json!({"rrf": {}, "weighted": {}})).is_err());
    assert!(parse_fusion(&json!("nope")).is_err());
  }

  #[test]
  fn search_request_validation() {
    let q = |s: &str| SearchRequest::parse(s.as_bytes())?.into_query();
    assert!(q(r#"{"text": "a"}"#).is_ok());
    assert!(q(r#"{"vector": [1.0], "text": "a", "k": 3}"#).is_ok());
    assert!(q(r#"{}"#).unwrap_err().to_string().contains("needs"));
    assert!(q(r#"{"text": "a", "k": 0}"#).is_err());
    assert!(q(r#"{"text": "a", "k": 10001}"#).is_err());
    assert!(q(r#"{"text": "a", "kk": 1}"#).is_err());
    let e = q(r#"{"text": "a", "filter": {"eq": 1}}"#).unwrap_err();
    assert!(e.to_string().contains("filter at $"), "{e}");
    assert!(q(r#"{"text": "a", "ids": [1, "b"]}"#).is_ok());
  }
}
