//! Collection schemas and their stored (JSON) form.

use muninn_core::{Error, Metric, Result};
use muninn_text::{AnalyzerConfig, StemmerChoice};
use muninn_vector::Quantization;
use serde_json::{Value, json};

/// Version of the schema JSON stored in every manifest.
const SCHEMA_FORMAT: u64 = 1;

/// What a collection stores and how it is indexed. Fixed at creation.
#[derive(Clone, Debug, PartialEq)]
pub struct Schema {
  /// The vector field, if documents carry vectors.
  pub vector: Option<VectorField>,
  /// Names of the full-text fields, each with its own inverted index.
  pub text_fields: Vec<String>,
  /// How text is split into terms (for every text field). Custom
  /// stemmers cannot be stored in a schema; use the built-in choices.
  pub analyzer: AnalyzerConfig,
  /// Payload fields that filters can use (see [`crate::Filter`]). Each
  /// gets a payload index in every segment. A name is a dot-separated
  /// path into the payload (`"meta.lang"`); arrays along the path are
  /// flattened, so fields can hold several values.
  pub filterable: Vec<FieldSpec>,
}

impl Schema {
  /// A schema with a vector field and nothing else.
  pub fn vector(field: VectorField) -> Self {
    Self {
      vector: Some(field),
      text_fields: Vec::new(),
      analyzer: AnalyzerConfig::default(),
      filterable: Vec::new(),
    }
  }

  /// A schema with only text fields.
  pub fn text<S: Into<String>>(fields: impl IntoIterator<Item = S>) -> Self {
    Self {
      vector: None,
      text_fields: fields.into_iter().map(Into::into).collect(),
      analyzer: AnalyzerConfig::default(),
      filterable: Vec::new(),
    }
  }

  /// Adds text fields.
  pub fn with_text_fields<S: Into<String>>(
    mut self,
    fields: impl IntoIterator<Item = S>,
  ) -> Self {
    self.text_fields.extend(fields.into_iter().map(Into::into));
    self
  }

  /// Sets the analyzer.
  pub fn with_analyzer(mut self, analyzer: AnalyzerConfig) -> Self {
    self.analyzer = analyzer;
    self
  }

  /// Declares a filterable payload field.
  pub fn with_filterable(mut self, field: FieldSpec) -> Self {
    self.filterable.push(field);
    self
  }

  /// The payload-index schema of the filterable fields.
  pub(crate) fn filter_schema(&self) -> Result<muninn_filter::Schema> {
    let mut fs = muninn_filter::Schema::new();
    for f in &self.filterable {
      fs.add(&f.name, f.kind.filter_type())?;
    }
    Ok(fs)
  }

  /// Index of text field `name`.
  pub(crate) fn text_field(&self, name: &str) -> Option<usize> {
    self.text_fields.iter().position(|f| f == name)
  }

  /// Checks the schema is usable.
  pub fn validate(&self) -> Result<()> {
    if let Some(v) = &self.vector {
      if v.dim == 0 || v.dim > 65_536 {
        return Err(Error::invalid(format!(
          "vector dimension must be in 1..=65536, got {}",
          v.dim
        )));
      }
      if let IndexKind::Hnsw { m, ef_construction } = v.index {
        if !(2..=muninn_vector::MAX_M).contains(&m) {
          return Err(Error::invalid(format!(
            "HNSW m must be in 2..={}, got {m}",
            muninn_vector::MAX_M
          )));
        }
        if ef_construction == 0 {
          return Err(Error::invalid("HNSW ef_construction must be positive"));
        }
      }
    }
    if self.vector.is_none() && self.text_fields.is_empty() {
      return Err(Error::invalid(
        "a schema needs a vector field or at least one text field",
      ));
    }
    for (i, f) in self.text_fields.iter().enumerate() {
      check_name(f, "text field")?;
      if self.text_fields[..i].contains(f) {
        return Err(Error::invalid(format!("text field {f:?} declared twice")));
      }
    }
    for (i, f) in self.filterable.iter().enumerate() {
      check_name(&f.name, "filterable field")?;
      if self.filterable[..i].iter().any(|g| g.name == f.name) {
        return Err(Error::invalid(format!(
          "filterable field {:?} declared twice",
          f.name
        )));
      }
    }
    self.filter_schema()?;
    if let StemmerChoice::Custom(name) = &self.analyzer.stemmer {
      return Err(Error::invalid(format!(
        "custom stemmer {name:?} cannot be stored in a schema"
      )));
    }
    muninn_text::Analyzer::new(self.analyzer.clone())?;
    Ok(())
  }

  /// The canonical JSON form, as stored in every manifest, with every
  /// setting spelled out:
  ///
  /// ```json
  /// {
  ///   "format": 1,
  ///   "vector": {"dim": 3, "metric": "cosine",
  ///              "index": {"kind": "hnsw", "m": 16, "ef_construction": 200},
  ///              "quantization": "none"},
  ///   "text_fields": ["title"],
  ///   "analyzer": {"fold_diacritics": false, "english_stop_words": false,
  ///                "stemmer": "none", "max_token_bytes": 255},
  ///   "filterable": [{"name": "lang", "kind": "keyword"}]
  /// }
  /// ```
  ///
  /// `metric` is `l2`, `dot` or `cosine`; `index.kind` is `hnsw` or
  /// `flat`; `quantization` is `none`, `sq8` or `binary`; `stemmer` is
  /// `none` or `porter_english`; a filterable `kind` is `keyword`,
  /// `integer`, `float` or `bool`. [`Schema::from_json`] reads it back.
  pub fn to_json(&self) -> Value {
    let vector = self.vector.as_ref().map(|v| {
      let index = match v.index {
        IndexKind::Flat => json!({"kind": "flat"}),
        IndexKind::Hnsw { m, ef_construction } => json!({
          "kind": "hnsw", "m": m, "ef_construction": ef_construction
        }),
      };
      json!({
        "dim": v.dim,
        "metric": metric_name(v.metric),
        "index": index,
        "quantization": quant_name(v.quantization),
      })
    });
    let a = &self.analyzer;
    let stemmer = match &a.stemmer {
      StemmerChoice::None => "none",
      StemmerChoice::PorterEnglish => "porter_english",
      // Rejected by validate().
      StemmerChoice::Custom(_) => "custom",
    };
    json!({
      "format": SCHEMA_FORMAT,
      "vector": vector,
      "text_fields": self.text_fields,
      "analyzer": {
        "fold_diacritics": a.fold_diacritics,
        "english_stop_words": a.english_stop_words,
        "stemmer": stemmer,
        "max_token_bytes": a.max_token_bytes,
      },
      "filterable": self.filterable.iter().map(|f| json!({
        "name": f.name, "kind": f.kind.name()
      })).collect::<Vec<_>>(),
    })
  }

  /// Parses the JSON form written by [`Schema::to_json`] and validates
  /// the schema. Keys may be left out: `format` (the current one),
  /// `vector` (none), `text_fields` and `filterable` (empty), `analyzer`
  /// and each of its settings ([`AnalyzerConfig::default`]),
  /// `vector.index` (HNSW with `m = 16`, `ef_construction = 200`) and
  /// `vector.quantization` (`none`). Unknown keys are refused, so a typo
  /// does not silently fall back to a default. Errors are
  /// [`Error::InvalidArgument`] naming the offending key.
  pub fn from_json(v: &Value) -> Result<Self> {
    let o = object(v, "schema")?;
    known(
      o,
      "schema",
      &["format", "vector", "text_fields", "analyzer", "filterable"],
    )?;
    match o.get("format") {
      None => {}
      Some(f) if f.as_u64() == Some(SCHEMA_FORMAT) => {}
      Some(f) => {
        return Err(Error::invalid(format!(
          "schema.format {f} is not supported (expected {SCHEMA_FORMAT})"
        )));
      }
    }
    let vector = match o.get("vector") {
      None | Some(Value::Null) => None,
      Some(f) => Some(parse_vector(f)?),
    };
    let text_fields = match o.get("text_fields") {
      None | Some(Value::Null) => Vec::new(),
      Some(Value::Array(a)) => a
        .iter()
        .enumerate()
        .map(|(i, f)| {
          f.as_str().map(String::from).ok_or_else(|| {
            Error::invalid(format!("schema.text_fields[{i}] must be a string"))
          })
        })
        .collect::<Result<Vec<_>>>()?,
      Some(_) => {
        return Err(Error::invalid(
          "schema.text_fields must be an array of strings",
        ));
      }
    };
    let analyzer = match o.get("analyzer") {
      None | Some(Value::Null) => AnalyzerConfig::default(),
      Some(a) => parse_analyzer(a)?,
    };
    let filterable = match o.get("filterable") {
      None | Some(Value::Null) => Vec::new(),
      Some(Value::Array(a)) => a
        .iter()
        .enumerate()
        .map(|(i, f)| parse_field(f, &format!("schema.filterable[{i}]")))
        .collect::<Result<Vec<_>>>()?,
      Some(_) => {
        return Err(Error::invalid("schema.filterable must be an array"));
      }
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

  /// Parses the schema stored in a manifest; any problem means the file
  /// is damaged.
  pub(crate) fn from_stored(v: &Value) -> Result<Self> {
    if v.get("format").and_then(Value::as_u64) != Some(SCHEMA_FORMAT) {
      return Err(Error::corrupt("schema: bad format version"));
    }
    Self::from_json(v).map_err(|e| match e {
      Error::InvalidArgument(m) => Error::corrupt(format!("stored {m}")),
      e => e,
    })
  }
}

fn object<'a>(
  v: &'a Value,
  at: &str,
) -> Result<&'a serde_json::Map<String, Value>> {
  v.as_object()
    .ok_or_else(|| Error::invalid(format!("{at} must be an object")))
}

/// Refuses keys of `o` (at path `at`) not in `allowed`.
fn known(
  o: &serde_json::Map<String, Value>,
  at: &str,
  allowed: &[&str],
) -> Result<()> {
  match o.keys().find(|k| !allowed.contains(&k.as_str())) {
    Some(k) => Err(Error::invalid(format!(
      "{at} has unknown key {k:?} (known: {})",
      allowed.join(", ")
    ))),
    None => Ok(()),
  }
}

fn string<'a>(v: Option<&'a Value>, at: &str) -> Result<&'a str> {
  v.and_then(Value::as_str)
    .ok_or_else(|| Error::invalid(format!("{at} must be a string")))
}

fn uint(v: Option<&Value>, at: &str) -> Result<u64> {
  v.and_then(Value::as_u64).ok_or_else(|| {
    Error::invalid(format!("{at} must be a non-negative integer"))
  })
}

fn usize_of(v: Option<&Value>, at: &str) -> Result<usize> {
  usize::try_from(uint(v, at)?)
    .map_err(|_| Error::invalid(format!("{at} is too large")))
}

fn parse_vector(v: &Value) -> Result<VectorField> {
  let o = object(v, "schema.vector")?;
  known(
    o,
    "schema.vector",
    &["dim", "metric", "index", "quantization"],
  )?;
  let dim = usize_of(o.get("dim"), "schema.vector.dim")?;
  let m = string(o.get("metric"), "schema.vector.metric")?;
  let metric = Metric::parse(m).ok_or_else(|| {
    Error::invalid(format!(
      "schema.vector.metric {m:?} is not one of l2, dot, cosine"
    ))
  })?;
  let quantization = match o.get("quantization") {
    None | Some(Value::Null) => Quantization::None,
    q => {
      let q = string(q, "schema.vector.quantization")?;
      parse_quant(q).ok_or_else(|| {
        Error::invalid(format!(
          "schema.vector.quantization {q:?} is not one of none, sq8, binary"
        ))
      })?
    }
  };
  let index = match o.get("index") {
    None | Some(Value::Null) => VectorField::hnsw(dim, metric).index,
    Some(ix) => {
      let at = "schema.vector.index";
      let io = object(ix, at)?;
      known(io, at, &["kind", "m", "ef_construction"])?;
      match string(io.get("kind"), "schema.vector.index.kind")? {
        "flat" => {
          known(io, at, &["kind"])?;
          IndexKind::Flat
        }
        "hnsw" => IndexKind::Hnsw {
          m: match io.get("m") {
            None => 16,
            m => usize_of(m, "schema.vector.index.m")?,
          },
          ef_construction: match io.get("ef_construction") {
            None => 200,
            e => usize_of(e, "schema.vector.index.ef_construction")?,
          },
        },
        k => {
          return Err(Error::invalid(format!(
            "schema.vector.index.kind {k:?} is not one of hnsw, flat"
          )));
        }
      }
    }
  };
  Ok(VectorField {
    dim,
    metric,
    index,
    quantization,
  })
}

fn parse_analyzer(v: &Value) -> Result<AnalyzerConfig> {
  let at = "schema.analyzer";
  let o = object(v, at)?;
  known(
    o,
    at,
    &[
      "fold_diacritics",
      "english_stop_words",
      "stemmer",
      "max_token_bytes",
    ],
  )?;
  let mut a = AnalyzerConfig::default();
  let flag = |name: &str| -> Result<Option<bool>> {
    match o.get(name) {
      None => Ok(None),
      Some(b) => b.as_bool().map(Some).ok_or_else(|| {
        Error::invalid(format!("{at}.{name} must be true or false"))
      }),
    }
  };
  if let Some(b) = flag("fold_diacritics")? {
    a.fold_diacritics = b;
  }
  if let Some(b) = flag("english_stop_words")? {
    a.english_stop_words = b;
  }
  if let Some(st) = o.get("stemmer") {
    a.stemmer = match string(Some(st), "schema.analyzer.stemmer")? {
      "none" => StemmerChoice::None,
      "porter_english" => StemmerChoice::PorterEnglish,
      x => {
        return Err(Error::invalid(format!(
          "schema.analyzer.stemmer {x:?} is not one of none, porter_english"
        )));
      }
    };
  }
  if let Some(n) = o.get("max_token_bytes") {
    let at = "schema.analyzer.max_token_bytes";
    a.max_token_bytes = u16::try_from(uint(Some(n), at)?)
      .map_err(|_| Error::invalid(format!("{at} must be at most 65535")))?;
  }
  Ok(a)
}

fn parse_field(v: &Value, at: &str) -> Result<FieldSpec> {
  let o = object(v, at)?;
  known(o, at, &["name", "kind"])?;
  let name = string(o.get("name"), &format!("{at}.name"))?;
  let k = string(o.get("kind"), &format!("{at}.kind"))?;
  let kind = FieldKind::parse(k).ok_or_else(|| {
    Error::invalid(format!(
      "{at}.kind {k:?} is not one of keyword, integer, float, bool"
    ))
  })?;
  Ok(FieldSpec {
    name: name.into(),
    kind,
  })
}

fn check_name(name: &str, what: &str) -> Result<()> {
  if name.is_empty() || name.len() > 255 {
    return Err(Error::invalid(format!(
      "{what} name must be 1 to 255 bytes long"
    )));
  }
  Ok(())
}

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

fn parse_quant(s: &str) -> Option<Quantization> {
  match s {
    "none" => Some(Quantization::None),
    "sq8" => Some(Quantization::Sq8),
    "binary" => Some(Quantization::Binary),
    _ => None,
  }
}

/// The vector field of a schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorField {
  /// Dimensions of every vector.
  pub dim: usize,
  /// How vectors are compared.
  pub metric: Metric,
  /// Index built for segments. The in-memory part of a collection is
  /// always searched exactly.
  pub index: IndexKind,
  /// Compression of segment vectors for the first search pass; the best
  /// candidates are re-scored with the full vectors.
  pub quantization: Quantization,
}

impl VectorField {
  /// An exact (flat) vector field.
  pub fn flat(dim: usize, metric: Metric) -> Self {
    Self {
      dim,
      metric,
      index: IndexKind::Flat,
      quantization: Quantization::None,
    }
  }

  /// An HNSW vector field with `m = 16`, `ef_construction = 200`.
  pub fn hnsw(dim: usize, metric: Metric) -> Self {
    Self {
      dim,
      metric,
      index: IndexKind::Hnsw {
        m: 16,
        ef_construction: 200,
      },
      quantization: Quantization::None,
    }
  }

  /// Sets the quantisation.
  pub fn with_quantization(mut self, q: Quantization) -> Self {
    self.quantization = q;
    self
  }
}

/// How segment vectors are indexed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexKind {
  /// Exact scan. Best below roughly 50 000 vectors per collection.
  Flat,
  /// HNSW graph: approximate, fast at any size. Segments smaller than
  /// [`crate::Options::hnsw_min_docs`] are still scanned exactly.
  Hnsw {
    /// Neighbours per node (level 0 keeps `2 * m`).
    m: usize,
    /// Candidate-list size during construction.
    ef_construction: usize,
  },
}

/// A payload field declared for filtering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldSpec {
  /// Dot-separated path into the payload, e.g. `"meta.lang"`.
  pub name: String,
  /// Value type.
  pub kind: FieldKind,
}

impl FieldSpec {
  /// A keyword (exact string match) field.
  pub fn keyword(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      kind: FieldKind::Keyword,
    }
  }

  /// An integer field.
  pub fn integer(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      kind: FieldKind::Integer,
    }
  }

  /// A floating-point field.
  pub fn float(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      kind: FieldKind::Float,
    }
  }

  /// A boolean field.
  pub fn boolean(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      kind: FieldKind::Bool,
    }
  }
}

/// Value type of a [`FieldSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
  /// Strings matched exactly.
  Keyword,
  /// 64-bit integers, for equality and ranges.
  Integer,
  /// 64-bit floats, for equality and ranges.
  Float,
  /// `true` / `false`.
  Bool,
}

impl FieldKind {
  fn name(self) -> &'static str {
    match self {
      Self::Keyword => "keyword",
      Self::Integer => "integer",
      Self::Float => "float",
      Self::Bool => "bool",
    }
  }

  fn filter_type(self) -> muninn_filter::FieldType {
    match self {
      Self::Keyword => muninn_filter::FieldType::Keyword,
      Self::Integer => muninn_filter::FieldType::Integer,
      Self::Float => muninn_filter::FieldType::Float,
      Self::Bool => muninn_filter::FieldType::Bool,
    }
  }

  fn parse(s: &str) -> Option<Self> {
    match s {
      "keyword" => Some(Self::Keyword),
      "integer" => Some(Self::Integer),
      "float" => Some(Self::Float),
      "bool" => Some(Self::Bool),
      _ => None,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn json_round_trip() {
    let s = Schema::vector(
      VectorField::hnsw(64, Metric::Cosine)
        .with_quantization(Quantization::Sq8),
    )
    .with_text_fields(["title", "body"])
    .with_analyzer(AnalyzerConfig::english())
    .with_filterable(FieldSpec::keyword("lang"))
    .with_filterable(FieldSpec::integer("year"))
    .with_filterable(FieldSpec::float("price"))
    .with_filterable(FieldSpec::boolean("meta.ok"));
    s.validate().unwrap();
    assert_eq!(Schema::from_json(&s.to_json()).unwrap(), s);
    let t = Schema::text(["a"]);
    assert_eq!(Schema::from_json(&t.to_json()).unwrap(), t);
  }

  #[test]
  fn validation() {
    assert!(Schema::text(Vec::<String>::new()).validate().is_err());
    assert!(Schema::text(["a", "a"]).validate().is_err());
    assert!(
      Schema::vector(VectorField::flat(0, Metric::L2))
        .validate()
        .is_err()
    );
    let mut s = Schema::text(["a"]);
    s.analyzer.stemmer = StemmerChoice::Custom("x".into());
    assert!(s.validate().is_err());
    let mut bad = Schema::text(["a"]).to_json();
    bad["format"] = json!(99);
    assert!(Schema::from_json(&bad).is_err());
    assert!(Schema::from_stored(&bad).is_err());
  }

  #[test]
  fn json_defaults_and_errors() {
    let s = Schema::from_json(&json!({
      "vector": {"dim": 3, "metric": "cosine"},
      "text_fields": ["title"],
    }))
    .unwrap();
    assert_eq!(
      s,
      Schema::vector(VectorField::hnsw(3, Metric::Cosine))
        .with_text_fields(["title"])
    );
    // The stored form needs its format version.
    assert!(matches!(
      Schema::from_stored(&json!({"text_fields": ["a"]})),
      Err(Error::Corrupt(_))
    ));
    let err = |v: Value| match Schema::from_json(&v) {
      Err(Error::InvalidArgument(m)) => m,
      other => panic!("{other:?}"),
    };
    assert!(err(json!({"text_fields": ["a"], "vectr": {}})).contains("vectr"));
    assert!(
      err(json!({"vector": {"dim": 3, "metric": "manhattan"}}))
        .contains("schema.vector.metric")
    );
    assert!(
      err(json!({"vector": {"dim": 3, "metric": "l2",
                 "index": {"kind": "flat", "m": 4}}}))
      .contains("\"m\"")
    );
    assert!(
      err(json!({"text_fields": ["a"],
                 "filterable": [{"name": "x", "kind": "date"}]}))
      .contains("schema.filterable[0].kind")
    );
    assert!(err(json!({})).contains("vector field"));
  }
}
