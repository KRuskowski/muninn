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

  /// The stored form, kept in every manifest.
  pub(crate) fn to_json(&self) -> Value {
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

  /// Parses [`Schema::to_json`].
  pub(crate) fn from_json(v: &Value) -> Result<Self> {
    let bad = |what: &str| Error::corrupt(format!("schema: bad {what}"));
    if v.get("format").and_then(Value::as_u64) != Some(SCHEMA_FORMAT) {
      return Err(bad("format version"));
    }
    let vector = match v.get("vector") {
      None | Some(Value::Null) => None,
      Some(f) => {
        let dim = f
          .get("dim")
          .and_then(Value::as_u64)
          .ok_or_else(|| bad("dim"))? as usize;
        let metric = f
          .get("metric")
          .and_then(Value::as_str)
          .and_then(Metric::parse)
          .ok_or_else(|| bad("metric"))?;
        let quantization = f
          .get("quantization")
          .and_then(Value::as_str)
          .and_then(parse_quant)
          .ok_or_else(|| bad("quantization"))?;
        let ix = f.get("index").ok_or_else(|| bad("index"))?;
        let index = match ix.get("kind").and_then(Value::as_str) {
          Some("flat") => IndexKind::Flat,
          Some("hnsw") => IndexKind::Hnsw {
            m: ix
              .get("m")
              .and_then(Value::as_u64)
              .ok_or_else(|| bad("m"))? as usize,
            ef_construction: ix
              .get("ef_construction")
              .and_then(Value::as_u64)
              .ok_or_else(|| bad("ef_construction"))?
              as usize,
          },
          _ => return Err(bad("index kind")),
        };
        Some(VectorField {
          dim,
          metric,
          index,
          quantization,
        })
      }
    };
    let text_fields = v
      .get("text_fields")
      .and_then(Value::as_array)
      .ok_or_else(|| bad("text_fields"))?
      .iter()
      .map(|f| {
        f.as_str()
          .map(String::from)
          .ok_or_else(|| bad("text field"))
      })
      .collect::<Result<Vec<_>>>()?;
    let a = v.get("analyzer").ok_or_else(|| bad("analyzer"))?;
    let flag = |name: &str| {
      a.get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| bad(name))
    };
    let stemmer = match a.get("stemmer").and_then(Value::as_str) {
      Some("none") => StemmerChoice::None,
      Some("porter_english") => StemmerChoice::PorterEnglish,
      _ => return Err(bad("stemmer")),
    };
    let max_token_bytes = a
      .get("max_token_bytes")
      .and_then(Value::as_u64)
      .and_then(|n| u16::try_from(n).ok())
      .ok_or_else(|| bad("max_token_bytes"))?;
    let analyzer = AnalyzerConfig {
      fold_diacritics: flag("fold_diacritics")?,
      english_stop_words: flag("english_stop_words")?,
      stemmer,
      max_token_bytes,
    };
    let filterable = v
      .get("filterable")
      .and_then(Value::as_array)
      .ok_or_else(|| bad("filterable"))?
      .iter()
      .map(|f| {
        let name = f.get("name").and_then(Value::as_str);
        let kind = f
          .get("kind")
          .and_then(Value::as_str)
          .and_then(FieldKind::parse);
        match (name, kind) {
          (Some(n), Some(k)) => Ok(FieldSpec {
            name: n.into(),
            kind: k,
          }),
          _ => Err(bad("filterable field")),
        }
      })
      .collect::<Result<Vec<_>>>()?;
    let s = Schema {
      vector,
      text_fields,
      analyzer,
      filterable,
    };
    s.validate()
      .map_err(|e| Error::corrupt(format!("schema: {e}")))?;
    Ok(s)
  }
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
  }
}
