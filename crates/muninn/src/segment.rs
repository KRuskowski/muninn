//! Immutable segments: building, writing and opening them.
//!
//! A segment is a set of container files named by the manifest:
//!
//! | role           | file                 | contents                       |
//! |----------------|----------------------|--------------------------------|
//! | `vectors`      | `seg-<id>.vec`       | `FlatIndex` or `HnswIndex`     |
//! | `text:<field>` | `seg-<id>.t<i>`      | `TextIndex` of text field `i`  |
//! | `docs`         | `seg-<id>.docs`      | stored records (`payload` file)|
//! | `ids`          | `seg-<id>.ids`       | `FrozenIdMap`                  |
//! | `pindex`       | `seg-<id>.pidx`      | payload index (`FrozenIndex`), |
//! |                |                      | if the schema has filterable   |
//! |                |                      | fields                         |
//!
//! Every file is written to a temporary name, fsynced and renamed, so a
//! file under its final name is always complete. Doc ids are dense and in
//! insertion order: a flushed memtable keeps its doc ids exactly, and a
//! merge concatenates its inputs in collection order, dropping deleted
//! docs. Order across and within segments therefore always follows the
//! order documents were written, which is how result ties are broken.

use std::collections::BTreeMap;
use std::path::Path;

use muninn_core::container::Container;
use muninn_core::{BitSet, DocId, Error, Result};
use muninn_filter::FrozenIndex;
use muninn_store::fs::write_atomic_with;
use muninn_store::{
  Advice, FrozenIdMap, IdMap, PayloadStore, SegmentEntry, map_file,
  map_file_with, segment_file_name, write_payloads,
};
use muninn_text::{Analyzer, DELETED, MergeInput, TextIndex, TextSearcher};
use muninn_vector::{
  FLAT_KIND, FlatIndex, HNSW_KIND, HnswIndex, HnswParams, Quantization,
  VectorIndex,
};
use serde_json::json;

use crate::memtable::{MemInner, Memtable};
use crate::options::Options;
use crate::schema::{IndexKind, Schema};

/// A segment's vector index. One per segment, so the size difference
/// between the variants does not matter.
#[allow(clippy::large_enum_variant)]
pub(crate) enum VecIndex {
  Flat(FlatIndex),
  Hnsw(HnswIndex),
}

impl VecIndex {
  pub(crate) fn index(&self) -> &dyn VectorIndex {
    match self {
      Self::Flat(f) => f,
      Self::Hnsw(h) => h,
    }
  }

  /// The stored (normalised for cosine) vector of `doc`.
  pub(crate) fn vector(&self, doc: DocId) -> &[f32] {
    match self {
      Self::Flat(f) => f.vector(doc),
      Self::Hnsw(h) => h.vector(doc),
    }
  }

  pub(crate) fn kind_name(&self) -> &'static str {
    match self {
      Self::Flat(_) => "flat",
      Self::Hnsw(_) => "hnsw",
    }
  }

  fn write(&self, path: &Path) -> Result<()> {
    write_atomic_with(path, |w| {
      match self {
        Self::Flat(f) => f.write_to(w)?,
        Self::Hnsw(h) => h.write_to(w)?,
      };
      Ok(())
    })?;
    Ok(())
  }
}

/// An open segment.
pub(crate) struct Segment {
  pub id: u64,
  pub len: u32,
  pub vectors: Option<VecIndex>,
  pub text: Vec<TextIndex>,
  pub stored: PayloadStore,
  pub ids: FrozenIdMap,
  pub pindex: Option<FrozenIndex>,
  /// The manifest entry, without tombstones (tracked by the engine).
  pub entry: SegmentEntry,
}

impl std::fmt::Debug for Segment {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Segment")
      .field("id", &self.id)
      .field("len", &self.len)
      .finish()
  }
}

/// Builds the vector index of a new segment from row-major `vectors`:
/// HNSW if the schema asks for it, `graph` is set and the segment is
/// large enough, else flat.
fn build_vectors(
  schema: &Schema,
  opts: &Options,
  vectors: &[f32],
  graph: bool,
) -> Result<Option<VecIndex>> {
  let Some(f) = &schema.vector else {
    return Ok(None);
  };
  let n = vectors.len() / f.dim;
  let idx = match f.index {
    IndexKind::Hnsw { m, ef_construction }
      if graph && n >= opts.hnsw_min_docs =>
    {
      let p = HnswParams::default()
        .with_m(m)
        .with_ef_construction(ef_construction)
        .with_quantization(f.quantization);
      VecIndex::Hnsw(HnswIndex::build(f.dim, f.metric, p, vectors)?)
    }
    _ => {
      let mut flat = FlatIndex::from_vectors(f.dim, f.metric, vectors)?;
      if f.quantization != Quantization::None {
        flat.quantize(f.quantization);
      }
      VecIndex::Flat(flat)
    }
  };
  Ok(Some(idx))
}

fn text_role(field: &str) -> String {
  format!("text:{field}")
}

/// Writes the files of segment `id` (in `dir`) from builders, returning
/// the manifest entry.
struct FileSet<'a> {
  dir: &'a Path,
  id: u64,
  files: BTreeMap<String, String>,
}

impl<'a> FileSet<'a> {
  fn new(dir: &'a Path, id: u64) -> Self {
    Self {
      dir,
      id,
      files: BTreeMap::new(),
    }
  }

  fn path(&mut self, role: String, ext: &str) -> std::path::PathBuf {
    let name = segment_file_name(self.id, ext);
    let p = self.dir.join(&name);
    self.files.insert(role, name);
    p
  }

  fn finish(self, doc_count: u32, vectors: Option<&VecIndex>) -> SegmentEntry {
    SegmentEntry {
      id: self.id,
      doc_count: u64::from(doc_count),
      files: self.files,
      tombstones: None,
      meta: json!({
        "vector_index": vectors.map_or("none", VecIndex::kind_name),
      }),
    }
  }
}

/// Writes a frozen memtable as segment `id`. `g` must be the memtable's
/// (no longer changing) contents. Without `graph` the vectors get a flat
/// index even when the schema asks for HNSW, leaving the graph to a
/// later merge.
pub(crate) fn write_from_memtable(
  dir: &Path,
  id: u64,
  mem: &Memtable,
  g: &MemInner,
  schema: &Schema,
  opts: &Options,
  graph: bool,
) -> Result<SegmentEntry> {
  let n = g.docs;
  let mut fs = FileSet::new(dir, id);
  let vectors = match &mem.vectors {
    Some(a) => {
      let mut v = Vec::with_capacity(n as usize * a.dim());
      for (_, run) in a.runs(n as usize) {
        v.extend_from_slice(run);
      }
      if v.len() != n as usize * a.dim() {
        return Err(Error::invalid("memtable vectors are not all published"));
      }
      build_vectors(schema, opts, &v, graph)?
    }
    None => None,
  };
  if let Some(vi) = &vectors {
    vi.write(&fs.path("vectors".into(), "vec"))?;
  }
  for (i, (field, t)) in schema.text_fields.iter().zip(&g.text).enumerate() {
    let p = fs.path(text_role(field), &format!("t{i}"));
    write_atomic_with(&p, |w| {
      t.write_to(w)?;
      Ok(())
    })?;
  }
  let p = fs.path("docs".into(), "docs");
  write_atomic_with(&p, |w| {
    g.stored.write_to(w)?;
    Ok(())
  })?;
  let p = fs.path("ids".into(), "ids");
  write_atomic_with(&p, |w| {
    g.ids.write_to(w)?;
    Ok(())
  })?;
  if let Some(pi) = &g.pindex {
    let p = fs.path("pindex".into(), "pidx");
    write_atomic_with(&p, |w| {
      pi.write_to(w)?;
      Ok(())
    })?;
  }
  Ok(fs.finish(n, vectors.as_ref()))
}

/// The HNSW graph of `seg` if a merge whose output starts with all of
/// `seg`'s documents, unchanged, can extend it instead of building a new
/// graph: same parameters as the schema asks for now.
fn reusable_graph<'a>(
  seg: &'a Segment,
  schema: &Schema,
) -> Option<&'a HnswIndex> {
  let f = schema.vector.as_ref()?;
  let IndexKind::Hnsw { m, ef_construction } = f.index else {
    return None;
  };
  match &seg.vectors {
    Some(VecIndex::Hnsw(h))
      if h.params().m == m
        && h.params().ef_construction == ef_construction
        && h.params().quantization == f.quantization =>
    {
      Some(h)
    }
    _ => None,
  }
}

/// Merges `inputs` (segments with the deletes to drop) into segment `id`,
/// keeping their order. Returns the entry and, per input, the new id of
/// each old doc ([`DELETED`] if dropped); `None` if nothing survives.
///
/// When the first input has an HNSW graph and no deletes, its documents
/// keep their ids, so the merged graph is that graph with the other
/// inputs' vectors inserted: a merge costs graph work only for the
/// documents that were not in a graph yet (or were in a smaller one),
/// not for all of them. Inserting into an existing graph is how every
/// HNSW graph is built, so the result has the same quality as a fresh
/// build.
pub(crate) fn write_merged(
  dir: &Path,
  id: u64,
  inputs: &[(&Segment, &BitSet)],
  schema: &Schema,
  opts: &Options,
) -> Result<Option<(SegmentEntry, Vec<Vec<DocId>>)>> {
  let mut maps = Vec::with_capacity(inputs.len());
  let mut n: u32 = 0;
  for (seg, tomb) in inputs {
    let mut m = vec![DELETED; seg.len as usize];
    for (d, slot) in m.iter_mut().enumerate() {
      if !tomb.contains(d as DocId) {
        *slot = n;
        n = n
          .checked_add(1)
          .filter(|&x| x < DocId::MAX)
          .ok_or_else(|| Error::invalid("merged segment would be too big"))?;
      }
    }
    maps.push(m);
  }
  if n == 0 {
    return Ok(None);
  }
  let live = |i: usize| {
    maps[i]
      .iter()
      .enumerate()
      .filter(|&(_, &new)| new != DELETED)
      .map(|(d, _)| d as DocId)
  };
  let mut fs = FileSet::new(dir, id);
  let first_intact = !maps[0].contains(&DELETED);
  let reuse = reusable_graph(inputs[0].0, schema)
    .filter(|_| first_intact && n as usize >= opts.hnsw_min_docs);
  let vectors = match &schema.vector {
    Some(f) => {
      let skip = usize::from(reuse.is_some());
      let rows = n as usize - if skip == 1 { maps[0].len() } else { 0 };
      let mut v = Vec::with_capacity(rows * f.dim);
      for (i, (seg, _)) in inputs.iter().enumerate().skip(skip) {
        let vi = seg
          .vectors
          .as_ref()
          .ok_or_else(|| Error::corrupt("segment has no vector index"))?;
        for d in live(i) {
          v.extend_from_slice(vi.vector(d));
        }
      }
      match reuse {
        Some(h) => {
          // A mapped index is copied into memory on the first insert.
          let mut h = h.clone();
          h.add_batch(&v)?;
          Some(VecIndex::Hnsw(h))
        }
        None => build_vectors(schema, opts, &v, true)?,
      }
    }
    None => None,
  };
  if let Some(vi) = &vectors {
    vi.write(&fs.path("vectors".into(), "vec"))?;
  }
  for (f, field) in schema.text_fields.iter().enumerate() {
    let merge_inputs: Vec<MergeInput<'_>> = inputs
      .iter()
      .zip(&maps)
      .map(|((seg, _), m)| {
        let t = &seg.text[f];
        MergeInput {
          index: t,
          doc_map: &m[..t.num_docs()],
        }
      })
      .collect();
    let p = fs.path(text_role(field), &format!("t{f}"));
    write_atomic_with(&p, |w| {
      TextIndex::merge(&merge_inputs, n as usize, w)?;
      Ok(())
    })?;
  }
  let mut records: Vec<&[u8]> = Vec::with_capacity(n as usize);
  let mut ids = IdMap::new();
  for (i, (seg, _)) in inputs.iter().enumerate() {
    for d in live(i) {
      records.push(seg.stored.get(d)?);
      ids.insert(seg.ids.external_id(d)?.into_owned())?;
    }
  }
  let p = fs.path("docs".into(), "docs");
  write_atomic_with(&p, |w| {
    write_payloads(w, records.iter().copied())?;
    Ok(())
  })?;
  let p = fs.path("ids".into(), "ids");
  write_atomic_with(&p, |w| {
    ids.write_to(w)?;
    Ok(())
  })?;
  if !schema.filterable.is_empty() {
    let fschema = schema.filter_schema()?;
    let pin: Vec<(&FrozenIndex, &[DocId])> = inputs
      .iter()
      .zip(&maps)
      .map(|((seg, _), m)| {
        seg
          .pindex
          .as_ref()
          .map(|x| (x, m.as_slice()))
          .ok_or_else(|| Error::corrupt("segment has no payload index"))
      })
      .collect::<Result<_>>()?;
    let p = fs.path("pindex".into(), "pidx");
    write_atomic_with(&p, |w| {
      FrozenIndex::merge(&fschema, &pin, w)?;
      Ok(())
    })?;
  }
  Ok(Some((fs.finish(n, vectors.as_ref()), maps)))
}

impl Segment {
  /// Opens the segment named by `entry` (its files memory-mapped).
  pub(crate) fn open(
    dir: &Path,
    entry: &SegmentEntry,
    schema: &Schema,
  ) -> Result<Segment> {
    let file = |role: &str| -> Result<std::path::PathBuf> {
      entry.files.get(role).map(|n| dir.join(n)).ok_or_else(|| {
        Error::corrupt(format!("segment {} has no {role} file", entry.id))
      })
    };
    let len = u32::try_from(entry.doc_count)
      .ok()
      .filter(|&n| n < DocId::MAX)
      .ok_or_else(|| Error::corrupt("segment doc count out of range"))?;
    let check = |what: &str, got: usize| {
      if got != len as usize {
        return Err(Error::corrupt(format!(
          "segment {}: {what} holds {got} docs, manifest says {len}",
          entry.id
        )));
      }
      Ok(())
    };
    let vectors = match &schema.vector {
      Some(f) => {
        let region = map_file_with(&file("vectors")?, Advice::Random)?;
        let kind = Container::open(region.clone())?.kind();
        let vi = if kind == FLAT_KIND {
          VecIndex::Flat(FlatIndex::open(region)?)
        } else if kind == HNSW_KIND {
          VecIndex::Hnsw(HnswIndex::open(region)?)
        } else {
          return Err(Error::corrupt("unknown vector index kind"));
        };
        let ix = vi.index();
        if ix.dim() != f.dim || ix.metric() != f.metric {
          return Err(Error::corrupt(format!(
            "segment {} vectors do not match the schema",
            entry.id
          )));
        }
        check("vector index", ix.len())?;
        Some(vi)
      }
      None => None,
    };
    let analyzer = Analyzer::new(schema.analyzer.clone())?;
    let mut text = Vec::with_capacity(schema.text_fields.len());
    for field in &schema.text_fields {
      let region = map_file(&file(&text_role(field))?)?;
      let t = TextIndex::open_with_analyzer(region, analyzer.clone())?;
      if t.num_docs() > len as usize {
        return Err(Error::corrupt(format!(
          "segment {} text index is larger than the segment",
          entry.id
        )));
      }
      text.push(t);
    }
    let stored = PayloadStore::open(map_file(&file("docs")?)?)?;
    check("stored records", stored.len())?;
    let ids = FrozenIdMap::open(map_file(&file("ids")?)?)?;
    check("id map", ids.len())?;
    let pindex = if schema.filterable.is_empty() {
      None
    } else {
      let p = FrozenIndex::open(map_file(&file("pindex")?)?)?;
      if *p.schema() != schema.filter_schema()? || p.doc_count() > len as usize
      {
        return Err(Error::corrupt(format!(
          "segment {} payload index does not match the schema",
          entry.id
        )));
      }
      Some(p)
    };
    let mut entry = entry.clone();
    entry.tombstones = None;
    Ok(Segment {
      id: entry.id,
      len,
      vectors,
      text,
      stored,
      ids,
      pindex,
      entry,
    })
  }
}
