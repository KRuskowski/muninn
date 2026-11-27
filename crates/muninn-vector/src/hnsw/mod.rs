//! Hierarchical navigable small world graphs (Malkov & Yashunin, 2016).

mod build;
mod graph;
mod search;

use std::io::Write;
use std::ops::Range;
use std::sync::Mutex;

use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{DocFilter, DocId, Error, Hit, Metric, Region, Result, TopK};
use rayon::prelude::*;

use self::build::{Builder, MAX_LEVEL, level_for};
use self::graph::{BuildView, Graph, GraphView, NONE, as_atomic};
use self::search::{Abort, Beam, beam, greedy, prefer_exact, with_scratch};
use crate::dist::{Dist, FullDist};
use crate::flat::{
  FlatIndex, check_capacity, check_dim, check_rows, prepare_query,
};
use crate::quant::{Quantization, Quantizer, with_code_dist};
use crate::scan::{exact_search, rerank};
use crate::slab::Slab;
use crate::{SearchParams, VectorIndex, meta};

/// Container kind of an [`HnswIndex`] file.
pub const HNSW_KIND: [u8; 8] = tag("hnsw");
/// Current [`HnswIndex`] file format version.
pub const HNSW_VERSION: u32 = 1;

const T_META: [u8; 8] = tag("meta");
const T_VECTORS: [u8; 8] = tag("vectors");
const T_LEVELS: [u8; 8] = tag("levels");
const T_L0: [u8; 8] = tag("l0");
const T_UPOFF: [u8; 8] = tag("upoff");
const T_UPPER: [u8; 8] = tag("upper");
const META_FIELDS: usize = 12;

/// Nodes a filtered search scores before judging the filter's
/// selectivity from what it saw.
const ABORT_MIN_VISITS: usize = 512;

/// How filtered searches are executed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FilterStrategy {
  /// Choose per query from a cost model: an exact scan over the passing
  /// documents when the filter is selective enough that the graph walk
  /// would cost more (judged from `DocFilter::selectivity_hint`, or from
  /// the pass rate seen during traversal), graph search otherwise.
  #[default]
  Auto,
  /// Always traverse the graph (still falls back to an exact scan if it
  /// finds fewer than `k` passing documents). With [`Metric::Dot`] some
  /// low-norm vectors may have no incoming edges, so under very selective
  /// filters a forced graph walk can miss passing documents.
  Graph,
  /// Always scan the passing documents exactly.
  Exact,
}

impl FilterStrategy {
  fn to_u8(self) -> u8 {
    match self {
      Self::Auto => 0,
      Self::Graph => 1,
      Self::Exact => 2,
    }
  }

  fn from_u8(v: u64) -> Result<Self> {
    match v {
      0 => Ok(Self::Auto),
      1 => Ok(Self::Graph),
      2 => Ok(Self::Exact),
      _ => Err(Error::corrupt(format!("unknown filter strategy {v}"))),
    }
  }
}

/// Largest supported `M` (level 0 then holds up to 512 neighbours).
pub const MAX_M: usize = 256;

/// Batches smaller than this are inserted on the calling thread.
const PAR_MIN_BATCH: usize = 64;

/// While the graph is smaller than this, a batch is inserted serially
/// first, so parallel inserts never race over a near-empty graph.
const SERIAL_WARMUP: usize = 256;

/// Construction and search settings for an [`HnswIndex`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HnswParams {
  /// Neighbours per node on upper levels; level 0 keeps `2 * m`.
  pub m: usize,
  /// Candidate-list size while inserting. Larger builds slower and
  /// produces a better graph.
  pub ef_construction: usize,
  /// Default candidate-list size for search (`SearchParams::ef == 0`).
  /// Search always uses at least `k`.
  pub ef_search: usize,
  /// Compressed codes used for graph traversal, re-ranked with the full
  /// vectors. The graph itself is always built from full vectors.
  pub quantization: Quantization,
  /// Seed for level assignment.
  pub seed: u64,
  /// How filtered searches choose between the graph and an exact scan.
  pub filter: FilterStrategy,
}

impl Default for HnswParams {
  fn default() -> Self {
    Self {
      m: 16,
      ef_construction: 200,
      ef_search: 64,
      quantization: Quantization::None,
      seed: 0x6d75_6e69_6e6e,
      filter: FilterStrategy::Auto,
    }
  }
}

impl HnswParams {
  /// Sets `m`.
  pub fn with_m(mut self, m: usize) -> Self {
    self.m = m;
    self
  }

  /// Sets `ef_construction`.
  pub fn with_ef_construction(mut self, ef: usize) -> Self {
    self.ef_construction = ef;
    self
  }

  /// Sets the default search `ef`.
  pub fn with_ef_search(mut self, ef: usize) -> Self {
    self.ef_search = ef;
    self
  }

  /// Sets the traversal quantisation.
  pub fn with_quantization(mut self, q: Quantization) -> Self {
    self.quantization = q;
    self
  }

  /// Sets the level-assignment seed.
  pub fn with_seed(mut self, seed: u64) -> Self {
    self.seed = seed;
    self
  }

  /// Sets the filtered-search strategy.
  pub fn with_filter_strategy(mut self, f: FilterStrategy) -> Self {
    self.filter = f;
    self
  }

  fn validate(&self) -> Result<()> {
    if !(2..=MAX_M).contains(&self.m) {
      return Err(Error::invalid(format!(
        "HNSW m must be in 2..={MAX_M}, got {}",
        self.m
      )));
    }
    if self.ef_construction == 0 {
      return Err(Error::invalid("HNSW ef_construction must be positive"));
    }
    Ok(())
  }
}

/// Approximate nearest-neighbour index: an HNSW graph over full-precision
/// vectors, with optional SQ8 or binary codes for traversal.
///
/// - **Build**: [`HnswIndex::add_batch`] inserts in parallel on rayon's
///   pool with per-node striped locks; [`HnswIndex::insert`] adds one
///   vector. Both work on an index reopened from a file.
/// - **Search**: greedy descent through the upper levels, then a beam
///   search of width `ef` at level 0. Filters are applied during the
///   beam (filtered-out nodes are traversed, never returned). When the
///   filter is very selective, or the graph cannot produce `k` passing
///   results, the passing documents are scanned exactly instead, so a
///   search returns `min(k, passing)` hits.
/// - **Files**: [`HnswIndex::write_to`] writes one container in one pass;
///   [`HnswIndex::open`] reads vectors, graph and codes in place.
///
/// Building with one thread is deterministic. Parallel builds produce
/// graphs of the same quality that differ in detail between runs;
/// searching a given graph is always deterministic.
#[derive(Clone, Debug)]
pub struct HnswIndex {
  dim: usize,
  metric: Metric,
  params: HnswParams,
  vectors: Slab<f32>,
  graph: Graph,
  entry: DocId,
  max_level: u8,
  quant: Option<Quantizer>,
  trained_on: usize,
}

/// Index structure sizes, for capacity planning and reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HnswStats {
  /// Number of vectors.
  pub len: usize,
  /// Highest level in the graph.
  pub max_level: usize,
  /// Bytes of full-precision vectors.
  pub vector_bytes: usize,
  /// Bytes of graph structure (all levels).
  pub graph_bytes: usize,
  /// Bytes of quantised codes and their parameters.
  pub code_bytes: usize,
  /// Average level-0 out-degree.
  pub avg_degree: usize,
}

impl HnswIndex {
  /// An empty index.
  pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Result<Self> {
    params.validate()?;
    if dim == 0 {
      return Err(Error::invalid("HNSW dimension must be positive"));
    }
    Ok(Self {
      dim,
      metric,
      params,
      vectors: Slab::default(),
      graph: Graph::default(),
      entry: NONE,
      max_level: 0,
      quant: None,
      trained_on: 0,
    })
  }

  /// Builds an index over row-major `vectors` (`dim` floats per row),
  /// in parallel.
  pub fn build(
    dim: usize,
    metric: Metric,
    params: HnswParams,
    vectors: &[f32],
  ) -> Result<Self> {
    let mut idx = Self::new(dim, metric, params)?;
    idx.add_batch(vectors)?;
    Ok(idx)
  }

  /// Builds an index over the vectors of a [`FlatIndex`] (for example a
  /// frozen memtable), keeping its ids.
  pub fn from_flat(flat: &FlatIndex, params: HnswParams) -> Result<Self> {
    Self::build(flat.dim(), flat.metric(), params, flat.vectors())
  }

  /// The settings this index was built with.
  pub fn params(&self) -> &HnswParams {
    &self.params
  }

  /// Switches graph traversal to codes of kind `q`, trained on the
  /// current vectors (or back to full precision with
  /// [`Quantization::None`]). The graph is unchanged: it is always built
  /// from full-precision vectors.
  pub fn set_quantization(&mut self, q: Quantization) {
    self.params.quantization = q;
    self.quant =
      Quantizer::train(q, self.dim, self.metric, self.vectors.as_slice());
    self.trained_on = self.len();
  }

  /// Sets the default search `ef` (used when `SearchParams::ef` is 0).
  pub fn set_ef_search(&mut self, ef: usize) {
    self.params.ef_search = ef;
  }

  /// Sets the filtered-search strategy.
  pub fn set_filter_strategy(&mut self, f: FilterStrategy) {
    self.params.filter = f;
  }

  /// The stored (possibly normalised) vector for `id`. Panics if `id` is
  /// out of range.
  pub fn vector(&self, id: DocId) -> &[f32] {
    let s = id as usize * self.dim;
    &self.vectors.as_slice()[s..s + self.dim]
  }

  /// Whether vectors, the level-0 graph and any codes are read in place
  /// from a mapped region (true after [`HnswIndex::open`], until a mutation).
  pub fn is_mapped(&self) -> bool {
    self.vectors.is_mapped()
      && self.graph.l0.is_mapped()
      && self.quant.as_ref().is_none_or(|q| q.is_mapped())
  }

  /// Structure sizes.
  pub fn stats(&self) -> HnswStats {
    let len = self.len();
    let m0 = 2 * self.params.m;
    let edges: usize = self
      .graph
      .l0
      .as_slice()
      .chunks_exact(1 + m0)
      .map(|r| (r[0] as usize).min(m0))
      .sum();
    HnswStats {
      len,
      max_level: self.max_level as usize,
      vector_bytes: self.vectors.bytes(),
      graph_bytes: self.graph.bytes(),
      code_bytes: self.quant.as_ref().map_or(0, |q| q.bytes()),
      avg_degree: edges.checked_div(len).unwrap_or(0),
    }
  }

  /// Adds one vector and returns its id.
  pub fn insert(&mut self, vector: &[f32]) -> Result<DocId> {
    check_dim(self.dim, vector)?;
    Ok(self.add_batch(vector)?.start)
  }

  /// Adds row-major `vectors` (`dim` floats per row) and returns their
  /// ids. Large batches are linked into the graph in parallel on the
  /// current rayon pool. A mapped index is copied into memory first.
  pub fn add_batch(&mut self, vectors: &[f32]) -> Result<Range<DocId>> {
    let rows = check_rows(self.dim, vectors)?;
    let first = self.len();
    check_capacity(first, rows)?;
    if rows == 0 {
      return Ok(first as DocId..first as DocId);
    }
    let (m, m0, dim) = (self.params.m, 2 * self.params.m, self.dim);
    let end = first + rows;

    let data = self.vectors.to_mut();
    data.extend_from_slice(vectors);
    if self.metric.normalizes() {
      for row in data[first * dim..].chunks_exact_mut(dim) {
        muninn_simd::normalize(row);
      }
    }
    match &mut self.quant {
      None => {
        self.quant =
          Quantizer::train(self.params.quantization, dim, self.metric, data);
        self.trained_on = end;
      }
      Some(q) => {
        q.append(&data[first * dim..]);
        self.trained_on =
          q.maybe_retrain(self.metric, dim, data, self.trained_on);
      }
    }
    debug_assert!(self.quant.as_ref().is_none_or(|q| q.len() == end));

    let levels = self.graph.levels.to_mut();
    let upoff = self.graph.upoff.to_mut();
    let upper = self.graph.upper.to_mut();
    for id in first..end {
      let l = level_for(self.params.seed, id as DocId, m);
      levels.push(l);
      if l == 0 {
        upoff.push(NONE);
      } else {
        let off = upper.len();
        let grow = l as usize * (1 + m);
        if off + grow >= NONE as usize {
          return Err(Error::invalid("HNSW upper levels exceed u32 range"));
        }
        upoff.push(off as u32);
        upper.resize(off + grow, 0);
      }
    }
    let l0 = self.graph.l0.to_mut();
    l0.resize(end * (1 + m0), 0);

    let mut start = first;
    if self.entry == NONE {
      self.entry = first as DocId;
      self.max_level = levels[first];
      start += 1;
    }
    let stripes = if rows < PAR_MIN_BATCH {
      1
    } else {
      end.next_power_of_two().min(1 << 16)
    };
    let locks: Vec<Mutex<()>> = (0..stripes).map(|_| Mutex::new(())).collect();
    let entry = Mutex::new((self.entry, self.max_level));
    let b = Builder {
      view: BuildView {
        m,
        m0,
        levels,
        upoff,
        l0: as_atomic(l0),
        upper: as_atomic(upper),
        locks: &locks,
      },
      data,
      dim,
      metric: self.metric,
      ef_construction: self.params.ef_construction,
      entry: &entry,
    };
    let warm = if first < SERIAL_WARMUP {
      end.min(SERIAL_WARMUP)
    } else {
      start
    };
    for id in start..warm.max(start) {
      b.insert(id as DocId);
    }
    let start = warm.max(start);
    if rows < PAR_MIN_BATCH {
      for id in start..end {
        b.insert(id as DocId);
      }
    } else {
      (start..end)
        .into_par_iter()
        .for_each(|id| b.insert(id as DocId));
    }
    (self.entry, self.max_level) =
      entry.into_inner().unwrap_or_else(|e| e.into_inner());
    Ok(first as DocId..end as DocId)
  }

  /// Writes the index as one container file in a single streaming pass.
  pub fn write_to<W: Write>(&self, out: W) -> Result<W> {
    let mut w = ContainerWriter::new(out, HNSW_KIND, HNSW_VERSION)?;
    let p = &self.params;
    let m: [u64; META_FIELDS] = [
      self.dim as u64,
      self.metric.to_u8() as u64,
      self.len() as u64,
      p.m as u64,
      p.ef_construction as u64,
      p.ef_search as u64,
      p.seed,
      p.quantization.to_u8() as u64,
      p.filter.to_u8() as u64,
      self.entry as u64,
      self.max_level as u64,
      self.trained_on as u64,
    ];
    w.pod_section(T_META, &m)?;
    w.pod_section(T_VECTORS, self.vectors.as_slice())?;
    w.pod_section(T_LEVELS, self.graph.levels.as_slice())?;
    w.pod_section(T_L0, self.graph.l0.as_slice())?;
    w.pod_section(T_UPOFF, self.graph.upoff.as_slice())?;
    w.pod_section(T_UPPER, self.graph.upper.as_slice())?;
    if let Some(q) = &self.quant {
      q.write_sections(&mut w)?;
    }
    w.finish()
  }

  /// Opens a file written by [`HnswIndex::write_to`]. Vectors, graph and
  /// codes are read in place from `region`: opening costs a few header
  /// reads, whatever the index size.
  pub fn open(region: Region) -> Result<Self> {
    let c = Container::open_kind(region, HNSW_KIND, HNSW_VERSION)?;
    let f = meta::read(c.section(T_META)?, META_FIELDS)?;
    let dim = meta::usize_field(f[0], "dimension")?;
    let metric = meta::metric(f[1])?;
    let n = meta::usize_field(f[2], "vector count")?;
    let params = HnswParams {
      m: meta::usize_field(f[3], "m")?,
      ef_construction: meta::usize_field(f[4], "ef_construction")?,
      ef_search: meta::usize_field(f[5], "ef_search")?,
      seed: f[6],
      quantization: meta::quantization(f[7])?,
      filter: FilterStrategy::from_u8(f[8])?,
    };
    params
      .validate()
      .map_err(|e| Error::corrupt(e.to_string()))?;
    if dim == 0 {
      return Err(Error::corrupt("HNSW dimension is zero"));
    }
    let entry = f[9];
    let max_level = f[10];
    let m0 = 2 * params.m;
    let vectors = Slab::from_region(c.section(T_VECTORS)?)?;
    let graph = Graph {
      levels: Slab::from_region(c.section(T_LEVELS)?)?,
      l0: Slab::from_region(c.section(T_L0)?)?,
      upoff: Slab::from_region(c.section(T_UPOFF)?)?,
      upper: Slab::from_region(c.section(T_UPPER)?)?,
    };
    let sizes_ok = Some(vectors.len()) == n.checked_mul(dim)
      && graph.levels.len() == n
      && graph.upoff.len() == n
      && Some(graph.l0.len()) == n.checked_mul(1 + m0);
    if !sizes_ok {
      return Err(Error::corrupt("HNSW sections have inconsistent sizes"));
    }
    let entry_ok = if n == 0 {
      entry == NONE as u64
    } else {
      entry < n as u64
        && max_level <= MAX_LEVEL as u64
        && graph.levels.as_slice()[entry as usize] as u64 == max_level
    };
    if !entry_ok {
      return Err(Error::corrupt("HNSW entry point is invalid"));
    }
    let quant = Quantizer::open(params.quantization, &c, dim, metric, n)?;
    Ok(Self {
      dim,
      metric,
      params,
      vectors,
      graph,
      entry: entry as DocId,
      max_level: max_level as u8,
      quant,
      trained_on: meta::usize_field(f[11], "training size")?,
    })
  }

  fn view(&self) -> GraphView<'_> {
    self.graph.view(self.params.m)
  }

  /// Exact scan over the documents passing `filter`.
  fn exact(&self, q: &[f32], k: usize, filter: Option<&dyn DocFilter>) -> TopK {
    exact_search(self.metric, self.dim, self.vectors.as_slice(), q, k, filter)
  }

  /// Descends the upper levels and runs the level-0 beam with `dist`.
  fn graph_search<D: Dist>(
    &self,
    dist: &D,
    ef: usize,
    filter: Option<&dyn DocFilter>,
  ) -> Beam {
    let g = self.view();
    let auto = self.params.filter == FilterStrategy::Auto;
    let abort = filter.filter(|_| auto).map(|_| Abort {
      n: self.len(),
      ef,
      min_visits: ABORT_MIN_VISITS,
    });
    with_scratch(|s| {
      let mut cur = (self.entry, dist.dist(self.entry));
      for lc in (1..=self.max_level as usize).rev() {
        cur = greedy(&g, dist, cur, lc, &mut s.buf);
      }
      beam(&g, dist, &[cur], ef, 0, filter, abort, s)
    })
  }
}

impl VectorIndex for HnswIndex {
  fn dim(&self) -> usize {
    self.dim
  }

  fn metric(&self) -> Metric {
    self.metric
  }

  fn len(&self) -> usize {
    self.graph.levels.len()
  }

  fn search(
    &self,
    query: &[f32],
    params: &SearchParams,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>> {
    check_dim(self.dim, query)?;
    let k = params.k;
    if k == 0 || self.is_empty() {
      return Ok(Vec::new());
    }
    let q = prepare_query(self.metric, query);
    let ef = match params.ef {
      0 => self.params.ef_search,
      ef => ef,
    }
    .max(k);
    if let Some(f) = filter {
      let exact = match self.params.filter {
        FilterStrategy::Exact => true,
        FilterStrategy::Graph => false,
        FilterStrategy::Auto => f
          .selectivity_hint()
          .is_some_and(|s| prefer_exact(self.len(), s, ef)),
      };
      if exact {
        return Ok(self.exact(&q, k, filter).into_hits(self.metric));
      }
    }
    let data = self.vectors.as_slice();
    let top = match &self.quant {
      None => {
        let dist = FullDist {
          metric: self.metric,
          dim: self.dim,
          query: &q,
          data,
        };
        let b = self.graph_search(&dist, ef, filter);
        if b.aborted || b.top.len() < k {
          self.exact(&q, k, filter)
        } else {
          let mut top = TopK::new(k);
          for (id, d) in b.top.iter() {
            top.push(id, d);
          }
          top
        }
      }
      Some(qz) => {
        let r = match params.rerank {
          0 => qz.kind().default_rerank(k),
          r => r,
        }
        .max(k);
        let qc = qz.prepare(self.metric, &q);
        let b = with_code_dist!(qz.dist(&qc), |d| {
          self.graph_search(&d, ef.max(r), filter)
        });
        if b.aborted || b.top.len() < k {
          self.exact(&q, k, filter)
        } else {
          let cands = b.top.into_sorted_vec();
          let ids = cands.iter().take(r).map(|&(id, _)| id);
          rerank(self.metric, self.dim, data, &q, ids, k)
        }
      }
    };
    Ok(top.into_hits(self.metric))
  }
}
