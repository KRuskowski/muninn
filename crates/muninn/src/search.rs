//! Query execution: snapshot, per-source search, merge, fusion.
//!
//! 1. **Snapshot.** Read-lock the active memtable, then load the current
//!    version; retry if a freeze replaced the active memtable in between.
//!    Under that lock the memtable holds exactly what the version says
//!    (see `memtable.rs`). Still under it, evaluate the filter on the
//!    memtables, gather BM25 statistics from *every* source and search
//!    the memtables' text with the summed statistics. Then unlock.
//! 2. **Fan out.** Search every segment's text with the same statistics,
//!    and every source's vectors, in parallel on rayon. Each source gets
//!    the filter "user filter AND not deleted AND visible".
//! 3. **Merge.** Order candidates by score, then by write order (source
//!    position in the version, then doc id). Every index breaks ties by
//!    smaller doc id, so per-source top-k lists merge into the exact
//!    global top-k under that total order.
//! 4. **Fuse** (hybrid only) and resolve external ids and payloads.
//!
//! Scores from different sources are comparable: vector scores are exact
//! distances converted by the metric, and text scores use one set of
//! corpus statistics summed over all sources (`CorpusStats::merge`).
//! Like Lucene, those statistics count documents that were deleted but
//! not yet compacted away.

use std::collections::HashMap;
use std::sync::{Arc, RwLockReadGuard};

use muninn_core::{DocFilter, DocId, Error, Hit, Metric, Result, TopK};
use muninn_text::{CorpusStats, TextSearcher};
use muninn_vector::SearchParams;
use rayon::prelude::*;

use crate::arena::VecArena;
use crate::codec;
use crate::collection::Core;
use crate::idfilter::{Selection, SourceFilter, SourceRef};
use crate::memtable::{MemInner, read};
use crate::query::{Fusion, Query};
use crate::segment::Segment;
use crate::types::{Document, Id, SearchHit};
use crate::version::{Source, Version};

/// A candidate: score (higher is better), source position, doc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cand {
  pub score: f32,
  pub view: u32,
  pub doc: DocId,
}

impl Cand {
  fn key(&self) -> (u32, DocId) {
    (self.view, self.doc)
  }
}

/// Best first; ties by write order.
fn sort_cands(v: &mut [Cand]) {
  v.sort_unstable_by(|a, b| {
    b.score
      .total_cmp(&a.score)
      .then(a.view.cmp(&b.view))
      .then(a.doc.cmp(&b.doc))
  });
}

fn to_cands(view: usize, hits: Vec<Hit>) -> impl Iterator<Item = Cand> {
  hits.into_iter().map(move |h| Cand {
    score: h.score,
    view: view as u32,
    doc: h.id,
  })
}

/// Rows scored per `distance_batch` call.
const BLOCK: usize = 256;
/// Rows per parallel task of a memtable scan.
const PAR_ROWS: usize = 4096;
/// Scans touching fewer floats than this stay on one thread.
const PAR_MIN_FLOATS: usize = 1 << 19;

/// Exact top-`k` over rows `0..len` of a memtable's vectors. `query` is
/// already normalised for cosine.
fn scan_memtable(
  arena: &VecArena,
  len: u32,
  metric: Metric,
  query: &[f32],
  k: usize,
  filter: &SourceFilter<'_>,
) -> Vec<Hit> {
  let dim = arena.dim();
  let all = filter.passes_all();
  let mut pieces: Vec<(usize, &[f32])> = Vec::new();
  for (start, floats) in arena.runs(len as usize) {
    let rows = floats.len() / dim;
    let mut r = 0;
    while r < rows {
      let e = (r + PAR_ROWS).min(rows);
      pieces.push((start + r, &floats[r * dim..e * dim]));
      r = e;
    }
  }
  let scan = |(start, floats): &(usize, &[f32])| {
    let mut top = TopK::new(k);
    let mut out = [0f32; BLOCK];
    let rows = floats.len() / dim;
    let mut b = 0;
    while b < rows {
      let e = (b + BLOCK).min(rows);
      let n = e - b;
      muninn_simd::distance_batch(
        metric,
        query,
        &floats[b * dim..e * dim],
        &mut out[..n],
      );
      for (i, &d) in out[..n].iter().enumerate() {
        let id = (start + b + i) as DocId;
        if d < top.threshold() && (all || filter.matches(id)) {
          top.push(id, d);
        }
      }
      b = e;
    }
    top
  };
  let floats = len as usize * dim;
  let top = if floats >= PAR_MIN_FLOATS && pieces.len() > 1 {
    pieces.par_iter().map(scan).reduce(
      || TopK::new(k),
      |mut a, b| {
        a.merge(b);
        a
      },
    )
  } else {
    let mut top = TopK::new(k);
    for p in &pieces {
      top.merge(scan(p));
    }
    top
  };
  top.into_hits(metric)
}

/// Read guards on every memtable of a version (`None` for segments).
type Guards<'a> = Vec<Option<RwLockReadGuard<'a, MemInner>>>;

/// What the locked phase produces; owns everything.
struct Locked {
  /// Filter selections of memtable views (`None` for segments).
  mem_sels: Vec<Option<Selection<'static>>>,
  /// Per text field searched: summed statistics and memtable hits.
  text: Vec<(usize, CorpusStats, Vec<Cand>)>,
}

/// The query's payload and id filters evaluated on one source. Lazy
/// parts may borrow only `lazy_seg`.
fn select<'s>(
  q: &Query,
  src: &SourceRef<'_>,
  lazy_seg: Option<&'s Segment>,
) -> Result<Selection<'s>> {
  let mut parts = Vec::with_capacity(2);
  if let Some(f) = &q.filter {
    parts.push(src.payload(f)?);
  }
  if let Some(f) = &q.id_filter {
    parts.push(f.select(src, lazy_seg)?);
  }
  Ok(Selection::and(parts))
}

/// Validated text fields of a query.
fn text_fields(core: &Core, q: &Query) -> Result<Vec<usize>> {
  if q.text.is_none() {
    return Ok(Vec::new());
  }
  if core.schema.text_fields.is_empty() {
    return Err(Error::invalid("the collection has no text fields"));
  }
  match &q.text_fields {
    None => Ok((0..core.schema.text_fields.len()).collect()),
    Some(names) => names
      .iter()
      .map(|n| {
        core
          .schema
          .text_field(n)
          .ok_or_else(|| Error::invalid(format!("unknown text field {n:?}")))
      })
      .collect(),
  }
}

/// Runs `f` with a snapshot whose memtables are read-locked and agree with
/// it exactly (see the module docs).
fn with_locked<R>(
  core: &Core,
  f: impl FnOnce(&Arc<Version>, &Guards<'_>) -> Result<R>,
) -> Result<(Arc<Version>, R)> {
  loop {
    let v0 = core.snapshot();
    let active = v0.active().clone();
    let g = read(&active.inner);
    let v = core.snapshot();
    if !Arc::ptr_eq(v.active(), &active) {
      continue;
    }
    let n = v.views.len();
    let mut guards: Guards<'_> = Vec::with_capacity(n);
    for (i, w) in v.views.iter().enumerate() {
      guards.push(match w.memtable() {
        Some(m) if i + 1 < n => Some(read(&m.inner)),
        _ => None,
      });
    }
    guards[n - 1] = Some(g);
    let r = f(&v, &guards)?;
    drop(guards);
    return Ok((v, r));
  }
}

pub(crate) fn search(core: &Core, q: &Query) -> Result<Vec<SearchHit>> {
  if let Some(v) = &q.vector {
    let f = core
      .schema
      .vector
      .as_ref()
      .ok_or_else(|| Error::invalid("the collection has no vector field"))?;
    if v.len() != f.dim {
      return Err(Error::DimensionMismatch {
        expected: f.dim,
        got: v.len(),
      });
    }
    codec::check_vector(v, "the query")?;
  }
  let fields = text_fields(core, q)?;
  if let Some(f) = &q.filter {
    core.filter_schema.check(f)?;
  }
  if q.vector.is_none() && q.text.is_none() {
    return Err(Error::invalid("a query needs a vector or text"));
  }
  if q.k == 0 {
    return Ok(Vec::new());
  }
  let side_k = q.side_k();
  let text = q.text.as_deref();

  let (v, locked) = with_locked(core, |v, guards| {
    let mut mem_sels = Vec::with_capacity(v.views.len());
    for (w, g) in v.views.iter().zip(guards) {
      mem_sels.push(match g {
        Some(g) => Some(select(q, &SourceRef::Mem(g, w.len), None)?),
        None => None,
      });
    }
    let mut out = Vec::new();
    if let Some(text) = text {
      for &f in &fields {
        let mut stats = CorpusStats::default();
        for (w, g) in v.views.iter().zip(guards) {
          let s = match (&w.src, g) {
            (_, Some(g)) => g.text[f].stats_for(text)?,
            (Source::Seg(seg), None) => seg.text[f].stats_for(text)?,
            (Source::Mem(_), None) => unreachable!("memtables are locked"),
          };
          stats.merge(&s);
        }
        let mut hits = Vec::new();
        for (i, (w, g)) in v.views.iter().zip(guards).enumerate() {
          let Some(g) = g else { continue };
          let sel = mem_sels[i].as_ref().expect("memtable selection");
          let sf = SourceFilter {
            sel,
            tomb: &w.tomb,
            deleted: w.deleted,
            len: w.len,
          };
          let filt: Option<&dyn DocFilter> =
            if sf.passes_all() { None } else { Some(&sf) };
          let h = g.text[f].search_with_stats(text, side_k, filt, &stats)?;
          hits.extend(to_cands(i, h));
        }
        out.push((f, stats, hits));
      }
    }
    Ok(Locked {
      mem_sels,
      text: out,
    })
  })?;

  // Phase 2, unlocked: segment selections, then the fan-out.
  let Locked {
    mem_sels,
    text: text_parts,
  } = locked;
  let mut sels: Vec<Selection<'_>> = Vec::with_capacity(v.views.len());
  for (w, ms) in v.views.iter().zip(mem_sels) {
    sels.push(match (ms, &w.src) {
      (Some(s), _) => s,
      (None, Source::Seg(seg)) => select(q, &SourceRef::Seg(seg), Some(seg))?,
      (None, Source::Mem(_)) => unreachable!("memtables have selections"),
    });
  }
  let filters: Vec<SourceFilter<'_>> = v
    .views
    .iter()
    .zip(&sels)
    .map(|(w, sel)| SourceFilter {
      sel,
      tomb: &w.tomb,
      deleted: w.deleted,
      len: w.len,
    })
    .collect();

  let vector_part = || -> Result<Vec<Cand>> {
    let Some(query) = &q.vector else {
      return Ok(Vec::new());
    };
    let metric = core.schema.vector.as_ref().expect("checked").metric;
    let mut prepared = query.clone();
    if metric.normalizes() {
      muninn_simd::normalize(&mut prepared);
    }
    let params = SearchParams::new(side_k)
      .with_ef(q.ef)
      .with_rerank(q.rerank);
    // Empty sources (typically the active memtable right after a flush)
    // cost nothing; skipping them often leaves a single source, searched
    // without a parallel fan-out.
    let nonempty: Vec<usize> =
      (0..v.views.len()).filter(|&i| v.views[i].len > 0).collect();
    let per: Vec<Result<Vec<Cand>>> = fan_out(&nonempty, |i| {
      let w = &v.views[i];
      let sf = &filters[i];
      let hits = match &w.src {
        Source::Mem(m) => match &m.vectors {
          Some(a) => scan_memtable(a, w.len, metric, &prepared, side_k, sf),
          None => Vec::new(),
        },
        Source::Seg(seg) => {
          let vi = seg
            .vectors
            .as_ref()
            .ok_or_else(|| Error::corrupt("segment without a vector index"))?;
          // A selective filter is answered by the index's exact scan
          // over the filter's id set (`SourceFilter::candidates`).
          let filt: Option<&dyn DocFilter> =
            if sf.passes_all() { None } else { Some(sf) };
          vi.index().search(query, &params, filt)?
        }
      };
      Ok(to_cands(i, hits).collect())
    });
    let mut all = Vec::new();
    for r in per {
      all.extend(r?);
    }
    sort_cands(&mut all);
    all.truncate(side_k);
    Ok(all)
  };

  let text_part = || -> Result<Vec<Cand>> {
    let Some(text) = text else {
      return Ok(Vec::new());
    };
    let mut best: HashMap<(u32, DocId), f32> = HashMap::new();
    for (f, stats, mem_hits) in &text_parts {
      let segs: Vec<usize> = (0..v.nseg).collect();
      let per: Vec<Result<Vec<Cand>>> = fan_out(&segs, |i| {
        let seg = v.views[i].segment().expect("segment");
        let sf = &filters[i];
        let filt: Option<&dyn DocFilter> =
          if sf.passes_all() { None } else { Some(sf) };
        let h = seg.text[*f].search_with_stats(text, side_k, filt, stats)?;
        Ok(to_cands(i, h).collect())
      });
      let mut all: Vec<Cand> = mem_hits.clone();
      for r in per {
        all.extend(r?);
      }
      sort_cands(&mut all);
      all.truncate(side_k);
      if text_parts.len() == 1 {
        return Ok(all);
      }
      // Several fields: a document scores its best field.
      for c in all {
        let e = best.entry(c.key()).or_insert(c.score);
        if c.score > *e {
          *e = c.score;
        }
      }
    }
    let mut all: Vec<Cand> = best
      .into_iter()
      .map(|((view, doc), score)| Cand { score, view, doc })
      .collect();
    sort_cands(&mut all);
    all.truncate(side_k);
    Ok(all)
  };

  let (vc, tc) = match (q.vector.is_some(), text.is_some()) {
    (true, true) => {
      let (vr, tr) = rayon::join(vector_part, text_part);
      (vr?, tr?)
    }
    (true, false) => (vector_part()?, Vec::new()),
    _ => (Vec::new(), text_part()?),
  };
  let fused = fuse(q, &vc, &tc);
  resolve(&v, &fused, q.with_payload)
}

/// Maps `f` over `items`, in parallel on rayon when there are several:
/// a one-item fan-out would only add a task hand-off to the query.
fn fan_out<T: Send>(
  items: &[usize],
  f: impl Fn(usize) -> T + Sync + Send,
) -> Vec<T> {
  if items.len() <= 1 {
    items.iter().map(|&i| f(i)).collect()
  } else {
    items.par_iter().map(|&i| f(i)).collect()
  }
}

/// A fused result before id resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fused {
  pub cand: Cand,
  pub vector: Option<f32>,
  pub text: Option<f32>,
}

/// A candidate's 1-based rank and score in one list, if present.
type Rank = Option<(usize, f32)>;

/// Combines the vector and text candidate lists (each best first) into
/// the final top `q.k`.
pub(crate) fn fuse(q: &Query, vc: &[Cand], tc: &[Cand]) -> Vec<Fused> {
  let k = q.k;
  match (q.vector.is_some(), q.text.is_some()) {
    (true, false) | (false, true) => {
      let src = if q.vector.is_some() { vc } else { tc };
      let vec_side = q.vector.is_some();
      return src
        .iter()
        .take(k)
        .map(|c| Fused {
          cand: *c,
          vector: vec_side.then_some(c.score),
          text: (!vec_side).then_some(c.score),
        })
        .collect();
    }
    _ => {}
  }
  let mut by: HashMap<(u32, DocId), (Rank, Rank)> = HashMap::new();
  for (r, c) in vc.iter().enumerate() {
    by.entry(c.key()).or_default().0 = Some((r + 1, c.score));
  }
  for (r, c) in tc.iter().enumerate() {
    by.entry(c.key()).or_default().1 = Some((r + 1, c.score));
  }
  let norm = |list: &[Cand]| {
    let max = list
      .iter()
      .map(|c| c.score)
      .fold(f32::NEG_INFINITY, f32::max);
    let min = list.iter().map(|c| c.score).fold(f32::INFINITY, f32::min);
    move |s: f32| {
      if max > min {
        (s - min) / (max - min)
      } else {
        1.0
      }
    }
  };
  let (nv, nt) = (norm(vc), norm(tc));
  let mut out: Vec<Fused> = by
    .into_iter()
    .map(|((view, doc), (v, t))| {
      let score = match q.fusion {
        Fusion::Rrf { k } => {
          let rrf =
            |x: Rank| x.map_or(0.0, |(r, _)| 1.0 / (k as f64 + r as f64));
          (rrf(v) + rrf(t)) as f32
        }
        Fusion::Weighted { vector, text } => {
          vector * v.map_or(0.0, |(_, s)| nv(s))
            + text * t.map_or(0.0, |(_, s)| nt(s))
        }
      };
      Fused {
        cand: Cand { score, view, doc },
        vector: v.map(|x| x.1),
        text: t.map(|x| x.1),
      }
    })
    .collect();
  out.sort_unstable_by(|a, b| {
    b.cand
      .score
      .total_cmp(&a.cand.score)
      .then(a.cand.view.cmp(&b.cand.view))
      .then(a.cand.doc.cmp(&b.cand.doc))
  });
  out.truncate(k);
  out
}

/// Maps candidates to external ids (and payloads).
fn resolve(
  v: &Version,
  fused: &[Fused],
  with_payload: bool,
) -> Result<Vec<SearchHit>> {
  let mut guards: HashMap<u32, RwLockReadGuard<'_, MemInner>> = HashMap::new();
  let mut out = Vec::with_capacity(fused.len());
  for f in fused {
    let w = &v.views[f.cand.view as usize];
    let d = f.cand.doc;
    let (id, rec): (Id, Option<&[u8]>) = match &w.src {
      Source::Seg(seg) => (
        seg.ids.external_id(d)?.into_owned(),
        if with_payload {
          Some(seg.stored.get(d)?)
        } else {
          None
        },
      ),
      Source::Mem(m) => {
        let g = guards.entry(f.cand.view).or_insert_with(|| read(&m.inner));
        let id = g
          .ids
          .external_id(d)
          .ok_or_else(|| Error::corrupt("memtable doc without id"))?
          .into_owned();
        let payload = if with_payload {
          let rec = g
            .stored
            .get(d)
            .ok_or_else(|| Error::corrupt("memtable doc without record"))?;
          Some(codec::parse_payload(codec::decode_stored(rec)?.payload)?)
        } else {
          None
        };
        out.push(SearchHit {
          id,
          score: f.cand.score,
          vector_score: f.vector,
          text_score: f.text,
          payload,
        });
        continue;
      }
    };
    let payload = match rec {
      Some(r) => Some(codec::parse_payload(codec::decode_stored(r)?.payload)?),
      None => None,
    };
    out.push(SearchHit {
      id,
      score: f.cand.score,
      vector_score: f.vector,
      text_score: f.text,
      payload,
    });
  }
  Ok(out)
}

/// Builds a document from its stored record and vector.
fn build_doc(
  core: &Core,
  id: &Id,
  vector: Option<&[f32]>,
  rec: &[u8],
) -> Result<Document> {
  let st = codec::decode_stored(rec)?;
  let mut doc = Document::new(id.clone());
  doc.vector = vector.map(<[f32]>::to_vec);
  for (f, t) in st.text {
    let name =
      core.schema.text_fields.get(f).ok_or_else(|| {
        Error::corrupt("stored record names an unknown field")
      })?;
    doc.text.insert(name.clone(), t.to_string());
  }
  doc.payload = codec::parse_payload(st.payload)?;
  Ok(doc)
}

pub(crate) fn get(core: &Core, id: &Id) -> Result<Option<Document>> {
  let (_, doc) = with_locked(core, |v, guards| {
    for (i, w) in v.views.iter().enumerate().rev() {
      match (&w.src, &guards[i]) {
        (Source::Mem(m), Some(g)) => {
          let Some(d) = g.ids.get(id.as_borrowed()) else {
            continue;
          };
          if d >= w.len || w.is_deleted(d) {
            continue;
          }
          let vec = m.vectors.as_ref().map(|a| a.row(d as usize));
          let rec = g
            .stored
            .get(d)
            .ok_or_else(|| Error::corrupt("memtable doc without record"))?;
          return Ok(Some(build_doc(core, id, vec, rec)?));
        }
        (Source::Seg(seg), _) => {
          let Some(d) = seg.ids.get(id.as_borrowed()) else {
            continue;
          };
          if w.is_deleted(d) {
            continue;
          }
          let vec = seg.vectors.as_ref().map(|x| x.vector(d));
          return Ok(Some(build_doc(core, id, vec, seg.stored.get(d)?)?));
        }
        (Source::Mem(_), None) => unreachable!("memtables are locked"),
      }
    }
    Ok(None)
  })?;
  Ok(doc)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn c(score: f32, view: u32, doc: DocId) -> Cand {
    Cand { score, view, doc }
  }

  fn ids(f: &[Fused]) -> Vec<(u32, DocId)> {
    f.iter().map(|x| x.cand.key()).collect()
  }

  #[test]
  fn rrf_sums_reciprocal_ranks() {
    let q = Query::hybrid(vec![0.0], "x").k(10);
    // a: vector rank 1, text rank 3; b: vector rank 2 only; c: text 1,2.
    let vc = [c(0.9, 0, 1), c(0.8, 0, 2)];
    let tc = [c(5.0, 0, 3), c(4.0, 1, 0), c(3.0, 0, 1)];
    let f = fuse(&q, &vc, &tc);
    let s = |key: (u32, DocId)| f.iter().find(|x| x.cand.key() == key).unwrap();
    let a = s((0, 1));
    assert!((a.cand.score - (1.0 / 61.0 + 1.0 / 63.0) as f32).abs() < 1e-7);
    assert_eq!(a.vector, Some(0.9));
    assert_eq!(a.text, Some(3.0));
    assert_eq!(s((0, 2)).text, None);
    assert_eq!(ids(&f)[0], (0, 1));
    // b (rank 2 vector) and (0,3) (rank 1 text) do not tie: 1/62 < 1/61.
    assert_eq!(ids(&f)[1], (0, 3));
    assert_eq!(f.len(), 4);
  }

  #[test]
  fn rrf_ties_break_by_write_order() {
    let q = Query::hybrid(vec![0.0], "x").k(4);
    // Same ranks on opposite sides: equal fused scores.
    let vc = [c(1.0, 2, 5), c(0.5, 0, 9)];
    let tc = [c(9.0, 1, 7), c(8.0, 0, 3)];
    let f = fuse(&q, &vc, &tc);
    assert_eq!(ids(&f), vec![(1, 7), (2, 5), (0, 3), (0, 9)]);
  }

  #[test]
  fn weighted_normalises_each_side() {
    let q = Query::hybrid(vec![0.0], "x").k(3).fusion(Fusion::Weighted {
      vector: 0.25,
      text: 0.75,
    });
    let vc = [c(10.0, 0, 1), c(0.0, 0, 2)];
    let tc = [c(2.0, 0, 2), c(1.0, 0, 3)];
    let f = fuse(&q, &vc, &tc);
    let s = |d: DocId| f.iter().find(|x| x.cand.doc == d).unwrap().cand.score;
    assert_eq!(s(1), 0.25); // vector max, absent from text
    assert_eq!(s(2), 0.75); // vector min, text max
    assert_eq!(s(3), 0.0); // text min only
    assert_eq!(ids(&f), vec![(0, 2), (0, 1), (0, 3)]);
    // A single candidate normalises to 1.
    let f = fuse(&q, &[c(3.0, 0, 1)], &[]);
    assert_eq!(f[0].cand.score, 0.25);
  }

  #[test]
  fn single_mode_passes_through_and_truncates() {
    let q = Query::vector(vec![0.0]).k(1);
    let f = fuse(&q, &[c(2.0, 0, 1), c(1.0, 0, 2)], &[]);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].vector, Some(2.0));
    assert_eq!(f[0].text, None);
    let q = Query::text("x").k(5);
    let f = fuse(&q, &[], &[c(2.0, 0, 1)]);
    assert_eq!(f[0].text, Some(2.0));
  }

  #[test]
  fn candidates_sort_by_score_then_write_order() {
    let mut v = vec![c(1.0, 1, 0), c(2.0, 3, 3), c(1.0, 0, 9), c(1.0, 0, 2)];
    sort_cands(&mut v);
    assert_eq!(
      v.iter().map(Cand::key).collect::<Vec<_>>(),
      vec![(3, 3), (0, 2), (0, 9), (1, 0)]
    );
  }
}
