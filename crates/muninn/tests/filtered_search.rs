//! Filtered vector search over HNSW and flat segments plus a memtable,
//! with deletes and replacements, against brute force: every metric and
//! quantisation, payload and id filters of every shape (sets, NOT, AND
//! and OR with predicates, combinations). Each hit must pass the filter,
//! be live, carry its current version's score, and the count must be
//! `min(k, matching)`. Sizes are kept small for debug CI.

mod common;

use std::collections::{BTreeMap, HashSet};

use common::{Rng, TestDir};
use muninn::{
  Collection, Database, Document, Durability, FieldSpec, Filter, Id, IdFilter,
  Metric, Options, Quantization, Query, Schema, VectorField,
};
use serde_json::json;

const DIM: usize = 16;

struct Live {
  docs: BTreeMap<u64, (Vec<f32>, i64)>,
}

fn dist(metric: Metric, q: &[f32], v: &[f32]) -> f32 {
  match metric {
    Metric::L2 => q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum(),
    Metric::Dot => -q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>(),
    Metric::Cosine => {
      let n = |x: &[f32]| x.iter().map(|a| a * a).sum::<f32>().sqrt();
      let d: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
      1.0 - d / (n(q) * n(v))
    }
  }
}

type Pred = Box<dyn Fn(u64, i64) -> bool>;

/// (name, id filter, payload filter, reference predicate)
fn filters(
  rng: &mut Rng,
  live: &Live,
) -> Vec<(String, Option<IdFilter>, Option<Filter>, Pred)> {
  let ids: Vec<u64> = live.docs.keys().copied().collect();
  let subset = |s: usize, rng: &mut Rng| -> HashSet<u64> {
    (0..s)
      .map(|_| ids[rng.below(ids.len() as u64) as usize])
      .collect()
  };
  let mut out: Vec<(String, Option<IdFilter>, Option<Filter>, Pred)> =
    Vec::new();
  for t in [1i64, 3, 10, 50, 200, 1000] {
    out.push((
      format!("g<{t}"),
      None,
      Some(Filter::lt("g", t)),
      Box::new(move |_, g| g < t),
    ));
  }
  for s in [3usize, 40, 400, 2000] {
    let set = subset(s, rng);
    let s2 = set.clone();
    out.push((
      format!("ids({s})"),
      Some(IdFilter::ids(set.iter().copied())),
      None,
      Box::new(move |id, _| s2.contains(&id)),
    ));
    let s2 = set.clone();
    out.push((
      format!("not ids({s})"),
      Some(IdFilter::ids(set.iter().copied()).not()),
      None,
      Box::new(move |id, _| !s2.contains(&id)),
    ));
    // Bits AND lazy predicate.
    let s2 = set.clone();
    out.push((
      format!("ids({s}) and pred%3"),
      Some(IdFilter::ids(set.iter().copied()).and(IdFilter::predicate(
        |id| matches!(id, muninn::IdRef::U64(v) if v % 3 != 0),
      ))),
      None,
      Box::new(move |id, _| s2.contains(&id) && id % 3 != 0),
    ));
    // Predicate first, then bits.
    let s2 = set.clone();
    out.push((
      format!("pred%2 and ids({s})"),
      Some(
        IdFilter::predicate(
          |id| matches!(id, muninn::IdRef::U64(v) if v % 2 == 0),
        )
        .and(IdFilter::ids(set.iter().copied())),
      ),
      None,
      Box::new(move |id, _| id % 2 == 0 && s2.contains(&id)),
    ));
    // OR: bits or predicate.
    let s2 = set.clone();
    out.push((
      format!("ids({s}) or pred%97"),
      Some(IdFilter::ids(set.iter().copied()).or(IdFilter::predicate(
        |id| matches!(id, muninn::IdRef::U64(v) if v % 97 == 0),
      ))),
      None,
      Box::new(move |id, _| s2.contains(&id) || id % 97 == 0),
    ));
    // Payload bits AND id bits.
    let s2 = set.clone();
    out.push((
      format!("g<500 and ids({s})"),
      Some(IdFilter::ids(set.iter().copied())),
      Some(Filter::lt("g", 500i64)),
      Box::new(move |id, g| g < 500 && s2.contains(&id)),
    ));
  }
  // Broad bits AND a very selective predicate (bits is not the most
  // selective term).
  out.push((
    "g<900 and pred%997".into(),
    Some(IdFilter::predicate(
      |id| matches!(id, muninn::IdRef::U64(v) if v % 997 == 0),
    )),
    Some(Filter::lt("g", 900i64)),
    Box::new(|id, g| g < 900 && id % 997 == 0),
  ));
  // NOT of a predicate.
  out.push((
    "not pred%2".into(),
    Some(
      IdFilter::predicate(
        |id| matches!(id, muninn::IdRef::U64(v) if v % 2 == 0),
      )
      .not(),
    ),
    None,
    Box::new(|id, _| id % 2 != 0),
  ));
  // NOT of (bits AND pred).
  let set = subset(1000, rng);
  let s2 = set.clone();
  out.push((
    "not (ids(1000) and pred%2)".into(),
    Some(
      IdFilter::ids(set.iter().copied())
        .and(IdFilter::predicate(
          |id| matches!(id, muninn::IdRef::U64(v) if v % 2 == 0),
        ))
        .not(),
    ),
    None,
    Box::new(move |id, _| !(s2.contains(&id) && id % 2 == 0)),
  ));
  out
}

fn check(
  c: &Collection,
  metric: Metric,
  live: &Live,
  rng: &mut Rng,
  ctx: &str,
  stats: &mut BTreeMap<String, (f64, usize)>,
) {
  let fs = filters(rng, live);
  for _ in 0..2 {
    let q: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
    for (name, idf, pf, pred) in &fs {
      for k in [1usize, 10, 50] {
        let mut query = Query::vector(q.clone()).k(k);
        if let Some(f) = idf {
          query = query.id_filter(f.clone());
        }
        if let Some(f) = pf {
          query = query.filter(f.clone());
        }
        let hits = c.search(&query).unwrap();
        let mut truth: Vec<(f32, u64)> = live
          .docs
          .iter()
          .filter(|(id, (_, g))| pred(**id, *g))
          .map(|(id, (v, _))| (dist(metric, &q, v), *id))
          .collect();
        truth.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let want = truth.len().min(k);
        assert_eq!(
          hits.len(),
          want,
          "{ctx} {name} k {k}: got {} of {} matching",
          hits.len(),
          truth.len()
        );
        for h in &hits {
          let Id::U64(id) = h.id else { panic!() };
          let (v, g) = live
            .docs
            .get(&id)
            .unwrap_or_else(|| panic!("{ctx} {name}: dead doc {id}"));
          assert!(pred(id, *g), "{ctx} {name}: doc {id} fails the filter");
          let d = dist(metric, &q, v);
          let s = metric.score(d);
          assert!(
            (s - h.score).abs() <= 1e-3 * (1.0 + s.abs()),
            "{ctx} {name}: doc {id} score {} want {s} (stale version?)",
            h.score
          );
        }
        // Recall by distance cut-off (ties count as found).
        if want > 0 {
          let cut = truth[want - 1].0;
          let found = hits
            .iter()
            .filter(|h| {
              let Id::U64(id) = h.id else { panic!() };
              dist(metric, &q, &live.docs[&id].0) <= cut + 1e-4 * cut.abs()
            })
            .count();
          let e = stats.entry(name.clone()).or_default();
          e.0 += found as f64 / want as f64;
          e.1 += 1;
        }
      }
    }
  }
}

fn run(metric: Metric, quant: Quantization, seed: u64) {
  let dir = TestDir::new(&format!("filtered-{seed}"));
  let opts = Options {
    durability: Durability::NoSync,
    memtable_max_docs: 100_000,
    auto_compact: false,
    hnsw_min_docs: 500,
    ..Options::default()
  };
  let db = Database::open(dir.path(), opts).unwrap();
  let schema =
    Schema::vector(VectorField::hnsw(DIM, metric).with_quantization(quant))
      .with_filterable(FieldSpec::integer("g"));
  let c = db.create_collection("r", schema).unwrap();
  let mut rng = Rng::new(seed);
  let mut live = Live {
    docs: BTreeMap::new(),
  };
  let mut stats = BTreeMap::new();
  let upsert = |c: &Collection, live: &mut Live, rng: &mut Rng, ids: &[u64]| {
    let docs: Vec<Document> = ids
      .iter()
      .map(|&id| {
        let v: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
        let g = rng.below(1000) as i64;
        live.docs.insert(id, (v.clone(), g));
        Document::new(id).vector(v).payload(json!({ "g": g }))
      })
      .collect();
    for ch in docs.chunks(500) {
      c.upsert(ch).unwrap();
    }
  };
  let ctx = format!("{metric:?} {quant:?}");
  upsert(&c, &mut live, &mut rng, &(0..1500).collect::<Vec<_>>());
  c.flush().unwrap();
  upsert(&c, &mut live, &mut rng, &(1500..3000).collect::<Vec<_>>());
  check(
    &c,
    metric,
    &live,
    &mut rng,
    &format!("{ctx} p1"),
    &mut stats,
  );
  c.flush().unwrap();
  // Deletes and replacements.
  let dels: Vec<u64> = (0..450).map(|_| rng.below(3000)).collect();
  c.delete(&dels.iter().map(|&i| Id::U64(i)).collect::<Vec<_>>())
    .unwrap();
  for d in &dels {
    live.docs.remove(d);
  }
  let reps: Vec<u64> = (0..300)
    .map(|_| rng.below(3000))
    .collect::<HashSet<_>>()
    .into_iter()
    .collect();
  upsert(&c, &mut live, &mut rng, &reps);
  check(
    &c,
    metric,
    &live,
    &mut rng,
    &format!("{ctx} p2"),
    &mut stats,
  );
  c.flush().unwrap();
  upsert(&c, &mut live, &mut rng, &(3000..4500).collect::<Vec<_>>());
  c.flush().unwrap();
  let dels: Vec<u64> = (0..750).map(|_| rng.below(4500)).collect();
  c.delete(&dels.iter().map(|&i| Id::U64(i)).collect::<Vec<_>>())
    .unwrap();
  for d in &dels {
    live.docs.remove(d);
  }
  let reps: Vec<u64> = (0..150)
    .map(|_| rng.below(4500))
    .collect::<HashSet<_>>()
    .into_iter()
    .collect();
  upsert(&c, &mut live, &mut rng, &reps);
  check(
    &c,
    metric,
    &live,
    &mut rng,
    &format!("{ctx} p3"),
    &mut stats,
  );
  c.compact().unwrap();
  check(
    &c,
    metric,
    &live,
    &mut rng,
    &format!("{ctx} p4"),
    &mut stats,
  );
  for (name, (r, n)) in &stats {
    println!("{ctx} recall {name}: {:.4} over {n}", r / *n as f64);
  }
}

#[test]
fn filtered_search_matches_brute_force() {
  for (i, metric) in [Metric::L2, Metric::Cosine, Metric::Dot]
    .into_iter()
    .enumerate()
  {
    for (j, q) in [Quantization::None, Quantization::Sq8, Quantization::Binary]
      .into_iter()
      .enumerate()
    {
      run(metric, q, 10 + (i * 3 + j) as u64);
    }
  }
}

/// A flat, quantised segment with a selective id filter is scored at full
/// precision over the filter's ids (the engine used to special-case this;
/// the index now does it), so it is exact.
#[test]
fn quantised_flat_selective_filter_is_exact() {
  for quant in [Quantization::Sq8, Quantization::Binary] {
    let dir = TestDir::new("filtered-flatq");
    let opts = Options {
      durability: Durability::NoSync,
      memtable_max_docs: 100_000,
      auto_compact: false,
      ..Options::default()
    };
    let db = Database::open(dir.path(), opts).unwrap();
    let schema = Schema::vector(
      VectorField::flat(DIM, Metric::L2).with_quantization(quant),
    );
    let c = db.create_collection("r", schema).unwrap();
    let mut rng = Rng::new(3);
    let mut vecs = Vec::new();
    let docs: Vec<Document> = (0..20_000u64)
      .map(|id| {
        let v: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
        vecs.push(v.clone());
        Document::new(id).vector(v)
      })
      .collect();
    for ch in docs.chunks(1000) {
      c.upsert(ch).unwrap();
    }
    c.flush().unwrap();
    let ids: Vec<u64> = (0..600).map(|_| rng.below(20_000)).collect();
    let set: HashSet<u64> = ids.iter().copied().collect();
    let (mut r, mut nq) = (0.0, 0);
    for _ in 0..50 {
      let q: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
      let hits = c
        .search(
          &Query::vector(q.clone())
            .k(10)
            .id_filter(IdFilter::ids(set.iter().copied())),
        )
        .unwrap();
      let mut t: Vec<(f32, u64)> = set
        .iter()
        .map(|&id| (dist(Metric::L2, &q, &vecs[id as usize]), id))
        .collect();
      t.sort_by(|a, b| a.0.total_cmp(&b.0));
      let truth: HashSet<u64> = t[..10].iter().map(|x| x.1).collect();
      let got = hits
        .iter()
        .filter(|h| matches!(h.id, Id::U64(v) if truth.contains(&v)))
        .count();
      r += got as f64 / 10.0;
      nq += 1;
    }
    // Scored at full precision over the id set: exact.
    let r = r / nq as f64;
    println!("flat {quant:?}, 600 of 20000 pass: recall@10 {r:.4}");
    assert_eq!(r, 1.0, "flat {quant:?}");
  }
}
