//! Randomized differential tests: index evaluation (mutable, frozen and
//! merged) must equal brute-force evaluation over the raw JSON.

mod common;

use common::{Rng, brute, filter, payload, schema};
use muninn_core::{AlignedBuf, DocId, Region};
use muninn_filter::{DELETED, FilterIndex, FrozenIndex, MemIndex, Schema};
use serde_json::Value as Json;

fn freeze(m: &MemIndex) -> FrozenIndex {
  let buf = m.write_to(AlignedBuf::new()).unwrap();
  let ix = FrozenIndex::open(Region::new(buf)).unwrap();
  ix.verify_checksum().unwrap();
  ix
}

/// Random docs with strictly increasing ids and occasional gaps.
fn docs(r: &mut Rng, n: usize) -> Vec<(u32, Json)> {
  let mut id = 0u32;
  (0..n)
    .map(|_| {
      id += if r.chance(0.1) {
        1 + r.below(3) as u32
      } else {
        0
      };
      let d = (id, payload(r));
      id += 1;
      d
    })
    .collect()
}

fn build(schema: &Schema, docs: &[(u32, Json)]) -> MemIndex {
  let mut m = MemIndex::new(schema.clone());
  for (id, p) in docs {
    m.add(*id, p).unwrap();
  }
  m
}

fn ids(ix: &dyn FilterIndex, f: &muninn_filter::Filter, n: usize) -> Vec<u32> {
  let b = ix.evaluate(f, n).unwrap();
  assert_eq!(b.capacity(), n);
  b.iter().collect()
}

#[test]
fn index_evaluation_equals_brute_force() {
  let schema = schema();
  let mut r = Rng::new(0x5EED);
  let mut checked = 0;
  let mut nonempty = 0;
  for round in 0..120 {
    let n = if round == 0 {
      0
    } else {
      1 + r.below(400) as usize
    };
    let docs = docs(&mut r, n);
    let mem = build(&schema, &docs);
    let frozen = freeze(&mem);
    assert_eq!(frozen.doc_count(), mem.num_docs());
    let max = mem.num_docs();
    for _ in 0..40 {
      let f = filter(&mut r, 4, false);
      // Exactly the indexed range, beyond it, and truncated.
      for num_docs in [max, max + 70, max / 2] {
        let want = brute(&schema, &f, &docs, num_docs);
        assert_eq!(ids(&mem, &f, num_docs), want, "mem {f:?} n={num_docs}");
        assert_eq!(
          ids(&frozen, &f, num_docs),
          want,
          "frozen {f:?} n={num_docs}"
        );
        for est in [mem.estimate(&f, num_docs), frozen.estimate(&f, num_docs)] {
          assert!((0.0..=1.0).contains(&est), "estimate {est} for {f:?}");
        }
        checked += 1;
        nonempty += usize::from(!want.is_empty() && want.len() < num_docs);
      }
    }
  }
  // The generator must produce informative filters, not all-or-nothing.
  assert!(
    nonempty * 3 > checked,
    "{nonempty} of {checked} informative"
  );
}

/// Merges `parts` with deletions under `order`, and checks the result
/// against a from-scratch build of the survivors and the brute force.
fn check_merge(r: &mut Rng, reverse: bool) {
  let schema = schema();
  let parts: Vec<Vec<(u32, Json)>> = (0..3)
    .map(|k| {
      let n = if k == 1 && r.chance(0.3) {
        0
      } else {
        r.below(300) as usize
      };
      docs(r, n)
    })
    .collect();
  let frozen: Vec<FrozenIndex> =
    parts.iter().map(|p| freeze(&build(&schema, p))).collect();
  // Survivors get consecutive new ids, in input order or reversed.
  let mut survivors: Vec<(usize, u32)> = Vec::new();
  let mut remaps: Vec<Vec<DocId>> = Vec::new();
  for (k, ix) in frozen.iter().enumerate() {
    let mut remap = vec![DELETED; ix.doc_count() + r.below(3) as usize];
    for (id, _) in &parts[k] {
      if !r.chance(0.3) {
        survivors.push((k, *id));
        remap[*id as usize] = 0;
      }
    }
    remaps.push(remap);
  }
  let total = survivors.len() as u32;
  for (i, &(k, id)) in survivors.iter().enumerate() {
    let new = if reverse {
      total - 1 - i as u32
    } else {
      i as u32
    };
    remaps[k][id as usize] = new;
  }
  let inputs: Vec<(&FrozenIndex, &[DocId])> = frozen
    .iter()
    .zip(&remaps)
    .map(|(ix, m)| (ix, m.as_slice()))
    .collect();
  let buf = FrozenIndex::merge(&schema, &inputs, AlignedBuf::new()).unwrap();
  let merged = FrozenIndex::open(Region::new(buf.clone())).unwrap();
  merged.verify_checksum().unwrap();

  // From scratch: survivors re-added under their new ids, in id order.
  let mut fresh: Vec<(u32, Json)> = survivors
    .iter()
    .map(|&(k, id)| {
      let p = parts[k].iter().find(|d| d.0 == id).unwrap().1.clone();
      (remaps[k][id as usize], p)
    })
    .collect();
  fresh.sort_by_key(|d| d.0);
  let scratch = build(&schema, &fresh);
  let scratch_buf = scratch.write_to(AlignedBuf::new()).unwrap();
  assert_eq!(merged.doc_count(), scratch.num_docs());
  assert!(
    buf.as_slice() == scratch_buf.as_slice(),
    "merged file differs from a fresh build"
  );
  let n = fresh.len();
  for _ in 0..30 {
    let f = filter(r, 4, false);
    assert_eq!(ids(&merged, &f, n), brute(&schema, &f, &fresh, n), "{f:?}");
  }
}

#[test]
fn merge_with_deletions_equals_fresh_build() {
  let mut r = Rng::new(77);
  for _ in 0..40 {
    check_merge(&mut r, false);
  }
}

#[test]
fn merge_with_reordering_remap_equals_fresh_build() {
  let mut r = Rng::new(78);
  for _ in 0..40 {
    check_merge(&mut r, true);
  }
}

#[test]
fn merge_of_merged_indexes() {
  // Merging is closed: merged indexes merge again correctly.
  let schema = schema();
  let mut r = Rng::new(79);
  let a = docs(&mut r, 200);
  let b = docs(&mut r, 150);
  let fa = freeze(&build(&schema, &a));
  let fb = freeze(&build(&schema, &b));
  let ka = (0..fa.doc_count() as u32).collect::<Vec<_>>();
  let off = fa.doc_count() as u32;
  let kb = (0..fb.doc_count() as u32)
    .map(|i| i + off)
    .collect::<Vec<_>>();
  let ab = FrozenIndex::merge(
    &schema,
    &[(&fa, ka.as_slice()), (&fb, kb.as_slice())],
    AlignedBuf::new(),
  )
  .unwrap();
  let ab = FrozenIndex::open(Region::new(ab)).unwrap();
  // Drop every third doc in a second merge.
  let mut remap = vec![DELETED; ab.doc_count()];
  let mut next = 0;
  let all: Vec<(u32, Json)> = a
    .iter()
    .cloned()
    .chain(b.iter().map(|(id, p)| (id + off, p.clone())))
    .collect();
  let mut fresh = Vec::new();
  for (id, p) in &all {
    if id % 3 != 0 {
      remap[*id as usize] = next;
      fresh.push((next, p.clone()));
      next += 1;
    }
  }
  let buf =
    FrozenIndex::merge(&schema, &[(&ab, remap.as_slice())], AlignedBuf::new())
      .unwrap();
  let merged = FrozenIndex::open(Region::new(buf)).unwrap();
  let n = fresh.len();
  for _ in 0..100 {
    let f = filter(&mut r, 4, false);
    assert_eq!(ids(&merged, &f, n), brute(&schema, &f, &fresh, n), "{f:?}");
  }
}
