//! File format round trips, corruption handling, merging, edge cases.

mod common;

use std::sync::Arc;

use common::{Reference, Rng, assert_same, random_query, zipf_corpus};
use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{AlignedBuf, BitSet, DocId, Error, Region};
use muninn_text::{
  Analyzer, AnalyzerConfig, CorpusStats, DELETED, FORMAT_VERSION, KIND,
  MemTextIndex, MergeInput, Stemmer, TextIndex, TextSearcher,
};

fn mem_from(docs: &[(DocId, String)], analyzer: &Analyzer) -> MemTextIndex {
  let mut m = MemTextIndex::new(analyzer.clone());
  for (id, text) in docs {
    m.add(*id, text).unwrap();
  }
  m
}

fn bytes_of(m: &MemTextIndex) -> AlignedBuf {
  m.write_to(AlignedBuf::new()).unwrap()
}

#[test]
fn round_trip_restores_analyzer_and_results() {
  let analyzer = Analyzer::new(AnalyzerConfig::english()).unwrap();
  let corpus = zipf_corpus(5, 2000, 1500);
  let mem = mem_from(&corpus.docs, &analyzer);
  let buf = bytes_of(&mem);
  let idx = TextIndex::open(Region::new(buf.clone())).unwrap();
  idx.verify_checksum().unwrap();
  assert_eq!(*idx.analyzer().config(), AnalyzerConfig::english());
  assert_eq!(idx.num_docs(), mem.num_docs());
  assert_eq!(idx.doc_count(), mem.doc_count());
  assert_eq!(idx.total_doc_len(), mem.total_doc_len());
  assert_eq!(idx.num_terms(), mem.num_terms() as u64);
  assert_eq!(idx.size_bytes(), buf.len());
  for id in 0..mem.num_docs() as DocId + 2 {
    assert_eq!(idx.doc_len(id), mem.doc_len(id));
  }
  // Writing is deterministic.
  assert_eq!(bytes_of(&mem).as_slice(), buf.as_slice());
  let mut rng = Rng::new(3);
  for _ in 0..200 {
    let q = random_query(&mut rng, 1500);
    assert_eq!(
      idx.search(&q, 10, None).unwrap(),
      mem.search(&q, 10, None).unwrap()
    );
  }
  // Query text is analysed with the stored analyzer: stemmed, folded,
  // stop words removed.
  let mut m = MemTextIndex::new(analyzer.clone());
  m.add(0, "The runners were running to the Café").unwrap();
  m.add(1, "nothing here").unwrap();
  let f = TextIndex::open(Region::new(bytes_of(&m))).unwrap();
  let hits = f.search("RUN cafe the", 10, None).unwrap();
  assert_eq!(hits.len(), 1);
  assert_eq!(hits[0].id, 0);
  assert!(f.search("the to", 10, None).unwrap().is_empty());
}

struct Trunc;
impl Stemmer for Trunc {
  fn name(&self) -> &str {
    "trunc4"
  }
  fn stem(&self, w: &mut String) {
    if let Some((i, _)) = w.char_indices().nth(4) {
      w.truncate(i);
    }
  }
}

#[test]
fn custom_stemmer_must_be_supplied_on_open() {
  let analyzer =
    Analyzer::with_stemmer(AnalyzerConfig::default(), Arc::new(Trunc)).unwrap();
  let mut m = MemTextIndex::new(analyzer.clone());
  m.add(0, "searching engines").unwrap();
  let buf = bytes_of(&m);
  let err = TextIndex::open(Region::new(buf.clone())).unwrap_err();
  assert!(err.to_string().contains("open_with_analyzer"), "{err}");
  let wrong = Analyzer::new(AnalyzerConfig::english()).unwrap();
  assert!(
    TextIndex::open_with_analyzer(Region::new(buf.clone()), wrong).is_err()
  );
  let idx = TextIndex::open_with_analyzer(Region::new(buf), analyzer).unwrap();
  assert_eq!(idx.search("searchlight", 5, None).unwrap().len(), 1);
}

#[test]
fn rejects_wrong_kind_version_and_truncation() {
  let mem = mem_from(&zipf_corpus(6, 300, 200).docs, &Analyzer::default());
  let buf = bytes_of(&mem);
  for cut in [0, 10, 64, 200, buf.len() / 2, buf.len() - 1] {
    let r = Region::copy_from(&buf.as_slice()[..cut]);
    assert!(TextIndex::open(r).is_err(), "truncated to {cut}");
  }
  let mut other =
    ContainerWriter::new(AlignedBuf::new(), tag("hnsw"), 1).unwrap();
  other.section(tag("meta"), b"x").unwrap();
  let other = other.finish().unwrap();
  assert!(TextIndex::open(Region::new(other)).is_err());
  let mut v2 = buf.clone();
  v2.as_mut_slice()[16..20]
    .copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
  assert!(matches!(
    TextIndex::open(Region::new(v2)),
    Err(Error::Corrupt(_))
  ));
  // A container of the right kind with sections missing.
  let mut empty =
    ContainerWriter::new(AlignedBuf::new(), KIND, FORMAT_VERSION).unwrap();
  empty.section(tag("norms"), b"").unwrap();
  let empty = empty.finish().unwrap();
  assert!(TextIndex::open(Region::new(empty)).is_err());
}

/// Flipping any byte of a file must never panic: open fails, or search
/// fails or returns something, and the checksum always catches it.
#[test]
fn every_single_byte_corruption_is_survivable() {
  let corpus = zipf_corpus(8, 400, 40);
  let mem = mem_from(&corpus.docs, &Analyzer::default());
  let buf = bytes_of(&mem);
  let queries: Vec<String> = {
    let mut rng = Rng::new(4);
    (0..4).map(|_| random_query(&mut rng, 40)).collect()
  };
  let mut opened = 0;
  for at in 0..buf.len() {
    for mask in [0x01u8, 0xff] {
      let mut bad = buf.clone();
      bad.as_mut_slice()[at] ^= mask;
      let Ok(idx) = TextIndex::open(Region::new(bad)) else {
        continue;
      };
      opened += 1;
      // The container footer's reserved zero padding (bytes 16..56 of
      // the last 64) is neither checksummed nor validated by core.
      let footer_pad = buf.len() - 48..buf.len() - 8;
      if !footer_pad.contains(&at) {
        assert!(idx.verify_checksum().is_err(), "flip at {at} undetected");
      }
      for q in &queries {
        let _ = idx.search(q, 10, None);
        let _ = idx.search_exhaustive(q, 10, None, None);
        let _ = idx.stats_for(q);
      }
      let map: Vec<DocId> = (0..idx.num_docs() as DocId).collect();
      let input = [MergeInput {
        index: &idx,
        doc_map: &map,
      }];
      let _ = TextIndex::merge(&input, idx.num_docs(), AlignedBuf::new());
    }
  }
  assert!(opened > 0);
}

#[test]
fn edge_cases() {
  let empty = MemTextIndex::new(Analyzer::default());
  assert_eq!(empty.num_docs(), 0);
  assert!(empty.search("anything", 10, None).unwrap().is_empty());
  let frozen = empty.freeze().unwrap();
  assert_eq!(frozen.num_docs(), 0);
  assert_eq!(frozen.num_terms(), 0);
  assert!(frozen.search("anything", 10, None).unwrap().is_empty());
  let s = frozen.stats_for("a b").unwrap();
  assert_eq!((s.doc_count, s.terms.len()), (0, 2));

  let mut m = MemTextIndex::new(Analyzer::default());
  m.add(0, "alpha beta").unwrap();
  m.add(3, "beta gamma gamma").unwrap();
  m.add(4, "").unwrap();
  assert!(m.add(4, "again").is_err());
  assert!(m.add(2, "back").is_err());
  assert!(m.add(DocId::MAX, "reserved").is_err());
  assert_eq!(m.num_docs(), 5);
  assert_eq!(m.doc_count(), 3);
  assert_eq!(m.doc_len(1), None);
  assert_eq!(m.doc_len(4), Some(0));
  let f = m.freeze().unwrap();
  for idx in [&m as &dyn TextSearcher, &f] {
    assert!(idx.search("", 10, None).unwrap().is_empty());
    assert!(idx.search(" ,;. ", 10, None).unwrap().is_empty());
    assert!(idx.search("unknown words", 10, None).unwrap().is_empty());
    assert!(idx.search("beta", 0, None).unwrap().is_empty());
    // Huge k must not try to allocate k slots.
    assert_eq!(idx.search("beta", usize::MAX, None).unwrap().len(), 2);
    let ids: Vec<DocId> = idx
      .search("beta gamma unknown", 10, None)
      .unwrap()
      .iter()
      .map(|h| h.id)
      .collect();
    assert_eq!(ids, [3, 0]);
    let none = BitSet::new(5);
    assert!(idx.search("beta", 10, Some(&none)).unwrap().is_empty());
    // Supplied stats must cover every query term.
    let partial = CorpusStats {
      doc_count: 3,
      total_doc_len: 5,
      terms: vec![],
    };
    assert!(matches!(
      idx.search_with_stats("beta", 10, None, &partial),
      Err(Error::InvalidArgument(_))
    ));
    // ...even when they make no sense (df above N, zero length).
    let odd = CorpusStats {
      doc_count: 1,
      total_doc_len: 0,
      terms: vec![muninn_text::TermDocFreq {
        term: "beta".into(),
        doc_freq: 50,
      }],
    };
    assert_eq!(
      idx.search_with_stats("beta", 10, None, &odd).unwrap().len(),
      2
    );
  }
}

#[test]
fn huge_term_frequency_and_long_documents() {
  let mut m = MemTextIndex::new(Analyzer::default());
  m.add(0, &"spam ".repeat(70_000)).unwrap();
  m.add(1, "spam eggs").unwrap();
  for i in 2..300 {
    m.add(i, &format!("spam {}", "filler ".repeat(i as usize % 50)))
      .unwrap();
  }
  let f = m.freeze().unwrap();
  let docs: Vec<(DocId, String)> = std::iter::once((0, "spam ".repeat(70_000)))
    .chain(std::iter::once((1, "spam eggs".to_string())))
    .chain(
      (2..300)
        .map(|i| (i, format!("spam {}", "filler ".repeat(i as usize % 50)))),
    )
    .collect();
  let r = Reference::new(&Analyzer::default(), &docs);
  for k in [1, 2, 10, 300] {
    let want = r.search("spam", k, None, None);
    assert_same(&m.search("spam", k, None).unwrap(), &want, "mem");
    assert_same(&f.search("spam", k, None).unwrap(), &want, "frozen");
  }
}

/// Merging with deletions must produce the same bytes as indexing the
/// surviving documents from scratch under their new ids.
#[test]
fn merge_with_deletions_equals_building_from_scratch() {
  let analyzer = Analyzer::new(AnalyzerConfig::english()).unwrap();
  let mut rng = Rng::new(21);
  let corpora: Vec<_> = (0..3)
    .map(|i| zipf_corpus(30 + i, 700 + 300 * i as usize, 900))
    .collect();
  let segs: Vec<TextIndex> = corpora
    .iter()
    .map(|c| mem_from(&c.docs, &analyzer).freeze().unwrap())
    .collect();
  for mode in ["compact", "permuted", "sparse", "all-deleted"] {
    // Survivors in input order, with their texts.
    let mut survivors: Vec<(usize, DocId, &str)> = Vec::new();
    for (si, c) in corpora.iter().enumerate() {
      for (id, text) in &c.docs {
        if mode != "all-deleted" && !rng.chance(0.2) {
          survivors.push((si, *id, text));
        }
      }
    }
    let n = survivors.len();
    let new_ids: Vec<DocId> = match mode {
      // Reversed: the merge has to sort every posting list.
      "permuted" => (0..n as DocId).rev().collect(),
      "sparse" => (0..n as DocId).map(|i| i * 2 + 1).collect(),
      _ => (0..n as DocId).collect(),
    };
    let num_docs = new_ids.iter().max().map_or(0, |&m| m as usize + 1);
    let mut maps: Vec<Vec<DocId>> =
      segs.iter().map(|s| vec![DELETED; s.num_docs()]).collect();
    for (&(si, old, _), &new) in survivors.iter().zip(&new_ids) {
      maps[si][old as usize] = new;
    }
    let inputs: Vec<MergeInput<'_>> = segs
      .iter()
      .zip(&maps)
      .map(|(index, m)| MergeInput { index, doc_map: m })
      .collect();
    let merged =
      TextIndex::merge(&inputs, num_docs, AlignedBuf::new()).unwrap();

    let mut scratch_docs: Vec<(DocId, String)> = survivors
      .iter()
      .zip(&new_ids)
      .map(|(&(_, _, t), &new)| (new, t.to_string()))
      .collect();
    scratch_docs.sort_by_key(|d| d.0);
    let scratch = mem_from(&scratch_docs, &analyzer);
    let want = bytes_of(&scratch);
    assert_eq!(merged.len(), want.len(), "{mode}: size");
    assert!(merged.as_slice() == want.as_slice(), "{mode}: bytes differ");

    let idx = TextIndex::open(Region::new(merged)).unwrap();
    idx.verify_checksum().unwrap();
    // The output id space may extend past the last surviving document.
    let padded = TextIndex::merge(&inputs, num_docs + 3, AlignedBuf::new())
      .and_then(|b| TextIndex::open(Region::new(b)))
      .unwrap();
    assert_eq!(padded.num_docs(), num_docs + 3);
    assert_eq!(padded.doc_count(), idx.doc_count());
    let reference = Reference::new(&analyzer, &scratch_docs);
    for _ in 0..50 {
      let q = random_query(&mut rng, 900);
      let want = reference.search(&q, 10, None, None);
      assert_same(&idx.search(&q, 10, None).unwrap(), &want, mode);
      assert_same(&padded.search(&q, 10, None).unwrap(), &want, mode);
    }
  }
}

#[test]
fn merge_rejects_bad_maps() {
  let a = Analyzer::default();
  let s1 = mem_from(&[(0, "x y".into()), (1, "y".into())], &a)
    .freeze()
    .unwrap();
  let s2 = mem_from(&[(0, "z".into())], &a).freeze().unwrap();
  let e = Analyzer::new(AnalyzerConfig::english()).unwrap();
  let s3 = mem_from(&[(0, "z".into())], &e).freeze().unwrap();
  let m = |index, doc_map| MergeInput { index, doc_map };
  let out = AlignedBuf::new;
  assert!(TextIndex::merge(&[], 0, out()).is_err());
  assert!(TextIndex::merge(&[m(&s1, &[0])], 5, out()).is_err());
  assert!(TextIndex::merge(&[m(&s1, &[0, 7])], 5, out()).is_err());
  assert!(
    TextIndex::merge(&[m(&s1, &[0, 1]), m(&s2, &[1])], 5, out()).is_err()
  );
  assert!(TextIndex::merge(&[m(&s2, &[0]), m(&s3, &[1])], 5, out()).is_err());
  let ok =
    TextIndex::merge(&[m(&s1, &[DELETED, 0]), m(&s2, &[1])], 2, out()).unwrap();
  let idx = TextIndex::open(Region::new(ok)).unwrap();
  assert_eq!(idx.doc_count(), 2);
  assert_eq!(idx.doc_freq("x").unwrap(), 0);
  assert_eq!(idx.doc_freq("y").unwrap(), 1);
  assert_eq!(idx.doc_freq("z").unwrap(), 1);
}

#[test]
fn huge_k_on_both_search_paths() {
  let mut m = MemTextIndex::new(Analyzer::default());
  for i in 0..300 {
    m.add(i, if i % 3 == 0 { "a b" } else { "a" }).unwrap();
  }
  let f = m.freeze().unwrap();
  let stats = f.stats_for("a b").unwrap();
  assert_eq!(
    m.search_exhaustive("a b", usize::MAX, None, None)
      .unwrap()
      .len(),
    300
  );
  assert_eq!(
    f.search_exhaustive("a b", usize::MAX, None, None)
      .unwrap()
      .len(),
    300
  );
  assert_eq!(
    f.search_with_stats("a b", usize::MAX, None, &stats)
      .unwrap()
      .len(),
    300
  );
}

/// Review finding: dictionary block offsets were trusted, so a corrupt
/// skip index overflowed (a panic in debug builds) and a corrupt
/// postings offset read the wrong bytes.
#[test]
fn corrupt_dictionary_index_is_rejected() {
  let mut m = MemTextIndex::new(Analyzer::default());
  for i in 0..300 {
    m.add(i, &format!("a w{}", i % 40)).unwrap();
  }
  let buf = bytes_of(&m);
  let region = Region::new(buf.clone());
  let base = region.bytes().as_ptr() as usize;
  let c = Container::open(region).unwrap();
  let idx = c.section(tag("dictidx")).unwrap();
  let off = idx.bytes().as_ptr() as usize - base;
  let skips = c.section(tag("skips")).unwrap().len() as u64 / 32;
  let postings = c.section(tag("postings")).unwrap().len() as u64;
  // Each dictionary block entry: dict offset, postings offset, skip index.
  for (field, value) in [
    (2, u64::MAX),
    (2, u64::MAX - 1),
    (2, skips + 1),
    (1, u64::MAX),
    (1, postings + 1),
  ] {
    let mut bad = buf.clone();
    let at = off + field * 8;
    bad.as_mut_slice()[at..at + 8].copy_from_slice(&value.to_le_bytes());
    let opened = TextIndex::open(Region::new(bad));
    assert!(opened.is_err(), "field {field} = {value} accepted");
  }
  // A dictionary block that legitimately starts at the end of the skip
  // records (only single-block terms follow) still opens.
  TextIndex::open(Region::new(buf)).unwrap();
}

#[test]
fn queries_with_too_many_distinct_terms_are_rejected() {
  let mut m = MemTextIndex::new(Analyzer::default());
  m.add(0, "w1 w2 w3").unwrap();
  let f = m.freeze().unwrap();
  let max = muninn_text::MAX_QUERY_TERMS;
  let words = |n: usize| {
    (0..n)
      .map(|i| format!("w{i}"))
      .collect::<Vec<_>>()
      .join(" ")
  };
  let ok = words(max);
  let too_many = format!("{} w1 w2", words(max + 1));
  for idx in [&m as &dyn TextSearcher, &f] {
    assert_eq!(idx.search(&ok, 5, None).unwrap().len(), 1);
    let stats = idx.stats_for(&ok).unwrap();
    assert_eq!(stats.terms.len(), max);
    assert_eq!(
      idx.search_with_stats(&ok, 5, None, &stats).unwrap().len(),
      1
    );
    assert!(matches!(
      idx.search(&too_many, 5, None),
      Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
      idx.stats_for(&too_many),
      Err(Error::InvalidArgument(_))
    ));
    // Repeating terms does not count against the limit.
    let repeated = "w1 ".repeat(100_000);
    assert_eq!(idx.search(&repeated, 5, None).unwrap().len(), 1);
  }
}
