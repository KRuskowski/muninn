//! The immutable on-disk text index and its container format.
//!
//! A `muninn_core::container` of kind `"text"`, version
//! [`FORMAT_VERSION`], with these sections (written in this order):
//!
//! | tag        | contents |
//! |------------|----------|
//! | `norms`    | `u8` per doc id: quantised length ([`crate::norm`]) |
//! | `doclens`  | `u32` per doc id: exact length, `u32::MAX` = absent |
//! | `postings` | per term, in term order: its blocks ([`crate::postings`]), then 8 zero bytes |
//! | `skips`    | 32-byte skip records of every multi-block term, in term order |
//! | `dict`     | front-coded term dictionary, blocks of 16 terms |
//! | `dictidx`  | per dictionary block, 3 × `u64`: dict offset, postings offset, first skip index |
//! | `meta`     | counts and the analyzer settings |
//!
//! Dictionary entry: the first term of a block is `varint len, bytes`,
//! later ones `varint shared_prefix, varint suffix_len, suffix`. Then
//! `varint df, varint data_len`; terms with at most 128 documents (one
//! block, no skip records) add `varint last_doc, u8 n, n × (u16 tf, u8
//! norm)` impacts. Postings offsets and skip indexes are implied by
//! summing `data_len` and `ceil(df / 128)` from the block's start.
//!
//! `meta`: `u32 num_docs` (id space), `u32 0`, `u64 doc_count`, `u64
//! total_doc_len`, `u64 num_terms`, `u64 num_skips`, then the analyzer
//! settings. All little-endian.
//!
//! Opening validates section sizes and the analyzer; posting blocks are
//! validated as they are decoded, so corrupt files produce errors, not
//! panics. [`TextIndex::verify_checksum`] checks the whole-file CRC.

use std::io::Write;

use muninn_core::container::{Container, ContainerWriter, tag};
use muninn_core::{DocFilter, DocId, Error, Hit, Region, Result};

use crate::TextSearcher;
use crate::analyzer::{Analyzer, AnalyzerConfig, decode_config, encode_config};
use crate::bitpack::BLOCK;
use crate::impacts::{IMPACTS, Impacts};
use crate::norm::encode_doc_len;
use crate::postings::{Cursor, END, Frozen, PostingList, SkipRec};
use crate::search::{self, Source};
use crate::stats::CorpusStats;
use crate::varint;

/// Container kind of text index files.
pub const KIND: [u8; 8] = tag("text");
/// Current format version.
pub const FORMAT_VERSION: u32 = 1;
/// Marks a deleted document in a [`MergeInput::doc_map`].
pub const DELETED: DocId = DocId::MAX;

/// Exact-length value of an id with no document.
pub(crate) const ABSENT: u32 = u32::MAX;

const DICT_BLOCK: u64 = 16;
const META: [u8; 8] = tag("meta");
const NORMS: [u8; 8] = tag("norms");
const DOCLENS: [u8; 8] = tag("doclens");
const POSTINGS: [u8; 8] = tag("postings");
const SKIPS: [u8; 8] = tag("skips");
const DICT: [u8; 8] = tag("dict");
const DICTIDX: [u8; 8] = tag("dictidx");
const META_FIXED: usize = 40;

/// Streams a text index container. Terms must arrive in byte order.
pub(crate) struct Writer<W: Write> {
  cw: ContainerWriter<W>,
  config: Vec<u8>,
  num_docs: u32,
  doc_count: u64,
  total_len: u64,
  post_pos: u64,
  skips: Vec<SkipRec>,
  dict: Vec<u8>,
  dictidx: Vec<u64>,
  num_terms: u64,
  prev: Vec<u8>,
}

impl<W: Write> Writer<W> {
  pub fn new(
    out: W,
    config: &AnalyzerConfig,
    norms: &[u8],
    lens: &[u32],
    doc_count: u64,
    total_len: u64,
  ) -> Result<Self> {
    let num_docs = u32::try_from(norms.len())
      .map_err(|_| Error::invalid("too many document ids"))?;
    debug_assert_eq!(norms.len(), lens.len());
    let mut cw = ContainerWriter::new(out, KIND, FORMAT_VERSION)?;
    cw.section(NORMS, norms)?;
    cw.pod_section(DOCLENS, lens)?;
    cw.begin_section(POSTINGS)?;
    let mut cfg = Vec::new();
    encode_config(config, &mut cfg);
    Ok(Self {
      cw,
      config: cfg,
      num_docs,
      doc_count,
      total_len,
      post_pos: 0,
      skips: Vec::new(),
      dict: Vec::new(),
      dictidx: Vec::new(),
      num_terms: 0,
      prev: Vec::new(),
    })
  }

  pub fn add_term(&mut self, term: &[u8], f: &Frozen<'_>) -> Result<()> {
    if self.num_terms > 0 && term <= self.prev.as_slice() {
      return Err(Error::invalid("terms must be written in byte order"));
    }
    let data_len = f.head.len() + f.tail_block.len();
    let data_len = u32::try_from(data_len)
      .map_err(|_| Error::invalid("postings of one term exceed 4 GiB"))?;
    if self.num_terms.is_multiple_of(DICT_BLOCK) {
      self.dictidx.extend_from_slice(&[
        self.dict.len() as u64,
        self.post_pos,
        self.skips.len() as u64,
      ]);
      varint::put(&mut self.dict, term.len() as u64);
      self.dict.extend_from_slice(term);
    } else {
      let shared = term
        .iter()
        .zip(&self.prev)
        .take_while(|(a, b)| a == b)
        .count();
      varint::put(&mut self.dict, shared as u64);
      varint::put(&mut self.dict, (term.len() - shared) as u64);
      self.dict.extend_from_slice(&term[shared..]);
    }
    varint::put(&mut self.dict, u64::from(f.df));
    varint::put(&mut self.dict, u64::from(data_len));
    if f.skips.len() == 1 {
      varint::put(&mut self.dict, u64::from(f.last_doc));
      let imp = &f.skips[0].impacts;
      let n = imp.iter().take_while(|&&i| i != 0).count();
      self.dict.push(n as u8);
      for &i in &imp[..n] {
        self.dict.extend_from_slice(&(i as u16).to_le_bytes());
        self.dict.push((i >> 16) as u8);
      }
    } else {
      self.skips.extend_from_slice(&f.skips);
    }
    self.cw.write(f.head)?;
    self.cw.write(&f.tail_block)?;
    self.post_pos += u64::from(data_len);
    self.prev.clear();
    self.prev.extend_from_slice(term);
    self.num_terms += 1;
    Ok(())
  }

  pub fn finish(mut self) -> Result<W> {
    // Slack so the block decoder's 8-byte reads stay inside the section.
    self.cw.write(&[0u8; 8])?;
    self.cw.end_section()?;
    self.cw.pod_section(SKIPS, &self.skips)?;
    self.cw.section(DICT, &self.dict)?;
    self.cw.pod_section(DICTIDX, &self.dictidx)?;
    let mut meta = Vec::with_capacity(META_FIXED + self.config.len());
    meta.extend_from_slice(&self.num_docs.to_le_bytes());
    meta.extend_from_slice(&0u32.to_le_bytes());
    meta.extend_from_slice(&self.doc_count.to_le_bytes());
    meta.extend_from_slice(&self.total_len.to_le_bytes());
    meta.extend_from_slice(&self.num_terms.to_le_bytes());
    meta.extend_from_slice(&(self.skips.len() as u64).to_le_bytes());
    meta.extend_from_slice(&self.config);
    self.cw.section(META, &meta)?;
    self.cw.finish()
  }
}

/// Where a term's postings live.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TermMeta {
  df: u32,
  data_off: u64,
  /// `Some` for single-block terms (impacts inline), else the index of
  /// the first skip record.
  single: Option<(DocId, Impacts)>,
  skip_idx: u64,
}

/// Sequential reader of one dictionary block.
struct BlockReader<'a> {
  dict: &'a [u8],
  pos: usize,
  left: u64,
  first: bool,
  term: Vec<u8>,
  post_off: u64,
  skip_idx: u64,
}

impl BlockReader<'_> {
  fn bad() -> Error {
    Error::corrupt("text index dictionary is corrupt")
  }

  /// Decodes the next entry into `self.term`.
  fn next(&mut self) -> Result<Option<TermMeta>> {
    if self.left == 0 {
      return Ok(None);
    }
    self.left -= 1;
    let d = self.dict;
    let p = &mut self.pos;
    let rd = |p: &mut usize| varint::get(d, p).ok_or_else(Self::bad);
    if self.first {
      self.first = false;
      let len = rd(p)? as usize;
      let bytes = d.get(*p..p.saturating_add(len)).ok_or_else(Self::bad)?;
      self.term.clear();
      self.term.extend_from_slice(bytes);
      *p += len;
    } else {
      let shared = rd(p)? as usize;
      let len = rd(p)? as usize;
      if shared > self.term.len() {
        return Err(Self::bad());
      }
      let bytes = d.get(*p..p.saturating_add(len)).ok_or_else(Self::bad)?;
      self.term.truncate(shared);
      self.term.extend_from_slice(bytes);
      *p += len;
    }
    let df = u32::try_from(rd(p)?).map_err(|_| Self::bad())?;
    let data_len = rd(p)?;
    if df == 0 {
      return Err(Self::bad());
    }
    let single = if df as usize <= BLOCK {
      let last = u32::try_from(rd(p)?).map_err(|_| Self::bad())?;
      let n = usize::from(*d.get(*p).ok_or_else(Self::bad)?);
      *p += 1;
      if n > IMPACTS {
        return Err(Self::bad());
      }
      let mut imp = [0u32; IMPACTS];
      for slot in imp.iter_mut().take(n) {
        let b = d.get(*p..*p + 3).ok_or_else(Self::bad)?;
        let tf = u32::from(u16::from_le_bytes([b[0], b[1]]));
        if tf == 0 {
          return Err(Self::bad());
        }
        *slot = tf | (u32::from(b[2]) << 16);
        *p += 3;
      }
      Some((last, imp))
    } else {
      None
    };
    let meta = TermMeta {
      df,
      data_off: self.post_off,
      single,
      skip_idx: self.skip_idx,
    };
    self.post_off = self.post_off.saturating_add(data_len);
    if single.is_none() {
      self.skip_idx = self
        .skip_idx
        .checked_add((df as u64).div_ceil(BLOCK as u64))
        .ok_or_else(Self::bad)?;
    }
    Ok(Some(meta))
  }
}

/// An immutable text index opened zero-copy from a [`Region`].
#[derive(Clone, Debug)]
pub struct TextIndex {
  container: Container,
  analyzer: Analyzer,
  file_len: usize,
  num_docs: u32,
  doc_count: u64,
  total_len: u64,
  num_terms: u64,
  norms: Region,
  lens: Region,
  postings: Region,
  skips: Region,
  dict: Region,
  dictidx: Region,
}

impl TextIndex {
  /// Opens a text index file. The analyzer is restored from the file;
  /// files that use a custom stemmer need [`Self::open_with_analyzer`].
  /// Checks structure, not the checksum.
  pub fn open(region: Region) -> Result<Self> {
    let (container, config, len) = Self::open_container(region)?;
    let analyzer = Analyzer::new(config).map_err(|e| {
      Error::invalid(format!(
        "{e}; open this text index with TextIndex::open_with_analyzer"
      ))
    })?;
    Self::from_container(container, analyzer, len)
  }

  /// Opens a text index with a caller-built analyzer, which must have
  /// exactly the settings stored in the file (including the custom
  /// stemmer's name).
  pub fn open_with_analyzer(
    region: Region,
    analyzer: Analyzer,
  ) -> Result<Self> {
    let (container, config, len) = Self::open_container(region)?;
    if *analyzer.config() != config {
      return Err(Error::invalid(format!(
        "analyzer {:?} does not match the index's {config:?}",
        analyzer.config()
      )));
    }
    Self::from_container(container, analyzer, len)
  }

  fn open_container(
    region: Region,
  ) -> Result<(Container, AnalyzerConfig, usize)> {
    let file_len = region.len();
    let c = Container::open_kind(region, KIND, FORMAT_VERSION)?;
    let meta = c.section(META)?;
    let b = meta.bytes();
    if b.len() < META_FIXED {
      return Err(Error::corrupt("text index meta too short"));
    }
    let (config, used) = decode_config(&b[META_FIXED..])?;
    if META_FIXED + used != b.len() {
      return Err(Error::corrupt("trailing bytes in text index meta"));
    }
    Ok((c, config, file_len))
  }

  fn from_container(
    c: Container,
    analyzer: Analyzer,
    file_len: usize,
  ) -> Result<Self> {
    let meta = c.section(META)?;
    let num_docs = u32::from_le(meta.read::<u32>(0)?);
    let doc_count = u64::from_le(meta.read::<u64>(8)?);
    let total_len = u64::from_le(meta.read::<u64>(16)?);
    let num_terms = u64::from_le(meta.read::<u64>(24)?);
    let num_skips = u64::from_le(meta.read::<u64>(32)?);
    let norms = c.section(NORMS)?;
    let lens = c.section(DOCLENS)?;
    let postings = c.section(POSTINGS)?;
    let skips = c.section(SKIPS)?;
    let dict = c.section(DICT)?;
    let dictidx = c.section(DICTIDX)?;
    let bad = |what: &str| {
      Err(Error::corrupt(format!(
        "text index {what} has the wrong size"
      )))
    };
    if norms.len() != num_docs as usize || doc_count > u64::from(num_docs) {
      return bad("norms");
    }
    if lens.len() != num_docs as usize * 4 {
      return bad("doclens");
    }
    lens.cast::<u32>()?;
    if Some(skips.len() as u64) != num_skips.checked_mul(32) {
      return bad("skips");
    }
    skips.cast::<SkipRec>()?;
    let blocks = num_terms.div_ceil(DICT_BLOCK);
    if Some(dictidx.len() as u64) != blocks.checked_mul(24) {
      return bad("dictidx");
    }
    let idx = dictidx.cast::<u64>()?;
    let mut prev = 0u64;
    let (mut prev_post, mut prev_skip) = (0u64, 0u64);
    for b in idx.chunks_exact(3) {
      let (d, post, skip) = (b[0], b[1], b[2]);
      if d < prev
        || d >= dict.len() as u64
        || post < prev_post
        || post > postings.len() as u64
        || skip < prev_skip
        || skip > num_skips
      {
        return bad("dictidx");
      }
      (prev, prev_post, prev_skip) = (d, post, skip);
    }
    Ok(Self {
      container: c,
      analyzer,
      file_len,
      num_docs,
      doc_count,
      total_len,
      num_terms,
      norms,
      lens,
      postings,
      skips,
      dict,
      dictidx,
    })
  }

  /// Reads the whole file and checks its CRC-32C.
  pub fn verify_checksum(&self) -> Result<()> {
    self.container.verify_checksum()
  }

  /// The analyzer restored from the file.
  pub fn analyzer(&self) -> &Analyzer {
    &self.analyzer
  }

  /// Number of documents (gaps in the id space excluded).
  pub fn doc_count(&self) -> u64 {
    self.doc_count
  }

  /// Sum of document lengths in tokens.
  pub fn total_doc_len(&self) -> u64 {
    self.total_len
  }

  /// Number of distinct terms.
  pub fn num_terms(&self) -> u64 {
    self.num_terms
  }

  /// Length in tokens of document `id`, or `None` if absent.
  pub fn doc_len(&self, id: DocId) -> Option<u32> {
    self
      .lens_slice()
      .get(id as usize)
      .copied()
      .filter(|&l| l != ABSENT)
  }

  fn lens_slice(&self) -> &[u32] {
    // Size and alignment were checked at open.
    self.lens.cast::<u32>().unwrap_or(&[])
  }

  fn skip_slice(&self) -> &[SkipRec] {
    self.skips.cast::<SkipRec>().unwrap_or(&[])
  }

  fn idx_slice(&self) -> &[u64] {
    self.dictidx.cast::<u64>().unwrap_or(&[])
  }

  fn block_reader(&self, b: usize) -> BlockReader<'_> {
    let idx = self.idx_slice();
    let dict = self.dict.bytes();
    let start = idx[3 * b] as usize;
    let end = idx
      .get(3 * (b + 1))
      .map_or(dict.len(), |&e| e as usize)
      .max(start);
    BlockReader {
      dict: &dict[..end],
      pos: start,
      left: (self.num_terms - b as u64 * DICT_BLOCK).min(DICT_BLOCK),
      first: true,
      term: Vec::new(),
      post_off: idx[3 * b + 1],
      skip_idx: idx[3 * b + 2],
    }
  }

  fn first_term(&self, b: usize) -> Result<&[u8]> {
    let dict = self.dict.bytes();
    let mut p = self.idx_slice()[3 * b] as usize;
    let len = varint::get(dict, &mut p).ok_or_else(BlockReader::bad)?;
    dict
      .get(p..p.saturating_add(len as usize))
      .ok_or_else(BlockReader::bad)
  }

  fn lookup(&self, term: &[u8]) -> Result<Option<TermMeta>> {
    let blocks = self.idx_slice().len() / 3;
    let (mut lo, mut hi) = (0, blocks);
    while lo < hi {
      let mid = (lo + hi) / 2;
      if self.first_term(mid)? <= term {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if lo == 0 {
      return Ok(None);
    }
    let mut r = self.block_reader(lo - 1);
    while let Some(m) = r.next()? {
      match r.term.as_slice().cmp(term) {
        std::cmp::Ordering::Equal => return Ok(Some(m)),
        std::cmp::Ordering::Greater => return Ok(None),
        std::cmp::Ordering::Less => {}
      }
    }
    Ok(None)
  }

  fn cursor_for(&self, m: &TermMeta) -> Result<Cursor<'_>> {
    let bad = || Error::corrupt("text index term points outside postings");
    let data = usize::try_from(m.data_off)
      .ok()
      .and_then(|o| self.postings.bytes().get(o..))
      .ok_or_else(bad)?;
    let num_docs = self.num_docs as usize;
    let c = match m.single {
      Some((last, imp)) => Cursor::single(data, m.df, last, imp, num_docs),
      None => {
        let nb = (m.df as usize).div_ceil(BLOCK);
        let s = usize::try_from(m.skip_idx).map_err(|_| bad())?;
        let recs = self
          .skip_slice()
          .get(s..s.saturating_add(nb))
          .ok_or_else(bad)?;
        let mut c = Cursor::empty(data, recs, m.df, num_docs);
        c.start();
        c
      }
    };
    Ok(c)
  }

  /// Document frequency of an analysed term.
  pub fn doc_freq(&self, term: &str) -> Result<u32> {
    Ok(self.lookup(term.as_bytes())?.map_or(0, |m| m.df))
  }

  /// Size of the file in bytes.
  pub fn size_bytes(&self) -> usize {
    self.file_len
  }

  /// BM25 search without pruning: scores every matching document. Same
  /// results as [`TextSearcher::search`]; a baseline for tests and
  /// benchmarks.
  pub fn search_exhaustive(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
    stats: Option<&CorpusStats>,
  ) -> Result<Vec<Hit>> {
    search::search(self, query, k, filter, stats, false)
  }

  /// Merges frozen indexes into one, for compaction, in one streaming
  /// pass over their dictionaries.
  ///
  /// `inputs[i].doc_map[old]` is the output id of input `i`'s document
  /// `old`, or [`DELETED`] to drop it. Output ids must be unique and
  /// below `num_docs`; they need not preserve order. All inputs must use
  /// the same analyzer settings. Terms whose documents are all deleted
  /// disappear. The result is byte-identical to indexing the surviving
  /// documents under their new ids from scratch.
  pub fn merge<W: Write>(
    inputs: &[MergeInput<'_>],
    num_docs: usize,
    out: W,
  ) -> Result<W> {
    let first = inputs
      .first()
      .ok_or_else(|| Error::invalid("merge needs at least one input"))?;
    let config = first.index.analyzer.config();
    if inputs.iter().any(|i| i.index.analyzer.config() != config) {
      return Err(Error::invalid("merge inputs use different analyzers"));
    }
    if num_docs >= END as usize {
      return Err(Error::invalid("num_docs must be below u32::MAX"));
    }
    let mut lens = vec![ABSENT; num_docs];
    let (mut doc_count, mut total_len) = (0u64, 0u64);
    for inp in inputs {
      if inp.doc_map.len() != inp.index.num_docs as usize {
        return Err(Error::invalid(format!(
          "doc_map has {} entries for an index of {} ids",
          inp.doc_map.len(),
          inp.index.num_docs
        )));
      }
      for (&new, &len) in inp.doc_map.iter().zip(inp.index.lens_slice()) {
        if new == DELETED || len == ABSENT {
          continue;
        }
        let slot = lens.get_mut(new as usize).ok_or_else(|| {
          Error::invalid(format!("mapped id {new} is not below {num_docs}"))
        })?;
        if *slot != ABSENT {
          return Err(Error::invalid(format!("id {new} is mapped twice")));
        }
        *slot = len;
        doc_count += 1;
        total_len += u64::from(len);
      }
    }
    let norms: Vec<u8> = lens
      .iter()
      .map(|&l| if l == ABSENT { 0 } else { encode_doc_len(l) })
      .collect();
    let mut w = Writer::new(out, config, &norms, &lens, doc_count, total_len)?;
    let mut iters: Vec<DictIter<'_>> =
      inputs.iter().map(|i| DictIter::new(i.index)).collect();
    let mut heads: Vec<Option<TermMeta>> = Vec::with_capacity(iters.len());
    for it in &mut iters {
      heads.push(it.next()?);
    }
    let mut postings: Vec<(DocId, u32)> = Vec::new();
    let mut term: Vec<u8> = Vec::new();
    loop {
      let mut min: Option<&[u8]> = None;
      for (it, h) in iters.iter().zip(&heads) {
        if h.is_some() && min.is_none_or(|m| it.term() < m) {
          min = Some(it.term());
        }
      }
      let Some(m) = min else { break };
      term.clear();
      term.extend_from_slice(m);
      postings.clear();
      for (i, inp) in inputs.iter().enumerate() {
        let Some(meta) = heads[i] else { continue };
        if iters[i].term() != term.as_slice() {
          continue;
        }
        let mut c = inp.index.cursor_for(&meta)?;
        while c.doc() != END {
          let new = inp.doc_map[c.doc() as usize];
          let tf = c.tf();
          if new != DELETED {
            postings.push((new, tf));
          }
          c.next();
        }
        if c.corrupt {
          return Err(Error::corrupt("text index postings are corrupt"));
        }
        heads[i] = iters[i].next()?;
      }
      if postings.is_empty() {
        continue;
      }
      if !postings.is_sorted_by_key(|p| p.0) {
        postings.sort_unstable_by_key(|p| p.0);
      }
      let mut list = PostingList::default();
      let mut prev: Option<DocId> = None;
      for &(doc, tf) in &postings {
        if prev.is_some_and(|p| doc <= p)
          || tf == 0
          || lens.get(doc as usize).is_none_or(|&l| l == ABSENT)
        {
          return Err(Error::corrupt(
            "merge input postings disagree with its document lengths",
          ));
        }
        prev = Some(doc);
        list.push(doc, tf, &norms);
      }
      w.add_term(&term, &list.freeze(&norms))?;
    }
    w.finish()
  }
}

/// One input of [`TextIndex::merge`].
#[derive(Clone, Copy, Debug)]
pub struct MergeInput<'a> {
  /// The index to merge.
  pub index: &'a TextIndex,
  /// Output id per input id, or [`DELETED`]. Length must equal the
  /// input's `num_docs()`.
  pub doc_map: &'a [DocId],
}

/// Iterates a whole dictionary in term order.
struct DictIter<'a> {
  index: &'a TextIndex,
  block: usize,
  reader: Option<BlockReader<'a>>,
}

impl<'a> DictIter<'a> {
  fn new(index: &'a TextIndex) -> Self {
    Self {
      index,
      block: 0,
      reader: None,
    }
  }

  fn term(&self) -> &[u8] {
    self.reader.as_ref().map_or(&[], |r| r.term.as_slice())
  }

  fn next(&mut self) -> Result<Option<TermMeta>> {
    loop {
      if let Some(r) = &mut self.reader
        && let Some(m) = r.next()?
      {
        return Ok(Some(m));
      }
      if self.block >= self.index.idx_slice().len() / 3 {
        self.reader = None;
        return Ok(None);
      }
      let mut r = self.index.block_reader(self.block);
      self.block += 1;
      if let Some(m) = r.next()? {
        self.reader = Some(r);
        return Ok(Some(m));
      }
    }
  }
}

impl Source for TextIndex {
  fn analyzer(&self) -> &Analyzer {
    &self.analyzer
  }

  fn doc_count(&self) -> u64 {
    self.doc_count
  }

  fn total_doc_len(&self) -> u64 {
    self.total_len
  }

  fn norms(&self) -> &[u8] {
    self.norms.bytes()
  }

  fn doc_freq(&self, term: &str) -> Result<u32> {
    TextIndex::doc_freq(self, term)
  }

  fn cursor(&self, term: &str) -> Result<Option<Cursor<'_>>> {
    match self.lookup(term.as_bytes())? {
      Some(m) => self.cursor_for(&m).map(Some),
      None => Ok(None),
    }
  }
}

impl TextSearcher for TextIndex {
  fn num_docs(&self) -> usize {
    self.num_docs as usize
  }

  fn search(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
  ) -> Result<Vec<Hit>> {
    search::search(self, query, k, filter, None, true)
  }

  fn stats_for(&self, query: &str) -> Result<CorpusStats> {
    search::stats_for(self, query)
  }

  fn search_with_stats(
    &self,
    query: &str,
    k: usize,
    filter: Option<&dyn DocFilter>,
    stats: &CorpusStats,
  ) -> Result<Vec<Hit>> {
    search::search(self, query, k, filter, Some(stats), true)
  }
}
