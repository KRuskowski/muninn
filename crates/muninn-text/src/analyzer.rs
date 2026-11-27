//! Text analysis: tokenising and normalising text into index terms.
//!
//! The pipeline, applied identically to documents and queries:
//!
//! 1. **Tokenise.** A token is a maximal run of alphanumeric characters
//!    (Unicode `Alphabetic` or `Numeric`). Combining marks (U+0300..U+036F
//!    and the other combining-mark blocks) stay inside the token they
//!    follow, so decomposed text such as `e` + U+0301 is not split.
//!    Everything else separates tokens. Scripts written without spaces
//!    (Chinese, Japanese, Thai) are not segmented into words.
//! 2. **Lowercase** with full Unicode case mapping; final sigma `ς` is
//!    mapped to `σ` so case variants of Greek words agree.
//! 3. **Fold diacritics** (optional): Latin letters with diacritics map to
//!    their base letters (`é` -> `e`, `ß` -> `ss`, `æ` -> `ae`, `ł` -> `l`)
//!    and combining marks are dropped.
//! 4. **Drop long tokens**: tokens longer than `max_token_bytes` (after
//!    normalisation) are skipped, typically hashes or base64.
//! 5. **Stop words** (optional): the English stop list Lucene uses.
//! 6. **Stem** (optional): the built-in Porter stemmer for English, or a
//!    custom [`Stemmer`]. The Porter stemmer only touches tokens made of
//!    ASCII letters.
//!
//! [`AnalyzerConfig`] is stored in every index file, and opening a file
//! restores the analyzer, so queries are always analysed like the text.

use std::fmt;
use std::sync::Arc;

use muninn_core::{Error, Result};

use crate::porter;

/// Which stemmer an analyzer applies.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum StemmerChoice {
  /// No stemming.
  #[default]
  None,
  /// The built-in Porter (1980) English stemmer.
  PorterEnglish,
  /// A caller-supplied [`Stemmer`], identified by its name. Files that
  /// use one must be opened with [`crate::TextIndex::open_with_analyzer`].
  Custom(String),
}

/// Analysis settings. Stored in the index file.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnalyzerConfig {
  /// Fold Latin diacritics to base letters and drop combining marks.
  pub fold_diacritics: bool,
  /// Remove English stop words ("the", "and", ...).
  pub english_stop_words: bool,
  /// Stemmer to apply after stop word removal.
  pub stemmer: StemmerChoice,
  /// Tokens longer than this many bytes (UTF-8, after normalisation) are
  /// skipped. Must be at least 1.
  pub max_token_bytes: u16,
}

impl Default for AnalyzerConfig {
  /// Lowercasing only: no folding, no stop words, no stemming.
  fn default() -> Self {
    Self {
      fold_diacritics: false,
      english_stop_words: false,
      stemmer: StemmerChoice::None,
      max_token_bytes: 255,
    }
  }
}

impl AnalyzerConfig {
  /// English: diacritic folding, stop words and Porter stemming.
  pub fn english() -> Self {
    Self {
      fold_diacritics: true,
      english_stop_words: true,
      stemmer: StemmerChoice::PorterEnglish,
      max_token_bytes: 255,
    }
  }
}

/// A stemming hook: reduces a normalised token to its stem.
pub trait Stemmer: Send + Sync {
  /// Stable name recorded in index files. Opening a file checks that the
  /// supplied stemmer has the same name.
  fn name(&self) -> &str;

  /// Stems `word` in place. Leaving it empty drops the token.
  fn stem(&self, word: &mut String);
}

/// The built-in Porter English stemmer as a [`Stemmer`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PorterStemmer;

impl Stemmer for PorterStemmer {
  fn name(&self) -> &str {
    "porter-english"
  }

  fn stem(&self, word: &mut String) {
    if word.bytes().all(|b| b.is_ascii_lowercase()) {
      let mut bytes = std::mem::take(word).into_bytes();
      porter::stem(&mut bytes);
      // Porter only ever writes ASCII lowercase letters.
      *word = String::from_utf8(bytes).expect("porter output is ASCII");
    }
  }
}

/// Reusable token buffer filled by [`Analyzer::analyze`].
///
/// All tokens share one string, so analysing a document allocates only
/// when the buffer grows.
#[derive(Clone, Debug, Default)]
pub struct Tokens {
  text: String,
  ends: Vec<usize>,
  scratch: String,
}

impl Tokens {
  /// An empty buffer.
  pub fn new() -> Self {
    Self::default()
  }

  /// Number of tokens.
  #[inline]
  pub fn len(&self) -> usize {
    self.ends.len()
  }

  /// Whether there are no tokens.
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.ends.is_empty()
  }

  /// Token `i`.
  #[inline]
  pub fn get(&self, i: usize) -> &str {
    let start = if i == 0 { 0 } else { self.ends[i - 1] };
    &self.text[start..self.ends[i]]
  }

  /// The tokens in text order.
  pub fn iter(&self) -> impl Iterator<Item = &str> + '_ {
    (0..self.len()).map(move |i| self.get(i))
  }

  /// Removes all tokens, keeping the allocation.
  pub fn clear(&mut self) {
    self.text.clear();
    self.ends.clear();
  }
}

/// Turns text into index terms. Cheap to clone. The default analyzer
/// only lowercases ([`AnalyzerConfig::default`]).
#[derive(Clone, Default)]
pub struct Analyzer {
  config: AnalyzerConfig,
  stemmer: Option<Arc<dyn Stemmer>>,
}

impl fmt::Debug for Analyzer {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Analyzer")
      .field("config", &self.config)
      .finish()
  }
}

impl Analyzer {
  /// An analyzer using only built-in components. Fails for
  /// [`StemmerChoice::Custom`] (use [`Analyzer::with_stemmer`]) and for a
  /// zero `max_token_bytes`.
  pub fn new(config: AnalyzerConfig) -> Result<Self> {
    if config.max_token_bytes == 0 {
      return Err(Error::invalid("max_token_bytes must be at least 1"));
    }
    let stemmer: Option<Arc<dyn Stemmer>> = match &config.stemmer {
      StemmerChoice::None => None,
      StemmerChoice::PorterEnglish => Some(Arc::new(PorterStemmer)),
      StemmerChoice::Custom(name) => {
        return Err(Error::invalid(format!(
          "custom stemmer {name:?} must be supplied with \
           Analyzer::with_stemmer"
        )));
      }
    };
    Ok(Self { config, stemmer })
  }

  /// An analyzer with a custom stemmer. `config.stemmer` is replaced by
  /// `StemmerChoice::Custom(stemmer.name())`.
  pub fn with_stemmer(
    mut config: AnalyzerConfig,
    stemmer: Arc<dyn Stemmer>,
  ) -> Result<Self> {
    let name = stemmer.name();
    if name.is_empty() || name.len() > u16::MAX as usize {
      return Err(Error::invalid("stemmer name must be 1..=65535 bytes"));
    }
    config.stemmer = StemmerChoice::Custom(name.to_string());
    let mut a = Self::new(AnalyzerConfig {
      stemmer: StemmerChoice::None,
      ..config.clone()
    })?;
    a.config = config;
    a.stemmer = Some(stemmer);
    Ok(a)
  }

  /// The settings this analyzer was built from.
  pub fn config(&self) -> &AnalyzerConfig {
    &self.config
  }

  /// Analyses `text` into `out`, replacing its previous contents.
  pub fn analyze(&self, text: &str, out: &mut Tokens) {
    out.clear();
    let bytes = text.as_bytes();
    let fold = self.config.fold_diacritics;
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
      let b = bytes[i];
      if b < 0x80 {
        i += 1;
        if b.is_ascii_alphanumeric() {
          out.text.push(b.to_ascii_lowercase() as char);
        } else {
          self.finish(out, start);
          start = out.text.len();
        }
        continue;
      }
      let c = text[i..].chars().next().expect("index at char boundary");
      i += c.len_utf8();
      if is_combining_mark(c) {
        if !fold && out.text.len() > start {
          out.text.push(c);
        }
      } else if c.is_alphanumeric() {
        for lc in c.to_lowercase() {
          push_normalised(&mut out.text, lc, fold, start);
        }
      } else {
        self.finish(out, start);
        start = out.text.len();
      }
    }
    self.finish(out, start);
  }

  /// Ends the token at `out.text[start..]`: applies the length limit,
  /// stop words and stemming, then records it or discards it.
  fn finish(&self, out: &mut Tokens, start: usize) {
    let len = out.text.len() - start;
    if len == 0 {
      return;
    }
    let max = usize::from(self.config.max_token_bytes);
    if len > max
      || (self.config.english_stop_words
        && is_english_stop_word(&out.text[start..]))
    {
      out.text.truncate(start);
      return;
    }
    if let Some(stemmer) = &self.stemmer {
      out.scratch.clear();
      out.scratch.push_str(&out.text[start..]);
      stemmer.stem(&mut out.scratch);
      out.text.truncate(start);
      if out.scratch.is_empty() || out.scratch.len() > max {
        return;
      }
      out.text.push_str(&out.scratch);
    }
    out.ends.push(out.text.len());
  }
}

/// Appends a lowercased character, applying sigma and diacritic folding.
#[inline]
fn push_normalised(text: &mut String, c: char, fold: bool, start: usize) {
  let c = if c == 'ς' { 'σ' } else { c };
  if is_combining_mark(c) {
    // `to_lowercase` can emit marks (U+0130 -> "i\u{307}").
    if !fold && text.len() > start {
      text.push(c);
    }
    return;
  }
  if fold {
    if let Some(s) = fold_latin(c) {
      text.push_str(s);
      return;
    }
  }
  text.push(c);
}

/// Combining diacritical mark blocks.
#[inline]
fn is_combining_mark(c: char) -> bool {
  matches!(
    c,
    '\u{0300}'..='\u{036F}'
      | '\u{1AB0}'..='\u{1AFF}'
      | '\u{1DC0}'..='\u{1DFF}'
      | '\u{20D0}'..='\u{20FF}'
      | '\u{FE20}'..='\u{FE2F}'
  )
}

/// ASCII replacement for a lowercase Latin letter with a diacritic.
fn fold_latin(c: char) -> Option<&'static str> {
  Some(match c {
    'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' | 'ǎ' | 'ǻ' | 'ȁ'
    | 'ȃ' | 'ạ' | 'ả' | 'ấ' | 'ầ' | 'ẩ' | 'ẫ' | 'ậ' | 'ắ' | 'ằ' | 'ẳ' | 'ẵ'
    | 'ặ' => "a",
    'æ' | 'ǽ' => "ae",
    'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => "c",
    'ď' | 'đ' | 'ð' => "d",
    'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' | 'ȅ' | 'ȇ' | 'ẹ'
    | 'ẻ' | 'ẽ' | 'ế' | 'ề' | 'ể' | 'ễ' | 'ệ' => "e",
    'ĝ' | 'ğ' | 'ġ' | 'ģ' | 'ǧ' => "g",
    'ĥ' | 'ħ' => "h",
    'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' | 'ǐ' | 'ȉ' | 'ȋ'
    | 'ỉ' | 'ị' => "i",
    'ĳ' => "ij",
    'ĵ' => "j",
    'ķ' | 'ǩ' => "k",
    'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => "l",
    'ñ' | 'ń' | 'ņ' | 'ň' | 'ŉ' | 'ǹ' => "n",
    'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' | 'ơ' | 'ǒ' | 'ǿ'
    | 'ȍ' | 'ȏ' | 'ọ' | 'ỏ' | 'ố' | 'ồ' | 'ổ' | 'ỗ' | 'ộ' | 'ớ' | 'ờ' | 'ở'
    | 'ỡ' | 'ợ' => "o",
    'œ' => "oe",
    'ŕ' | 'ŗ' | 'ř' | 'ȑ' | 'ȓ' => "r",
    'ś' | 'ŝ' | 'ş' | 'š' | 'ș' | 'ſ' => "s",
    'ß' => "ss",
    'ţ' | 'ť' | 'ŧ' | 'ț' => "t",
    'þ' => "th",
    'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' | 'ư' | 'ǔ'
    | 'ǖ' | 'ǘ' | 'ǚ' | 'ǜ' | 'ȕ' | 'ȗ' | 'ụ' | 'ủ' | 'ứ' | 'ừ' | 'ử' | 'ữ'
    | 'ự' => "u",
    'ŵ' => "w",
    'ý' | 'ÿ' | 'ŷ' | 'ỳ' | 'ỵ' | 'ỷ' | 'ỹ' => "y",
    'ź' | 'ż' | 'ž' => "z",
    _ => return None,
  })
}

/// Lucene's English stop word list.
fn is_english_stop_word(t: &str) -> bool {
  matches!(
    t,
    "a"
      | "an"
      | "and"
      | "are"
      | "as"
      | "at"
      | "be"
      | "but"
      | "by"
      | "for"
      | "if"
      | "in"
      | "into"
      | "is"
      | "it"
      | "no"
      | "not"
      | "of"
      | "on"
      | "or"
      | "such"
      | "that"
      | "the"
      | "their"
      | "then"
      | "there"
      | "these"
      | "they"
      | "this"
      | "to"
      | "was"
      | "will"
      | "with"
  )
}

/// Analyses a query: its distinct terms in order of first occurrence.
pub(crate) fn query_terms(analyzer: &Analyzer, query: &str) -> Vec<String> {
  let mut toks = Tokens::new();
  analyzer.analyze(query, &mut toks);
  let mut out: Vec<String> = Vec::with_capacity(toks.len());
  for t in toks.iter() {
    if !out.iter().any(|o| o == t) {
      out.push(t.to_string());
    }
  }
  out
}

/// Serialised form of an [`AnalyzerConfig`], stored in index files.
pub(crate) fn encode_config(c: &AnalyzerConfig, out: &mut Vec<u8>) {
  let (kind, name): (u8, &str) = match &c.stemmer {
    StemmerChoice::None => (0, ""),
    StemmerChoice::PorterEnglish => (1, ""),
    StemmerChoice::Custom(n) => (2, n.as_str()),
  };
  out.push(u8::from(c.fold_diacritics));
  out.push(u8::from(c.english_stop_words));
  out.push(kind);
  out.push(0);
  out.extend_from_slice(&c.max_token_bytes.to_le_bytes());
  out.extend_from_slice(&(name.len() as u16).to_le_bytes());
  out.extend_from_slice(name.as_bytes());
}

/// Inverse of [`encode_config`]; returns the config and bytes consumed.
pub(crate) fn decode_config(b: &[u8]) -> Result<(AnalyzerConfig, usize)> {
  let bad = || Error::corrupt("bad analyzer settings in text index");
  if b.len() < 8 {
    return Err(bad());
  }
  let flag = |x: u8| match x {
    0 => Ok(false),
    1 => Ok(true),
    _ => Err(bad()),
  };
  let fold_diacritics = flag(b[0])?;
  let english_stop_words = flag(b[1])?;
  let max_token_bytes = u16::from_le_bytes([b[4], b[5]]);
  let name_len = usize::from(u16::from_le_bytes([b[6], b[7]]));
  let name = b.get(8..8 + name_len).ok_or_else(bad)?;
  let stemmer = match (b[2], name_len) {
    (0, 0) => StemmerChoice::None,
    (1, 0) => StemmerChoice::PorterEnglish,
    (2, 1..) => StemmerChoice::Custom(
      String::from_utf8(name.to_vec()).map_err(|_| bad())?,
    ),
    _ => return Err(bad()),
  };
  if max_token_bytes == 0 {
    return Err(bad());
  }
  Ok((
    AnalyzerConfig {
      fold_diacritics,
      english_stop_words,
      stemmer,
      max_token_bytes,
    },
    8 + name_len,
  ))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn toks(a: &Analyzer, text: &str) -> Vec<String> {
    let mut t = Tokens::new();
    a.analyze(text, &mut t);
    t.iter().map(str::to_string).collect()
  }

  #[test]
  fn splits_and_lowercases() {
    let a = Analyzer::default();
    assert_eq!(
      toks(&a, "Hello, World! It's 2024-01-02."),
      ["hello", "world", "it", "s", "2024", "01", "02"]
    );
    assert!(toks(&a, "").is_empty());
    assert!(toks(&a, "  ,.;  ").is_empty());
  }

  #[test]
  fn reuses_buffer_without_leaking_previous_tokens() {
    let a = Analyzer::default();
    let mut t = Tokens::new();
    a.analyze("one two three", &mut t);
    a.analyze("four", &mut t);
    assert_eq!(t.iter().collect::<Vec<_>>(), ["four"]);
    assert_eq!(t.len(), 1);
  }

  #[test]
  fn unicode_lowercase_and_scripts() {
    let a = Analyzer::default();
    assert_eq!(toks(&a, "ÉCOLE Straße"), ["école", "straße"]);
    assert_eq!(toks(&a, "ΟΔΟΣ οδος"), ["οδοσ", "οδοσ"]);
    assert_eq!(toks(&a, "Привет, мир"), ["привет", "мир"]);
    // CJK runs stay one token (no word segmentation).
    assert_eq!(toks(&a, "東京タワー 2"), ["東京タワー", "2"]);
    // Non-alphanumeric symbols separate tokens.
    assert_eq!(toks(&a, "a€b😀c"), ["a", "b", "c"]);
  }

  #[test]
  fn combining_marks_stay_in_token() {
    let plain = Analyzer::default();
    let decomposed = "re\u{301}sume\u{301}";
    assert_eq!(toks(&plain, decomposed), [decomposed]);
    let fold = Analyzer::new(AnalyzerConfig {
      fold_diacritics: true,
      ..Default::default()
    })
    .unwrap();
    assert_eq!(toks(&fold, decomposed), ["resume"]);
    assert_eq!(toks(&fold, "Résumé"), ["resume"]);
    assert_eq!(toks(&fold, "İstanbul"), ["istanbul"]);
  }

  #[test]
  fn folding() {
    let a = Analyzer::new(AnalyzerConfig {
      fold_diacritics: true,
      ..Default::default()
    })
    .unwrap();
    assert_eq!(
      toks(&a, "Crème Brûlée Straße Æsir Łódź naïve Ørsted Œuvre"),
      [
        "creme", "brulee", "strasse", "aesir", "lodz", "naive", "orsted",
        "oeuvre"
      ]
    );
    // Non-Latin scripts are untouched.
    assert_eq!(toks(&a, "Ελλάδα"), ["ελλάδα"]);
  }

  #[test]
  fn stop_words_and_stemming() {
    let a = Analyzer::new(AnalyzerConfig::english()).unwrap();
    assert_eq!(
      toks(&a, "The connections are running into the Cafés"),
      ["connect", "run", "cafe"]
    );
    // Non-ASCII tokens pass through the Porter stemmer untouched.
    assert_eq!(toks(&a, "печеньки"), ["печеньки"]);
  }

  #[test]
  fn long_tokens_are_dropped() {
    let a = Analyzer::new(AnalyzerConfig {
      max_token_bytes: 4,
      ..Default::default()
    })
    .unwrap();
    assert_eq!(toks(&a, "abcd abcde ab éé ééé"), ["abcd", "ab", "éé"]);
    assert!(
      Analyzer::new(AnalyzerConfig {
        max_token_bytes: 0,
        ..Default::default()
      })
      .is_err()
    );
  }

  struct Truncate3;
  impl Stemmer for Truncate3 {
    fn name(&self) -> &str {
      "truncate3"
    }
    fn stem(&self, word: &mut String) {
      if let Some((i, _)) = word.char_indices().nth(3) {
        word.truncate(i);
      }
      if word == "dro" {
        word.clear();
      }
    }
  }

  #[test]
  fn custom_stemmer_hook() {
    let a =
      Analyzer::with_stemmer(AnalyzerConfig::default(), Arc::new(Truncate3))
        .unwrap();
    assert_eq!(
      a.config().stemmer,
      StemmerChoice::Custom("truncate3".into())
    );
    assert_eq!(toks(&a, "Searching drop über"), ["sea", "übe"]);
    assert!(Analyzer::new(a.config().clone()).is_err());
  }

  #[test]
  fn config_round_trips() {
    let configs = [
      AnalyzerConfig::default(),
      AnalyzerConfig::english(),
      AnalyzerConfig {
        stemmer: StemmerChoice::Custom("x-stem".into()),
        max_token_bytes: 7,
        ..Default::default()
      },
    ];
    for c in configs {
      let mut b = Vec::new();
      encode_config(&c, &mut b);
      let (d, used) = decode_config(&b).unwrap();
      assert_eq!(d, c);
      assert_eq!(used, b.len());
      assert!(decode_config(&b[..b.len() - 1]).is_err());
    }
    assert!(decode_config(&[2, 0, 0, 0, 1, 0, 0, 0]).is_err());
  }

  #[test]
  fn query_terms_dedupe_in_order() {
    let a = Analyzer::default();
    assert_eq!(query_terms(&a, "b a B c a"), ["b", "a", "c"]);
  }
}
