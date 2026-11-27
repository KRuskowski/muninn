//! The Porter (1980) English stemmer.
//!
//! A direct port of Martin Porter's reference ANSI C implementation
//! (tartarus.org/martin/PorterStemmer), including its documented
//! departures from the paper (`logi` -> `log`, `bli` -> `ble`). Works on
//! lowercase ASCII letters; callers pass nothing else.

struct Stem<'a> {
  b: &'a mut Vec<u8>,
  /// Index of the last character of the current word.
  k: isize,
  /// End of the stem before a matched suffix (may be -1).
  j: isize,
}

impl Stem<'_> {
  #[inline]
  fn at(&self, i: isize) -> u8 {
    self.b[i as usize]
  }

  /// Whether `b[i]` is a consonant.
  ///
  /// A `y` is a consonant at the start or after a vowel, and a vowel
  /// after a consonant, so in a run of `y`s it alternates from the
  /// character before the run. Computed without recursion.
  fn cons(&self, i: isize) -> bool {
    let mut j = i;
    while j > 0 && self.at(j) == b'y' {
      j -= 1;
    }
    let base = match self.at(j) {
      b'a' | b'e' | b'i' | b'o' | b'u' => false,
      _ => true, // Includes a `y` at position 0.
    };
    base ^ ((i - j) % 2 == 1)
  }

  /// The measure `m` of `b[0..=j]`: the number of VC sequences.
  fn m(&self) -> u32 {
    let mut n = 0;
    let mut i = 0isize;
    loop {
      if i > self.j {
        return n;
      }
      if !self.cons(i) {
        break;
      }
      i += 1;
    }
    i += 1;
    loop {
      loop {
        if i > self.j {
          return n;
        }
        if self.cons(i) {
          break;
        }
        i += 1;
      }
      i += 1;
      n += 1;
      loop {
        if i > self.j {
          return n;
        }
        if !self.cons(i) {
          break;
        }
        i += 1;
      }
      i += 1;
    }
  }

  /// Whether `b[0..=j]` contains a vowel.
  fn vowel_in_stem(&self) -> bool {
    (0..=self.j).any(|i| !self.cons(i))
  }

  /// Whether `b[j-1..=j]` is a double consonant.
  fn doublec(&self, j: isize) -> bool {
    j >= 1 && self.at(j) == self.at(j - 1) && self.cons(j)
  }

  /// Whether `b[i-2..=i]` is consonant-vowel-consonant and the last
  /// consonant is not w, x or y.
  fn cvc(&self, i: isize) -> bool {
    if i < 2 || !self.cons(i) || self.cons(i - 1) || !self.cons(i - 2) {
      return false;
    }
    !matches!(self.at(i), b'w' | b'x' | b'y')
  }

  /// Whether the word ends with `s`; sets `j` to the end of the stem.
  fn ends(&mut self, s: &[u8]) -> bool {
    let len = s.len() as isize;
    if s[s.len() - 1] != self.at(self.k) || len > self.k + 1 {
      return false;
    }
    let start = (self.k - len + 1) as usize;
    if &self.b[start..=self.k as usize] != s {
      return false;
    }
    self.j = self.k - len;
    true
  }

  /// Replaces `b[j+1..=k]` with `s`.
  fn set_to(&mut self, s: &[u8]) {
    let start = (self.j + 1) as usize;
    self.b.truncate(start);
    self.b.extend_from_slice(s);
    self.k = self.j + s.len() as isize;
  }

  fn r(&mut self, s: &[u8]) {
    if self.m() > 0 {
      self.set_to(s);
    }
  }

  fn truncate_to_k(&mut self) {
    self.b.truncate((self.k + 1) as usize);
  }

  fn step1ab(&mut self) {
    if self.at(self.k) == b's' {
      if self.ends(b"sses") {
        self.k -= 2;
      } else if self.ends(b"ies") {
        self.set_to(b"i");
      } else if self.at(self.k - 1) != b's' {
        self.k -= 1;
      }
      self.truncate_to_k();
    }
    if self.ends(b"eed") {
      if self.m() > 0 {
        self.k -= 1;
        self.truncate_to_k();
      }
    } else if (self.ends(b"ed") || self.ends(b"ing")) && self.vowel_in_stem() {
      self.k = self.j;
      self.truncate_to_k();
      if self.ends(b"at") {
        self.set_to(b"ate");
      } else if self.ends(b"bl") {
        self.set_to(b"ble");
      } else if self.ends(b"iz") {
        self.set_to(b"ize");
      } else if self.doublec(self.k) {
        self.k -= 1;
        if matches!(self.at(self.k), b'l' | b's' | b'z') {
          self.k += 1;
        }
        self.truncate_to_k();
      } else if {
        self.j = self.k;
        self.m() == 1
      } && self.cvc(self.k)
      {
        self.set_to(b"e");
      }
    }
  }

  fn step1c(&mut self) {
    if self.ends(b"y") && self.vowel_in_stem() {
      let k = self.k as usize;
      self.b[k] = b'i';
    }
  }

  /// Tries each `(suffix, replacement)` in order; the first suffix that
  /// matches ends the step, whether or not the replacement applies.
  fn try_rules(&mut self, rules: &[(&[u8], &[u8])]) {
    for (suffix, repl) in rules {
      if self.ends(suffix) {
        self.r(repl);
        return;
      }
    }
  }

  fn step2(&mut self) {
    let rules: &[(&[u8], &[u8])] = match self.at(self.k - 1) {
      b'a' => &[(b"ational", b"ate"), (b"tional", b"tion")],
      b'c' => &[(b"enci", b"ence"), (b"anci", b"ance")],
      b'e' => &[(b"izer", b"ize")],
      b'l' => &[
        (b"bli", b"ble"),
        (b"alli", b"al"),
        (b"entli", b"ent"),
        (b"eli", b"e"),
        (b"ousli", b"ous"),
      ],
      b'o' => &[(b"ization", b"ize"), (b"ation", b"ate"), (b"ator", b"ate")],
      b's' => &[
        (b"alism", b"al"),
        (b"iveness", b"ive"),
        (b"fulness", b"ful"),
        (b"ousness", b"ous"),
      ],
      b't' => &[(b"aliti", b"al"), (b"iviti", b"ive"), (b"biliti", b"ble")],
      b'g' => &[(b"logi", b"log")],
      _ => &[],
    };
    self.try_rules(rules);
  }

  fn step3(&mut self) {
    let rules: &[(&[u8], &[u8])] = match self.at(self.k) {
      b'e' => &[(b"icate", b"ic"), (b"ative", b""), (b"alize", b"al")],
      b'i' => &[(b"iciti", b"ic")],
      b'l' => &[(b"ical", b"ic"), (b"ful", b"")],
      b's' => &[(b"ness", b"")],
      _ => &[],
    };
    self.try_rules(rules);
  }

  fn step4(&mut self) {
    let found = match self.at(self.k - 1) {
      b'a' => self.ends(b"al"),
      b'c' => self.ends(b"ance") || self.ends(b"ence"),
      b'e' => self.ends(b"er"),
      b'i' => self.ends(b"ic"),
      b'l' => self.ends(b"able") || self.ends(b"ible"),
      b'n' => {
        self.ends(b"ant")
          || self.ends(b"ement")
          || self.ends(b"ment")
          || self.ends(b"ent")
      }
      b'o' => {
        (self.ends(b"ion")
          && self.j >= 0
          && matches!(self.at(self.j), b's' | b't'))
          || self.ends(b"ou")
      }
      b's' => self.ends(b"ism"),
      b't' => self.ends(b"ate") || self.ends(b"iti"),
      b'u' => self.ends(b"ous"),
      b'v' => self.ends(b"ive"),
      b'z' => self.ends(b"ize"),
      _ => false,
    };
    if found && self.m() > 1 {
      self.k = self.j;
      self.truncate_to_k();
    }
  }

  fn step5(&mut self) {
    self.j = self.k;
    if self.at(self.k) == b'e' {
      let a = self.m();
      if a > 1 || (a == 1 && !self.cvc(self.k - 1)) {
        self.k -= 1;
        self.truncate_to_k();
      }
    }
    if self.at(self.k) == b'l' && self.doublec(self.k) && {
      self.j = self.k;
      self.m() > 1
    } {
      self.k -= 1;
      self.truncate_to_k();
    }
  }
}

/// Longer words are left unstemmed: the algorithm is quadratic in the
/// length of runs of `y`, and no English word comes close.
pub(crate) const MAX_STEM_BYTES: usize = 255;

/// Stems `word` in place. `word` must consist of lowercase ASCII letters;
/// words of one or two letters, or over [`MAX_STEM_BYTES`], are left
/// alone.
pub(crate) fn stem(word: &mut Vec<u8>) {
  debug_assert!(word.iter().all(u8::is_ascii_lowercase));
  if word.len() <= 2 || word.len() > MAX_STEM_BYTES {
    return;
  }
  let k = word.len() as isize - 1;
  let mut z = Stem { b: word, k, j: 0 };
  z.step1ab();
  if z.k > 0 {
    z.step1c();
    z.step2();
    z.step3();
    z.step4();
    z.step5();
  }
  z.truncate_to_k();
}

#[cfg(test)]
mod tests {
  use super::*;

  fn s(w: &str) -> String {
    let mut v = w.as_bytes().to_vec();
    stem(&mut v);
    String::from_utf8(v).unwrap()
  }

  #[test]
  fn porter_paper_examples() {
    // Examples from Porter (1980), "An algorithm for suffix stripping",
    // and the reference vocabulary.
    let cases = [
      ("caresses", "caress"),
      ("ponies", "poni"),
      ("ties", "ti"),
      ("caress", "caress"),
      ("cats", "cat"),
      ("feed", "feed"),
      ("agreed", "agre"),
      ("plastered", "plaster"),
      ("bled", "bled"),
      ("motoring", "motor"),
      ("sing", "sing"),
      ("conflated", "conflat"),
      ("troubled", "troubl"),
      ("sized", "size"),
      ("hopping", "hop"),
      ("tanned", "tan"),
      ("falling", "fall"),
      ("hissing", "hiss"),
      ("fizzed", "fizz"),
      ("failing", "fail"),
      ("filing", "file"),
      ("happy", "happi"),
      ("sky", "sky"),
      ("relational", "relat"),
      ("conditional", "condit"),
      ("rational", "ration"),
      ("valenci", "valenc"),
      ("hesitanci", "hesit"),
      ("digitizer", "digit"),
      ("conformabli", "conform"),
      ("radicalli", "radic"),
      ("differentli", "differ"),
      ("vileli", "vile"),
      ("analogousli", "analog"),
      ("vietnamization", "vietnam"),
      ("predication", "predic"),
      ("operator", "oper"),
      ("feudalism", "feudal"),
      ("decisiveness", "decis"),
      ("hopefulness", "hope"),
      ("callousness", "callous"),
      ("formaliti", "formal"),
      ("sensitiviti", "sensit"),
      ("sensibiliti", "sensibl"),
      ("triplicate", "triplic"),
      ("formative", "form"),
      ("formalize", "formal"),
      ("electriciti", "electr"),
      ("electrical", "electr"),
      ("hopeful", "hope"),
      ("goodness", "good"),
      ("revival", "reviv"),
      ("allowance", "allow"),
      ("inference", "infer"),
      ("airliner", "airlin"),
      ("gyroscopic", "gyroscop"),
      ("adjustable", "adjust"),
      ("defensible", "defens"),
      ("irritant", "irrit"),
      ("replacement", "replac"),
      ("adjustment", "adjust"),
      ("dependent", "depend"),
      ("adoption", "adopt"),
      ("homologou", "homolog"),
      ("communism", "commun"),
      ("activate", "activ"),
      ("angulariti", "angular"),
      ("homologous", "homolog"),
      ("effective", "effect"),
      ("bowdlerize", "bowdler"),
      ("probate", "probat"),
      ("rate", "rate"),
      ("cease", "ceas"),
      ("controll", "control"),
      ("roll", "roll"),
      ("generalizations", "gener"),
      ("oscillators", "oscil"),
      ("abilities", "abil"),
      ("able", "abl"),
      ("connection", "connect"),
      ("connections", "connect"),
      ("connective", "connect"),
      ("connected", "connect"),
      ("connecting", "connect"),
      ("running", "run"),
      ("runs", "run"),
      ("ion", "ion"),
      ("is", "is"),
      ("as", "as"),
    ];
    for (word, want) in cases {
      assert_eq!(s(word), want, "stem({word})");
    }
  }
}
