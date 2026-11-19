// lyricsPronunciation.ts — the lyric token as the CTC aligner needs to hear it.
//
// MMS_FA scores the audio against the LETTERS of the lyric, and the engine keeps
// only `[a-z']` from each word (`yue2_cursor_words_of`, engine/src/yue2/
// yue2-ctc-align.h). That is fine for "hold the line" and wrong for the words a
// song makes hard:
//
//   AGI    becomes "agi"        — nobody sings "agi", they sing "ay gee eye"
//   DJ     becomes "dj"         — no vowel for the model to stretch
//   2024   becomes ""           — the word has NO target, so it gets no span and
//                                 silently disappears from the line
//
// The fix is not in the model, it is in the text handed to it. This is the same
// move the pdoom-video project makes for its own word-level lyrics
// (analysis/pron.py): an explicit table for the tokens an artist spells out, and
// a fallback that reduces anything else to pronounceable letters. Its notes
// record where the payoff is — acronyms and initialisms are exactly the words a
// char-level aligner smears across the rest of the line.
//
// Everything here is pure and offset-preserving: `alignerLyrics()` also returns
// the map from each unit back to the lyric word it spells, so a span the engine
// returns can be folded onto the word the sidecar stores. The engine's contract
// is that `char0`/`char1` index the string it was given, so the map has to be
// built here, where that string is made — never re-derived by searching it.

/** A token's pronunciation, when the plain lowercase spelling is not what the
 *  singer sings. Keys are the token with case and punctuation removed, so
 *  `P(doom)`, `p.doom` and `P-DOOM` all hit `pdoom`. */
const SPOKEN_AS: Record<string, string> = {
  // Spelled out letter by letter.
  agi: 'ay gee eye',
  ai: 'ay eye',
  api: 'ay pee eye',
  cpu: 'see pee you',
  gpu: 'gee pee you',
  llm: 'el el em',
  mlp: 'em el pee',
  nvda: 'en vee dee ay',
  rlhf: 'are el aitch eff',
  dj: 'dee jay',
  mc: 'em see',
  tv: 'tee vee',
  vr: 'vee are',
  ar: 'ay are',
  uk: 'you kay',
  usa: 'you ess ay',
  ok: 'oh kay',
  // Sung as a word even though it is written in capitals (pdoom's FOOM).
  foom: 'foom',
};

/** How each letter is named when a token is spelled out. pdoom's own list
 *  (`pron.py`), which is the one its alignment was checked against. */
const LETTER_NAMES: Record<string, string> = {
  a: 'ay', b: 'bee', c: 'see', d: 'dee', e: 'ee', f: 'eff', g: 'gee',
  h: 'aitch', i: 'eye', j: 'jay', k: 'kay', l: 'el', m: 'em', n: 'en',
  o: 'oh', p: 'pee', q: 'cue', r: 'are', s: 'ess', t: 'tee', u: 'you',
  v: 'vee', w: 'double you', x: 'ex', y: 'why', z: 'zed',
};

const ONES = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
  'seventeen', 'eighteen', 'nineteen',
];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];

/** English words for 0..9999; anything longer is read digit by digit, because a
 *  number nobody can sing as a number is not worth a guess. */
function numberWords(value: number): string[] {
  if (!Number.isFinite(value) || value < 0 || value > 9999) return [];
  if (value < 20) return [ONES[value]];
  if (value < 100) {
    const tens = TENS[Math.floor(value / 10)];
    const rest = value % 10;
    return rest ? [tens, ONES[rest]] : [tens];
  }
  if (value < 1000) {
    const hundreds = `${ONES[Math.floor(value / 100)]} hundred`;
    const rest = value % 100;
    return rest ? [hundreds, ...numberWords(rest)] : [hundreds];
  }
  const thousands = `${ONES[Math.floor(value / 1000)]} thousand`;
  const rest = value % 1000;
  return rest ? [thousands, ...numberWords(rest)] : [thousands];
}

/** A run of digits as the words a singer uses for it. */
function spelledNumber(digits: string): string[] {
  if (digits.length > 4) return [...digits].map(digit => ONES[Number(digit)]);
  const value = Number(digits);
  // A year is sung in pairs — "twenty twenty four", "nineteen ninety nine" —
  // which is neither the cardinal reading nor digit by digit. It is the one
  // number where the reading changes where the syllables land.
  if (/^(1\d|20)\d\d$/.test(digits)) {
    const high = Math.floor(value / 100);
    const low = value % 100;
    if (low > 0) {
      return [high === 20 ? 'twenty' : ONES[high], numberWords(low).join(' ')];
    }
  }
  const words = numberWords(value);
  return words.length ? words : [...digits].map(digit => ONES[Number(digit)]);
}

/**
 * What the aligner should look for, one entry per pronounced unit.
 *
 * Order matters: an explicit table first (the tokens an artist really spells
 * out), then digits, then the initialism rule, then the plain token. A word with
 * no letters at all — a dash, a note symbol, "…" — returns `[]`, which is the
 * engine's own verdict: it has no alignable character, so it gets no span and
 * the sidecar keeps it without a time rather than inventing one.
 */
export function pronounceWord(token: string): string[] {
  const trimmed = token.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
  if (!trimmed) return [];

  const key = trimmed.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  const known = SPOKEN_AS[key];
  if (known) return normaliseUnits(known);

  // Digits: one word per number run, in the words a singer uses for it.
  if (/\d/.test(trimmed)) {
    const units: string[] = [];
    for (const part of trimmed.split(/(\d+)/)) {
      if (!part) continue;
      if (/^\d+$/.test(part)) units.push(...spelledNumber(part));
      else units.push(...normaliseUnits(part));
    }
    if (units.length) return units;
  }

  // An initialism: capitals with no vowel to sustain. Anything with a vowel is
  // sung as a word (NASA, FOOM), so it is only lowercased — spelling those out
  // is the failure this rule exists to avoid.
  const lettersOnly = /^\p{Lu}{2,6}$/u.test(trimmed);
  if (lettersOnly && !/[AEIOUY]/i.test(trimmed.replace(/[^A-Za-z]/g, ''))) {
    const spelled = [...trimmed.toLowerCase()].map(letter => LETTER_NAMES[letter]).filter(Boolean);
    if (spelled.length) return spelled;
  }

  return normaliseUnits(trimmed);
}

/** Lowercase, keep the letters and the internal apostrophes, split on everything
 *  else (a hyphen is a word break to the aligner either way). */
function normaliseUnits(text: string): string[] {
  return text
    .replace(/[’‘]/g, "'")
    .toLowerCase()
    .split(/[^a-z']+/)
    .map(part => part.replace(/^'+|'+$/g, ''))
    .filter(part => part.length > 0);
}

/** One pronunciation unit, located in the string that will be posted. */
export interface AlignerUnit {
  /** Codepoint span in `AlignerLyrics.text`. */
  start: number;
  end: number;
  /** The unit itself, as posted. */
  text: string;
}

/** One lyric word, spelled out as one or more units. */
export interface AlignerWord {
  /** Codepoint span of the ORIGINAL token in the lyrics handed in. */
  start: number;
  end: number;
  /** Indices into `AlignerLyrics.units`, inclusive. */
  firstUnit: number;
  lastUnit: number;
}

export interface AlignerLyrics {
  /** The string to POST to /yue2/align; the returned offsets index into it. */
  text: string;
  /** Every unit in performance order. */
  units: AlignerUnit[];
  /** Every word that has something to align, in lyrics order. */
  words: AlignerWord[];
  /** Words whose target is not simply the lowercased token: the ones this
   *  module changed, so a run can say whether it did anything. */
  expanded: number;
}

interface SourceWord {
  start: number;
  end: number;
  token: string;
}

interface SourceLine {
  /** A `[Section]` label: not lyrics, and the engine skips it too. */
  tag?: string;
  words: SourceWord[];
}

/** Split the lyrics into lines of tokens, keeping each token's original
 *  codepoint span. The tag test matches the engine's own. */
function sourceLines(chars: string[]): SourceLine[] {
  const out: SourceLine[] = [];
  let start = 0;
  for (let i = 0; i <= chars.length; i++) {
    if (i < chars.length && chars[i] !== '\n') continue;
    const text = chars.slice(start, i).join('');
    const trimmed = text.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']') && trimmed.length > 1) {
      out.push({ tag: trimmed, words: [] });
    } else {
      const words: SourceWord[] = [];
      let k = 0;
      while (k < text.length) {
        while (k < text.length && /\s/.test(text[k])) k++;
        if (k >= text.length) break;
        const from = k;
        while (k < text.length && !/\s/.test(text[k])) k++;
        words.push({ start: start + from, end: start + k, token: text.slice(from, k) });
      }
      if (words.length) out.push({ words });
    }
    start = i + 1;
  }
  return out;
}

/**
 * Build the lyrics string the CTC aligner should be given, plus the map back.
 *
 * For a lyric of ordinary words the text this produces differs from the source
 * only in case, punctuation and spacing — the same characters the engine would
 * have kept anyway — so nothing but the hard tokens changes. `[Section]` labels
 * are carried through untouched: the engine reads them as labels, not words.
 */
export function alignerLyrics(lyrics: string): AlignerLyrics {
  const chars = Array.from(lyrics);
  const lines = sourceLines(chars);
  const units: AlignerUnit[] = [];
  const words: AlignerWord[] = [];
  let text = '';
  // Codepoint offset of the next character appended. Tracked rather than read
  // off `text.length` so an astral character in a lyric cannot shift every span
  // after it by one per character.
  let at = 0;
  let expanded = 0;

  const push = (piece: string) => {
    text += piece;
    at += Array.from(piece).length;
  };

  lines.forEach((line, index) => {
    if (index > 0) push('\n');
    if (line.tag) {
      push(line.tag);
      return;
    }
    line.words.forEach((word, wordIndex) => {
      const spoken = pronounceWord(word.token);
      // Nothing to align (a dash, a glyph): leave it out of the text, exactly as
      // the engine would. It keeps no span, and the sidecar says nothing about
      // it rather than guessing.
      if (spoken.length === 0) return;
      if (wordIndex > 0) push(' ');
      const firstUnit = units.length;
      spoken.forEach((unit, unitIndex) => {
        if (unitIndex > 0) push(' ');
        units.push({ start: at, end: at + Array.from(unit).length, text: unit });
        push(unit);
      });
      words.push({ start: word.start, end: word.end, firstUnit, lastUnit: units.length - 1 });
      if (normaliseUnits(word.token).join(' ') !== spoken.join(' ')) expanded++;
    });
  });

  return { text, units, words, expanded };
}
