import assert from 'node:assert/strict';
import test from 'node:test';
import { alignerLyrics, type AlignerLyrics } from '../../lyricsPronunciation.js';
import { yue2LyricsJson, type Yue2AlignWord } from './align.js';

const LYRICS = '[Verse 1]\nhold the line\n\n[Chorus]\nlet it go\n';

/** Span for the word at `word`'s first occurrence, timed at [start, end). */
function span(word: string, start: number, end: number): Yue2AlignWord {
  const c0 = Array.from(LYRICS).join('').indexOf(word);
  return { start, end, score: 0.9, char0: c0, char1: c0 + word.length };
}

test('words land on their own line, with that line’s section', () => {
  const json = yue2LyricsJson(LYRICS, [
    span('hold', 1, 1.4), span('the', 1.4, 1.6), span('line', 1.6, 2.2),
    span('let', 9, 9.3), span('it', 9.3, 9.5), span('go', 9.5, 10.4),
  ]);

  assert.equal(json.method, 'mms-fa');
  assert.deepEqual(json.lines.map(l => l.text), ['hold the line', 'let it go']);
  assert.deepEqual(json.lines.map(l => l.section), ['Verse 1', 'Chorus']);
  assert.deepEqual(json.lines.map(l => [l.start, l.end]), [[1, 2.2], [9, 10.4]]);
  assert.deepEqual(json.lines[0].words.map(w => w.word), ['hold', 'the', 'line']);
});

test('a line the aligner timed no word on is dropped, not invented', () => {
  const json = yue2LyricsJson(LYRICS, [span('let', 9, 9.3)]);
  assert.deepEqual(json.lines.map(l => l.text), ['let it go']);
});

/** One span per word of the aligner text, in order, so a test says nothing about
 *  where the engine put it — only that every unit was placed. The lyric text in
 *  these fixtures is ASCII, where a UTF-16 index is a codepoint index. */
function spansOf(plan: AlignerLyrics, from = 1, step = 0.25): Yue2AlignWord[] {
  const out: Yue2AlignWord[] = [];
  const words = /\S+/g;
  let match: RegExpExecArray | null;
  while ((match = words.exec(plan.text))) {
    out.push({ start: from, end: from + step, score: 0.8, char0: match.index, char1: match.index + match[0].length });
    from += step;
  }
  return out;
}

function close(actual: number, expected: number, because = ''): void {
  assert.ok(
    Math.abs(actual - expected) < 1e-9,
    `${actual} should be ${expected}${because ? ` (${because})` : ''}`,
  );
}

const PRONOUNCED = 'you are an AGI now';

test('an acronym spelled out comes back as the one word the sidecar stores', () => {
  // The aligner is asked for five units of "you are an AGI now" — three of them
  // spell AGI — because "agi" is not what is sung. The word takes the first
  // unit's onset, the last unit's end, and the mean of their scores.
  const plan = alignerLyrics(PRONOUNCED);
  const spans = spansOf(plan);
  spans[3].score = 0.6;
  const json = yue2LyricsJson(PRONOUNCED, spans, plan);
  const line = json.lines[0];

  assert.deepEqual(line.words.map(w => w.word), ['you', 'are', 'an', 'AGI', 'now']);
  const agi = line.words[3];
  assert.equal(agi.start, 1.75, 'the onset of "ay"');
  close(agi.end, 2.5, 'the end of "eye"');
  assert.ok(Math.abs((agi.confidence ?? 0) - (0.6 + 0.8 + 0.8) / 3) < 1e-9, 'the mean of its units');
  assert.equal(agi.word, 'AGI', 'the lyric’s own spelling, not the pronunciation');
});

test('a number is timed as one word instead of being dropped', () => {
  // "2024" has no letter in the engine’s alphabet, so without the pronunciation
  // map it gets no span at all and the line loses a word.
  const lyrics = 'in 2024 we sang';
  const plan = alignerLyrics(lyrics);
  const json = yue2LyricsJson(lyrics, spansOf(plan), plan);

  assert.deepEqual(json.lines[0].words.map(w => w.word), ['in', '2024', 'we', 'sang']);
  const year = json.lines[0].words[1];
  assert.equal(year.start, 1.25, 'the first word of "twenty"');
  close(year.end, 2, 'the end of "four"');
});

test('a unit the engine dropped leaves a hole, not a wrong span', () => {
  const plan = alignerLyrics(PRONOUNCED);
  const spans = spansOf(plan).filter(span => plan.text.slice(span.char0, span.char1) !== 'gee');
  const json = yue2LyricsJson(PRONOUNCED, spans, plan);
  const agi = json.lines[0].words[3];

  assert.equal(agi.start, 1.75, 'still the first unit it did place');
  close(agi.end, 2.5, 'and the last');
  assert.ok(Math.abs((agi.confidence ?? 0) - 0.8) < 1e-9);
});

test('offsets are codepoints, not UTF-16 units', () => {
  // The engine counts 🎵 as ONE character; a naive lyrics[i] counts two.
  const lyrics = '[Verse]\n🎵 sing\n';
  const c0 = Array.from('[Verse]\n🎵 ').length;
  const json = yue2LyricsJson(lyrics, [{ start: 2, end: 2.5, score: 1, char0: c0, char1: c0 + 4 }]);
  assert.equal(json.lines[0].words[0].word, 'sing');
  assert.equal(json.lines[0].text, '🎵 sing');
});
