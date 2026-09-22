import assert from 'node:assert/strict';
import test from 'node:test';
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

test('offsets are codepoints, not UTF-16 units', () => {
  // The engine counts 🎵 as ONE character; a naive lyrics[i] counts two.
  const lyrics = '[Verse]\n🎵 sing\n';
  const c0 = Array.from('[Verse]\n🎵 ').length;
  const json = yue2LyricsJson(lyrics, [{ start: 2, end: 2.5, score: 1, char0: c0, char1: c0 + 4 }]);
  assert.equal(json.lines[0].words[0].word, 'sing');
  assert.equal(json.lines[0].text, '🎵 sing');
});
