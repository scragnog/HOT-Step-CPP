import assert from 'node:assert/strict';
import test from 'node:test';
import { measureCoverDrift } from './coverDrift.js';
import { buildYue2LyricSchedule } from './lyricSchedule.js';

const options = { mode: 'bias' as const, bias: -4, abc: true, leadSec: 0, behind: -1 };
const cps = (text: string, [a, b]: [number, number]) => Array.from(text).slice(a, b).join('');

// Rest-led sections, an inline meter change and an instrumental break.
const abc = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C',
  '% intro', 'V: Vocal', 'z4|', 'V: Ins', 'C4|',
  '% verse', 'V: Vocal', 'z2C2|', 'M:1/4', 'D|', 'M:4/4',
  '% break', 'V: Vocal', 'z4|',
  '% chorus', 'V: Vocal', 'z1E3|F4|', ''].join('\n');
const lyrics = '[Intro]\n\n[Verse]\nfirst line\n\n[Break]\n\n[Chorus]\nsecond line\n\n[Outro]\nlast words';

test('each section is revealed at the S13 v4 first Vocal note onset', () => {
  const { wire } = buildYue2LyricSchedule(abc, lyrics, options);
  const words = [['first', 6], ['second', 14]].map(([text, start]) => {
    const char0 = Array.from(lyrics.slice(0, lyrics.indexOf(text as string))).length;
    return { char0, char1: char0 + (text as string).length, start: start as number, end: (start as number) + 0.5, score: 1 };
  });
  const drift = measureCoverDrift(abc, abc, lyrics, words);
  const onsets = drift.sections.filter(s => s.sung).map(s => s.expected.start);
  // Intro bar 0-4 s; verse bar 2 starts at 4 s, first note after a half rest
  // = 6 s; the 1/4 bar is 8-9 s, the break 9-13 s, so the chorus's first note
  // follows a quarter rest at 14 s. A single-meter clock would say 17 s.
  assert.deepEqual(onsets, [6, 14]);
  assert.deepEqual(wire.sections.map(s => s.start_sec), onsets);
});

test('spans cover the exact lyric block and the section score lines', () => {
  const { wire, untimed } = buildYue2LyricSchedule(abc, lyrics, options);
  assert.deepEqual(wire.sections.map(s => cps(lyrics, s.lyric)), ['[Verse]\nfirst line', '[Chorus]\nsecond line']);
  assert.deepEqual(wire.sections.map(s => cps(abc, s.abc!)),
    ['% verse\nV: Vocal\nz2C2|\nM:1/4\nD|\nM:4/4\n', '% chorus\nV: Vocal\nz1E3|F4|\n']);
  assert.deepEqual(untimed, ['Outro']);
  assert.deepEqual(wire, { mode: 'bias', bias: -4, abc: true, sections: wire.sections });
});

test('codepoint offsets survive non-ASCII lyrics', () => {
  const text = '[Verse]\nnaïve café ✓\n\n[Chorus]\nsecond line';
  const { wire } = buildYue2LyricSchedule(abc, text, { ...options, mode: 'mask', abc: false });
  assert.deepEqual(wire.sections.map(s => cps(text, s.lyric)), ['[Verse]\nnaïve café ✓', '[Chorus]\nsecond line']);
  assert.equal(wire.sections[0].abc, undefined);
  assert.equal(wire.bias, undefined);
});

test('a free-tempo score or CRLF lyrics is refused rather than guessed', () => {
  assert.throws(() => buildYue2LyricSchedule(abc.replace(/^Q:.*\n/m, ''), lyrics, options), /tempo/);
  assert.throws(() => buildYue2LyricSchedule(abc, lyrics.replace(/\n/g, '\r\n'), options), /line endings/);
});
