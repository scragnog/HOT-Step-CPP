import assert from 'node:assert/strict';
import test from 'node:test';
import { measureCoverDrift } from './coverDrift.js';
import { buildYue2LyricSchedule } from './lyricSchedule.js';
import { scoreBarClock } from './scoreClock.js';
import { classifyYue2Score } from './scoreHealth.js';

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
  // Field lines (V:, M:, Q:, K:) stay visible: they outlive their section.
  assert.deepEqual(wire.sections.map(s => s.abc!.map(span => cps(abc, span))),
    [['% verse\n', 'z2C2|\n', 'D|\n'], ['% chorus\n', 'z1E3|F4|\n']]);
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

test('hand-written score with blank lines and final barline builds a mask', () => {
  const score = "X:1\nT:Twinkle Twinkle Little Star (Test)\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\nV:1 name=\"Vocal\" clef=treble\n\n%% Intro\n\"C\" C2 G2 E2 G2 | \"F\" A2 F2 \"C\" E4 ||\n\n%% Verse 1\n\"C\" C2 C2 G2 G2 | \"F\" A2 A2 \"C\" G4 |\nw: Twin-kle, twin-kle, lit-tle star,\n\"F\" F2 F2 \"C\" E2 E2 | \"G7\" D2 D2 \"C\" C4 ||\nw: How I won-der what you are!\n\n%% Verse 2\n\"C\" G2 G2 \"F\" F2 F2 | \"C\" E2 E2 \"G7\" D4 |\nw: Up a-bove the world so high,\n\"C\" G2 G2 \"F\" F2 F2 | \"C\" E2 E2 \"G7\" D4 ||\nw: Like a dia-mond in the sky.\n\n%% Verse 3\n\"C\" C2 C2 G2 G2 | \"F\" A2 A2 \"C\" G4 |\nw: Twin-kle, twin-kle, lit-tle star,\n\"F\" F2 F2 \"C\" E2 E2 | \"G7\" D2 D2 \"C\" C4 |]\nw: How I won-der what you are!\n";
  const words = "[Intro]\n\n[Verse 1]\nTwinkle, twinkle, little star,\nHow I wonder what you are!\n\n[Verse 2]\nUp above the world so high,\nLike a diamond in the sky.\n\n[Verse 3]\nTwinkle, twinkle, little star,\nHow I wonder what you are!\n";
  assert.equal(classifyYue2Score(score, 'eos').bars, 14);
  assert.equal(scoreBarClock(score).length, 14);
  assert.equal(buildYue2LyricSchedule(score, words, { ...options, mode: 'mask' }).wire.sections.length, 3);
});

test('inline meter changes and an indented tempo preserve the section clock', () => {
  const score = ['X:1', 'M:4/4', 'L:1/4', '  Q:1/4=120', 'K:C',
    'V:1 name="Vocal"', '% verse', 'C D E F | [M: 2/4] G A |',
    '% chorus', '[M: 3/4] [L: 1/4] B c d |]', 'w: one two three'].join('\n');
  const bars = scoreBarClock(score);
  assert.deepEqual(bars.map(bar => bar.meter), ['4/4', '2/4', '3/4']);
  assert.deepEqual(bars.map(bar => bar.end), [2, 3, 4.5]);
  const result = buildYue2LyricSchedule(score, '[Verse]\nwords\n[Chorus]\nmore words', { ...options, mode: 'mask' });
  assert.deepEqual(result.wire.sections.map(section => section.start_sec), [0, 3]);
});
