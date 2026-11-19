import assert from 'node:assert/strict';
import test from 'node:test';
import { handCheckedLines, scoreFixture, type LyricsFixture } from './lyricsFixture.js';
import type { LyricsLine, LyricsWord } from './lyricsReconcile.js';

function word(text: string, start: number, end: number): LyricsWord {
  return { word: text, start, end, confidence: 1, source: 'matched' };
}

function line(words: LyricsWord[], section?: string): LyricsLine {
  return {
    start: words[0].start,
    end: words[words.length - 1].end,
    text: words.map(w => w.word).join(' '),
    ...(section ? { section } : {}),
    words,
  };
}

function fixtureOf(lines: LyricsLine[]): LyricsFixture {
  return { version: 1, audioUrl: 'x.mp3', createdAt: '2026-01-01T00:00:00.000Z', lines };
}

/** Shift every word by `delta`, for a run that is uniformly early or late. */
function shifted(lines: LyricsLine[], delta: number): LyricsLine[] {
  return lines.map(l => ({
    ...l,
    start: l.start + delta,
    end: l.end + delta,
    words: l.words.map(w => ({ ...w, start: w.start + delta, end: w.end + delta })),
  }));
}

const track = (): LyricsLine[] => [
  line([word('one', 1, 1.4), word('two', 1.4, 1.8)], 'Verse 1'),
  line([word('three', 2, 2.4), word('four', 2.4, 2.8)]),
  line([word('five', 5, 5.4), word('six', 5.4, 5.8)], 'Chorus'),
  line([word('seven', 9, 9.4), word('eight', 9.4, 9.8)], 'Chorus'),
];

test('a fixture keeps the lines a person checked, and nothing else', () => {
  const lines = track();
  lines[1] = { ...lines[1], checked: true };
  lines[3] = { ...lines[3], checked: true };
  // A machine-placed line is not a reference: keeping it would score every
  // later run mostly against the previous run's guess.
  const kept = handCheckedLines(lines);
  assert.deepEqual(kept.map(l => l.text), ['three four', 'seven eight']);
  assert.deepEqual(handCheckedLines(track()), []);
});

test('a perfect run scores zero error', () => {
  const result = scoreFixture(fixtureOf(track()), track());
  assert.equal(result.words, 8);
  assert.equal(result.matched, 8);
  assert.equal(result.missing, 0);
  assert.equal(result.medianMs, 0);
  assert.equal(result.p90Ms, 0);
  assert.equal(result.within100, 1);
  assert.equal(result.monotonicViolations, 0);
  assert.equal(result.secondChorusOffsetMs, 0);
});

test('a uniform shift shows up as its own offset, with no drift', () => {
  const result = scoreFixture(fixtureOf(track()), shifted(track(), 0.08));
  assert.equal(result.medianMs, 80);
  assert.equal(result.within100, 1);
  assert.equal(result.within250, 1);
  // Both choruses moved the same amount, so the repeat is not evidence of drift.
  assert.equal(result.secondChorusOffsetMs, 0);
});

test('a second chorus timed somewhere else is the number that exposes it', () => {
  const candidate = shifted(track(), 0);
  // The repeated chorus lands half a second late while the rest is right.
  candidate[3] = shifted([candidate[3]], 0.5)[0];
  const result = scoreFixture(fixtureOf(track()), candidate);
  assert.equal(result.medianMs, 0);
  assert.equal(result.secondChorusOffsetMs, 500);
  const chorus = result.sections.filter(s => s.section === 'Chorus');
  assert.deepEqual(chorus.map(s => s.occurrence), [1, 2]);
  assert.deepEqual(chorus.map(s => s.medianMs), [0, 500]);
});

test('a word the run dropped does not shift every later error', () => {
  const candidate = track();
  candidate[0] = { ...candidate[0], words: [candidate[0].words[1]], text: 'two' };
  const result = scoreFixture(fixtureOf(track()), candidate);
  assert.equal(result.missing, 1);
  assert.equal(result.matched, 7);
  // "three" is still compared to "three", not to "four".
  assert.equal(result.medianMs, 0);
});

test('words that overlap count as monotonic violations', () => {
  const candidate = track();
  candidate[1] = {
    ...candidate[1],
    // "three" starts before "two" ended; "four" still follows it.
    words: [word('three', 1.7, 2.4), word('four', 2.4, 2.8)],
  };
  const result = scoreFixture(fixtureOf(track()), candidate);
  assert.equal(result.monotonicViolations, 1);
});

test('a coarse run reports how little landed inside the tolerances', () => {
  const result = scoreFixture(fixtureOf(track()), shifted(track(), 0.3));
  assert.equal(result.medianMs, 300);
  assert.equal(result.within250, 0);
  assert.equal(result.maxMs, 300);
});
