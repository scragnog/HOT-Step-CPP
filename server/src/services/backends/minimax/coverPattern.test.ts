import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyMm3ForcePattern, parseMm3ForcePattern, trimMm3Codes } from './coverPattern.js';

// Row 0 warm-up + 10 s of frames, codes 100.. so a pinned row is easy to spot.
const sem = Array.from({ length: 251 }, (_, i) => 100 + i);
const free = (out: number[]) => out.flatMap((c, i) => (c === -1 ? [i] : []));

test('prefix pins the first N seconds and frees the rest', () => {
  const out = applyMm3ForcePattern(sem, { prefixSeconds: 2 });
  assert.equal(out[50], sem[50]);  // frame 49 = 1.96 s
  assert.equal(out[51], -1);       // frame 50 = 2.00 s
  assert.equal(free(out).length, 200);
});

test('anchors pin every Nth frame', () => {
  const out = applyMm3ForcePattern(sem, { anchorEvery: 4 });
  assert.equal(out[1], sem[1]);
  assert.deepEqual(out.slice(2, 6), [-1, -1, -1, sem[5]]);
});

test('ranges pin [start, end) and invert frees them instead', () => {
  const p = { pinnedRanges: [[2, 4], [8, 9]] as Array<[number, number]> };
  const out = applyMm3ForcePattern(sem, p);
  assert.equal(out[50], -1);
  assert.equal(out[51], sem[51]);
  assert.equal(out[101], -1);
  assert.equal(free(out).length, 250 - 75);
  const inv = applyMm3ForcePattern(sem, { ...p, invert: true });
  assert.equal(inv[51], -1);
  assert.equal(inv[50], sem[50]);
  assert.equal(free(inv).length, 75);
});

test('row 0 stays pinned, even inverted', () => {
  assert.equal(applyMm3ForcePattern(sem, { prefixSeconds: 0.01, invert: true })[0], sem[0]);
  assert.equal(applyMm3ForcePattern(sem, { pinnedRanges: [[5, 6]] })[0], sem[0]);
});

test('parse keeps well-typed rules and returns null for none', () => {
  assert.equal(parseMm3ForcePattern(undefined), null);
  assert.equal(parseMm3ForcePattern({ invert: true }), null);
  assert.deepEqual(parseMm3ForcePattern({ prefixSeconds: 20, anchorEvery: 'x', pinnedRanges: [[0, 5], [9, 3], 'a'] }),
    { prefixSeconds: 20, pinnedRanges: [[0, 5]] });
});

test('trim keeps the warm-up row plus N seconds of both arrays', () => {
  const ac = sem.flatMap(c => Array(7).fill(c));
  const t = trimMm3Codes(sem, ac, 4);
  assert.equal(t.sem.length, 101);
  assert.equal(t.ac.length, 707);
  assert.equal(trimMm3Codes(sem, ac, 60).sem, sem);
});
