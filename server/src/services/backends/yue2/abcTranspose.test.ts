import assert from 'node:assert/strict';
import test from 'node:test';
import { soundingPitches, transposeAbc } from './abcTranspose.js';

const score = (key: string, vocal: string, ins = 'Z|') =>
  `X:1\nT:\nM:4/4\nL:1/32\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:${key}\n% verse\nV: Vocal\n${vocal}\nV: Ins\n${ins}\n`;
const body = (abc: string) => abc.slice(abc.indexOf('V: Vocal\n', abc.indexOf('K:')) + 9).split('\n')[0];
const key = (abc: string) => abc.match(/^K:(.*)$/m)![1];

function assertMoved(abc: string, semitones: number) {
  const out = transposeAbc(abc, semitones);
  const before = soundingPitches(abc), after = soundingPitches(out);
  assert.deepEqual(after.map(t => t.length), before.map(t => t.length));
  before.forEach((track, i) => track.forEach((p, j) => {
    assert.equal(after[i][j] - p, semitones, `track ${i + 1} note ${j + 1}`);
  }));
  return out;
}

test('a natural that cancels the key signature or a carried accidental keeps its pitch', () => {
  // abcjs strTranspose moves these bars to the wrong pitches at -7 to -12, and at
  // +2 leaves the upper g unmarked, which YuE2's letter-wide carry reads as G natural.
  const abc = score('D', 'F8=F8F8f8|^G8G8=G8g8|');
  const out = assertMoved(abc, 2);
  assert.equal(key(out), 'E');
  assert.equal(body(out), 'G8=G8G8^g8|^A8A8=A8a8|');
  // Into Db the natural F becomes F-flat; the upper f then needs its natural back
  // because YuE2 carries the flat to every octave of F.
  assert.equal(body(assertMoved(abc, -1)), 'F8_F8F8=f8|=G8G8_G8g8|');
});

test("the mountain king's cancelled accidentals, every shift", () => {
  // Bar 3 of the reported score: E#, C#, E#, then E and C cancelled in the same bar.
  const abc = score('D', '"B"^E,,4C,,4E,,8=E,,4=C,,4E,,8|"Bm"B,,,4C,,4D,,4E,,4F,,4D,,4F,,8|');
  for (let s = -12; s <= 12; s++) assertMoved(abc, s);
  assert.equal(body(transposeAbc(abc, 1)), '"C"^F,,4D,,4F,,8=F,,4_D,,4F,,8|"Cm"C,,4D,,4E,,4F,,4G,,4E,,4G,,8|');
  assert.equal(body(transposeAbc(abc, -12)), '"B"^E,,,4C,,,4E,,,8=E,,,4=C,,,4E,,,8|"Bm"B,,,,4C,,,4D,,,4E,,,4F,,,4D,,,4F,,,8|');
});

test('keys move with the fewest accidentals and keep their mode', () => {
  assert.equal(key(transposeAbc(score('D', 'D32|'), 6)), 'Ab');
  assert.equal(key(transposeAbc(score('C', 'C32|'), 6)), 'F#');
  assert.equal(key(transposeAbc(score('C', 'C32|'), -6)), 'Gb');
  assert.equal(key(transposeAbc(score('Am', 'A32|'), 1)), 'Bbm');
  assert.equal(key(transposeAbc(score('Bb', 'B32|'), 3)), 'Db');
  assert.equal(key(transposeAbc(score('C', 'C32|'), 12)), 'C');
  assert.throws(() => transposeAbc(score('Dxyz', 'D32|'), 2), /unsupported key mode/);
});

test('modal keys (SheetSage2-Refined) move with their own signature', () => {
  // D dorian has no accidentals; up a tone it is E dorian, two sharps.
  const dor = assertMoved(score('Ddor', 'D8F8B8c8|'), 2);
  assert.equal(key(dor), 'Edor');
  assert.equal(body(dor), 'E8G8c8d8|');
  for (let s = -12; s <= 12; s++) {
    for (const k of ['Ephr', 'Flyd', 'Gmix', 'Bloc', 'F#dor']) assertMoved(score(k, 'C8^F8B8_e8|'), s);
  }
  assert.equal(key(transposeAbc(score('Gmix', 'G32|'), 6)), 'C#mix'); // six either way: up prefers sharps, as C +6 = F#;
});

test('a key change mid-piece moves its own signature and spelling', () => {
  const abc = score('C', 'F8^F8[K:Eb]B8=B8|B8[K:D]F8=F8F8|', 'z8z8[K:Eb]z16|z8[K:D]z24|');
  const out = assertMoved(abc, 3);
  assert.equal(key(out), 'Eb');
  assert.equal(body(out), 'A8=A8[K:Gb]d8=d8|d8[K:F]A8_A8A8|');
  assert.match(out, /^z8z8\[K:Gb\]z16\|z8\[K:F\]z24\|$/m);
});

test('a tie into the next bar keeps its pitch and sets no accidental there', () => {
  const abc = score('C', '_A32-|A16A16|');
  const out = assertMoved(abc, 2);
  assert.equal(body(out), '_B32-|B16B16|');
  assert.deepEqual(soundingPitches(out)[0], [70, 71]);
});

test('chord symbols move root and bass; annotations stay', () => {
  const out = assertMoved(score('G', '"D7/F#"A16"^rit."B16|"Em"G32|'), -2);
  assert.equal(body(out), '"C7/E"G16"^rit."A16|"Dm"F32|');
});

test('zero is a no-op, fractions and a missing key are refused', () => {
  const abc = score('C', 'C32|');
  assert.equal(transposeAbc(abc, 0), abc);
  assert.throws(() => transposeAbc(abc, 1.5), /whole number/);
  assert.throws(() => transposeAbc('X:1\nC32|\n', 1), /without K:/);
});
