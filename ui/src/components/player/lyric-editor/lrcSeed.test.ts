import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { seedLinesFromLrc } from './lrcSeed';

const lrc = [
  '[00:10.00][Verse 1]',
  '[00:12.00]hello there world',
  '[00:20.50]second line',
].join('\n');

describe('seeding word chips from a line-level LRC', () => {
  it('starts every line where the .lrc says, with its words inside that line', () => {
    const lines = seedLinesFromLrc(lrc, 30);
    assert.equal((lines).length, 2);
    assert.equal(lines[0].section, 'Verse 1');
    assert.deepEqual(lines[0].words.map(w => w.word), ['hello', 'there', 'world']);
    assert.equal(lines[0].start, 12);
    // Three words at the normal spread, well inside the 8.5 s to the next line.
    assert.ok(Math.abs((lines[0].end) - (12.84)) < 10 ** -3 / 2);
    // Contiguous within a line: no gaps and no overlaps.
    for (const line of lines) {
      for (let i = 1; i < line.words.length; i++) {
        assert.equal(line.words[i].start, line.words[i - 1].end);
      }
    }
  });

  it('does not stretch a line across a long instrumental', () => {
    const lines = seedLinesFromLrc('[00:05.00]one two\n[01:30.00]three', 200);
    assert.ok(Math.abs((lines[0].end) - (5.56)) < 10 ** -3 / 2);
    assert.equal(lines[1].start, 90);
  });

  it('gives the last line room when the track length is unknown', () => {
    const lines = seedLinesFromLrc('[00:05.00]one two', 0);
    assert.equal((lines).length, 1);
    assert.ok(Math.abs((lines[0].end) - (5.56)) < 10 ** -3 / 2);
  });

  it('returns nothing for lyrics with no times', () => {
    assert.deepEqual(seedLinesFromLrc(''), []);
    assert.deepEqual(seedLinesFromLrc('[Verse 1]\nwords without times'), []);
  });
});
