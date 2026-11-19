import { describe, expect, it } from 'vitest';
import { seedLinesFromLrc } from './lrcSeed';

const lrc = [
  '[00:10.00][Verse 1]',
  '[00:12.00]hello there world',
  '[00:20.50]second line',
].join('\n');

describe('seeding word chips from a line-level LRC', () => {
  it('starts every line where the .lrc says, with its words inside that line', () => {
    const lines = seedLinesFromLrc(lrc, 30);
    expect(lines).toHaveLength(2);
    expect(lines[0].section).toBe('Verse 1');
    expect(lines[0].words.map(w => w.word)).toEqual(['hello', 'there', 'world']);
    expect(lines[0].start).toBe(12);
    // Three words at the normal spread, well inside the 8.5 s to the next line.
    expect(lines[0].end).toBeCloseTo(12.84, 3);
    // Contiguous within a line: no gaps and no overlaps.
    for (const line of lines) {
      for (let i = 1; i < line.words.length; i++) {
        expect(line.words[i].start).toBe(line.words[i - 1].end);
      }
    }
  });

  it('does not stretch a line across a long instrumental', () => {
    const lines = seedLinesFromLrc('[00:05.00]one two\n[01:30.00]three', 200);
    expect(lines[0].end).toBeCloseTo(5.56, 3);
    expect(lines[1].start).toBe(90);
  });

  it('gives the last line room when the track length is unknown', () => {
    const lines = seedLinesFromLrc('[00:05.00]one two', 0);
    expect(lines).toHaveLength(1);
    expect(lines[0].end).toBeCloseTo(5.56, 3);
  });

  it('returns nothing for lyrics with no times', () => {
    expect(seedLinesFromLrc('')).toEqual([]);
    expect(seedLinesFromLrc('[Verse 1]\nwords without times')).toEqual([]);
  });
});
