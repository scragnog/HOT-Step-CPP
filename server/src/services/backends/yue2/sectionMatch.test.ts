import assert from 'node:assert/strict';
import test from 'node:test';
import { matchSectionsToScore } from './sectionMatch.js';
import type { Yue2AlignWord } from './align.js';

// Two 3-second bars per section: intro 0, verse 6, chorus 12, verse 18,
// chorus 24, outro 30. Intro and outro have no Vocal note.
const score = [
  'X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C',
  '% intro', 'V: Vocal', 'z6|z6|',
  '% verse', 'V: Vocal', 'C6|D6|',
  '% chorus', 'V: Vocal', 'E6|F6|',
  '% verse', 'V: Vocal', 'C6|D6|',
  '% chorus', 'V: Vocal', 'E6|F6|',
  '% outro', 'V: Vocal', 'z6|z6|', '',
].join('\n');

function words(lyrics: string, times: Array<[string, number, number?]>): Yue2AlignWord[] {
  return times.map(([text, start, score = 0.9]) => {
    const char0 = Array.from(lyrics.slice(0, lyrics.indexOf(text))).length;
    return { char0, char1: char0 + Array.from(text).length, start, end: start + 0.4, score };
  });
}

test('retags by sung time, keeps same-kind tags, fills a missing chorus and instrumental tags', () => {
  const lyrics = '[Intro]\n\n[Verse 1]\nfirst line\n\n[Hook]\nsecond line\n\n[Verse 2]\nthird line';
  // "second" lands half a bar before the chorus: a pickup into it.
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.2], ['second', 11.5], ['third', 18.3]]));
  assert.equal(result.lyrics, '[Intro]\n\n[Verse 1]\nfirst line\n\n[Chorus]\nsecond line\n\n[Verse 2]\nthird line\n\n[Chorus]\nsecond line\n\n[Outro]');
  assert.deepEqual(result.blocks.map(b => b.status), ['dropped', 'kept', 'renamed', 'kept']);
  assert.deepEqual(result.filled.map(f => [f.section, f.copiedFrom]), [[1, null], [5, 3], [6, null]]);
  assert.equal(result.unchanged, false);
});

test('a block with no aligned words keeps its tag and place, and blocks copies into its gap', () => {
  const lyrics = '[verse]\nfirst line\n[hook]\nsecond line\n[verse]\nthird line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.2], ['second', 12.1]]));
  assert.equal(result.lyrics, '[intro]\n\n[verse]\nfirst line\n\n[chorus]\nsecond line\n\n[verse]\nthird line\n\n[outro]');
  assert.equal(result.blocks[2].status, 'unsure');
  assert.equal(result.blocks[2].section, null);
});

test('a block after an unsure one is never merged under the unsure tag', () => {
  const lyrics = '[Verse]\nfirst line\n\n[Bridge]\nlost line\n\n[Verse 2]\nthird line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.1], ['third', 7.5]]));
  assert.equal(result.lyrics.includes('[Bridge]\nlost line\n\n[Verse 2]\nthird line'), true);
  assert.deepEqual(result.blocks.map(b => b.status), ['kept', 'unsure', 'unsure']);
});

test('two blocks sung in one section share one tag; words are untouched', () => {
  const lyrics = 'first line\n\n[Pre]\nsecond line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.1], ['second', 7.5]]));
  assert.equal(result.lyrics.startsWith('[Intro]\n\n[Verse]\nfirst line\n\nsecond line'), true);
  assert.deepEqual(result.blocks.map(b => b.status), ['renamed', 'merged']);
});

test('only chorus-type sections get copies, and a block sung where the score has no Vocal stays unsure', () => {
  // A verse the lyrics leave out stays empty-handed; the chorus is copied.
  const twoBlocks = '[Verse 1]\nfirst line\n\n[Chorus]\nsecond line';
  const copied = matchSectionsToScore(score, twoBlocks, words(twoBlocks, [['first', 6.2], ['second', 12.1]]));
  assert.equal(copied.lyrics, '[Intro]\n\n[Verse 1]\nfirst line\n\n[Chorus]\nsecond line\n\n[Chorus]\nsecond line\n\n[Outro]');
  assert.deepEqual(copied.filled.map(f => [f.section, f.copiedFrom]), [[1, null], [5, 2], [6, null]]);

  // "third" starts in the outro, which has no Vocal note: tag and place kept, flagged.
  const lyrics = '[Verse 1]\nfirst line\n\n[Chorus]\nsecond line\n\n[Verse 2]\nthird line';
  const result = matchSectionsToScore(score, lyrics, words(lyrics, [['first', 6.2], ['second', 12.1], ['third', 30.2]]));
  assert.deepEqual(result.blocks.map(b => b.status), ['kept', 'kept', 'unsure']);
  assert.equal(result.blocks[2].newTag, 'Verse 2');
  assert.equal(result.lyrics.includes('[Verse 2]\nthird line'), true);
  assert.equal(result.lyrics.includes('[Outro]\nthird line'), false);
  assert.equal(result.filled.some(f => f.copiedFrom === 1), false);
});
