// check-song-display.ts — the title/subtext display rules.
// Run: npx tsx ui/scripts/check-song-display.ts
//
// The model-tag stripper is a regex over names people typed by hand, so the
// interesting cases are the ones it must NOT touch.

import assert from 'node:assert';
import { stripModelTag, displayTitle, displaySubtext } from '../src/utils/songDisplay';
import type { Song } from '../src/types';

const song = (over: Partial<Song> & Record<string, unknown>) => over as unknown as Song;

// ── stripModelTag: tags that go ──
for (const [input, want] of [
  ['Ninety Minute Tape - Claude Opus', 'Ninety Minute Tape'],
  ['Ninety Minute Tape - Claude Opus 5', 'Ninety Minute Tape'],
  ['Sunny Delaney - Opus 5', 'Sunny Delaney'],
  ['This Is My Flatmate (Opus 5)', 'This Is My Flatmate'],
  ['Wasterstein [GPT-5]', 'Wasterstein'],
  ['Something - Gemini 2.5 Pro', 'Something'],
  ['Track — Sonnet 4.5', 'Track'],
] as const) {
  assert.equal(stripModelTag(input), want, `strip ${input}`);
}

// ── stripModelTag: names that stay ──
for (const input of [
  'Ninety Minute Tape',
  'Sixteen Tons - Live',
  'The Opus',                    // no separator
  'Opus 5',                      // stripping leaves nothing
  'Song - Part Two',
  'Hey - Grok About It',         // a family word, but what follows is not a version
  'Nothing - Novacane',
] as const) {
  assert.equal(stripModelTag(input), input, `left alone: ${input}`);
}

// ── displayTitle ──
assert.equal(
  displayTitle(song({ title: 'Ninety Minute Tape - Claude Opus', generationParams: { artist: 'PRESIDENT' } })),
  'PRESIDENT - Ninety Minute Tape',
  'artist prepended, tag stripped',
);
assert.equal(
  displayTitle(song({ title: 'PRESIDENT - Ninety Minute Tape', generationParams: { artist: 'PRESIDENT' } })),
  'PRESIDENT - Ninety Minute Tape',
  'artist not doubled',
);
assert.equal(
  displayTitle(song({ title: 'Ninety Minute Tape', artistName: 'PRESIDENT' })),
  'PRESIDENT - Ninety Minute Tape',
  'artistName is a source too',
);
assert.equal(
  displayTitle(song({ title: 'Ninety Minute Tape' })),
  'Ninety Minute Tape',
  'no artist, no change',
);
assert.equal(displayTitle(song({})), 'Untitled', 'missing title');

// ── displaySubtext ──
assert.equal(
  displaySubtext(song({ style: 'nu-metalcore, male vocals', generationParams: { subject: 'Clearing the garage' } })),
  'Clearing the garage',
  'subject beats caption',
);
assert.equal(
  displaySubtext(song({ style: 'nu-metalcore, male vocals' })),
  'nu-metalcore, male vocals',
  'falls back to style',
);
assert.equal(displaySubtext(song({ caption: 'a caption' })), 'a caption', 'then caption');
assert.equal(displaySubtext(song({})), '', 'nothing to show');

console.log('check-song-display: all assertions passed');
