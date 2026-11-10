import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveDuration } from './estimateDuration';

// "Use LLM duration" off: the lyric estimate is in force.
(globalThis as { localStorage?: unknown }).localStorage = { getItem: () => 'false' };

test('an instrumental track keeps its stored duration when the lyric estimate is on', () => {
  assert.equal(resolveDuration(145, '[Instrumental]', 106), 145);
  assert.equal(resolveDuration(undefined, ' [Instrumental]\n', 106), 180);
});

test('a vocal track is still estimated from its lyrics', () => {
  const lyrics = ['[Verse]', ...Array(30).fill('a line'), '[Chorus]', ...Array(10).fill('a line')].join('\n');
  assert.equal(resolveDuration(60, lyrics, 120), 148);
});
