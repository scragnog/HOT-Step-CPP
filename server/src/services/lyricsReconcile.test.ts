// lyricsReconcile.test.ts — where a second voice enters the sidecar.
//
// The reconcile pass is the only place the pipeline decides which line a heard
// word belongs to. A harmony that starts inside the lead's held note is not the
// lead's next syllable, so it has to become its own line there; nothing
// downstream can recover the lead line's bounds once the word has stretched
// them.

import test from 'node:test';
import assert from 'node:assert/strict';
import { reconcileLyrics } from './lyricsReconcile.js';

const heard = (word: string, start: number, end: number) => ({ word, start, end, probability: 0.9 });

/** A word as the decoder reports it when it gives no probability at all. */
const unmeasured = (word: string, start: number, end: number) => ({ word, start, end });

function run(words: Array<{ word: string; start: number; end: number; probability?: number }>) {
  return reconcileLyrics(
    { segments: [{ words }] } as any,
    'hold the line now',
    'test-model',
    false,
  );
}

test('a word heard inside the lead’s tail becomes its own backing line', () => {
  // The harmony starts 0.7 s before the lead's held note ends: a second voice,
  // not the next word. The lead line must keep 15.5 as its end.
  const result = run([
    heard('hold', 13.5, 13.9), heard('the', 13.9, 14.3), heard('line', 14.3, 15.5),
    heard('ooh', 14.8, 15.2),
    heard('now', 16.0, 16.4),
  ]);

  assert.equal(result.lines.length, 3);
  assert.equal(result.lines[0].text, 'hold the line');
  assert.equal(result.lines[0].end, 15.5, 'the lead line is not stretched by the harmony');
  assert.equal(result.lines[0].layer, undefined, 'an untagged line is the lead');
  assert.equal(result.lines[1].layer, 'backing');
  assert.deepEqual(result.lines[1].words.map(w => w.word), ['ooh']);
  assert.equal(result.lines[1].words[0].layer, 'backing');
  assert.equal(result.lines[2].text, 'now', 'the lead resumes in its own line');
});

test('a word Whisper gave no probability is left unmeasured', () => {
  // `?? 1` used to turn "the decoder said nothing" into "certainly right",
  // which is the one value the snapping gate must never read as confidence.
  const result = run([
    unmeasured('hold', 13.5, 13.9), unmeasured('the', 13.9, 14.3), unmeasured('line', 14.3, 14.7),
  ]);
  const words = result.lines.flatMap(line => line.words);

  assert.equal(words.length, 3);
  assert.equal(words[0].confidence, undefined);

  const known = run([heard('hold', 13.5, 13.9), heard('the', 13.9, 14.3)]);
  assert.equal(known.lines[0].words[0].confidence, 0.9, 'a reported probability is kept as it was heard');
});

test('an ad-lib that follows the lead stays in the lead’s line', () => {
  // 15.6 starts after the lead ended, so it is the same singer adding a word:
  // no layer, no split, exactly the behaviour before layers existed.
  const result = run([
    heard('hold', 13.5, 13.9), heard('the', 13.9, 14.3), heard('line', 14.3, 15.5),
    heard('ooh', 15.6, 16.0),
    heard('now', 16.2, 16.6),
  ]);

  assert.equal(result.lines.length, 1);
  assert.equal(result.lines[0].layer, undefined);
  assert.equal(result.lines[0].words.length, 5);
  assert.equal(result.lines[0].words[3].layer, undefined);
});
