// lyricTimings.test.ts — the editor's save path, including the file it creates.
//
// A track with a line-level `.lrc` and no word sidecar is opened by the editor
// with its lines split into words, and the first save is what puts a
// `.lyrics.json` beside the audio. That creation path is the one worth pinning:
// it must still refuse a track with nothing to place against, and it must not
// overwrite a sidecar that appeared while the editor was open.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { saveLyricTimings } from './lyricTimings.js';

const lines = [{
  start: 1,
  end: 1.5,
  words: [{ word: 'a', start: 1, end: 1.2 }, { word: 'b', start: 1.2, end: 1.5 }],
}];

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-timings-'));
}

test('the first save creates the word sidecar for a track with a line-level .lrc', () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'track.lrc'), '[00:01.00]a b\n');

  const saved = saveLyricTimings(dir, '/audio/track.wav', lines, { title: 'Track' }, 0);
  assert.equal(saved.ok, true);

  const json = JSON.parse(fs.readFileSync(path.join(dir, 'track.lyrics.json'), 'utf-8'));
  assert.equal(json.revision, 1);
  assert.equal(json.method, 'lrc-lines');
  assert.equal(json.editedByHand, true);
  assert.deepEqual(json.lines.map((l: any) => l.words.length), [2]);
  assert.match(fs.readFileSync(path.join(dir, 'track.lrc'), 'utf-8'), /\[00:01\.00\]a b/);
});

test('a track with neither file is still refused', () => {
  const dir = tmp();
  assert.deepEqual(saveLyricTimings(dir, '/audio/none.wav', lines), { ok: false, reason: 'no-sidecar' });
});

test('a sidecar that appeared after the editor opened is stale, not overwritten', () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'moved.lrc'), '[00:01.00]a b\n');
  fs.writeFileSync(path.join(dir, 'moved.lyrics.json'), JSON.stringify({ version: 1, revision: 3, lines: [] }));

  assert.deepEqual(
    saveLyricTimings(dir, '/audio/moved.wav', lines, {}, 0),
    { ok: false, reason: 'stale', revision: 3 },
  );
});

test('the created file is editable again at the revision it returned', () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'next.lrc'), '[00:01.00]a b\n');

  const first = saveLyricTimings(dir, '/audio/next.wav', lines, {}, 0);
  assert.equal(first.ok && first.revision, 1);
  const second = saveLyricTimings(dir, '/audio/next.wav', lines, {}, 1);
  assert.equal(second.ok && second.revision, 2);
});

test('a layer survives the editor’s save, word and line alike', () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'harmony.lrc'), '[00:01.00]a b\n');

  const saved = saveLyricTimings(dir, '/audio/harmony.wav', [{
    start: 1, end: 1.5, layer: 'backing',
    words: [
      { word: 'a', start: 1, end: 1.2, layer: 'backing' },
      { word: 'b', start: 1.2, end: 1.5, layer: 'backing' },
    ],
  }], {}, 0);
  assert.equal(saved.ok, true);

  const json = JSON.parse(fs.readFileSync(path.join(dir, 'harmony.lyrics.json'), 'utf-8'));
  assert.equal(json.lines[0].layer, 'backing');
  assert.deepEqual(json.lines[0].words.map((w: any) => w.layer), ['backing', 'backing']);
});
