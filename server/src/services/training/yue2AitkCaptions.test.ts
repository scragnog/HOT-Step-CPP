import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { jointCaptionTracks } from './yue2AitkCaptions.js';
import type { Yue2AitkRunRecord } from './yue2AitkRuns.js';

test('joint caption picker returns the exact prepared style, including its trigger', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-joint-captions-'));
  try {
    const dataset = path.join(dir, 'dataset.json');
    const legacyManifest = path.join(dir, 'legacy.json');
    fs.writeFileSync(dataset, JSON.stringify({ items: [{ id: 'song.flac',
      style: 'artist_album, in the style of artist_album. Fast guitars.' }] }));
    fs.writeFileSync(legacyManifest, JSON.stringify({ sources: [{ name: 'song.flac', bpm: '120', key: 'E' }] }));
    const run = { options: { dataset, preparation: { legacyManifest } } } as unknown as Yue2AitkRunRecord;
    assert.deepEqual(jointCaptionTracks(run), [{ name: 'song.flac',
      caption: 'artist_album, in the style of artist_album. Fast guitars.',
      styled: 'artist_album, in the style of artist_album. Fast guitars.',
      genre: '', bpm: '120', key: 'E' }]);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});
