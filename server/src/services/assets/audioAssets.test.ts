import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import Database from 'better-sqlite3';
import { recordAudioAsset, resolveAudioAsset } from './audioAssets.js';

test('an uploaded asset resolves only for its owner and only while its file exists', () => {
  const db = new Database(':memory:');
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'assets-'));
  fs.mkdirSync(path.join(dataDir, 'references'));
  const file = path.join(dataDir, 'references', 'abc.mp3');
  fs.writeFileSync(file, 'x');
  try {
    const a = recordAudioAsset(db, { userId: 'u', url: '/references/abc.mp3', filename: 'song.mp3', size: 1, sha256: 'h' });
    const r = resolveAudioAsset(db, a.id, 'u', dataDir);
    assert.equal(r.url, '/references/abc.mp3');
    assert.equal(r.path, file);
    assert.equal(r.filename, 'song.mp3');
    assert.throws(() => resolveAudioAsset(db, a.id, 'other', dataDir), (e: any) => e.status === 404);
    assert.throws(() => resolveAudioAsset(db, 'missing', 'u', dataDir), (e: any) => e.status === 404);
    const anon = recordAudioAsset(db, { userId: null, url: '/references/abc.mp3', filename: 's', size: 1, sha256: 'h' });
    assert.throws(() => resolveAudioAsset(db, anon.id, 'u', dataDir), (e: any) => e.status === 404);
    fs.rmSync(file);
    assert.throws(() => resolveAudioAsset(db, a.id, 'u', dataDir), /no file any more/);
  } finally {
    fs.rmSync(dataDir, { recursive: true, force: true });
  }
});
