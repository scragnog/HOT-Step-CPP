import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mock } from 'node:test';
import fs from 'node:fs';
import type Database from 'better-sqlite3';
import { exportRequest } from './contracts.js';
import { resolveExports } from './operations.js';

test('bulk export reports missing and unavailable items independently', () => {
  mock.method(fs, 'existsSync', () => true);
  const db = {
    prepare: () => ({ get: (id: string, userId: string) =>
      id === 'own' && userId === 'owner' ? { id, title: 'Track', audio_url: '/audio/raw.wav' } : undefined }),
  } as unknown as Database.Database;
  const items = resolveExports(db, 'owner', exportRequest.parse({
    items: [{ songId: 'missing' }, { songId: 'own', variant: 'mastered' }, { songId: 'own', variant: 'original' }],
    format: 'wav',
  }));
  assert.equal(items.length, 3);
  assert.deepEqual(items.map(item => item.index), [0, 1, 2]);
  assert.equal(items[0].error, 'Song not found');
  assert.equal(items[1].variant, 'original');
  assert.equal(items[2].songId, 'own');
  mock.restoreAll();
});

test('both resolves on the server and retains queue-only audio overrides', () => {
  mock.method(fs, 'existsSync', () => true);
  const db = { prepare: () => ({ get: (id: string) => id === 'mastered'
    ? { id, title: 'Track', audio_url: '/audio/raw.wav', mastered_audio_url: '/audio/master.wav' }
    : undefined }) } as unknown as Database.Database;
  const items = resolveExports(db, 'owner', exportRequest.parse({
    items: [{ songId: 'mastered' }, { songId: 'queue', audioUrl: '/audio/queue.wav', srcUrl: '/audio/noadapter.wav', variant: 'noadapter' }],
    format: 'flac', downloadVersion: 'both',
  }));
  assert.deepEqual(items.map(item => [item.index, item.variant]), [[0, 'original'], [0, 'mastered'], [1, 'noadapter']]);
  assert.match(items[2].url!, /srcUrl=%2Faudio%2Fnoadapter.wav/);
  mock.restoreAll();
});
