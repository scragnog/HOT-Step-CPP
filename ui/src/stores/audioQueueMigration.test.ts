import test from 'node:test';
import assert from 'node:assert/strict';
import { exportQueueBackup, restoreQueueBackup, type QueueBackup, type QueueBackupStore } from './audioQueueMigration.js';

test('versioned localStorage and IndexedDB backup reads back and restores exactly', async () => {
  let legacy: string | null = JSON.stringify({ items: [{ id: 'old', jobId: 'job-old' }] });
  let indexed: unknown = { items: [{ id: 'new', jobId: 'job-new' }] };
  const saved = new Map<string, QueueBackup>();
  const store: QueueBackupStore = {
    readLegacy: () => legacy,
    writeLegacy: value => { legacy = value; },
    readIndexedDb: async () => indexed,
    writeIndexedDb: async value => { indexed = value; },
    saveBackup: async backup => { saved.set(backup.id, structuredClone(backup)); },
    readBackup: async id => saved.get(id) ?? null,
  };
  const backup = await exportQueueBackup(store, 'e5726199-d56b-4528-87de-37af04662579');
  assert.deepEqual(backup.itemIds.sort(), ['new', 'old']);
  assert.deepEqual(backup.jobIds.sort(), ['job-new', 'job-old']);
  legacy = null; indexed = null;
  await restoreQueueBackup(store, backup);
  assert.equal(legacy, backup.legacyRaw);
  assert.deepEqual(indexed, backup.indexedDb);
});
