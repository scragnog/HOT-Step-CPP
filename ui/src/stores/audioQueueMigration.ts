/** Browser queue backup and rollback data. This module has no browser globals
 * so the exact readback/restore path can be tested with an in-memory store. */

export interface QueueBackup {
  version: 1;
  id: string;
  exportedAt: number;
  legacyRaw: string | null;
  indexedDb: unknown | null;
  itemIds: string[];
  jobIds: string[];
}

export interface QueueBackupStore {
  readLegacy(): string | null;
  writeLegacy(raw: string | null): void;
  readIndexedDb(): Promise<unknown | null>;
  writeIndexedDb(value: unknown | null): Promise<void>;
  saveBackup(backup: QueueBackup): Promise<void>;
  readBackup(id: string): Promise<QueueBackup | null>;
}

function itemsOf(value: unknown): Record<string, unknown>[] {
  if (!value || typeof value !== 'object') return [];
  const items = (value as Record<string, unknown>).items;
  return Array.isArray(items) ? items.filter(item => item && typeof item === 'object') : [];
}

export function parseQueueBackup(value: unknown): QueueBackup {
  if (!value || typeof value !== 'object') throw new Error('Queue backup is not an object');
  const backup = value as QueueBackup;
  if (backup.version !== 1 || typeof backup.id !== 'string' || !backup.id
    || typeof backup.exportedAt !== 'number'
    || (backup.legacyRaw !== null && typeof backup.legacyRaw !== 'string')
    || !Array.isArray(backup.itemIds) || !backup.itemIds.every(id => typeof id === 'string')
    || !Array.isArray(backup.jobIds) || !backup.jobIds.every(id => typeof id === 'string')) {
    throw new Error('Invalid queue backup version or fields');
  }
  return backup;
}

export async function exportQueueBackup(store: QueueBackupStore, id = crypto.randomUUID()): Promise<QueueBackup> {
  const legacyRaw = store.readLegacy();
  const indexedDb = await store.readIndexedDb();
  let legacy: unknown = null;
  if (legacyRaw) {
    try { legacy = JSON.parse(legacyRaw); }
    catch { throw new Error('Legacy localStorage queue is invalid JSON; no import was attempted'); }
  }
  const entries = [...itemsOf(legacy), ...itemsOf(indexedDb)];
  const backup: QueueBackup = {
    version: 1, id, exportedAt: Date.now(), legacyRaw,
    indexedDb: indexedDb ?? null,
    itemIds: [...new Set(entries.map(item => item.id).filter((id): id is string => typeof id === 'string'))],
    jobIds: [...new Set(entries.map(item => item.jobId).filter((id): id is string => typeof id === 'string'))],
  };
  await store.saveBackup(backup);
  const readback = parseQueueBackup(await store.readBackup(backup.id));
  if (JSON.stringify(readback) !== JSON.stringify(backup)) throw new Error('Queue backup readback differs from export');
  return backup;
}

export async function restoreQueueBackup(store: QueueBackupStore, raw: unknown): Promise<QueueBackup> {
  const backup = parseQueueBackup(raw);
  store.writeLegacy(backup.legacyRaw);
  await store.writeIndexedDb(backup.indexedDb);
  if (store.readLegacy() !== backup.legacyRaw
    || JSON.stringify(await store.readIndexedDb()) !== JSON.stringify(backup.indexedDb)) {
    throw new Error('Queue backup restore did not read back exactly');
  }
  return backup;
}
