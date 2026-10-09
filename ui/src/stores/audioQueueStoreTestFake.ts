// audioQueueStoreTestFake.ts — in-memory indexedDB and /api/audio-queue
// fakes for audioGenQueueStore's own migration/rollback tests. Not imported
// by the app. Modeled on services/preferencesTestFake.ts (same idea, a
// different route surface); reuses its MemoryStorage.

export { MemoryStorage } from '../services/preferencesTestFake';

interface Call { method: string; path: string; body: any }

/** Minimal fake of the subset of IndexedDB that _openDB/_idbGet/_idbSet use:
 *  one object store, get/put by key. Each call's work is deferred a
 *  microtask, matching real IndexedDB's async callback shape closely enough
 *  that the store's `onsuccess`/`oncomplete` wiring (assigned synchronously
 *  right after the call that schedules them) sees them in time. */
export function createFakeIndexedDB(): { indexedDB: IDBFactory; data: Map<string, unknown> } {
  const data = new Map<string, unknown>();
  const indexedDB = {
    open(_name: string, _version?: number) {
      const req: any = {};
      queueMicrotask(() => {
        const db = {
          objectStoreNames: { contains: () => true },
          createObjectStore() { /* no-op: `data` already stands in for the one store */ },
          transaction(_storeName: string, _mode: string) {
            const tx: any = {};
            tx.objectStore = () => ({
              get(key: string) {
                const r: any = {};
                queueMicrotask(() => { r.result = data.get(key); r.onsuccess?.(); });
                return r;
              },
              put(value: unknown, key: string) {
                const r: any = {};
                queueMicrotask(() => { data.set(key, value); r.onsuccess?.(); tx.oncomplete?.(); });
                return r;
              },
            });
            return tx;
          },
        };
        req.result = db;
        req.onupgradeneeded?.();
        req.onsuccess?.();
      });
      return req;
    },
  };
  return { indexedDB: indexedDB as unknown as IDBFactory, data };
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status });
}

/** Fakes the /api/audio-queue routes the migration/rollback paths use.
 *  Mirrors the real server's two idempotency guarantees
 *  (server/src/services/audioQueue/intentQueue.ts): an exact backupId replay
 *  returns the cached receipt verbatim, and a never-seen backupId still
 *  dedupes per item by legacyId. */
export class AudioQueueApiFake {
  calls: Call[] = [];
  rollbackCalls = 0;
  /** Items the server holds, for rollback-export, GET /items and cancel. */
  serverItems: Array<{ id: string; status: string; jobId?: string | null; meta?: Record<string, unknown>; request?: Record<string, unknown>; createdAt?: number; error?: string | null }> = [];
  paused = false;
  private receiptsByBackupId = new Map<string, any>();
  private importedLegacyIds = new Set<string>();
  private failures: Array<(c: Call) => boolean> = [];
  private persistentFailures: Array<(c: Call) => boolean> = [];

  /** Fail the next matching request, regardless of what the real server
   *  would have answered — models a dropped response after the server has
   *  already committed the write. */
  failNext(match: (c: Call) => boolean): void { this.failures.push(match); }

  /** Fail every matching request from here on — a connection that stays down. */
  failAlways(match: (c: Call) => boolean): void { this.persistentFailures.push(match); }

  install(): () => void {
    const saved = globalThis.fetch;
    globalThis.fetch = (async (url: string, init?: RequestInit) => {
      const u = new URL(url, 'http://local');
      const call: Call = { method: init?.method ?? 'GET', path: u.pathname, body: init?.body ? JSON.parse(String(init.body)) : undefined };
      this.calls.push(call);
      const fi = this.failures.findIndex(f => f(call));
      if (fi >= 0) { this.failures.splice(fi, 1); return json(500, { error: 'Injected failure' }); }
      if (this.persistentFailures.some(f => f(call))) return json(500, { error: 'Injected failure' });
      return this.answer(call);
    }) as typeof fetch;
    return () => { globalThis.fetch = saved; };
  }

  private answer(c: Call): Response {
    if (c.path === '/api/audio-queue/items' && c.method === 'GET') return json(200, { items: this.serverItems });
    if (c.path === '/api/audio-queue/migration/import') {
      const { backupId, items } = c.body as { backupId: string; items: { legacyId: string; status: string }[] };
      const cached = this.receiptsByBackupId.get(backupId);
      if (cached) return json(200, cached);
      let imported = 0, existing = 0;
      const resultItems = items.map(item => {
        if (this.importedLegacyIds.has(item.legacyId)) { existing++; return { legacyId: item.legacyId, itemId: `srv-${item.legacyId}`, status: item.status }; }
        this.importedLegacyIds.add(item.legacyId);
        imported++;
        return { legacyId: item.legacyId, itemId: `srv-${item.legacyId}`, status: item.status === 'pending' ? 'pending' : item.status };
      });
      const receipt = { backupId, imported, existing, items: resultItems };
      this.receiptsByBackupId.set(backupId, receipt);
      return json(200, receipt);
    }
    if (c.path === '/api/audio-queue/migration/rollback-export') {
      this.rollbackCalls++;
      // Same order as the real route (routes/audioQueue.ts): pause first, then
      // refuse while a submission is in flight.
      this.paused = true;
      if (this.serverItems.some(item => item.status === 'submitting')) {
        return json(409, { error: 'Submission still in flight; retry export when it settles' });
      }
      return json(200, { version: 1, exportedAt: Date.now(), state: {}, items: structuredClone(this.serverItems) });
    }
    const cancel = c.path.match(/^\/api\/audio-queue\/items\/([^/]+)\/cancel$/);
    if (cancel) {
      const item = this.serverItems.find(i => i.id === decodeURIComponent(cancel[1]));
      if (!item) return json(404, { error: 'Queue item not found' });
      if (item.status !== 'pending' && item.status !== 'held') return json(409, { error: `Queue item is already ${item.status}` });
      item.status = 'cancelled';
      return json(200, { item });
    }
    return json(404, { error: `No route ${c.method} ${c.path}` });
  }
}
