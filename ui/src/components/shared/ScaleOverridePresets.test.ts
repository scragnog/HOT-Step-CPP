// ScaleOverridePresets.test.ts — a same-name import conflict is reported,
// never duplicated with a guessed keep-both. Run with the server's tsx:
//   (cd ui && npx tsx --test src/components/shared/ScaleOverridePresets.test.ts)
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadServerPresets, STORAGE_KEY, MIGRATED_FLAG } from './ScaleOverridePresets.js';

class MemoryStorage {
  private store = new Map<string, string>();
  getItem(key: string): string | null { return this.store.has(key) ? this.store.get(key)! : null; }
  setItem(key: string, value: string): void { this.store.set(key, value); }
  removeItem(key: string): void { this.store.delete(key); }
  clear(): void { this.store.clear(); }
}

function withFetch<T>(handler: (url: string, init?: RequestInit) => { status: number; body: unknown },
  run: () => Promise<T>): Promise<T> {
  const saved = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const { status, body } = handler(url, init);
    return new Response(JSON.stringify(body), { status });
  }) as typeof fetch;
  return run().finally(() => { globalThis.fetch = saved; });
}

function withStorage<T>(run: (storage: MemoryStorage) => Promise<T>): Promise<T> {
  const saved = (globalThis as { localStorage?: Storage }).localStorage;
  const storage = new MemoryStorage();
  (globalThis as { localStorage?: unknown }).localStorage = storage;
  return run(storage).finally(() => { (globalThis as { localStorage?: unknown }).localStorage = saved; });
}

const preset = (name: string) => ({ name, overallScale: 1, groupScales: { self_attn: 1, cross_attn: 1, mlp: 1, cond_embed: 1 } });

test('a same-name import conflict is reported, never duplicated with a guessed keep-both', async () => {
  await withStorage(async storage => {
    storage.setItem(STORAGE_KEY, JSON.stringify([preset('Lead')]));
    const importCalls: Array<Array<{ resolution?: string }>> = [];
    await withFetch((url, init) => {
      if (url.endsWith('/import')) {
        importCalls.push(JSON.parse(String(init?.body)).items);
        return { status: 200, body: { results: [{ storageKey: `${STORAGE_KEY}:Lead`, outcome: 'name-conflict', documentId: 'doc1', storedName: 'Lead' }] } };
      }
      return { status: 200, body: { documents: [] } };
    }, () => loadServerPresets());

    assert.equal(importCalls.length, 1);
    assert.equal('resolution' in importCalls[0]![0]!, false); // no hardcoded keep-both sent
    assert.equal(storage.getItem(MIGRATED_FLAG), null); // unresolved conflict leaves the flag unset
  });
});

test('an import with no conflicts commits the migrated flag', async () => {
  await withStorage(async storage => {
    storage.setItem(STORAGE_KEY, JSON.stringify([preset('Fresh')]));
    await withFetch(url => {
      if (url.endsWith('/import')) return { status: 200, body: { results: [{ storageKey: `${STORAGE_KEY}:Fresh`, outcome: 'imported', documentId: 'doc2', storedName: 'Fresh' }] } };
      return { status: 200, body: { documents: [] } };
    }, () => loadServerPresets());

    assert.equal(storage.getItem(MIGRATED_FLAG), '1');
  });
});

test('no legacy presets commits the migrated flag without calling import', async () => {
  await withStorage(async storage => {
    let importCalled = false;
    await withFetch(url => {
      if (url.endsWith('/import')) { importCalled = true; return { status: 200, body: { results: [] } }; }
      return { status: 200, body: { documents: [] } };
    }, () => loadServerPresets());

    assert.equal(importCalled, false);
    assert.equal(storage.getItem(MIGRATED_FLAG), '1');
  });
});
