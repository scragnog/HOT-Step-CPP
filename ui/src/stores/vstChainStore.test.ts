// vstChainStore.test.ts — preset migration conflicts and failed save/delete
// CRUD, against a mocked /api/preferences. Run with the server's tsx:
//   (cd ui && npx tsx --test src/stores/vstChainStore.test.ts)
import test from 'node:test';
import assert from 'node:assert/strict';
import { importLegacyPresetsOnce, useVstChainStore, PRESETS_KEY, MIGRATED_FLAG } from './vstChainStore.js';

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

const entry = (uid: string) => ({ uid, name: `Plugin ${uid}`, vendor: 'V', path: `/p/${uid}`, enabled: true, statePath: `/s/${uid}` });

test('a same-name import conflict is reported, never duplicated with a guessed keep-both', async () => {
  await withStorage(async storage => {
    storage.setItem(PRESETS_KEY, JSON.stringify({ Lead: [entry('a')] }));
    const importCalls: Array<Array<{ resolution?: string }>> = [];
    await withFetch((url, init) => {
      if (url.endsWith('/import')) {
        importCalls.push(JSON.parse(String(init?.body)).items);
        return { status: 200, body: { results: [{ storageKey: `${PRESETS_KEY}:Lead`, outcome: 'name-conflict', documentId: 'doc1', storedName: 'Lead' }] } };
      }
      return { status: 404, body: { error: `unexpected call to ${url}` } };
    }, () => importLegacyPresetsOnce());

    assert.equal(importCalls.length, 1);
    assert.equal('resolution' in importCalls[0]![0]!, false); // no hardcoded keep-both sent
    assert.equal(storage.getItem(MIGRATED_FLAG), null); // unresolved conflict leaves the flag unset
  });
});

test('an import with no conflicts commits the migrated flag', async () => {
  await withStorage(async storage => {
    storage.setItem(PRESETS_KEY, JSON.stringify({ Fresh: [entry('b')] }));
    await withFetch((url) => {
      if (url.endsWith('/import')) return { status: 200, body: { results: [{ storageKey: `${PRESETS_KEY}:Fresh`, outcome: 'imported', documentId: 'doc2', storedName: 'Fresh' }] } };
      return { status: 404, body: { error: `unexpected call to ${url}` } };
    }, () => importLegacyPresetsOnce());

    assert.equal(storage.getItem(MIGRATED_FLAG), '1');
  });
});

test('a failed save reverts the optimistic preset and reports it, instead of looking saved', async () => {
  await withStorage(async () => {
    useVstChainStore.setState({ chain: [entry('c')], presets: {}, presetDocs: {}, presetError: null });
    await withFetch(() => ({ status: 409, body: { error: 'Preset changed elsewhere' } }),
      () => useVstChainStore.getState().savePreset('My Chain'));

    const state = useVstChainStore.getState();
    assert.equal(state.presets['My Chain'], undefined); // never looked saved after the write actually failed
    assert.match(state.presetError ?? '', /My Chain/);
  });
});

test('a failed delete restores the preset instead of leaving it hidden', async () => {
  await withStorage(async () => {
    const existing = [entry('d')];
    useVstChainStore.setState({
      presets: { Keeper: existing }, presetDocs: { Keeper: { id: 'doc3', revision: 2 } }, presetError: null,
    });
    await withFetch(() => ({ status: 409, body: { error: 'Preset changed elsewhere', currentRevision: 3 } }),
      () => useVstChainStore.getState().deletePreset('Keeper'));

    const state = useVstChainStore.getState();
    assert.deepEqual(state.presets['Keeper'], existing); // restored, not silently dropped
    assert.match(state.presetError ?? '', /Keeper/);
  });
});
