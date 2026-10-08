// AiContinuePresetModal.test.ts — template hydration/save serialization,
// preset import conflicts, and a pending-create deleted before it resolves.
// Run with the server's tsx:
//   (cd ui && npx tsx --test src/components/storm/AiContinuePresetModal.test.ts)
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  hydrateTemplateFromServer, queueTemplateSave, loadServerPresets, resolvePresetCreate,
  _resetTemplateStateForTests, TEMPLATE_KEY, USER_STYLE_KEY, MIGRATED_FLAG,
} from './AiContinuePresetModal.js';

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

test('a second browser hydrates the server template instead of falling back to the default', async () => {
  await withStorage(async storage => {
    _resetTemplateStateForTests();
    const result = await withFetch(url => {
      if (url.endsWith('/settings/ai-continue-template')) return { status: 200, body: { document: { id: 'doc-t', revision: 3, body: { template: 'saved on another browser' } } } };
      return { status: 404, body: { error: `unexpected call to ${url}` } };
    }, () => hydrateTemplateFromServer());

    assert.equal(result, 'saved on another browser');
    assert.equal(storage.getItem(TEMPLATE_KEY), 'saved on another browser'); // mirrored for the sync readers
  });
});

test('two quick template edits serialize: the second waits for the first instead of racing a stale revision', async () => {
  await withStorage(async () => {
    _resetTemplateStateForTests();
    const puts: Array<{ expectedRevision?: number }> = [];
    let revision = 0;
    const errors: string[] = [];
    await withFetch((url, init) => {
      if (!url.endsWith('/settings/ai-continue-template')) return { status: 404, body: { error: `unexpected call to ${url}` } };
      if (!init || init.method === 'GET') return { status: 200, body: { document: null } };
      const parsed = JSON.parse(String(init.body));
      puts.push(parsed);
      revision += 1;
      return { status: 200, body: { document: { id: 'doc-t', revision, body: parsed.body } } };
    }, async () => {
      queueTemplateSave('first', m => errors.push(m));
      queueTemplateSave('second', m => errors.push(m));
      await new Promise(r => setTimeout(r, 20));
    });

    assert.deepEqual(errors, []);
    assert.equal(puts.length, 2);
    assert.equal(puts[0]!.expectedRevision, undefined); // first write: no document yet
    assert.equal(puts[1]!.expectedRevision, 1); // waited for the first write's revision instead of also sending stale/undefined
  });
});

test('a same-name preset import conflict is reported, never duplicated with a guessed keep-both', async () => {
  await withStorage(async storage => {
    storage.setItem(USER_STYLE_KEY, JSON.stringify([{ id: 'u1', label: 'Lead', value: 'go darker' }]));
    const importCalls: Array<Array<{ resolution?: string }>> = [];
    await withFetch((url, init) => {
      if (url.endsWith('/import')) {
        importCalls.push(JSON.parse(String(init?.body)).items);
        return { status: 200, body: { results: [{ storageKey: `${USER_STYLE_KEY}:u1`, outcome: 'name-conflict', documentId: 'doc1', storedName: 'Lead' }] } };
      }
      return { status: 200, body: { documents: [] } };
    }, () => loadServerPresets('style'));

    assert.equal(importCalls.length, 1);
    assert.equal('resolution' in importCalls[0]![0]!, false); // no hardcoded keep-both sent
    assert.equal(storage.getItem(MIGRATED_FLAG.style), null); // unresolved conflict leaves the flag unset
  });
});

test('deleting a pending create removes the eventual server document instead of letting it reappear', async () => {
  const removed: Array<{ id: string; revision: number }> = [];
  const result = await resolvePresetCreate(
    async () => ({ document: { id: 'doc-new', revision: 1 } }),
    async (id, revision) => { removed.push({ id, revision }); },
    () => true, // the row was deleted while the create was still in flight
  );
  assert.equal(result, 'deleted');
  assert.deepEqual(removed, [{ id: 'doc-new', revision: 1 }]);
});

test('a create that was not deleted meanwhile settles normally', async () => {
  const result = await resolvePresetCreate(
    async () => ({ document: { id: 'doc-new', revision: 1 } }),
    async () => { throw new Error('should not be called'); },
    () => false,
  );
  assert.deepEqual(result, { id: 'doc-new', revision: 1 });
});
