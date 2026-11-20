// audioGenQueueStore.migration.test.ts — the real store logic behind queue
// ownership: the one-shot import of a v1.3.x browser queue, from every state a
// profile can load in (never decided, a v1.4.x rollback's 'browser', a stuck
// 'migrating'), and a migration interrupted by reload. Exercises the actual exported
// functions (not just the pure decision table in audioQueueOwnerDecision.ts)
// against fake IndexedDB/localStorage/fetch.
//
// audioGenQueueStore.ts can't be imported under plain node:test without two
// things first: a `window` (it calls addEventListener at module scope) and a
// CSS-import stub (its module graph pulls in a component that imports a
// stylesheet, which Vite handles but Node cannot) — hence the loader
// registration and the dynamic import below, both before the first test.
import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { createFakeIndexedDB, AudioQueueApiFake, MemoryStorage } from './audioQueueStoreTestFake.js';
import type { AudioQueueItem } from './audioGenQueueStore.js';

register('./cssStubLoader.mjs', import.meta.url);

// downloadQueueExport() (migrateAudioQueue) schedules a
// bare 60s setTimeout to revoke its object URL — harmless in a browser tab,
// but under node:test it is a real, un-unref'd handle that keeps the process
// alive for a full minute after every test that downloads anything. unref
// it here; production code never sees this file.
const realSetTimeout = globalThis.setTimeout;
(globalThis as any).setTimeout = ((fn: (...a: any[]) => void, ms?: number, ...args: any[]) => {
  const handle = realSetTimeout(fn, ms, ...args);
  (handle as unknown as { unref?: () => void })?.unref?.();
  return handle;
}) as typeof setTimeout;

// The shared axios client sniffs `window` to pick its browser platform, which
// reads window.location — the stub must provide it or module load crashes.
(globalThis as any).window = { addEventListener() {}, removeEventListener() {}, location: { href: 'http://localhost' } };
(globalThis as any).document = { createElement: () => ({ click() {}, href: '', download: '' }) };
(globalThis.URL as any).createObjectURL = () => 'blob:fake';
(globalThis.URL as any).revokeObjectURL = () => {};
const initialIdb = createFakeIndexedDB();
(globalThis as any).localStorage = new MemoryStorage();
(globalThis as any).indexedDB = initialIdb.indexedDB;

const {
  retryQueueMigration, resumeQueue,
  ensureQueueOwnerDecided, getAudioQueueOwner, _resetAudioQueueForTests,
  OWNER_KEY, BACKUP_KEY, PENDING_IMPORT_KEY,
} = await import('./audioGenQueueStore.js') as any;

const fixtureItem = (id: string): AudioQueueItem => ({
  id,
  generation: { id: 1, profile_id: 1, provider: 'create', model: 'test', lyrics: '' } as any,
  artistId: 1, artistName: 'Test', preset: null, profileId: 1, lyricsSetId: 1,
  globalParams: { prompt: 'test' },
  status: 'pending',
});

async function withFakeQueue(
  items: AudioQueueItem[],
  run: (ctx: { storage: MemoryStorage; idbData: Map<string, unknown>; api: AudioQueueApiFake }) => Promise<void>,
): Promise<void> {
  const storage = new MemoryStorage();
  const { indexedDB, data: idbData } = createFakeIndexedDB();
  const api = new AudioQueueApiFake();
  const g = globalThis as any;
  const savedStorage = g.localStorage, savedIdb = g.indexedDB;
  g.localStorage = storage; g.indexedDB = indexedDB;
  const uninstallFetch = api.install();
  _resetAudioQueueForTests(items);
  try { await run({ storage, idbData, api }); }
  finally {
    _resetAudioQueueForTests(); // clears any server-projection/persist timer this run started
    uninstallFetch(); g.localStorage = savedStorage; g.indexedDB = savedIdb;
  }
}

/** The v1.3.x queue as the browser stored it, which the import reads. */
const seedBrowserQueue = (idbData: Map<string, unknown>, items: AudioQueueItem[]) =>
  idbData.set('state', { items, completionCounter: 0 });
const importCalls = (api: AudioQueueApiFake) => api.calls.filter((c: any) => c.path === '/api/audio-queue/migration/import');
const importedIds = (api: AudioQueueApiFake) => importCalls(api).flatMap((c: any) => c.body.items.map((i: any) => i.legacyId));

test('a never-decided profile with nothing pending becomes server-owned silently, no network call', async () => {
  await withFakeQueue([], async ({ api }) => {
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.equal(api.calls.length, 0);
  });
});

test('start state: never decided, items pending — imported once, silently, and server-owned', async () => {
  const items = [fixtureItem('a'), fixtureItem('b')];
  await withFakeQueue(items, async ({ idbData, api }) => {
    seedBrowserQueue(idbData, items);
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.equal(importCalls(api).length, 1);
    assert.deepEqual(importedIds(api), ['a', 'b']);
    assert.ok(idbData.get('state'), 'the browser copy is kept');
  });
});

test("start state: 'browser' (a v1.4.x rollback), items pending — imported, not left for an executor that is gone", async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'browser');
    seedBrowserQueue(idbData, items);
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.deepEqual(importedIds(api), ['a']);
  });
});

test("'browser' after a rollback: items the first import landed and the server still holds are not sent again", async () => {
  const items = [fixtureItem('old'), fixtureItem('new')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'browser');
    seedBrowserQueue(idbData, items);
    // The pre-rollback import's receipt: 'old' is on the server already, as
    // the cancelled row the rollback left, which Retry can reach.
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-0', items: [{ legacyId: 'old', itemId: 'srv-old', status: 'cancelled' }] });
    api.serverItems = [{ id: 'srv-old', status: 'cancelled', meta: { legacyId: 'old' }, request: { title: 'old' }, createdAt: 1 }];
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.deepEqual(importedIds(api), ['new']);
  });
});

test("'browser' after a rollback: a receipt item whose server row was dismissed since is imported again", async () => {
  const items = [fixtureItem('old'), fixtureItem('new')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'browser');
    seedBrowserQueue(idbData, items);
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-0', items: [{ legacyId: 'old', itemId: 'srv-old', status: 'cancelled' }] });
    api.serverItems = []; // the cancelled row was cleared after the rollback
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.deepEqual(importedIds(api), ['old', 'new'], 'the still-pending browser item is not lost');
  });
});

test("a finished item is never replayed, receipt or not", async () => {
  const done = { ...fixtureItem('done'), status: 'succeeded' as const, songId: 's1' };
  const items = [done, fixtureItem('new')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'browser');
    seedBrowserQueue(idbData, items);
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-0', items: [{ legacyId: 'done', itemId: 'srv-done', status: 'succeeded' }] });
    await ensureQueueOwnerDecided('tok');
    assert.deepEqual(importedIds(api), ['new']);
  });
});

test("start state: 'migrating' — the load retries the import, even with a v1.4.x rollback marker left over", async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'migrating');
    storage.setItem('lireek-audio-queue-rollback-requested-v1', 'bk-x');
    // No persisted payload: the earlier attempt died before its POST, so the
    // retry rebuilds it from the browser copy.
    seedBrowserQueue(idbData, items);
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.deepEqual(importedIds(api), ['a']);
  });
});

test('import landed before an interrupted reload: a failed replay stays fenced, and Resume does not resubmit', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    const payload = {
      backupId: 'bk-1', choice: 'resume',
      items: [{ legacyId: 'a', request: { prompt: 'test' }, meta: { view: items[0] }, status: 'pending' }],
    };
    // The original attempt's POST actually reached the server...
    await fetch('/api/audio-queue/migration/import', { method: 'POST', body: JSON.stringify(payload) });
    assert.equal(api.calls.length, 1);
    // ...but the client died before it could record that locally — exactly
    // what migrateAudioQueue persists right before its own POST.
    storage.setItem(OWNER_KEY, 'migrating');
    idbData.set(PENDING_IMPORT_KEY, payload);

    // The connection stays down: every further import attempt fails too.
    api.failAlways((c: any) => c.path === '/api/audio-queue/migration/import');

    await assert.rejects(() => retryQueueMigration('tok'));
    assert.equal(getAudioQueueOwner(), 'migrating', 'must NOT fall back to browser — the server may already own this item');

    await resumeQueue('tok');
    assert.equal(getAudioQueueOwner(), 'migrating', 'still fenced after Resume');
    assert.equal(items[0].status, 'pending');
    assert.equal(items[0].jobId, undefined, 'Resume did not resubmit the item locally');
    assert.equal(api.calls.filter((c: any) => c.path.startsWith('/api/generate')).length, 0);
  });
});

test('a stuck migration recovers once the replay succeeds', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    const payload = {
      backupId: 'bk-2', choice: 'resume',
      items: [{ legacyId: 'a', request: { prompt: 'test' }, meta: { view: items[0] }, status: 'pending' }],
    };
    storage.setItem(OWNER_KEY, 'migrating');
    idbData.set(PENDING_IMPORT_KEY, payload);

    await retryQueueMigration('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.equal(importCalls(api).length, 1);

    await resumeQueue('tok');
    assert.equal(getAudioQueueOwner(), 'server', 'resumeQueue takes the server branch, not local execution');
  });
});

/** Make the next matching localStorage write throw once, like a tab dying at that line. */
function crashOnce(storage: MemoryStorage, method: 'setItem' | 'removeItem', match: (key: string, value?: string) => boolean) {
  const real = (storage as any)[method].bind(storage);
  let armed = true;
  (storage as any)[method] = (key: string, value?: string) => {
    if (armed && match(key, value)) { armed = false; throw new Error('tab closed'); }
    return real(key, value);
  };
}

test('an import interrupted before server ownership is recorded is replayed on reload', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    const payload = {
      backupId: 'bk-c3', choice: 'resume',
      items: [{ legacyId: 'a', request: { prompt: 'test' }, meta: { view: items[0] }, status: 'pending' }],
    };
    storage.setItem(OWNER_KEY, 'migrating');
    idbData.set(PENDING_IMPORT_KEY, payload);
    crashOnce(storage, 'setItem', (key, value) => key === OWNER_KEY && value === 'server');

    await assert.rejects(() => retryQueueMigration('tok'), /tab closed/);
    assert.equal(getAudioQueueOwner(), 'migrating');

    _resetAudioQueueForTests(items);
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server', 'reload replays the import');
    assert.equal(importCalls(api).length, 2);
  });
});
