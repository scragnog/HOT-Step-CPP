// audioGenQueueStore.migration.test.ts — the real store logic behind queue
// ownership: never-decided -> server/migrate, a migration interrupted by
// reload, retry, and explicit rollback. Exercises the actual exported
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

// downloadQueueExport() (migrateAudioQueue/rollbackAudioQueue) schedules a
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

(globalThis as any).window = { addEventListener() {}, removeEventListener() {} };
(globalThis as any).document = { createElement: () => ({ click() {}, href: '', download: '' }) };
(globalThis.URL as any).createObjectURL = () => 'blob:fake';
(globalThis.URL as any).revokeObjectURL = () => {};
const initialIdb = createFakeIndexedDB();
(globalThis as any).localStorage = new MemoryStorage();
(globalThis as any).indexedDB = initialIdb.indexedDB;

const {
  migrateAudioQueue, retryQueueMigration, rollbackAudioQueue, resumeQueue,
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

/** Seeds the browser backup `rollbackAudioQueue` reads back by id — as if
 *  `exportQueueBackup` had produced it for the interrupted attempt. */
function seedBackup(idbData: Map<string, unknown>, id: string, itemIds: string[]): void {
  idbData.set(`${BACKUP_KEY}:${id}`, { version: 1, id, exportedAt: Date.now(), legacyRaw: null, indexedDb: null, itemIds, jobIds: [] });
}

test('a never-decided profile with nothing pending becomes server-owned silently, no network call', async () => {
  await withFakeQueue([], async ({ api }) => {
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.equal(api.calls.length, 0);
  });
});

test('a never-decided profile with pending items migrates once, silently', async () => {
  await withFakeQueue([fixtureItem('a')], async ({ api }) => {
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.equal(api.calls.filter((c: any) => c.path === '/api/audio-queue/migration/import').length, 1);
  });
});

test('import landed before an interrupted reload: a failed replay stays fenced, and Resume does not resubmit', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    const payload = {
      backupId: 'bk-1', choice: 'resume',
      items: [{ legacyId: 'a', request: { prompt: 'test' }, meta: { view: items[0] }, status: 'pending' }],
    };
    seedBackup(idbData, 'bk-1', ['a']);
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
    assert.equal(api.calls.filter((c: any) => c.path === '/api/audio-queue/migration/import').length, 1);

    await resumeQueue('tok');
    assert.equal(getAudioQueueOwner(), 'server', 'resumeQueue takes the server branch, not local execution');
  });
});

test('an explicit rollback is the only way back to browser ownership from a stuck migration', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    const payload = {
      backupId: 'bk-3', choice: 'resume',
      items: [{ legacyId: 'a', request: { prompt: 'test' }, meta: { view: items[0] }, status: 'pending' }],
    };
    seedBackup(idbData, 'bk-3', ['a']);
    storage.setItem(OWNER_KEY, 'migrating');
    idbData.set(PENDING_IMPORT_KEY, payload);

    // Even a repeatedly-failing retry must not unblock rollback's own path.
    api.failAlways((c: any) => c.path === '/api/audio-queue/migration/import');
    await assert.rejects(() => retryQueueMigration('tok'));
    assert.equal(getAudioQueueOwner(), 'migrating');

    await rollbackAudioQueue('tok');
    assert.equal(getAudioQueueOwner(), 'browser');
    assert.equal(api.rollbackCalls, 1);

    // Browser ownership is no longer fenced — resumeQueue may now run locally.
    await resumeQueue('tok');
    assert.equal(getAudioQueueOwner(), 'browser');
  });
});

// ── Rollback failures (round 4): every failure stays 'migrating', fenced ──

const { ROLLBACK_PLAN_KEY, isAudioQueueRollbackPending } = await import('./audioGenQueueStore.js') as any;
const restoredItems = (idbData: Map<string, unknown>) => (idbData.get('state') as { items: AudioQueueItem[] } | undefined)?.items ?? [];
/** A server queue item imported from browser item `legacyId`. */
const srvItem = (legacyId: string, status: string) =>
  ({ id: `srv-${legacyId}`, status, meta: { legacyId }, request: { title: legacyId }, createdAt: 1 });
const generateCalls = (api: AudioQueueApiFake) => api.calls.filter((c: any) => c.path.startsWith('/api/generate'));

test('rollback refused with 409 (submission in flight) stays fenced; reload neither replays the import nor runs locally; a later rollback succeeds', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'server');
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-r1' });
    seedBackup(idbData, 'bk-r1', ['a']);
    api.serverItems = [srvItem('a', 'submitting')];

    await assert.rejects(() => rollbackAudioQueue('tok'), /in flight/);
    assert.equal(getAudioQueueOwner(), 'migrating', 'never back to server after a failed rollback');
    assert.equal(isAudioQueueRollbackPending(), true);
    assert.equal(api.paused, true, 'the server queue stays paused, as the real route leaves it');

    // Reload: the load-time decision must not replay the import over the
    // user's rollback, and Resume must not run anything in the browser.
    _resetAudioQueueForTests(items);
    await ensureQueueOwnerDecided('tok');
    await resumeQueue('tok');
    assert.equal(getAudioQueueOwner(), 'migrating');
    assert.equal(api.calls.filter((c: any) => c.path === '/api/audio-queue/migration/import').length, 0);
    assert.deepEqual(generateCalls(api), []);
    assert.equal(items[0].jobId, undefined);

    // The submission settles; the user retries the rollback.
    api.serverItems[0].status = 'pending';
    await rollbackAudioQueue('tok');
    assert.equal(getAudioQueueOwner(), 'browser');
    assert.equal(isAudioQueueRollbackPending(), false);
    assert.equal(api.serverItems[0].status, 'cancelled');
    assert.deepEqual(restoredItems(idbData).map(i => [i.status, i.stage]), [['pending', 'Held after rollback']]);
    assert.equal(idbData.get(ROLLBACK_PLAN_KEY), null, 'the finished plan is cleared');
  });
});

test('a rollback that fails part-way through its cancels finishes the same plan on retry, without re-exporting', async () => {
  await withFakeQueue([fixtureItem('a'), fixtureItem('b')], async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'server');
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-r2' });
    seedBackup(idbData, 'bk-r2', ['a', 'b']);
    api.serverItems = [
      srvItem('a', 'pending'),
      srvItem('b', 'pending'),
    ];
    api.failNext((c: any) => c.path === '/api/audio-queue/items/srv-b/cancel');

    await assert.rejects(() => rollbackAudioQueue('tok'));
    assert.equal(getAudioQueueOwner(), 'migrating');
    assert.deepEqual(api.serverItems.map(i => i.status), ['cancelled', 'pending']);
    assert.ok(idbData.get(ROLLBACK_PLAN_KEY), 'the plan was saved before the first cancel');

    await rollbackAudioQueue('tok');
    assert.equal(api.rollbackCalls, 1, 'the retry reuses the plan instead of exporting a half-cancelled server');
    assert.deepEqual(api.serverItems.map(i => i.status), ['cancelled', 'cancelled']);
    assert.equal(getAudioQueueOwner(), 'browser');
    // Both come back pending: the first one's cancel by the earlier attempt
    // does not turn it into a failed item.
    assert.deepEqual(restoredItems(idbData).map(i => i.status), ['pending', 'pending']);
  });
});

test('after a failed rollback, choosing Retry hands the queue back to the server and clears the rollback state', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    const payload = {
      backupId: 'bk-r3', choice: 'resume',
      items: [{ legacyId: 'a', request: { prompt: 'test' }, meta: { view: items[0] }, status: 'pending' }],
    };
    storage.setItem(OWNER_KEY, 'server');
    idbData.set(PENDING_IMPORT_KEY, payload);
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-r3' });
    seedBackup(idbData, 'bk-r3', ['a']);
    api.serverItems = [srvItem('a', 'pending')];
    api.failNext((c: any) => c.path === '/api/audio-queue/migration/rollback-export');

    await assert.rejects(() => rollbackAudioQueue('tok'));
    assert.equal(getAudioQueueOwner(), 'migrating');
    assert.equal(isAudioQueueRollbackPending(), true);

    await retryQueueMigration('tok');
    assert.equal(getAudioQueueOwner(), 'server');
    assert.equal(isAudioQueueRollbackPending(), false);
    assert.equal(idbData.get(ROLLBACK_PLAN_KEY), null);
    assert.equal(api.serverItems[0].status, 'pending', 'nothing was cancelled');
  });
});

// ── Interrupted cleanup (round 5): recovery data outlives the ownership change ──

/** Make the next matching localStorage write throw once, like a tab dying at that line. */
function crashOnce(storage: MemoryStorage, method: 'setItem' | 'removeItem', match: (key: string, value?: string) => boolean) {
  const real = (storage as any)[method].bind(storage);
  let armed = true;
  (storage as any)[method] = (key: string, value?: string) => {
    if (armed && match(key, value)) { armed = false; throw new Error('tab closed'); }
    return real(key, value);
  };
}

const { ROLLBACK_REQUESTED_KEY } = await import('./audioGenQueueStore.js') as any;

test('a rollback interrupted before browser ownership is recorded keeps its plan; the retry restores the saved pending state', async () => {
  await withFakeQueue([fixtureItem('a')], async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'server');
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-c1' });
    seedBackup(idbData, 'bk-c1', ['a']);
    api.serverItems = [srvItem('a', 'pending')];
    crashOnce(storage, 'setItem', (key, value) => key === OWNER_KEY && value === 'browser');

    await assert.rejects(() => rollbackAudioQueue('tok'), /tab closed/);
    assert.equal(getAudioQueueOwner(), 'migrating');
    assert.equal(api.serverItems[0].status, 'cancelled', 'the cancel already happened');
    assert.ok(idbData.get(ROLLBACK_PLAN_KEY), 'the plan survives the interruption');

    _resetAudioQueueForTests([fixtureItem('a')]);
    await rollbackAudioQueue('tok');
    assert.equal(api.rollbackCalls, 1, 'no second export of the already-cancelled server');
    assert.equal(getAudioQueueOwner(), 'browser');
    assert.deepEqual(restoredItems(idbData).map(i => i.status), ['pending'], 'not restored as failed');
  });
});

test('a rollback interrupted after browser ownership is recorded stands; the next migration clears its leftovers', async () => {
  await withFakeQueue([fixtureItem('a')], async ({ storage, idbData, api }) => {
    storage.setItem(OWNER_KEY, 'server');
    idbData.set(`${BACKUP_KEY}:receipt`, { backupId: 'bk-c2' });
    seedBackup(idbData, 'bk-c2', ['a']);
    api.serverItems = [srvItem('a', 'pending')];
    crashOnce(storage, 'removeItem', key => key === ROLLBACK_REQUESTED_KEY);

    await rollbackAudioQueue('tok');
    assert.equal(getAudioQueueOwner(), 'browser', 'a failed tidy-up does not reopen the rollback');
    assert.equal(isAudioQueueRollbackPending(), true, 'the marker was left behind');

    _resetAudioQueueForTests([fixtureItem('a')]);
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'browser', 'browser ownership is final');
    // Later, the browser queue holds new work and moves to the server again.
    idbData.set('state', { items: [fixtureItem('b')], completionCounter: 0 });
    _resetAudioQueueForTests([fixtureItem('b')]);
    await migrateAudioQueue('tok', 'resume', false);
    assert.equal(getAudioQueueOwner(), 'server');
    assert.equal(isAudioQueueRollbackPending(), false);
    assert.equal(idbData.get(ROLLBACK_PLAN_KEY), null);
  });
});

test('an import interrupted after clearing the rollback state but before server ownership is replayed on reload', async () => {
  const items = [fixtureItem('a')];
  await withFakeQueue(items, async ({ storage, idbData, api }) => {
    const payload = {
      backupId: 'bk-c3', choice: 'resume',
      items: [{ legacyId: 'a', request: { prompt: 'test' }, meta: { view: items[0] }, status: 'pending' }],
    };
    storage.setItem(OWNER_KEY, 'migrating');
    storage.setItem(ROLLBACK_REQUESTED_KEY, 'bk-c3');
    idbData.set(PENDING_IMPORT_KEY, payload);
    crashOnce(storage, 'setItem', (key, value) => key === OWNER_KEY && value === 'server');

    await assert.rejects(() => retryQueueMigration('tok'), /tab closed/);
    assert.equal(getAudioQueueOwner(), 'migrating');
    assert.equal(isAudioQueueRollbackPending(), false);

    _resetAudioQueueForTests(items);
    await ensureQueueOwnerDecided('tok');
    assert.equal(getAudioQueueOwner(), 'server', 'no rollback pending, so reload replays the import');
    assert.equal(api.calls.filter((c: any) => c.path === '/api/audio-queue/migration/import').length, 2);
  });
});
