// audioGenQueueStore.yue2Pick.test.ts — a Create render carries the YuE2
// picker's pick, a save still in flight included (#319). Store imported as in
// audioGenQueueStore.delivery.test.ts.
import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { createFakeIndexedDB, MemoryStorage } from './audioQueueStoreTestFake.js';

register('./cssStubLoader.mjs', import.meta.url);
Object.assign(globalThis, {
  window: { addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; } },
  CustomEvent: class extends Event {},
  localStorage: new MemoryStorage(),
  indexedDB: createFakeIndexedDB().indexedDB,
});

const { OWNER_KEY, enqueueSimpleGen, resumeQueue, _resetAudioQueueForTests } = await import('./audioGenQueueStore.js');
const { useBackendStore } = await import('./backendStore.js');

const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });

test('Create enqueue sends the pending picker pick over a reused one; a build save keeps its own dials', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  const posted: Record<string, unknown>[] = [];
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  let defaults: Record<string, unknown> = { lmAdapterAr: 'C:/a/old.safetensors', lmAdapterArScale: 0.65 };
  const saved = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const path = new URL(url, 'http://local').pathname;
    if (path === '/api/backends/models' && method === 'POST') {
      await held;
      defaults = { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: 0.6 };
      return ok({ changed: true });
    }
    if (path === '/api/backends/models') return ok({ buckets: {}, defaults });
    if (path === '/api/resolve/preview') {
      const intent = JSON.parse(String(init!.body));
      return ok({ request: intent.params, uiEffects: [] });
    }
    if (path === '/api/audio-queue/items' && method === 'POST') {
      const { idempotencyKey, request, meta } = JSON.parse(String(init!.body));
      posted.push(request);
      return ok({ item: { id: `srv${posted.length}`, idempotencyKey, status: 'submitted', request, meta, createdAt: 1 }, created: true });
    }
    if (path === '/api/audio-queue/items') return ok({ items: [] });
    return new Response(JSON.stringify({ error: `no route ${method} ${path}` }), { status: 404 });
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  try {
    await resumeQueue('t');
    await useBackendStore.getState().fetchModels('yue2');
    const save = useBackendStore.getState().selectModels({ lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: '0.6' }, 'yue2');
    const reused = { backend: 'yue2', title: 'T', yue2Pick: { lmAdapterArScale: 1 } };
    await enqueueSimpleGen(reused as never, 't', undefined, { kind: 'create', params: reused } as never);
    await enqueueSimpleGen(reused as never, 't');
    await enqueueSimpleGen({ backend: 'ace', title: 'A' } as never, 't');
    release();
    assert.equal(await save, true);
    assert.deepEqual(posted[0].yue2Pick, { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: 0.6 }, 'one-shot: the pending pick wins');
    assert.deepEqual(posted[1].yue2Pick, { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: 1 }, 'build save: its own dial, picker path');
    assert.equal(posted[2].yue2Pick, undefined, 'other engines untouched');
  } finally {
    release();
    _resetAudioQueueForTests();
    globalThis.fetch = saved;
  }
});
