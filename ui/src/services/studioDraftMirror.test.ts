// studioDraftMirror.test.ts — revisioned draft mirroring with a fake API.
//   (cd server && node --import tsx --test ../ui/src/services/studioDraftMirror.test.ts)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { StudioDraftBody } from '../../../server/src/contracts/studioDrafts';
import { createDraftMirror, type DraftPointer } from './studioDraftMirror';

const conflict = Object.assign(new Error('Document changed elsewhere'), { status: 409 });

function harness(failNext?: (call: string) => Error | null) {
  const docs = new Map<string, { revision: number; body: StudioDraftBody }>();
  const calls: string[] = [];
  let pointer: DraftPointer | null = null;
  let error = '';
  let next = 0;
  const api = {
    async create(_token: string, body: StudioDraftBody) {
      calls.push('create');
      const failure = failNext?.('create'); if (failure) throw failure;
      const id = `d${++next}`;
      docs.set(id, { revision: 1, body: structuredClone(body) });
      return { document: { id, revision: 1, body } };
    },
    async update(_token: string, id: string, expectedRevision: number, body: StudioDraftBody) {
      calls.push(`update ${id}@${expectedRevision}`);
      const failure = failNext?.('update'); if (failure) throw failure;
      const doc = docs.get(id)!;
      if (doc.revision !== expectedRevision) throw conflict;
      doc.revision += 1; doc.body = structuredClone(body);
      return { document: { id, revision: doc.revision, body } };
    },
  };
  const mirror = createDraftMirror({ api: api as never, pointer: { read: () => pointer, write: v => { pointer = v; } }, onError: m => { error = m; } });
  return { mirror, docs, calls, pointer: () => pointer, error: () => error };
}

const body = (fields: Record<string, unknown>): StudioDraftBody => ({ studio: 'create', fields });

test('first save creates, later saves update with the expected revision, unchanged bodies are skipped', async () => {
  const h = harness();
  await h.mirror.save('t', body({ 'hs-caption': 'a', 'hs-bpm': 0, 'hs-instrumental': false }));
  await h.mirror.save('t', body({ 'hs-caption': 'a', 'hs-bpm': 0, 'hs-instrumental': false }));
  await h.mirror.save('t', body({ 'hs-caption': 'b', 'hs-bpm': 0, 'hs-instrumental': false }));
  assert.deepEqual(h.calls, ['create', 'update d1@1']);
  // Resumed draft: false and zero survive exactly.
  assert.deepEqual(h.docs.get('d1')!.body.fields, { 'hs-caption': 'b', 'hs-bpm': 0, 'hs-instrumental': false });
  assert.equal(h.pointer()!.revision, 2);
  assert.equal(h.error(), '');
});

test('a conflict keeps both: the other writer keeps its draft, this edit becomes a new one', async () => {
  const h = harness();
  await h.mirror.save('t', body({ 'hs-caption': 'mine' }));
  // Another client (or a headless caller) writes the same draft.
  h.docs.get('d1')!.revision = 5; h.docs.get('d1')!.body = body({ 'hs-caption': 'theirs' });
  await h.mirror.save('t', body({ 'hs-caption': 'mine, edited' }));
  assert.deepEqual(h.docs.get('d1')!.body.fields, { 'hs-caption': 'theirs' });
  assert.deepEqual(h.docs.get('d2')!.body.fields, { 'hs-caption': 'mine, edited' });
  assert.equal(h.pointer()!.id, 'd2');
  assert.match(h.error(), /new draft/);
});

test('a failed save shows an error, keeps the pointer and retries with the next edit', async () => {
  let fail = true;
  const h = harness(call => call === 'update' && fail ? Object.assign(new Error('offline'), { status: 503 }) : null);
  await h.mirror.save('t', body({ 'hs-caption': 'a' }));
  await h.mirror.save('t', body({ 'hs-caption': 'b' }));
  assert.match(h.error(), /Draft not saved: offline/);
  assert.equal(h.pointer()!.revision, 1);
  fail = false;
  await h.mirror.save('t', body({ 'hs-caption': 'c' }));
  assert.deepEqual(h.docs.get('d1')!.body.fields, { 'hs-caption': 'c' });
  assert.equal(h.error(), '');
});

test('edits made while a save is in flight are written last, once', async () => {
  const h = harness();
  const first = h.mirror.save('t', body({ 'hs-caption': '1' }));
  void h.mirror.save('t', body({ 'hs-caption': '2' }));
  void h.mirror.save('t', body({ 'hs-caption': '3' }));
  await first;
  assert.deepEqual(h.calls, ['create', 'update d1@1']);
  assert.deepEqual(h.docs.get('d1')!.body.fields, { 'hs-caption': '3' });
});
