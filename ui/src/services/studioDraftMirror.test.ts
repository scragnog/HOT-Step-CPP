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
  const store = { read: () => pointer, write: (v: DraftPointer) => { pointer = v; } };
  const mirror = createDraftMirror({ api: api as never, pointer: store, onError: m => { error = m; } });
  /** Another editor (a second tab) on the same browser pointer and server. */
  const another = () => {
    let own = '';
    return { mirror: createDraftMirror({ api: api as never, pointer: store, onError: m => { own = m; } }), error: () => own };
  };
  return { mirror, docs, calls, another, pointer: () => pointer, error: () => error };
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
  // A headless caller writes the same draft.
  h.docs.get('d1')!.revision = 5; h.docs.get('d1')!.body = body({ 'hs-caption': 'theirs' });
  await h.mirror.save('t', body({ 'hs-caption': 'mine, edited' }));
  assert.deepEqual(h.docs.get('d1')!.body.fields, { 'hs-caption': 'theirs' });
  assert.deepEqual(h.docs.get('d2')!.body.fields, { 'hs-caption': 'mine, edited' });
  assert.equal(h.pointer()!.id, 'd2');
  assert.match(h.error(), /new draft/);
});

test('two editors sharing one browser pointer never overwrite each other', async () => {
  const h = harness();
  await h.mirror.save('t', body({ 'hs-caption': 'start' }));
  // Two tabs open from the same pointer (d1 at revision 1).
  const tabA = h.another(), tabB = h.another();
  await tabA.mirror.save('t', body({ 'hs-caption': 'A' }));
  assert.equal(h.pointer()!.revision, 2);
  // B still holds the form it opened with; the shared pointer must not lend it A's revision.
  await tabB.mirror.save('t', body({ 'hs-caption': 'B' }));
  assert.deepEqual(h.docs.get('d1')!.body.fields, { 'hs-caption': 'A' });
  assert.deepEqual(h.docs.get('d2')!.body.fields, { 'hs-caption': 'B' });
  assert.ok(h.calls.includes('update d1@1'));
  assert.match(tabB.error(), /new draft/);
});

test('loading a draft adopts it and drops the old form queued and in-flight saves', async () => {
  const h = harness();
  await h.mirror.save('t', body({ 'hs-caption': 'old' }));
  const other = { id: 'd9', revision: 4, body: body({ 'hs-caption': 'loaded' }) };
  h.docs.set('d9', { revision: 4, body: other.body });
  const inFlight = h.mirror.save('t', body({ 'hs-caption': 'old, edited' }));
  void h.mirror.save('t', body({ 'hs-caption': 'old, edited again' }));
  assert.equal(h.mirror.unsaved(body({ 'hs-caption': 'old, edited again' })), true);
  h.mirror.adopt(other as never);
  await inFlight;
  assert.deepEqual(h.pointer(), { id: 'd9', revision: 4, saved: JSON.stringify(other.body) });
  assert.equal(h.mirror.unsaved(other.body), false);
  // The loaded form saves into the loaded draft at its revision.
  await h.mirror.save('t', body({ 'hs-caption': 'loaded, edited' }));
  assert.deepEqual(h.docs.get('d9')!.body.fields, { 'hs-caption': 'loaded, edited' });
  assert.equal(h.calls.at(-1), 'update d9@4');
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
