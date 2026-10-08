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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(r => { resolve = r; });
  return { promise, resolve };
}
const doc = (id: string, revision: number, fields: Record<string, unknown>) => ({ document: { id, revision, body: body(fields) } as never, sourceError: null });

test('load decides after the fetch: an edit made while the draft loads is protected', async () => {
  const h = harness();
  await h.mirror.save('t', body({ 'hs-caption': 'clean' }));
  let form = body({ 'hs-caption': 'clean' });
  const get = deferred<ReturnType<typeof doc>>();
  const applied: StudioDraftBody[] = [];
  const asked: string[] = [];
  const loading = h.mirror.load('t', 'd7', { get: () => get.promise, current: () => form,
    confirm: m => { asked.push(m); return false; }, apply: b => applied.push(b) });
  form = body({ 'hs-caption': 'typed while loading' });   // clean at selection, dirty when the GET lands
  get.resolve(doc('d7', 3, { 'hs-caption': 'saved draft' }));
  assert.equal(await loading, false);
  assert.equal(asked.length, 1);
  assert.deepEqual(applied, []);
  assert.equal(h.pointer()!.id, 'd1');
});

test('competing loads: the last one chosen wins, whatever order the fetches finish in', async () => {
  const h = harness();
  const first = deferred<ReturnType<typeof doc>>(), second = deferred<ReturnType<typeof doc>>();
  const applied: unknown[] = [];
  const hooks = (p: Promise<ReturnType<typeof doc>>) => ({ get: () => p, current: () => body({}), confirm: () => true,
    apply: (b: StudioDraftBody) => applied.push(b.fields['hs-caption']) });
  const a = h.mirror.load('t', 'dA', hooks(first.promise));
  const b = h.mirror.load('t', 'dB', hooks(second.promise));
  second.resolve(doc('dB', 2, { 'hs-caption': 'B' }));
  assert.equal(await b, true);
  first.resolve(doc('dA', 5, { 'hs-caption': 'A' }));
  assert.equal(await a, false);
  assert.deepEqual(applied, ['B']);
  assert.equal(h.pointer()!.id, 'dB');
});

test('a refused draft (wrong backend) changes nothing and says why', async () => {
  const h = harness();
  await h.mirror.save('t', body({ 'hs-caption': 'mine' }));
  const applied: unknown[] = [];
  const ok = await h.mirror.load('t', 'd5', { get: async () => doc('d5', 1, { 'hs-caption': 'mm3 draft' }),
    current: () => body({ 'hs-caption': 'mine' }), confirm: () => true, apply: b => applied.push(b),
    refuse: () => 'This draft was saved for MiniMax-Music3. Switch the backend to MiniMax-Music3 to load it.' });
  assert.equal(ok, false);
  assert.deepEqual(applied, []);
  assert.equal(h.pointer()!.id, 'd1');
  assert.match(h.error(), /saved for MiniMax-Music3/);
});
