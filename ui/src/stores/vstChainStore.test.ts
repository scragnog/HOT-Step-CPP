// vstChainStore.test.ts — the VST preset paths the global bar uses, through
// the real store against an in-memory /api/preferences:
//   (cd server && node --import tsx --test ../ui/src/stores/vstChainStore.test.ts)
import test from 'node:test';
import assert from 'node:assert/strict';
import { useVstChainStore, _resetVstPresetsForTests, PRESETS_KEY, MIGRATED_FLAG } from './vstChainStore';
import { tick, withBrowser } from '../services/preferencesTestFake';

const plugin = (uid: string) => ({ uid, name: `Plugin ${uid}`, vendor: 'V', path: `/p/${uid}`, enabled: true, statePath: `/s/${uid}` });
const store = () => useVstChainStore.getState();
const entries = () => store().presetSnapshot.entries;

test('a colliding browser preset waits for the user: keep both shows two entries keyed by id', async () => {
  await withBrowser(async (fake, storage) => {
    _resetVstPresetsForTests();
    const saved = fake.seed('vst-chain', { name: 'Warm', entries: [plugin('a')] });
    storage.setItem(PRESETS_KEY, JSON.stringify({ Warm: [plugin('b')], Bright: [plugin('c')] }));
    await store().loadPresets();

    // The non-colliding preset came in on its own; the collision did not.
    assert.deepEqual(entries().map(e => e.body.name).sort(), ['Bright', 'Warm']);
    assert.deepEqual(store().presetSnapshot.importConflicts.map(c => [c.name, c.existingId]), [['Warm', saved.id]]);
    assert.equal(storage.getItem(MIGRATED_FLAG), null);

    await store().presetCollection.resolveImport(`${PRESETS_KEY}:Warm`, 'keep-both');
    const warm = entries().filter(e => e.body.name === 'Warm');
    assert.equal(warm.length, 2);
    assert.notEqual(warm[0]!.key, warm[1]!.key);
    assert.equal(warm.every(e => e.key === e.id), true);
    assert.deepEqual(store().presetSnapshot.importConflicts, []);
    assert.equal(storage.getItem(MIGRATED_FLAG), '1');

    // Each same-name entry applies its own chain.
    const legacy = warm.find(e => e.body.entries[0]!.uid === 'b')!;
    await store().loadPreset(legacy.key);
    assert.equal(store().chain[0]!.uid, 'b');
  });
});

test('replace overwrites the saved preset in place', async () => {
  await withBrowser(async (fake, storage) => {
    _resetVstPresetsForTests();
    const saved = fake.seed('vst-chain', { name: 'Warm', entries: [plugin('a')] });
    storage.setItem(PRESETS_KEY, JSON.stringify({ Warm: [plugin('b')] }));
    await store().loadPresets();
    await store().presetCollection.resolveImport(`${PRESETS_KEY}:Warm`, 'replace');
    assert.deepEqual(entries().map(e => [e.id, e.body.entries[0]!.uid]), [[saved.id, 'b']]);
  });
});

test('a failed save keeps the chain in the entry; reapply saves it', async () => {
  await withBrowser(async fake => {
    _resetVstPresetsForTests();
    await store().loadPresets();
    useVstChainStore.setState({ chain: [plugin('x')] });
    fake.failNext(c => c.method === 'POST' && c.path === '/api/preferences/presets/vst-chain');
    await store().savePreset('Live');
    const failed = entries()[0]!;
    assert.equal(failed.status, 'failed');
    assert.equal(failed.body.entries[0]!.uid, 'x');
    assert.match(failed.error!, /Save failed/);

    await store().presetCollection.reapply(failed.key);
    assert.equal(entries()[0]!.status, 'saved');
    assert.equal(fake.docs.length, 1);
  });
});

test('a save refused because another client changed the preset pauses until reapply', async () => {
  await withBrowser(async fake => {
    _resetVstPresetsForTests();
    const doc = fake.seed('vst-chain', { name: 'Live', entries: [plugin('a')] });
    await store().loadPresets();
    fake.touch(doc.id, { name: 'Live', entries: [plugin('other')] });

    useVstChainStore.setState({ chain: [plugin('mine')] });
    await store().savePreset('Live');
    await tick(); await tick();
    let e = entries()[0]!;
    assert.equal(e.status, 'conflict');
    assert.equal(e.revision, 2);
    assert.equal((e.serverBody as any).entries[0].uid, 'other');
    assert.equal(e.body.entries[0]!.uid, 'mine');

    // A further edit while paused is kept but not sent.
    const puts = fake.count('PUT');
    useVstChainStore.setState({ chain: [plugin('mine2')] });
    await store().savePreset('Live');
    await tick();
    assert.equal(fake.count('PUT'), puts);

    await store().presetCollection.reapply(e.key);
    e = entries()[0]!;
    assert.equal(e.status, 'saved');
    assert.equal(fake.docs[0]!.revision, 3);
    assert.equal((fake.docs[0]!.body as any).entries[0].uid, 'mine2');
  });
});

test('deleting a preset while its create is in flight deletes the created document; a failed delete is shown', async () => {
  await withBrowser(async fake => {
    _resetVstPresetsForTests();
    await store().loadPresets();
    useVstChainStore.setState({ chain: [plugin('x')] });

    const release = fake.hold(c => c.method === 'POST' && c.path === '/api/preferences/presets/vst-chain');
    const saving = store().savePreset('Gone');
    await tick();
    await store().deletePreset(entries()[0]!.key);
    release();
    await saving; await tick();
    assert.deepEqual(entries(), []);
    assert.equal(fake.docs.length, 0);
    assert.equal(fake.count('DELETE'), 1);

    // Same race, but the follow-up delete fails: the entry stays, with its id and the error.
    const release2 = fake.hold(c => c.method === 'POST' && c.path === '/api/preferences/presets/vst-chain');
    fake.failNext(c => c.method === 'DELETE');
    const saving2 = store().savePreset('Stays');
    await tick();
    await store().deletePreset(entries()[0]!.key);
    release2();
    await saving2; await tick();
    const left = entries()[0]!;
    assert.equal(left.id, fake.docs[0]!.id);
    assert.equal(left.status, 'failed');
    assert.equal(left.pendingOp, 'delete');
    assert.match(left.error!, /Delete failed/);
  });
});
