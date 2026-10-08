// ScaleOverridePresets.test.ts — the scale presets' server state
// (scalePresetCollection(), which the component renders and edits through)
// against an in-memory /api/preferences:
//   (cd server && node --import tsx --test ../ui/src/components/shared/ScaleOverridePresets.test.ts)
import test from 'node:test';
import assert from 'node:assert/strict';
import { scalePresetCollection, _resetScalePresetsForTests, STORAGE_KEY, MIGRATED_FLAG } from './ScaleOverridePresets';
import { tick, withBrowser } from '../../services/preferencesTestFake';

const scales = { self_attn: 1, cross_attn: 1, mlp: 1, cond_embed: 1 };
const preset = (name: string, overallScale: number) => ({ name, overallScale, groupScales: { ...scales } });

test('scale presets: a collision waits for a choice, and keep-both keeps both under one name', async () => {
  await withBrowser(async (fake, storage) => {
    _resetScalePresetsForTests();
    fake.seed('scale-override', preset('Soft', 0.5));
    storage.setItem(STORAGE_KEY, JSON.stringify([preset('Soft', 0.8), preset('Hard', 1.2)]));
    const presets = scalePresetCollection();
    await presets.load();
    assert.deepEqual(presets.getSnapshot().importConflicts.map(c => c.name), ['Soft']);
    assert.equal(storage.getItem(MIGRATED_FLAG), null);
    await presets.resolveImport(`${STORAGE_KEY}:Soft`, 'keep-both');
    const soft = presets.getSnapshot().entries.filter(e => e.body.name === 'Soft').map(e => e.body.overallScale).sort();
    assert.deepEqual(soft, [0.5, 0.8]);
    assert.equal(storage.getItem(MIGRATED_FLAG), '1');
  });
});

test('scale presets: a delete refused because another client changed the preset stays listed until the user acts', async () => {
  await withBrowser(async fake => {
    _resetScalePresetsForTests();
    const doc = fake.seed('scale-override', preset('Soft', 0.5));
    const presets = scalePresetCollection();
    await presets.load();
    fake.touch(doc.id, preset('Soft', 0.6));
    await presets.remove(doc.id);
    await tick();
    const e = presets.getSnapshot().entries.find(x => x.key === doc.id)!;
    assert.equal(e.status, 'conflict');
    assert.equal(e.pendingOp, 'delete');
    assert.equal((e.serverBody as any).overallScale, 0.6);
    // Delete again, now over the current revision.
    await presets.reapply(doc.id);
    assert.deepEqual(presets.getSnapshot().entries, []);
    assert.equal(fake.docs.length, 0);
  });
});
