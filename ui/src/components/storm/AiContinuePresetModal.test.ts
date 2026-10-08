// AiContinuePresetModal.test.ts — the modal's server state (aiContinueState(),
// which the modal renders and edits through) against an in-memory
// /api/preferences:
//   (cd server && node --import tsx --test ../ui/src/components/storm/AiContinuePresetModal.test.ts)
import test from 'node:test';
import assert from 'node:assert/strict';
import { aiContinueState, _resetAiContinueStateForTests, TEMPLATE_KEY, USER_STYLE_KEY, DEFAULT_TEMPLATE, loadTemplate } from './AiContinuePresetModal';
import { tick, withBrowser } from '../../services/preferencesTestFake';

const isGet = (c: { method: string; path: string }) => c.method === 'GET' && c.path.endsWith('/settings/ai-continue-template');
const isPut = (c: { method: string; path: string }) => c.method === 'PUT' && c.path.endsWith('/settings/ai-continue-template');
const serverTemplate = (fake: { docs: Array<{ family: string; body: any; revision: number }> }) =>
  fake.docs.find(d => d.family === 'ai-continue-template');

test('a second browser hydrates the saved template and mirrors it for other readers', async () => {
  await withBrowser(async fake => {
    _resetAiContinueStateForTests();
    fake.seed('ai-continue-template', { template: 'Saved elsewhere {lyrics}' });
    const { template } = aiContinueState();
    assert.equal(template.getSnapshot().value, DEFAULT_TEMPLATE);
    await template.hydrate();
    assert.equal(template.getSnapshot().value, 'Saved elsewhere {lyrics}');
    assert.equal(loadTemplate(), 'Saved elsewhere {lyrics}');
  });
});

test('an edit made while the first read is slow is not overwritten by it, and saves at the read revision', async () => {
  await withBrowser(async (fake, storage) => {
    _resetAiContinueStateForTests();
    storage.setItem(TEMPLATE_KEY, 'Base {lyrics}');
    fake.seed('ai-continue-template', { template: 'Base {lyrics}' });
    const release = fake.hold(isGet);
    const { template } = aiContinueState();
    const hydrating = template.hydrate();
    await tick();
    template.edit('Mine {lyrics}');
    release();
    await hydrating; await template.settled();
    assert.equal(template.getSnapshot().value, 'Mine {lyrics}');
    assert.equal(template.getSnapshot().status, 'idle');
    assert.equal(storage.getItem(TEMPLATE_KEY), 'Mine {lyrics}');
    assert.equal(serverTemplate(fake)!.body.template, 'Mine {lyrics}');
    assert.equal(serverTemplate(fake)!.revision, 2);
  });
});

test('an edit of a stale copy is not sent over a newer server value; the user chooses', async () => {
  await withBrowser(async (fake, storage) => {
    _resetAiContinueStateForTests();
    storage.setItem(TEMPLATE_KEY, 'Old {lyrics}');
    fake.seed('ai-continue-template', { template: 'Newer {lyrics}' });
    const release = fake.hold(isGet);
    const { template } = aiContinueState();
    const hydrating = template.hydrate();
    await tick();
    template.edit('Mine {lyrics}');
    release();
    await hydrating; await template.settled();
    assert.equal(template.getSnapshot().status, 'conflict');
    assert.equal(template.getSnapshot().serverValue, 'Newer {lyrics}');
    assert.equal(fake.count('PUT'), 0);
    assert.equal(storage.getItem(TEMPLATE_KEY), 'Mine {lyrics}');
    template.useServer();
    assert.equal(template.getSnapshot().value, 'Newer {lyrics}');
    assert.equal(storage.getItem(TEMPLATE_KEY), 'Newer {lyrics}');
  });
});

test('fast edits are saved in order and coalesced: the last one wins, revisions only move forward', async () => {
  await withBrowser(async fake => {
    _resetAiContinueStateForTests();
    fake.seed('ai-continue-template', { template: 'v0' });
    const { template } = aiContinueState();
    await template.hydrate();
    const release = fake.hold(isPut);
    template.edit('a');
    await tick();
    template.edit('b');
    template.edit('c');
    release();
    await template.settled();
    assert.deepEqual(fake.calls.filter(isPut).map(c => c.body.body.template), ['a', 'c']);
    assert.deepEqual(fake.calls.filter(isPut).map(c => c.body.expectedRevision), [1, 2]);
    assert.equal(serverTemplate(fake)!.body.template, 'c');
    assert.equal(template.getSnapshot().status, 'idle');
  });
});

test('a 409 pauses saving until reapply; edits made meanwhile are kept and sent then', async () => {
  await withBrowser(async fake => {
    _resetAiContinueStateForTests();
    const doc = fake.seed('ai-continue-template', { template: 'v0' });
    const { template } = aiContinueState();
    await template.hydrate();
    fake.touch(doc.id, { template: 'theirs' });
    template.edit('mine');
    await template.settled();
    assert.equal(template.getSnapshot().status, 'conflict');
    assert.equal(template.getSnapshot().serverValue, 'theirs');

    template.edit('mine, edited');
    await template.settled();
    assert.equal(fake.count('PUT'), 1);

    await template.reapply();
    assert.equal(template.getSnapshot().status, 'idle');
    assert.equal(serverTemplate(fake)!.body.template, 'mine, edited');
    assert.equal(fake.calls.filter(isPut).at(-1)!.body.expectedRevision, 2);
  });
});

test('presets: a failed create keeps its label and value; a failed delete stays listed; import collisions wait', async () => {
  await withBrowser(async (fake, storage) => {
    _resetAiContinueStateForTests();
    fake.seed('ai-continue-style', { label: 'Jazz', value: 'more swing' });
    storage.setItem(USER_STYLE_KEY, JSON.stringify([{ id: 'u1', label: 'Jazz', value: 'less swing' }]));
    const { style } = aiContinueState();
    await style.load();
    assert.equal(style.getSnapshot().importConflicts.length, 1);

    fake.failNext(c => c.method === 'POST' && c.path === '/api/preferences/presets/ai-continue-style');
    await style.create({ label: 'Calm', value: 'slow down' });
    const failed = style.getSnapshot().entries.find(e => e.body.label === 'Calm')!;
    assert.equal(failed.status, 'failed');
    assert.equal(failed.body.value, 'slow down');

    const saved = style.getSnapshot().entries.find(e => e.body.label === 'Jazz')!;
    fake.failNext(c => c.method === 'DELETE');
    await style.remove(saved.key);
    const still = style.getSnapshot().entries.find(e => e.key === saved.key)!;
    assert.equal(still.status, 'failed');
    assert.match(still.error!, /Delete failed/);
    await style.reapply(saved.key);
    assert.equal(style.getSnapshot().entries.some(e => e.key === saved.key), false);
  });
});
