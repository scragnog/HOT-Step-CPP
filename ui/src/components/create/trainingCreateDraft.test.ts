import assert from 'node:assert/strict';
import test from 'node:test';
import { applyTrainingCreateDraft, type ApplyDeps } from './trainingCreateDraft';

function fakeDeps(state: Record<string, unknown>, stored: Record<string, string> = {}) {
  const writes: Record<string, unknown> = {};
  const setState: Record<string, unknown>[] = [];
  const deps: ApplyDeps = {
    store: { getState: () => state, setState: p => { setState.push(p); } },
    write: (k, v) => { writes[k] = v; },
    storage: { getItem: k => stored[k] ?? null, setItem: (k, v) => { stored[k] = v; } },
    scopedKey: base => `${base}:ace`,
  };
  return { deps, writes, setState, stored };
}

test('a draft applies content, params via setters, scoped pluginParams and the codes cache setting', () => {
  const set: Record<string, unknown> = {};
  const { deps, writes, setState, stored } = fakeDeps(
    { setSeed: (v: unknown) => { set.seed = v; } }, { 'ace-settings': JSON.stringify({ cacheLmCodes: true, other: 1 }) });
  applyTrainingCreateDraft({ content: { 'hs-caption': 'x' }, params: { seed: 0, pluginParams: {} },
    settings: { cacheLmCodes: false } }, deps);
  assert.equal(writes['hs-caption'], 'x');
  assert.equal(set.seed, 0);
  assert.deepEqual(setState, [{ pluginParams: {} }]);
  assert.equal(stored['hs-pluginParams:ace'], '{}');
  assert.deepEqual(writes['ace-settings'], { cacheLmCodes: false, other: 1 });
});

test('a draft with a key this build cannot apply changes nothing', () => {
  const { deps, writes, setState } = fakeDeps({ setSeed: () => assert.fail('applied') });
  assert.throws(() => applyTrainingCreateDraft({ content: { 'hs-caption': 'x' }, params: { seed: 1, gone: 2 } }, deps), /gone/);
  assert.deepEqual(writes, {});
  assert.deepEqual(setState, []);
});
