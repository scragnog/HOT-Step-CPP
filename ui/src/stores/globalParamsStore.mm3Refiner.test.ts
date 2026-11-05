import assert from 'node:assert/strict';
import test from 'node:test';
import { register } from 'node:module';
import type { GenerationParams } from '../types';

register('data:text/javascript,' + encodeURIComponent(
  'export async function load(url, ctx, next) { if (url.endsWith(".css")) return { format: "module", source: "export default {}", shortCircuit: true }; return next(url, ctx); }'));
const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => storage.set(key, value),
} });
const { normalizeMm3RefinerSettings, useGlobalParamsStore } = await import('./globalParamsStore');

test('refiner normalization supplies defaults for empty and old settings', () => {
  const defaults = {
    mm3RefinerEnabled: false,
    mm3RefinerSteps: 32,
    mm3RefinerWindowSeconds: 30,
    mm3RefinerOverlapSeconds: 2,
    mm3RefinerMix: 1,
  };
  assert.deepEqual(normalizeMm3RefinerSettings({}), defaults);
  const oldSettings: Partial<GenerationParams> = { masteringEnabled: true, stableStepOn: false };
  assert.deepEqual(normalizeMm3RefinerSettings(oldSettings), defaults);
  assert.deepEqual(normalizeMm3RefinerSettings({
    mm3RefinerSteps: NaN, mm3RefinerWindowSeconds: Infinity,
    mm3RefinerOverlapSeconds: NaN, mm3RefinerMix: -Infinity,
  }), defaults);
});

test('refiner normalization bounds steps, mix, window and overlap', () => {
  for (const [input, expected] of [[0, 1], [65, 64], [7.6, 8]]) {
    assert.equal(normalizeMm3RefinerSettings({ mm3RefinerSteps: input }).mm3RefinerSteps, expected);
  }
  for (const [input, expected] of [[-1, 0], [2, 1], [0.5, 0.5]]) {
    assert.equal(normalizeMm3RefinerSettings({ mm3RefinerMix: input }).mm3RefinerMix, expected);
  }
  assert.equal(normalizeMm3RefinerSettings({ mm3RefinerWindowSeconds: -1 }).mm3RefinerWindowSeconds, 0);
  assert.equal(normalizeMm3RefinerSettings({ mm3RefinerOverlapSeconds: -1 }).mm3RefinerOverlapSeconds, 0);
  for (const window of [0.05, 1, 30]) {
    for (const overlap of [window, window + 1]) {
      const normalized = normalizeMm3RefinerSettings({
        mm3RefinerWindowSeconds: window, mm3RefinerOverlapSeconds: overlap,
      });
      assert.equal(normalized.mm3RefinerOverlapSeconds, Math.max(0, window - 0.1));
      assert.ok(normalized.mm3RefinerOverlapSeconds < window);
    }
  }
  assert.equal(normalizeMm3RefinerSettings({
    mm3RefinerWindowSeconds: 15, mm3RefinerOverlapSeconds: 2,
  }).mm3RefinerOverlapSeconds, 2);
  const direct = normalizeMm3RefinerSettings({
    mm3RefinerEnabled: true, mm3RefinerWindowSeconds: 0, mm3RefinerOverlapSeconds: 40,
  });
  assert.equal(direct.mm3RefinerEnabled, true);
  assert.equal(direct.mm3RefinerWindowSeconds, 0);
  assert.equal(direct.mm3RefinerOverlapSeconds, 40);
});

test('committing the final window preserves overlap unless the window is shorter', () => {
  const store = useGlobalParamsStore.getState();
  store.setMm3RefinerSettings({ mm3RefinerWindowSeconds: 30, mm3RefinerOverlapSeconds: 2 });
  store.setMm3RefinerWindowSeconds(15);
  assert.equal(useGlobalParamsStore.getState().mm3RefinerOverlapSeconds, 2);
  assert.equal(storage.get('hs-mm3RefinerOverlapSeconds'), '2');
  store.setMm3RefinerWindowSeconds(1);
  assert.equal(useGlobalParamsStore.getState().mm3RefinerOverlapSeconds, 0.9);
  assert.equal(storage.get('hs-mm3RefinerOverlapSeconds'), '0.9');
  const request = store.getGlobalParams({
    mm3RefinerEnabled: true, mm3RefinerWindowSeconds: 1, mm3RefinerOverlapSeconds: 2,
  });
  assert.equal(request.mm3RefinerOverlapSeconds, 0.9);
});
