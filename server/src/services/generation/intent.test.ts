import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveGenerationIntent } from './intent.js';
import type { BackendExtensionParam } from '../backends/types.js';

// Safe field projections from INDEX_BATCH1_2026_10_07.json. The local capture
// directory is ignored by git and contains private free text, so tests keep
// only the wire values needed to catch defaults and omission regressions.
const captures = [
  { engine: 'ace', zero: ['lmTopK'], false: ['skipLm', 'dcwEnabled', 'masteringEnabled'] },
  { engine: 'minimax-m3', zero: ['lmTopK'], false: ['mm3Stream', 'mm3ReuseAr', 'masteringEnabled'] },
  { engine: 'yue2', zero: ['lmTopK', 'yue2NarCacheRatio', 'yue2NarChunkSeconds'], false: ['yue2PreviewScore', 'masteringEnabled'] },
];

test('captured zero and false values survive compact intent input on all engines', () => {
  for (const fixture of captures) {
    const input = Object.fromEntries([
      ...fixture.zero.map(key => [key, 0]), ...fixture.false.map(key => [key, false]),
    ]);
    const resolved = resolveGenerationIntent({ contract: 'generation-intent/1', params: {}, input }, fixture.engine, []);
    for (const key of fixture.zero) assert.equal(resolved[key], 0, `${fixture.engine}: ${key}`);
    for (const key of fixture.false) assert.equal(resolved[key], false, `${fixture.engine}: ${key}`);
  }
});

test('blend budget preserves two and three adapter weights and zero budget', () => {
  for (const count of [2, 3]) {
    const adapterStack = Array.from({ length: count }, (_, i) => ({ path: `${i}.safetensors`, scale: i + 1 }));
    const params = { advancedAdapters: true, adapterStackMode: 'blend', adapterStackBudget: 0.75, adapterStack };
    const result = resolveGenerationIntent({ contract: 'generation-intent/1', params }, 'ace', []);
    const stack = result.loraStack as { scale: number }[];
    assert.equal(stack.length, count);
    assert.ok(Math.abs(stack.reduce((sum, entry) => sum + entry.scale, 0) - 0.75) < 0.0002);
  }
  const zero = resolveGenerationIntent({ contract: 'generation-intent/1', params: {
    advancedAdapters: true, adapterStackMode: 'blend', adapterStackBudget: 0,
    adapterStack: [{ path: 'a', scale: 1 }, { path: 'b', scale: 1 }],
  } }, 'ace', []);
  assert.deepEqual((zero.loraStack as { scale: number }[]).map(entry => entry.scale), [0, 0]);
});

test('captured two and three adapter stacks retain their normalized scales', () => {
  for (const scales of [[0.375, 0.375], [0.25, 0.25, 0.25]]) {
    const stack = scales.map((scale, i) => ({ path: `adapter-${i + 1}.safetensors`, scale }));
    const result = resolveGenerationIntent({ contract: 'generation-intent/1', params: {
      advancedAdapters: true, adapterStackMode: 'blend',
      adapterStackBudget: 0.75, adapterStack: stack,
    } }, 'ace', []);
    assert.deepEqual(result.loraStack, stack);
    assert.equal(result.loraPath, stack[0].path);
  }
});

test('manifest supplies backend defaults and rejects invalid extension values', () => {
  const extensions: BackendExtensionParam[] = [
    { key: 'mm3Steps', type: 'slider', label: 'Steps', default: 30, min: 2, max: 60 },
    { key: 'mm3Stream', type: 'toggle', label: 'Stream', default: false },
  ];
  const defaulted = resolveGenerationIntent({ contract: 'generation-intent/1', params: {} }, 'minimax-m3', extensions);
  assert.equal(defaulted.mm3Steps, 30);
  assert.equal(defaulted.mm3Stream, false);
  const explicit = resolveGenerationIntent({ contract: 'generation-intent/1', params: {
    backendParams: { mm3Steps: 2, mm3Stream: false },
  } }, 'minimax-m3', extensions);
  assert.equal(explicit.mm3Steps, 2);
  assert.equal(explicit.mm3Stream, false);
  assert.throws(() => resolveGenerationIntent({ contract: 'generation-intent/1', params: {
    backendParams: { mm3Steps: 0 },
  } }, 'minimax-m3', extensions), /Invalid backend parameter/);
});

test('inactive controls and seed linkage match UI assembly', () => {
  const result = resolveGenerationIntent({ contract: 'generation-intent/1', params: {
    postProcessingEnabled: false, spectralLifterEnabled: true, slDenoiseStrength: 0.9,
    seed: 0, randomSeed: false, lmSeed: 0, lmSeedFollowsDit: false,
  } }, 'ace', []);
  assert.equal(result.spectralLifterEnabled, false);
  assert.equal(result.slDenoiseStrength, undefined);
  assert.equal(result.seed, 0);
  assert.equal(result.randomSeed, false);
  assert.equal(result.lmSeed, 0);
  assert.equal(result.lmSeedFollowsDit, false);
});

test('filename trigger settings follow the adapter they describe', () => {
  const result = resolveGenerationIntent({ contract: 'generation-intent/1', params: {
    adapter: 'C:/adapters/fixture.safetensors', adapterScale: 0,
  }, settings: { triggerUseFilename: true, triggerPlacement: 'append' } }, 'ace', []);
  assert.equal(result.loraScale, 0);
  assert.equal(result.triggerWord, 'fixture');
  assert.equal(result.triggerPlacement, 'append');
  assert.deepEqual(result.triggerSpecs, [{
    word: 'fixture', placement: 'append', source: 'filename', path: 'C:/adapters/fixture.safetensors',
  }]);
});
