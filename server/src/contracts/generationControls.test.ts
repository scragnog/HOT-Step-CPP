import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { generationIntentSchema } from './generation.js';
import { generationControlsSchema, parseGenerationControls } from './generationControls.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const goldenPath = path.join(__dirname, '../services/generation/fixtures/batch1-intent-golden.json');
const golden: Array<{ id: string; engine: string; branch: string; intent: { contract: string; params: Record<string, unknown> } }> =
  JSON.parse(fs.readFileSync(goldenPath, 'utf8'));

test('golden intent fixtures parse under the existing generationIntentSchema', () => {
  assert.ok(golden.length > 0, 'fixture file must not be empty');
  for (const entry of golden) {
    const result = generationIntentSchema.safeParse(entry.intent);
    assert.equal(result.success, true, `${entry.id} (${entry.branch}) rejected by generationIntentSchema`);
  }
});

test('golden fixtures\' params accept under the new documented control dictionary (parity, not a tightening)', () => {
  for (const entry of golden) {
    const result = generationControlsSchema.safeParse(entry.intent.params);
    assert.equal(
      result.success, true,
      `${entry.id} (${entry.branch}) rejected by generationControlsSchema: ${!result.success ? result.error.message : ''}`,
    );
  }
});

test('every engine family (ace, yue2, minimax-m3) appears in the golden set', () => {
  const engines = new Set(golden.map(entry => entry.engine));
  for (const engine of ['ace', 'yue2', 'minimax-m3']) {
    assert.ok(engines.has(engine), `no golden fixture exercises ${engine}`);
  }
});

test('zero and false values that must survive (falsy but meaningful) still parse', () => {
  const zeroFalseFixture = {
    lmTopK: 0,
    skipLm: false,
    dcwEnabled: false,
    masteringEnabled: false,
    randomSeed: false,
    backendParams: {
      mm3ReuseAr: false,
      mm3Stream: false,
      yue2NarCacheRatio: 0,
      yue2NarChunkSeconds: 0,
      yue2PreviewScore: false,
    },
  };
  const result = generationControlsSchema.safeParse(zeroFalseFixture);
  assert.equal(result.success, true);
  if (result.success) {
    assert.equal(result.data.lmTopK, 0);
    assert.equal(result.data.skipLm, false);
    assert.equal((result.data.backendParams as Record<string, unknown>).mm3ReuseAr, false);
  }
});

test('an arbitrary/unknown field (future control or plugin param) passes through rather than being rejected', () => {
  const result = parseGenerationControls({ inferenceSteps: 12, someFutureControlNobodyHasAddedYet: 'x' });
  assert.equal(result.success, true);
});

test('backendParams and pluginParams stay free-form extension channels', () => {
  const result = parseGenerationControls({
    backendParams: { yue2OdeSteps: 32, mm3Steps: 30, anythingElse: { nested: true } },
    pluginParams: { 'my-plugin:threshold': '0.5' },
  });
  assert.equal(result.success, true);
});

test('an empty params object (every field omitted) is valid — defaults apply downstream, not here', () => {
  assert.equal(generationControlsSchema.safeParse({}).success, true);
});
