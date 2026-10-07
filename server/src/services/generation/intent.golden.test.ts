import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolveGenerationIntent } from './intent.js';
import type { BackendExtensionParam } from '../backends/types.js';

interface GoldenCase {
  id: string;
  engine: string;
  branch: string;
  intent: { contract: 'generation-intent/1'; params: Record<string, unknown>; input: Record<string, unknown> };
  expected: Record<string, unknown>;
  extensions: BackendExtensionParam[];
}

const cases = JSON.parse(fs.readFileSync(path.join(
  path.dirname(fileURLToPath(import.meta.url)), 'fixtures/batch1-intent-golden.json',
), 'utf8')) as GoldenCase[];

// The source capture is already the resolved body. We reconstructed UI state
// from its control values; only caller-owned content/settings and the known
// written-song post-assembly overrides enter `input`. This guard prevents a
// new policy mismatch from being hidden by copying it into input.
const allowedInput = new Set([
  'artist', 'bpm', 'cacheLmCodes', 'caption', 'coResident', 'duration',
  'generationTimeoutMinutes', 'instrumental', 'keyScale', 'lyrics',
  'parallelCoverArt', 'parallelQualityEval', 'parallelWhisper',
  'randomizeTimbreRef', 'source', 'subject', 'taskType', 'timeSignature',
  'title', 'vocalLanguage', 'yue2Pick',
  // audioGenQueueStore.ts overrides the album preset after global params
  // assembly; it can leave the old global masking/trigger fields in place.
  'adapterSectionAlignAt', 'adapterSectionIsolation', 'masteringReference',
  'timbreReference', 'triggerPlacement', 'triggerSpecs', 'triggerWord', 'triggerWords',
]);
const queueOverrides = new Set([
  'adapterSectionAlignAt', 'adapterSectionIsolation', 'masteringReference',
  'timbreReference', 'triggerPlacement', 'triggerSpecs', 'triggerWord', 'triggerWords',
]);

test('all 31 representative captured bodies match Node intent normalization', () => {
  assert.equal(cases.length, 31);
  assert.equal(new Set(cases.map(entry => entry.id)).size, 31);
  for (const entry of cases) {
    for (const key of Object.keys(entry.intent.input)) {
      assert.ok(allowedInput.has(key), `${entry.id}: unexpected caller override ${key}`);
      if (queueOverrides.has(key)) {
        assert.equal(entry.intent.input.source, 'lyric-studio', `${entry.id}: ${key} must come from written-song queue`);
      }
    }
    const actual = JSON.parse(JSON.stringify(resolveGenerationIntent(
      entry.intent, entry.engine, entry.extensions,
    ))) as Record<string, unknown>;
    const keys = new Set([...Object.keys(actual), ...Object.keys(entry.expected)]);
    const changed = [...keys].filter(key => JSON.stringify(actual[key]) !== JSON.stringify(entry.expected[key]));
    assert.deepEqual(changed, [], `${entry.id} (${entry.branch}) differs at: ${changed.join(', ')}`);
    assert.deepEqual(actual, entry.expected, `${entry.id} (${entry.branch})`);
  }
});
