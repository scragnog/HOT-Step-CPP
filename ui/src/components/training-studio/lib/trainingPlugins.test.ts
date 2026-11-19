import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { DatasetAssets } from '@/services/trainingApi';

import { aceChips, heartmulaChips, heartmulaReadiness, mm3Chips, sa3Chips, sa3Readiness, yue2Chips } from '@/components/training-studio/plugins/chips';

const CHIPS = { ace: aceChips, 'minimax-m3': mm3Chips, yue2: yue2Chips, sa3: sa3Chips, heartmula: heartmulaChips };

// Echo keys so the matrix pins which chip, state and tooltip each backend shows.
const ctx = { t: (key: string, o?: Record<string, unknown>) => o ? `${key}${JSON.stringify(o)}` : key, formatWhen: (iso: string) => iso };
const base: DatasetAssets = {
  labeled: true, built: true, tensorVariants: 2, tensorVariantKey: 'v1', tensorSamples: 9, ditBase: 'turbo',
  lm: { trainedAt: 'T1', detail: '4B' } as DatasetAssets['lm'],
  dit: { trainedAt: 'T2', detail: 'turbo' } as DatasetAssets['dit'],
  mm3: { codesReady: true, codesCount: 5, lmAdapter: { trainedAt: 'T3' } as DatasetAssets['lm'] },
  yue2: { latentsReady: true, clips: 7, codesReady: false, cursorReady: false, narAdapter: { trainedAt: 'T4' } as DatasetAssets['lm'], arAdapter: null },
};
const summary = (id: keyof typeof CHIPS, a: DatasetAssets) =>
  CHIPS[id](a, ctx).map(c => `${c.id}:${c.on ? 1 : 0}${c.on && c.suffix ? `:${c.suffix}` : ''}`);

test('chip ids, order, states and suffixes per backend', () => {
  assert.deepEqual(summary('ace', base), ['tensors:1:9', 'dit:1', 'lm:1:4B']);
  assert.deepEqual(summary('minimax-m3', base), ['codes:1:5', 'lmAdapter:1']);
  assert.deepEqual(summary('yue2', base), ['latents:1:7', 'codes:0', 'cursorSpans:0', 'narAdapter:1', 'arAdapter:0']);
});

test('a missing backend block renders every chip OFF instead of dropping the row', () => {
  const bare = { labeled: false, built: false, tensorVariants: 0, tensorVariantKey: '', tensorSamples: 0, ditBase: '', lm: null, dit: null } as DatasetAssets;
  assert.deepEqual(summary('minimax-m3', bare), ['codes:0', 'lmAdapter:0']);
  assert.deepEqual(summary('yue2', bare), ['latents:0', 'codes:0', 'cursorSpans:0', 'narAdapter:0', 'arAdapter:0']);
  assert.deepEqual(summary('sa3', bare), ['audio:0', 'encoded:0', 'snapshots:0', 'adapter:0']);
});

test('SA3 chips report verified assets only', () => {
  const sa3 = { audioPrepared: true, preparedClips: 12, encodingCacheValid: false, encodedClips: 12, snapshots: 2,
    adapter: { name: 'x', trainedAt: 'T5', run: null } };
  assert.deepEqual(summary('sa3', { ...base, sa3 } as DatasetAssets), ['audio:1:12', 'encoded:0', 'snapshots:1:2', 'adapter:1']);
  const r = sa3Readiness({ ...base, sa3 } as DatasetAssets);
  assert.deepEqual(r, { prepared: false, trained: true });
});

test('HeartMuLa chips report the verified prepare cache and published adapter', () => {
  assert.deepEqual(summary('heartmula', base), ['codes:0', 'adapter:0']);
  const heartmula = { prepared: true, preparedSamples: 8, adapter: { name: 'x', trainedAt: 'T5', run: null } };
  assert.deepEqual(summary('heartmula', { ...base, heartmula } as DatasetAssets), ['codes:1:8', 'adapter:1']);
  assert.deepEqual(heartmulaReadiness({ ...base, heartmula } as DatasetAssets), { prepared: true, trained: true });
  assert.deepEqual(heartmulaReadiness(undefined), { prepared: false, trained: false });
});
