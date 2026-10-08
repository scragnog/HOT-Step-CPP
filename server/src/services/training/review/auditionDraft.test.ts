import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import type { AuditionPreview, AuditionSideResult } from '../types.js';
import { createTrainingCreateDraft, getTrainingCreateDraft, mirroredGenerationDraft } from './auditionDraft.js';
import Database from 'better-sqlite3';
import { WorkflowDocuments } from '../../workflows/revisions.js';

const preview = {
  previewId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', datasetId: 'dataset', kind: 'ab',
  createdAt: '2026-01-01T00:00:00.000Z', seed: 73, caption: 'tagged caption',
  captionInput: 'plain caption', lyrics: 'lines', durationSec: 12, lmModel: 'lm',
  ditModel: 'dit', renderDitModel: 'render dit', vaeModel: 'vae', renderDitAdapter: 'dit adapter',
  renderSteps: 8, bpm: 120, keyscale: 'C', timesignature: '4',
  lmTemperature: 0.7, lmTopP: 0.8, lmCfgScale: 3, lmRepPenalty: 1.2,
} as AuditionPreview;
const sides = [
  { slot: 'base', lmAdapter: '', lmAdapterScale: 1, ok: true },
  { slot: 'adapter', lmAdapter: 'lm adapter', lmAdapterScale: 0.8, ok: true },
] as AuditionSideResult[];
type Former = { side: 'base' | 'adapter'; cell: 'bare' | 'adapter';
  content: Record<string, unknown>; params: Record<string, unknown>; settings: Record<string, unknown> };
const former = JSON.parse(readFileSync(new URL('./former-audition-drafts.json', import.meta.url), 'utf8')) as Former[];

for (const fixture of former) {
  test(`server draft matches former complete handoff for ${fixture.side}/${fixture.cell}`, () => {
    const side = sides.find(item => item.slot === fixture.side)!;
    const actual = mirroredGenerationDraft(preview, side, fixture.cell);
    assert.deepEqual(actual.content, fixture.content);
    assert.deepEqual(actual.params, fixture.params);
    assert.deepEqual(actual.settings, fixture.settings);
    assert.deepEqual(actual.source, { datasetId: preview.datasetId, previewId: preview.previewId,
      slot: fixture.side, cell: fixture.cell });
  });
}

test('older audition without captionInput uses the tagged caption on both sides', () => {
  const old = { ...preview, captionInput: undefined };
  assert.equal(mirroredGenerationDraft(old, sides[0], 'bare').content['hs-caption'], 'tagged caption');
  assert.equal(mirroredGenerationDraft(old, sides[1], 'bare').content['hs-caption'], 'tagged caption');
});

test('draft idempotency and ownership keep another client from replacing or reading the draft', () => {
  const db = new Database(':memory:');
  try {
    const docs = new WorkflowDocuments(db);
    const draft = mirroredGenerationDraft(preview, sides[1], 'adapter');
    const first = createTrainingCreateDraft('user-a', 'same-click', draft, docs);
    const retry = createTrainingCreateDraft('user-a', 'same-click', draft, docs);
    assert.equal(retry.id, first.id);
    assert.deepEqual(getTrainingCreateDraft('user-a', first.id, docs).data.content, draft.content);
    assert.throws(() => getTrainingCreateDraft('user-b', first.id, docs));
    assert.equal(createTrainingCreateDraft('user-b', 'same-click', draft, docs).id === first.id, false);
  } finally { db.close(); }
});