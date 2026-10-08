import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { WorkflowError } from '../workflows/workflowJobs.js';
import { TypedDocuments, WorkflowDocuments, type TypedDocumentKind } from '../workflows/revisions.js';
import { getSingleton, importPreset, importSingleton, LOCAL_OWNER, upsertSingleton } from './presets.js';
import { scaleOverridePresetBodySchema, type ScaleOverridePresetBody } from '../../contracts/preferences.js';
import { aiContinueTemplateBodySchema, type AiContinueTemplateBody } from '../../contracts/preferences.js';

const uniq = (() => { let n = 0; return () => `t.preset-test-${process.pid}-${n++}`; })();
const uniqSingleton = (() => { let n = 0; return () => `t.singleton-test-${process.pid}-${n++}`; })();

function presetStore(): TypedDocuments<ScaleOverridePresetBody> {
  const db = new Database(':memory:');
  const def: TypedDocumentKind<ScaleOverridePresetBody> = {
    kind: uniq(), scope: 'installation', schemaVersion: 1, schema: scaleOverridePresetBodySchema,
  };
  return new TypedDocuments(new WorkflowDocuments(db), db, def);
}
function singletonStore(): TypedDocuments<AiContinueTemplateBody> {
  const db = new Database(':memory:');
  const def: TypedDocumentKind<AiContinueTemplateBody> = {
    kind: uniqSingleton(), scope: 'installation', schemaVersion: 1, schema: aiContinueTemplateBodySchema,
  };
  return new TypedDocuments(new WorkflowDocuments(db), db, def);
}
const nameOf = (b: ScaleOverridePresetBody) => b.name;
const preset = (name: string, overallScale = 1): ScaleOverridePresetBody =>
  ({ name, overallScale, groupScales: { self_attn: 1, cross_attn: 1, mlp: 1, cond_embed: 1 } });

test('importPreset: a fresh name imports; retrying the same source is a no-op, not a duplicate', () => {
  const td = presetStore();
  const item = { storageKey: 'k', sourceHash: 'sha256:a', body: preset('Light touch') };
  const first = importPreset(td, nameOf, item);
  assert.equal(first.outcome, 'imported');
  const again = importPreset(td, nameOf, item);
  assert.equal(again.outcome, 'unchanged');
  assert.equal(again.documentId, first.documentId);
  assert.equal(td.list(LOCAL_OWNER).length, 1);
});

test('importPreset: same name, different content refuses without a resolution; nothing is written', () => {
  const td = presetStore();
  importPreset(td, nameOf, { storageKey: 'k1', sourceHash: 'sha256:a', body: preset('Light touch', 0.5) });
  const conflict = importPreset(td, nameOf, { storageKey: 'k2', sourceHash: 'sha256:b', body: preset('Light touch', 0.9) });
  assert.equal(conflict.outcome, 'name-conflict');
  assert.equal(td.list(LOCAL_OWNER).length, 1);
  assert.equal(td.list(LOCAL_OWNER)[0]!.body.overallScale, 0.5);
});

test('importPreset: resolution "replace" updates the existing document in place', () => {
  const td = presetStore();
  const original = importPreset(td, nameOf, { storageKey: 'k1', sourceHash: 'sha256:a', body: preset('Light touch', 0.5) });
  const replaced = importPreset(td, nameOf, {
    storageKey: 'k2', sourceHash: 'sha256:b', body: preset('Light touch', 0.9), resolution: 'replace',
  });
  assert.equal(replaced.outcome, 'replaced');
  assert.equal(replaced.documentId, original.documentId);
  assert.equal(td.list(LOCAL_OWNER).length, 1);
  assert.equal(td.list(LOCAL_OWNER)[0]!.body.overallScale, 0.9);
});

test('importPreset: resolution "keep-both" imports a second document under the same name', () => {
  const td = presetStore();
  importPreset(td, nameOf, { storageKey: 'k1', sourceHash: 'sha256:a', body: preset('Light touch', 0.5) });
  const kept = importPreset(td, nameOf, {
    storageKey: 'k2', sourceHash: 'sha256:b', body: preset('Light touch', 0.9), resolution: 'keep-both',
  });
  assert.equal(kept.outcome, 'imported');
  assert.equal(td.list(LOCAL_OWNER).length, 2);
  assert.deepEqual(td.list(LOCAL_OWNER).map(d => d.body.overallScale).sort(), [0.5, 0.9]);
});

test('importPreset: a credential-free body that fails its own schema is refused before anything is written', () => {
  const td = presetStore();
  assert.throws(() => importPreset(td, nameOf, { storageKey: 'k', sourceHash: 'sha256:a', body: { name: 'x' } }), WorkflowError);
  assert.deepEqual(td.list(LOCAL_OWNER), []);
});

test('singleton: create on first write, update by revision after; a stale expectedRevision 409s', () => {
  const td = singletonStore();
  assert.equal(getSingleton(td), null);
  const created = upsertSingleton(td, undefined, { template: 'v1' });
  assert.equal(created.revision, 1);
  const updated = upsertSingleton(td, 1, { template: 'v2' });
  assert.equal(updated.revision, 2);
  assert.equal(getSingleton(td)!.body.template, 'v2');
  assert.throws(() => upsertSingleton(td, 1, { template: 'stale' }), (err: unknown) => err instanceof WorkflowError && err.status === 409);
});

test('singleton import: identical content is a no-op; different content needs resolution "replace"', () => {
  const td = singletonStore();
  const first = importSingleton(td, { storageKey: 'hs-ai-continue-template', sourceHash: 'sha256:a', body: { template: 'v1' } });
  assert.equal(first.outcome, 'imported');
  const same = importSingleton(td, { storageKey: 'hs-ai-continue-template', sourceHash: 'sha256:a', body: { template: 'v1' } });
  assert.equal(same.outcome, 'unchanged');
  const conflict = importSingleton(td, { storageKey: 'hs-ai-continue-template', sourceHash: 'sha256:b', body: { template: 'v2' } });
  assert.equal(conflict.outcome, 'name-conflict');
  assert.equal(getSingleton(td)!.body.template, 'v1');
  const replaced = importSingleton(td, {
    storageKey: 'hs-ai-continue-template', sourceHash: 'sha256:b', body: { template: 'v2' }, resolution: 'replace',
  });
  assert.equal(replaced.outcome, 'replaced');
  assert.equal(getSingleton(td)!.body.template, 'v2');
});
