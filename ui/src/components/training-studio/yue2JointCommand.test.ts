import assert from 'node:assert/strict';
import { test } from 'node:test';
import { YUE2_JOINT_FORM_DEFAULTS } from '../../../../server/src/contracts/trainingRecipes';
import type { Yue2AitkPrepareRequest, Yue2JointTrainRequest } from '../../services/trainingApi';
import { captureJointOverrides, submitJointStart } from './yue2JointCommand';
import type { Yue2PreparationSummary } from '../../services/trainingPreparationApi';

test('joint command captures current form and preparation before the chain starts', () => {
  const form = { ...YUE2_JOINT_FORM_DEFAULTS, saveEvery: 17,
    preview: { ...YUE2_JOINT_FORM_DEFAULTS.preview, seconds: 12, parallel: true } } as Yue2JointTrainRequest;
  const prepare = { force: false } as Yue2AitkPrepareRequest;
  const captured = captureJointOverrides(form, prepare, true, 'CUDA0', '');
  form.saveEvery = 99;
  prepare.force = true;
  assert.equal(captured.saveEvery, 17);
  assert.deepEqual(captured.preparation, { force: false });
  assert.equal(captured.device, 'CUDA0');
  assert.equal(captured.lyricTiming, true);
  assert.equal(captured.alignmentEnabled, true);
  assert.equal(captured.autoPrepare, true);
  assert.equal((captured.preview as { everySteps: number }).everySteps, 0);
  assert.equal((captured.preview as { previewMaxFrames: number }).previewMaxFrames, 300);
});

test('joint resume retains its selected run and skips automatic preparation', () => {
  const form = { ...YUE2_JOINT_FORM_DEFAULTS } as Yue2JointTrainRequest;
  const captured = captureJointOverrides(form, {} as Yue2AitkPrepareRequest, false, '', 'run-7|42');
  assert.equal(captured.resumeRunId, 'run-7');
  assert.equal(captured.resumeStep, 42);
  assert.equal(captured.autoPrepare, false);
  assert.equal(captured.cursorWeight, 0);
});

test('direct start captures one joint-only operation with the current source and form', async () => {
  const overrides = { steps: 71, lyricTiming: false, autoPrepare: true };
  const received: unknown[] = [];
  const runner = {
    getContext: async () => ({ dataset: { id: 'album', revision: 7 },
      source: { kind: 'dataset-sources' as const, id: 'album', revision: 'source-3' } }),
    start: async (snapshot: Parameters<typeof submitJointStart>[2] extends { start: (s: infer S) => unknown } ? S : never) => {
      received.push(snapshot);
      return { id: 'pipeline-1' } as Yue2PreparationSummary;
    },
  };
  const result = await submitJointStart('album', overrides, runner, 'same-click');
  overrides.steps = 99;
  assert.equal(result.id, 'pipeline-1');
  assert.equal(received.length, 1);
  const snapshot = received[0] as { operation: { idempotencyKey: string }; dataset: { revision: number };
    sources: { revision: string }[]; payload: { mode: string; stages: string[]; recipes: { joint: { overrides: { steps: number }; version: number } } } };
  assert.equal(snapshot.operation.idempotencyKey, 'same-click');
  assert.equal(snapshot.dataset.revision, 7);
  assert.equal(snapshot.sources[0].revision, 'source-3');
  assert.equal(snapshot.payload.mode, 'train-after-preparation');
  assert.deepEqual(snapshot.payload.stages, ['joint']);
  assert.equal(snapshot.payload.recipes.joint.overrides.steps, 71);
  assert.equal(typeof snapshot.payload.recipes.joint.version, 'number');
});

test('direct start keeps the clicked worker and form while context is pending', async () => {
  const overrides = { steps: 71, preview: { seconds: 12 } };
  let releaseContext!: (context: Awaited<ReturnType<Parameters<typeof submitJointStart>[2]['getContext']>>) => void;
  const context = new Promise<Awaited<ReturnType<Parameters<typeof submitJointStart>[2]['getContext']>>>(resolve => {
    releaseContext = resolve;
  });
  let submitted: ReturnType<typeof import('./yue2JointCommand').captureJointStart> | undefined;
  let selectedWorker = { kind: 'remote' as const, name: 'worker-a' };
  const pending = submitJointStart('album', overrides, {
    getContext: async () => context,
    start: async snapshot => {
      submitted = snapshot;
      return { id: 'pipeline-1' } as Yue2PreparationSummary;
    },
  }, 'clicked', selectedWorker);
  selectedWorker = { kind: 'remote', name: 'worker-b' };
  overrides.steps = 99;
  overrides.preview.seconds = 60;
  releaseContext({ dataset: { id: 'album', revision: 7 },
    source: { kind: 'dataset-sources', id: 'album', revision: 'source-3' } });
  await pending;
  assert.equal(selectedWorker.name, 'worker-b');
  assert.deepEqual(submitted?.worker, { kind: 'remote', name: 'worker-a' });
  assert.deepEqual(submitted?.payload.recipes.joint.overrides.preview, { seconds: 12 });
  assert.equal(submitted?.payload.recipes.joint.overrides.steps, 71);
});

test('direct start surfaces context and admission failures without a second submission', async () => {
  let calls = 0;
  const failContext = { getContext: async () => { throw new Error('stale source'); },
    start: async () => { calls++; return {} as Yue2PreparationSummary; } };
  await assert.rejects(submitJointStart('album', {}, failContext, 'stale'), /stale source/);
  assert.equal(calls, 0);
  const failAdmission = { getContext: async () => ({ dataset: { id: 'album', revision: 7 },
    source: { kind: 'dataset-sources' as const, id: 'album', revision: 'source-3' } }),
    start: async () => { calls++; throw new Error('Run preparation first'); } };
  await assert.rejects(submitJointStart('album', {}, failAdmission, 'missing'), /Run preparation first/);
  assert.equal(calls, 1);
});

test('direct start preserves active-pipeline and prerequisite refusals from the runner', async () => {
  const context = { dataset: { id: 'album', revision: 7 },
    source: { kind: 'dataset-sources' as const, id: 'album', revision: 'source-3' } };
  for (const reason of ['Active preparation pipeline for this dataset',
    'Missing latent cache; use Perform all stages', 'Dataset changed before admission']) {
    let submissions = 0;
    const runner = { getContext: async () => context, start: async () => {
      submissions++;
      throw new Error(reason);
    } };
    await assert.rejects(submitJointStart('album', { steps: 71 }, runner, reason),
      error => error instanceof Error && error.message === reason);
    assert.equal(submissions, 1);
  }
});
