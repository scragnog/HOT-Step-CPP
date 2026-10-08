import assert from 'node:assert/strict';
import { test } from 'node:test';
import { YUE2_JOINT_FORM_DEFAULTS } from '../../../../server/src/contracts/trainingRecipes';
import type { Yue2AitkPrepareRequest, Yue2JointTrainRequest } from '../../services/trainingApi';
import { captureJointOverrides } from './yue2JointCommand';

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
