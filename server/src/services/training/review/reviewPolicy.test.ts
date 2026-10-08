import assert from 'node:assert/strict';
import test from 'node:test';
import { narContinuationRequest, reviewUseDecision } from './reviewPolicy.js';

test('review keeps the NAR skip/reject/chain decisions on Node', () => {
  const local = { options: {} };
  const remote = { options: {}, origin: { worker: 'worker', remoteJobId: 'job' } };
  assert.equal(reviewUseDecision(local, false), 'skip');
  assert.equal(reviewUseDecision(remote, false), 'skip');
  assert.equal(reviewUseDecision(remote, true), 'reject');
  assert.equal(reviewUseDecision({ options: { freezePlannerNow: true } }, true), 'skip');
  assert.equal(reviewUseDecision(local, true), 'chain');
});

test('NAR continuation preserves the former full request and custom controls', () => {
  const body = narContinuationRequest('run', 40, { budget: 250, lrScale: 0.2,
    keepDelta: 0.003, target: 0.25, knee: true });
  assert.deepEqual(body, {
    trainingMethod: 'aitk', refine: true, resumeRunId: 'run', resumeStep: 40,
    steps: 290, saveEvery: 10, stopMode: 'kl', narExtraSteps: 290, freezePlannerNow: true,
    narLrScale: 0.2, reconStop: 0.005, reconStopWindow: 10, reconKeepDelta: 0.003,
    reconTarget: 0.25, stopEngine: false, lyricTiming: true, alignmentEnabled: true,
    autoPrepare: false, checkpoint: '', output: '',
    preview: { enabled: false, everySteps: 0, seconds: 90, seed: 424242,
      previewMaxFrames: 2250, baseline: false, control: false },
  });
  const blank = narContinuationRequest('run', 40, { budget: 10, lrScale: 0.05,
    keepDelta: 0, target: null, knee: false });
  assert.equal(blank.steps, 50);
  assert.equal(blank.reconStop, 0);
  assert.equal('reconTarget' in blank, false);
});
