import assert from 'node:assert/strict';
import test from 'node:test';
import { engineNotice, noModelsRefetchInterval, NO_MODELS_POLL_MS } from './noModelsPoll';

const snap = (ready: boolean, noModels: boolean) => ({
  state: { data: { engine: { ready, bootStatus: '', noModels } }, status: 'success', errorUpdateCount: 0 },
}) as any;

test('keeps polling through ready=true so a later noModels=true is seen', () => {
  assert.equal(noModelsRefetchInterval(snap(true, false)), NO_MODELS_POLL_MS);   // startup snapshot
  assert.equal(noModelsRefetchInterval(snap(false, true)), NO_MODELS_POLL_MS);   // after the scan
});

test('chooses failure before missing models, and stays silent for healthy or starting engines', () => {
  assert.equal(engineNotice({ failed: true, noModels: false }), 'failed');
  assert.equal(engineNotice({ failed: true, noModels: true }), 'failed');
  assert.equal(engineNotice({ failed: false, noModels: true }), 'no-models');
  assert.equal(engineNotice({ failed: false, noModels: false }), null);
  assert.equal(engineNotice(undefined), null);
});

test('keeps polling through failure and recovery', () => {
  const query = snap(false, false);
  query.state.data.engine.failed = true;
  assert.equal(noModelsRefetchInterval(query), NO_MODELS_POLL_MS);
  query.state.data.engine.failed = false;
  assert.equal(noModelsRefetchInterval(query), NO_MODELS_POLL_MS);
});
