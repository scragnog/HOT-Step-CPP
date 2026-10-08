import assert from 'node:assert/strict';
import test from 'node:test';
import { createPresetLoadGate } from './Yue2AitkTrainCard';

test('a delayed preset response cannot submit the old form through batch or exposed Start', async () => {
  const gate = createPresetLoadGate();
  let resolveResponse!: () => void;
  const response = new Promise<void>(resolve => { resolveResponse = resolve; });
  let commitForm!: () => void;
  const committed = new Promise<void>(resolve => { commitForm = resolve; });
  let form = { steps: 50 };
  const submitted: number[] = [];
  const startBatch = () => { if (!gate.pending) submitted.push(form.steps); };
  const exposedStart = async () => gate.capture(() => { submitted.push(form.steps); });

  const load = (async () => {
    assert.equal(gate.begin(), true);
    await response;
    form = { steps: 200 };
    await committed;
    gate.release();
  })();

  startBatch();
  await assert.rejects(exposedStart(), /Wait for the preset/);
  assert.deepEqual(submitted, []);

  resolveResponse();
  await response;
  startBatch();
  await assert.rejects(exposedStart(), /Wait for the preset/);
  assert.deepEqual(submitted, []);

  commitForm();
  await load;
  startBatch();
  await exposedStart();
  assert.deepEqual(submitted, [200, 200]);
});
