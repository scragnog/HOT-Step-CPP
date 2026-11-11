import assert from 'node:assert/strict';
import test from 'node:test';
import router from './health.js';
import { aceClient } from '../services/aceClient.js';
import { setEngineFailed, setEngineNoModels, setEngineReady } from '../engineState.js';
import type { HealthResponse } from '../contracts/health.js';

test('health exposes engine failure and its reason independently of missing models', async t => {
  t.mock.method(aceClient, 'health', async () => ({ status: 'ok' }));
  t.mock.method(aceClient, 'props', async () => ({}));
  const handler = router.stack.find(layer => layer.route?.path === '/')!.route!.stack[0].handle;
  try {
    for (const [failed, noModels, ready, reason] of [
      [true, false, false, 'ace-server not found at C:/fixture/ace-server.exe'],
      [true, true, false, 'Engine crashed'],
      [false, true, false, 'No models installed'],
      [false, false, false, 'Starting engine...'],
      [false, false, true, 'Ready'],
    ] as const) {
      setEngineFailed(failed);
      setEngineNoModels(noModels);
      setEngineReady(ready, reason);
      let response: HealthResponse | undefined;
      await handler({} as any, { json: (body: HealthResponse) => { response = body; } } as any, () => {});
      assert.equal(response?.engine.failed, failed);
      assert.equal(response?.engine.noModels, noModels);
      assert.equal(response?.engine.ready, ready);
      assert.equal(response?.engine.bootStatus, reason);
    }
  } finally {
    setEngineFailed(false);
    setEngineNoModels(false);
    setEngineReady(false, 'Initializing...');
  }
});
