import assert from 'node:assert/strict';
import test from 'node:test';
import { setTrainingWorker } from './trainingApi';
import { assertRecipeWorker, getTrainingRecipe, resolveTrainingRecipe } from './trainingRecipesApi';

test('recipe reads follow the selected worker; resolution captures it and preserves zero and false', async () => {
  const saved = globalThis.fetch;
  const calls: Array<{ url: string; body?: any }> = [];
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    return new Response(JSON.stringify({ recipeVersion: 1, family: 'mm3-lm', worker: { kind: 'remote', name: 'Den' }, builtin: {}, stored: {},
      resolved: {}, execution: { steps: 0, keepResumeState: false }, deferred: [] }), { status: 200 });
  }) as typeof fetch;
  try {
    setTrainingWorker('Den');
    await getTrainingRecipe('mm3-lm', 'balanced');
    const resolved = await resolveTrainingRecipe('mm3-lm', { steps: 0, keepResumeState: false }, 'balanced');
    assert.doesNotThrow(() => assertRecipeWorker(resolved));
    setTrainingWorker(null);
    assert.throws(() => assertRecipeWorker(resolved), /worker changed/);
    assert.equal(calls[0].url, '/api/training/ops/recipes/mm3-lm?preset=balanced&worker=Den');
    assert.equal(calls[1].url, '/api/training/ops/recipes/resolve');
    assert.deepEqual(calls[1].body.worker, { kind: 'remote', name: 'Den' });
    assert.equal(calls[1].body.payload.recipeVersion, 1);
    assert.deepEqual(calls[1].body.payload.overrides, { steps: 0, keepResumeState: false });
  } finally {
    globalThis.fetch = saved;
    setTrainingWorker(null);
  }
});
