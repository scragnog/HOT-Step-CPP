import assert from 'node:assert/strict';
import test from 'node:test';
import { register } from 'node:module';

register('data:text/javascript,' + encodeURIComponent(
  'export async function load(url, ctx, next) { if (url.endsWith(".css")) return { format: "module", source: "export default {}", shortCircuit: true }; return next(url, ctx); }'));
const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => storage.set(key, value),
} });
const { useGlobalParamsStore } = await import('../stores/globalParamsStore');
const { setParam, resetParam, isEditableParam, defaultOf, paramHistory } = await import('./paramStore');

const steps = () => useGlobalParamsStore.getState().inferenceSteps;

test('params write through the store setters, so the hs-* key follows', () => {
  const start = steps();
  setParam('inferenceSteps', 33);
  assert.equal(steps(), 33);
  assert.equal(storage.get('hs-inferenceSteps'), '33');
  paramHistory.undo();
  assert.equal(steps(), start);
  assert.equal(storage.get('hs-inferenceSteps'), JSON.stringify(start));
  paramHistory.redo();
  assert.equal(steps(), 33);
});

test('reset writes the descriptor default and is undoable', () => {
  setParam('guidanceScale', 9);
  resetParam('guidanceScale');
  assert.equal(useGlobalParamsStore.getState().guidanceScale, defaultOf('guidanceScale'));
  paramHistory.undo();
  assert.equal(useGlobalParamsStore.getState().guidanceScale, 9);
});

test('a backend knob lands in the active backend bag', () => {
  setParam('backendParams.knob', 5);
  assert.equal(useGlobalParamsStore.getState().backendParams.knob, 5);
  paramHistory.undo();
  assert.equal(useGlobalParamsStore.getState().backendParams.knob, undefined);
});

test('content fields and unknown ids are refused', () => {
  assert.equal(isEditableParam('inferenceSteps'), true);
  assert.equal(isEditableParam('caption'), false);
  assert.throws(() => setParam('notAParam', 1), /not an editable parameter/);
  assert.throws(() => setParam('bpm', 120), /not an editable parameter/);
});
