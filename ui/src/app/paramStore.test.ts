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
const { setParam, resetParam, isEditableParam, defaultOf, paramHistory, readParam, snapshot } = await import('./paramStore');

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

test('undo after a backend switch restores the backend that was edited, not the active one', async () => {
  const { useBackendStore } = await import('../stores/backendStore');
  useBackendStore.setState({ activeBackendId: 'ace' });
  useGlobalParamsStore.getState().setInferMethod('euler');
  setParam('inferMethod', 'heun');
  useBackendStore.setState({ activeBackendId: 'mm3' });
  const mm3 = useGlobalParamsStore.getState().inferMethod;
  paramHistory.undo();
  assert.equal(storage.get('hs-inferMethod'), '"euler"');
  assert.equal(useGlobalParamsStore.getState().inferMethod, mm3, 'the live (mm3) value is untouched');
  assert.notEqual(storage.get('hs-inferMethod@mm3'), '"euler"');
  useBackendStore.setState({ activeBackendId: 'ace' });
});

test('only declared ids read; store actions and objects do not leak', () => {
  assert.equal(readParam(useGlobalParamsStore.getState(), 'setSeed'), undefined);
  assert.equal(readParam(useGlobalParamsStore.getState(), 'getGlobalParams'), undefined);
  assert.equal(readParam(useGlobalParamsStore.getState(), 'backendParams.__proto__'), undefined);
  const copy = snapshot(useGlobalParamsStore.getState().adapterGroupScales);
  assert.notEqual(copy, useGlobalParamsStore.getState().adapterGroupScales);
});
