import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizePpParams } from './postProcessing.js';

// The ONNX StableStep backend is retired: a saved 'onnx' must load as 'auto'
// (GGML), so no request built from an old profile can ask the engine for ONNX.
test('normalizePpParams: stableStepBackend keeps only gguf, everything else is auto', () => {
  const backend = (v: unknown) => normalizePpParams({ stableStepBackend: v }, []).stableStepBackend;
  assert.equal(backend('onnx'), 'auto');
  assert.equal(backend('gguf'), 'gguf');
  assert.equal(backend('auto'), 'auto');
  assert.equal(backend(undefined), 'auto');
  assert.equal(backend('trt'), 'auto');
});
