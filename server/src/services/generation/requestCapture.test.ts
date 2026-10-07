import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGenerateCapture, generateCaptureMode, redactCredentials, REDACTED, GENERATE_CAPTURE_SCHEMA } from './requestCapture.js';

const ctx = {
  mode: 'record' as const, commit: 'abc123', dirty: false, activeBackendId: 'yue2',
  refererPath: '/create', now: new Date('2026-10-07T20:00:00Z'), id: 'fixed-id',
};

function sampleBody() {
  return {
    source: 'create', backend: 'yue2', caption: 'a caption', lyrics: '[Verse]\nline',
    seed: 0, randomSeed: false, lmSeed: '18446744073709551615', duration: 0, instrumental: false,
    maxTokens: 4096, apiKey: 'sk-secret', nested: { hfToken: 'hf_x', authToken: '', items: [{ password: 'p' }] },
    adapters: [{ path: 'a.safetensors', scale: 1 }],
  };
}

test('capture mode is off unless a dev server names a mode', () => {
  const dev = { HOT_STEP_DEV: '1' };
  assert.equal(generateCaptureMode({}), 'off');
  assert.equal(generateCaptureMode({ ...dev }), 'off');
  assert.equal(generateCaptureMode({ ...dev, HOTSTEP_GENERATE_CAPTURE: '1' }), 'off');
  assert.equal(generateCaptureMode({ ...dev, HOTSTEP_GENERATE_CAPTURE: 'record' }), 'record');
  assert.equal(generateCaptureMode({ ...dev, HOTSTEP_GENERATE_CAPTURE: ' Capture-Only ' }), 'capture-only');
  // not under dev.bat: the end-user launchers never set HOT_STEP_DEV
  assert.equal(generateCaptureMode({ HOTSTEP_GENERATE_CAPTURE: 'record' }), 'off');
});

test('capture never alters the request body it records', () => {
  const body = sampleBody();
  const before = structuredClone(body);
  const fixture = buildGenerateCapture(body, ctx);
  assert.deepEqual(body, before);
  // and the fixture holds a copy, not a reference
  assert.notEqual(fixture.body, body);
  (fixture.body as any).caption = 'changed';
  (fixture.body as any).adapters[0].scale = 9;
  assert.deepEqual(body, before);
});

test('credential-like keys are redacted, ordinary fields and falsy values are kept', () => {
  const fixture = buildGenerateCapture(sampleBody(), ctx);
  const b = fixture.body as any;
  assert.equal(b.apiKey, REDACTED);
  assert.equal(b.nested.hfToken, REDACTED);
  assert.equal(b.nested.items[0].password, REDACTED);
  assert.equal(b.nested.authToken, '');          // empty: nothing to hide
  assert.equal(b.maxTokens, 4096);               // "tokens" is not a credential
  assert.equal(b.seed, 0);
  assert.equal(b.randomSeed, false);
  assert.equal(b.duration, 0);
  assert.deepEqual(fixture.redactedPaths.sort(), ['apiKey', 'nested.hfToken', 'nested.items[0].password']);
});

test('fixture records commit, caller, settings and seed', () => {
  const fixture = buildGenerateCapture(sampleBody(), { ...ctx, callerLabel: 'harness' });
  assert.equal(fixture.schema, GENERATE_CAPTURE_SCHEMA);
  assert.equal(fixture.id, 'fixed-id');
  assert.equal(fixture.capturedAt, '2026-10-07T20:00:00.000Z');
  assert.equal(fixture.commit, 'abc123');
  assert.deepEqual(fixture.caller, { source: 'create', label: 'harness', refererPath: '/create' });
  assert.deepEqual(fixture.settings, { activeBackendId: 'yue2', submittedBackend: 'yue2' });
  assert.deepEqual(fixture.seed, { seed: 0, randomSeed: false, lmSeed: '18446744073709551615' });
});

test('a missing or non-object body still produces a fixture', () => {
  for (const body of [undefined, null, 'text', [1, 2]]) {
    const fixture = buildGenerateCapture(body, ctx);
    assert.equal(fixture.caller.source, null);
    assert.deepEqual(fixture.seed, { seed: undefined, randomSeed: undefined, lmSeed: undefined });
  }
});

test('redactCredentials leaves its input untouched', () => {
  const input = { token: 't', list: [{ secret: 's' }] };
  const before = structuredClone(input);
  const { value } = redactCredentials(input);
  assert.deepEqual(input, before);
  assert.deepEqual(value, { token: REDACTED, list: [{ secret: REDACTED }] });
});
