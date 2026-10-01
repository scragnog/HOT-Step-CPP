import test from 'node:test';
import assert from 'node:assert/strict';
import { expectedBackendMismatch } from './generate.js';

test('expectedBackendMismatch is null when the caller named no expectation', () => {
  assert.equal(expectedBackendMismatch({}, 'ace'), null);
  assert.equal(expectedBackendMismatch({ backend: 'yue2' }, 'ace'), null); // backend is log-only, not read here
  assert.equal(expectedBackendMismatch(null, 'ace'), null);
  assert.equal(expectedBackendMismatch(undefined, 'ace'), null);
});

test('expectedBackendMismatch is null when the expectation already matches the active backend', () => {
  assert.equal(expectedBackendMismatch({ expectedBackend: 'ace' }, 'ace'), null);
});

test('expectedBackendMismatch reports both ids when the active backend moved on', () => {
  assert.deepEqual(
    expectedBackendMismatch({ expectedBackend: 'yue2' }, 'ace'),
    { expectedBackend: 'yue2', activeBackend: 'ace' },
  );
});

test('expectedBackendMismatch ignores a non-string expectedBackend rather than comparing it', () => {
  for (const bad of [42, true, {}, [], null]) {
    assert.equal(expectedBackendMismatch({ expectedBackend: bad }, 'ace'), null);
  }
});
