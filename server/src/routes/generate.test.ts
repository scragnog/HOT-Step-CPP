import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { expectedBackendMismatch, markGenerationJobCancelled, setGenerationJobLedgerForTests } from './generate.js';
import { GenerationJobLedger } from '../services/generation/jobLedger.js';

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

test('a job cancelled before its lane turn stays cancelled across a restart', () => {
  const db = new Database(':memory:');
  const ledger = new GenerationJobLedger(db);
  setGenerationJobLedgerForTests(ledger);
  try {
    const job = { id: 'waiting', userId: 'u', createdAt: Date.now(), status: 'pending' } as Parameters<typeof markGenerationJobCancelled>[0];
    ledger.opened(job);
    markGenerationJobCancelled(job);
    assert.equal(job.status, 'cancelled');
    // Restart before the lane ever ran the job: a new process opens the same database.
    assert.equal(new GenerationJobLedger(db).get('waiting')!.status, 'cancelled');
  } finally { setGenerationJobLedgerForTests(null); }
});
