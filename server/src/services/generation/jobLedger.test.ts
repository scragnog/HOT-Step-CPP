import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { GenerationJobLedger, INTERRUPTED_ERROR, ledgerStatusBody } from './jobLedger.js';

const job = (id: string, createdAt = 1000) => ({ id, userId: 'u', createdAt });

test('a restart turns a running job into a visible failure, and finished jobs keep their outcome', () => {
  const db = new Database(':memory:');
  let clock = 1000;
  const before = new GenerationJobLedger(db, () => clock);
  before.opened(job('running'));
  before.opened(job('queued'));
  before.opened(job('done'));
  before.finished({ id: 'done', status: 'succeeded', result: { audioUrls: ['/audio/a.wav'], songIds: ['s1'] } as never });
  before.opened(job('stopped'));
  before.finished({ id: 'stopped', status: 'cancelled' });
  before.opened(job('broke'));
  before.finished({ id: 'broke', status: 'failed', error: 'Engine error' });
  before.finished({ id: 'running', status: 'running' as never });
  assert.equal(before.get('running')!.status, 'open', 'an active job stays open');

  // The process dies here. A new process builds its ledger on the same database.
  clock = 5000;
  const after = new GenerationJobLedger(db, () => clock);
  for (const id of ['running', 'queued']) {
    const entry = after.get(id)!;
    assert.deepEqual([entry.status, entry.error, entry.finishedAt], ['interrupted', INTERRUPTED_ERROR, 5000]);
    const body = ledgerStatusBody(entry);
    assert.equal(body.status, 'failed');
    assert.equal(body.interrupted, true);
    assert.equal(body.stage, 'Interrupted');
    // The browser queue stops polling only on an error containing "failed".
    assert.match(String(body.error), /failed/);
    assert.match(String(body.error), /not resubmitted/);
  }
  assert.deepEqual(ledgerStatusBody(after.get('done')!).result, { audioUrls: ['/audio/a.wav'], songIds: ['s1'] });
  assert.equal(ledgerStatusBody(after.get('done')!).status, 'succeeded');
  assert.equal(ledgerStatusBody(after.get('stopped')!).status, 'cancelled');
  assert.deepEqual([ledgerStatusBody(after.get('broke')!).status, ledgerStatusBody(after.get('broke')!).error], ['failed', 'Engine error']);
  assert.equal(after.get('nope'), null);

  // A job opened by the new process is not touched by its own startup pass.
  after.opened(job('new', 5000));
  assert.equal(after.get('new')!.status, 'open');
});

test('rows older than a week are pruned at startup', () => {
  const db = new Database(':memory:');
  const week = 7 * 24 * 60 * 60 * 1000;
  const first = new GenerationJobLedger(db, () => 0);
  first.opened(job('old', 0));
  first.opened(job('recent', week));
  new GenerationJobLedger(db, () => week + 1);
  assert.equal(new GenerationJobLedger(db, () => week + 1).get('old'), null);
  assert.equal(new GenerationJobLedger(db, () => week + 1).get('recent')!.status, 'interrupted');
});
