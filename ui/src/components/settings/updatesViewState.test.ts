import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { UpdateApplyResult, UpdatesStatus, UpdatePrepareStatus } from '../../services/updatesApi';
import { completedUpdate, updateRefusal, updatesViewState, updateVersion } from './updatesViewState';

function fixture(phase: UpdatePrepareStatus['phase'] = 'idle'): UpdatesStatus {
  return {
    channel: 'release', installed: { kind: 'portable', version: '1.5.0', tag: 'v1.5.0', sha: null, variant: 'cpu', buildInfo: null, buildInfoProblem: null, git: null },
    status: 'update-available', latest: { release: { tag: 'v1.6.0', name: 'Update', url: 'https://example.invalid/release', publishedAt: null }, master: null },
    gitRelation: 'unknown', notesUrl: 'https://example.invalid/release', lastSuccessAt: null, lastAttemptAt: null, stale: false, error: null, retryAfter: null, checking: false,
    prepare: { offered: ['idle', 'failed', 'cancelled'].includes(phase), reason: null, candidate: { id: 'candidate', tag: 'v1.6.0', sha: 'a'.repeat(40), variant: 'cpu', assetName: 'update.zip', bytes: 100 }, phase, downloadedBytes: 25, totalBytes: 100, error: null, preparedId: phase === 'ready' ? 'staged' : null },
    install: { supported: true, reason: null, phase: 'idle', lastResult: null },
    apply: { offered: false, reason: 'notices-only', blockers: [] },
  };
}

test('protocol 0 has no dead button and explains the restriction', () => {
  const s = fixture();
  s.prepare!.offered = false; s.prepare!.reason = 'Manual upgrade required'; s.install!.supported = false;
  assert.equal(updatesViewState(s).action, null);
  assert.equal(updatesViewState(s).reason, 'Manual upgrade required');
  s.prepare!.reason = null; s.install!.reason = 'Start with the launcher';
  assert.equal(updatesViewState(s).reason, 'Start with the launcher');
});
test('git checkout gives update.bat instructions; portable hides Master and resets it once requested', () => {
  const s = fixture(); s.installed.kind = 'git';
  assert.equal(updatesViewState(s).gitInstructions, true);
  assert.equal(updatesViewState(s).action, null);
  assert.equal(updatesViewState(s).showChannel, true);
  s.installed.kind = 'portable'; s.channel = 'master';
  assert.equal(updatesViewState(s).showChannel, false);
  assert.equal(updatesViewState(s).resetToRelease, true);
  assert.equal(updatesViewState(s).offered, false);
});
for (const phase of ['idle', 'checking', 'downloading', 'verifying', 'extracting', 'ready', 'failed', 'cancelled'] as const) {
  test(`prepare ${phase}`, () => {
    const s = fixture(phase); s.prepare!.error = 'Checksum mismatch';
    const v = updatesViewState(s);
    const active = ['checking', 'downloading', 'verifying', 'extracting'].includes(phase);
    assert.equal(v.active, active);
    assert.equal(v.action, phase === 'ready' ? 'install' : active ? null : 'download');
    assert.equal(v.progress, 0.25);
    assert.equal(v.prepareError, phase === 'failed' ? 'Checksum mismatch' : null);
  });
}
test('all offer guards must hold and missing byte totals are indeterminate', () => {
  for (const change of [
    (s: UpdatesStatus) => { s.status = 'up-to-date'; },
    (s: UpdatesStatus) => { s.prepare!.candidate = null; },
    (s: UpdatesStatus) => { s.prepare!.offered = false; },
    (s: UpdatesStatus) => { s.install!.supported = false; },
    (s: UpdatesStatus) => { delete s.prepare; delete s.install; },
  ]) { const s = fixture(); change(s); assert.equal(updatesViewState(s).offered, false); }
  const s = fixture(); s.prepare!.totalBytes = 0;
  assert.equal(updatesViewState(s).progress, null);
  s.prepare!.totalBytes = 1;
  assert.equal(updatesViewState(s).progress, 1);
});
for (const [status, body, blockers, errors] of [
  [409, { error: 'Busy', blockers: ['Training is running'] }, ['Training is running'], []],
  [400, { error: 'Preflight', errors: ['Not enough disk space'] }, [], ['Not enough disk space']],
  [409, { error: 'Candidate expired' }, [], []],
  [403, { error: 'Cross-origin request refused' }, [], []],
] as const) {
  test(`refusal ${status}: ${body.error}`, () => {
    const refusal = updateRefusal({ status, body });
    assert.deepEqual(refusal, { error: body.error, blockers, errors });
    assert.equal(updatesViewState(fixture('ready'), refusal).action, 'install');
  });
}
for (const outcome of ['installed', 'rolled-back', 'recovery-required', 'failed'] as const) {
  test(`last result ${outcome} persists and completes only the matching new attempt`, () => {
    const s = fixture();
    const result: UpdateApplyResult = { outcome, from: 'v1.5.0-0123456789ab-cpu', to: 'v1.6.0-abcdef012345-cpu', at: '2026-11-11T20:00:00Z', message: 'Launcher details' };
    s.install!.lastResult = result;
    assert.deepEqual(updatesViewState(s).lastResult, result);
    assert.equal(updateVersion(result), 'v1.6.0');
    const attempt = { from: result.from, to: result.to, previousResult: null, startedAt: 1 };
    assert.deepEqual(completedUpdate(s, attempt), result);
    assert.equal(completedUpdate(s, { ...attempt, previousResult: { ...result } }), null);
    assert.equal(completedUpdate(s, { ...attempt, to: 'different' }), null);
    s.install!.phase = 'stopping';
    assert.equal(completedUpdate(s, attempt), null);
  });
}
