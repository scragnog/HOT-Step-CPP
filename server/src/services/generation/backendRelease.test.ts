import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { trackBackendRelease, awaitBackendRelease, _resetBackendReleases } from './backendRelease.js';

// #204: no render loads weights while any backend switch's eviction is unconfirmed.

const tick = () => new Promise(r => setTimeout(r, 5));
function deferred() {
  let resolve!: () => void; let reject!: (e: Error) => void;
  const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

beforeEach(() => _resetBackendReleases());

test('a later switch cannot hide an earlier failed eviction (YuE2 -> ACE -> MM3)', async () => {
  let yue2Calls = 0;
  trackBackendRelease('yue2', async () => { yue2Calls++; throw new Error('unload not confirmed'); });
  trackBackendRelease('ace', async () => {});
  await assert.rejects(awaitBackendRelease(), /\(yue2\) has not confirmed/);
  assert.equal(yue2Calls, 2);  // the switch's attempt, then one retry
  // Still tracked: the next render tries again rather than loading next to it.
  await assert.rejects(awaitBackendRelease(), /\(yue2\)/);
});

test('a switch made while a render waits is waited for too', async () => {
  const a = deferred();
  const b = deferred();
  let bDone = false;
  trackBackendRelease('yue2', () => a.promise);
  let waited = false;
  const wait = awaitBackendRelease().then(() => { waited = true; });
  await tick();
  trackBackendRelease('minimax-m3', () => b.promise.then(() => { bDone = true; }));
  a.resolve();
  await tick();
  assert.equal(waited, false, 'returned after the older eviction while the newer one was pending');
  b.resolve();
  await wait;
  assert.equal(bDone, true);
});

test('a failed eviction the engine finished later passes on the retry', async () => {
  let calls = 0;
  trackBackendRelease('yue2', async () => { if (++calls === 1) throw new Error('client timed out'); });
  await awaitBackendRelease();
  assert.equal(calls, 2);
  await awaitBackendRelease();  // nothing left
});

test('no tracked eviction means no wait', async () => {
  await awaitBackendRelease();
});
