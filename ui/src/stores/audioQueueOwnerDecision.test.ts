import test from 'node:test';
import assert from 'node:assert/strict';
import { decideQueueOwnerAction, shouldShowMigrationBanner } from './audioQueueOwnerDecision.js';

test('decideQueueOwnerAction: a never-decided, empty profile goes straight to server-owned', () => {
  assert.equal(decideQueueOwnerAction(null, false), 'set-server');
  assert.equal(decideQueueOwnerAction('garbage', false), 'set-server');
});

test('decideQueueOwnerAction: a never-decided profile with pending work migrates', () => {
  assert.equal(decideQueueOwnerAction(null, true), 'migrate');
  assert.equal(decideQueueOwnerAction('garbage', true), 'migrate');
});

test('decideQueueOwnerAction: server and browser are terminal, regardless of pending work', () => {
  assert.equal(decideQueueOwnerAction('server', true), 'none');
  assert.equal(decideQueueOwnerAction('server', false), 'none');
  assert.equal(decideQueueOwnerAction('browser', true), 'none');
  assert.equal(decideQueueOwnerAction('browser', false), 'none');
});

test('decideQueueOwnerAction: a migration interrupted by reload is retried, not left stuck', () => {
  // Reload after OWNER_KEY is set to 'migrating' used to leave resumeQueue()
  // returning early forever. It must resolve to a real decision instead.
  assert.equal(decideQueueOwnerAction('migrating', true), 'migrate');
  assert.equal(decideQueueOwnerAction('migrating', false), 'set-server');
});

test('shouldShowMigrationBanner: hidden for a successful/silent outcome', () => {
  assert.equal(shouldShowMigrationBanner(true, true, 'server', 5), false, 'server-owned, even with history, stays silent');
  assert.equal(shouldShowMigrationBanner(true, true, 'browser', 0), false, 'nothing pending, nothing to show');
});

test('shouldShowMigrationBanner: hidden before the owner decision has landed, or with no token', () => {
  assert.equal(shouldShowMigrationBanner(true, false, 'browser', 3), false, 'decision still in flight — no manual-prompt flash');
  assert.equal(shouldShowMigrationBanner(false, true, 'browser', 3), false);
});

test('shouldShowMigrationBanner: shown only once a failed migration leaves pending work stuck browser-side', () => {
  assert.equal(shouldShowMigrationBanner(true, true, 'browser', 3), true);
});
