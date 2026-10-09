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

test('decideQueueOwnerAction: a migration interrupted by reload always retries, regardless of pending work', () => {
  // Reload after OWNER_KEY is set to 'migrating' used to leave resumeQueue()
  // returning early forever, or (an earlier fix) fall back to 'browser' and
  // risk resubmitting work the server might already own. It must always
  // retry instead — never silently decide 'browser' is safe again.
  assert.equal(decideQueueOwnerAction('migrating', true), 'retry');
  assert.equal(decideQueueOwnerAction('migrating', false), 'retry');
});

test('shouldShowMigrationBanner: hidden for a successful/silent outcome', () => {
  assert.equal(shouldShowMigrationBanner(true, true, 'server'), false, 'server-owned stays silent');
  assert.equal(shouldShowMigrationBanner(true, true, 'browser'), false, 'browser-owned (never attempted, or rolled back) stays silent');
});

test('shouldShowMigrationBanner: hidden before the owner decision has landed, or with no token', () => {
  assert.equal(shouldShowMigrationBanner(true, false, 'migrating'), false, 'decision still in flight — no manual-prompt flash');
  assert.equal(shouldShowMigrationBanner(false, true, 'migrating'), false);
});

test('shouldShowMigrationBanner: shown only once stuck in migrating — the silent retry on load already failed', () => {
  assert.equal(shouldShowMigrationBanner(true, true, 'migrating'), true);
});
