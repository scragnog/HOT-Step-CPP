import assert from 'node:assert/strict';
import { test } from 'node:test';
import { withReviewMutation } from './operations.js';

test('a second client cannot select another rung while cleanup is deleting runs', async () => {
  let release!: () => void;
  let entered!: () => void;
  const started = new Promise<void>(resolve => { entered = resolve; });
  const deletion = new Promise<void>(resolve => { release = resolve; });
  const cleanup = withReviewMutation('dataset-race', async () => {
    entered();
    await deletion;
  });
  await started;
  await assert.rejects(withReviewMutation('dataset-race', () => {
    throw new Error('selection ran during cleanup');
  }), /already in progress/);
  release();
  await cleanup;
  await withReviewMutation('dataset-race', () => {});
});
