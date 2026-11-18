import assert from 'node:assert/strict';
import test from 'node:test';
import { promises as fs } from 'node:fs';
import router from './analyze.js';

test('missing Essentia returns a plain unavailable message before reading audio', async t => {
  const access = t.mock.method(fs, 'access', async () => { throw new Error('ENOENT'); });
  const handler = router.stack.find(layer => layer.route?.path === '/')!.route!.stack[0].handle;
  let status = 200;
  let body: any;
  const res = { status: (n: number) => { status = n; return res; }, json: (value: unknown) => { body = value; } };
  await handler({ body: { audioUrl: '/audio/track.wav' } } as any, res as any, () => {});
  assert.equal(status, 503);
  assert.equal(body.error, 'Essentia is not installed. No metadata was saved.');
  assert.equal(access.mock.callCount(), 1);
});
