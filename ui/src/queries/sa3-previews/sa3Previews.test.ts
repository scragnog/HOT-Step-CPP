import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { createQueryClient } from '@/queries/client';
import { sa3PreviewKeys } from './keys';
import { invalidateSa3Previews, manualRenderRequest, renderSa3PreviewCommand, sa3PreviewOptions, sa3PreviewsPending } from './queries';

const originalFetch = globalThis.fetch;
const clients: ReturnType<typeof createQueryClient>[] = [];
afterEach(() => { globalThis.fetch = originalFetch; clients.splice(0).forEach(c => c.clear()); });
const cache = () => { const c = createQueryClient(); clients.push(c); return c; };
const list = (status: string) => ({ run: 'a/1', settings: { everySteps: 4, prompts: ['p'], seconds: 5, seeds: [1] }, previews: [{ run: 'a/1', step: 4, promptIndex: 0, prompt: 'p', seed: 1, seconds: 5, status, audioUrl: null, error: null, final: false, createdAt: 1 }] });
const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });

test('the list read goes to the worker in its key and stays per run', async () => {
  const c = cache();
  const urls: string[] = [];
  globalThis.fetch = async input => { urls.push(String(input)); return json(list('done')); };
  await c.query(sa3PreviewOptions('box', 'ds 1', 'a/1'));
  await c.query(sa3PreviewOptions(null, 'ds 1', 'a/2'));
  assert.match(urls[0], /\/api\/workers\/box\/api\/training\/datasets\/ds%201\/sa3-previews\?run=a%2F1$/);
  assert.match(urls[1], /^(\S*)\/api\/training\/datasets\/ds%201\/sa3-previews\?run=a%2F2$/);
});

test('only queued or rendering rows keep the list polling', () => {
  assert.equal(sa3PreviewsPending(list('rendering') as never), true);
  assert.equal(sa3PreviewsPending(list('queued') as never), true);
  assert.equal(sa3PreviewsPending(list('done') as never), false);
  assert.equal(sa3PreviewsPending(list('failed') as never), false);
  assert.equal(sa3PreviewsPending(undefined), false);
});

test('a render invalidates that worker and dataset only, and a failed one invalidates nothing', async () => {
  const c = cache();
  globalThis.fetch = async () => json(list('done'));
  const mine = sa3PreviewOptions('box', 'ds', 'a/1');
  const otherWorker = sa3PreviewOptions('other', 'ds', 'a/1');
  const otherSet = sa3PreviewOptions('box', 'ds2', 'a/1');
  await Promise.all([mine, otherWorker, otherSet].map(o => c.query(o)));
  await renderSa3PreviewCommand(c, 'box', 'ds', { run: 'a/1', step: 4 });
  assert.equal(c.getQueryState(mine.queryKey)?.isInvalidated, true);
  assert.equal(c.getQueryState(otherWorker.queryKey)?.isInvalidated, false);
  assert.equal(c.getQueryState(otherSet.queryKey)?.isInvalidated, false);
  c.setQueryData(mine.queryKey, list('done'));
  globalThis.fetch = async () => new Response(JSON.stringify({ error: 'no' }), { status: 400, headers: { 'Content-Type': 'application/json' } });
  await assert.rejects(renderSa3PreviewCommand(c, 'box', 'ds', { run: 'a/1', step: 8 }));
  assert.equal(c.getQueryState(mine.queryKey)?.isInvalidated, false);
  await invalidateSa3Previews(c, null, 'ds');
  assert.equal(sa3PreviewKeys.dataset(null, 'ds')[1], 'training-runs/sa3-previews');
});

test('a manual render sends only run and step, so the run keeps its saved prompts after a reload, resume or previews off', async () => {
  assert.deepEqual(manualRenderRequest('a/1', 8), { run: 'a/1', step: 8 });
  const c = cache();
  let sent: unknown;
  globalThis.fetch = async (_url, init) => { sent = JSON.parse(String(init?.body)); return json(list('done')); };
  await renderSa3PreviewCommand(c, null, 'ds', manualRenderRequest('a/1', 8));
  assert.deepEqual(sent, { run: 'a/1', step: 8 });
});
