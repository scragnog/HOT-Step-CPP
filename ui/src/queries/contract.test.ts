// contract.test.ts — retry policy, poll helper and client defaults.
// Run with the ui suite: node --import ../server/node_modules/tsx/dist/loader.mjs --test "src/**/*.test.ts"
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { ApiError } from '@/services/httpClient';
import {
  ERROR_POLL_MAX_MS, isRetryableError, pollWhile, retryDelay, retryQuery,
} from './contract';
import { createQueryClient as makeClient } from './client';

// gcTime keeps a 5 min timer per query; clear every client so node can exit.
const made: ReturnType<typeof makeClient>[] = [];
const createQueryClient = () => { const c = makeClient(); made.push(c); return c; };
after(() => made.forEach(c => c.clear()));

test('only network errors and 5xx are retryable', () => {
  assert.equal(isRetryableError(new TypeError('Failed to fetch')), true);
  assert.equal(isRetryableError(new ApiError(0, 'closed', null)), true);
  assert.equal(isRetryableError(new ApiError(503, 'busy', null)), true);
  assert.equal(isRetryableError(new ApiError(404, 'gone', null)), false);
  assert.equal(isRetryableError(new ApiError(409, 'stale', null)), false);
  assert.equal(isRetryableError(new DOMException('Aborted', 'AbortError')), false);
  assert.equal(isRetryableError(new Error('plain')), false);
});

test('at most two retries, with a capped exponential delay', () => {
  const err = new ApiError(500, 'x', null);
  assert.equal(retryQuery(0, err), true);
  assert.equal(retryQuery(1, err), true);
  assert.equal(retryQuery(2, err), false);
  assert.deepEqual([0, 1, 2, 3, 10].map(retryDelay), [500, 1000, 2000, 4000, 4000]);
});

test('pollWhile: real fetches back off per failed cycle, reset on success, stop when terminal', async () => {
  const client = createQueryClient();
  const queryKey = ['hs', 'test/poll'];
  let reply: () => { status: string } = () => ({ status: 'running' });
  let calls = 0;
  const cycle = () => client.query({
    queryKey, staleTime: 0, retryDelay: 1,
    queryFn: async () => { calls++; return reply(); },
  }).catch(() => undefined);
  const poll = pollWhile<{ status: string }>(d => d?.status === 'running', 1500);
  const interval = () => poll(client.getQueryCache().find({ queryKey })!);

  await cycle();
  assert.equal(interval(), 1500);
  reply = () => { throw new ApiError(500, 'down', null); };
  const backoff: Array<number | false> = [];
  for (let i = 0; i < 5; i++) { await cycle(); backoff.push(interval()); }
  assert.equal(calls, 1 + 5 * 3, 'each failed cycle used its two default retries');
  assert.deepEqual(backoff, [3000, 6000, 12000, 24000, ERROR_POLL_MAX_MS]);

  reply = () => ({ status: 'running' });
  await cycle();
  assert.equal(interval(), 1500, 'a success resets the backoff');
  reply = () => { throw new ApiError(503, 'busy', null); };
  await cycle();
  assert.equal(interval(), 3000, 'counting restarts from the success');

  reply = () => ({ status: 'done' });
  await cycle();
  assert.equal(interval(), false);
});

test('client: a 5xx read is retried twice, then fails without caching data', async () => {
  const client = createQueryClient();
  let calls = 0;
  const key = ['hs', 'test/a'];
  await assert.rejects(client.query({
    queryKey: key,
    queryFn: async () => { calls++; throw new ApiError(500, 'boom', null); },
    retryDelay: 1,
  }));
  assert.equal(calls, 3);
  assert.equal(client.getQueryData(key), undefined);
  assert.equal(client.getQueryState(key)?.status, 'error');
  client.clear();
});

test('client: a 4xx read is not retried', async () => {
  const client = createQueryClient();
  let calls = 0;
  await assert.rejects(client.query({
    queryKey: ['hs', 'test/b'],
    queryFn: async () => { calls++; throw new ApiError(404, 'missing', null); },
  }));
  assert.equal(calls, 1);
  client.clear();
});

test('client: mutations never retry, even on a 5xx', async () => {
  const client = createQueryClient();
  let calls = 0;
  const mutation = client.getMutationCache().build(client, {
    mutationFn: async () => { calls++; throw new ApiError(503, 'busy', null); },
  });
  await assert.rejects(mutation.execute(undefined));
  assert.equal(calls, 1);
  client.clear();
});

test('client: project defaults are applied', () => {
  const d = createQueryClient().getDefaultOptions();
  assert.equal(d.queries?.staleTime, 30_000);
  assert.equal(d.queries?.gcTime, 300_000);
  assert.equal(d.queries?.refetchOnWindowFocus, false);
  assert.equal(d.queries?.refetchOnReconnect, true);
  assert.equal(d.mutations?.retry, false);
});
