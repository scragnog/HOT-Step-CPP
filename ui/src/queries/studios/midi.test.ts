import test from 'node:test';
import assert from 'node:assert/strict';
import { QueryObserver } from '@tanstack/react-query';
import { createQueryClient } from '@/queries/client';
import { midiJobsOptions, midiStatusOptions, midiNotesOptions, removeMidiJob } from './midi';
import { ApiError } from '@/services/httpClient';
import type { MidiJobSummary } from '@/services/midiStudioApi';

const job: MidiJobSummary = { id: 'test-job', status: 'transcribing', sourceFileName: 'test.wav', model: 'small', noteCount: 0, durationSec: 1, createdAt: '' };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(r => { resolve = r; });
  return { promise, resolve };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('slow simultaneous MIDI reads share a single cycle and consume cancellation', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  const held = deferred<Response>();
  let calls = 0;
  let signal: AbortSignal | undefined;
  globalThis.fetch = async (_url, opts) => { calls++; signal = opts?.signal ?? undefined; return held.promise; };
  try {
    const options = midiJobsOptions();
    const first = client.query(options).catch(error => error);
    const second = client.query(options).catch(error => error);
    await tick();
    assert.equal(calls, 1);
    await client.cancelQueries({ queryKey: options.queryKey });
    assert.equal(signal?.aborted, true);
    held.resolve(json([job]));
    await Promise.all([first, second]);
    assert.equal(client.getQueryData(options.queryKey), undefined);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('delete during a slow progress read cannot resurrect the row, even if fetch ignores abort', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  const progress = deferred<Response>();
  let progressSignal: AbortSignal | undefined;
  let deleted = false;
  globalThis.fetch = async (url, opts) => {
    if (opts?.method === 'DELETE') { deleted = true; return json({}); }
    if (String(url).endsWith('/jobs')) return json(deleted ? [] : [job]);
    progressSignal = opts?.signal ?? undefined;
    return progress.promise;
  };
  try {
    const options = midiJobsOptions();
    client.setQueryData(options.queryKey, [job]);
    client.setQueryData(midiNotesOptions(job.id).queryKey, { notes: [] });
    const pending = client.query({ ...options, staleTime: 0 }).catch(error => error);
    await tick();
    await removeMidiJob(client, job.id);
    assert.equal(progressSignal?.aborted, true);
    progress.resolve(json({ status: 'done', noteCount: 9 }));
    await pending; await tick();
    assert.deepEqual(client.getQueryData(options.queryKey), []);
    assert.equal(client.getQueryData(midiNotesOptions(job.id).queryKey), undefined);
    assert.deepEqual(await client.query(options), []);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('a poll started during DELETE is cancelled again after success', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  const deletion = deferred<Response>();
  const list = deferred<Response>();
  let signal: AbortSignal | undefined;
  globalThis.fetch = async (_url, opts) => {
    if (opts?.method === 'DELETE') return deletion.promise;
    signal = opts?.signal ?? undefined;
    return list.promise;
  };
  try {
    const options = midiJobsOptions();
    client.setQueryData(options.queryKey, [job]);
    const removing = removeMidiJob(client, job.id);
    await tick();
    const reading = client.query({ ...options, staleTime: 0 }).catch(error => error);
    await tick();
    deletion.resolve(json({}));
    await removing;
    assert.equal(signal?.aborted, true);
    list.resolve(json([job]));
    await reading; await tick();
    assert.deepEqual(client.getQueryData(options.queryKey), []);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('failed deletion retains the row and reports the error without retrying', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  let deletes = 0;
  globalThis.fetch = async () => { deletes++; return json({ error: 'Cannot delete' }, 500); };
  try {
    client.setQueryData(midiJobsOptions().queryKey, [job]);
    await assert.rejects(removeMidiJob(client, job.id), /Cannot delete/);
    assert.equal(deletes, 1);
    assert.deepEqual(client.getQueryData(midiJobsOptions().queryKey), [job]);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('MIDI polls stop at terminal status and downloads stop when complete', async () => {
  const client = createQueryClient();
  try {
    const options = midiJobsOptions();
    client.setQueryData(options.queryKey, [job]);
    const query = client.getQueryCache().find({ queryKey: options.queryKey })!;
    const interval = options.refetchInterval as (query: any) => number | false;
    assert.equal(interval(query), 1500);
    client.setQueryData(options.queryKey, [{ ...job, status: 'done' }]);
    assert.equal(interval(query), false);
    client.setQueryData(options.queryKey, []);
    assert.equal(interval(query), false);
    const status = midiStatusOptions();
    const statusInterval = status.refetchInterval as (query: any) => number | false;
    assert.equal(statusInterval({ state: { data: { models: { small: { downloading: true } } }, fetchFailureCount: 0 } }), 1500);
    assert.equal(statusInterval({ state: { data: { models: { small: { downloading: false } } }, fetchFailureCount: 0 } }), false);
    assert.equal(interval({ state: { data: [job], status: 'error', errorUpdateCount: 9 } }), 30_000);
  } finally { client.clear(); }
});

test('returning to MIDI reuses its fresh snapshot and mutation invalidation refreshes observers', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  let calls = 0;
  globalThis.fetch = async () => { calls++; return json([]); };
  try {
    const options = midiJobsOptions();
    await client.query(options);
    const first = new QueryObserver(client, options);
    first.subscribe(() => {})();
    const second = new QueryObserver(client, options);
    const unsubscribe = second.subscribe(() => {});
    assert.equal(calls, 1);
    await client.invalidateQueries({ queryKey: options.queryKey });
    assert.equal(calls, 2);
    unsubscribe();
    assert.equal(client.getQueryCache().find({ queryKey: options.queryKey })?.getObserversCount(), 0);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('MIDI read errors carry HTTP status instead of a failed empty snapshot', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  globalThis.fetch = async () => json({ error: 'Missing job' }, 404);
  try {
    await assert.rejects(client.query(midiJobsOptions()), (error: unknown) => error instanceof ApiError && error.status === 404);
    assert.equal(client.getQueryData(midiJobsOptions().queryKey), undefined);
  } finally { globalThis.fetch = original; client.clear(); }
});
