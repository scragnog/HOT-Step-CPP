// workflowApi.test.ts — followJob against the real /api/workflows router: a
// stream that drops mid-job resumes from the last event, with nothing lost or
// repeated. No UI test runner is wired up; run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/services/workflowApi.test.ts)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startWorkflowTestServer, z } from '../../../server/src/services/workflows/testServer';
import type { WorkflowEvent } from '../../../server/src/contracts/workflow';
import { followJob, revisionedRequest, workflowApi, WorkflowRequestError } from './workflowApi';

test('followJob resumes after a dropped stream without losing or repeating events', async () => {
  const http = await startWorkflowTestServer();
  const { jobs } = http;
  let release!: () => void;
  const step = new Promise<void>(r => { release = r; });
  jobs.register({
    kind: 'count', input: z.object({ n: z.number() }),
    run: async ctx => {
      ctx.emit('progress', { i: 1 });
      await step;
      for (let i = 2; i <= ctx.input.n; i++) ctx.emit('progress', { i });
      return 'done';
    },
  });
  const base = http.origin;

  // Relative URLs go to the test server; the first stream is cut after its first event.
  const realFetch = globalThis.fetch;
  let streams = 0;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const response = await realFetch(base + url, init);
    if (!url.includes('/events') || streams++ > 0) return response;
    const reader = response.body!.getReader();
    let seen = '';
    const cut = new ReadableStream<Uint8Array>({
      async pull(controller) {
        const { value, done } = await reader.read();
        if (done) { controller.close(); return; }
        controller.enqueue(value);
        seen += new TextDecoder().decode(value);
        if (seen.includes('"progress"')) { void reader.cancel(); controller.error(new Error('network dropped')); }
      },
    });
    return new Response(cut, { status: response.status, headers: response.headers });
  }) as typeof fetch;

  try {
    const { job } = await workflowApi.submit('t', 'count', 'k', { n: 5 });
    const events: WorkflowEvent[] = [];
    const errors: unknown[] = [];
    const snapshots: boolean[] = [];
    const done = followJob('t', job.id, {
      onEvent: e => { events.push(e); if (e.type === 'progress' && (e.data as { i: number }).i === 1) setTimeout(release, 20); },
      onSnapshot: (_, gap) => snapshots.push(gap),
      onError: e => errors.push(e),
    }, { retryMs: 10 });
    assert.equal(await done, 'succeeded');
    assert.equal(errors.length, 1, 'one drop');
    assert.deepEqual(snapshots, [false, false]);
    const total = jobs.get(job.id).lastSeq;
    assert.deepEqual(events.map(e => e.seq), Array.from({ length: total }, (_, i) => i + 1));

    // A stale document write surfaces the current revision.
    const { document } = await workflowApi.createDocument('t', 'draft', { a: 1 });
    await workflowApi.updateDocument('t', document.id, 1, { a: 2 });
    await assert.rejects(workflowApi.updateDocument('t', document.id, 1, { a: 3 }),
      (e: unknown) => e instanceof WorkflowRequestError && e.status === 409 && e.currentRevision === 2);
    await assert.rejects(followJob('t', 'missing', { onEvent: () => {} }),
      (e: unknown) => e instanceof WorkflowRequestError && e.status === 404);
  } finally {
    globalThis.fetch = realFetch;
    await http.close();
  }
});

test('followJob on a retried job ends at the current attempt, not the first failure', async () => {
  const http = await startWorkflowTestServer();
  http.jobs.register({
    kind: 'flaky', input: z.object({}),
    run: async ctx => { if (ctx.attempt === 1) throw new Error('first try fails'); return 'ok'; },
  });
  const realFetch = globalThis.fetch;
  globalThis.fetch = ((url: string, init?: RequestInit) => realFetch(http.origin + url, init)) as typeof fetch;
  try {
    const { job } = await workflowApi.submit('t', 'flaky', 'k', {});
    await http.jobs.settled();
    await workflowApi.retry('t', job.id);
    await http.jobs.settled();
    const seen: string[] = [];
    const status = await followJob('t', job.id, {
      onEvent: e => { if (e.type === 'status') seen.push((e.data as { status: string }).status); },
    });
    assert.equal(status, 'succeeded');
    assert.deepEqual(seen, ['pending', 'running', 'failed', 'pending', 'running', 'succeeded']);
  } finally {
    globalThis.fetch = realFetch;
    await http.close();
  }
});

test('revisionedRequest carries the stale revision and the unsupported-version reason', async () => {
  const realFetch = globalThis.fetch;
  const replies: Array<[number, unknown]> = [
    [409, { error: 'Stale revision', currentRevision: 4 }],
    [409, { error: 'Saved as version 3', reason: 'unsupported-version', schemaVersion: 3, supportedVersion: 2 }],
    [200, { ok: true }],
  ];
  const seen: Array<{ url: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    seen.push({ url, init });
    const [status, body] = replies.shift()!;
    return new Response(JSON.stringify(body), { status });
  }) as typeof fetch;
  try {
    const stale = await revisionedRequest('tok', '/api/presets/x', 'PUT', { expectedRevision: 3 }).catch(e => e);
    assert.ok(stale instanceof WorkflowRequestError);
    assert.equal(stale.status, 409);
    assert.equal(stale.currentRevision, 4);
    const newer = await revisionedRequest('tok', '/api/presets/y').catch(e => e);
    assert.equal(newer.reason, 'unsupported-version');
    assert.equal(newer.message, 'Saved as version 3');
    assert.deepEqual(await revisionedRequest('tok', '/api/presets/z'), { ok: true });
    assert.equal((seen[0].init?.headers as Record<string, string>).Authorization, 'Bearer tok');
    assert.equal(seen[0].init?.body, JSON.stringify({ expectedRevision: 3 }));
  } finally {
    globalThis.fetch = realFetch;
  }
});
