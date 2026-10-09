// client.mm3stream.test.ts — the live MM3 stream through its documented
// sequence (docs/dev/frontend-media.md): make MiniMax-Music3 the active
// backend, submit a streaming render, poll status until mm3_streaming, open
// GET /api/generate/mm3/stream/:id, read the windows with WavFrameReader, and
// find them in the stream's Node session. The real MiniMax backend and the
// real stream route run; only the engine is fake (fakeMm3Engine.ts).

import test from 'node:test';
import assert from 'node:assert/strict';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient } from './client.js';
import { installFakeMm3, mm3Window, MM3_WINDOWS } from './fakeMm3Engine.js';
import { STREAM_SESSION_HEADER, WavFrameReader, type Mm3StreamStatus } from '../../src/contracts/streaming.js';
import type { StreamSessionSummary } from '../../src/contracts/streamSessions.js';

let server: FakeServer;
let client: ReferenceClient;

test.before(async () => {
  server = await startFakeServer();
  installFakeMm3(server.engine);
  client = new ReferenceClient(server.origin);
  await client.login();
});
test.after(() => server.close());

test('MM3: a streaming render is heard window by window and recorded by its session', async () => {
  const active = await fetch(`${server.origin}/api/backends/active`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: 'minimax-m3' }) });
  assert.deepEqual(await active.json(), { activeId: 'minimax-m3' });

  const { jobId } = await client.submitGeneration({ caption: 'fixture mm3', lyrics: '[Instrumental]', instrumental: true,
    duration: 10, seed: 1, randomSeed: false, mm3Stream: true, expectedBackend: 'minimax-m3' });
  let status: Record<string, unknown> & Partial<Mm3StreamStatus> = {};
  for (let i = 0; i < 100; i++) {
    status = await client.generationStatus(jobId);
    if (status.mm3_streaming === true || ['failed', 'cancelled'].includes(String(status.status))) break;
    await new Promise(r => setTimeout(r, 100));
  }
  assert.equal(status.mm3_streaming, true, JSON.stringify(status));
  assert.equal(status.mm3_takes, 1);
  assert.equal(typeof status.mm3_duration, 'number', 'the length (0 = auto, unknown) is published before any audio');

  const res = await fetch(`${server.origin}/api/generate/mm3/stream/${jobId}`);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'audio/wav');
  const sessionId = res.headers.get(STREAM_SESSION_HEADER);
  assert.ok(sessionId);
  const reader = new WavFrameReader();
  const windows: Uint8Array[] = [];
  const body = res.body!.getReader();
  for (;;) {
    const { done, value } = await body.read();
    if (done) break;
    windows.push(...reader.push(value));
  }
  assert.equal(reader.end().truncatedBytes, 0);
  assert.equal(windows.length, MM3_WINDOWS);
  windows.forEach((w, i) => assert.ok(Buffer.from(w).equals(mm3Window(i)), `window ${i} arrives byte for byte`));

  for (let i = 0; i < 50; i++) {
    if ((await (await fetch(`${server.origin}/api/stream-sessions/${sessionId}`)).json() as { session: StreamSessionSummary }).session.status === 'ended') break;
    await new Promise(r => setTimeout(r, 50));
  }
  const { session } = await (await fetch(`${server.origin}/api/stream-sessions/${sessionId}`)).json() as { session: StreamSessionSummary };
  assert.deepEqual([session.kind, session.source, session.status, session.chunkCount], ['mm3', { jobId, take: 0 }, 'ended', MM3_WINDOWS]);
  assert.equal(session.chunks[0].sampleRate, 44100);

  const second = await fetch(`${server.origin}/api/generate/mm3/stream/${jobId}`);
  assert.equal(second.status, 409, 'the engine allows one reader per take');

  const finished = await client.waitForGeneration(jobId);
  assert.equal(finished.status, 'succeeded', JSON.stringify(finished.error));
});

test('Safety: no real subprocess exec or off-origin network call occurred', () => {
  assert.deepEqual(server.violations, []);
});
