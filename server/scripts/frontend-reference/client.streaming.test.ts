// client.streaming.test.ts — the streaming/recording audit group through the
// documented sequence (docs/dev/frontend-media.md): open a STORM stream,
// read its concatenated WAVs with the published WavFrameReader, steer it with
// /storm/control, record it canonically through its stream session, stop it,
// export the recording; then disconnect, rejected input and the MM3 stream's
// refusals. The fake engine renders each STORM slot; every Node route is real.
//
// Not covered here: a live MM3 stream. It needs a MiniMax-Music3 job (backend
// switch plus the engine's MM3 stream endpoint), which the fake engine does
// not serve. Recording exports other than WAV run ffmpeg, so only WAV is
// exported.

import test from 'node:test';
import assert from 'node:assert/strict';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient } from './client.js';
import { STREAM_SESSION_HEADER, WavFrameReader, type StormControl, type StormStreamStart } from '../../src/contracts/streaming.js';
import type { StreamSessionSummary } from '../../src/contracts/streamSessions.js';

let server: FakeServer;
let client: ReferenceClient;

test.before(async () => {
  server = await startFakeServer();
  client = new ReferenceClient(server.origin);
  await client.login();
});
test.after(() => server.close());

const json = async (method: string, path: string, body?: unknown) => {
  const res = await fetch(`${server.origin}${path}`, { method, headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: res.status, body: await res.json() as any };
};
const session = async (id: string) => (await json('GET', `/api/stream-sessions/${id}`)).body.session as StreamSessionSummary;

/** Open a STORM stream and hand back its reader and session id. */
async function openStorm(streamId: string, abort = new AbortController()) {
  const start: StormStreamStart = { streamId, caption: 'fixture storm', lyrics: '[Instrumental]', instrumental: true,
    seed: 11, skipLm: true, duration: 5 };
  const res = await fetch(`${server.origin}/api/generate/storm/stream`, { method: 'POST', signal: abort.signal,
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(start) });
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'audio/wav');
  const sessionId = res.headers.get(STREAM_SESSION_HEADER);
  assert.ok(sessionId, 'the stream names its Node session');
  const body = res.body!.getReader();
  const frames = new WavFrameReader();
  const wavs: Uint8Array[] = [];
  /** Read until `n` WAVs have arrived (or the stream ends). */
  const readUntil = async (n: number) => {
    while (wavs.length < n) {
      const { done, value } = await body.read();
      if (done) return true;
      wavs.push(...frames.push(value));
    }
    return false;
  };
  const drain = async () => { while (!(await body.read()).done) { /* until EOF */ } return frames.end(); };
  return { sessionId, wavs, readUntil, drain, abort };
}

test('STORM: stream, control, canonical recording, stop, WAV export of exactly the recorded slots', async () => {
  const storm = await openStorm('deck-a');
  assert.equal((await json('POST', `/api/stream-sessions/${storm.sessionId}/recording`, { action: 'start' })).status, 200);
  await storm.readUntil(1);

  const control: StormControl = { streamId: 'deck-a', guidance_scale: 5.5, next_bpm: 96, seed_lock: true };
  assert.deepEqual((await json('POST', '/api/generate/storm/control', control)).body, { ok: true, streamId: 'deck-a' });
  const live = (await json('GET', '/api/generate/storm/control?streamId=deck-a')).body;
  assert.deepEqual([live.running, live.guidanceScale, live.seedLock], [true, 5.5, true]);

  await storm.readUntil(3);
  assert.deepEqual((await json('POST', '/api/generate/storm/stop', { streamId: 'deck-a' })).body, { ok: true });
  assert.equal((await storm.drain()).truncatedBytes, 0, 'stop ends between WAVs');
  assert.ok(storm.wavs.length >= 3);

  const s = await session(storm.sessionId!);
  assert.deepEqual([s.kind, s.status, s.recording.status], ['storm', 'ended', 'stopped']);
  assert.equal(s.chunkCount, storm.wavs.length, 'Node saw every slot the client received');
  const { fromChunk, toChunk } = s.recording;
  assert.ok(fromChunk !== null && toChunk !== null);

  const exported = await fetch(`${server.origin}/api/stream-sessions/${storm.sessionId}/recording/export?format=wav`);
  assert.equal(exported.status, 200);
  assert.match(exported.headers.get('content-disposition') ?? '', /attachment; filename="storm-recording-.*\.wav"/);
  const file = Buffer.from(await exported.arrayBuffer());
  const expected = Buffer.concat(storm.wavs.slice(fromChunk!, toChunk! + 1).map(w => Buffer.from(w).subarray(44)));
  assert.ok(file.subarray(44).equals(expected), 'the export is the engine samples of the recorded slots, byte for byte');

  assert.equal((await json('POST', `/api/stream-sessions/${storm.sessionId}/recording`, { action: 'start' })).status, 409,
    'an ended stream cannot start a new recording');
});

test('STORM: closing the connection stops the stream and ends its session', async () => {
  const storm = await openStorm('deck-b');
  await storm.readUntil(1);
  storm.abort.abort();
  for (let i = 0; i < 50 && (await json('GET', '/api/generate/storm/control?streamId=deck-b')).body.running; i++) await new Promise(r => setTimeout(r, 100));
  assert.equal((await json('GET', '/api/generate/storm/control?streamId=deck-b')).body.running, false);
  for (let i = 0; i < 50 && (await session(storm.sessionId!)).status !== 'ended'; i++) await new Promise(r => setTimeout(r, 100));
  assert.equal((await session(storm.sessionId!)).status, 'ended');
});

test('Stream sessions: rejected input and unknown ids', async () => {
  const storm = await openStorm('deck-c');
  await storm.readUntil(1);
  await json('POST', '/api/generate/storm/stop', { streamId: 'deck-c' });
  await storm.drain();
  assert.equal((await json('POST', `/api/stream-sessions/${storm.sessionId}/recording`, { action: 'pause' })).status, 400);
  assert.equal((await fetch(`${server.origin}/api/stream-sessions/${storm.sessionId}/recording/export?format=aac`)).status, 400);
  assert.equal((await fetch(`${server.origin}/api/stream-sessions/${storm.sessionId}/recording/export`)).status, 409, 'nothing was recorded');
  assert.equal((await json('GET', '/api/stream-sessions/00000000-0000-4000-8000-000000000000')).status, 404);
  const list = (await json('GET', '/api/stream-sessions')).body.sessions as StreamSessionSummary[];
  assert.ok(list.some(x => x.id === storm.sessionId));
});

test('MM3 stream: 404 for an unknown job; 409 for a finished job that is not streaming', async () => {
  assert.equal((await fetch(`${server.origin}/api/generate/mm3/stream/no-such-job`)).status, 404);
  const { jobId } = await client.submitGeneration({ caption: 'not streaming', lyrics: '[Instrumental]', instrumental: true,
    seed: 4, randomSeed: false, skipLm: true });
  const status = await client.waitForGeneration(jobId);
  assert.equal(status.status, 'succeeded');
  assert.equal(status.mm3_streaming, false, 'status says not to open the stream');
  const res = await fetch(`${server.origin}/api/generate/mm3/stream/${jobId}`);
  assert.equal(res.status, 409);
  assert.equal((await res.json() as { error: string }).error, 'This job is not streaming');
});

test('Safety: no real subprocess exec or off-origin network call occurred', () => {
  assert.deepEqual(server.violations, []);
});
