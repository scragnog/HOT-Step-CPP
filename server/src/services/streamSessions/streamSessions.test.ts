import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { LIMITS, StreamSessions, StreamSessionError } from './index.js';
import { channel0, parseWav, WavSplitter, wavHeader, type WavFormat } from './wav.js';
import { detectBpm, detectKey } from './analysis.js';

const F32: WavFormat = { audioFormat: 3, channels: 2, sampleRate: 48000, bitsPerSample: 32 };

/** A stereo float WAV whose left channel clicks at `bpm` over a sustained A3+C#4+E4 triad. */
function fixture(seconds: number, bpm: number, format = F32, seed = 0): Uint8Array {
  const frames = Math.round(seconds * format.sampleRate);
  const data = new Uint8Array(frames * format.channels * format.bitsPerSample / 8);
  const view = new DataView(data.buffer);
  const beat = Math.round(60 / bpm * format.sampleRate);
  for (let i = 0; i < frames; i++) {
    const t = i / format.sampleRate;
    let v = 0.1 * (Math.sin(2 * Math.PI * 220 * t) + Math.sin(2 * Math.PI * 277.18 * t) + Math.sin(2 * Math.PI * 329.63 * t));
    if (i % beat < 200) v += ((((i + seed) * 1103515245 + 12345) >>> 16) % 2000 / 1000 - 1) * 0.8;
    for (let c = 0; c < format.channels; c++) {
      const at = (i * format.channels + c) * format.bitsPerSample / 8;
      if (format.audioFormat === 3) view.setFloat32(at, c === 0 ? v : -v, true);
      else view.setInt16(at, Math.max(-32768, Math.min(32767, Math.round(v * 32767))), true);
    }
  }
  const out = new Uint8Array(44 + data.length);
  out.set(wavHeader(format, data.length));
  out.set(data, 44);
  return out;
}

function store(limits: Partial<typeof LIMITS> = {}, clock = { t: 0 }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stream-sessions-'));
  const transcoded: string[] = [];
  const sessions = new StreamSessions(dir, async (wav, format, _bitrate, out) => {
    transcoded.push(format);
    fs.copyFileSync(wav, out);
  }, { ...LIMITS, ...limits }, () => clock.t);
  return { sessions, dir, clock, transcoded, done: () => fs.rmSync(dir, { recursive: true, force: true }) };
}

test('parseWav reads plain and extensible float headers and channel0 de-interleaves', () => {
  const wav = fixture(0.1, 120);
  const parsed = parseWav(wav);
  assert.deepEqual([parsed.audioFormat, parsed.channels, parsed.sampleRate, parsed.bitsPerSample, parsed.frames], [3, 2, 48000, 32, 4800]);
  const left = channel0(parsed);
  assert.equal(left[100], new DataView(wav.buffer).getFloat32(44 + 100 * 8, true));

  // WAVE_FORMAT_EXTENSIBLE (fmt size 40) plus a LIST chunk before data.
  const ext = new Uint8Array(12 + 48 + 12 + 8 + parsed.data.length);
  const v = new DataView(ext.buffer);
  const put = (at: number, s: string) => { for (let i = 0; i < 4; i++) ext[at + i] = s.charCodeAt(i); };
  put(0, 'RIFF'); v.setUint32(4, ext.length - 8, true); put(8, 'WAVE');
  put(12, 'fmt '); v.setUint32(16, 40, true); v.setUint16(20, 0xfffe, true); v.setUint16(22, 2, true);
  v.setUint32(24, 48000, true); v.setUint16(34, 32, true); v.setUint16(44, 3, true);
  put(60, 'LIST'); v.setUint32(64, 4, true); put(68, 'INFO');
  put(72, 'data'); v.setUint32(76, parsed.data.length, true); ext.set(parsed.data, 80);
  const e = parseWav(ext);
  assert.equal(e.audioFormat, 3);
  assert.deepEqual(e.data, parsed.data);

  assert.throws(() => parseWav(new Uint8Array(64)), /RIFF/);
  const pcm8 = fixture(0.01, 120); new DataView(pcm8.buffer).setUint16(34, 8, true);
  assert.throws(() => parseWav(pcm8), /Unsupported/);
});

test('WavSplitter frames whole WAVs across arbitrary reads and skips garbage', () => {
  const a = fixture(0.05, 120), b = fixture(0.07, 120);
  const stream = new Uint8Array(3 + a.length + b.length);
  stream.set([1, 2, 3]); stream.set(a, 3); stream.set(b, 3 + a.length);
  const splitter = new WavSplitter();
  const out: Uint8Array[] = [];
  for (let at = 0; at < stream.length; at += 777) out.push(...splitter.push(stream.subarray(at, at + 777)));
  assert.equal(out.length, 2);
  assert.deepEqual(out[0], a);
  assert.deepEqual(out[1], b);
});

test('chunks are analysed with the shared detector on channel 0', () => {
  const { sessions, done } = store();
  try {
    const s = sessions.open('storm', { streamId: 'x' })!;
    const wav = fixture(6, 120);
    s.chunk(wav);
    const [a] = s.summary().chunks;
    const left = channel0(parseWav(wav));
    assert.deepEqual({ bpm: a.bpm, key: a.key }, { bpm: detectBpm(left, 48000).bpm, key: detectKey(left, 48000) });
    assert.equal(a.bpm, 120);
    assert.equal(a.key, 'A');
    assert.equal(a.frames, 6 * 48000);
  } finally { done(); }
});

test('a recording is the byte-exact concatenation of the chunks between start and stop', async () => {
  const { sessions, done, transcoded } = store();
  try {
    const s = sessions.open('mm3', { jobId: 'j', take: 0 })!;
    const c0 = fixture(0.2, 120, F32, 1), c1 = fixture(0.3, 120, F32, 2), c2 = fixture(0.1, 120, F32, 3), c3 = fixture(0.1, 120, F32, 4);
    s.chunk(c0);
    s.record('start');
    s.chunk(c1); s.chunk(c2);
    s.record('stop');
    s.chunk(c3);
    const sum = s.summary().recording;
    assert.deepEqual([sum.status, sum.fromChunk, sum.toChunk, sum.frames], ['stopped', 1, 2, 0.4 * 48000]);

    const { file, cleanup } = await s.exportTo('wav');
    const out = fs.readFileSync(file);
    const expected = Buffer.concat([parseWav(c1).data, parseWav(c2).data]);
    assert.ok(Buffer.from(parseWav(out).data).equals(expected));
    assert.equal(parseWav(out).sampleRate, 48000);
    cleanup();
    assert.equal(fs.existsSync(file), false);

    const flac = await s.exportTo('flac');
    assert.deepEqual(transcoded, ['flac']);
    flac.cleanup();
  } finally { done(); }
});

test('recording refuses bad transitions and stops on a format change', async () => {
  const { sessions, done } = store();
  try {
    const s = sessions.open('storm', { streamId: 'x' })!;
    await assert.rejects(s.exportTo('wav'), /Nothing has been recorded/);
    s.record('start');
    await assert.rejects(s.exportTo('wav'), /Stop the recording/);
    s.chunk(fixture(0.1, 120));
    s.chunk(fixture(0.1, 120, { audioFormat: 1, channels: 2, sampleRate: 44100, bitsPerSample: 16 }));
    assert.equal(s.summary().recording.status, 'failed');
    assert.match(s.summary().recording.error!, /changed the audio format/);
    await assert.rejects(s.exportTo('wav'), /changed the audio format/);
    assert.throws(() => s.record('start'), (e: unknown) => e instanceof StreamSessionError && e.status === 409);
    s.record('discard');
    assert.equal(s.summary().recording.status, 'idle');
    s.record('start');
    s.chunk(new Uint8Array([1, 2, 3]));
    assert.equal(s.summary().recording.status, 'failed');
    s.end();
    assert.throws(() => s.record('start'), /ended/);
  } finally { done(); }
});

test('the byte cap stops a recording without losing what it holds', async () => {
  const one = parseWav(fixture(0.1, 120)).data.length;
  const { sessions, done } = store({ maxRecordingBytes: one * 2 });
  try {
    const s = sessions.open('storm', { streamId: 'x' })!;
    s.record('start');
    for (let i = 0; i < 3; i++) s.chunk(fixture(0.1, 120));
    assert.deepEqual([s.summary().recording.status, s.summary().recording.bytes], ['capped', one * 2]);
    const { file, cleanup } = await s.exportTo('wav');
    assert.equal(parseWav(fs.readFileSync(file)).data.length, one * 2);
    cleanup();
  } finally { done(); }
});

test('session cap evicts the oldest ended session; idle and ended sessions expire', async () => {
  const clock = { t: 0 };
  const { sessions, done, dir } = store({ maxSessions: 2, endedTtlMs: 1000, liveIdleMs: 5000 }, clock);
  try {
    const a = sessions.open('storm', { streamId: 'a' })!;
    a.record('start'); a.chunk(fixture(0.05, 120)); a.end();
    clock.t = 1;
    const b = sessions.open('storm', { streamId: 'b' })!;
    clock.t = 2;
    const c = sessions.open('storm', { streamId: 'c' })!;
    assert.ok(c);
    assert.throws(() => sessions.get(a.id), /not found/);
    assert.equal(sessions.open('storm', { streamId: 'd' }), null, 'all live: no session, stream still plays');
    await new Promise(r => setTimeout(r, 20));
    assert.deepEqual(fs.readdirSync(dir), [], 'evicted recording deleted');

    clock.t = 5003;
    sessions.sweep();
    assert.equal(sessions.get(b.id).summary().status, 'ended');
    assert.equal(sessions.get(c.id).summary().expiresAt, 6003);
    clock.t = 6004;
    assert.deepEqual(sessions.list(), []);
  } finally { done(); }
});

test('route: 404 for unknown, 400 for bad input, start/stop/export round trip', async () => {
  const express = (await import('express')).default;
  const { createStreamSessionsRouter } = await import('../../routes/streamSessions.js');
  const { sessions, done } = store();
  const app = express();
  app.use(express.json());
  app.use('/s', createStreamSessionsRouter(() => sessions));
  const server = app.listen(0);
  const url = `http://127.0.0.1:${(server.address() as { port: number }).port}/s`;
  try {
    assert.equal((await fetch(`${url}/nope`)).status, 404);
    const s = sessions.open('storm', { streamId: 'x' })!;
    const post = (action: unknown) => fetch(`${url}/${s.id}/recording`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
    });
    assert.equal((await post('pause')).status, 400);
    assert.equal((await fetch(`${url}/${s.id}/recording/export?format=aac`)).status, 400);
    assert.equal((await fetch(`${url}/${s.id}/recording/export`)).status, 409);
    assert.equal((await post('start')).status, 200);
    const chunk = fixture(0.1, 120);
    s.chunk(chunk);
    assert.equal(((await (await post('stop')).json()) as { session: { recording: { status: string } } }).session.recording.status, 'stopped');
    const res = await fetch(`${url}/${s.id}/recording/export?format=wav`);
    assert.equal(res.status, 200);
    assert.match(res.headers.get('content-disposition') ?? '', /storm-recording-.*\.wav/);
    assert.ok(Buffer.from(parseWav(new Uint8Array(await res.arrayBuffer())).data).equals(Buffer.from(parseWav(chunk).data)));
    const list = await (await fetch(url)).json() as { sessions: { id: string }[] };
    assert.deepEqual(list.sessions.map(x => x.id), [s.id]);
  } finally { server.close(); done(); }
});

test('a write that fails after stop or end fails the recording and export refuses it', async (t) => {
  for (const after of ['stop', 'end'] as const) {
    const { sessions, done } = store();
    try {
      const s = sessions.open('storm', { streamId: 'x' })!;
      s.record('start');
      s.chunk(fixture(0.05, 120));
      let reject!: (err: Error) => void;
      const append = t.mock.method(fs.promises, 'appendFile', () => new Promise<void>((_, r) => { reject = r; }));
      s.chunk(fixture(0.05, 120));
      if (after === 'stop') s.record('stop'); else s.end();
      assert.equal(s.summary().recording.status, 'stopped');
      await new Promise(r => setImmediate(r));
      reject(new Error('disk full'));
      append.mock.restore();
      await assert.rejects(s.exportTo('wav'), /disk full/);
      assert.equal(s.summary().recording.status, 'failed');
    } finally { done(); }
  }
});

test('stream routes get no session, not an error, when the store cannot open one', async () => {
  const { openStreamSession } = await import('./index.js');
  assert.equal(openStreamSession('storm', { streamId: 'x' }, () => { throw new Error('EPERM'); }), null);
  const { sessions, done } = store();
  try {
    const s = openStreamSession('mm3', { jobId: 'j', take: 0 }, () => sessions)!;
    const a = fixture(0.05, 120), b = fixture(0.05, 120);
    const both = new Uint8Array(a.length + b.length); both.set(a); both.set(b, a.length);
    s.feed(both.subarray(0, 100)); s.feed(both.subarray(100));
    assert.equal(s.summary().chunkCount, 2);
  } finally { done(); }
});

test('concurrent exports are capped and a released slot is reusable', async () => {
  const { sessions, done } = store({ maxConcurrentExports: 1 });
  try {
    const s = sessions.open('storm', { streamId: 'x' })!;
    s.record('start'); s.chunk(fixture(0.05, 120)); s.record('stop');
    const first = await s.exportTo('wav');
    await assert.rejects(s.exportTo('wav'), (e: unknown) => e instanceof StreamSessionError && e.status === 429);
    first.cleanup(); first.cleanup();
    const second = await s.exportTo('flac');
    second.cleanup();
    // The double cleanup above released once: one slot, not two.
    sessions.admitExport();
    assert.throws(() => sessions.admitExport(), /Too many/);
  } finally { done(); }
});

test('export cleanup that cannot delete its temp file never throws and frees the slot', async (t) => {
  const { sessions, done } = store({ maxConcurrentExports: 1 });
  try {
    const s = sessions.open('storm', { streamId: 'x' })!;
    s.record('start'); s.chunk(fixture(0.05, 120)); s.record('stop');
    const first = await s.exportTo('wav');
    const rm = t.mock.method(fs, 'rmSync', () => { throw Object.assign(new Error('EPERM: operation not permitted'), { code: 'EPERM' }); });
    t.mock.method(console, 'warn', () => {});
    assert.doesNotThrow(() => first.cleanup());
    assert.equal(rm.mock.callCount(), 1);
    rm.mock.restore();
    const second = await s.exportTo('wav');
    second.cleanup();
    fs.rmSync(first.file, { force: true });
  } finally { done(); }
});
