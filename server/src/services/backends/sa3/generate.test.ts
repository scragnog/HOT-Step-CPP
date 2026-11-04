import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import {
  SA3_LOADING_STAGE, sa3RenderStage, resolveSa3Sampler, resolveSa3SeedBlend,
  prepareSa3Cover, runSa3Generation,
} from './generate.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

function sourceWav(rate: number, seconds: number): Buffer {
  const dataSize = rate * seconds * 2;
  const wav = Buffer.alloc(44 + dataSize);
  wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 2, 28);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(dataSize, 40);
  return wav;
}

test('SA3 cover reads the full source at its native rate, including tracks over ten minutes', async t => {
  const root = fs.mkdtempSync(path.resolve('.tmp-sa3-cover-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  for (const [rate, seconds] of [[8000, 601], [22050, 2], [48000, 1]]) {
    const source = path.join(root, `${rate}.wav`);
    const wav = sourceWav(rate, seconds);
    fs.writeFileSync(source, wav);
    const cover = await prepareSa3Cover({ sourceAudioUrl: source,
      sa3Cover: { transform: 0.9, adapters: 'texture:0.5, space:0' } });
    assert.deepEqual(cover.source, wav);
    assert.equal(cover.sampleRate, rate);
    assert.equal(cover.durationSec, seconds);
    assert.equal(cover.strength, 0.9);
    assert.deepEqual(cover.adapters, [{ name: 'texture', scale: 0.5 }, { name: 'space', scale: 0 }]);
  }
  const source = path.join(root, '48000.wav');
  const mp3 = Buffer.alloc(417 * 10);
  for (let offset = 0; offset < mp3.length; offset += 417) mp3.set([0xff, 0xfb, 0x90, 0x64], offset);
  const compressed = path.join(root, 'source.mp3');
  fs.writeFileSync(compressed, mp3);
  const mp3Cover = await prepareSa3Cover({ sourceAudioUrl: compressed });
  assert.deepEqual(mp3Cover.source, mp3);
  assert.equal(mp3Cover.sampleRate, 44100);
  assert.ok(mp3Cover.durationSec > 0);
  for (const adapters of ['texture', 'texture:', ':1', 'texture:NaN', 'texture:1,']) {
    await assert.rejects(prepareSa3Cover({ sourceAudioUrl: source, sa3Cover: { adapters } }), /name:scale/);
  }
  await assert.rejects(prepareSa3Cover({ sourceAudioUrl: source,
    sa3Cover: { transform: 1 } }), /transform/);
  await assert.rejects(prepareSa3Cover({}), /source audio/);
});

test('SA3 cover decodes FLAC at the native rate without cropping', async t => {
  const { config, getFFmpegPath } = await import('../../../config.js');
  const { parseBuffer } = await import('music-metadata');
  const ffmpeg = getFFmpegPath();
  assert.ok(ffmpeg, 'the server provides ffmpeg for uploaded formats');
  const root = fs.mkdtempSync(path.resolve('.tmp-sa3-cover-flac-'));
  const previous = config.data.dir;
  t.after(() => { config.data.dir = previous; fs.rmSync(root, { recursive: true, force: true }); });
  config.data.dir = root;
  const wav = path.join(root, 'source.wav'), flac = path.join(root, 'source.flac');
  fs.writeFileSync(wav, sourceWav(22050, 2));
  execFileSync(ffmpeg, ['-y', '-i', wav, flac], { timeout: 30_000, windowsHide: true, stdio: 'pipe' });
  const prepared = await prepareSa3Cover({ sourceAudioUrl: flac });
  const { format } = await parseBuffer(prepared.source, undefined, { duration: true });
  assert.equal(format.sampleRate, 22050);
  assert.equal(format.duration, 2);
  assert.equal(prepared.sampleRate, 22050);
  assert.equal(prepared.durationSec, 2);
});

test('SA3 cover submits refine then saves its source duration and rate through the normal runner', async t => {
  const { config } = await import('../../../config.js');
  const { initDb, closeDb, getDb } = await import('../../../db/database.js');
  const { sa3Backend } = await import('./index.js');
  const { TokenizerLoader } = await import('@lenml/tokenizers');
  const root = fs.mkdtempSync(path.resolve('.tmp-sa3-cover-run-'));
  const previous = { dir: config.data.dir, models: config.aceServer.models };
  const realFetch = globalThis.fetch;
  t.after(() => {
    closeDb();
    config.data.dir = previous.dir;
    config.aceServer.models = previous.models; globalThis.fetch = realFetch;
    fs.rmSync(root, { recursive: true, force: true });
  });
  config.data.dir = root;
  config.aceServer.models = path.join(root, 'models');
  const tokenizerDir = path.join(config.aceServer.models, 'onnx', 'sa3');
  fs.mkdirSync(tokenizerDir, { recursive: true });
  fs.writeFileSync(path.join(tokenizerDir, 'tokenizer.json'), '{}');
  fs.writeFileSync(path.join(tokenizerDir, 'tokenizer_config.json'), '{}');
  let prompt = '';
  t.mock.method(TokenizerLoader, 'fromPreTrained', () => Object.assign((text: string) => {
    prompt = text;
    return { input_ids: [1, 2], attention_mask: [1, 1] };
  }, { pad_token_id: 0 }));
  initDb();
  getDb().prepare('INSERT INTO users (id, username) VALUES (?, ?)').run('cover-user', 'cover-user');
  assert.ok(sa3Backend.operations.includes('cover'));
  assert.equal(sa3Backend.resolveRequest({ taskType: 'cover', style: 'piano' }).operation, 'cover');
  assert.equal(sa3Backend.resolveRequest({ style: 'piano' }).operation, 'text2music');
  const wav = sourceWav(8000, 601);
  const source = path.join(root, 'source.wav');
  fs.writeFileSync(source, wav);
  const calls: URL[] = [];
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    const u = new URL(String(url));
    calls.push(u);
    if (u.pathname === '/mm3/unload') return Response.json({ unloaded: false });
    assert.equal(u.pathname, '/sa3-refine');
    assert.deepEqual(Buffer.from(init!.body as Uint8Array), wav);
    assert.equal(u.searchParams.get('strength'), '0.7');
    assert.equal(u.searchParams.get('seed'), '47');
    assert.equal(u.searchParams.get('adapters'), 'texture:0.6');
    assert.equal(u.searchParams.get('steps'), '12');
    assert.equal(u.searchParams.get('sampler'), 'pingpong');
    assert.equal(u.searchParams.get('out_sr'), '8000');
    return new Response(new Uint8Array(wav));
  }) as typeof fetch;
  const job: GenerationJob = { id: 'sa3-cover-test', userId: 'cover-user', status: 'pending',
    attempts: [], createdAt: Date.now(),
    params: { taskType: 'cover', sourceAudioUrl: source, style: 'stale global caption',
      caption: '  soft vocals, piano\n[Verse]  ', seed: 47, randomSeed: false, duration: 30,
      sa3Steps: 12, stableStepSteps: 32, stableStepOn: true, postProcessingEnabled: false,
      sa3Cover: { transform: 0.7, adapters: 'texture:0.6' } },
    envelope: { models: { dit: 'sa3-dit-BF16.gguf' } } as unknown as GenerationJob['envelope'],
  };
  await runSa3Generation(job, { signal: new AbortController().signal });
  assert.equal(job.status, 'succeeded', job.error);
  assert.equal(job.result?.duration, 601);
  assert.equal(prompt, '  soft vocals, piano\n[Verse]  \nLength: 601 seconds');
  assert.equal(calls.filter(u => u.pathname === '/sa3-refine').length, 1);
  assert.equal(calls.some(u => u.pathname === '/sa3-synth'), false);
  const row = getDb().prepare('SELECT generation_params FROM songs WHERE id = ?')
    .get(job.result!.songIds[0]) as { generation_params: string };
  const params = JSON.parse(row.generation_params);
  assert.equal(params.sa3.sample_rate, 8000);
  assert.equal(params.sa3Request.durationSec, 601);
  assert.equal(params.sa3Request.strength, 0.7);
});

test('SA3 sends the captured DiT and reports the response filename, including fallback', async () => {
  const { aceClient } = await import('../../aceClient.js');
  const originalFetch = globalThis.fetch;
  const requested = 'sa3-dit-Q4_K_M-imat.gguf';
  const seen: string[] = [];
  let returned: string | undefined = requested;
  globalThis.fetch = (async (url: string | URL | Request) => {
    assert.equal(new URL(String(url)).searchParams.get('synth_model'), requested);
    return new Response(new Uint8Array([1, 2, 3]), {
      headers: returned ? { 'X-SA3-DiT': returned } : {},
    });
  }) as typeof fetch;
  try {
    for (returned of [requested, 'sa3-dit-BF16.gguf', undefined]) {
      const wav = await aceClient.submitSa3Synth({
        tokens: [1], nTokens: 1, durationSec: 1, synthModel: requested,
        onModelResolved: filename => seen.push(filename),
      });
      assert.deepEqual([...wav], [1, 2, 3]);
    }
    assert.deepEqual(seen, [requested, 'sa3-dit-BF16.gguf']);
  } finally { globalThis.fetch = originalFetch; }
});

test('SA3 selects its default Lua sampler while preserving native and plugin overrides', () => {
  assert.deepEqual(resolveSa3Sampler({}, {}), { sampler: 'pingpong', solver: 'md_pingpong_blend' });
  assert.deepEqual(resolveSa3Sampler({ sa3Sampler: 'pingpong' }, {}), { sampler: 'pingpong' });
  assert.deepEqual(resolveSa3Sampler({ sa3Sampler: 'euler' }, {}), { sampler: 'euler' });
  assert.deepEqual(resolveSa3Sampler({}, { solver: 'heun' }), { sampler: 'pingpong' });
});

test('SA3 blend defaults preserve explicit endpoints and reject non-finite values', () => {
  for (const v of [undefined, null, '', NaN, Infinity, 'invalid']) {
    assert.equal(resolveSa3SeedBlend(v), 0.5);
  }
  for (const [v, expected] of [[0, 0], ['0', 0], [1, 1], ['1', 1], [-1, 0], [2, 1], [0.25, 0.25]] as const) {
    assert.equal(resolveSa3SeedBlend(v), expected);
  }
});

test('SA3 synth sends the default and full blend, and disables breeding at zero or without seed B', async () => {
  const { aceClient } = await import('../../aceClient.js');
  const realFetch = globalThis.fetch;
  const requests: URL[] = [];
  globalThis.fetch = (async (url: string | URL | Request) => {
    requests.push(new URL(String(url)));
    return new Response(new Uint8Array([1, 2, 3]));
  }) as typeof fetch;
  try {
    for (const input of [undefined, 0, 1]) {
      await aceClient.submitSa3Synth({
        tokens: [1], nTokens: 1, durationSec: 1, seed: 123,
        seedB: 456, seedBlend: resolveSa3SeedBlend(input),
      });
    }
    await aceClient.submitSa3Synth({ tokens: [1], nTokens: 1, durationSec: 1, seedBlend: 1 });
    assert.equal(requests.length, 4);
    assert.ok(requests.every(url => url.pathname === '/sa3-synth'));
    assert.equal(requests[0].searchParams.get('seed_b'), '456');
    assert.equal(requests[0].searchParams.get('seed_blend'), '0.5');
    assert.equal(requests[1].searchParams.has('seed_b'), false);
    assert.equal(requests[1].searchParams.has('seed_blend'), false);
    assert.equal(requests[2].searchParams.get('seed_b'), '456');
    assert.equal(requests[2].searchParams.get('seed_blend'), '1');
    assert.equal(requests[3].searchParams.has('seed_b'), false);
    assert.equal(requests[3].searchParams.has('seed_blend'), false);
  } finally {
    globalThis.fetch = realFetch;
  }
});

// Copied from the DAW facade (pr/29 dawFacade.ts:457): a stage matching this
// reports phase loading_model and eta_s null. pr/06 has no facade, so the
// regex is asserted directly here.
const FACADE_LOADING_RE = /^loading\b/i;

test('SA3 stage at submit reads as loading', () => {
  console.log('submit stage:', SA3_LOADING_STAGE);
  assert.match(SA3_LOADING_STAGE, FACADE_LOADING_RE);
});

test('SA3 stage once the render fetch is sent does not read as loading', () => {
  const s = sa3RenderStage(30.4, 8);
  console.log('render stage:', s);
  assert.doesNotMatch(s, FACADE_LOADING_RE);
  assert.ok(s.startsWith('SA3: rendering'));
  assert.ok(s.includes('30s') && s.includes('8 steps'));
});

test('real progress (saving audio) replaces the loading stage', () => {
  const saving = 'Stable Audio 3: saving audio...';
  assert.doesNotMatch(saving, FACADE_LOADING_RE);
});

// The lane must stay held until the engine's worker has provably finished the
// SA3 call. Each test scripts the engine: the render POST, then the replies to
// successive /worker-drain calls ('fail' = socket error, '404' = an engine
// without the route, 'ok' = held until released by the test).
type DrainReply = 'fail' | '404' | 'ok';
async function withScriptedEngine(
  post: 'hang' | 'socket-error',
  drains: DrainReply[],
  body: (h: { calls: string[]; releaseDrain: () => void }) => Promise<void>,
) {
  const realFetch = globalThis.fetch;
  const calls: string[] = [];
  let releaseDrain: () => void = () => {};
  const socketError = () => Object.assign(new TypeError('fetch failed'), { cause: { code: 'UND_ERR_SOCKET' } });
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    const path = new URL(String(url)).pathname;
    calls.push(path);
    if (path === '/worker-drain') {
      const reply = drains.shift() ?? 'ok';
      if (reply === 'fail') throw socketError();
      if (reply === '404') return new Response('Not Found', { status: 404 });
      await new Promise<void>(r => { releaseDrain = r; });
      return new Response('{"status":"ok"}');
    }
    if (post === 'socket-error') throw socketError();
    return new Promise<Response>((_, reject) => {
      init!.signal!.addEventListener('abort',
        () => reject(new DOMException('aborted', 'AbortError')), { once: true });
    });
  }) as typeof fetch;
  try {
    await body({ calls, releaseDrain: () => releaseDrain() });
  } finally {
    globalThis.fetch = realFetch;
  }
}

async function startSynth(signal?: AbortSignal) {
  const { aceClient } = await import('../../aceClient.js');
  const state = { settled: false };
  const p = aceClient.submitSa3Synth({ tokens: [1], nTokens: 1, durationSec: 1, signal })
    .catch((e: Error) => { state.settled = true; return e; });
  return { p, state };
}

const tick = (ms = 20) => new Promise(r => setTimeout(r, ms));

test('an aborted SA3 render settles only after the engine worker drains', async () => {
  await withScriptedEngine('hang', ['ok'], async ({ calls, releaseDrain }) => {
    const ac = new AbortController();
    const { p, state } = await startSynth(ac.signal);
    await tick(0);
    ac.abort();
    await tick();
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain']);
    assert.equal(state.settled, false, 'must not settle while the engine is still stopping');
    releaseDrain();
    assert.equal(((await p) as Error).name, 'AbortError');
  });
});

test('an aborted SA3 cover stops refine and holds the lane until the worker drains', async () => {
  await withScriptedEngine('hang', ['ok'], async ({ calls, releaseDrain }) => {
    const { aceClient } = await import('../../aceClient.js');
    const ac = new AbortController();
    let settled = false;
    const result = aceClient.submitSa3Refine(sourceWav(8000, 1), {
      tokens: [1], nTokens: 1, signal: ac.signal,
    }).catch((error: Error) => { settled = true; return error; });
    await tick(0);
    ac.abort();
    await tick();
    assert.deepEqual(calls, ['/sa3-refine', '/worker-drain']);
    assert.equal(settled, false);
    releaseDrain();
    assert.equal((await result as Error).name, 'AbortError');
  });
});

test('a failed drain is retried, not taken as proof the engine stopped', async () => {
  await withScriptedEngine('hang', ['fail', 'ok'], async ({ calls, releaseDrain }) => {
    const ac = new AbortController();
    const { p, state } = await startSynth(ac.signal);
    await tick(0);
    ac.abort();
    await tick(1300);  // past the 1 s retry delay
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain', '/worker-drain']);
    assert.equal(state.settled, false, 'a socket error on the drain must keep the lane held');
    releaseDrain();
    assert.equal(((await p) as Error).name, 'AbortError');
  });
});

test('a render POST that dies on a socket error drains before it rejects', async () => {
  await withScriptedEngine('socket-error', ['ok'], async ({ calls, releaseDrain }) => {
    const { p, state } = await startSynth();
    await tick();
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain']);
    assert.equal(state.settled, false);
    releaseDrain();
    assert.equal(((await p) as Error).message, 'fetch failed');
  });
});

test('a 404 from /worker-drain keeps the lane held', async () => {
  await withScriptedEngine('hang', ['404', 'ok'], async ({ calls, releaseDrain }) => {
    const ac = new AbortController();
    const { p, state } = await startSynth(ac.signal);
    await tick(0);
    ac.abort();
    await tick(1300);  // past the 1 s retry delay
    assert.deepEqual(calls, ['/sa3-synth', '/worker-drain', '/worker-drain']);
    assert.equal(state.settled, false, 'a 404 proves nothing about the render');
    releaseDrain();
    assert.equal(((await p) as Error).name, 'AbortError');
  });
});
