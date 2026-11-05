import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { aceClient } from '../aceClient.js';
import { encodeWav } from './audioLevel.js';
import { normalizeMm3RefinerParams, runMm3Refiner } from './mm3Refiner.js';
import { modelDownloadService } from '../modelDownloadService.js';
import { resolveGenerationIntent } from './intent.js';

const wav = (rate = 48000, frames = 257) => encodeWav({
  channels: [new Float32Array(frames).fill(0.1)], sampleRate: rate,
  numChannels: 1, sourceFormat: 's16',
}, 's16');

test('MM3 refiner defaults and saved settings preserve direct mode and overlap zero', () => {
  assert.deepEqual(normalizeMm3RefinerParams({}), {
    mm3RefinerEnabled: false, mm3RefinerSteps: 32, mm3RefinerWindowSeconds: 30,
    mm3RefinerOverlapSeconds: 2, mm3RefinerMix: 1,
  });
  const saved = { mm3RefinerEnabled: true, mm3RefinerSteps: 8, mm3RefinerWindowSeconds: 0,
    mm3RefinerOverlapSeconds: 0, mm3RefinerMix: 0.25 };
  assert.deepEqual(normalizeMm3RefinerParams(JSON.parse(JSON.stringify(saved))), saved);
});

test('off and mix=0 bypass without engine calls or WAV re-encoding', async t => {
  const submit = t.mock.method(aceClient, 'submitMm3Refine', async () => { throw new Error('must not call'); });
  const source = wav();
  const original = Buffer.from(source);
  for (const params of [{}, { mm3RefinerEnabled: true, mm3RefinerMix: 0 }]) {
    assert.equal(await runMm3Refiner(source, params, () => {}, () => {}), null);
    assert.deepEqual(source, original);
  }
  assert.equal(submit.mock.callCount(), 0);
});

test('client sends WAV content type and the frozen query, retaining HTTP reasons', async t => {
  const source = wav();
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  t.mock.method(globalThis, 'fetch', async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), init });
    return calls.length === 1 ? Response.json({ id: 'refine-job' })
      : Response.json({ error: 'no style source' }, { status: 501 });
  });
  const opts = { steps: 8, window: 0, overlap: 0, mix: 0.5 };
  assert.equal(await aceClient.submitMm3Refine(source, opts), 'refine-job');
  const url = new URL(calls[0].url);
  assert.equal(url.pathname, '/mm3/refine');
  assert.deepEqual(Object.fromEntries(url.searchParams), { steps: '8', window: '0', overlap: '0', mix: '0.5' });
  assert.equal(new Headers(calls[0].init?.headers).get('Content-Type'), 'audio/wav');
  assert.equal(calls[0].init?.body, source);
  await assert.rejects(aceClient.submitMm3Refine(source, opts), (error: any) => error.status === 501 && /no style source/.test(error.message));
});

test('condition counters can reset; returned audio retains input rate and length', async t => {
  const source = wav();
  const result = wav();
  const submit = t.mock.method(aceClient, 'submitMm3Refine', async () => 'refine-job');
  const phases = [
    { status: 'running', phase: 'refine_condition', phase_step: 2, phase_total: 2 },
    { status: 'running', phase: 'refine_condition', phase_step: 1, phase_total: 2 },
    { status: 'done', phase: 'done' },
  ];
  t.mock.method(aceClient, 'pollJob', async () => phases.shift() as any);
  t.mock.method(aceClient, 'getJobResult', async () => new Response(result));
  const stages: string[] = [];
  assert.deepEqual(await runMm3Refiner(source, { mm3RefinerEnabled: true }, () => {}, stage => stages.push(stage)), result);
  assert.deepEqual(submit.mock.calls[0].arguments[1], { steps: 32, window: 30, overlap: 2, mix: 1 });
  assert.ok(stages.some(stage => stage.endsWith('(2/2)')));
  assert.ok(stages.some(stage => stage.endsWith('(1/2)')));
});

test('missing files, no style source and the sequence limit warn and preserve source', async t => {
  for (const reason of ['501: missing mm3-refiner-mert-*.gguf', '501: no style source', '413: sequence exceeds 5168 frames']) {
    t.mock.method(aceClient, 'submitMm3Refine', async () => { throw new Error(reason); });
    const messages: string[] = [];
    const source = wav(), original = Buffer.from(source);
    assert.equal(await runMm3Refiner(source, { mm3RefinerEnabled: true }, (level, msg) => {
      assert.equal(level, 'WARNING'); messages.push(msg);
    }, () => {}), null);
    assert.ok(messages[0].includes(reason));
    assert.deepEqual(source, original);
    t.mock.restoreAll();
  }
});

test('invalid settings decline before queueing', async t => {
  const submit = t.mock.method(aceClient, 'submitMm3Refine', async () => 'unexpected');
  for (const params of [
    { mm3RefinerSteps: 0 }, { mm3RefinerSteps: 1.5 }, { mm3RefinerSteps: 65 },
    { mm3RefinerWindowSeconds: -1 }, { mm3RefinerWindowSeconds: NaN },
    { mm3RefinerWindowSeconds: 2, mm3RefinerOverlapSeconds: 2 },
    { mm3RefinerOverlapSeconds: -1 }, { mm3RefinerMix: 2 }, { mm3RefinerMix: Infinity },
  ]) {
    assert.equal(await runMm3Refiner(wav(), { mm3RefinerEnabled: true, ...params }, () => {}, () => {}), null);
  }
  assert.equal(submit.mock.callCount(), 0);
});

test('failed and cancelled engine jobs keep raw audio and their reason', async t => {
  for (const status of ['failed', 'cancelled']) {
    t.mock.method(aceClient, 'submitMm3Refine', async () => 'refine-job');
    t.mock.method(aceClient, 'pollJob', async () => ({ status, error: 'conditioner load failed' }) as any);
    const result = t.mock.method(aceClient, 'getJobResult', async () => new Response(wav()));
    const messages: string[] = [];
    assert.equal(await runMm3Refiner(wav(), { mm3RefinerEnabled: true }, (_, msg) => messages.push(msg), () => {}), null);
    assert.match(messages[0], /conditioner load failed/);
    assert.equal(result.mock.callCount(), 0);
    t.mock.restoreAll();
  }
});

test('poll transport failure cancels the outstanding job before declining', async t => {
  t.mock.method(aceClient, 'submitMm3Refine', async () => 'refine-job');
  let polls = 0;
  t.mock.method(aceClient, 'pollJob', async () => {
    if (++polls === 1) throw new Error('connection lost');
    return { status: 'cancelled' };
  });
  const cancel = t.mock.method(aceClient, 'cancelJob', async () => {});
  assert.equal(await runMm3Refiner(wav(), { mm3RefinerEnabled: true }, () => {}, () => {}), null);
  assert.equal(cancel.mock.calls[0].arguments[0], 'refine-job');
  assert.equal(polls, 2);
});

test('wrong-rate, wrong-length and non-WAV results cannot overwrite raw audio', async t => {
  for (const result of [wav(44100), wav(48000, 258), Buffer.from('not a WAV')]) {
    t.mock.method(aceClient, 'submitMm3Refine', async () => 'refine-job');
    t.mock.method(aceClient, 'pollJob', async () => ({ status: 'done' }));
    t.mock.method(aceClient, 'getJobResult', async () => new Response(result));
    assert.equal(await runMm3Refiner(wav(), { mm3RefinerEnabled: true }, () => {}, () => {}), null);
    t.mock.restoreAll();
  }
});

test('public refiner pack has measured downloads, required licences and a non-commercial MERT notice', () => {
  const registry = JSON.parse(fs.readFileSync(new URL('../../data/model-registry.json', import.meta.url), 'utf8'));
  assert.equal(registry.pending, undefined);
  const pack = registry.packs.find((p: any) => p.id === 'mm3-latent-refiner');
  assert.ok(pack);
  for (const id of pack.fileIds) {
    const file = registry.files.find((f: any) => f.id === id);
    assert.ok(file, id);
    assert.ok(file.sizeBytes > 0, id);
    assert.ok(file.companions.length > 0, id);
  }
  const refiner = registry.files.find((f: any) => f.id === 'mm3-refiner-refiner-f32');
  const mert = registry.files.find((f: any) => f.id === 'mm3-refiner-mert-f32');
  assert.equal(refiner.sizeBytes, 549104864);
  assert.equal(mert.sizeBytes, 377505664);
  for (const file of [refiner, mert]) {
    assert.equal(file.repo, 'scragnog/MiniMax-Music3-GGUF');
    assert.match(file.sha256, /^[a-f0-9]{64}$/);
    assert.ok(file.companions.every((c: any) => c.required));
  }
  assert.match(mert.licence, /CC-BY-NC-4.0/);
  for (const text of [mert.displayName, mert.description, mert.licence, pack.description]) {
    assert.match(text, /non-commercial/i);
  }
  assert.equal(pack.pendingComponents, undefined);
  assert.deepEqual(pack.fileIds, ['mm3-refiner-refiner-f32', 'mm3-refiner-mert-f32', 'mm3-refiner-clap-f32', 'mm3-enc-f16', 'mm3-voc-f16']);
  const clap = registry.files.find((f: any) => f.id === 'mm3-refiner-clap-f32');
  assert.equal(clap.sizeBytes, 274780768);
  assert.equal(clap.repo, 'scragnog/MiniMax-Music3-GGUF');
  assert.match(clap.sha256, /^[a-f0-9]{64}$/);
  assert.equal(clap.licence, 'Apache-2.0');
  assert.ok(clap.companions.every((c: any) => c.required));
  const visible = modelDownloadService.getRegistry();
  assert.ok(visible.packs.some(p => p.id === pack.id));
  assert.ok([refiner, mert, clap].every(file => visible.files.some(f => f.id === file.id)));
});

test('compact generation intents retain refiner settings on every backend and omit them when off', () => {
  const explicit = { mm3RefinerEnabled: true, mm3RefinerSteps: 8,
    mm3RefinerWindowSeconds: 0, mm3RefinerOverlapSeconds: 0, mm3RefinerMix: 0 };
  const keys = Object.keys(explicit);
  for (const backend of ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula']) {
    const resolve = (params: Record<string, unknown>) => JSON.parse(JSON.stringify(
      resolveGenerationIntent({ contract: 'generation-intent/1', params }, backend, []),
    ));
    const enabled = resolve({ postProcessingEnabled: true, ...explicit });
    for (const key of keys) assert.equal(enabled[key], explicit[key as keyof typeof explicit], `${backend}: ${key}`);
    const defaults = resolve({ mm3RefinerEnabled: true });
    assert.deepEqual(Object.fromEntries(keys.map(key => [key, defaults[key]])), {
      mm3RefinerEnabled: true, mm3RefinerSteps: 32, mm3RefinerWindowSeconds: 30,
      mm3RefinerOverlapSeconds: 2, mm3RefinerMix: 1,
    });
    for (const params of [{}, { ...explicit, mm3RefinerEnabled: false },
      { ...explicit, postProcessingEnabled: false }]) {
      const off = resolve(params);
      for (const key of keys) assert.ok(!(key in off), `${backend}: off must omit ${key}`);
    }
  }
});
