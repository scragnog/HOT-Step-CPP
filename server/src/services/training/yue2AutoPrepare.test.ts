import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ensureYue2PreparedDataset } from './yue2AutoPrepare.js';
import type { TrainingJob } from './labelingQueue.js';
import type { ResolvedYue2AitkPrepareOptions } from './yue2AitkPrepareRunner.js';

function fixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-auto-prepare-'));
  const file = (name: string, value = 'fixture') => { const p = path.join(dir, name); fs.writeFileSync(p, value); return p; };
  const o: ResolvedYue2AitkPrepareOptions = {
    legacyManifest: file('legacy.json', JSON.stringify({ sources: [{ latents: 'latents', codec_ids: 'codes', cursor_words: 'words' }] })),
    checkpoint: file('base'), tokenizer: file('tokenizer'), output: path.join(dir, 'unused'),
    models: { vae: file('vae'), semantic: file('semantic'), sheetsage: file('sheet') }, lyricTiming: true,
  };
  file('latents'); file('codes'); file('words');
  const job = (): TrainingJob => ({ id: 'fixture', datasetId: 'fixture', kind: 'yue2-joint-train', status: 'queued',
    total: 1, done: 0, failed: 0, currentSampleId: null, phase: '', engineQueueDepth: 0, error: null,
    createdAt: Date.now(), startedAt: null, finishedAt: null, sampleIds: [], opts: {}, events: [], listeners: new Set(), controller: new AbortController() });
  let calls = 0;
  const prepare = async (_job: TrainingJob, opts?: ResolvedYue2AitkPrepareOptions) => {
    ++calls; assert.ok(opts);
    fs.mkdirSync(opts.output);
    fs.writeFileSync(path.join(opts.output, 'latent.bin'), 'prepared');
    fs.writeFileSync(path.join(opts.output, 'dataset.json'), JSON.stringify({ schema_version: 1,
      recipe_version: 'aitk-yue2-2026-09-16', items: [{ latent_file: 'latent.bin' }] }));
  };
  return { dir, o, job, prepare, calls: () => calls };
}

test('automatic preparation reuses unchanged inputs, but invalidates changed settings, caches and models', async () => {
  const f = fixture();
  const first = await ensureYue2PreparedDataset(f.job(), f.o, f.prepare);
  assert.equal(await ensureYue2PreparedDataset(f.job(), f.o, f.prepare), first);
  assert.equal(f.calls(), 1);
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare), first);
  fs.appendFileSync(path.join(f.dir, 'codes'), 'new');
  await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare);
  fs.appendFileSync(f.o.checkpoint, 'changed');
  await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare);
  fs.appendFileSync(f.o.legacyManifest, ' ');
  await ensureYue2PreparedDataset(f.job(), { ...f.o, lyricTiming: false }, f.prepare);
  assert.equal(f.calls(), 5);
});

test('missing prepared payload is regenerated instead of reused', async () => {
  const f = fixture();
  const first = await ensureYue2PreparedDataset(f.job(), f.o, f.prepare);
  assert.ok(first);
  fs.unlinkSync(path.join(path.dirname(first), 'latent.bin'));
  assert.notEqual(await ensureYue2PreparedDataset(f.job(), f.o, f.prepare), first);
  assert.equal(f.calls(), 2);
});

test('cancelled or failed preparation cannot hand a dataset to training', async () => {
  const f = fixture();
  const cancelled = f.job(); cancelled.controller.abort();
  assert.equal(await ensureYue2PreparedDataset(cancelled, f.o, f.prepare), undefined);
  assert.equal(f.calls(), 0);
  const during = f.job();
  assert.equal(await ensureYue2PreparedDataset(during, f.o, async j => { j.controller.abort(); }), undefined);
  const failed = f.job();
  assert.equal(await ensureYue2PreparedDataset(failed, f.o, async j => { j.status = 'failed'; }), undefined);
  assert.equal(fs.existsSync(path.join(f.dir, 'aitk-auto-prepare-v1.json')), false);
});

test('source changes during preparation prevent handoff and cache publication', async () => {
  const f = fixture();
  await assert.rejects(ensureYue2PreparedDataset(f.job(), f.o, async (j, o) => {
    await f.prepare(j, o); fs.appendFileSync(path.join(f.dir, 'words'), 'updated');
  }), /changed during preparation/);
  assert.equal(fs.existsSync(path.join(f.dir, 'aitk-auto-prepare-v1.json')), false);
});
