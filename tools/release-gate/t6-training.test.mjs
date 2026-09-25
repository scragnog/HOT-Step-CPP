// Tier 6 — training, a few steps of each trainer, then use what it made.
//
// The dataset is built from this run's own renders (no committed audio), so
// the tier also covers folder ingestion. One epoch on three clips is three
// steps: enough to prove the path runs end to end and writes an adapter the
// engine can load. Nothing here says anything about adapter quality.
import test, { before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  MIN, RUN_DIR, api, apiRaw, generate, assertAudible, pollTrainingJob, waitEngineReady, setBackend, state,
  CAPTION, LYRICS, MM3_CAPTION,
} from './lib.mjs';

const NAME = 'release-gate';
const dsDir = path.join(RUN_DIR, 'dataset');
let caps = null;
let ds = null;

before(async () => {
  await setBackend('ace');
  caps = await api('GET', '/api/training/capabilities');

  // Fixture folder: at least three clips, each with caption + lyrics sidecars
  // and an MM3 caption beside it.
  let renders = state.get('renders', []).filter((r) => r.file.endsWith('.wav') && fs.existsSync(r.file));
  while (renders.length < 3) {
    const g = await generate(`t6-fixture-${renders.length}`, {
      prompt: CAPTION, lyrics: LYRICS, instrumental: false, duration: 10, inferenceSteps: 6,
      seed: 1000 + renders.length, randomSeed: false, skipLm: true,
    });
    renders = state.get('renders', []).filter((r) => r.file.endsWith('.wav'));
    void g;
  }
  fs.rmSync(dsDir, { recursive: true, force: true });
  fs.mkdirSync(dsDir, { recursive: true });
  renders.slice(0, 4).forEach((r, i) => {
    const stem = path.join(dsDir, `gate-${i + 1}`);
    fs.copyFileSync(r.file, `${stem}.wav`);
    fs.writeFileSync(`${stem}.caption.txt`, CAPTION);
    fs.writeFileSync(`${stem}.lyrics.txt`, LYRICS);
    fs.writeFileSync(`${stem}.mm3.txt`, MM3_CAPTION);
  });

  // Previous gate datasets: drop the record (the route leaves source folders alone).
  const { datasets = [] } = await api('GET', '/api/training/datasets');
  for (const d of datasets) if (d.name === NAME) await api('DELETE', `/api/training/datasets/${d.id}`);
});

test('dataset: create from the fixture folder', { timeout: 5 * MIN }, async (t) => {
  const r = await api('POST', '/api/training/datasets', {
    name: NAME, sourceDir: dsDir, customTag: 'rgate',
    defaultArtist: 'Release Gate', defaultAlbum: 'Gate', defaultGenre: 'indie pop', defaultLanguage: 'english',
  });
  ds = r.dataset;
  assert.ok(ds?.id, `no dataset in ${JSON.stringify(r).slice(0, 300)}`);
  state.set('datasetId', ds.id);
  const detail = await api('GET', `/api/training/datasets/${ds.id}`);
  const samples = detail.samples ?? detail.dataset?.samples ?? [];
  t.diagnostic(`dataset ${ds.id} (${ds.slug ?? ''}), ${samples.length} samples`);
});

test('dataset: build dataset.json', { timeout: 10 * MIN }, async (t) => {
  if (!ds) return t.skip('no dataset');
  const { jobId } = await api('POST', `/api/training/datasets/${ds.id}/build`, {});
  await pollTrainingJob(jobId, 'build', { timeoutMs: 10 * MIN });
});

test('preprocess: cache tensors (stops and restarts the engine)', { timeout: 30 * MIN }, async (t) => {
  if (!ds) return t.skip('no dataset');
  if (!caps.preprocess?.available) return t.skip('preprocess unavailable (ace-train missing?)');
  const { jobId } = await api('POST', `/api/training/datasets/${ds.id}/preprocess`, { maxDuration: 30 });
  await pollTrainingJob(jobId, 'preprocess', { timeoutMs: 25 * MIN });
  await waitEngineReady();
});

test('train-dit: one epoch of LoRA, then generate with the adapter', { timeout: 45 * MIN }, async (t) => {
  if (!ds) return t.skip('no dataset');
  if (!caps.trainDit?.available) return t.skip('DiT training unavailable');
  const start = await apiRaw('POST', `/api/training/datasets/${ds.id}/train-dit`, {
    epochs: 1, rank: 8, alpha: 16, adapterType: 'lora', adapterName: NAME,
  });
  if (start.status === 400) return t.skip(`train-dit refused: ${start.body?.error}`);
  assert.equal(start.status, 202, JSON.stringify(start.body).slice(0, 300));
  await pollTrainingJob(start.body.jobId, 'train-dit', { timeoutMs: 30 * MIN });
  const s = await api('GET', `/api/training/datasets/${ds.id}/train-dit?adapterName=${NAME}`);
  assert.ok(s.adapterExists, `no adapter written at ${s.adapterDir}`);
  state.set('trainedDitAdapter', s.adapterDir);
  await waitEngineReady();
  const g = await generate('t6-dit-adapter', {
    prompt: CAPTION, instrumental: true, duration: 6, inferenceSteps: 6, seed: 42, randomSeed: false, skipLm: true,
    loraPath: s.adapterDir,
  });
  assertAudible(assert, g.file, 'trained DiT adapter');
  t.diagnostic(`adapter ${s.adapterDir} (${s.adapterBytes} bytes), render ${g.seconds} s`);
});

test('train-lm: one epoch on the 0.6B planner, then generate with the adapter', { timeout: 45 * MIN }, async (t) => {
  if (!ds) return t.skip('no dataset');
  if (!caps.trainLm?.available) return t.skip('LM training unavailable');
  if (!caps.trainLm.defaultLmBySize?.['0.6B']) return t.skip('no 0.6B BF16 LM installed');
  const start = await apiRaw('POST', `/api/training/datasets/${ds.id}/train-lm`, {
    epochs: 1, rank: 8, lmSize: '0.6B', adapterName: NAME,
  });
  if (start.status === 400) return t.skip(`train-lm refused: ${start.body?.error}`);
  assert.equal(start.status, 202, JSON.stringify(start.body).slice(0, 300));
  await pollTrainingJob(start.body.jobId, 'train-lm', { timeoutMs: 30 * MIN });
  const s = await api('GET', `/api/training/datasets/${ds.id}/train-lm?adapterName=${NAME}`);
  assert.ok(s.adapterExists, `no adapter written at ${s.adapterDir}`);
  state.set('trainedLmAdapter', s.adapterDir);
  await waitEngineReady();
  const g = await generate('t6-lm-adapter', {
    prompt: CAPTION, lyrics: LYRICS, instrumental: false, duration: 10, inferenceSteps: 6, seed: 42, randomSeed: false,
    skipLm: false, lmAdapter: s.adapterDir,
  });
  assertAudible(assert, g.file, 'trained LM adapter');
  t.diagnostic(`adapter ${s.adapterDir}, render ${g.seconds} s`);
});

test('MiniMax-Music3: encode codes, train the planner LM for 5 steps', { timeout: 60 * MIN }, async (t) => {
  if (!ds) return t.skip('no dataset');
  const codes = await apiRaw('POST', `/api/training/datasets/${ds.id}/mm3-codes`, {});
  if (codes.status === 400 || codes.status === 503) return t.skip(`mm3-codes refused: ${codes.body?.error}`);
  assert.equal(codes.status, 200, JSON.stringify(codes.body).slice(0, 300));
  await pollTrainingJob(codes.body.jobId, 'mm3-codes', { timeoutMs: 20 * MIN });
  const train = await apiRaw('POST', `/api/training/datasets/${ds.id}/mm3-train-lm`, { steps: 5, saveEvery: 5 });
  if (train.status === 400 || train.status === 503) return t.skip(`mm3-train-lm refused: ${train.body?.error}`);
  assert.equal(train.status, 200, JSON.stringify(train.body).slice(0, 300));
  await pollTrainingJob(train.body.jobId, 'mm3-train-lm', { timeoutMs: 30 * MIN });
  const runs = await api('GET', `/api/training/datasets/${ds.id}/mm3-runs`);
  assert.ok((runs.runs ?? []).length > 0, 'training finished but no MM3 run is listed');
  await waitEngineReady();
  t.diagnostic(`run ${runs.runs[0]?.dir ?? runs.runs[0]?.path ?? JSON.stringify(runs.runs[0]).slice(0, 120)}`);
});

test('YuE2: joint train 5 steps with automatic preparation', { timeout: 60 * MIN }, async (t) => {
  if (!ds) return t.skip('no dataset');
  const start = await apiRaw('POST', `/api/training/datasets/${ds.id}/yue2-joint-train`, {
    trainingMethod: 'aitk', steps: 5, saveEvery: 5, autoPrepare: true,
  });
  if (start.status === 400 || start.status === 503) return t.skip(`yue2-joint-train refused: ${start.body?.error}`);
  assert.equal(start.status, 200, JSON.stringify(start.body).slice(0, 300));
  await pollTrainingJob(start.body.jobId, 'yue2-joint-train', { timeoutMs: 50 * MIN });
  const runs = await api('GET', `/api/training/datasets/${ds.id}/yue2-joint-runs`);
  assert.ok((runs.runs ?? []).length > 0, 'training finished but no YuE2 joint run is listed');
  await waitEngineReady();
});
