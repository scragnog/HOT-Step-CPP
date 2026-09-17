import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  YUE2_JOINT_PREVIEW_DEFAULTS,
  listYue2JointPreviews, parseYue2JointPreviewOptions, recordYue2JointPreview,
  renderYue2JointPreview, resolveYue2JointPreview, Yue2PreviewCleanupError,
} from './yue2JointPreview.js';
import { checkpointRecords } from './yue2AitkRuns.js';

test('preview options default off and clamp bounded values', () => {
  const d = parseYue2JointPreviewOptions(undefined, 50);
  assert.equal(d.enabled, false);
  assert.equal(d.everySteps, 50);
  const p = parseYue2JointPreviewOptions({ enabled: true, everySteps: 10, seconds: 999,
    previewMaxFrames: 9000, seed: 3, caption: '  song  ' }, 50);
  assert.deepEqual({ enabled: true, everySteps: 10, seconds: 40,
    previewMaxFrames: 9000, seed: 3, baseline: false, control: false,
    caption: 'song' }, p);
});

test('preview catalogue is durable and rejects traversal', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-preview-'));
  const now = Date.now();
  recordYue2JointPreview(root, { id: 'p1', step: 10, kind: 'artist', status: 'done',
    file: 'step-10-artist.wav', seconds: 40, seed: 424242, previewMaxFrames: 1000,
    createdAt: now, updatedAt: now });
  assert.equal(listYue2JointPreviews(root)[0]?.id, 'p1');
  assert.equal(resolveYue2JointPreview(root, 'step-10-artist.wav'), null);
  fs.mkdirSync(path.join(root, 'previews'), { recursive: true });
  fs.writeFileSync(path.join(root, 'previews', 'step-10-artist.wav'), 'RIFF');
  assert.equal(resolveYue2JointPreview(root, 'step-10-artist.wav') !== null, true);
});

test('checkpoint catalogue scans only the known segment layout', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-segments-'));
  const dir = path.join(root, 'segments', 'segment-000001', 'checkpoint-step5');
  fs.mkdirSync(dir, { recursive: true });
  for (const name of ['adapter.safetensors', 'optimizer.resume', 'native-ar.safetensors', 'native-nar.safetensors']) fs.writeFileSync(path.join(dir, name), 'x');
  fs.mkdirSync(path.join(root, 'random', 'checkpoint-step99'), { recursive: true });
  assert.deepEqual(checkpointRecords(root).map(c => c.step), [5]);
});

function previewFixture() {
  const output = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-render-'));
  const dataset = path.join(output, 'dataset.json');
  fs.writeFileSync(dataset, JSON.stringify({ items: [
    { id: 'other', style: 'wrong', lyrics: 'wrong' },
    { id: 'chosen', style: 'chosen style', lyrics: 'chosen lyrics' },
  ] }));
  return { output, dataset };
}

function mockDeps(overrides: Record<string, unknown> = {}) {
  const calls: Array<{ name: string; value?: unknown }> = [];
  const deps = {
    select: async (value: unknown) => { calls.push({ name: 'select', value }); return {}; },
    warm: async (value: unknown) => { calls.push({ name: 'warm', value }); return { warm: true }; },
    synth: async (value: unknown) => { calls.push({ name: 'synth', value }); return { job_id: 'preview-job' }; },
    poll: async () => ({ status: 'done' }),
    result: async () => new Response(Buffer.from('RIFF-test')),
    cancel: async (value: unknown) => { calls.push({ name: 'cancel', value }); },
    unload: async () => { calls.push({ name: 'unload' }); return { unloaded: true }; },
    persisted: () => ({ lm: 'base-lm', vae_variant: 'standard', adapters: {
      ar: { path: '', scales: { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 } },
      nar: { path: '', scales: { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 } },
    } }),
    ...overrides,
  };
  return { deps: deps as any, calls };
}

test('renderer uses selected style/lyrics, writes audio, and restores selection', async () => {
  const f = previewFixture();
  const { deps, calls } = mockDeps();
  const record = await renderYue2JointPreview({ output: f.output, dataset: f.dataset, step: 4,
    options: { ...YUE2_JOINT_PREVIEW_DEFAULTS, enabled: true, previewSongId: 'chosen' },
    arAdapter: 'ar.safetensors', narAdapter: 'nar.safetensors', deps });
  assert.equal(record.status, 'done');
  assert.deepEqual((calls.find(c => c.name === 'synth')?.value as any).lyrics, 'chosen lyrics');
  assert.equal((calls.find(c => c.name === 'synth')?.value as any).style, 'chosen style');
  assert.equal(calls.filter(c => c.name === 'unload').length, 1);
  assert.equal(calls.at(-1)?.name, 'select');
  assert.equal(fs.existsSync(path.join(f.output, 'previews', 'step-4-artist.wav')), true);
});

test('renderer cancels a polling job on render failure and still unloads', async () => {
  const f = previewFixture();
  let polls = 0;
  const { deps, calls } = mockDeps({ poll: async () => { if (polls++ === 0) throw new Error('poll broke'); return { status: 'cancelled' }; } });
  await assert.rejects(() => renderYue2JointPreview({ output: f.output, dataset: f.dataset, step: 5,
    options: { ...YUE2_JOINT_PREVIEW_DEFAULTS, enabled: true }, arAdapter: 'ar', narAdapter: 'nar', deps }), /poll broke/);
  assert.equal(calls.some(c => c.name === 'cancel'), true);
  assert.equal(calls.some(c => c.name === 'unload'), true);
  assert.equal(listYue2JointPreviews(f.output)[0]?.status, 'failed');
});

test('renderer aborts before synth and reports cleanup failure', async () => {
  const f = previewFixture();
  const controller = new AbortController(); controller.abort();
  const { deps, calls } = mockDeps({ unload: async () => { calls.push({ name: 'unload' }); return null; } });
  await assert.rejects(() => renderYue2JointPreview({ output: f.output, dataset: f.dataset, step: 6,
    options: { ...YUE2_JOINT_PREVIEW_DEFAULTS, enabled: true }, arAdapter: 'ar', narAdapter: 'nar', signal: controller.signal, deps }),
    (err: unknown) => err instanceof Yue2PreviewCleanupError);
  assert.equal(calls.some(c => c.name === 'synth'), false);
  assert.equal(calls.some(c => c.name === 'unload'), true);
});
