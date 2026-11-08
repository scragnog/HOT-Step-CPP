import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { MM3_LM_DEFAULTS, buildMm3TrainLmArgs, prepareMm3CodesInput, type ResolvedMm3TrainLmOptions } from './mm3Train.js';
import { resumeOptionsFor, writeMm3RunManifest } from './mm3Runs.js';
import { mm3ExecutionBody } from './recipes/operations.js';

function opts(over: Partial<ResolvedMm3TrainLmOptions>): ResolvedMm3TrainLmOptions {
  return {
    ...MM3_LM_DEFAULTS, manifest: 'm.json', captionsDir: 'c', codesDir: 'd', outDir: 'o',
    trigger: '', datasetName: 'ds', ...over,
  } as unknown as ResolvedMm3TrainLmOptions;
}
const flag = (args: string[], name: string) => {
  const i = args.indexOf(name);
  return i < 0 ? null : args[i + 1];
};

test('training precision maps to the trainer flags', () => {
  const cases: Array<[Partial<ResolvedMm3TrainLmOptions>, string, string | null, string | null]> = [
    [{ basePrecision: 'q8_0' }, 'bf16', 'flash-bf16', 'on'],                              // the default = fastest
    [{ basePrecision: 'q8_0', trainPrecision: undefined }, 'f32-window', 'flash', null],  // absent (old manifest) = full
    [{ basePrecision: 'q8_0', trainPrecision: 'full' }, 'f32-window', 'flash', null],
    [{ basePrecision: 'q8_0', trainPrecision: 'fast' }, 'bf16', 'flash', 'on'],
    [{ basePrecision: 'q8_0', trainPrecision: 'fastest' }, 'bf16', 'flash-bf16', 'on'],
    // Attention off (or coerced to exact): Fastest trains as Fast, no --attn.
    [{ basePrecision: 'q8_0', trainPrecision: 'fastest', attnBackend: 'exact' }, 'bf16', null, 'on'],
    [{ basePrecision: 'q8_0', trainPrecision: 'full', attnBackend: 'exact' }, 'f32-window', null, null],
    // A BF16 base always trains Lever A, as before; Full still keeps strict FP32.
    [{ basePrecision: 'bf16', trainPrecision: 'full' }, 'bf16', 'flash', null],
  ];
  for (const [over, weights, attn, tf32] of cases) {
    const args = buildMm3TrainLmArgs(opts(over));
    assert.equal(flag(args, '--weights'), weights, JSON.stringify(over));
    assert.equal(flag(args, '--attn'), attn, JSON.stringify(over));
    assert.equal(flag(args, '--tf32'), tf32, JSON.stringify(over));
  }
});

test('a preview relaunch keeps the precision flags', () => {
  const o = opts({ basePrecision: 'q8_0', trainPrecision: 'fastest' });
  const args = buildMm3TrainLmArgs({ ...o, resumeFrom: 'state.bin' });
  assert.equal(flag(args, '--weights'), 'bf16');
  assert.equal(flag(args, '--attn'), 'flash-bf16');
  assert.equal(flag(args, '--tf32'), 'on');
  assert.equal(flag(args, '--resume'), 'state.bin');
});

test('a resumed run reads its precision back from the run manifest', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-precision-'));
  try {
    writeMm3RunManifest(opts({ outDir: dir, basePrecision: 'q8_0', trainPrecision: 'fastest' }),
                        { datasetId: 'id', datasetSlug: 'ds', datasetName: 'ds' });
    const back = resumeOptionsFor(dir);
    assert.ok(back);
    const args = buildMm3TrainLmArgs({ ...back, resumeFrom: 'state.bin' });
    assert.equal(flag(args, '--weights'), 'bf16');
    assert.equal(flag(args, '--attn'), 'flash-bf16');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('the default training precision is fastest', () => {
  assert.equal(MM3_LM_DEFAULTS.trainPrecision, 'fastest');
});

test('the recipe projection sends the chosen precision, Full included', () => {
  const form = { ...MM3_LM_DEFAULTS, trigger: '', previewCaption: '', regDatasetId: '' } as Record<string, any>;
  assert.equal(mm3ExecutionBody(form, 'custom', false, true).trainPrecision, 'fastest');
  assert.equal(mm3ExecutionBody({ ...form, trainPrecision: 'full' }, 'custom', false, true).trainPrecision, 'full');
  assert.equal('trainPrecision' in mm3ExecutionBody({ ...form, trainPrecision: undefined }, 'custom', false, true), false);
});

test('mm3ExecutionBody carries warmup, lyric dropout, LR end fraction and preview lyrics', () => {
  const form = { ...MM3_LM_DEFAULTS, trigger: '', cropAnchor: 'song', prefixFrames: 0, adapterType: 'lora',
    warmup: 7, lyricsDropout: 0.2, lrEndFrac: 0.1, previewEverySteps: 50, previewCaption: 'c', previewLyrics: ' la la ' };
  const body = mm3ExecutionBody(form as any, 'custom', false, true) as any;
  assert.equal(body.warmup, 7);
  assert.equal(body.lyricsDropout, 0.2);
  assert.equal(body.lrEndFrac, 0.1);
  assert.equal(body.preview.lyrics, 'la la');
});

test('prepareMm3CodesInput: force clears the cache, filter and limit write a subset beside the full manifest', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3codes-'));
  const full = path.join(dir, 'dataset.json');
  const samples = ['Alpha one.flac', 'beta.flac', 'alpha two.flac'].map((f, i) => ({ id: String(i), filename: f, audio_path: `/x/${f}` }));
  fs.writeFileSync(full, JSON.stringify({ metadata: {}, samples }));
  const outDir = path.join(dir, 'mm3-codes');
  fs.mkdirSync(path.join(outDir, 'codes'), { recursive: true });
  fs.writeFileSync(path.join(outDir, 'codes', 'a.codes'), 'x');

  assert.equal(prepareMm3CodesInput({ datasetJson: full, outDir }), full);
  assert.ok(fs.existsSync(path.join(outDir, 'codes', 'a.codes')));

  const subset = prepareMm3CodesInput({ datasetJson: full, outDir, force: true, only: 'ALPHA', limit: 1 });
  assert.ok(!fs.existsSync(outDir));
  assert.notEqual(subset, full);
  assert.deepEqual(JSON.parse(fs.readFileSync(subset, 'utf8')).samples.map((s: any) => s.id), ['0']);
  assert.equal(JSON.parse(fs.readFileSync(full, 'utf8')).samples.length, 3);
  assert.throws(() => prepareMm3CodesInput({ datasetJson: full, outDir, only: 'zzz' }), /No tracks match/);
  fs.rmSync(dir, { recursive: true, force: true });
});
