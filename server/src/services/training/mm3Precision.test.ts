import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { MM3_LM_DEFAULTS, buildMm3TrainLmArgs, type ResolvedMm3TrainLmOptions } from './mm3Train.js';
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
  const cases: Array<[Partial<ResolvedMm3TrainLmOptions>, string, string | null]> = [
    [{ basePrecision: 'q8_0' }, 'f32-window', 'flash'],                                   // absent = full
    [{ basePrecision: 'q8_0', trainPrecision: 'full' }, 'f32-window', 'flash'],
    [{ basePrecision: 'q8_0', trainPrecision: 'fast' }, 'bf16', 'flash'],
    [{ basePrecision: 'q8_0', trainPrecision: 'fastest' }, 'bf16', 'flash-bf16'],
    // Attention off (or coerced to exact): Fastest trains as Fast, no --attn.
    [{ basePrecision: 'q8_0', trainPrecision: 'fastest', attnBackend: 'exact' }, 'bf16', null],
    [{ basePrecision: 'q8_0', trainPrecision: 'full', attnBackend: 'exact' }, 'f32-window', null],
    // A BF16 base always trains Lever A, as before.
    [{ basePrecision: 'bf16', trainPrecision: 'full' }, 'bf16', 'flash'],
  ];
  for (const [over, weights, attn] of cases) {
    const args = buildMm3TrainLmArgs(opts(over));
    assert.equal(flag(args, '--weights'), weights, JSON.stringify(over));
    assert.equal(flag(args, '--attn'), attn, JSON.stringify(over));
    assert.ok(!args.includes('--tf32'), 'tf32 stays command-line only');
  }
});

test('a preview relaunch keeps the precision flags', () => {
  const o = opts({ basePrecision: 'q8_0', trainPrecision: 'fastest' });
  const args = buildMm3TrainLmArgs({ ...o, resumeFrom: 'state.bin' });
  assert.equal(flag(args, '--weights'), 'bf16');
  assert.equal(flag(args, '--attn'), 'flash-bf16');
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

test('the recipe projection sends trainPrecision only when it is not full', () => {
  const form = { ...MM3_LM_DEFAULTS, trigger: '', previewCaption: '', regDatasetId: '' } as Record<string, any>;
  assert.equal('trainPrecision' in mm3ExecutionBody(form, 'custom', false, true), false);
  assert.equal(mm3ExecutionBody({ ...form, trainPrecision: 'fast' }, 'custom', false, true).trainPrecision, 'fast');
});
