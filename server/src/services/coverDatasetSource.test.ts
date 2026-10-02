import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { resolveCoverDatasetSource, saveDatasetCoverAbc } from './coverDatasetSource.js';
import { readAbcSidecar } from './training/abcSidecar.js';
import { sampleIdFor } from './training/paths.js';
import type { TrainingDatasetRow } from './training/types.js';

test('cover source resolves by dataset path, then exact copied bytes', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cover-dataset-'));
  try {
    const sourceDir = path.join(root, 'dataset');
    fs.mkdirSync(sourceDir);
    const original = path.join(sourceDir, 'track.wav');
    const copied = path.join(root, 'upload.wav');
    fs.writeFileSync(original, 'the same source bytes');
    fs.copyFileSync(original, copied);
    const dataset = { id: 'dataset-1', sourceDir, recursive: false } as TrainingDatasetRow;
    const expected = { datasetId: 'dataset-1', sampleId: sampleIdFor('track.wav'), audioPath: original };
    assert.deepEqual(resolveCoverDatasetSource(original, [dataset]), expected);
    assert.deepEqual(resolveCoverDatasetSource(copied, [dataset]), expected);
    assert.equal(saveDatasetCoverAbc(copied, 'X:1\nK:C\nC', file => resolveCoverDatasetSource(file, [dataset])), true);
    assert.equal(readAbcSidecar(original), 'X:1\nK:C\nC');
    fs.writeFileSync(copied, 'different source bytes');
    assert.equal(resolveCoverDatasetSource(copied, [dataset]), null);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('ambiguous byte matches do not choose a sidecar', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cover-dataset-'));
  try {
    const sourceDir = path.join(root, 'dataset');
    fs.mkdirSync(sourceDir);
    fs.writeFileSync(path.join(sourceDir, 'a.wav'), 'same');
    fs.writeFileSync(path.join(sourceDir, 'b.wav'), 'same');
    const copied = path.join(root, 'upload.wav');
    fs.writeFileSync(copied, 'same');
    const dataset = { id: 'dataset-1', sourceDir, recursive: false } as TrainingDatasetRow;
    assert.equal(resolveCoverDatasetSource(copied, [dataset]), null);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('repeated lookup uses the canonical index and row updates invalidate it', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cover-index-'));
  const originalReaddir = fs.readdirSync;
  let walks = 0;
  try {
    const sourceDir = path.join(root, 'dataset');
    fs.mkdirSync(sourceDir);
    const first = path.join(sourceDir, 'first.wav');
    fs.writeFileSync(first, 'first');
    const dataset = { id: `index-${path.basename(root)}`, sourceDir, recursive: false,
      updatedAt: 'one', sampleCount: 1 } as TrainingDatasetRow;
    (fs as any).readdirSync = (...args: any[]) => { walks++; return (originalReaddir as any)(...args); };
    assert.ok(resolveCoverDatasetSource(first, [dataset]));
    const firstWalks = walks;
    assert.ok(firstWalks > 0);
    assert.ok(resolveCoverDatasetSource(first, [dataset]));
    assert.equal(walks, firstWalks);
    const second = path.join(sourceDir, 'second.wav');
    fs.writeFileSync(second, 'second');
    assert.ok(resolveCoverDatasetSource(second, [{ ...dataset, updatedAt: 'two' }]));
    assert.ok(walks > firstWalks);
    const secondWalks = walks;
    const third = path.join(sourceDir, 'third.wav');
    fs.writeFileSync(third, 'third');
    assert.ok(resolveCoverDatasetSource(third, [{ ...dataset, updatedAt: 'two', sampleCount: 3 }]));
    assert.ok(walks > secondWalks);
  } finally {
    (fs as any).readdirSync = originalReaddir;
    fs.rmSync(root, { recursive: true, force: true });
  }
});
