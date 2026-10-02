import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { resolveCoverDatasetSource } from './coverDatasetSource.js';
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
