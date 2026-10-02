import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  buildYue2SheetArgs, readYue2CoverAbc, writeYue2CoverSheetManifest,
  type ResolvedYue2SheetOptions,
} from './yue2Sheet.js';

const trainingOptions: ResolvedYue2SheetOptions = {
  manifest: 'training.json', only: '', force: false, fast: false, datasetSlug: 'fixture',
};

test('dataset sheet arguments keep the full-score default', () => {
  const args = buildYue2SheetArgs(trainingOptions);
  assert.equal(args.includes('--melody-only'), false);
  assert.deepEqual(buildYue2SheetArgs({ ...trainingOptions, melodyOnly: false }), args);
});

test('cover sheet arguments request melody-only notation', () => {
  const args = buildYue2SheetArgs({ ...trainingOptions, melodyOnly: true });
  assert.equal(args.filter(arg => arg === '--melody-only').length, 1);
});

test('a cover source gets a private one-source manifest and usable ABC', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cover-sheet-'));
  try {
    const audio = path.join(root, 'source.wav');
    fs.writeFileSync(audio, 'audio fixture');
    const { manifest, name } = writeYue2CoverSheetManifest(audio, path.join(root, 'job'));
    assert.deepEqual(JSON.parse(fs.readFileSync(manifest, 'utf8')), {
      sources: [{ name: 'source.wav', source: audio }],
    });
    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name, source: audio, abc: 'X:1\nK:C' }] }));
    assert.equal(readYue2CoverAbc(manifest, name), 'X:1\nK:C');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('abc_error fails cover transcription even when ABC is present', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cover-error-'));
  try {
    const manifest = path.join(root, 'cover.json');
    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name: 'source', abc: 'X:1', abc_error: 'notation failed' }] }));
    assert.throws(() => readYue2CoverAbc(manifest, 'source'), /notation failed/);
    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name: 'source', abc: 'X:1', abc_error: '  ' }] }));
    assert.throws(() => readYue2CoverAbc(manifest, 'source'), /unknown notation error/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('blank ABC fails cover transcription', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cover-blank-'));
  try {
    const manifest = path.join(root, 'cover.json');
    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name: 'source', abc: '   ' }] }));
    assert.throws(() => readYue2CoverAbc(manifest, 'source'), /returned no ABC/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
