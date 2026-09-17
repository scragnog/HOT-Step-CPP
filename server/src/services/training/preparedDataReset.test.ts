import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { clearPreparedCaches, listPreparedCaches } from './preparedDataReset.js';
import { datasetDir } from './paths.js';
import { tensorsRoot } from './aceTrain.js';

test('prepared-data reset removes generated caches but preserves labels and source files', () => {
  const slug = `reset-fixture-${process.pid}-${Date.now()}`;
  const root = datasetDir(slug);
  const tensor = tensorsRoot(slug);
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'hotstep-reset-source-'));
  try {
    fs.mkdirSync(path.join(root, 'labels'), { recursive: true });
    fs.writeFileSync(path.join(root, 'labels', 'edited.json'), '{}');
    fs.mkdirSync(path.join(root, 'yue2-latents', 'codes'), { recursive: true });
    fs.writeFileSync(path.join(root, 'yue2-latents', 'codes', 'track.codes'), 'data');
    fs.mkdirSync(tensor, { recursive: true });
    fs.writeFileSync(path.join(tensor, 'tensor.bin'), 'data');
    fs.writeFileSync(path.join(source, 'track.wav'), 'source');
    assert.equal(listPreparedCaches(slug, source).length, 2);
    assert.equal(clearPreparedCaches(slug, source).length, 2);
    assert.equal(fs.existsSync(path.join(root, 'yue2-latents')), false);
    assert.equal(fs.existsSync(tensor), false);
    assert.equal(fs.existsSync(path.join(root, 'labels', 'edited.json')), true);
    assert.equal(fs.existsSync(path.join(source, 'track.wav')), true);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(tensor, { recursive: true, force: true });
    fs.rmSync(source, { recursive: true, force: true });
  }
});

test('prepared-data reset rejects a source directory inside a cache', () => {
  const slug = `reset-guard-${process.pid}-${Date.now()}`;
  const root = datasetDir(slug);
  const source = path.join(root, 'yue2-latents', 'source');
  try {
    fs.mkdirSync(source, { recursive: true });
    fs.writeFileSync(path.join(source, 'track.wav'), 'source');
    assert.throws(() => clearPreparedCaches(slug, source), /source files/);
    assert.equal(fs.existsSync(path.join(source, 'track.wav')), true);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
