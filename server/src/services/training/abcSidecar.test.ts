import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { abcSidecarPath, readAbcSidecar, writeAbcSidecar } from './abcSidecar.js';
import { seedAbcManifestFromSidecars, saveAbcManifestSidecars, buildYue2SheetArgs } from './yue2Sheet.js';
import { runYue2CoverSheetJob, runYue2SheetJob } from './yue2ArTrainRunner.js';
import { createJob } from './labelingQueue.js';
import { runYue2AceTrain } from './yue2TrainRunner.js';

test('lead sheets survive manifest reset and seed the next stage by source name', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'abc-sidecar-'));
  try {
    const audio = path.join(dir, 'track.wav');
    const manifest = path.join(dir, 'preprocess.json');
    fs.writeFileSync(audio, 'audio');
    assert.equal(writeAbcSidecar(audio, 'X:1\nK:C\nC'), true);
    assert.equal(path.basename(abcSidecarPath(audio)), 'track.abc');
    fs.writeFileSync(manifest, JSON.stringify({ sources: [{ name: 'track.wav', source: audio }] }));
    assert.equal(seedAbcManifestFromSidecars(manifest), 1);
    assert.equal(JSON.parse(fs.readFileSync(manifest, 'utf8')).sources[0].abc, 'X:1\nK:C\nC');
    assert.equal(buildYue2SheetArgs({ manifest, only: '', force: false, fast: false, datasetSlug: '' }).includes('--force'), false);
    fs.rmSync(manifest);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:C\nC');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('successful force result replaces the sidecar, while empty and error rows do not', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'abc-sidecar-'));
  try {
    const audio = path.join(dir, 'track.wav');
    const manifest = path.join(dir, 'preprocess.json');
    fs.writeFileSync(audio, 'audio');
    writeAbcSidecar(audio, 'X:1\nK:C\nC');
    const row = { name: 'track.wav', source: audio, abc: '', abc_error: 'failed' };
    fs.writeFileSync(manifest, JSON.stringify({ sources: [row] }));
    assert.equal(saveAbcManifestSidecars(manifest), 0);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:C\nC');
    row.abc = 'X:1\nK:D\nD';
    fs.writeFileSync(manifest, JSON.stringify({ sources: [row] }));
    assert.equal(saveAbcManifestSidecars(manifest), 0);
    delete (row as { abc_error?: string }).abc_error;
    fs.writeFileSync(manifest, JSON.stringify({ sources: [row] }));
    assert.equal(buildYue2SheetArgs({ manifest, only: '', force: true, fast: false, datasetSlug: '' }).includes('--force'), true);
    assert.equal(saveAbcManifestSidecars(manifest), 1);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:D\nD');
    assert.equal(writeAbcSidecar(audio, '  '), false);
    assert.equal(writeAbcSidecar(audio, 'X:1\nV: Vocal\nZ4|z4|\nV: Ins\nC4|'), false);
    assert.equal(readAbcSidecar(audio), 'X:1\nK:D\nD');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('cover transcription fails for all-rest Vocal bars and accepts a sounding bar', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cover-sheet-job-'));
  try {
    const audio = path.join(dir, 'track.wav');
    fs.writeFileSync(audio, 'audio');
    let abc = 'X:1\nV: Vocal\nZ4|z4|\nV: Ins\nC4|';
    const run = (async (_job, _kind, _args, _idleMs, verify) => {
      const manifest = path.join(dir, 'cover-sheet.json');
      const data = JSON.parse(fs.readFileSync(manifest, 'utf8'));
      data.sources[0].abc = abc;
      fs.writeFileSync(manifest, JSON.stringify(data));
      if (verify?.()) throw new Error(verify()!);
    }) as typeof runYue2AceTrain;
    const deps = { missingModels: () => [], run };
    const silent = createJob('yue2-sheet', 'fixture', [], {});
    await assert.rejects(runYue2CoverSheetJob(silent, audio, dir, deps), /The transcriber heard no melody in this source/);
    assert.equal(silent.status, 'failed');
    assert.equal(fs.existsSync(abcSidecarPath(audio)), false);
    abc = 'X:1\nV: Vocal\nz4|C4|\nV: Ins\nC4|';
    const sounding = createJob('yue2-sheet', 'fixture', [], {});
    assert.equal(await runYue2CoverSheetJob(sounding, audio, dir, deps), abc);
    assert.equal(sounding.status, 'done');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('dataset sheet job seeds saved rows, writes new scores and replaces them on force', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'abc-sheet-job-'));
  try {
    const first = path.join(dir, 'first.wav');
    const second = path.join(dir, 'second.wav');
    const manifest = path.join(dir, 'preprocess.json');
    fs.writeFileSync(first, 'audio 1'); fs.writeFileSync(second, 'audio 2');
    writeAbcSidecar(first, 'X:1\nK:C\nC');
    const sources = [{ name: 'first.wav', source: first }, { name: 'second.wav', source: second }];
    fs.writeFileSync(manifest, JSON.stringify({ sources }));
    let force = false;
    const run = (async (_job, _kind, args, _idleMs, _verify, _onLine, state) => {
      assert.equal(args.includes('--force'), force);
      const data = JSON.parse(fs.readFileSync(manifest, 'utf8'));
      assert.equal(data.sources[0].abc, 'X:1\nK:C\nC');
      data.sources[0].abc = force ? 'X:1\nK:D\nD' : data.sources[0].abc;
      data.sources[1].abc = 'X:1\nK:G\nG';
      fs.writeFileSync(manifest, JSON.stringify(data));
      return state;
    }) as typeof runYue2AceTrain;
    const deps = { missingModels: () => [], run };
    const job = createJob('yue2-sheet', 'fixture', [], { manifest, only: '', force: false, fast: false, datasetSlug: '' });
    await runYue2SheetJob(job, deps);
    assert.equal(job.status, 'done');
    assert.equal(readAbcSidecar(first), 'X:1\nK:C\nC');
    assert.equal(readAbcSidecar(second), 'X:1\nK:G\nG');
    force = true;
    const retry = createJob('yue2-sheet', 'fixture', [], { manifest, only: '', force: true, fast: false, datasetSlug: '' });
    await runYue2SheetJob(retry, deps);
    assert.equal(retry.status, 'done');
    assert.equal(readAbcSidecar(first), 'X:1\nK:D\nD');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});
