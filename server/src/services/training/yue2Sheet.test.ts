import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { runYue2CoverSheetJob } from './yue2ArTrainRunner.js';
import { createJob, type TrainingJob } from './labelingQueue.js';
import { gpuLaneOwner, runOnGpuLane } from '../generation/gpuLane.js';
import { type RelayState, type Yue2Kind, runYue2AceTrain } from './yue2TrainRunner.js';
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

function coverRunFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cover-fallback-'));
  const audio = path.join(root, 'source.wav');
  const jobDir = path.join(root, 'job');
  fs.writeFileSync(audio, 'audio fixture');
  const job = createJob('yue2-sheet', 'cover:test', [], {});
  const writeAbc = () => fs.writeFileSync(path.join(jobDir, 'cover-sheet.json'),
    JSON.stringify({ sources: [{ name: 'source.wav', source: audio, abc: 'X:1\nK:C\nC|' }] }));
  return { root, audio, jobDir, job, writeAbc };
}

type CoverRun = typeof runYue2AceTrain;
function mockCoverRun(action: (call: number, stopEngine: boolean, args: string[],
  onLine: (line: string, st: RelayState) => void, st: RelayState) => Promise<void>): CoverRun {
  let calls = 0;
  return async <S extends RelayState>(
    _job: TrainingJob, _kind: Yue2Kind, args: string[], _idleMs: number,
    _verify: () => string | null, onLine: (line: string, st: S) => void,
    st: S, _env?: NodeJS.ProcessEnv, stopEngine = true,
  ): Promise<S> => {
    await action(++calls, stopEngine, args, onLine as (line: string, st: RelayState) => void, st);
    return st;
  };
}

test('successful cover transcription keeps the engine and never takes the generation lane', async () => {
  const f = coverRunFixture();
  try {
    let calls = 0;
    const run = mockCoverRun(async (_call, stopEngine, args) => {
      calls++;
      assert.equal(stopEngine, false);
      assert.equal(args.includes('--melody-only'), false);
      assert.equal(gpuLaneOwner(), null);
      f.writeAbc();
    });
    assert.equal(await runYue2CoverSheetJob(f.job, f.audio, f.jobDir,
      { missingModels: () => [], run }), 'X:1\nK:C\nC|');
    assert.equal(calls, 1);
  } finally { fs.rmSync(f.root, { recursive: true, force: true }); }
});

test('only a SheetSage2 load failure retries with the engine stopped inside the generation lane', async () => {
  const f = coverRunFixture();
  try {
    const stops: boolean[] = [];
    const run = mockCoverRun(async (call, stopEngine, args, onLine, st) => {
      stops.push(stopEngine);
      if (call === 1) {
        onLine('[yue2-sheet] SheetSage2 load failed (model): backend buffer allocation failed', st);
        throw new Error(st.fatalMessage);
      }
      assert.equal(gpuLaneOwner()?.family, 'yue2');
      assert.equal(args.includes('--force'), false);
      f.writeAbc();
    });
    assert.equal(await runYue2CoverSheetJob(f.job, f.audio, f.jobDir,
      { missingModels: () => [], run }), 'X:1\nK:C\nC|');
    assert.deepEqual(stops, [false, true]);
    assert.equal(gpuLaneOwner(), null);
  } finally { fs.rmSync(f.root, { recursive: true, force: true }); }
});

test('other cover failures neither retry nor stop the engine', async () => {
  const f = coverRunFixture();
  try {
    let calls = 0;
    const run = mockCoverRun(async (_call, stopEngine, _args, onLine, st) => {
      calls++;
      assert.equal(stopEngine, false);
      onLine('[yue2-sheet] cannot decode source', st);
      throw new Error('cannot decode source');
    });
    await assert.rejects(runYue2CoverSheetJob(f.job, f.audio, f.jobDir,
      { missingModels: () => [], run }), /cannot decode source/);
    assert.equal(calls, 1);
    assert.equal(gpuLaneOwner(), null);
  } finally { fs.rmSync(f.root, { recursive: true, force: true }); }
});

for (const retryThrows of [false, true]) {
  test(`a render queued during fallback waits for engine restart (${retryThrows ? 'retry throws' : 'retry succeeds'})`, async () => {
    const f = coverRunFixture();
    try {
      let enterRetry!: () => void;
      const retryEntered = new Promise<void>(resolve => { enterRetry = resolve; });
      let completeRestart!: () => void;
      const restart = new Promise<void>(resolve => { completeRestart = resolve; });
      let engineStopped = false;
      const run = mockCoverRun(async (call, stopEngine, _args, onLine, st) => {
        if (call === 1) {
          assert.equal(stopEngine, false);
          onLine('[yue2-sheet] SheetSage2 load failed (model): backend buffer allocation failed', st);
          throw new Error(st.fatalMessage);
        }
        assert.equal(stopEngine, true);
        engineStopped = true;
        enterRetry();
        await restart; // Models runYue2AceTrain resolving only after its restart finally.
        engineStopped = false;
        if (retryThrows) throw new Error('retry failed');
        f.writeAbc();
      });
      const cover = runYue2CoverSheetJob(f.job, f.audio, f.jobDir,
        { missingModels: () => [], run });
      await retryEntered;
      let rendered = false;
      const render = runOnGpuLane(async () => {
        assert.equal(engineStopped, false);
        rendered = true;
      }, { family: 'yue2', label: 'queued render' });
      await Promise.resolve();
      assert.equal(rendered, false);
      completeRestart();
      if (retryThrows) await assert.rejects(cover, /retry failed/);
      else assert.equal(await cover, 'X:1\nK:C\nC|');
      await render;
      assert.equal(rendered, true);
    } finally { fs.rmSync(f.root, { recursive: true, force: true }); }
  });
}

test('development fallback hook reruns a completed cover with force', async () => {
  const f = coverRunFixture();
  const oldDev = process.env.HOT_STEP_DEV;
  const oldForce = process.env.HOTSTEP_YUE2_COVER_FORCE_FALLBACK;
  try {
    process.env.HOT_STEP_DEV = '1';
    process.env.HOTSTEP_YUE2_COVER_FORCE_FALLBACK = '1';
    const stops: boolean[] = [];
    const run = mockCoverRun(async (_call, stopEngine, args) => {
      stops.push(stopEngine);
      assert.equal(args.includes('--force'), stopEngine);
      if (!stopEngine) f.writeAbc();
    });
    assert.equal(await runYue2CoverSheetJob(f.job, f.audio, f.jobDir,
      { missingModels: () => [], run }), 'X:1\nK:C\nC|');
    assert.deepEqual(stops, [false, true]);
  } finally {
    if (oldDev === undefined) delete process.env.HOT_STEP_DEV; else process.env.HOT_STEP_DEV = oldDev;
    if (oldForce === undefined) delete process.env.HOTSTEP_YUE2_COVER_FORCE_FALLBACK;
    else process.env.HOTSTEP_YUE2_COVER_FORCE_FALLBACK = oldForce;
    fs.rmSync(f.root, { recursive: true, force: true });
  }
});
