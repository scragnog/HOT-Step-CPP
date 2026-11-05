import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { applyFixedRecipe, relayJsonLine, buildYue2JointTrainArgs, planPlannerGateHoldout, resolvePlannerGateItems, yue2GateSongIds, yue2PlannerGatePreview, legacyYue2DoraError, parseYue2JointEvent, parseYue2JointStopMode, reconKeepDecision, updateYue2JointParams, yue2JointExpectedStep, yue2JointLivePace } from './yue2JointTrainRunner.js';
import { createJob } from './labelingQueue.js';

test('Fixed recipe forces its recipe and retains the companion decoder', () => {
  const recipe = applyFixedRecipe({ method: 'fixed-recipe', steps: 2000, seed: 123, saveEvery: 100, rank: 16, alpha: 256, adapterType: 'lokr',
    optimizer: 'prodigy', gradAccum: 8, arCropFrames: 500, narCropFrames: 0,
    lrSchedule: 'wsd', lyricTiming: true, cursorWeight: 0.08, klWeight: 0,
    abcDropout: 1, captionDropout: 0, textDropout: 0.1, lyricDropout: 0.1, bothDropout: 0.1 });
  assert.equal(recipe.rank, 64);
  assert.equal(recipe.alpha, 64);
  assert.equal(recipe.adapterType, 'lora');
  assert.equal(recipe.optimizer, 'adamw-lm');
  assert.equal(recipe.gradAccum, 1);
  assert.equal(recipe.arCropFrames, 0);
  assert.equal(recipe.narCropFrames, 1500);
  assert.equal(recipe.lrSchedule, 'constant');
  assert.equal(recipe.lyricTiming, false);
  assert.equal(recipe.cursorWeight, 0);
  assert.equal(recipe.klWeight, 0.2);
  assert.equal(recipe.weightDecay, 0.01);
  assert.equal(recipe.beta1, 0.9);
  assert.equal(recipe.beta2, 0.999);
  assert.equal(recipe.lr, 1e-4);
  assert.equal(recipe.plannerLrScale, 0.5);
  assert.equal(recipe.abcDropout, 0.5);
  assert.equal(recipe.captionDropout, 0.5);
  assert.deepEqual([recipe.textDropout, recipe.lyricDropout, recipe.bothDropout], [0, 0, 0]);
  assert.deepEqual([recipe.steps, recipe.seed, recipe.saveEvery], [1500, 42, 200]);
  assert.equal(recipe.arTargets, 'tuned');
  assert.equal(recipe.base, 'bf16');
  assert.equal(applyFixedRecipe({ method: 'fixed-recipe', base: 'Q4_K_M' }).base, 'Q4_K_M');
  const base = { checkpoint: 'base.gguf', dataset: 'dataset.json', outDir: 'run',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', companion: 'nar-v9.safetensors' };
  const parity = buildYue2JointTrainArgs({ ...base, ...recipe, method: 'fixed-recipe' });
  assert.deepEqual(parity.slice(parity.indexOf('--abc-dropout'), parity.indexOf('--abc-dropout') + 4),
    ['--abc-dropout', '0.5', '--caption-dropout', '0.5']);
  for (const flag of ['--text-dropout', '--lyric-dropout', '--both-dropout', '--trigger']) assert.equal(parity.includes(flag), false);
  assert.equal(parity.includes('--nar-drift'), false);
  assert.deepEqual(parity.slice(parity.indexOf('--adam-eps'), parity.indexOf('--adam-eps') + 2), ['--adam-eps', '1e-6']);
  assert.deepEqual(parity.slice(-2), ['--companion', 'nar-v9.safetensors']);
  const oldArgs = ['yue2-joint-train', '--checkpoint', 'base.gguf', '--dataset', 'dataset.json',
    '--output', 'run', '--steps', '100', '--save-every', '10', '--seed', '3', '--device', 'CUDA0',
    '--nar-drift', '--companion', 'nar-v9.safetensors'];
  assert.deepEqual(buildYue2JointTrainArgs({ ...base, method: 'base-matched' }), oldArgs);
  assert.deepEqual(buildYue2JointTrainArgs({ ...base }), oldArgs);
});

test('joint stop mode accepts the explicit step count sent by the UI', () => {
  assert.equal(parseYue2JointStopMode('steps'), 'steps');
  assert.equal(parseYue2JointStopMode(undefined), 'steps');
  assert.equal(parseYue2JointStopMode('loss'), 'loss');
  assert.equal(parseYue2JointStopMode('other'), null);
});

test('AITK native joint JSON event maps to the server metric contract', () => {
  assert.deepEqual(parseYue2JointEvent(
    '{"stage":"joint","step":7,"ar_ce":1.2,"ar_kl":0.3,"nar_mse":0.004,"gradient_norm":2.5}', 100),
    { stage: 'joint', step: 7, loss: 1.264, arCe: 1.2, arKl: 0.3, narMse: 0.004, gradNorm: 2.5, totalSteps: 100 });
});

test('AITK native joint JSON forwards the measured step duration', () => {
  assert.equal(parseYue2JointEvent(
    '{"stage":"joint","step":3,"ar_ce":1,"ar_kl":0,"nar_mse":0,"gradient_norm":1,"step_ms":42.5}', 10)?.stepMs, 42.5);
});

test('AITK joint metric includes weighted cursor loss when present', () => {
  const event = parseYue2JointEvent('{"stage":"joint","step":2,"ar_ce":1,"ar_kl":2,"nar_mse":3,"cursor_ce":4,"cursor_weight":0.08}', 10);
  assert.ok(Math.abs((event?.loss ?? 0) - 4.72) < 1e-12);
});

test('AITK CLI contract preserves checkpoint, dataset, output and resume order', () => {
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'base.gguf', dataset: 'dataset.json', outDir: 'run-new',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', resume: 'optimizer.resume',
  }), [
    'yue2-joint-train', '--checkpoint', 'base.gguf', '--dataset', 'dataset.json',
    '--output', 'run-new', '--steps', '100', '--save-every', '10', '--seed', '3',
    '--device', 'CUDA0', '--nar-drift', '--resume', 'optimizer.resume',
  ]);
});

test('pause resume CLI stays on the absolute step axis and preview cap is synth-only', () => {
  const args = buildYue2JointTrainArgs({ checkpoint: 'b', dataset: 'd', outDir: 'segment-2',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', resume: 'segment-1/checkpoint-step50/optimizer.resume',
    pauseAt: 75, preview: { enabled: true, everySteps: 25, seconds: 40, seed: 424242,
      previewMaxFrames: 1000, baseline: false, control: false } });
  assert.deepEqual(args.slice(-4), ['--resume', 'segment-1/checkpoint-step50/optimizer.resume', '--pause-at', '75']);
  assert.equal(args.includes('--preview-max-frames'), false);
});

test('decoder steps after KL ride only on a KL stop', () => {
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 750, saveEvery: 25, seed: 42, device: 'CUDA0', narExtraSteps: 100 };
  const kl = buildYue2JointTrainArgs({ ...base, stopMode: 'kl', targetKl: 1.1 });
  assert.deepEqual(kl.slice(kl.indexOf('--nar-extra-steps'), kl.indexOf('--nar-extra-steps') + 2), ['--nar-extra-steps', '100']);
  assert.equal(buildYue2JointTrainArgs({ ...base, stopMode: 'steps' }).includes('--nar-extra-steps'), false);
  assert.equal(buildYue2JointTrainArgs({ ...base, stopMode: 'kl', targetKl: 1.1, narExtraSteps: 0 }).includes('--nar-extra-steps'), false);
});

test('spike guard flags ride only when the factor is on', () => {
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 700, saveEvery: 50, seed: 42, device: 'CUDA0' };
  const on = buildYue2JointTrainArgs({ ...base, spikeFactor: 5, spikeStop: 3, spikeStopWindow: 20 });
  assert.deepEqual(on.slice(on.indexOf('--spike-factor'), on.indexOf('--spike-factor') + 6),
    ['--spike-factor', '5', '--spike-stop', '3', '--spike-stop-window', '20']);
  assert.equal(buildYue2JointTrainArgs({ ...base, spikeFactor: 0, spikeStop: 3 }).includes('--spike-stop'), false);
});

test('planner crop rides only when set', () => {
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 100, saveEvery: 10, seed: 42, device: 'CUDA0' };
  const on = buildYue2JointTrainArgs({ ...base, arCropFrames: 3000 });
  assert.deepEqual(on.slice(on.indexOf('--ar-crop-frames'), on.indexOf('--ar-crop-frames') + 2), ['--ar-crop-frames', '3000']);
  assert.equal(buildYue2JointTrainArgs({ ...base, arCropFrames: 0 }).includes('--ar-crop-frames'), false);
  assert.equal(buildYue2JointTrainArgs(base).includes('--ar-crop-frames'), false);
});

test('style norms ride as one JSON argument only when known', () => {
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 100, saveEvery: 10, seed: 42, device: 'CUDA0' };
  const json = JSON.stringify({ exempt: ['bars'], sheets: 12 });
  const on = buildYue2JointTrainArgs({ ...base, styleNorms: json });
  assert.deepEqual(on.slice(on.indexOf('--style-norms'), on.indexOf('--style-norms') + 2), ['--style-norms', json]);
  assert.equal(buildYue2JointTrainArgs(base).includes('--style-norms'), false);
});

test('paused JSON event is recognized without becoming terminal done', () => {
  assert.deepEqual(parseYue2JointEvent('{"stage":"paused","step":50,"resume":"optimizer.resume"}', 100),
    { stage: 'paused', step: 50, totalSteps: 100 });
});

test('target-loss event carries its step for the post-exit validator', () => {
  const ev = parseYue2JointEvent('{"stage":"target","step":212,"ar_ce":0.4,"ar_kl":0.1,"nar_mse":0.01}', 400);
  assert.equal(ev?.stage, 'target');
  assert.equal(ev?.step, 212);
  assert.equal(ev?.totalSteps, 400);
  assert.ok(ev && Math.abs((ev.loss ?? 0) - 0.43) < 1e-12);
});

test('prodigy run carries optimizer, rank/alpha and target-loss flags', () => {
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'base.gguf', dataset: 'dataset.json', outDir: 'run-new',
    steps: 400, saveEvery: 50, seed: 3, device: 'CUDA0',
    optimizer: 'prodigy', prodigyD0: 1e-6, rank: 16, alpha: 32,
    stopMode: 'loss', targetLoss: 0.9,
  }), [
    'yue2-joint-train', '--checkpoint', 'base.gguf', '--dataset', 'dataset.json',
    '--output', 'run-new', '--steps', '400', '--save-every', '50', '--seed', '3',
    '--device', 'CUDA0', '--nar-drift', '--rank', '16', '--alpha', '32',
    '--optimizer', 'prodigy', '--prodigy-d0', '0.000001', '--target-loss', '0.9',
  ]);
});

test('recon-keep decision keeps the first checkpoint and every real improvement, drops a flat one', () => {
  const first = reconKeepDecision(undefined, 0.05, 0.01);
  assert.deepEqual(first, { keep: true, bestRecon: 0.05 });
  const improved = reconKeepDecision(0.05, 0.03, 0.01);
  assert.deepEqual(improved, { keep: true, bestRecon: 0.03 });
  const flat = reconKeepDecision(0.03, 0.029, 0.01);
  assert.deepEqual(flat, { keep: false, bestRecon: 0.03 });
  const wellPastDelta = reconKeepDecision(0.03, 0.018, 0.01);
  assert.deepEqual(wellPastDelta, { keep: true, bestRecon: 0.018 });
});

test('muon run carries lr-scale and Newton-Schulz step count, adamw adds no optimizer flag', () => {
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'b', dataset: 'd', outDir: 'run-new',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0',
    optimizer: 'muon', muonLrScale: 1.2, muonNsSteps: 7,
  }), [
    'yue2-joint-train', '--checkpoint', 'b', '--dataset', 'd', '--output', 'run-new',
    '--steps', '100', '--save-every', '10', '--seed', '3', '--device', 'CUDA0', '--nar-drift',
    '--optimizer', 'muon', '--muon-lr-scale', '1.2', '--muon-ns-steps', '7',
  ]);
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'b', dataset: 'd', outDir: 'run-new',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', optimizer: 'adamw',
  }), [
    'yue2-joint-train', '--checkpoint', 'b', '--dataset', 'd', '--output', 'run-new',
    '--steps', '100', '--save-every', '10', '--seed', '3', '--device', 'CUDA0', '--nar-drift',
  ]);
});

test('wsd overshoot margin rides only with the wsd schedule', () => {
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 500, saveEvery: 25, seed: 42, device: 'CUDA0', stopMode: 'kl' as const, targetKl: 1 };
  const wsd = buildYue2JointTrainArgs({ ...base, lrSchedule: 'wsd', lrDecaySteps: 40, klOvershootMargin: 0.15 });
  assert.deepEqual(wsd.slice(wsd.indexOf('--lr-decay-steps'), wsd.indexOf('--lr-decay-steps') + 4), ['--lr-decay-steps', '40', '--kl-overshoot-margin', '0.15']);
  assert.equal(buildYue2JointTrainArgs({ ...base, lrSchedule: 'constant', klOvershootMargin: 0.15 }).includes('--kl-overshoot-margin'), false);
});

test('DoRA reaches the joint trainer as --adapter-type dora; legacy endpoints refuse it', () => {
  const base = { checkpoint: 'base.gguf', dataset: 'dataset.json', outDir: 'run',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', companion: 'nar-v9.safetensors' };
  const args = buildYue2JointTrainArgs({ ...base, adapterType: 'dora' });
  assert.equal(args[args.indexOf('--adapter-type') + 1], 'dora');
  assert.equal(args.includes('--lokr-dim'), false);
  assert.equal(buildYue2JointTrainArgs({ ...base, adapterType: 'lora' }).includes('--adapter-type'), false);
  assert.match(legacyYue2DoraError({ adapterType: 'dora' }) ?? '', /only available in joint/);
  assert.match(legacyYue2DoraError({ dora: true }) ?? '', /only available in joint/);
  assert.equal(legacyYue2DoraError({ adapterType: 'lora', dora: false }), null);
});

test('live interval edit: validates, re-paces the preview pause and writes .params.json for the trainer', () => {
  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-params-'));
  try {
    const opts = { checkpoint: 'b', dataset: 'd', outDir, steps: 1000, saveEvery: 100, seed: 1, device: 'CUDA0' };
    const job = { id: 'live-1', kind: 'yue2-joint-train', status: 'running', opts, events: [], listeners: new Set() } as any;
    assert.equal((updateYue2JointParams(job, { saveEvery: 50 }) as any).status, 409, 'no live pace yet');
    const preview = { enabled: true, everySteps: 200 } as any;
    yue2JointLivePace.set(job.id, { preview, start: 400, pauseAt: 600, previewPaced: true, rev: 0 });
    assert.equal((updateYue2JointParams(job, { saveEvery: 0 }) as any).status, 400);
    assert.equal((updateYue2JointParams(job, { saveEvery: 1001 }) as any).status, 400);
    assert.equal((updateYue2JointParams(job, { previewEvery: 2.5 }) as any).status, 400);
    assert.equal((updateYue2JointParams(job, {}) as any).status, 400);
    assert.deepEqual(updateYue2JointParams(job, { saveEvery: 25, previewEvery: 50 }), { saveEvery: 25, previewEvery: 50 });
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(outDir, '.params.json'), 'utf8')), { rev: 1, save_every: 25, pause_at: 450 });
    assert.equal(opts.saveEvery, 25);
    assert.equal(preview.everySteps, 50);
    // A pause past the end clamps to the last step; a run not paced by previews refuses previewEvery.
    updateYue2JointParams(job, { previewEvery: 900 });
    assert.equal(JSON.parse(fs.readFileSync(path.join(outDir, '.params.json'), 'utf8')).pause_at, 1000);
    yue2JointLivePace.set(job.id, { start: 0, pauseAt: 0, previewPaced: false, rev: 7 });
    assert.equal((updateYue2JointParams(job, { previewEvery: 10 }) as any).status, 400);
    updateYue2JointParams(job, { saveEvery: 75 });
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(outDir, '.params.json'), 'utf8')), { rev: 8, save_every: 75 });
    assert.deepEqual(buildYue2JointTrainArgs({ ...opts, paramsFile: 'p.json', paramsRev: 8 }).slice(-4), ['--params-file', 'p.json', '--params-rev', '8']);
  } finally {
    yue2JointLivePace.delete('live-1');
    fs.rmSync(outDir, { recursive: true, force: true });
  }
});

test('expected checkpoint follows the engine, never a pause a late edit moved', () => {
  // A shorter preview edit landing during the final save: no paused event, so the cap.
  assert.equal(yue2JointExpectedStep({ lastStep: 1500, targetStopped: false }, 1200, 1500), 1500);
  assert.equal(yue2JointExpectedStep({ pausedAt: 1212, lastStep: 1212, targetStopped: false }, 1200, 1500), 1212);
  assert.equal(yue2JointExpectedStep({ lastStep: 1340, targetStopped: true }, 1200, 1500), 1340);
});

test('planner gate flags ride on every segment, with the decoder budget on any stop mode', () => {
  const gate = { every: 10, minEpochs: 5, patience: 3, epsilon: 0.001, evalSeed: 42, holdoutIds: ['a', 'c'], holdoutItems: [0, 2] };
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 750, saveEvery: 25, seed: 42, device: 'CUDA0', narExtraSteps: 100, stopMode: 'steps' as const };
  const args = buildYue2JointTrainArgs({ ...base, plannerGate: gate });
  const at = (flag: string) => args[args.indexOf(flag) + 1];
  assert.equal(at('--nar-extra-steps'), '100');
  assert.equal(at('--planner-gate-every'), '10');
  assert.equal(at('--planner-gate-min-epochs'), '5');
  assert.equal(at('--planner-gate-patience'), '3');
  assert.equal(at('--planner-gate-epsilon'), '0.001');
  assert.equal(at('--holdout-items'), '0,2');
  assert.equal(at('--eval-seed'), '42');
  // A resumed or paused segment carries the same flags.
  const resumed = buildYue2JointTrainArgs({ ...base, plannerGate: gate, resume: 'r', pauseAt: 100 });
  assert.deepEqual(resumed.slice(resumed.indexOf('--planner-gate-every'), resumed.indexOf('--planner-gate-every') + 8), args.slice(args.indexOf('--planner-gate-every'), args.indexOf('--planner-gate-every') + 8));
  // Off: no gate flags, and steps mode still passes no budget.
  const off = buildYue2JointTrainArgs(base);
  assert.equal(off.some(a => a.startsWith('--planner-gate') || a === '--holdout-items' || a === '--eval-seed'), false);
  assert.equal(off.includes('--nar-extra-steps'), false);
  // An unresolved split must never be dropped silently.
  assert.throws(() => buildYue2JointTrainArgs({ ...base, plannerGate: { ...gate, holdoutItems: undefined } }), /not resolved/);
});

test('planner gate hold-out: 20% rounded up, at least 3, deterministic, leaves training songs', () => {
  const ids = (n: number) => Array.from({ length: n }, (_, i) => `song-${String(i).padStart(2, '0')}`);
  const pick = (n: number, seed = 42) => planPlannerGateHoldout(ids(n), seed) as { holdoutIds: string[] };
  assert.equal(pick(10).holdoutIds.length, 3);
  assert.equal(pick(16).holdoutIds.length, 4);
  assert.equal(pick(21).holdoutIds.length, 5);
  assert.deepEqual(pick(16).holdoutIds, pick(16).holdoutIds);
  assert.notDeepEqual(pick(16, 1).holdoutIds, pick(16, 2).holdoutIds);
  assert.ok(pick(16).holdoutIds.every(id => ids(16).includes(id)));
  assert.match((planPlannerGateHoldout(ids(3), 42) as { error: string }).error, /at least one more/);
  assert.ok('holdoutIds' in planPlannerGateHoldout(ids(4), 42));
});

test('planner gate items resolve by id and a resume refuses a moved split', () => {
  const gate = { every: 10, minEpochs: 5, patience: 3, epsilon: 0.001, evalSeed: 42, holdoutIds: ['c', 'a'] };
  assert.deepEqual(resolvePlannerGateItems(['a', 'b', 'c', 'd'], gate), [2, 0]);
  assert.match((resolvePlannerGateItems(['a', 'b'], gate) as { error: string }).error, /not in the prepared dataset: c/);
  assert.deepEqual(resolvePlannerGateItems(['a', 'b', 'c', 'd'], { ...gate, holdoutItems: [2, 0] }), [2, 0]);
  assert.match((resolvePlannerGateItems(['x', 'a', 'b', 'c'], { ...gate, holdoutItems: [2, 0] }) as { error: string }).error, /no longer matches/);
  assert.match((resolvePlannerGateItems(['a', 'c'], gate) as { error: string }).error, /no song to train on/);
});

test('planner gate reading is a metric and leaves the phase alone; the freeze reason picks the log text', () => {
  const job = createJob('yue2-joint-train', 'fixture', [], {});
  job.phase = 'training';
  const state = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: 100 };
  relayJsonLine(job, JSON.stringify({ stage: 'planner_gate', step: 30, epoch: 6.5, ce: 2.31, best: 2.25, best_step: 20, stale: 2, patience: 3, armed: true, items: 4, tokens: 900, ms: 4300 }), state);
  assert.equal(job.phase, 'training');
  const metric = job.events.find(e => e.type === 'metric' && e.metric === 'eval') as any;
  assert.deepEqual(metric.plannerGate, { ce: 2.31, best: 2.25, bestStep: 20, stale: 2, patience: 3, armed: true, epoch: 6.5, items: 4 });
  const messages = () => job.events.filter(e => e.type === 'log').map((e: any) => e.message);
  relayJsonLine(job, JSON.stringify({ stage: 'planner_frozen', step: 40, reason: 'planner_ce' }), state);
  assert.equal((state as any).frozenAt, 40);
  assert.ok(messages().some(m => /held-out loss stopped improving at step 40/.test(m)));
  assert.ok(!messages().some(m => /KL target/.test(m)));
  relayJsonLine(job, JSON.stringify({ stage: 'planner_frozen', step: 50, reason: 'kl' }), state);
  assert.ok(messages().some(m => /reached its KL target at step 50/.test(m)));
});

test('planner gate preview works before any stage ran, from the current audio files alone', () => {
  const sample = (relPath: string, extra: Record<string, unknown> = {}) => ({ audioPath: `/data/set/${relPath}`, relPath, excluded: false, fileMissing: false, ...extra });
  const samples = [
    ...Array.from({ length: 8 }, (_, i) => sample(`song-${i}.wav`)),
    sample('a/same.flac'), sample('b/same.flac'),            // colliding basenames get flattened names
    sample('skipped.wav', { excluded: true }), sample('gone.wav', { fileMissing: true }), sample('notes.txt'),
    sample('song-0.engine.wav'),                              // the old conversion cache the engine skips
  ];
  const ids = yue2GateSongIds(samples);
  assert.deepEqual(ids.slice(0, 8), Array.from({ length: 8 }, (_, i) => `song-${i}.wav`));
  assert.deepEqual(ids.slice(8), ['a__same.flac', 'b__same.flac']);
  const preview = yue2PlannerGatePreview(samples, 42);
  assert.equal(preview.error, undefined);
  assert.equal(preview.songs, 10);
  assert.equal(preview.holdoutIds.length, 3);
  assert.equal(preview.training, 7);
  assert.ok(preview.holdoutIds.every(id => ids.includes(id)));
  // A changed inventory changes the answer: nothing is cached.
  assert.equal(yue2PlannerGatePreview([...samples, sample('new.wav')], 42).songs, 11);
  assert.match(yue2PlannerGatePreview([], 42).error ?? '', /no included audio/);
  assert.match(yue2PlannerGatePreview(samples.slice(0, 3), 42).error ?? '', /at least one more/);
});
