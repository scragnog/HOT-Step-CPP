import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import { checkpointRecords, isCheckpointLadder, rungsOf, type Yue2AitkCheckpointRecord, type Yue2AitkRunRecord } from './yue2AitkRuns.js';
import { readYue2ArRun } from './yue2ArRuns.js';
import { readYue2Run } from './yue2Runs.js';
import { uniqueDatasetTrigger } from './datasetTrigger.js';

test('a taken trigger gains the dataset slug suffix; unique triggers stay unchanged', () => {
  assert.equal(uniqueDatasetTrigger('base', 'base_covers', new Set(['base'])), 'base_covers');
  assert.equal(uniqueDatasetTrigger('fresh', 'fresh_covers', new Set(['base'])), 'fresh');
  assert.equal(uniqueDatasetTrigger('base', 'base_covers', new Set(['base', 'base_covers'])), 'base_covers_2');
});

test('dataset creation assigns a distinct trigger and leaves a unique one alone', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'dataset-trigger-'));
  try {
    const script = [
      "import fs from 'node:fs'; import path from 'node:path'; import assert from 'node:assert/strict';",
      "import { initDb } from './src/db/database.js'; import { createDatasetFromFolder } from './src/services/training/datasetCreate.js';",
      "initDb(); const root = process.env.TRAINING_DIR;",
      "async function create(name, tag) { const sourceDir = path.join(root, name); fs.mkdirSync(sourceDir, { recursive: true }); fs.writeFileSync(path.join(sourceDir, 'track.wav'), 'audio'); return createDatasetFromFolder({ name, sourceDir, customTag: tag }); }",
      "assert.equal((await create('Base', 'base')).customTag, 'base');",
      "assert.equal((await create('Base Covers', 'base')).customTag, 'base_covers');",
      "assert.equal((await create('Solo', 'solo')).customTag, 'solo');",
    ].join('\n');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)),
      env: { ...process.env, DATA_DIR: path.join(root, 'data'), TRAINING_DIR: path.join(root, 'training') }, stdio: 'pipe',
    });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('AITK checkpoint discovery exposes combined and native AR/NAR outputs', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-runs-'));
  try {
    const step2 = path.join(root, 'checkpoint-step2');
    const step10 = path.join(root, 'checkpoint-step10');
    fs.mkdirSync(step2); fs.mkdirSync(step10);
    for (const name of ['adapter.safetensors', 'optimizer.resume', 'native-ar.safetensors', 'native-nar.safetensors']) {
      fs.writeFileSync(path.join(step10, name), 'x');
    }
    fs.writeFileSync(path.join(step2, 'native-ar.safetensors'), 'x');
    const rows = checkpointRecords(root);
    assert.deepEqual(rows.map(r => r.step), [10, 2]);
    assert.equal(rows[0].arPath, path.join(step10, 'native-ar.safetensors'));
    assert.equal(rows[0].narPath, path.join(step10, 'native-nar.safetensors'));
    assert.equal(rows[1].adapterPath, undefined);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('AITK checkpoint discovery finds trigger-named AR/NAR weights (#220)', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-runs-'));
  try {
    const step37 = path.join(root, 'checkpoint-step37');
    const step5 = path.join(root, 'checkpoint-step5');
    fs.mkdirSync(step37); fs.mkdirSync(step5);
    for (const name of ['my-band_yue2_joint_ar_step37.safetensors', 'my-band_yue2_joint_nar_step37.safetensors']) fs.writeFileSync(path.join(step37, name), 'x');
    fs.writeFileSync(path.join(step5, 'yue2_joint_nar_step5.safetensors'), 'x');
    const rows = checkpointRecords(root);
    assert.equal(rows[0].arPath, path.join(step37, 'my-band_yue2_joint_ar_step37.safetensors'));
    assert.equal(rows[0].narPath, path.join(step37, 'my-band_yue2_joint_nar_step37.safetensors'));
    assert.equal(rows[1].arPath, undefined, 'a NAR file is never taken for the AR half');
    assert.equal(rows[1].narPath, path.join(step5, 'yue2_joint_nar_step5.safetensors'));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('AITK checkpoints show the trailing 20-step mean from their own segment', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-loss-'));
  try {
    for (const [segment, step, arCe] of [[1, 50, 2], [2, 100, 3]] as const) {
      const dir = path.join(root, 'segments', `segment-${String(segment).padStart(6, '0')}`);
      fs.mkdirSync(path.join(dir, `checkpoint-step${step}`), { recursive: true });
      fs.writeFileSync(path.join(dir, 'train.jsonl'), [
        ...Array.from({ length: 20 }, (_, i) => JSON.stringify({
          stage: 'joint', step: step - 19 + i,
          ar_ce: i === 19 ? arCe + 20 : arCe,
          ar_kl: 0.5, nar_mse: 1, cursor_ce: 0.25, cursor_weight: 0.08,
        })),
        '{incomplete',
      ].join('\n'));
    }
    const rows = checkpointRecords(root);
    assert.deepEqual(rows.map(row => row.step), [100, 50]);
    assert.ok(Math.abs((rows[0].loss ?? 0) - 5.12) < 1e-10);
    assert.ok(Math.abs((rows[1].loss ?? 0) - 4.12) < 1e-10);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a checkpoint above the planner freeze carries the decoder loss, not the freeze-time value', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-frozen-'));
  try {
    // 20 planner steps (composite 1.4 + 0.2 * 1 + 0 = 1.6), then 20 frozen
    // steps that report only the decoder's flow loss.
    fs.writeFileSync(path.join(root, 'train.jsonl'), [
      ...Array.from({ length: 20 }, (_, i) => JSON.stringify({
        stage: 'joint', step: i + 1, ar_ce: 1.4, ar_kl: 1, nar_mse: 0,
      })),
      ...Array.from({ length: 20 }, (_, i) => JSON.stringify({
        stage: 'joint', step: i + 21, planner_frozen: true, nar_mse: 0.5,
      })),
    ].join('\n'));
    // No meters.json on purpose: this label must not depend on it.
    fs.mkdirSync(path.join(root, 'checkpoint-step20'));
    fs.mkdirSync(path.join(root, 'checkpoint-step40'));
    const rows = checkpointRecords(root);
    const at = (step: number) => rows.find(row => row.step === step)!;
    assert.ok(Math.abs((at(20).loss ?? 0) - 1.6) < 1e-10);
    assert.equal(at(20).decoderOnly, undefined);
    assert.ok(Math.abs((at(40).loss ?? 0) - 0.5) < 1e-10, 'the decoder phase must report its own mean');
    assert.equal(at(40).decoderOnly, true, 'and say it is not the composite');
    assert.notEqual(at(40).loss, at(20).loss, 'a post-freeze label must not repeat the freeze-time value');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a checkpoint window that crosses the freeze averages only the newest objective', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-crossing-'));
  try {
    // 20 planner steps at 1.6, then frozen steps at 0.5 and 0.7.
    fs.writeFileSync(path.join(root, 'train.jsonl'), [
      ...Array.from({ length: 20 }, (_, i) => JSON.stringify({
        stage: 'joint', step: i + 1, ar_ce: 1.4, ar_kl: 1, nar_mse: 0,
      })),
      JSON.stringify({ stage: 'joint', step: 21, planner_frozen: true, nar_mse: 0.5 }),
      JSON.stringify({ stage: 'joint', step: 22, planner_frozen: true, nar_mse: 0.7 }),
    ].join('\n'));
    fs.mkdirSync(path.join(root, 'checkpoint-step21'));
    fs.mkdirSync(path.join(root, 'checkpoint-step22'));
    const rows = checkpointRecords(root);
    const at = (step: number) => rows.find(row => row.step === step)!;
    assert.ok(Math.abs((at(21).loss ?? 0) - 0.5) < 1e-10, 'one frozen reading, not a blend with 1.6');
    assert.equal(at(21).decoderOnly, true);
    assert.ok(Math.abs((at(22).loss ?? 0) - 0.6) < 1e-10);
    assert.equal(at(22).decoderOnly, true);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('AR checkpoint mean includes a spike only as one of the last 20 logged steps', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ar-loss-'));
  try {
    fs.writeFileSync(path.join(root, 'yue2_ar_lora_step25.safetensors'), 'x');
    fs.writeFileSync(path.join(root, 'train-log.jsonl'), [
      ...Array.from({ length: 25 }, (_, i) => JSON.stringify({
        type: 'step', step: i + 1, loss: i === 24 ? 21 : 1,
      })),
      JSON.stringify({ type: 'milestone', step: 25, loss: 21 }),
      JSON.stringify({ type: 'eval', step: 25, mintedVal: 3.5 }),
    ].join('\n'));
    const run = readYue2ArRun(root);
    assert.equal(run?.checkpoints[0]?.loss, 2);
    assert.equal(run?.checkpoints[0]?.valLoss, 3.5);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('NAR checkpoint loss is the same trailing 20-step mean', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-nar-loss-'));
  try {
    fs.writeFileSync(path.join(root, 'yue2_nar_lora_step25.safetensors'), 'x');
    fs.writeFileSync(path.join(root, 'train-log.jsonl'), [
      ...Array.from({ length: 25 }, (_, i) => JSON.stringify({
        type: 'step', step: i + 1, loss: i === 24 ? 21 : 1,
      })),
      JSON.stringify({ type: 'milestone', step: 25, loss: 21 }),
    ].join('\n'));
    const run = readYue2Run(root);
    assert.equal(run?.checkpoints[0]?.loss, 2);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('AITK run catalogue writes and rereads atomically in an isolated training root', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-index-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { recordYue2AitkRun, listYue2AitkRuns, listAllYue2AitkRuns, aitkRunIndexPath } from './src/services/training/yue2AitkRuns.js';",
      "const out = path.join(process.env.TRAINING_DIR, 'run'); fs.mkdirSync(path.join(out, 'checkpoint-step4'), { recursive: true });",
      "fs.writeFileSync(path.join(out, 'checkpoint-step4', 'native-ar.safetensors'), 'x');",
      "recordYue2AitkRun({ version: 1, jobId: 'job-test', datasetId: 'ds-test', datasetSlug: 'slug-test', method: 'aitk', output: out, options: {}, status: 'running', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "const rows = listYue2AitkRuns('ds-test'); if (rows.length !== 1 || rows[0].checkpoints[0].arPath === undefined) throw new Error('catalogue reread failed');",
      "const all = listAllYue2AitkRuns(); if (all.length !== 1 || all[0].jobId !== 'job-test') throw new Error('all-runs catalogue failed');",
      "if (!fs.existsSync(aitkRunIndexPath())) throw new Error('catalogue missing');",
    ].join('');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)), env: { ...process.env, TRAINING_DIR: root, ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') }, stdio: 'pipe',
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('moving a run renames its output directory and rewrites the index in place', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-move-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { recordYue2AitkRun, listYue2AitkRuns, moveYue2AitkRun } from './src/services/training/yue2AitkRuns.js';",
      "const out = path.join(process.env.TRAINING_DIR, 'run'); fs.mkdirSync(path.join(out, 'checkpoint-step9'), { recursive: true });",
      "fs.writeFileSync(path.join(out, 'checkpoint-step9', 'native-ar.safetensors'), 'x');",
      "recordYue2AitkRun({ version: 1, jobId: 'job-move', datasetId: 'ds-move', datasetSlug: 'slug-move', method: 'aitk', output: out, options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "const target = path.join(process.env.TRAINING_DIR, 'refined', 'run');",
      "moveYue2AitkRun('job-move', target);",
      "if (fs.existsSync(out)) throw new Error('old output still exists');",
      "if (!fs.existsSync(path.join(target, 'checkpoint-step9', 'native-ar.safetensors'))) throw new Error('checkpoint did not move');",
      "const rows = listYue2AitkRuns('ds-move'); if (rows[0].output !== target) throw new Error('index was not rewritten to the new path');",
      "let refused = false; try { moveYue2AitkRun('job-move', target); } catch { refused = true; }",
      "if (!refused) throw new Error('moving onto an existing path should have been refused');",
    ].join('');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)), env: { ...process.env, TRAINING_DIR: root, ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') }, stdio: 'pipe',
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('a run folder moved by hand into refined/ is found again by the index', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-heal-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { recordYue2AitkRun, jointRunForAdapter, listYue2AitkRuns } from './src/services/training/yue2AitkRuns.js';",
      "const out = path.join(process.env.TRAINING_DIR, 'run'); fs.mkdirSync(path.join(out, 'checkpoint-step9'), { recursive: true });",
      "fs.writeFileSync(path.join(out, 'checkpoint-step9', 'native-ar.safetensors'), 'x');",
      "fs.writeFileSync(path.join(out, 'checkpoint-step9', 'native-nar.safetensors'), 'x');",
      "recordYue2AitkRun({ version: 1, jobId: 'job-heal', datasetId: 'ds-heal', datasetSlug: 'slug-heal', method: 'aitk', output: out, options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "const target = path.join(process.env.TRAINING_DIR, 'refined', 'run'); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.renameSync(out, target);",
      "const rows = listYue2AitkRuns('ds-heal'); if (rows[0].output !== target) throw new Error('index did not follow the moved folder: ' + rows[0].output);",
      "if (rows[0].checkpoints[0]?.arPath === undefined) throw new Error('checkpoints not rescanned at the new path');",
      "if (!jointRunForAdapter(path.join(target, 'checkpoint-step9', 'native-ar.safetensors'))) throw new Error('adapter under refined/ not resolved to its run');",
    ].join('');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)), env: { ...process.env, TRAINING_DIR: root, ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') }, stdio: 'pipe',
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('joint catalogue reconciles copied folders, stale records and active runs', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-reconcile-'));
  try {
    const script = [
      "import fs from 'node:fs'; import path from 'node:path'; import assert from 'node:assert/strict';",
      "import { initDb } from './src/db/database.js'; import { insertDataset, listDatasets } from './src/services/training/datasetsRepo.js';",
      "import { recordYue2AitkRun, listAllYue2AitkRuns, findYue2JointAdaptersFor, reconcileYue2AitkRunsAtStartup, aitkRunIndexPath } from './src/services/training/yue2AitkRuns.js';",
      "initDb(); const root = path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters'); fs.mkdirSync(root, { recursive: true });",
      "const now = new Date().toISOString();",
      "function dataset(id, tag, slug = id) { insertDataset({ id, slug, name: id, sourceDir: path.join(process.env.TRAINING_DIR, id), recursive: true, customTag: tag, tagPosition: 'prepend', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now }); }",
      "dataset('a', 'alpha'); dataset('b', 'beta'); dataset('9a036717', 'shared', 'shared'); dataset('e945817c', 'shared', 'shared_covers');",
      "function folder(name) { const out = path.join(root, name + '_2026-10-02_12-00-00'); const ckpt = path.join(out, 'checkpoint-step10'); fs.mkdirSync(ckpt, { recursive: true }); fs.writeFileSync(path.join(ckpt, 'native-ar.safetensors'), 'x'); fs.writeFileSync(path.join(ckpt, 'native-nar.safetensors'), 'x'); return out; }",
      "const plain = folder('alpha'); const copied = folder('beta'); folder('shared'); folder('unknown');",
      "const record = (jobId, datasetId, output, status = 'done') => ({ version: 1, jobId, datasetId, datasetSlug: datasetId, method: 'aitk', output, options: { method: 'base-matched' }, status, createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "fs.writeFileSync(path.join(copied, 'run.json'), JSON.stringify(record('remote-job', 'b', 'Z:/remote/run', 'running')));",
      "const missing = path.join(root, 'missing_2026-10-02_12-00-00'); recordYue2AitkRun(record('stale', 'a', missing)); recordYue2AitkRun(record('active', 'a', path.join(root, 'active'), 'running')); recordYue2AitkRun(record('paused', 'a', path.join(root, 'paused'), 'interrupted'));",
      "reconcileYue2AitkRunsAtStartup(); const runs = listAllYue2AitkRuns();",
      "assert.equal(runs.some(r => r.jobId === 'stale'), false); assert.equal(runs.find(r => r.jobId === 'active')?.status, 'interrupted'); assert.ok(runs.some(r => r.jobId === 'paused'));",
      "assert.equal(runs.find(r => r.jobId === 'remote-job')?.output, copied); assert.equal(runs.find(r => r.jobId === 'remote-job')?.status, 'done'); assert.equal(JSON.parse(fs.readFileSync(path.join(copied, 'run.json'))).output, copied);",
      "assert.equal(runs.filter(r => r.datasetId === 'a' && r.output === plain).length, 1); assert.equal(runs.find(r => r.datasetId === '9a036717')?.output, path.join(root, 'shared_2026-10-02_12-00-00')); assert.equal(runs.some(r => r.datasetId === 'e945817c'), false);",
      "assert.equal(listDatasets().find(ds => ds.id === '9a036717')?.customTag, 'shared'); assert.equal(listDatasets().find(ds => ds.id === 'e945817c')?.customTag, 'shared_covers'); assert.equal(listDatasets().find(ds => ds.id === 'a')?.customTag, 'alpha');",
      "assert.equal(JSON.parse(fs.readFileSync(aitkRunIndexPath())).length, runs.length);",
      "dataset('e', 'unknown'); assert.ok(findYue2JointAdaptersFor([{ id: 'e', slug: 'e' }]).has('e'), 'a newly added dataset resolves an unchanged folder set');",
      "const linked = findYue2JointAdaptersFor([{ id: 'a', slug: 'a' }, { id: 'b', slug: 'b' }]); assert.ok(linked.has('a')); assert.ok(linked.has('b'));",
      "fs.rmSync(plain, { recursive: true }); assert.equal(findYue2JointAdaptersFor([{ id: 'a', slug: 'a' }]).has('a'), false);",
      "const live = folder('live'); recordYue2AitkRun(record('live-job', 'a', live, 'running'));",
      "const write = fs.writeFileSync; const mirrored = []; fs.writeFileSync = function(file, ...args) { if (String(file).endsWith('run.json')) mirrored.push(String(file)); return write.call(this, file, ...args); };",
      "recordYue2AitkRun(record('live-job', 'a', live, 'done')); fs.writeFileSync = write; assert.deepEqual(mirrored, [path.join(live, 'run.json')]); assert.equal(JSON.parse(fs.readFileSync(path.join(live, 'run.json'))).status, 'done');",
    ].join('\n');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)),
      env: { ...process.env, DATA_DIR: path.join(root, 'data'), TRAINING_DIR: path.join(root, 'training'), ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') },
      stdio: 'pipe',
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('startup reconcile only interrupts a local running row, never a remote-origin one sitting beside it', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-reconcile-mixed-'));
  try {
    const script = [
      "import path from 'node:path'; import assert from 'node:assert/strict';",
      "import { initDb } from './src/db/database.js'; import { insertDataset } from './src/services/training/datasetsRepo.js';",
      "import { recordYue2AitkRun, listAllYue2AitkRuns, reconcileYue2AitkRunsAtStartup } from './src/services/training/yue2AitkRuns.js';",
      "initDb();",
      "const now = new Date().toISOString();",
      "insertDataset({ id: 'a', slug: 'a', name: 'a', sourceDir: path.join(process.env.TRAINING_DIR, 'a'), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now });",
      "const record = (jobId, output, extra = {}) => ({ version: 1, jobId, datasetId: 'a', datasetSlug: 'a', method: 'aitk', output, options: {}, status: 'running', createdAt: 1, updatedAt: 2, checkpoints: [], ...extra });",
      "recordYue2AitkRun(record('local-job', path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters', 'local-job')));",
      "recordYue2AitkRun(record('remote-job', path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters', 'remote-job'), { origin: { worker: 'W', remoteJobId: 'r1' } }));",
      "const stale = reconcileYue2AitkRunsAtStartup();",
      "assert.equal(stale, 1, 'only the local row counts as interrupted-by-restart');",
      "const runs = listAllYue2AitkRuns();",
      "assert.equal(runs.find(r => r.jobId === 'local-job')?.status, 'interrupted');",
      "assert.equal(runs.find(r => r.jobId === 'remote-job')?.status, 'running');",
    ].join('\n');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)),
      env: { ...process.env, DATA_DIR: path.join(root, 'data'), TRAINING_DIR: path.join(root, 'training'), ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') },
      stdio: 'pipe',
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('base-matched and fixed-recipe runs are checkpoint ladders; tuned runs are not', () => {
  assert.equal(isCheckpointLadder({ method: 'base-matched' }), true);
  assert.equal(isCheckpointLadder({ method: 'fixed-recipe' }), true);
  assert.equal(isCheckpointLadder({ method: 'tuned' }), false);
  assert.equal(isCheckpointLadder(undefined), false);
});

test('a tuned run with no KL rungs gets every complete checkpoint as a rung', () => {
  const run = { options: {} } as Yue2AitkRunRecord;
  const full = { step: 100, dir: 'a', arPath: 'ar', narPath: 'nar' } as Yue2AitkCheckpointRecord;
  const partial = { step: 50, dir: 'b' } as Yue2AitkCheckpointRecord;
  assert.deepEqual(rungsOf(run, [full, partial]).map(c => !!c.rung), [true, false]);
  const kept = { ...partial, rung: true } as Yue2AitkCheckpointRecord;
  assert.deepEqual(rungsOf(run, [full, kept]).map(c => !!c.rung), [false, true]);
});

test('review catalogue scans each matching run once and sees changes on the next request', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-review-catalogue-'));
  try {
    const script = [
      "import fs from 'node:fs'; import path from 'node:path'; import assert from 'node:assert/strict';",
      "import { recordYue2AitkRun, listYue2AitkRuns, listYue2AitkRunsForDatasets, aitkRunIndexPath, setYue2ReviewComplete, yue2ReviewComplete, setYue2RunFinished, yue2RunFinished } from './src/services/training/yue2AitkRuns.js';",
      "const root = path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters');",
      "function checkpoint(output, step) { const dir = path.join(output, 'checkpoint-step' + step); fs.mkdirSync(dir, { recursive: true }); for (const half of ['ar', 'nar']) fs.writeFileSync(path.join(dir, 'native-' + half + '.safetensors'), 'x'); }",
      "function run(jobId, datasetId, datasetSlug, updatedAt, status = 'done') { const output = path.join(root, jobId); checkpoint(output, 10); fs.writeFileSync(path.join(output, 'train.jsonl'), ''); const record = { version: 1, jobId, datasetId, datasetSlug, method: 'aitk', output, options: { method: 'base-matched' }, status, createdAt: 1, updatedAt, checkpoints: [] }; recordYue2AitkRun(record); return record; }",
      "const a = run('a', 'ds-a', 'alpha', 1); const b = run('b', 'ds-b', 'beta', 2, 'running'); const unrelated = run('unrelated', 'other', 'other', 3);",
      "const datasets = [{ id: 'ds-a', slug: 'alpha' }, { id: 'ds-b', slug: 'beta' }, { id: 'new-id', slug: 'alpha' }, { id: 'empty' }];",
      "const expected = new Map(datasets.map(ds => [ds.id, listYue2AitkRuns(ds.id, ds.slug)]));",
      "const read = fs.readFileSync; const reads = new Map(); fs.readFileSync = function(file, ...args) { const key = String(file); reads.set(key, (reads.get(key) ?? 0) + 1); return read.call(this, file, ...args); };",
      "let snapshot; try { snapshot = listYue2AitkRunsForDatasets(datasets); } finally { fs.readFileSync = read; }",
      "assert.deepEqual(snapshot, expected, 'batch preserves ID matching, slug fallback, labels and ordering');",
      "assert.equal(reads.get(aitkRunIndexPath()), 1, 'one index read for the entire warm request');",
      "for (const r of [a, b]) assert.equal(reads.get(path.join(r.output, 'train.jsonl')), 1, 'scan a shared run only once');",
      "assert.equal(reads.has(path.join(unrelated.output, 'train.jsonl')), false, 'do not scan unrelated runs');",
      "checkpoint(a.output, 20); fs.rmSync(path.join(a.output, 'checkpoint-step10', 'native-nar.safetensors'));",
      "recordYue2AitkRun({ ...b, status: 'done', updatedAt: 4 }); const newer = run('newer', 'ds-a', 'alpha', 5);",
      "setYue2ReviewComplete(a.output, true); setYue2RunFinished(a.output, { pickedStep: 20, pickedBlind: false, pickedLabel: '' });",
      "const fresh = listYue2AitkRunsForDatasets(datasets); const changed = fresh.get('ds-a').find(r => r.jobId === 'a');",
      "assert.deepEqual(fresh.get('ds-a').map(r => r.jobId), ['newer', 'a']); assert.equal(fresh.get('ds-b')[0].status, 'done');",
      "assert.deepEqual(changed.checkpoints.map(c => c.step), [20, 10]); assert.equal(changed.checkpoints[1].narPath, undefined); assert.ok(changed.blindLabels[20]);",
      "assert.equal(changed.blindLabels[10], snapshot.get('ds-a')[0].blindLabels[10], 'existing labels survive a new checkpoint');",
      "assert.ok(yue2ReviewComplete(changed.output)); assert.ok(yue2RunFinished(changed.output));",
      "setYue2ReviewComplete(a.output, false); assert.equal(yue2ReviewComplete(listYue2AitkRunsForDatasets(datasets).get('ds-a')[1].output), false);",
      "assert.equal(snapshot.get('ds-a').length, 1, 'earlier request keeps its own snapshot');",
    ].join('\n');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)),
      env: { ...process.env, DATA_DIR: path.join(root, 'data'), TRAINING_DIR: path.join(root, 'training'), ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') },
      stdio: 'pipe', timeout: 30_000,
    });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('the all-runs snapshot coalesces refreshes and never restores a run deleted mid-refresh', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-snapshot-'));
  try {
    const script = [
      "import fs from 'node:fs'; import path from 'node:path'; import assert from 'node:assert/strict';",
      "import { recordYue2AitkRun, listAllYue2AitkRuns, deleteYue2AitkRun, refreshYue2AitkSnapshot, jointRunForAdapter, yue2AitkWeightStamp } from './src/services/training/yue2AitkRuns.js';",
      "const root = path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters');",
      "function checkpoint(output, step) { const dir = path.join(output, 'checkpoint-step' + step); fs.mkdirSync(dir, { recursive: true }); for (const half of ['ar', 'nar']) fs.writeFileSync(path.join(dir, 'native-' + half + '.safetensors'), 'x'); return dir; }",
      "function run(jobId, updatedAt) { const output = path.join(root, jobId); checkpoint(output, 10); recordYue2AitkRun({ version: 1, jobId, datasetId: 'ds', datasetSlug: '', method: 'aitk', output, options: {}, status: 'done', createdAt: 1, updatedAt, checkpoints: [] }); return output; }",
      "const keep = run('keep', 2); const gone = run('gone', 1);",
      "const steps = () => listAllYue2AitkRuns().map(r => r.jobId + ':' + r.checkpoints.map(c => c.step).join(','));",
      "assert.deepEqual(steps(), ['keep:10', 'gone:10']);",
      "const first = refreshYue2AitkSnapshot(); assert.equal(refreshYue2AitkSnapshot(), first, 'concurrent callers share one refresh');",
      // Delete once the refresh has read the index and is scanning: its stale view must be discarded.
      "const readdir = fs.promises.readdir; let deleted = false;",
      "fs.promises.readdir = function(dir, ...args) { if (!deleted && String(dir) === keep) { deleted = true; deleteYue2AitkRun('gone'); } return readdir.call(this, dir, ...args); };",
      "await first; fs.promises.readdir = readdir; assert.ok(deleted);",
      "assert.deepEqual(steps(), ['keep:10'], 'a run deleted during a refresh stays deleted');",
      "assert.equal(jointRunForAdapter(path.join(gone, 'checkpoint-step10', 'native-ar.safetensors')), undefined);",
      "checkpoint(keep, 20);",
      "assert.deepEqual(steps(), ['keep:10'], 'reads are served from memory');",
      "await refreshYue2AitkSnapshot(); assert.deepEqual(steps(), ['keep:20,10'], 'a refresh picks up a checkpoint written outside the app');",
      // Files rewritten in place, with the checkpoint folder's mtime put back: only file-level checks can see it.
      "const ckpt = path.join(keep, 'checkpoint-step20'); const ar = path.join(ckpt, 'native-ar.safetensors'); const dirTimes = fs.statSync(ckpt);",
      "const kl = () => listAllYue2AitkRuns()[0].checkpoints.find(c => c.step === 20).kl; const stamp = yue2AitkWeightStamp(ar); assert.notEqual(stamp, '');",
      "fs.writeFileSync(path.join(ckpt, 'meters.json'), JSON.stringify({ kl_reading: 0.9 })); fs.writeFileSync(ar, 'a longer weight'); fs.utimesSync(ckpt, dirTimes.atime, dirTimes.mtime);",
      "await refreshYue2AitkSnapshot(); assert.equal(kl(), 0.9, 'a meters.json rewritten in place is read by the next refresh');",
      "assert.notEqual(yue2AitkWeightStamp(ar), stamp, 'a weight rewritten in place changes its stamp, so the picker re-reads its header');",
    ].join('\n');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--input-type=module', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)),
      env: { ...process.env, DATA_DIR: path.join(root, 'data'), TRAINING_DIR: path.join(root, 'training'), ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') },
      stdio: 'pipe', timeout: 30_000,
    });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
