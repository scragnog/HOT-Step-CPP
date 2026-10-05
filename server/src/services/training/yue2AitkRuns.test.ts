import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import { checkpointRecords } from './yue2AitkRuns.js';
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
