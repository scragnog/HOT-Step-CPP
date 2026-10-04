import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { config } from '../../config.js';
import { listWorkers, resolveDatasetFile } from './trainingWorkers.js';
import { labelsDir } from './paths.js';

test('pushed file paths stay inside the dataset or its labels folder', () => {
  const root = path.resolve('/tmp/ds');
  assert.equal(resolveDatasetFile(root, 'band', 'a/song.flac'), path.join(root, 'a', 'song.flac'));
  assert.equal(resolveDatasetFile(root, 'band', '__labels/x.json'), path.join(labelsDir('band'), 'x.json'));
  for (const bad of ['../evil.flac', 'a/../../evil', '__labels/../x.json', '', '.', path.resolve('/elsewhere/x')]) {
    assert.throws(() => resolveDatasetFile(root, 'band', bad), /Refused path/, bad);
  }
});

test('TRAINING_WORKERS parses name=url pairs and skips junk', () => {
  const before = config.workers.list;
  config.workers.list = 'LivingRoom=http://192.168.50.50:3001/, bad, =http://x, Ftp=ftp://y ;Other = https://o:1';
  try {
    assert.deepEqual(listWorkers(), [
      { name: 'LivingRoom', url: 'http://192.168.50.50:3001' },
      { name: 'Other', url: 'https://o:1' },
    ]);
  } finally { config.workers.list = before; }
});

const TRAINING_SRC_ROOT = fileURLToPath(new URL('../../../', import.meta.url));
function runPullInIsolatedRoot(root: string, script: string): void {
  const env = {
    ...process.env, TRAINING_DIR: path.join(root, 'training'),
    ACESTEPCPP_ADAPTERS: path.join(root, 'adapters'), DATA_DIR: path.join(root, 'data'),
  };
  execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], { cwd: TRAINING_SRC_ROOT, env, stdio: 'pipe' });
}

test('workerLinkedPairs reads keptStep from the checkpoint directory itself, not a misleading ancestor folder name', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-worker-keptstep-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { recordYue2AitkRun } from './src/services/training/yue2AitkRuns.js';",
      "import { workerLinkedPairs } from './src/services/training/trainingWorkers.js';",
      // A run folder whose own name happens to look like a checkpoint dir —
      // yue2JointOutputDirectory's trigger sanitization does not forbid this.
      "const runDir = path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters', 'checkpoint-step999_2026-10-01_12-00-00');",
      "const ckptDir = path.join(runDir, 'segments', 'segment-000001', 'checkpoint-step120');",
      "fs.mkdirSync(ckptDir, { recursive: true });",
      "const arPath = path.join(ckptDir, 'native-ar.safetensors'); const narPath = path.join(ckptDir, 'native-nar.safetensors');",
      "fs.writeFileSync(arPath, 'AR'); fs.writeFileSync(narPath, 'NAR');",
      "recordYue2AitkRun({ version: 1, jobId: 'job1', datasetId: 'ds1', datasetSlug: 'album', method: 'aitk', output: runDir, options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "fs.mkdirSync(process.env.TRAINING_DIR, { recursive: true });",
      "fs.writeFileSync(path.join(process.env.TRAINING_DIR, 'yue2-linked.json'), JSON.stringify({ album: { arPath, narPath, at: new Date().toISOString() } }));",
      "const pairs = workerLinkedPairs();",
      "if (pairs.length !== 1) throw new Error('expected one linked pair: ' + JSON.stringify(pairs));",
      "if (pairs[0].keptStep !== 120) throw new Error('keptStep picked up the ancestor folder name, not the checkpoint dir: ' + pairs[0].keptStep);",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

// ── pullLinked: a fake worker over a real loopback HTTP server, with
// TRAINING_DIR/ACESTEPCPP_ADAPTERS/DATA_DIR isolated to a temp root so the
// pull never touches this checkout's real adapters, training dir or db. ──

// Shared boilerplate every pull test needs: initDb, a dataset row per slug,
// and a tiny HTTP server standing in for the worker's /linked + /adapter-file.
const PULL_HARNESS = [
  "import fs from 'node:fs';",
  "import path from 'node:path';",
  "import http from 'node:http';",
  "import { initDb } from './src/db/database.js';",
  "import * as repo from './src/services/training/datasetsRepo.js';",
  "import { pullLinked } from './src/services/training/trainingWorkers.js';",
  "import { trainLogArchiveDir } from './src/services/training/datasetProfile.js';",
  "initDb();",
  "const now = new Date().toISOString();",
  "function addDataset(slug) { repo.insertDataset({ id: 'ds-' + slug, slug, name: slug, sourceDir: path.join(process.env.TRAINING_DIR, 'src-' + slug), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now }); }",
  "async function serve(linked, files) {",
  "  const server = http.createServer((req, res) => {",
  "    const u = new URL(req.url, 'http://x');",
  "    if (u.pathname === '/api/training/worker/linked') { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ linked })); return; }",
  "    if (u.pathname === '/api/training/worker/adapter-file') { const body = files[u.searchParams.get('rel')]; if (!body) { res.statusCode = 404; res.end('{}'); return; } res.end(body); return; }",
  "    res.statusCode = 404; res.end();",
  "  });",
  "  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));",
  "  return server;",
  "}",
].join('');

test('pullLinked fetches a linked pair\'s train log and notes the kept step', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-pull-logs-'));
  try {
    const script = PULL_HARNESS + [
      "addDataset('album');",
      "const AR = Buffer.from('AR-BYTES'); const NAR = Buffer.from('NAR-BYTES'); const LOG = Buffer.from('{\"stage\":\"joint\",\"step\":1}\\n');",
      "const files = { 'album/ar.safetensors': AR, 'album/nar.safetensors': NAR, 'album/job1-logs/segment-000001/train.jsonl': LOG };",
      "const linked = [{ slug: 'album', at: now, ar: { rel: 'album/ar.safetensors', size: AR.length, mtimeMs: 1 }, nar: { rel: 'album/nar.safetensors', size: NAR.length, mtimeMs: 1 }, jobId: 'job1', keptStep: 120, logs: [{ seg: 'segment-000001', rel: 'album/job1-logs/segment-000001/train.jsonl', size: LOG.length, mtimeMs: 1 }] }];",
      "const server = await serve(linked, files);",
      "try {",
      "  const pulled = await pullLinked({ name: 'worker1', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled.length !== 1 || pulled[0].status !== 'fetched') throw new Error('unexpected pull result: ' + JSON.stringify(pulled));",
      "  const arDest = path.join(process.env.ACESTEPCPP_ADAPTERS, 'album', 'ar.safetensors');",
      "  if (fs.readFileSync(arDest, 'utf8') !== AR.toString()) throw new Error('ar file missing or wrong content');",
      "  const logDest = path.join(trainLogArchiveDir('album'), 'job1-segment-000001.jsonl');",
      "  if (fs.readFileSync(logDest, 'utf8') !== LOG.toString()) throw new Error('log file missing or wrong content');",
      "  const sidecar = JSON.parse(fs.readFileSync(path.join(trainLogArchiveDir('album'), 'job1.json'), 'utf8'));",
      "  if (sidecar.keptStep !== 120 || sidecar.pulledFrom !== 'worker1') throw new Error('sidecar missing fields: ' + JSON.stringify(sidecar));",
      "} finally { server.close(); }",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('pullLinked still pulls adapters only when the worker sends none of the new log fields', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-pull-old-worker-'));
  try {
    const script = PULL_HARNESS + [
      "addDataset('album');",
      "const AR = Buffer.from('AR-BYTES'); const NAR = Buffer.from('NAR-BYTES');",
      "const files = { 'album/ar.safetensors': AR, 'album/nar.safetensors': NAR };",
      "const linked = [{ slug: 'album', at: now, ar: { rel: 'album/ar.safetensors', size: AR.length, mtimeMs: 1 }, nar: { rel: 'album/nar.safetensors', size: NAR.length, mtimeMs: 1 } }];",
      "const server = await serve(linked, files);",
      "try {",
      "  const pulled = await pullLinked({ name: 'worker1', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled.length !== 1 || pulled[0].status !== 'fetched' || pulled[0].bytes !== AR.length + NAR.length) throw new Error('unexpected pull result: ' + JSON.stringify(pulled));",
      "  if (fs.existsSync(trainLogArchiveDir('album'))) throw new Error('train-logs dir created for a pair with no logs');",
      "} finally { server.close(); }",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('scoring a run already known locally through a worker is refused with 409', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-score-local-guard-'));
  try {
    const script = [
      "import path from 'node:path';",
      "import { Readable } from 'node:stream';",
      "import { initDb } from './src/db/database.js';",
      "import * as repo from './src/services/training/datasetsRepo.js';",
      "import { recordYue2AitkRun } from './src/services/training/yue2AitkRuns.js';",
      "import { proxyToWorker } from './src/services/training/trainingWorkers.js';",
      "import { config } from './src/config.js';",
      "initDb();",
      "const now = new Date().toISOString();",
      "config.workers.list = 'Mock=http://127.0.0.1:1';",
      "repo.insertDataset({ id: 'ds-album', slug: 'album', name: 'album', sourceDir: path.join(process.env.TRAINING_DIR, 'src-album'), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now });",
      "recordYue2AitkRun({ version: 1, jobId: 'local-run', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output: path.join(process.env.TRAINING_DIR, 'out'), options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "const req = Readable.from([Buffer.from(JSON.stringify({ refineRun: 'local-run', step: 10, likeness: 4 }))]);",
      "Object.assign(req, { url: '/training/datasets/ds-album/yue2-rung-scores', method: 'PUT', params: { name: 'Mock' }, query: {}, headers: {} });",
      "let status = 0; let payload;",
      "const res = { status(c) { status = c; return this; }, json(p) { payload = p; }, headersSent: false, on() {} };",
      "await proxyToWorker(req, res);",
      "if (status !== 409) throw new Error('expected 409, got ' + status + ' ' + JSON.stringify(payload));",
      "if (!/already known on this machine/.test(payload?.error ?? '')) throw new Error('unexpected message: ' + JSON.stringify(payload));",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('deleting a worker ladder folder is refused while another job for the dataset is active', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-delete-active-job-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { initDb } from './src/db/database.js';",
      "import * as repo from './src/services/training/datasetsRepo.js';",
      "import { recordYue2AitkRun } from './src/services/training/yue2AitkRuns.js';",
      "import { deleteWorkerYue2LadderFolder } from './src/services/training/trainingWorkers.js';",
      "import * as queue from './src/services/training/labelingQueue.js';",
      "initDb();",
      "const now = new Date().toISOString();",
      "repo.insertDataset({ id: 'ds-album', slug: 'album', name: 'album', sourceDir: path.join(process.env.TRAINING_DIR, 'src-album'), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now });",
      // The ladder itself reports 'done' — only a *different* job for the
      // same dataset (a NAR follow-up resuming from this checkpoint, or GPU
      // preview work) is still active (Reviewer, blocker #2).
      "const output = path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters', 'job1');",
      "fs.mkdirSync(output, { recursive: true });",
      "recordYue2AitkRun({ version: 1, jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output, options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "queue.createJob('yue2-ar-train', 'ds-album', [], {});",
      "let threw = false;",
      "try { deleteWorkerYue2LadderFolder('ds-album', 'job1'); } catch (err) { threw = true; if (err?.status !== 409) throw new Error('expected 409, got ' + err?.status); }",
      "if (!threw) throw new Error('expected the delete to be refused while another job is active');",
      "if (!fs.existsSync(output)) throw new Error('fixture bug: output should still be here');",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('deleting a worker ladder folder is refused while the GPU lane is busy or queued (a manual preview render)', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-delete-gpu-lane-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { initDb } from './src/db/database.js';",
      "import * as repo from './src/services/training/datasetsRepo.js';",
      "import { recordYue2AitkRun } from './src/services/training/yue2AitkRuns.js';",
      "import { deleteWorkerYue2LadderFolder } from './src/services/training/trainingWorkers.js';",
      "import { runOnGpuLane } from './src/services/generation/gpuLane.js';",
      "initDb();",
      "const now = new Date().toISOString();",
      "repo.insertDataset({ id: 'ds-album', slug: 'album', name: 'album', sourceDir: path.join(process.env.TRAINING_DIR, 'src-album'), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now });",
      // A manual preview render uses the GPU lane directly, never the
      // labeling queue — activeJobForDataset alone would miss it entirely
      // (Reviewer, round 3 P1).
      "const output = path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters', 'job1');",
      "fs.mkdirSync(output, { recursive: true });",
      "recordYue2AitkRun({ version: 1, jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output, options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "void runOnGpuLane(() => new Promise(() => {}), { label: 'manual preview' });",
      "await new Promise(resolve => setImmediate(resolve));",
      "let threw = false;",
      "try { deleteWorkerYue2LadderFolder('ds-album', 'job1'); } catch (err) { threw = true; if (err?.status !== 409) throw new Error('expected 409, got ' + err?.status); }",
      "if (!threw) throw new Error('expected the delete to be refused while the GPU lane is held');",
      "if (!fs.existsSync(output)) throw new Error('fixture bug: output should still be here');",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('deleting a worker ladder folder refuses a run record whose output IS the ladder root itself', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-delete-root-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { initDb } from './src/db/database.js';",
      "import * as repo from './src/services/training/datasetsRepo.js';",
      "import { recordYue2AitkRun } from './src/services/training/yue2AitkRuns.js';",
      "import { deleteWorkerYue2LadderFolder } from './src/services/training/trainingWorkers.js';",
      "initDb();",
      "const now = new Date().toISOString();",
      "repo.insertDataset({ id: 'ds-album', slug: 'album', name: 'album', sourceDir: path.join(process.env.TRAINING_DIR, 'src-album'), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now });",
      // isInside treats the root itself as "inside" by design for its other
      // callers; a corrupted run record whose output IS the ladder root must
      // still be refused, or it recursively deletes every other ladder too
      // (Reviewer, round 3 P1 — reproduced with an unrelated sibling marker).
      "const ladderRoot = path.join(process.env.ACESTEPCPP_ADAPTERS, 'yue2-joint-adapters');",
      "fs.mkdirSync(ladderRoot, { recursive: true });",
      "const marker = path.join(ladderRoot, 'sibling-marker.txt');",
      "fs.writeFileSync(marker, 'x');",
      "recordYue2AitkRun({ version: 1, jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output: ladderRoot, options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "let threw = false;",
      "try { deleteWorkerYue2LadderFolder('ds-album', 'job1'); } catch (err) { threw = true; }",
      "if (!threw) throw new Error('expected the delete to be refused when output is the ladder root itself');",
      "if (!fs.existsSync(marker)) throw new Error('the sibling marker must survive: the whole ladder tree must never be wiped');",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('deleting a worker ladder folder refuses a run record whose output points outside the ladder tree', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-delete-containment-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { initDb } from './src/db/database.js';",
      "import * as repo from './src/services/training/datasetsRepo.js';",
      "import { recordYue2AitkRun } from './src/services/training/yue2AitkRuns.js';",
      "import { deleteWorkerYue2LadderFolder } from './src/services/training/trainingWorkers.js';",
      "initDb();",
      "const now = new Date().toISOString();",
      "repo.insertDataset({ id: 'ds-album', slug: 'album', name: 'album', sourceDir: path.join(process.env.TRAINING_DIR, 'src-album'), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: now, updatedAt: now });",
      // A corrupted index entry pointing outside the ladder tree entirely.
      "const outside = path.join(process.env.TRAINING_DIR, 'not-a-ladder-folder');",
      "fs.mkdirSync(outside, { recursive: true }); fs.writeFileSync(path.join(outside, 'keep.txt'), 'x');",
      "recordYue2AitkRun({ version: 1, jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output: outside, options: {}, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "let threw = false;",
      "try { deleteWorkerYue2LadderFolder('ds-album', 'job1'); } catch (err) { threw = true; }",
      "if (!threw) throw new Error('expected the delete to be refused for a path outside the ladder tree');",
      "if (!fs.existsSync(path.join(outside, 'keep.txt'))) throw new Error('the outside folder must never be touched');",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('pullLinked rejects traversal in a linked pair\'s jobId or log seg', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-pull-traversal-'));
  try {
    const script = PULL_HARNESS + [
      "addDataset('alpha'); addDataset('beta');",
      "const AR = Buffer.from('AR'); const NAR = Buffer.from('NAR'); const LOG = Buffer.from('log');",
      "const files = {",
      "  'alpha/ar.safetensors': AR, 'alpha/nar.safetensors': NAR, 'alpha/escape.jsonl': LOG,",
      "  'beta/ar.safetensors': AR, 'beta/nar.safetensors': NAR, 'beta/escape2.jsonl': LOG,",
      "};",
      "const linked = [",
      "  { slug: 'alpha', at: now, ar: { rel: 'alpha/ar.safetensors', size: AR.length, mtimeMs: 1 }, nar: { rel: 'alpha/nar.safetensors', size: NAR.length, mtimeMs: 1 }, jobId: '../evil', keptStep: 1, logs: [{ seg: 'segment-000001', rel: 'alpha/escape.jsonl', size: LOG.length, mtimeMs: 1 }] },",
      "  { slug: 'beta', at: now, ar: { rel: 'beta/ar.safetensors', size: AR.length, mtimeMs: 1 }, nar: { rel: 'beta/nar.safetensors', size: NAR.length, mtimeMs: 1 }, jobId: 'job2', keptStep: 1, logs: [{ seg: '..', rel: 'beta/escape2.jsonl', size: LOG.length, mtimeMs: 1 }] },",
      "];",
      "const server = await serve(linked, files);",
      "try {",
      "  const pulled = await pullLinked({ name: 'worker1', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled.length !== 2 || pulled.some(p => p.status !== 'fetched')) throw new Error('unexpected pull result: ' + JSON.stringify(pulled));",
      "  const trainingRoot = path.join(process.env.TRAINING_DIR);",
      "  const escaped = fs.readdirSync(process.env.TRAINING_DIR, { recursive: true }).filter(f => String(f).includes('evil') || String(f).endsWith('..jsonl'));",
      "  if (escaped.length) throw new Error('a log landed outside its archive dir: ' + JSON.stringify(escaped));",
      "  if (fs.existsSync(trainLogArchiveDir('alpha')) && fs.readdirSync(trainLogArchiveDir('alpha')).some(f => f.includes('evil'))) throw new Error('unsafe jobId was not rejected');",
      "  if (fs.existsSync(trainLogArchiveDir('beta')) && fs.readdirSync(trainLogArchiveDir('beta')).some(f => f.endsWith('-..jsonl'))) throw new Error('unsafe seg was not rejected');",
      "} finally { server.close(); }",
    ].join('');
    runPullInIsolatedRoot(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
