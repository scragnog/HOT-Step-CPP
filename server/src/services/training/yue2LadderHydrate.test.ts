// "Use this rung" on a pulled ladder: fetching just the chosen rung's
// checkpoint from its worker before linking (trainingWorkers.ts's
// hydrateYue2LadderCheckpoint, wired into POST .../yue2-joint-preset and
// yue2BatchRunner.ts's finishLadder), over a fake worker HTTP server.
// Every test runs in an isolated subprocess (own TRAINING_DIR/
// ACESTEPCPP_ADAPTERS/DATA_DIR) — config and the run index are module-level
// singletons, so tests sharing a process would see each other's state.
//
// Checksums are computed here, in this normal file, and spliced into each
// generated script as a literal hex string — see yue2LadderPull.test.ts's
// header for why (Node's --eval auto-wraps any script containing the bare
// word "crypto" in a way that breaks a script that also `import`s).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const sha256Hex = (s: string): string => createHash('sha256').update(s).digest('hex');

const TRAINING_SRC_ROOT = fileURLToPath(new URL('../../../', import.meta.url));
function runInIsolatedRoot(root: string, script: string): void {
  const env = {
    ...process.env, TRAINING_DIR: path.join(root, 'training'),
    ACESTEPCPP_ADAPTERS: path.join(root, 'adapters'), DATA_DIR: path.join(root, 'data'),
  };
  execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], { cwd: TRAINING_SRC_ROOT, env, stdio: 'pipe' });
}

// Shared boilerplate: initDb, a dataset row, a pulled remote-origin run with
// one remote-availability checkpoint, a fake worker HTTP server standing in
// for /yue2-ladder-checkpoint, /yue2-ladder-checkpoint-file and the whole-
// ladder DELETE, and a tiny express app mounting the real training router so
// POST .../yue2-joint-preset runs for real, not a mock.
const HARNESS = [
  "import fs from 'node:fs';",
  "import path from 'node:path';",
  "import http from 'node:http';",
  "import express from 'express';",
  "import { initDb } from './src/db/database.js';",
  "import * as repo from './src/services/training/datasetsRepo.js';",
  "import { config } from './src/config.js';",
  "import { recordYue2AitkRun, listYue2AitkRuns } from './src/services/training/yue2AitkRuns.js';",
  "import trainingRoutes from './src/routes/training.js';",
  "initDb();",
  "const now = Date.now();",
  "function addDataset(slug) { const iso = new Date(now).toISOString(); repo.insertDataset({ id: 'ds-' + slug, slug, name: slug, sourceDir: path.join(process.env.TRAINING_DIR, 'src-' + slug), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: iso, updatedAt: iso }); }",
  "function serve(state) {",
  "  const server = http.createServer((req, res) => {",
  "    const u = new URL(req.url, 'http://x');",
  "    if (req.method === 'GET' && u.pathname === '/api/training/worker/yue2-ladder-checkpoint') { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(state.manifest)); return; }",
  "    if (req.method === 'GET' && u.pathname === '/api/training/worker/yue2-ladder-checkpoint-file') {",
  "      const body = state.files[u.searchParams.get('file')];",
  "      if (!body) { res.statusCode = 404; res.end('{}'); return; }",
  "      res.end(body); return;",
  "    }",
  "    if (req.method === 'DELETE' && u.pathname.startsWith('/api/training/worker/yue2-ladders/')) { state.deleted.push(u.pathname); res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ ok: true })); return; }",
  "    res.statusCode = 404; res.end();",
  "  });",
  "  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));",
  "}",
  "async function appServer() { const app = express(); app.use(express.json()); app.use('/api/training', trainingRoutes); return new Promise(resolve => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); }); }",
  "function recordPulledRun(jobId, remoteJobId, worker, step) {",
  "  const output = path.join(config.aceServer.adapters, 'yue2-joint-adapters', 'remote-' + worker.toLowerCase() + '-' + remoteJobId);",
  "  const dir = path.join(output, 'checkpoint-step' + step);",
  "  recordYue2AitkRun({ version: 1, jobId, datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output,",
  "    options: { method: 'base-matched' }, status: 'done', createdAt: now, updatedAt: now,",
  "    checkpoints: [{ step, dir, kl: 1.2, rung: true, availability: 'remote' }], origin: { worker, remoteJobId } });",
  "  return dir;",
  "}",
].join('');

test('Use this rung fetches, verifies and links a remote-only checkpoint, then tells the worker to drop the whole ladder', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-ok-'));
  try {
    const ar = Buffer.from('AR-BYTES'); const nar = Buffer.from('NAR-BYTES');
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: ${ar.length} }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: ${nar.length} }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  const body = await res.json();",
      "  if (!res.ok) throw new Error('expected success: ' + JSON.stringify(body));",
      "  if (!fs.readFileSync(path.join(dir, 'native-ar.safetensors')).equals(Buffer.from('AR-BYTES'))) throw new Error('ar bytes do not match');",
      "  if (!fs.readFileSync(path.join(dir, 'native-nar.safetensors')).equals(Buffer.from('NAR-BYTES'))) throw new Error('nar bytes do not match');",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (!run.checkpoints[0].arPath || !run.checkpoints[0].narPath) throw new Error('local scan did not pick up the hydrated checkpoint: ' + JSON.stringify(run.checkpoints));",
      "  if (state.deleted.length !== 1 || !state.deleted[0].endsWith('/job1')) throw new Error('worker was not asked to drop the ladder: ' + JSON.stringify(state.deleted));",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a failed fetch links nothing and never asks the worker to delete', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-fail-'));
  try {
    const ar = Buffer.from('AR-BYTES');
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      // The manifest promises a nar file; the file route never serves it (offline/404).
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: ${ar.length} }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: 4 }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  if (res.ok) throw new Error('expected a non-2xx response for a failed fetch');",
      "  if (fs.existsSync(path.join(dir, 'native-nar.safetensors'))) throw new Error('a file that never arrived must not be present');",
      // The ar half may legitimately have landed (a partial download resumes
      // next time); what must never happen is a complete, usable checkpoint
      // or a link — narPath is the proof nothing was ever considered ready.
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (run.checkpoints[0].narPath) throw new Error('nothing should have linked: ' + JSON.stringify(run.checkpoints));",
      "  if (state.deleted.length) throw new Error('no delete on a failed fetch: ' + JSON.stringify(state.deleted));",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a hash mismatch links nothing and never asks the worker to delete', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-mismatch-'));
  try {
    const good = Buffer.from('AR-BYTES'); const corrupt = Buffer.from('AR-XXXXX'); // same length, different bytes
    if (good.length !== corrupt.length) throw new Error('fixture bug: buffers must be same size');
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: ${good.length} }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: 9 }], files: { 'native-ar.safetensors': Buffer.from('AR-XXXXX'), 'native-nar.safetensors': Buffer.from('NAR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  if (res.ok) throw new Error('expected a non-2xx response for a hash mismatch');",
      "  if (fs.existsSync(path.join(dir, 'native-ar.safetensors'))) throw new Error('a corrupt transfer must not land at its destination name');",
      // The nar half may legitimately have landed (its hash was fine); what
      // must never happen is a complete checkpoint or a link — arPath is
      // the proof the corrupt half was never accepted as ready.
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (run.checkpoints[0].arPath) throw new Error('nothing should have linked: ' + JSON.stringify(run.checkpoints));",
      "  if (state.deleted.length) throw new Error('no delete on a hash mismatch: ' + JSON.stringify(state.deleted));",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a worker manifest missing the nar file is refused before any fetch attempt', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-partial-manifest-'));
  try {
    const ar = Buffer.from('AR-BYTES');
    const hashAr = sha256Hex('AR-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      // Manifest reports only the ar file — the worker never trained far enough, or lost the nar half.
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: ${ar.length} }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  if (res.ok) throw new Error('expected a non-2xx response for an incomplete manifest');",
      "  if (fs.existsSync(path.join(dir, 'native-ar.safetensors'))) throw new Error('an incomplete manifest must not even attempt a fetch');",
      "  if (state.deleted.length) throw new Error('no delete when the worker cannot supply a complete checkpoint: ' + JSON.stringify(state.deleted));",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
