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

test('using an already-verified, now worker-deleted rung again revalidates locally instead of asking a worker that is gone', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-retired-'));
  try {
    // Reviewer, round 3 P2: after a successful "Use this rung" link, the
    // worker's whole ladder folder is deleted (deleteWorkerYue2Ladder).
    // Using the same rung again — Finish scored's direct link path hits the
    // same route — must not require a live worker manifest, since the
    // worker legitimately has nothing left to report.
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
      "  const first = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  if (!first.ok) throw new Error('expected the first use to succeed: ' + JSON.stringify(await first.json()));",
      "  if (state.deleted.length !== 1) throw new Error('expected the worker ladder to be dropped after the first successful link');",
      // The worker's copy is gone now: every route 404s, as deleteWorkerYue2LadderFolder leaves it.
      "  state.manifest = []; state.files = {};",
      "  const second = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  const body = await second.json();",
      "  if (!second.ok) throw new Error('expected the second use to succeed from the persisted verified record: ' + JSON.stringify(body));",
      "  if (!fs.readFileSync(path.join(dir, 'native-ar.safetensors')).equals(Buffer.from('AR-BYTES'))) throw new Error('ar bytes changed unexpectedly');",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a corrupt local file with no verified record and a dead worker is refused, not trusted', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-dead-worker-'));
  try {
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      // No manifest/files at all: the worker is simply gone, and this
      // checkpoint was never verified before (no prior successful hydrate).
      "const state = { manifest: [], files: {}, deleted: [] };",
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  if (res.ok) throw new Error('expected a non-2xx response: nothing was ever verified and the worker has nothing to serve');",
      "  if (fs.existsSync(path.join(dir, 'native-ar.safetensors'))) throw new Error('nothing should have been written');",
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

test('a manifest entry outside the fixed checkpoint filename set is refused, never written to disk', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-bad-name-'));
  try {
    // Reviewer, blocker #3: an unvalidated f.name let a malicious/buggy
    // manifest entry escape ckpt.dir via path.join. Ar/nar are still valid
    // and complete, so the rung must still link — only the bad entry drops.
    const ar = Buffer.from('AR-BYTES'); const nar = Buffer.from('NAR-BYTES');
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: ${ar.length} }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: ${nar.length} }, { name: '../../escaped.json', sha256: '${hashAr}', bytes: ${ar.length} }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES'), '../../escaped.json': Buffer.from('AR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  const body = await res.json();",
      "  if (!res.ok) throw new Error('expected success despite the one bad entry: ' + JSON.stringify(body));",
      "  if (fs.existsSync(path.join(dir, '..', '..', 'escaped.json'))) throw new Error('the unsafe filename must never be written outside the checkpoint dir');",
      "  if (fs.existsSync(path.join(path.dirname(path.dirname(dir)), 'escaped.json'))) throw new Error('the unsafe filename must never land one level up either');",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (!run.checkpoints[0].arPath || !run.checkpoints[0].narPath) throw new Error('the valid half of the manifest must still link: ' + JSON.stringify(run.checkpoints));",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a corrupt local file left by a prior attempt is caught and repaired, never linked as-is', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-stale-corrupt-'));
  try {
    // Reviewer, blocker #4: the old code trusted an existing arPath/narPath
    // without re-checking the worker's manifest, so a file corrupted after a
    // prior partial attempt would be linked unverified. The fix always
    // re-fetches a fresh manifest and re-hashes every file, even ones
    // already present, before considering the checkpoint ready.
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      // A stale, corrupt ar file already sitting on disk from an earlier attempt.
      "fs.mkdirSync(dir, { recursive: true });",
      "fs.writeFileSync(path.join(dir, 'native-ar.safetensors'), Buffer.from('CORRUPT!'));",
      "fs.writeFileSync(path.join(dir, 'native-nar.safetensors'), Buffer.from('NAR-BYTES'));",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: 8 }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: 9 }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  const body = await res.json();",
      "  if (!res.ok) throw new Error('expected the repair to succeed: ' + JSON.stringify(body));",
      "  if (!fs.readFileSync(path.join(dir, 'native-ar.safetensors')).equals(Buffer.from('AR-BYTES'))) throw new Error('the stale corrupt ar file was linked instead of repaired');",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a failed durable link write stops before the worker delete, even though the preset count is 0', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-link-write-fail-'));
  try {
    // Reviewer, blocker #1: make yue2-linked.json a directory so the write
    // throws. No preset is linked to this dataset either, so `updated` was
    // always going to be 0 — the old code read that as success and still
    // told the worker to drop its copy. The fix must distinguish "nothing
    // needed updating" from "the record write failed" and must never delete.
    const ar = Buffer.from('AR-BYTES'); const nar = Buffer.from('NAR-BYTES');
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      "fs.mkdirSync(path.join(process.env.TRAINING_DIR, 'yue2-linked.json'), { recursive: true });",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: ${ar.length} }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: ${nar.length} }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "try {",
      "  const res = await fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "  if (res.ok) throw new Error('expected a non-2xx response when the link record cannot be written');",
      "  if (state.deleted.length) throw new Error('no worker delete when the durable link write failed: ' + JSON.stringify(state.deleted));",
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

// Reviewer, 135990e3 P1: the worker declares an optimizer.resume it cannot
// serve. The first use is partial; a retry must stay partial too (asking the
// worker again), never link the ar/nar subset and retire the worker's copy.
test('a partial transfer stays partial on retry: nothing links and the worker keeps its copy', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-partial-retry-'));
  try {
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES'); const hashOpt = sha256Hex('OPT-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: 8 }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: 9 }, { name: 'optimizer.resume', sha256: '${hashOpt}', bytes: 9 }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "const use = () => fetch('http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "try {",
      "  const first = await use();",
      "  if (first.ok) throw new Error('expected the first use to be partial: ' + JSON.stringify(await first.json()));",
      "  if (listYue2AitkRuns('ds-album', 'album')[0].checkpoints[0].manifestSha256) throw new Error('a partial transfer must record no manifest');",
      "  const second = await use();",
      "  if (second.ok) throw new Error('expected the retry to stay partial: ' + JSON.stringify(await second.json()));",
      "  if (state.deleted.length) throw new Error('the worker copy must not be retired while optimizer.resume is missing: ' + JSON.stringify(state.deleted));",
      // Once the worker can serve it, the same rung completes and links.
      "  state.files['optimizer.resume'] = Buffer.from('OPT-BYTES');",
      "  const third = await use();",
      "  if (!third.ok) throw new Error('expected the complete transfer to link: ' + JSON.stringify(await third.json()));",
      "  if (!fs.readFileSync(path.join(dir, 'optimizer.resume')).equals(Buffer.from('OPT-BYTES'))) throw new Error('optimizer.resume did not land');",
      "  if (state.deleted.length !== 1) throw new Error('expected the worker ladder to be dropped once complete: ' + JSON.stringify(state.deleted));",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

// Reviewer, 135990e3 P2: cleanup prunes optimizer.resume on purpose. After
// the worker has retired its copy, using the rung again must revalidate the
// retained ar/nar weights against the record and not demand the pruned file.
test('use, cleanup with the resume file pruned, then reuse after the worker is retired', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-cleanup-reuse-'));
  try {
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES'); const hashOpt = sha256Hex('OPT-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: 8 }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: 9 }, { name: 'optimizer.resume', sha256: '${hashOpt}', bytes: 9 }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES'), 'optimizer.resume': Buffer.from('OPT-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "const base = 'http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/';",
      "let dir2 = dir;",
      "const use = () => fetch(base + 'yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir2 }) });",
      "try {",
      "  const first = await use();",
      "  if (!first.ok) throw new Error('expected the first use to succeed: ' + JSON.stringify(await first.json()));",
      "  const cleanup = await fetch(base + 'yue2-cleanup', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ run: 'remote:W:job1', step: 10, resume: true }) });",
      "  if (!cleanup.ok) throw new Error('cleanup failed: ' + JSON.stringify(await cleanup.json()));",
      // A finished pulled run moves to its local name; follow it through the index.
      "  dir2 = listYue2AitkRuns('ds-album', 'album')[0].checkpoints.find(c => c.step === 10).dir;",
      "  if (fs.existsSync(path.join(dir2, 'optimizer.resume'))) throw new Error('cleanup did not prune the resume file');",
      "  state.manifest = []; state.files = {};",
      "  const second = await use();",
      "  if (!second.ok) throw new Error('expected reuse to revalidate locally: ' + JSON.stringify(await second.json()));",
      "  if (fs.existsSync(path.join(dir2, 'optimizer.resume'))) throw new Error('a pruned file must not come back');",
      // The retained weights are still checked: a flipped byte is refused, not trusted.
      "  fs.writeFileSync(path.join(dir2, 'native-ar.safetensors'), Buffer.from('AR-XXXXX'));",
      "  const third = await use();",
      "  if (third.ok) throw new Error('a corrupt retained weight must be refused once the worker is gone');",
      "} finally { workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

// Reviewer, 84b3edf6 P2: the run index write that records the prune fails
// (recordYue2AitkRun swallows it). Cleanup must refuse and keep the resume
// file, so reuse after the worker is retired still validates.
test('cleanup keeps the resume file and fails when the prune record cannot be written', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-hydrate-prune-write-fail-'));
  try {
    const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES'); const hashOpt = sha256Hex('OPT-BYTES');
    const script = HARNESS + [
      "addDataset('album');",
      "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: 8 }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: 9 }, { name: 'optimizer.resume', sha256: '${hashOpt}', bytes: 9 }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES'), 'optimizer.resume': Buffer.from('OPT-BYTES') }, deleted: [] };`,
      "const workerServer = await serve(state);",
      "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
      "const app = await appServer();",
      "const base = 'http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/';",
      "const use = () => fetch(base + 'yue2-joint-preset', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ checkpointDir: dir }) });",
      "const rename = fs.renameSync;",
      "try {",
      "  const first = await use();",
      "  if (!first.ok) throw new Error('expected the first use to succeed: ' + JSON.stringify(await first.json()));",
      "  fs.renameSync = function (from, to) { if (String(to).endsWith('yue2-aitk-runs.json')) throw Object.assign(new Error('test: index write denied'), { code: 'EACCES' }); return rename.apply(this, arguments); };",
      "  const cleanup = await fetch(base + 'yue2-cleanup', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ run: 'remote:W:job1', step: 10, resume: true }) });",
      "  fs.renameSync = rename;",
      "  if (cleanup.ok) throw new Error('cleanup must fail when the prune cannot be recorded: ' + JSON.stringify(await cleanup.json()));",
      "  if (!fs.existsSync(path.join(dir, 'optimizer.resume'))) throw new Error('the resume file must be kept when its prune was not recorded');",
      "  state.manifest = []; state.files = {};",
      "  const second = await use();",
      "  if (!second.ok) throw new Error('expected reuse to revalidate locally: ' + JSON.stringify(await second.json()));",
      "} finally { fs.renameSync = rename; workerServer.close(); app.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

// Pulled runs keep readable folders: a finished one moves from staging to the
// local `<trigger>_<stamp>` name, taking a -2 suffix when a local run already
// owns that name, and every record that named its files follows it.
const FOLDERS = [
  "import { runStamp } from './src/services/training/adapterLayout.js';",
  "import { getOrCreateArtist, saveLyricsSet, upsertPreset, getAllPresets } from './src/db/lireekDb.js';",
  "import { readYue2Linked } from './src/services/training/lyricStudioExport.js';",
  "import { migrateYue2RemoteFolders } from './src/services/training/yue2Cleanup.js';",
  "import { setYue2RunFinished } from './src/services/training/yue2AitkRuns.js';",
  "const joint = path.join(config.aceServer.adapters, 'yue2-joint-adapters');",
  "const stamp = runStamp(new Date(now));",
  "function presetFor(ar, nar) { const artist = getOrCreateArtist('A'); const set = saveLyricsSet(artist.id, 'B', 1, []); upsertPreset(Number(set.id), { yue2ArAdapterPath: ar, yue2NarAdapterPath: nar }); return Number(set.id); }",
  "const presetOf = id => getAllPresets().find(p => p.lyrics_set_id === id);",
].join('');

function folderScript(body: string[]): string {
  const hashAr = sha256Hex('AR-BYTES'); const hashNar = sha256Hex('NAR-BYTES');
  return HARNESS + FOLDERS + [
    "addDataset('album');",
    "const dir = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
    `const state = { manifest: [{ name: 'native-ar.safetensors', sha256: '${hashAr}', bytes: 8 }, { name: 'native-nar.safetensors', sha256: '${hashNar}', bytes: 9 }], files: { 'native-ar.safetensors': Buffer.from('AR-BYTES'), 'native-nar.safetensors': Buffer.from('NAR-BYTES') }, deleted: [] };`,
    "const workerServer = await serve(state);",
    "config.workers.list = 'W=http://127.0.0.1:' + workerServer.address().port;",
    "const app = await appServer();",
    "const base = 'http://127.0.0.1:' + app.address().port + '/api/training/datasets/ds-album/';",
    "const post = (route, body) => fetch(base + route, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });",
    "try {",
    "  const first = await post('yue2-joint-preset', { checkpointDir: dir });",
    "  if (!first.ok) throw new Error('expected the first use to succeed: ' + JSON.stringify(await first.json()));",
    "  const setId = presetFor(path.join(dir, 'native-ar.safetensors'), path.join(dir, 'native-nar.safetensors'));",
    ...body,
    "} finally { workerServer.close(); app.close(); }",
  ].join('');
}

test('cleanup moves a finished pulled run to its local name, -2 on a collision, and repoints index, link and preset', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-remote-folder-move-'));
  try {
    runInIsolatedRoot(root, folderScript([
      // A local run already owns the plain name.
      "  fs.mkdirSync(path.join(joint, 'album_' + stamp), { recursive: true });",
      "  const cleanup = await post('yue2-cleanup', { run: 'remote:W:job1', step: 10 });",
      "  const body = await cleanup.json();",
      "  if (!cleanup.ok || body.moveError) throw new Error('cleanup failed: ' + JSON.stringify(body));",
      "  const want = path.join(joint, 'album_' + stamp + '-2');",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (run.output !== want) throw new Error('run not at its local name: ' + run.output);",
      "  if (fs.existsSync(path.dirname(dir))) throw new Error('the staging folder is still there');",
      "  if (!fs.existsSync(path.join(joint, 'album_' + stamp))) throw new Error('the local run that owned the name was touched');",
      "  const ckpt = run.checkpoints.find(c => c.step === 10);",
      "  if (ckpt.dir !== path.join(want, 'checkpoint-step10') || !ckpt.arPath) throw new Error('checkpoint not repointed: ' + JSON.stringify(ckpt));",
      "  const linked = readYue2Linked().album;",
      "  if (linked.arPath !== path.join(want, 'checkpoint-step10', 'native-ar.safetensors')) throw new Error('yue2-linked.json not repointed: ' + JSON.stringify(linked));",
      "  if (presetOf(setId).yue2_nar_adapter_path !== path.join(want, 'checkpoint-step10', 'native-nar.safetensors')) throw new Error('preset not repointed: ' + JSON.stringify(presetOf(setId)));",
      "  state.manifest = []; state.files = {};",
      "  const again = await post('yue2-joint-preset', { checkpointDir: ckpt.dir });",
      "  if (!again.ok) throw new Error('the moved rung no longer links: ' + JSON.stringify(await again.json()));",
    ]));
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a move whose link records cannot be rewritten puts the folder and the index back', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-remote-folder-move-fail-'));
  try {
    runInIsolatedRoot(root, folderScript([
      "  const linkedFile = path.join(process.env.TRAINING_DIR, 'yue2-linked.json');",
      "  const linkedBefore = fs.readFileSync(linkedFile, 'utf8');",
      "  fs.chmodSync(linkedFile, 0o444);",
      "  const cleanup = await post('yue2-cleanup', { run: 'remote:W:job1', step: 10 });",
      "  fs.chmodSync(linkedFile, 0o644);",
      "  const body = await cleanup.json();",
      "  if (!cleanup.ok || !body.moveError) throw new Error('expected a finished cleanup that reports the failed move: ' + JSON.stringify(body));",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (run.output !== path.dirname(dir) || !fs.existsSync(path.join(dir, 'native-ar.safetensors'))) throw new Error('the folder did not stay where it was: ' + run.output);",
      "  if (fs.existsSync(path.join(joint, 'album_' + stamp))) throw new Error('a half-moved folder was left behind');",
      "  if (fs.readFileSync(linkedFile, 'utf8') !== linkedBefore) throw new Error('yue2-linked.json changed');",
      "  if (presetOf(setId).yue2_ar_adapter_path !== path.join(dir, 'native-ar.safetensors')) throw new Error('preset changed');",
      // Run again later (here, via the migration) and it moves cleanly.
      "  const moved = migrateYue2RemoteFolders(true);",
      "  if (moved.length !== 1 || moved[0].error || moved[0].to !== path.join(joint, 'album_' + stamp)) throw new Error('retry did not move it: ' + JSON.stringify(moved));",
    ]));
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('the migration lists then moves old remote-<worker>-<id> folders: finished to the local name, the rest to staging', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-remote-folder-migrate-'));
  try {
    const script = HARNESS + FOLDERS + [
      "addDataset('album');",
      "const done = recordPulledRun('remote:W:job1', 'job1', 'W', 10);",
      "const open = recordPulledRun('remote:W:job2', 'job2', 'W', 20);",
      "for (const d of [done, open]) { fs.mkdirSync(d, { recursive: true }); fs.writeFileSync(path.join(d, 'native-ar.safetensors'), 'AR'); fs.writeFileSync(path.join(d, 'native-nar.safetensors'), 'NAR'); }",
      "setYue2RunFinished(path.dirname(done), { pickedStep: 10, pickedBlind: false, pickedLabel: '' });",
      "const dry = migrateYue2RemoteFolders(false);",
      "if (dry.length !== 2) throw new Error('dry run should list both: ' + JSON.stringify(dry));",
      "if (!fs.existsSync(done) || !fs.existsSync(open)) throw new Error('a dry run moved something');",
      "const finishedTo = path.join(joint, 'album_' + stamp), stagedTo = path.join(joint, '_remote', 'w', 'album_' + stamp);",
      "const byJob = Object.fromEntries(dry.map(r => [r.jobId, r]));",
      "if (byJob['remote:W:job1'].to !== finishedTo || byJob['remote:W:job2'].to !== stagedTo) throw new Error('unexpected targets: ' + JSON.stringify(dry));",
      "const applied = migrateYue2RemoteFolders(true);",
      "if (applied.some(r => r.error)) throw new Error('apply failed: ' + JSON.stringify(applied));",
      "if (!fs.existsSync(path.join(finishedTo, 'checkpoint-step10', 'native-ar.safetensors')) || !fs.existsSync(path.join(stagedTo, 'checkpoint-step20', 'native-nar.safetensors'))) throw new Error('files did not move');",
      "const outputs = listYue2AitkRuns('ds-album', 'album').map(r => r.output).sort();",
      "if (JSON.stringify(outputs) !== JSON.stringify([finishedTo, stagedTo].sort())) throw new Error('index not repointed: ' + JSON.stringify(outputs));",
      "if (migrateYue2RemoteFolders(false).length) throw new Error('a second run should find nothing to move');",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
