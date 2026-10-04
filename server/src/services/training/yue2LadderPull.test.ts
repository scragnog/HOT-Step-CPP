// Pulling a worker's YuE2 ladders into this machine's own index
// (trainingWorkers.ts's pullYue2Ladders), over a fake worker HTTP server.
// Every test runs in an isolated subprocess (own TRAINING_DIR/
// ACESTEPCPP_ADAPTERS/DATA_DIR) — config and the run index are module-level
// singletons, so tests sharing a process would see each other's state.
//
// Checksums are computed here, in this normal file, and spliced into each
// generated script as a literal hex string. Node's `--eval` auto-wraps any
// script whose source contains the bare word "crypto" in a non-module
// function that injects `require('node:crypto')` as a parameter — a
// convenience for one-liners that breaks outright on a script that also
// `import`s. Computing the hash out here sidesteps it entirely.
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

// Shared boilerplate: initDb, a dataset row, and a fake worker HTTP server
// standing in for /yue2-ladders and /yue2-ladder-file. `state.deleted` keeps
// recording any DELETE the puller sends (there is no such route any more —
// slice 1 never touches the worker's files — so every test can assert it
// stays empty as a regression guard). `state.ladders`/`state.files` are
// mutated by each test between pulls to simulate rungs or previews landing
// later. `preview(...)`'s sha256 is a literal hex string this file computed,
// never recomputed in-process.
const LADDER_HARNESS = [
  "import fs from 'node:fs';",
  "import path from 'node:path';",
  "import http from 'node:http';",
  "import { initDb } from './src/db/database.js';",
  "import * as repo from './src/services/training/datasetsRepo.js';",
  "import { pullYue2Ladders } from './src/services/training/trainingWorkers.js';",
  "import { listYue2AitkRuns, reconcileYue2AitkRunsAtStartup } from './src/services/training/yue2AitkRuns.js';",
  "import { listYue2JointPreviews, pruneYue2JointPreviews } from './src/services/training/yue2JointPreview.js';",
  "initDb();",
  "const now = Date.now();",
  "function addDataset(slug) { const iso = new Date(now).toISOString(); repo.insertDataset({ id: 'ds-' + slug, slug, name: slug, sourceDir: path.join(process.env.TRAINING_DIR, 'src-' + slug), recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: iso, updatedAt: iso }); }",
  "function serve(state) {",
  "  const server = http.createServer((req, res) => {",
  "    const u = new URL(req.url, 'http://x');",
  "    if (req.method === 'GET' && u.pathname === '/api/training/worker/yue2-ladders') { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ ladders: state.ladders })); return; }",
  "    if (req.method === 'GET' && u.pathname === '/api/training/worker/yue2-ladder-file') {",
  "      const key = u.searchParams.get('run') + '/' + u.searchParams.get('file');",
  "      const body = state.files[key];",
  "      if (!body) { res.statusCode = 404; res.end('{}'); return; }",
  "      if (body.truncate) { res.setHeader('content-length', String(body.buf.length)); res.write(body.buf.subarray(0, 3)); res.destroy(); return; }",
  "      res.end(body.buf); return;",
  "    }",
  "    if (req.method === 'DELETE') { state.deleted.push(u.pathname); res.setHeader('content-type', 'application/json'); res.end(JSON.stringify({ ok: true })); return; }",
  "    res.statusCode = 404; res.end();",
  "  });",
  "  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));",
  "}",
  "function preview(id, step, file, buf, sha256) { return { id, step, kind: 'artist', status: 'done', file, seconds: 30, seed: 1, previewMaxFrames: 100, createdAt: now, updatedAt: now, ...(sha256 !== undefined ? { sha256 } : {}), bytes: buf.length }; }",
].join('');

test('pulls a preview-only ladder: listable without weights, previews byte-identical, worker left untouched', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-basic-'));
  try {
    const hash = sha256Hex('WAV-BYTES-1');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const buf = Buffer.from('WAV-BYTES-1');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'done', createdAt: now, updatedAt: now, options: { method: 'base-matched' }, blindLabels: { '10': 'A' }, checkpoints: [{ step: 10, kl: 1.5, rung: true }], previews: [preview('p1', 10, 'p1.wav', buf, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled.length !== 1 || pulled[0].status !== 'pulled' || pulled[0].previewsFetched !== 1) throw new Error('unexpected pull result: ' + JSON.stringify(pulled));",
      "  const runs = listYue2AitkRuns('ds-album', 'album');",
      "  if (runs.length !== 1) throw new Error('expected one run, got ' + runs.length);",
      "  const run = runs[0];",
      "  if (run.jobId !== 'remote:W:job1') throw new Error('unexpected jobId: ' + run.jobId);",
      "  if (run.origin?.worker !== 'W' || run.origin?.remoteJobId !== 'job1') throw new Error('missing origin: ' + JSON.stringify(run.origin));",
      "  if (run.checkpoints.length !== 1 || run.checkpoints[0].availability !== 'remote' || run.checkpoints[0].arPath) throw new Error('unexpected checkpoints: ' + JSON.stringify(run.checkpoints));",
      "  const previews = listYue2JointPreviews(run.output);",
      "  if (previews.length !== 1 || !previews[0].file) throw new Error('preview not recorded: ' + JSON.stringify(previews));",
      "  if (!fs.readFileSync(path.join(run.output, 'previews', previews[0].file)).equals(buf)) throw new Error('preview bytes do not match');",
      "  if (state.deleted.length) throw new Error('slice 1 must never delete anything on the worker: ' + JSON.stringify(state.deleted));",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a later rung landing on the worker is added to the pulled ladder without losing the first one', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-later-rung-'));
  try {
    const hash10 = sha256Hex('WAV-10'); const hash20 = sha256Hex('WAV-20');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const buf10 = Buffer.from('WAV-10'); const buf20 = Buffer.from('WAV-20');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'running', createdAt: now, updatedAt: now, options: { method: 'base-matched' }, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p10', 10, 'p10.wav', buf10, '${hash10}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p10.wav': { buf: buf10 } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (state.deleted.length) throw new Error('a still-running ladder must never have its worker copies deleted');",
      "  ladder.checkpoints.push({ step: 20, kl: 1.8, rung: true });",
      `  ladder.previews.push(preview('p20', 20, 'p20.wav', buf20, '${hash20}'));`,
      "  state.files['job1/p20.wav'] = { buf: buf20 };",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled[0].previewsFetched !== 1) throw new Error('expected only the new preview to be fetched: ' + JSON.stringify(pulled));",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  const steps = run.checkpoints.map(c => c.step).sort((a, b) => a - b);",
      "  if (JSON.stringify(steps) !== JSON.stringify([10, 20])) throw new Error('expected both rungs, got ' + JSON.stringify(steps));",
      "  if (listYue2JointPreviews(run.output).filter(p => p.file).length !== 2) throw new Error('expected both previews on disk');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a repull after local cleanup never resurrects a pruned preview', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-repull-pruned-'));
  try {
    const hash = sha256Hex('WAV-BYTES-1');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const buf = Buffer.from('WAV-BYTES-1');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'running', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, 'p1.wav', buf, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  pruneYue2JointPreviews(run.output, -1);",
      "  const prunedBefore = listYue2JointPreviews(run.output);",
      "  if (prunedBefore[0].file) throw new Error('prune did not remove the file reference');",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled[0].previewsFetched !== 0) throw new Error('repull must not re-fetch a pruned preview: ' + JSON.stringify(pulled));",
      "  const after = listYue2JointPreviews(run.output);",
      "  if (after[0].file) throw new Error('repull restored a pruned preview file');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a byte flip with the correct size is rejected by its checksum, not silently accepted', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-corrupt-'));
  try {
    const hash = sha256Hex('WAV-BYTES-1'); // the hash of the ORIGINAL bytes; the worker serves a different, same-length buffer below
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const good = Buffer.from('WAV-BYTES-1'); const corrupt = Buffer.from('WAV-BYTES-2');",
      "if (good.length !== corrupt.length) throw new Error('fixture bug: buffers must be same size');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'done', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, 'p1.wav', good, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf: corrupt } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled[0].status !== 'partial' || !pulled[0].errors.length) throw new Error('expected a partial result with errors: ' + JSON.stringify(pulled));",
      "  if (state.deleted.length) throw new Error('no delete on a hash mismatch');",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (fs.existsSync(path.join(run.output, 'previews', 'p1.wav'))) throw new Error('a corrupt transfer must not land at its destination name');",
      "  const rec = listYue2JointPreviews(run.output)[0];",
      "  if (rec.file) throw new Error('a corrupt preview must not be published as available');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('an interrupted transfer is cleaned up, not renamed into place, and nothing is deleted on the worker', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-interrupted-'));
  try {
    const hash = sha256Hex('WAV-BYTES-LONGER-THAN-THREE');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const buf = Buffer.from('WAV-BYTES-LONGER-THAN-THREE');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'done', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, 'p1.wav', buf, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf, truncate: true } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled[0].status !== 'partial' || !pulled[0].errors.length) throw new Error('expected a partial result with errors: ' + JSON.stringify(pulled));",
      "  if (state.deleted.length) throw new Error('no delete on an interrupted transfer');",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  const left = fs.readdirSync(path.join(run.output, 'previews')).filter(f => f.endsWith('.part'));",
      "  if (left.length) throw new Error('a .part file was left behind: ' + JSON.stringify(left));",
      "  if (fs.existsSync(path.join(run.output, 'previews', 'p1.wav'))) throw new Error('a truncated transfer must not land at its destination name');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a pulled ladder still training on its worker survives this machine restarting', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-restart-'));
  try {
    const hash = sha256Hex('WAV-BYTES-1');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const buf = Buffer.from('WAV-BYTES-1');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'running', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, 'p1.wav', buf, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  const before = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (before.status !== 'running') throw new Error('expected running before restart, got ' + before.status);",
      "  const stale = reconcileYue2AitkRunsAtStartup();",
      "  if (stale !== 0) throw new Error('a pulled ladder must not count as a local job this process killed');",
      "  const after = listYue2AitkRuns('ds-album', 'album')[0];",
      "  if (after.status !== 'running') throw new Error('restart flipped a worker-owned run to interrupted: ' + after.status);",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a remote-reported preview filename cannot escape the previews directory', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-traversal-'));
  try {
    const buf = Buffer.from('EVIL');
    const hash = sha256Hex('EVIL');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'done', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, '../../escaped.json', Buffer.from('EVIL'), '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/../../escaped.json': { buf: Buffer.from('EVIL') } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled[0].status !== 'partial' || !pulled[0].errors.length) throw new Error('expected an unsafe filename to be refused with an error: ' + JSON.stringify(pulled));",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  const adaptersRoot = path.resolve(process.env.ACESTEPCPP_ADAPTERS);",
      "  if (fs.existsSync(path.join(adaptersRoot, 'escaped.json')) || fs.existsSync(path.join(path.dirname(adaptersRoot), 'escaped.json'))) throw new Error('a file landed outside the previews directory');",
      "  const rec = listYue2JointPreviews(run.output)[0];",
      "  if (rec.file) throw new Error('an unsafe filename must never be published as available');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a preview corrupted at rest between pulls is re-fetched and repaired, not trusted because it exists', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-repair-'));
  try {
    const hash = sha256Hex('WAV-BYTES-1');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const buf = Buffer.from('WAV-BYTES-1');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'done', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, 'p1.wav', buf, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  const dest = path.join(run.output, 'previews', 'p1.wav');",
      "  fs.writeFileSync(dest, Buffer.from('CORRUPTED-ON-DISK'));",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled[0].status !== 'pulled' || pulled[0].previewsFetched !== 1) throw new Error('expected the corrupted file to be refetched: ' + JSON.stringify(pulled));",
      "  if (!fs.readFileSync(dest).equals(buf)) throw new Error('the repaired file does not match the worker\\'s bytes');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('a repull with no checksum never verifies a byte-flipped local file just because it is present', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-no-checksum-'));
  try {
    const hash = sha256Hex('WAV-BYTES-1');
    const script = LADDER_HARNESS + [
      "addDataset('album');",
      "const buf = Buffer.from('WAV-BYTES-1');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-album', datasetSlug: 'album', status: 'done', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, 'p1.wav', buf, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  const run = listYue2AitkRuns('ds-album', 'album')[0];",
      "  const dest = path.join(run.output, 'previews', 'p1.wav');",
      "  fs.writeFileSync(dest, Buffer.from('CORRUPTED-ON-DISK'));",
      "  delete ladder.previews[0].sha256;", // the worker's manifest omits the checksum this round (e.g. a read failure there)
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled[0].status !== 'partial' || !pulled[0].errors.length) throw new Error('expected a partial result with errors: ' + JSON.stringify(pulled));",
      "  if (pulled[0].previewsFetched !== 0) throw new Error('no checksum means no fetch attempt: ' + JSON.stringify(pulled));",
      "  if (fs.readFileSync(dest).toString() !== 'CORRUPTED-ON-DISK') throw new Error('the corrupted file must not be silently left in place as if fine, nor repaired without a checksum to repair against');",
      "  const rec = listYue2JointPreviews(run.output)[0];",
      "  if (rec.file) throw new Error('a preview this machine cannot currently verify must not be published as available');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('no dataset for the worker\'s ladder is reported, not guessed at', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-ladder-pull-no-dataset-'));
  try {
    const hash = sha256Hex('WAV');
    const script = LADDER_HARNESS + [
      "const buf = Buffer.from('WAV');",
      `const ladder = { jobId: 'job1', datasetId: 'ds-missing', datasetSlug: 'missing', status: 'done', createdAt: now, updatedAt: now, options: {}, checkpoints: [{ step: 10, kl: 1.0, rung: true }], previews: [preview('p1', 10, 'p1.wav', buf, '${hash}')] };`,
      "const state = { ladders: [ladder], files: { 'job1/p1.wav': { buf } }, deleted: [] };",
      "const server = await serve(state);",
      "try {",
      "  const pulled = await pullYue2Ladders({ name: 'W', url: 'http://127.0.0.1:' + server.address().port });",
      "  if (pulled.length !== 1 || pulled[0].status !== 'no-dataset') throw new Error('unexpected result: ' + JSON.stringify(pulled));",
      "  if (listYue2AitkRuns('ds-missing', 'missing').length) throw new Error('a run was recorded with no matching dataset');",
      "} finally { server.close(); }",
    ].join('');
    runInIsolatedRoot(root, script);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
