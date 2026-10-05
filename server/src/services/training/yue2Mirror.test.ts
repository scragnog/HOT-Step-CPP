// The worker mirror (yue2Mirror.ts) against a fake worker HTTP server, plus
// the worker's own manifest and file route (trainingWorkers.ts). One process
// with its own TRAINING_DIR / ACESTEPCPP_ADAPTERS / DATA_DIR; each test uses
// its own worker name and folder, so they never see each other's runs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test, { after } from 'node:test';
import { createHash } from 'node:crypto';
import express from 'express';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-mirror-'));
process.env.DATA_DIR = path.join(root, 'data');
process.env.TRAINING_DIR = path.join(root, 'training');
process.env.ACESTEPCPP_ADAPTERS = path.join(root, 'adapters');
const { initDb, closeDb } = await import('../../db/database.js');
const repo = await import('./datasetsRepo.js');
const { syncYue2Mirror, deleteYue2Run, migrateYue2OriginRuns } = await import('./yue2Mirror.js');
const { listAllYue2AitkRuns, recordYue2AitkRun, setYue2RunFinished } = await import('./yue2AitkRuns.js');
const { listYue2JointPreviews } = await import('./yue2JointPreview.js');
const { workerRouter } = await import('../../routes/workers.js');
const { workerMirrorManifest } = await import('./trainingWorkers.js');
initDb();
after(() => { closeDb(); fs.rmSync(root, { recursive: true, force: true }); });

const joint = path.join(root, 'adapters', 'yue2-joint-adapters');
const sha = (b: Buffer) => createHash('sha256').update(b).digest('hex');
const iso = new Date().toISOString();
repo.insertDataset({ id: 'ds-album', slug: 'album', name: 'album', sourceDir: path.join(root, 'src'), recursive: true, customTag: '', tagPosition: 'prefix',
  genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0,
  status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: iso, updatedAt: iso } as any);

type FakeFile = { buf: Buffer; mtimeMs: number; corrupt?: number };
interface FakeRun {
  jobId: string; folder: string; status: 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';
  previewsBusy: boolean; files: Record<string, FakeFile>; previews: any[];
}
interface Fake { runs: FakeRun[]; gets: string[]; heads: string[]; deletes: string[]; deleteStatus: number; url: string; close: () => void }

async function fakeWorker(runs: FakeRun[]): Promise<Fake> {
  const fake = { runs, gets: [] as string[], heads: [] as string[], deletes: [] as string[], deleteStatus: 200 } as Fake;
  const server = http.createServer((req, res) => {
    const u = new URL(req.url!, 'http://x');
    if (req.method === 'GET' && u.pathname === '/api/training/worker/yue2-mirror') {
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ runs: fake.runs.map(r => ({
        jobId: r.jobId, datasetId: 'ds-album', datasetSlug: 'album', folder: r.folder, status: r.status, createdAt: 1, updatedAt: 2,
        options: { method: 'base-matched' }, previewsBusy: r.previewsBusy, previews: r.previews,
        files: Object.entries(r.files).map(([rel, f]) => ({ rel, size: f.buf.length, mtimeMs: f.mtimeMs })),
      })) }));
      return;
    }
    if ((req.method === 'GET' || req.method === 'HEAD') && u.pathname === '/api/training/worker/yue2-mirror-file') {
      const rel = u.searchParams.get('rel')!;
      const f = fake.runs.find(r => r.jobId === u.searchParams.get('jobId'))?.files[rel];
      (req.method === 'HEAD' ? fake.heads : fake.gets).push(rel);
      if (!f) { res.statusCode = 404; res.end('{}'); return; }
      res.setHeader('x-sha256', sha(f.buf));
      res.setHeader('x-mtime', String(f.mtimeMs));
      if (req.method === 'HEAD') { res.end(); return; }
      if (f.corrupt) { f.corrupt--; const bad = Buffer.from(f.buf); bad[0] ^= 0xff; res.end(bad); return; }
      res.end(f.buf);
      return;
    }
    if (req.method === 'DELETE' && u.pathname.startsWith('/api/training/worker/yue2-mirror/')) {
      const jobId = decodeURIComponent(u.pathname.split('/').pop()!);
      fake.deletes.push(jobId);
      res.statusCode = fake.deleteStatus;
      if (fake.deleteStatus === 200) fake.runs = fake.runs.filter(r => r.jobId !== jobId);
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify(fake.deleteStatus === 200 ? { ok: true } : { error: 'busy' }));
      return;
    }
    res.statusCode = 404; res.end();
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  fake.url = `http://127.0.0.1:${(server.address() as any).port}`;
  fake.close = () => server.close();
  return fake;
}

const file = (text: string, mtimeMs = 1000): FakeFile => ({ buf: Buffer.from(text), mtimeMs });
const preview = (id: string, step: number, name: string) => ({ id, step, kind: 'artist', status: 'done', file: name, seconds: 30, seed: 1, previewMaxFrames: 100, createdAt: 1, updatedAt: 1 });
function ladder(jobId: string, folder: string, status: FakeRun['status'] = 'running'): FakeRun {
  return { jobId, folder, status, previewsBusy: false, previews: [preview(`${jobId}-p10`, 10, 'step-10.wav')], files: {
    'train.jsonl': file('{"stage":"joint","step":1}\n'),
    'checkpoint-step10/native-ar.safetensors': file('AR-10'),
    'checkpoint-step10/native-nar.safetensors': file('NAR-10'),
    'checkpoint-step10/meters.json': file('{"kl_rung":true}'),
    'previews/step-10.wav': file('WAV-10'),
    'previews/step-10.score.abc': file('X:1'),
  } };
}
const localRun = (worker: string, jobId: string) => listAllYue2AitkRuns().find(r => r.origin?.worker === worker && r.origin.remoteJobId === jobId);
const sync = (name: string, fake: Fake) => syncYue2Mirror({ name, url: fake.url });

test('a new worker run appears here as an ordinary local run, under the worker jobId and folder name', async () => {
  const fake = await fakeWorker([ladder('job-new', 'album_2026-10-01_00-00-00')]);
  try {
    const status = await sync('NewW', fake);
    assert.equal(status.lastError, null);
    assert.equal(status.pendingFiles, 0);
    assert.equal(status.runs, 1);
    const run = localRun('NewW', 'job-new')!;
    assert.equal(run.jobId, 'job-new');
    assert.equal(run.output, path.join(joint, 'album_2026-10-01_00-00-00'));
    assert.equal(run.status, 'running');
    const ckpt = run.checkpoints.find(c => c.step === 10)!;
    assert.ok(ckpt.arPath && ckpt.narPath && ckpt.rung);
    assert.equal(fs.readFileSync(ckpt.arPath, 'utf8'), 'AR-10');
    const previews = listYue2JointPreviews(run.output);
    assert.equal(previews.length, 1);
    assert.equal(previews[0].status, 'done');
    assert.equal(fs.readFileSync(path.join(run.output, 'previews', 'step-10.score.abc'), 'utf8'), 'X:1');
    assert.deepEqual(fake.deletes, [], 'a running run is never deleted on the worker');
  } finally { fake.close(); }
});

test('a growing train.jsonl is fetched again; a finished checkpoint is fetched once', async () => {
  const run = ladder('job-grow', 'album_grow');
  const fake = await fakeWorker([run]);
  try {
    await sync('GrowW', fake);
    run.files['train.jsonl'] = file('{"stage":"joint","step":1}\n{"stage":"joint","step":2}\n', 2000);
    await sync('GrowW', fake);
    await sync('GrowW', fake);
    const count = (rel: string) => fake.gets.filter(g => g === rel).length;
    assert.equal(count('train.jsonl'), 2);
    assert.equal(count('checkpoint-step10/native-ar.safetensors'), 1);
    assert.equal(count('previews/step-10.wav'), 1);
    assert.match(fs.readFileSync(path.join(localRun('GrowW', 'job-grow')!.output, 'train.jsonl'), 'utf8'), /"step":2/);
  } finally { fake.close(); }
});

test('a done preview whose audio has not landed yet shows as rendering, not failed', async () => {
  const run = ladder('job-pending', 'album_pending');
  run.files['previews/step-10.wav'].corrupt = 2;
  const fake = await fakeWorker([run]);
  try {
    const status = await sync('PendW', fake);
    assert.match(status.lastError ?? '', /step-10\.wav: checksum mismatch/);
    assert.equal(status.pendingFiles, 1);
    const output = localRun('PendW', 'job-pending')!.output;
    assert.equal(listYue2JointPreviews(output)[0].status, 'rendering');
    assert.ok(!fs.existsSync(path.join(output, 'previews', 'step-10.wav')));
    assert.ok(!fs.readdirSync(path.join(output, 'previews')).some(f => f.includes('.part')), 'no .part left behind');
  } finally { fake.close(); }
});

test('a checksum mismatch is retried once and lands; twice is reported and blocks the worker delete', async () => {
  const once = ladder('job-once', 'album_once', 'done');
  once.files['checkpoint-step10/native-nar.safetensors'].corrupt = 1;
  const twice = ladder('job-twice', 'album_twice', 'done');
  twice.files['checkpoint-step10/native-nar.safetensors'].corrupt = 2;
  const fake = await fakeWorker([once, twice]);
  try {
    const status = await sync('HashW', fake);
    assert.equal(fs.readFileSync(path.join(joint, 'album_once', 'checkpoint-step10', 'native-nar.safetensors'), 'utf8'), 'NAR-10');
    assert.match(status.lastError ?? '', /album_twice\/checkpoint-step10\/native-nar\.safetensors: checksum mismatch/);
    assert.ok(!fs.existsSync(path.join(joint, 'album_twice', 'checkpoint-step10', 'native-nar.safetensors')));
    assert.deepEqual(fake.deletes, ['job-once']);
  } finally { fake.close(); }
});

test('a path outside the mirrored set, such as optimizer.resume, is refused and never requested', async () => {
  const run = ladder('job-opt', 'album_opt');
  run.files['checkpoint-step10/optimizer.resume'] = file('OPT');
  run.files['checkpoint-step10/adapter.safetensors'] = file('ADA');
  run.files['../escape.wav'] = file('BAD');
  const fake = await fakeWorker([run]);
  try {
    const status = await sync('OptW', fake);
    assert.match(status.lastError ?? '', /refused checkpoint-step10\/optimizer\.resume/);
    assert.ok(!fake.gets.some(g => /optimizer|adapter\.safetensors|escape/.test(g)));
    assert.ok(!fs.existsSync(path.join(joint, 'escape.wav')));
  } finally { fake.close(); }
});

test('terminal, idle and complete: the worker copy is deleted; interrupted or previews busy: it is kept', async () => {
  const done = ladder('job-done', 'album_done', 'done');
  const interrupted = ladder('job-int', 'album_int', 'interrupted');
  const busy = ladder('job-busy', 'album_busy', 'cancelled');
  busy.previewsBusy = true;
  const fake = await fakeWorker([done, interrupted, busy]);
  try {
    await sync('TermW', fake);
    assert.deepEqual(fake.deletes, ['job-done']);
    assert.ok(localRun('TermW', 'job-done'), 'the local copy stays');
    busy.previewsBusy = false;
    fake.deleteStatus = 409;
    await sync('TermW', fake);
    assert.deepEqual(fake.deletes, ['job-done', 'job-busy'], 'a refused delete is asked again later');
    fake.deleteStatus = 200;
    await sync('TermW', fake);
    assert.deepEqual(fake.deletes, ['job-done', 'job-busy', 'job-busy']);
    assert.ok(!fake.deletes.includes('job-int'));
  } finally { fake.close(); }
});

test('a run finished here is never fetched again, and its worker copy is dropped', async () => {
  const run = ladder('job-fin', 'album_fin', 'done');
  const fake = await fakeWorker([run]);
  fake.deleteStatus = 409;
  try {
    await sync('FinW', fake);
    const output = localRun('FinW', 'job-fin')!.output;
    setYue2RunFinished(output, { pickedStep: 10, pickedBlind: false, pickedLabel: '' });
    fs.rmSync(path.join(output, 'previews', 'step-10.wav'));
    const before = fake.gets.length;
    await sync('FinW', fake);
    assert.equal(fake.gets.length, before);
    assert.ok(!fs.existsSync(path.join(output, 'previews', 'step-10.wav')));
    assert.equal(fake.deletes.length, 2);
  } finally { fake.close(); }
});

test('a mirrored run discarded here is tombstoned: no pass recreates it, and the worker delete is retried', async () => {
  const run = ladder('job-tomb', 'album_tomb', 'done');
  const fake = await fakeWorker([run]);
  fake.deleteStatus = 409;
  try {
    await sync('TombW', fake);
    const local = localRun('TombW', 'job-tomb')!;
    await deleteYue2Run(local.jobId);
    assert.ok(!fs.existsSync(local.output));
    const before = fake.gets.length;
    await sync('TombW', fake);
    assert.equal(localRun('TombW', 'job-tomb'), undefined);
    assert.ok(!fs.existsSync(local.output));
    assert.equal(fake.gets.length, before);
    assert.ok(fake.deletes.filter(d => d === 'job-tomb').length >= 2);
  } finally { fake.close(); }
});

test('the migration moves a pulled run out of _remote, keeps its jobId, and the mirror then fills it in', async () => {
  const staged = path.join(joint, '_remote', 'migw', 'album_mig');
  fs.mkdirSync(path.join(staged, 'previews'), { recursive: true });
  fs.mkdirSync(path.join(staged, 'checkpoint-step10'), { recursive: true });
  fs.writeFileSync(path.join(staged, 'previews', 'step-10.wav'), 'WAV-10');
  fs.writeFileSync(path.join(staged, 'checkpoint-step10', 'native-ar.safetensors'), 'STALE');
  // Temp files a restart cut off mid-download or mid-ledger-write.
  const orphans = ['checkpoint-step10/native-nar.safetensors.part-0b5f5a0e-4a3c-4e6b-9d1f-2b8f9c7e6a51',
    '.mirror.json.6c1d2e3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f.tmp'];
  for (const rel of orphans) fs.writeFileSync(path.join(staged, ...rel.split('/')), 'partial');
  recordYue2AitkRun({ version: 1, jobId: 'remote:MigW:job-mig', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output: staged,
    options: {}, status: 'running', createdAt: 1, updatedAt: 1, checkpoints: [{ step: 10, dir: path.join(staged, 'checkpoint-step10'), availability: 'remote' } as any],
    origin: { worker: 'MigW', remoteJobId: 'job-mig' } });
  await migrateYue2OriginRuns();
  const moved = listAllYue2AitkRuns().find(r => r.jobId === 'remote:MigW:job-mig')!;
  assert.equal(moved.output, path.join(joint, 'album_mig'));
  assert.ok(!fs.existsSync(path.join(joint, '_remote')), 'the empty staging tree is gone');
  const index = JSON.parse(fs.readFileSync(path.join(root, 'training', 'yue2-aitk-runs.json'), 'utf8')) as any[];
  assert.ok(!JSON.stringify(index.find(r => r.jobId === 'remote:MigW:job-mig')).includes('availability'));
  assert.deepEqual((await migrateYue2OriginRuns()).moved, 0, 'idempotent');
  const fake = await fakeWorker([ladder('job-mig', 'album_mig_on_worker')]);
  try {
    await sync('MigW', fake);
    const run = listAllYue2AitkRuns().find(r => r.jobId === 'remote:MigW:job-mig')!;
    assert.equal(run.output, path.join(joint, 'album_mig'));
    assert.ok(run.checkpoints.find(c => c.step === 10)?.arPath);
    assert.ok(!fs.existsSync(path.join(joint, 'album_mig_on_worker')));
    assert.ok(!fake.gets.includes('previews/step-10.wav'), 'a file already here and matching the worker is adopted, not downloaded');
    assert.ok(fake.heads.includes('previews/step-10.wav'));
    assert.equal(fs.readFileSync(path.join(run.output, 'checkpoint-step10', 'native-ar.safetensors'), 'utf8'), 'AR-10', 'a different file of the same size is fetched');
    assert.equal(listYue2JointPreviews(run.output)[0].status, 'done');
    for (const rel of orphans) assert.ok(!fs.existsSync(path.join(run.output, ...rel.split('/'))), `${rel} swept`);
    const heads = fake.heads.length;
    await sync('MigW', fake);
    assert.equal(fake.heads.length, heads, 'an adopted file is checked once, then trusted from the ledger');
  } finally { fake.close(); }
});

test('the worker manifest lists weights, meters, logs and previews but never the optimizer or the combined adapter; the file route sends a checksum', async () => {
  const output = path.join(joint, 'album_worker_side');
  const ckpt = path.join(output, 'checkpoint-step10');
  fs.mkdirSync(path.join(output, 'previews'), { recursive: true });
  fs.mkdirSync(ckpt, { recursive: true });
  for (const name of ['native-ar.safetensors', 'native-nar.safetensors', 'meters.json', 'optimizer.resume', 'adapter.safetensors']) fs.writeFileSync(path.join(ckpt, name), name);
  fs.writeFileSync(path.join(output, 'train.jsonl'), '{}\n');
  fs.writeFileSync(path.join(output, 'previews', 'step-10.wav'), 'WAV');
  fs.writeFileSync(path.join(output, 'previews', 'index.json'), JSON.stringify([preview('wp', 10, 'step-10.wav')]));
  // A render a worker restart cut off: its record says 'rendering' for good.
  fs.writeFileSync(path.join(output, 'previews', 'index.json'), JSON.stringify([preview('wp', 10, 'step-10.wav'),
    { ...preview('stale', 20, 'step-20.wav'), status: 'rendering', file: undefined }]));
  recordYue2AitkRun({ version: 1, jobId: 'worker-side', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output, options: {},
    status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });
  const shipped = path.join(joint, 'album_worker_shipped');
  fs.mkdirSync(shipped, { recursive: true });
  setYue2RunFinished(shipped, { pickedStep: 10, pickedBlind: false, pickedLabel: '' });
  recordYue2AitkRun({ version: 1, jobId: 'worker-shipped', datasetId: 'ds-album', datasetSlug: 'album', method: 'aitk', output: shipped, options: {},
    status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });
  const manifest = workerMirrorManifest();
  assert.ok(!manifest.some(r => r.jobId === 'worker-shipped'), 'a run finished on the worker is never mirrored');
  const entry = manifest.find(r => r.jobId === 'worker-side')!;
  assert.deepEqual(entry.files.map(f => f.rel).sort(), ['checkpoint-step10/meters.json', 'checkpoint-step10/native-ar.safetensors',
    'checkpoint-step10/native-nar.safetensors', 'previews/step-10.wav', 'train.jsonl']);
  assert.equal(entry.previewsBusy, false, 'a rendering record with no render alive does not hold the run');
  assert.equal(entry.previews.find(p => p.id === 'stale')!.status, 'failed');
  const app = express();
  app.use('/w', workerRouter);
  const server = await new Promise<http.Server>(resolve => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  try {
    const base = `http://127.0.0.1:${(server.address() as any).port}/w/yue2-mirror-file?jobId=worker-side&rel=`;
    const ok = await fetch(base + encodeURIComponent('checkpoint-step10/native-ar.safetensors'));
    assert.equal(ok.status, 200);
    assert.equal(ok.headers.get('x-sha256'), sha(Buffer.from('native-ar.safetensors')));
    assert.equal(await ok.text(), 'native-ar.safetensors');
    assert.equal((await fetch(base + encodeURIComponent('checkpoint-step10/optimizer.resume'))).status, 404);
    assert.equal((await fetch(base + encodeURIComponent('../album/../album_worker_side/train.jsonl'))).status, 404);
  } finally { server.close(); }
});
