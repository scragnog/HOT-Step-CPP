// MM3 on a training worker: the dispatch's pre-push checks, the worker's MM3
// manifest (trainingWorkers.ts), and the controller copying runs back
// (yue2Mirror.ts, mirrorMm3Runs) against a fake worker. One process with its
// own TRAINING_DIR / ACESTEPCPP_ADAPTERS / DATA_DIR.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test, { after } from 'node:test';
import { createHash } from 'node:crypto';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-worker-'));
process.env.DATA_DIR = path.join(root, 'data');
process.env.TRAINING_DIR = path.join(root, 'training');
process.env.ACESTEPCPP_ADAPTERS = path.join(root, 'adapters');
const { initDb, closeDb } = await import('../../db/database.js');
const repo = await import('./datasetsRepo.js');
const { syncYue2Mirror } = await import('./yue2Mirror.js');
const { listMm3Runs, mm3AdapterRoot } = await import('./mm3Runs.js');
const { getDispatch, startDispatch, workerDatasetsDir, workerMm3MirrorFile, workerMm3MirrorManifest } = await import('./trainingWorkers.js');
initDb();
after(() => { closeDb(); fs.rmSync(root, { recursive: true, force: true }); });

const iso = new Date().toISOString();
function dataset(id: string, slug: string, sourceDir: string, extra: Record<string, unknown> = {}) {
  repo.insertDataset({ id, slug, name: slug, sourceDir, recursive: true, customTag: '', tagPosition: 'prefix',
    genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '', sampleCount: 0, labeledCount: 0, excludedCount: 0,
    status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: iso, updatedAt: iso, ...extra } as any);
}
const manifest = (run: string, datasetId: string, extra: Record<string, unknown> = {}) => JSON.stringify({
  version: 1, runName: run, datasetId, datasetSlug: 'album', datasetName: 'album', createdAt: 1, updatedAt: 2, launches: 1,
  options: { steps: 300 }, ...extra });
const sha = (b: Buffer) => createHash('sha256').update(b).digest('hex');
const until = async (ok: () => boolean) => { for (let i = 0; i < 200 && !ok(); i++) await new Promise(r => setTimeout(r, 10)); };

dataset('ds-album', 'album', path.join(root, 'src', 'album'));

// ── Controller: copying an MM3 run back ─────────────────────────────────────

type FakeFile = { buf: Buffer; mtimeMs: number };
interface FakeMm3 { run: string; datasetId: string; live: boolean; files: Record<string, FakeFile> }

async function fakeWorker(runs: FakeMm3[], hold?: (rel: string, res: http.ServerResponse, buf: Buffer) => boolean) {
  const gets: string[] = [];
  const server = http.createServer((req, res) => {
    const u = new URL(req.url!, 'http://x');
    res.setHeader('content-type', 'application/json');
    if (u.pathname === '/api/training/worker/yue2-mirror') { res.end('{"runs":[]}'); return; }
    if (u.pathname === '/api/training/worker/mm3-mirror') {
      res.end(JSON.stringify({ runs: runs.map(r => ({ run: r.run, datasetId: r.datasetId, live: r.live,
        files: Object.entries(r.files).map(([rel, f]) => ({ rel, size: f.buf.length, mtimeMs: f.mtimeMs })) })) }));
      return;
    }
    if (u.pathname === '/api/training/worker/mm3-mirror-file') {
      const rel = u.searchParams.get('rel')!;
      const f = runs.find(r => r.run === u.searchParams.get('run'))?.files[rel];
      gets.push(rel);
      if (!f) { res.statusCode = 404; res.end('{}'); return; }
      res.setHeader('content-type', 'application/octet-stream');
      res.setHeader('x-sha256', sha(f.buf));
      res.setHeader('x-mtime', String(f.mtimeMs));
      if (!hold?.(rel, res, f.buf)) res.end(f.buf);
      return;
    }
    res.statusCode = 404; res.end('{}');
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${(server.address() as any).port}`;
  return { gets, url, close: () => server.close() };
}
const file = (text: string, mtimeMs = 1000): FakeFile => ({ buf: Buffer.from(text), mtimeMs });

test('an MM3 worker run lands under mm3-lm-adapters as an ordinary run, and local cleanups stick', async () => {
  const run: FakeMm3 = { run: 'album-2026-11-08_10-00-00', datasetId: 'ds-album', live: true, files: {
    'hotstep-run.json': file(manifest('album-2026-11-08_10-00-00', 'ds-album')),
    'train-log.jsonl': file('{"type":"step","step":1,"loss":2.5}\n'),
    'ckpt-100/adapter_config.json': file('{}'),
    'ckpt-100/adapter_model.safetensors': file('W-100'),
    'ckpt-100/adapter_model.safetensors.json': file('{"name":"album ckpt-100"}'),
    'previews/step-000100-artist.wav': file('WAV'),
  } };
  const fake = await fakeWorker([run]);
  const w = { name: 'Mm3W', url: fake.url };
  try {
    const first = await syncYue2Mirror(w);
    assert.equal(first.lastError, null);
    assert.equal(first.pendingFiles, 0);
    const dir = path.join(mm3AdapterRoot(), run.run);
    const listed = listMm3Runs('ds-album', 'album');
    assert.deepEqual(listed.map(r => r.runName), [run.run]);
    assert.deepEqual(listed[0].checkpoints.map(c => c.step), [100]);
    assert.equal(fs.readFileSync(path.join(dir, 'previews', 'step-000100-artist.wav'), 'utf8'), 'WAV');

    // Nothing changed on the worker: nothing is fetched again.
    fake.gets.length = 0;
    await syncYue2Mirror(w);
    assert.deepEqual(fake.gets, []);

    // A ladder review here gave a blind letter; the worker then rewrote its
    // manifest and the log grew. Both come down, and the letter survives.
    const local = JSON.parse(fs.readFileSync(path.join(dir, 'hotstep-run.json'), 'utf8'));
    fs.writeFileSync(path.join(dir, 'hotstep-run.json'), JSON.stringify({ ...local, blindLabels: { 100: 'C' } }));
    run.files['hotstep-run.json'] = file(manifest(run.run, 'ds-album', { updatedAt: 3 }), 2000);
    run.files['train-log.jsonl'] = file('{"type":"step","step":1,"loss":2.5}\n{"type":"done"}\n', 2000);
    // A cleanup here removed the checkpoint: it stays removed.
    fs.rmSync(path.join(dir, 'ckpt-100'), { recursive: true });
    run.live = false;
    fake.gets.length = 0;
    const done = await syncYue2Mirror(w);
    assert.equal(done.lastError, null);
    assert.deepEqual(fake.gets.sort(), ['hotstep-run.json', 'train-log.jsonl']);
    const merged = JSON.parse(fs.readFileSync(path.join(dir, 'hotstep-run.json'), 'utf8'));
    assert.equal(merged.updatedAt, 3);
    assert.deepEqual(merged.blindLabels, { 100: 'C' });
    assert.equal(fs.existsSync(path.join(dir, 'ckpt-100')), false);

    // The whole run deleted here: no later pass brings it back.
    fs.rmSync(dir, { recursive: true });
    await syncYue2Mirror(w);
    assert.equal(fs.existsSync(dir), false);
  } finally { fake.close(); }
});

test('an MM3 worker run whose name a run trained here already has is refused, not merged', async () => {
  const name = 'album-2026-11-08_11-00-00';
  fs.mkdirSync(path.join(mm3AdapterRoot(), name), { recursive: true });
  fs.writeFileSync(path.join(mm3AdapterRoot(), name, 'hotstep-run.json'), 'LOCAL');
  const fake = await fakeWorker([{ run: name, datasetId: 'ds-album', live: false, files: { 'hotstep-run.json': file(manifest(name, 'ds-album')) } }]);
  try {
    const s = await syncYue2Mirror({ name: 'ClashW', url: fake.url });
    assert.match(s.lastError ?? '', /trained here/);
    assert.equal(fs.readFileSync(path.join(mm3AdapterRoot(), name, 'hotstep-run.json'), 'utf8'), 'LOCAL');
  } finally { fake.close(); }
});

// ── Worker: what it offers ──────────────────────────────────────────────────

test('the worker offers MM3 runs of pushed datasets only: finished checkpoints, never the optimizer state', async () => {
  dataset('ds-pushed', 'pushed', path.join(workerDatasetsDir(), 'pushed'));
  dataset('ds-own', 'own', path.join(root, 'src', 'own'));
  const mk = (run: string, datasetId: string, files: Record<string, string>) => {
    const dir = path.join(mm3AdapterRoot(), run);
    for (const [rel, text] of Object.entries({ 'hotstep-run.json': manifest(run, datasetId), ...files })) {
      fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true });
      fs.writeFileSync(path.join(dir, rel), text);
    }
  };
  mk('pushed-2026-11-08_12-00-00', 'ds-pushed', {
    'train-log.jsonl': '{}', 'resume-state.bin': 'OPT', 'resume-state.json': '{}',
    'ckpt-100/adapter_model.safetensors': 'W', 'ckpt-100/adapter_model.safetensors.json': '{}',
    'ckpt-200/adapter_model.safetensors': 'half-written',
    'previews/step-000100-artist.wav': 'WAV', 'previews/step-000100-artist.wav.json': '{}',
  });
  mk('own-2026-11-08_12-00-00', 'ds-own', { 'train-log.jsonl': '{}' });
  const offered = workerMm3MirrorManifest();
  assert.deepEqual(offered.map(r => r.run), ['pushed-2026-11-08_12-00-00']);
  assert.deepEqual(offered[0].files.map(f => f.rel).sort(), [
    'ckpt-100/adapter_model.safetensors', 'ckpt-100/adapter_model.safetensors.json', 'hotstep-run.json',
    'previews/step-000100-artist.wav', 'previews/step-000100-artist.wav.json', 'train-log.jsonl',
  ]);
  const served = await workerMm3MirrorFile('pushed-2026-11-08_12-00-00', 'ckpt-100/adapter_model.safetensors');
  assert.equal(served.sha256, sha(Buffer.from('W')));
  await assert.rejects(workerMm3MirrorFile('pushed-2026-11-08_12-00-00', 'resume-state.bin'), /Not a mirrored file/);
  await assert.rejects(workerMm3MirrorFile('own-2026-11-08_12-00-00', 'train-log.jsonl'), /Run not found/);
  await assert.rejects(workerMm3MirrorFile('..', 'hotstep-run.json'), /Run not found/);
});

// ── Dispatch: checked here before any audio is sent ─────────────────────────

test('an MM3 dispatch refuses an unbuilt or uncaptioned dataset before contacting the worker, and never mixes families', async () => {
  // A worker that accepts connections and never answers: anything that reaches it stays "pushing".
  const held: http.ServerResponse[] = [];
  const server = http.createServer((_req, res) => { held.push(res); });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const w = { name: 'DispatchW', url: `http://127.0.0.1:${(server.address() as any).port}` };
  try {
    const built = (slug: string, captioned: boolean) => {
      const dir = path.join(root, 'src', slug);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, 'dataset.json'), JSON.stringify({ samples: [{ filename: 'one.flac' }] }));
      if (captioned) fs.writeFileSync(path.join(dir, 'one.mm3.txt'), 'caption');
      dataset(`ds-${slug}`, slug, dir, { builtAt: iso, datasetJsonPath: path.join(dir, 'dataset.json') });
    };
    dataset('ds-unbuilt', 'unbuilt', path.join(root, 'src', 'unbuilt'));
    built('nocaps', false);
    built('ready', true);

    startDispatch(w, { family: 'mm3', datasetIds: ['ds-unbuilt', 'ds-nocaps'], lyricTiming: false, recipe: {} });
    await until(() => getDispatch('DispatchW')?.running === false);
    const items = getDispatch('DispatchW')!.items;
    assert.match(items[0].error ?? '', /Build the dataset first/);
    assert.match(items[1].error ?? '', /MiniMax-Music3 caption/);
    assert.equal(held.length, 0, 'nothing reached the worker');

    const live = startDispatch(w, { family: 'mm3', datasetIds: ['ds-ready'], lyricTiming: false, recipe: {} });
    assert.ok(!('error' in live) && live.family === 'mm3');
    await until(() => held.length > 0);
    const mixed = startDispatch(w, { family: 'yue2', datasetIds: ['ds-ready'], lyricTiming: true, recipe: {} });
    assert.match('error' in mixed ? mixed.error : '', /MiniMax-Music3 dispatch to DispatchW is still sending/);
  } finally {
    for (const res of held) res.destroy();
    server.closeAllConnections();
    server.close();
  }
});

// HOTSTEP-1078: deleting a mirrored MM3 run here (the Review page, Previous
// runs, the ladder) drops its folder and its mirror record, waits out a pass,
// and no later pass brings it back. The Review page lists the run first.
test('a mirrored MM3 run shows on the Review page with its worker, and deleting it here sticks', async () => {
  const { deleteMm3RunFolder, mm3RunOrigin } = await import('./yue2Mirror.js');
  const { mm3ReviewRows } = await import('./mm3Ladder.js');
  const run: FakeMm3 = { run: 'album-2026-11-08_12-00-00', datasetId: 'ds-album', live: false, files: {
    'hotstep-run.json': file(manifest('album-2026-11-08_12-00-00', 'ds-album')),
    'train-log.jsonl': file('{"type":"step","step":1,"loss":2.5}\n{"type":"done"}\n'),
    'ckpt-100/adapter_model.safetensors': file('W-100'),
    'previews/step-000100-artist.wav': file('WAV'),
  } };
  const fake = await fakeWorker([run]);
  const w = { name: 'DelW', url: fake.url };
  try {
    await syncYue2Mirror(w);
    const dir = path.join(mm3AdapterRoot(), run.run);
    assert.equal(mm3RunOrigin(run.run), 'DelW');
    const row = mm3ReviewRows({ id: 'ds-album', slug: 'album', name: 'album' }).find(r => r.refineRun === run.run);
    assert.ok(row);
    assert.equal(row.family, 'mm3');
    assert.equal(row.origin, 'DelW');
    assert.equal(row.rungs, 1);
    assert.equal(row.unscored, 1);
    assert.equal(row.best, null);

    // Outside the adapters root, or under another run's name: refused, nothing removed.
    await assert.rejects(deleteMm3RunFolder(run.run, path.join(root, 'src')), /outside the MM3 adapters folder/);
    await assert.rejects(deleteMm3RunFolder('other', dir), /outside the MM3 adapters folder/);
    assert.equal(fs.existsSync(dir), true);

    await Promise.all([syncYue2Mirror(w), deleteMm3RunFolder(run.run, dir)]);
    assert.equal(fs.existsSync(dir), false);
    assert.equal(mm3RunOrigin(run.run), null);
    run.files['train-log.jsonl'] = file('{"type":"done"}\n', 3000);
    await syncYue2Mirror(w);
    assert.equal(fs.existsSync(dir), false);
    assert.equal(mm3ReviewRows({ id: 'ds-album', slug: 'album', name: 'album' }).some(r => r.refineRun === run.run), false);
  } finally { fake.close(); }
});

// From review (HOTSTEP-1078): a pass and a delete racing on one run, and a
// delete during a ladder render.
test('a delete during the final mirror pass wins: the pass does not link the deleted checkpoint', async () => {
  const { deleteMm3Run } = await import('./mm3Ladder.js');
  const { getOrCreateArtist, saveLyricsSet, upsertPreset, getPreset } = await import('../../db/lireekDb.js');
  const artist = getOrCreateArtist('fixture');
  const lyrics = saveLyricsSet(Number(artist.id), 'fixture', 0, []);
  dataset('ds-race', 'race', path.join(root, 'src', 'race'), { lyricsSetId: Number(lyrics.id) });
  upsertPreset(Number(lyrics.id), { mm3AdapterPath: 'race-old/ckpt-100/adapter_model.safetensors' });
  const name = 'race-2026-11-08_13-00-00';
  const run: FakeMm3 = { run: name, datasetId: 'ds-race', live: true, files: {
    'hotstep-run.json': file(manifest(name, 'ds-race', { datasetSlug: 'race' })),
    'train-log.jsonl': file('{"type":"step","step":100,"loss":2.5}\n'),
    'ckpt-100/adapter_model.safetensors': file('W-100'),
  }};
  let release: (() => void) | undefined;
  let entered: (() => void) | undefined;
  const reached = new Promise<void>(r => { entered = r; });
  let holding = false;
  const fake = await fakeWorker([run], (rel, res, buf) => {
    if (holding && rel === 'train-log.jsonl') {
      release = () => res.end(buf); entered!(); return true;
    }
    return false;
  });
  const w = { name: 'RaceW', url: fake.url };
  try {
    await syncYue2Mirror(w);
    run.live = false;
    run.files['train-log.jsonl'] = file('{"type":"step","step":100,"loss":2.5}\n{"type":"done"}\n', 2000);
    holding = true;
    const pass = syncYue2Mirror(w);
    await reached;
    const removal = deleteMm3Run('ds-race', name);
    release!();
    await Promise.all([pass, removal]);
    const dir = path.join(mm3AdapterRoot(), name);
    assert.equal(fs.existsSync(dir), false);
    const preset = getPreset(Number(lyrics.id))!;
    assert.equal(preset.mm3_adapter_path, 'race-old/ckpt-100/adapter_model.safetensors');
  } finally { release?.(); fake.close(); }
});

test('a delete is refused while an on-demand preview holds the GPU', async () => {
  const { deleteMm3Run } = await import('./mm3Ladder.js');
  const { reserveGpuForPreview, runReservedPreview } = await import('./labelingQueue.js');
  const name = 'preview-2026-11-08_14-00-00';
  dataset('ds-preview', 'preview', path.join(root, 'src', 'preview'));
  const dir = path.join(mm3AdapterRoot(), name);
  fs.mkdirSync(path.join(dir, 'ckpt-100'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'hotstep-run.json'), manifest(name, 'ds-preview', { datasetSlug: 'preview' }));
  fs.writeFileSync(path.join(dir, 'ckpt-100', 'adapter_model.safetensors'), 'W');
  assert.equal(reserveGpuForPreview(), true);
  try { await assert.rejects(deleteMm3Run('ds-preview', name), /preview|busy/i); }
  finally { await runReservedPreview(async () => {}); }
});

test('cleanup revalidates the linked checkpoint after waiting for a finishing mirror pass', async () => {
  await import('./mm3Ladder.js');
  const { trainingOperationsRouter } = await import('./operations.js');
  const { withReviewMutation } = await import('./review/operations.js');
  const { getOrCreateArtist, saveLyricsSet, upsertPreset, getPreset } = await import('../../db/lireekDb.js');
  const { default: express } = await import('express');
  const artist = getOrCreateArtist('fixture');
  const lyrics = saveLyricsSet(Number(artist.id), 'fixture', 0, []);
  dataset('ds-cleanrace', 'cleanrace', path.join(root, 'src', 'cleanrace'));
  const name = 'cleanrace-2026-11-08_15-00-00';
  const ref = (step: number) => name + '/ckpt-' + step + '/adapter_model.safetensors';
  const run: FakeMm3 = { run: name, datasetId: 'ds-cleanrace', live: true, files: {
    'hotstep-run.json': file(manifest(name, 'ds-cleanrace', { datasetSlug: 'cleanrace' })),
    'train-log.jsonl': file('{"type":"step","step":200,"loss":2.5}\n'),
    'ckpt-100/adapter_model.safetensors': file('W-100'),
    'ckpt-200/adapter_model.safetensors': file('W-200'),
  }};
  let release: (() => void) | undefined;
  let entered: (() => void) | undefined;
  const reached = new Promise<void>(r => { entered = r; });
  let holding = false;
  const fake = await fakeWorker([run], (rel, res, buf) => {
    if (holding && rel === 'train-log.jsonl') { release = () => res.end(buf); entered!(); return true; }
    return false;
  });
  const w = { name: 'CleanRaceW', url: fake.url };
  const app = express();
  app.use(express.json());
  app.use('/ops', trainingOperationsRouter);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(r => server.once('listening', r));
  try {
    await syncYue2Mirror(w);
    upsertPreset(Number(lyrics.id), { mm3AdapterPath: ref(100) });
    run.live = false;
    run.files['train-log.jsonl'] = file('{"type":"step","step":200,"loss":2.5}\n{"type":"done"}\n', 2000);
    holding = true;
    const pass = syncYue2Mirror(w);
    await reached;
    const response = fetch('http://127.0.0.1:' + (server.address() as any).port + '/ops/mm3/cleanup', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({
        version: 1, operation: { kind: 'mm3:cleanup', idempotencyKey: 'review-cleanrace' },
        worker: { kind: 'local' }, dataset: { id: 'ds-cleanrace', revision: iso }, sources: [],
        payload: { datasetId: 'ds-cleanrace', runName: name, step: 100, choice: { otherCheckpoints: true } },
      }),
    });
    let waiting = false;
    for (let i = 0; i < 200 && !waiting; i++) {
      await new Promise(r => setTimeout(r, 5));
      try { await withReviewMutation('ds-cleanrace', () => {}); } catch { waiting = true; }
    }
    assert.ok(waiting, 'cleanup reached the review lock');
    release!();
    await pass;
    const { readMm3Linked } = await import('./lyricStudioExport.js');
    assert.equal(readMm3Linked().cleanrace?.ref, path.join(name, 'ckpt-200', 'adapter_model.safetensors'));
    const res = await response;
    const result = await res.json();
    const linked = String(getPreset(Number(lyrics.id))!.mm3_adapter_path);
    const linkedExists = fs.existsSync(path.join(mm3AdapterRoot(), linked));
    assert.equal(res.status, 409, JSON.stringify(result));
    assert.equal(linkedExists, true, 'cleanup must never remove the newly linked checkpoint');
  } finally { release?.(); fake.close(); server.closeAllConnections(); await new Promise<void>(r => server.close(() => r())); }
});
