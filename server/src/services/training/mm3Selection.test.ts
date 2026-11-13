import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import express from 'express';

test('MM3 picks persist without presets and cleanup protects both the record and album references', async () => {
  const root = fs.mkdtempSync(path.resolve('.test-mm3-selection-'));
  process.env.DATA_DIR = path.join(root, 'data');
  process.env.TRAINING_DIR = path.join(root, 'training');
  process.env.ACESTEPCPP_ADAPTERS = path.join(root, 'adapters');
  const { initDb, closeDb } = await import('../../db/database.js');
  const repo = await import('./datasetsRepo.js');
  const links = await import('./lyricStudioExport.js');
  const { mm3AdapterRoot } = await import('./mm3Runs.js');
  const { mm3Ladder, deleteMm3Run } = await import('./mm3Ladder.js');
  const { trainingOperationsRouter } = await import('./operations.js');
  const presets = await import('../../db/lireekDb.js');
  const app = express();
  app.use(express.json()); app.use('/ops', trainingOperationsRouter);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.on('listening', resolve));
  const origin = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  try {
    initDb();
    const iso = new Date().toISOString();
    repo.insertDataset({ id: 'selection', slug: 'selection', name: 'Selection', sourceDir: path.join(root, 'source'),
      recursive: true, customTag: '', tagPosition: 'prefix', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: '',
      sampleCount: 0, labeledCount: 0, excludedCount: 0, status: 'draft', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: iso, updatedAt: iso } as any);
    const runName = 'selection-2026-11-10_10-00-00';
    const dir = path.join(mm3AdapterRoot(), runName);
    for (const step of [100, 200]) {
      fs.mkdirSync(path.join(dir, `ckpt-${step}`), { recursive: true });
      fs.writeFileSync(path.join(dir, `ckpt-${step}`, 'adapter_model.safetensors'), 'weights');
    }
    fs.writeFileSync(path.join(dir, 'hotstep-run.json'), JSON.stringify({ version: 1, datasetId: 'selection', datasetSlug: 'selection', runName, options: {} }));
    const command = async (op: string, step: number, choice = {}) => {
      const res = await fetch(`${origin}/ops/mm3/${op}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({
        version: 1, operation: { kind: `mm3:${op}`, idempotencyKey: crypto.randomUUID() }, worker: { kind: 'local' },
        dataset: mm3Ladder('selection', runName).dataset, sources: [], payload: { datasetId: 'selection', runName, step, choice },
      }) });
      return { status: res.status, body: await res.json() as any };
    };
    const first = await command('select', 100);
    assert.equal(first.status, 200); assert.equal(first.body.presets, 0); assert.ok(first.body.plan);
    assert.equal(mm3Ladder('selection', runName).inUse, first.body.ref);
    assert.equal(links.readMm3Linked().selection.ref, first.body.ref);
    await assert.rejects(deleteMm3Run('selection', runName), /dataset uses a checkpoint/);
    const second = await command('select', 200);
    assert.equal(second.status, 200); assert.ok(second.body.plan);
    assert.equal(mm3Ladder('selection', runName).inUse, second.body.ref);
    const otherRun = 'selection-2026-11-09_10-00-00';
    const otherDir = path.join(mm3AdapterRoot(), otherRun);
    fs.mkdirSync(path.join(otherDir, 'ckpt-100'), { recursive: true });
    fs.writeFileSync(path.join(otherDir, 'ckpt-100', 'adapter_model.safetensors'), 'weights');
    fs.writeFileSync(path.join(otherDir, 'hotstep-run.json'), JSON.stringify({ version: 1, datasetId: 'selection', datasetSlug: 'selection', runName: otherRun, options: {} }));
    const otherRef = path.join(otherRun, 'ckpt-100', 'adapter_model.safetensors');
    links.pointMm3PresetsAt({ slug: 'another-dataset' }, otherRef);
    const protectedRun = await command('select', 200);
    assert.equal(protectedRun.body.plan.otherRuns.count, 0);
    await assert.rejects(deleteMm3Run('selection', otherRun), /dataset uses a checkpoint/);
    links.clearMm3LinkedRun(otherRun);
    const unlinkedRun = await command('select', 200);
    assert.equal(unlinkedRun.body.plan.otherRuns.count, 1);
    const stale = await command('cleanup', 100, { otherCheckpoints: true });
    assert.equal(stale.status, 409); assert.match(stale.body.error, /no longer the linked adapter/);
    const artist = presets.getOrCreateArtist('Fixture');
    const album = presets.saveLyricsSet(Number(artist.id), 'Fixture', 0, []);
    presets.upsertPreset(Number(album.id), { mm3AdapterPath: first.body.ref.replace(/\\/g, '/').toUpperCase() });
    const protectedPick = await command('cleanup', 200, { otherCheckpoints: true });
    assert.equal(protectedPick.status, 409); assert.match(protectedPick.body.error, /album preset uses/);
    assert.ok(fs.existsSync(path.join(dir, 'ckpt-100', 'adapter_model.safetensors')));
    presets.clearMm3AdapterPath(Number(album.id));
    const cleaned = await command('cleanup', 200, { otherCheckpoints: true });
    assert.equal(cleaned.status, 200);
    assert.equal(fs.existsSync(path.join(dir, 'ckpt-100')), false);
    assert.ok(fs.existsSync(path.join(dir, 'ckpt-200', 'adapter_model.safetensors')));

    // A linked album with no preset row stays without a row, as in YuE2.
    const emptyAlbum = presets.saveLyricsSet(Number(artist.id), 'Empty fixture', 0, []);
    assert.equal(links.pointMm3PresetsAt({ slug: 'empty', lyricsSetId: Number(emptyAlbum.id) }, second.body.ref), 0);
    assert.equal(presets.getPreset(Number(emptyAlbum.id)), null);
    assert.equal(links.readMm3Linked().empty.ref, second.body.ref);
    links.pointMm3PresetsAt({ slug: 'selection', lyricsSetId: Number(album.id) }, second.body.ref);
    assert.equal(presets.getPreset(Number(album.id))?.mm3_adapter_path, second.body.ref);

    // The local runner uses the same durable link even with no album.
    const { refreshLinkedMm3Presets } = await import('./mm3TrainRunner.js');
    links.clearMm3LinkedRun(runName);
    assert.equal(links.readMm3Linked().selection, undefined);
    refreshLinkedMm3Presets({ id: 'fixture-job', datasetId: 'selection' } as any, {}, path.join(dir, 'ckpt-200'));
    assert.equal(links.readMm3Linked().selection.ref, second.body.ref);
    await deleteMm3Run('selection', runName, true);
    assert.equal(links.readMm3Linked().selection, undefined);
    assert.equal(links.readMm3Linked().empty, undefined);
  } finally {
    await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
    closeDb(); fs.rmSync(root, { recursive: true, force: true });
  }
});
