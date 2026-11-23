import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import express from 'express';

test('MM3 status and training use an installed BF16 base without Q8_0', async t => {
  const root = fs.mkdtempSync(path.join(process.cwd(), '.test-mm3-admission-'));
  process.env.TRAINING_DIR = path.join(root, 'training');
  process.env.DATA_DIR = path.join(root, 'library');
  const { config } = await import('../config.js');
  const { initDb, closeDb } = await import('../db/database.js');
  const { insertDataset } = await import('../services/training/datasetsRepo.js');
  const { mm3CodesDir } = await import('../services/training/mm3Train.js');
  const queue = await import('../services/training/labelingQueue.js');
  const { default: training } = await import('./training.js');

  const models = path.join(root, 'models', 'mm3');
  config.aceServer.models = path.dirname(models);
  config.aceServer.adapters = path.join(root, 'adapters');
  config.aceServer.exe = path.join(root, 'ace-server');
  fs.mkdirSync(models, { recursive: true });
  for (const model of ['mm3-lm-bf16.gguf', 'mm3-depth-f16.gguf']) {
    fs.writeFileSync(path.join(models, model), 'fixture');
  }
  fs.writeFileSync(path.join(root, process.platform === 'win32' ? 'ace-train.exe' : 'ace-train'), 'fixture');
  const manifest = path.join(root, 'dataset.json');
  fs.writeFileSync(manifest, JSON.stringify({ samples: [{ filename: 'track.wav' }] }));
  fs.writeFileSync(path.join(root, 'track.mm3.txt'), 'fixture caption');
  const codes = path.join(mm3CodesDir('fixture'), 'codes');
  fs.mkdirSync(codes, { recursive: true });
  fs.writeFileSync(path.join(codes, 'track.codes'), 'fixture');

  assert.ok(config.data.dbPath.startsWith(root + path.sep));
  fs.mkdirSync(config.data.dir, { recursive: true });
  initDb();
  const now = new Date().toISOString();
  insertDataset({ id: 'fixture', slug: 'fixture', name: 'Fixture', sourceDir: root, recursive: false,
    customTag: '', tagPosition: 'prepend', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '',
    defaultLanguage: '', sampleCount: 1, labeledCount: 1, excludedCount: 0, status: 'built', builtAt: now,
    datasetJsonPath: manifest, albumName: '', createdAt: now, updatedAt: now });
  queue.overrideTrainingRunner('mm3-train-lm', async job => { queue.finishJob(job, 'done'); });

  const app = express();
  app.use(express.json());
  app.get('/vram', (_req, res) => { res.json({ total_mb: 0 }); });
  app.use(training);
  const server = app.listen(0, '127.0.0.1');
  t.after(async () => {
    await new Promise<void>(resolve => server.close(() => resolve()));
    closeDb();
    fs.rmSync(root, { recursive: true, force: true });
  });
  await new Promise<void>(resolve => server.once('listening', resolve));
  config.aceServer.host = '127.0.0.1';
  config.aceServer.port = (server.address() as { port: number }).port;
  const datasetUrl = `${config.aceServer.url}/datasets/fixture`;
  const status = async () => await (await fetch(`${datasetUrl}/mm3`)).json() as {
    defaults: { basePrecision: string }; missingForTrain: string[]; missingForLaunder: string[];
  };
  const start = (body: object) => fetch(`${datasetUrl}/mm3-train-lm`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });

  const bf16 = await status();
  assert.deepEqual(bf16.missingForTrain, []);
  assert.equal(bf16.defaults.basePrecision, 'bf16');
  assert.ok(bf16.missingForLaunder.includes('mm3-lm-q8_0.gguf'));

  const missingQ8 = await start({ basePrecision: 'q8_0' });
  assert.equal(missingQ8.status, 400);
  assert.match(await missingQ8.text(), /mm3-lm-q8_0\.gguf/);

  const selected = await start({ basePrecision: 'bf16' });
  assert.equal(selected.status, 200);
  const selectedBody = await selected.json() as { jobId: string; previewWarning?: string };
  assert.equal(selectedBody.previewWarning, undefined);
  assert.equal((queue.getJob(selectedBody.jobId)?.opts as { basePrecision: string }).basePrecision, 'bf16');

  const started = await start({ preview: { everySteps: 100 } });
  const body = await started.json() as { jobId: string; previewWarning?: string };
  assert.equal(started.status, 200, JSON.stringify(body));
  assert.match(body.previewWarning ?? '', /Previews skipped.*mm3-lm-q8_0\.gguf/);
  const job = queue.getJob(body.jobId);
  assert.deepEqual(job?.opts && {
    basePrecision: (job.opts as { basePrecision: string }).basePrecision,
    preview: (job.opts as { preview?: unknown }).preview,
  }, { basePrecision: 'bf16', preview: undefined });
  assert.ok(job?.events.some(event => event.type === 'log' && event.message === body.previewWarning));

  for (const model of ['mm3-lm-q8_0.gguf', 'mm3-lm-q8_0.pissa-r128.safetensors']) {
    fs.writeFileSync(path.join(models, model), 'fixture');
  }
  const q8 = await status();
  assert.equal(q8.defaults.basePrecision, 'q8_0');
  assert.deepEqual(q8.missingForTrain, []);
  const withQ8 = await start({ basePrecision: 'bf16', preview: { everySteps: 100 } });
  const withQ8Body = await withQ8.json() as { jobId: string; previewWarning?: string };
  assert.equal(withQ8.status, 200, JSON.stringify(withQ8Body));
  assert.equal(withQ8Body.previewWarning, undefined);
  const opts = queue.getJob(withQ8Body.jobId)?.opts as { basePrecision: string; preview: { everySteps: number } };
  assert.equal(opts.basePrecision, 'bf16');
  assert.equal(opts.preview.everySteps, 100);
});
