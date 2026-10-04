// routes/workers.ts — training on another PC (services/training/trainingWorkers.ts)
//
// Two routers. `workerRouter` (/api/training/worker) is what a worker exposes
// to the machine driving it. The default router (/api/workers) is that
// driving machine's side: status, dispatch, pull, and the proxy the Training
// Studio talks through. It is mounted before the body parsers so proxied
// request bodies stream through untouched; its own JSON routes parse inline.

import express, { Router, type Request, type Response } from 'express';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import * as repo from '../services/training/datasetsRepo.js';
import { APP_VERSION, config } from '../config.js';
import { aceClient } from '../services/aceClient.js';
import { activeTraining, dirtyCheckout, getUpdate, cancelUpdate, receiveUpdate, startUpdate, startupCommit } from '../services/training/workerUpdate.js';
import {
  getDispatch, getWorker, listWorkers, proxyToWorker, pullLinked, pullYue2Ladders, receiveDatasetFile, startDispatch,
  upsertPushedDataset, workerAdapterFile, workerDatasetFiles, workerLinkedPairs, workerStatus, workerYue2LadderFile, workerYue2Ladders,
} from '../services/training/trainingWorkers.js';

const fail = (res: Response, err: any) => res.status(err?.status ?? 500).json({ error: err?.message || String(err) });

// ── Worker side ─────────────────────────────────────────────────────────────

export const workerRouter = Router();

workerRouter.get('/status', async (_req: Request, res: Response) => {
  let engineVersion = '';
  let engineStatus = 'disconnected';
  try {
    const health = await aceClient.health();
    engineStatus = health.status;
  } catch { /* engine may be stopped while training */ }
  try { engineVersion = String((await aceClient.props() as { version?: string }).version ?? ''); }
  catch { /* properties may be unavailable during engine startup */ }
  let gpu: { memoryUsedMiB: number; utilization: number } | null = null;
  try {
    const data = execFileSync('nvidia-smi', ['--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], { encoding: 'utf8', timeout: 3000, windowsHide: true }).trim().split(/\r?\n/)[0]?.split(',');
    if (data?.length === 2) gpu = { memoryUsedMiB: Number(data[0].trim()), utilization: Number(data[1].trim()) };
  } catch { /* no NVIDIA GPU */ }
  const job = activeTraining();
  let engineBuiltAt: string | null = null;
  try { engineBuiltAt = fs.statSync(config.aceServer.exe).mtime.toISOString(); } catch { /* binary absent */ }
  res.json({ commit: startupCommit, dirty: dirtyCheckout(), version: APP_VERSION, aceServer: { status: engineStatus, version: engineVersion }, engineBuiltAt, gpu, job, idle: job === null });
});

workerRouter.post('/update', async (req: Request, res: Response) => {
  try {
    const base = String(req.query.base ?? '');
    const target = String(req.query.target ?? '');
    res.setHeader('content-type', 'application/x-ndjson');
    await receiveUpdate(req, base, target, (line, phase) => res.write(JSON.stringify({ line, phase }) + '\n'));
    res.end(JSON.stringify({ phase: 'done' }) + '\n');
  } catch (err: any) {
    if (res.headersSent) res.end(JSON.stringify({ error: err?.message || String(err) }) + '\n');
    else fail(res, err);
  }
});

/** GET /api/training/worker/datasets/:id/files?slug= — what this worker already holds. */
workerRouter.get('/datasets/:id/files', (req: Request, res: Response) => {
  try { res.json({ files: workerDatasetFiles(String(req.query.slug ?? '')) }); } catch (err) { fail(res, err); }
});

/** PUT /api/training/worker/datasets/:id/file?slug=&rel=&mtime= — raw file body. */
workerRouter.put('/datasets/:id/file', async (req: Request, res: Response) => {
  try {
    await receiveDatasetFile(req.params.id as string, String(req.query.slug ?? ''), String(req.query.rel ?? ''), Number(req.query.mtime), req);
    res.json({ ok: true });
  } catch (err) { fail(res, err); }
});

/** POST /api/training/worker/datasets/:id — body { row }: create or refresh the row. */
workerRouter.post('/datasets/:id', async (req: Request, res: Response) => {
  try {
    const row = req.body?.row;
    if (!row || typeof row.slug !== 'string' || typeof row.name !== 'string') { res.status(400).json({ error: 'row is required' }); return; }
    await upsertPushedDataset(req.params.id as string, row);
    res.json({ dataset: repo.getDataset(req.params.id as string) });
  } catch (err) { fail(res, err); }
});

workerRouter.get('/linked', (_req: Request, res: Response) => {
  try { res.json({ linked: workerLinkedPairs() }); } catch (err) { fail(res, err); }
});

workerRouter.get('/adapter-file', (req: Request, res: Response) => {
  try { res.sendFile(workerAdapterFile(String(req.query.rel ?? ''))); } catch (err) { fail(res, err); }
});

/** GET /api/training/worker/yue2-ladders — every rung-bearing run this
 *  worker knows of, with a checksum per rendered preview. */
workerRouter.get('/yue2-ladders', (_req: Request, res: Response) => {
  try { res.json({ ladders: workerYue2Ladders() }); } catch (err) { fail(res, err); }
});

/** GET /api/training/worker/yue2-ladder-file?datasetId=&run=&file= — one
 *  rung's preview audio, validated the same way local playback resolves it. */
workerRouter.get('/yue2-ladder-file', (req: Request, res: Response) => {
  try { res.sendFile(workerYue2LadderFile(String(req.query.datasetId ?? ''), String(req.query.run ?? ''), String(req.query.file ?? ''))); }
  catch (err) { fail(res, err); }
});

// Deleting a worker's ladder is slice 3's job, once its chosen rung's
// checkpoint has also been pulled and linked locally — this slice never
// removes anything from the worker.

// ── Controller side ─────────────────────────────────────────────────────────

const router = Router();
const json = express.json({ limit: '5mb' });

router.get('/', async (_req: Request, res: Response) => {
  res.json({ workers: await Promise.all(listWorkers().map(workerStatus)) });
});

router.post('/:name/update', (req: Request, res: Response) => {
  const w = getWorker(req.params.name as string);
  if (!w) { res.status(404).json({ error: 'No such worker' }); return; }
  try { res.status(202).json({ update: startUpdate(w.name, w.url, config.workers.token) }); } catch (err) { fail(res, err); }
});
router.get('/:name/update', (req: Request, res: Response) => res.json({ update: getUpdate(req.params.name as string) }));
router.delete('/:name/update', (req: Request, res: Response) => {
  if (!cancelUpdate(req.params.name as string)) { res.status(409).json({ error: 'Update cannot be cancelled after upload starts' }); return; }
  res.json({ ok: true });
});

/** POST /api/workers/:name/yue2-dispatch — the batch-start body
 *  { datasetIds, lyricTiming, clearCache, recipe }. Joins a dispatch already running. */
router.post('/:name/yue2-dispatch', json, (req: Request, res: Response) => {
  const w = getWorker(req.params.name as string);
  if (!w) { res.status(404).json({ error: 'No such worker' }); return; }
  const b = (req.body || {}) as Record<string, unknown>;
  const datasetIds = Array.isArray(b.datasetIds) ? b.datasetIds.filter((x): x is string => typeof x === 'string' && x.length > 0) : [];
  const recipe = b.recipe && typeof b.recipe === 'object' ? b.recipe as Record<string, unknown> : {};
  const result = startDispatch(w, { datasetIds, lyricTiming: b.lyricTiming !== false, clearCache: b.clearCache === true, recipe });
  if ('error' in result) { res.status(400).json(result); return; }
  res.status(202).json({ dispatch: result });
});

router.get('/:name/yue2-dispatch', (req: Request, res: Response) => {
  res.json({ dispatch: getDispatch(req.params.name as string) });
});

/** POST /api/workers/:name/pull — fetch linked adapters back and link them to presets here. */
router.post('/:name/pull', async (req: Request, res: Response) => {
  const w = getWorker(req.params.name as string);
  if (!w) { res.status(404).json({ error: 'No such worker' }); return; }
  try { res.json({ pulled: await pullLinked(w) }); } catch (err) { fail(res, err); }
});

/** POST /api/workers/:name/pull-ladders — pull every rung-bearing ladder
 *  (finished or still rendering) into this machine's own index. */
router.post('/:name/pull-ladders', async (req: Request, res: Response) => {
  const w = getWorker(req.params.name as string);
  if (!w) { res.status(404).json({ error: 'No such worker' }); return; }
  try { res.json({ pulled: await pullYue2Ladders(w) }); } catch (err) { fail(res, err); }
});

router.use('/:name/api', (req: Request, res: Response) => { void proxyToWorker(req, res); });

export default router;
