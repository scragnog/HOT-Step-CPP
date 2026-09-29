// routes/workers.ts — training on another PC (services/training/trainingWorkers.ts)
//
// Two routers. `workerRouter` (/api/training/worker) is what a worker exposes
// to the machine driving it. The default router (/api/workers) is that
// driving machine's side: status, dispatch, pull, and the proxy the Training
// Studio talks through. It is mounted before the body parsers so proxied
// request bodies stream through untouched; its own JSON routes parse inline.

import express, { Router, type Request, type Response } from 'express';
import * as repo from '../services/training/datasetsRepo.js';
import {
  getDispatch, getWorker, listWorkers, proxyToWorker, pullLinked, receiveDatasetFile, startDispatch,
  upsertPushedDataset, workerAdapterFile, workerDatasetFiles, workerLinkedPairs, workerStatus,
} from '../services/training/trainingWorkers.js';

const fail = (res: Response, err: any) => res.status(err?.status ?? 500).json({ error: err?.message || String(err) });

// ── Worker side ─────────────────────────────────────────────────────────────

export const workerRouter = Router();

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

// ── Controller side ─────────────────────────────────────────────────────────

const router = Router();
const json = express.json({ limit: '5mb' });

router.get('/', async (_req: Request, res: Response) => {
  res.json({ workers: await Promise.all(listWorkers().map(workerStatus)) });
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

router.use('/:name/api', (req: Request, res: Response) => { void proxyToWorker(req, res); });

export default router;
