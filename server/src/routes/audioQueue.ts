// routes/audioQueue.ts — /api/audio-queue, the durable audio intent queue
// (services/audioQueue/intentQueue.ts).
//
//   POST /items               enqueue { idempotencyKey, request, meta? }
//   GET  /items[?status=]     list, oldest first
//   GET  /items/:id
//   POST /items/:id/cancel
//   POST /items/:id/retry     failed, cancelled or interrupted only
//   GET  /state               { paused, maxInFlight, counts }
//   POST /pause | /resume

import { Router } from 'express';
import type { Request } from 'express';
import { getUserId } from './auth.js';
import { getDb } from '../db/database.js';
import { getActiveBackendId, getBackend } from '../services/backends/registry.js';
import { cancelGenerationJob, getGenerationJob, submitGeneration } from './generate.js';
import { AudioIntentQueue, AudioQueueError } from '../services/audioQueue/intentQueue.js';
import { enqueueAudioIntentSchema, importAudioQueueSchema, type AudioIntentStatus } from '../contracts/audioQueue.js';

const STATUSES = new Set(['held', 'pending', 'submitting', 'submitted', 'succeeded', 'failed', 'cancelled', 'interrupted']);

export function createAudioQueueRouter(queue: AudioIntentQueue, userIdOf: (req: Request) => string | null): Router {
  const router = Router();
  router.use((req, res, next) => {
    if (!userIdOf(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    next();
  });
  const handle = (fn: (req: Request) => unknown) => (req: Request, res: any) => {
    try { res.json(fn(req)); }
    catch (err) {
      if (err instanceof AudioQueueError) { res.status(err.status).json({ error: err.message }); return; }
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  };

  router.post('/items', (req, res) => {
    const parsed = enqueueAudioIntentSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid queue item', issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })) });
      return;
    }
    handle(() => {
      const { item, created } = queue.enqueue(parsed.data, userIdOf(req)!);
      res.status(created ? 201 : 200);
      void queue.tick();
      return { item, created };
    })(req, res);
  });
  router.get('/items', handle(req => {
    const s = typeof req.query.status === 'string' ? req.query.status : undefined;
    if (s && !STATUSES.has(s)) throw new AudioQueueError(400, `Unknown status '${s}'`);
    return { items: queue.list(s as AudioIntentStatus | undefined) };
  }));
  router.get('/items/:id', handle(req => ({ item: queue.get(String(req.params.id)) })));
  router.delete('/items/:id', handle(req => { queue.dismiss(String(req.params.id)); return { removed: true }; }));
  router.post('/items/:id/cancel', handle(req => ({ item: queue.cancel(String(req.params.id)) })));
  router.post('/items/:id/retry', handle(req => { const item = queue.retry(String(req.params.id)); void queue.tick(); return { item }; }));
  router.post('/migration/import', (req, res) => {
    const parsed = importAudioQueueSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid queue import', issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })) });
      return;
    }
    handle(() => { const receipt = queue.importLegacy(parsed.data, userIdOf(req)!); if (parsed.data.choice === 'resume') { queue.setPaused(false); void queue.tick(); } return receipt; })(req, res);
  });
  router.post('/migration/resume-held', (req, res) => {
    const ids = req.body?.ids;
    if (!Array.isArray(ids) || !ids.every((id: unknown) => typeof id === 'string')) {
      res.status(400).json({ error: 'ids must be an array of queue item ids' }); return;
    }
    handle(() => { const items = queue.resumeHeld(ids); queue.setPaused(false); void queue.tick(); return { items }; })(req, res);
  });
  router.post('/migration/rollback-export', handle(() => {
    queue.setPaused(true);
    if (queue.list('submitting').length) throw new AudioQueueError(409, 'Submission still in flight; retry export when it settles');
    return { version: 1, exportedAt: Date.now(), state: queue.state(), items: queue.list() };
  }));
  router.get('/state', handle(() => queue.state()));
  router.post('/pause', handle(() => queue.setPaused(true)));
  router.post('/resume', handle(() => { const s = queue.setPaused(false); void queue.tick(); return s; }));
  return router;
}

/** The process's queue, on the app database. Created on first use, after
 *  initDb(); `startAudioIntentQueue` reconciles and starts the executor. */
let instance: AudioIntentQueue | null = null;
export function audioIntentQueue(): AudioIntentQueue {
  instance ??= new AudioIntentQueue({
    db: getDb(),
    submit: (body, userId) => submitGeneration(body, userId),
    getJob: id => getGenerationJob(id),
    cancelJob: id => cancelGenerationJob(id),
    activeEngine: getActiveBackendId,
    isEngine: id => !!getBackend(id),
  });
  return instance;
}

export function startAudioIntentQueue(): void {
  audioIntentQueue().start();
}

/** Mounted at /api/audio-queue. Built on its first request, because route
 *  modules load before initDb() runs and the queue needs the database. */
const lazy = Router();
let inner: Router | null = null;
lazy.use((req, res, next) => { (inner ??= createAudioQueueRouter(audioIntentQueue(), getUserId))(req, res, next); });
export default lazy;
