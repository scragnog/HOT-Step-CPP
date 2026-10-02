// /api/yue2-cover: authenticated source validation and lead-sheet jobs.
import { Router, type Request, type Response } from 'express';
import { getUserId } from './auth.js';
import { CoverRequestError, yue2CoverService } from '../services/yue2Cover.js';

type CoverService = typeof yue2CoverService;

export function createYue2CoverRouter(service: CoverService = yue2CoverService, authenticate = getUserId) {
  const router = Router();
  router.use((req, res, next) => {
    if (!authenticate(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    next();
  });

  function fail(res: Response, err: unknown) {
    if (err instanceof CoverRequestError) { res.status(err.status).json({ error: err.message }); return; }
    console.error('[Yue2Cover] Request failed:', err);
    res.status(500).json({ error: 'Cover transcription failed' });
  }

  // A supplied ABC does not need the optional SheetSage2 model.
  router.get('/readiness', (_req: Request, res: Response) => { res.json(service.readiness()); });

  router.post('/transcriptions', async (req: Request, res: Response) => {
    try {
      const result = await service.start(req.body || {}, authenticate(req)!);
      res.status(result.status === 'queued' ? 202 : 200).json(result);
    } catch (err) { fail(res, err); }
  });

  router.get('/transcriptions/:jobId', (req: Request, res: Response) => {
    try { res.json(service.find(req.params.jobId as string, authenticate(req)!)); }
    catch (err) { fail(res, err); }
  });

  router.delete('/transcriptions/:jobId', (req: Request, res: Response) => {
    try { res.json(service.cancel(req.params.jobId as string, authenticate(req)!)); }
    catch (err) { fail(res, err); }
  });
  return router;
}

export default createYue2CoverRouter();
