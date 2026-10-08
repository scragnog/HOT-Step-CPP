// routes/streamSessions.ts — /api/stream-sessions: Batch 5 row 8a.
//
//   GET    /                          sessions, newest first
//   GET    /:id                       one session: status, chunk analysis, recording
//   POST   /:id/recording   { action: 'start' | 'stop' | 'discard' }
//   GET    /:id/recording/export?format=wav|flac|mp3|opus[&bitrate=kbps]
//
// Sessions are created by the stream routes themselves (POST
// /api/generate/storm/stream, GET /api/generate/mm3/stream/:id), which name
// theirs in the X-Stream-Session response header. Same access policy as those
// routes: installation-scoped, no per-user auth. See services/streamSessions.
import { Router, type Request, type Response } from 'express';
import fs from 'fs';
import { exportQuerySchema, recordingCommandSchema } from '../contracts/streamSessions.js';
import { getStreamSessions, StreamSessionError, type StreamSessions } from '../services/streamSessions/index.js';

export function createStreamSessionsRouter(store: () => StreamSessions): Router {
  const router = Router();
  const fail = (res: Response, err: unknown) => {
    if (err instanceof StreamSessionError) { res.status(err.status).json({ error: err.message }); return; }
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  };
  router.get('/', (_req, res) => { res.json({ sessions: store().list() }); });
  router.get('/:id', (req, res) => {
    try { res.json({ session: store().get(String(req.params.id)).summary() }); } catch (err) { fail(res, err); }
  });
  router.post('/:id/recording', (req: Request, res: Response) => {
    const input = recordingCommandSchema.safeParse(req.body);
    if (!input.success) { res.status(400).json({ error: 'action must be start, stop or discard' }); return; }
    try {
      const session = store().get(String(req.params.id));
      session.record(input.data.action);
      res.json({ session: session.summary() });
    } catch (err) { fail(res, err); }
  });
  router.get('/:id/recording/export', async (req, res) => {
    const query = exportQuerySchema.safeParse(req.query);
    if (!query.success) { res.status(400).json({ error: 'format must be wav, flac, mp3 or opus; bitrate 32-512' }); return; }
    try {
      const session = store().get(String(req.params.id));
      const { file, cleanup } = await session.exportTo(query.data.format, query.data.bitrate);
      const stamp = new Date(session.createdAt).toISOString().slice(0, 19).replace(/[T:]/g, '-');
      res.download(file, `${session.kind}-recording-${stamp}.${query.data.format}`, () => cleanup());
      res.on('close', () => { if (fs.existsSync(file)) cleanup(); });
    } catch (err) { fail(res, err); }
  });
  return router;
}

export default createStreamSessionsRouter(getStreamSessions);
