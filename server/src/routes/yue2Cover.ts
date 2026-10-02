// /api/yue2-cover: authenticated source validation and lead-sheet jobs.
import { Router, type Request, type Response } from 'express';
import { getUserId } from './auth.js';
import { CoverRequestError, yue2CoverService } from '../services/yue2Cover.js';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { config } from '../config.js';
import { getDb } from '../db/database.js';
import { yue2Align } from '../services/backends/yue2/client.js';
import type { Yue2AlignWord } from '../services/backends/yue2/align.js';
import { measureCoverDrift } from '../services/backends/yue2/coverDrift.js';

type CoverService = typeof yue2CoverService;
type DriftSong = { id: string; audio_url: string; lyrics: string; generation_params: string };
const driftDeps = {
  song: (id: string, userId: string): DriftSong | undefined => getDb().prepare(
    'SELECT id, audio_url, lyrics, generation_params FROM songs WHERE id = ? AND user_id = ?',
  ).get(id, userId) as DriftSong | undefined,
  save: (id: string, userId: string, result: object) => { getDb().prepare(
    `UPDATE songs SET generation_params = json_set(generation_params, '$.yue2CoverDrift', json(?)) WHERE id = ? AND user_id = ?`,
  ).run(JSON.stringify(result), id, userId); },
  align: (audio: Buffer, lyrics: string): Promise<{ words: Yue2AlignWord[] }> => yue2Align(audio, lyrics),
  readAudio: (file: string) => fs.readFileSync(file),
  statAudio: (file: string) => fs.statSync(file),
  audioDir: config.data.audioDir,
};

export function createYue2CoverRouter(service: CoverService = yue2CoverService, authenticate = getUserId,
  drift: typeof driftDeps = driftDeps) {
  const router = Router();
  router.use((req, res, next) => {
    if (!authenticate(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    next();
  });

  function fail(res: Response, err: unknown) {
    if (err instanceof CoverRequestError) { res.status(err.status).json({ error: err.message }); return; }
    console.error('[Yue2Cover] Request failed:', err);
    res.status(500).json({ error: 'YuE2 cover request failed' });
  }

  // A supplied ABC does not need the optional SheetSage2 model.
  router.get('/readiness', (_req: Request, res: Response) => { res.json(service.readiness()); });

  router.post('/source-metadata', async (req: Request, res: Response) => {
    try { res.json(await service.lookup(req.body || {}, authenticate(req)!)); }
    catch (err) { fail(res, err); }
  });

  router.post('/sections/review', (req: Request, res: Response) => {
    try { res.json(service.reviewScore(req.body || {})); }
    catch (err) { fail(res, err); }
  });

  // Explicitly requested from a saved cover's details. The result is cached
  // on that song row; opening details never starts an aligner or a render.
  router.post('/drift/:songId', async (req: Request, res: Response) => {
    try {
      const id = req.params.songId as string;
      const userId = authenticate(req)!;
      const song = drift.song(id, userId);
      if (!song) throw new CoverRequestError('Cover song not found.', 404);
      let params: Record<string, any>;
      try { params = JSON.parse(song.generation_params || '{}'); }
      catch { throw new CoverRequestError('This song has no readable cover provenance.'); }
      const rendered = params.yue2Abc;
      const full = params.yue2Cover?.fullScore;
      const lyrics = params.yue2Request?.lyrics ?? song.lyrics;
      if (params.backend !== 'yue2' || !params.yue2Cover || typeof rendered !== 'string' ||
          typeof full !== 'string' || typeof lyrics !== 'string' || !lyrics.trim()) {
        throw new CoverRequestError('Drift needs a saved YuE2 cover with score and lyrics.');
      }
      const filename = path.basename(song.audio_url || '');
      if (song.audio_url !== `/audio/${filename}` || !/^[a-zA-Z0-9][a-zA-Z0-9._-]*\.wav$/i.test(filename)) {
        throw new CoverRequestError('Cover audio is not a local WAV file.');
      }
      const audioPath = path.join(drift.audioDir, filename);
      let stat: { size: number; mtimeMs: number };
      try { stat = drift.statAudio(audioPath); }
      catch { throw new CoverRequestError('Cover audio is missing.', 404); }
      const alignText = lyrics.replace(/\r\n?/g, '\n');
      const inputHash = createHash('sha256').update(JSON.stringify([
        song.audio_url, stat.size, stat.mtimeMs, alignText, rendered, full,
      ])).digest('hex');
      if (params.yue2CoverDrift?.inputHash === inputHash) { res.json(params.yue2CoverDrift); return; }
      try { measureCoverDrift(rendered, full, alignText, []); }
      catch (err) { throw new CoverRequestError((err as Error).message, 422); }
      const aligned = await drift.align(drift.readAudio(audioPath), alignText);
      const result = { ...measureCoverDrift(rendered, full, alignText, aligned.words), inputHash };
      drift.save(id, userId, result);
      res.json(result);
    } catch (err) { fail(res, err); }
  });

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
