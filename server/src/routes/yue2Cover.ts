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
import { COVER_DRIFT_METRIC_VERSION, measureCoverDrift } from '../services/backends/yue2/coverDrift.js';
import { remeterTwoFourToFourFour, shiftBarLines } from '../services/backends/yue2/coverScoreTransform.js';
import { coverScoreDetails, registerCoverWorkflows } from '../services/workflows/coverWorkflow.js';
import { workflowDocuments } from './workflows.js';
import { resolveAudioAsset } from '../services/assets/audioAssets.js';
import { WorkflowError } from '../services/workflows/workflowJobs.js';
import { TakesError, applyBuildLyrics, applyBuildSettings,cancelBuildTakes, createBuildSession, cropBuildTake, deleteBuild, deleteBuildTake, editBuildPlan, generateBuildTakes, getBuildSession, keepBuildTake, lyricsProblem, renderBuildPreview, setBuildTrim, setOccurrenceLyrics, undoBuildKeep } from '../services/yue2Takes.js';
import { buildSessionHasActiveJob } from './generate.js';
import { buildEnvelope } from '../services/generation/envelope.js';
import { yue2RequestForJob } from '../services/backends/yue2/generate.js';
import type { GenerationJob } from '../services/generation/jobTypes.js';
import { randomUUID } from 'node:crypto';

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

  // Optional source: sheetsage2 | sheetsage2-refined | muscriptor | yourmt3-chordnet.
  // Omission uses the stored sheet-transcriber preference. Supplied ABC needs no model.
  router.get('/readiness', (req: Request, res: Response) => {
    try { res.json(service.readiness(req.query.source)); }
    catch (err) { fail(res, err); }
  });

  router.post('/source-metadata', async (req: Request, res: Response) => {
    try { res.json(await service.lookup(req.body || {}, authenticate(req)!)); }
    catch (err) { fail(res, err); }
  });

  // Score editing stays in the studio. Approval is a revisioned server write
  // bound to the source asset, so a late transcript cannot approve itself.
  router.post('/drafts/:id/approve-score', async (req: Request, res: Response) => {
    try {
      const userId = authenticate(req)!;
      const revision = req.body?.revision;
      const abc = req.body?.abc;
      if (!Number.isInteger(revision) || revision < 1 || typeof abc !== 'string' || !abc.trim()) {
        throw new WorkflowError(400, 'revision and non-empty ABC are required');
      }
      const docs = workflowDocuments();
      const document = docs.get(String(req.params.id), userId);
      if (document.kind !== 'cover-draft') throw new WorkflowError(400, 'Document is not a cover draft');
      if (document.revision !== revision) throw new WorkflowError(409, 'Stale cover draft revision');
      const assetId = document.data.assetId;
      if (typeof assetId !== 'string') throw new WorkflowError(400, 'Cover draft has no asset id');
      const asset = resolveAudioAsset(getDb(), assetId, userId, config.data.dir);
      const result = await service.start({ sourceAudioUrl: asset.url, abc, sourceLabel: asset.filename.slice(0, 120) }, userId);
      const updated = docs.update(document.id, userId, revision, current => {
        if (current.assetId !== assetId) throw new WorkflowError(409, 'Cover source changed');
        return { ...current, abc: result.abc || abc.trim(), approvedAbc: result.abc || abc.trim() };
      });
      res.json({ document: updated, sourceId: result.sourceId, sourceLabel: result.sourceLabel, abc: result.abc });
    } catch (err) {
      if (err instanceof WorkflowError) { res.status(err.status).json({ error: err.message, ...err.extra }); return; }
      fail(res, err);
    }
  });

  router.post('/drafts/:id/save-score-details', async (req: Request, res: Response) => {
    try {
      const userId = authenticate(req)!;
      const { revision, abc, factor } = req.body || {};
      if (!Number.isInteger(revision) || revision < 1 || typeof abc !== 'string') {
        throw new WorkflowError(400, 'revision and ABC are required');
      }
      const document = workflowDocuments().get(String(req.params.id), userId);
      if (document.kind !== 'cover-draft') throw new WorkflowError(400, 'Document is not a cover draft');
      if (document.revision !== revision) throw new WorkflowError(409, 'Stale cover draft revision');
      const assetId = document.data.assetId;
      if (typeof assetId !== 'string') throw new WorkflowError(400, 'Cover draft has no asset id');
      const asset = resolveAudioAsset(getDb(), assetId, userId, config.data.dir);
      const { bpm, key } = coverScoreDetails(abc, factor);
      const saved = await service.saveDatasetDetails({ sourceAudioUrl: asset.url, bpm, key }, userId);
      res.json({ ...saved, bpm, key });
    } catch (err) {
      if (err instanceof WorkflowError) { res.status(err.status).json({ error: err.message, ...err.extra }); return; }
      fail(res, err);
    }
  });

  // A preview only: nothing is stored. The studio applies the result as a score
  // edit, so it has to be approved again before a cover uses it.
  router.post('/score/remeter', (req: Request, res: Response) => {
    try {
      const { abc, pickup } = req.body || {};
      if (typeof abc !== 'string' || !abc.trim() || typeof pickup !== 'boolean') {
        throw new CoverRequestError('abc and a boolean pickup are required.');
      }
      let result;
      try { result = remeterTwoFourToFourFour(abc, pickup); }
      catch (err) { throw new CoverRequestError((err as Error).message, 422); }
      res.json(result);
    } catch (err) { fail(res, err); }
  });

  // A preview too: every bar line moved by whole beats, for a score whose downbeats landed off.
  router.post('/score/shift-bars', (req: Request, res: Response) => {
    try {
      const { abc, beats } = req.body || {};
      if (typeof abc !== 'string' || !abc.trim() || !Number.isInteger(beats)) {
        throw new CoverRequestError('abc and a whole number of beats are required.');
      }
      let result;
      try { result = shiftBarLines(abc, beats); }
      catch (err) { throw new CoverRequestError((err as Error).message, 422); }
      res.json(result);
    } catch (err) { fail(res, err); }
  });

  router.post('/sections/review', (req: Request, res: Response) => {
    try { res.json(service.reviewScore(req.body || {})); }
    catch (err) { fail(res, err); }
  });

  router.post('/sections/match', async (req: Request, res: Response) => {
    try { res.json(await service.matchSections(req.body || {}, authenticate(req)!)); }
    catch (err) { fail(res, err); }
  });

  router.post('/sections/plan-score', async (req: Request, res: Response) => {
    try { res.json(await service.planScoreSections(req.body || {}, authenticate(req)!)); }
    catch (err) { fail(res, err); }
  });

  router.post('/sections/suggest', async (req: Request, res: Response) => {
    try { res.json(await service.suggestAssignments(req.body || {}, authenticate(req)!)); }
    catch (err) { fail(res, err); }
  });

  router.post('/sections/save-dataset', async (req: Request, res: Response) => {
    try { res.json(await service.saveDatasetDetails(req.body || {}, authenticate(req)!)); }
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
        COVER_DRIFT_METRIC_VERSION, song.audio_url, stat.size, stat.mtimeMs, alignText, rendered, full,
      ])).digest('hex');
      if (params.yue2CoverDrift?.metricVersion === COVER_DRIFT_METRIC_VERSION &&
          params.yue2CoverDrift.inputHash === inputHash) { res.json(params.yue2CoverDrift); return; }
      try { measureCoverDrift(rendered, full, alignText, []); }
      catch (err) { throw new CoverRequestError((err as Error).message, 422); }
      const aligned = await drift.align(drift.readAudio(audioPath), alignText);
      const result = { ...measureCoverDrift(rendered, full, alignText, aligned.words),
        metricVersion: COVER_DRIFT_METRIC_VERSION, inputHash };
      drift.save(id, userId, result);
      res.json(result);
    } catch (err) { fail(res, err); }
  });

  // Build by section (services/yue2Takes.ts). A session starts from the same
  // params a one-shot cover submits, resolved exactly as a render would.
  const takes = (res: Response, fn: () => unknown) => Promise.resolve().then(fn).then(
    out => { res.json(out); },
    err => { if (err instanceof TakesError) res.status(err.status).json({ error: err.message }); else fail(res, err); });
  const buildRequest = (params: Record<string, unknown>, userId: string) => {
    const id = `build-${randomUUID()}`;
    let envelope;
    try { envelope = buildEnvelope(params, userId, id, Date.now()); }
    catch (err) { throw new TakesError((err as Error).message); }
    if (envelope.backendId !== 'yue2') throw new TakesError('Build by section needs the YuE2 backend.');
    const job = { id, userId, envelope, params, attempts: [], status: 'pending' } as unknown as GenerationJob;
    return yue2RequestForJob(job);
  };
  // `lyrics` is the build's own text, set on the Build page. It replaces the
  // Prepare lyrics in the session's params and never reaches a dataset.
  router.post('/build', (req: Request, res: Response) => takes(res, () => {
    const userId = authenticate(req)!;
    let params = req.body?.params;
    if (!params || typeof params !== 'object' || Array.isArray(params)) throw new TakesError('params are required');
    if (req.body.lyrics !== undefined) {
      const problem = lyricsProblem(req.body.lyrics);
      if (problem) throw new TakesError(problem);
      params = { ...params, lyrics: req.body.lyrics };
    }
    return createBuildSession(params, buildRequest(params, userId), userId);
  }));
  router.post('/build/lyrics-check', (req: Request, res: Response) => takes(res, () => {
    const problem = lyricsProblem(req.body?.lyrics);
    if (problem) throw new TakesError(problem);
    return { ok: true };
  }));
  router.get('/build/:id', (req: Request, res: Response) =>
    takes(res, () => getBuildSession(String(req.params.id), authenticate(req)!)));
  // `caption` (optional) is these takes' own caption; empty uses the session's.
  router.post('/build/:id/generate', (req: Request, res: Response) => {
    const userId = authenticate(req)!;
    return takes(res, () => generateBuildTakes(String(req.params.id), userId, req.body?.to, req.body?.count, req.body?.revision,
      req.body?.caption, params => buildRequest(params, userId).style));
  });
  router.post('/build/:id/cancel', (req: Request, res: Response) =>
    takes(res, () => cancelBuildTakes(String(req.params.id), authenticate(req)!)));
  router.post('/build/:id/keep', (req: Request, res: Response) =>
    takes(res, () => keepBuildTake(String(req.params.id), authenticate(req)!, String(req.body?.take ?? ''), req.body?.revision)));
  router.post('/build/:id/undo', (req: Request, res: Response) =>
    takes(res, () => undoBuildKeep(String(req.params.id), authenticate(req)!, req.body?.section, req.body?.revision)));
  // Cuts the last kept take at codec frame `frame` (null restores it);
  // `mode` is 'regenerate' or 'advance' when the cut falls inside a section.
  router.post('/build/:id/crop', (req: Request, res: Response) =>
    takes(res, () => cropBuildTake(String(req.params.id), authenticate(req)!, String(req.body?.take ?? ''), req.body?.revision, req.body?.frame, req.body?.mode)));
  // Renders everything kept so far once, for Play build (takes hold only their own part).
  router.post('/build/:id/preview', (req: Request, res: Response) =>
    takes(res, () => renderBuildPreview(String(req.params.id), authenticate(req)!)));
  // Where Save song starts the song: seconds into a first-section take, or null for the top.
  router.post('/build/:id/trim', (req: Request, res: Response) =>
    takes(res, () => setBuildTrim(String(req.params.id), authenticate(req)!, String(req.body?.take ?? ''), req.body?.revision, req.body?.seconds ?? null)));
  router.post('/build/:id/lyrics', (req: Request, res: Response) => {
    const userId = authenticate(req)!;
    return takes(res, () => applyBuildLyrics(String(req.params.id), userId, req.body?.revision, req.body?.lyrics,
      params => buildRequest(params, userId)));
  });
  // Unbuilt occurrences' own sung lines: `edits` maps occurrence ids
  // (o3...) to text, or null to sing the build's lyrics again.
  router.post('/build/:id/occurrence-lyrics', (req: Request, res: Response) =>
    takes(res, () => setOccurrenceLyrics(String(req.params.id), authenticate(req)!, req.body?.revision, req.body?.edits)));
  // Before the first keep: `settings` holds the cover's voices, tempo, key,
  // keyShift and cfgScale (as params.yue2Cover carries them) and arStrength.
  router.post('/build/:id/settings', (req: Request, res: Response) => {
    const userId = authenticate(req)!;
    return takes(res, () => applyBuildSettings(String(req.params.id), userId, req.body?.revision, req.body?.settings,
      params => buildRequest(params, userId)));
  });
  // The order the next sections are built in: `upcoming` lists source
  // section ids (s0...) or upcoming occurrence ids (o3...); `finished` ends the song.
  router.post('/build/:id/plan', (req: Request, res: Response) =>
    takes(res, () => editBuildPlan(String(req.params.id), authenticate(req)!, req.body?.revision, req.body?.upcoming, req.body?.finished)));
  router.delete('/build/:id', (req: Request, res: Response) =>
    takes(res, () => { deleteBuild(String(req.params.id), authenticate(req)!, buildSessionHasActiveJob); return { ok: true }; }));
  router.delete('/build/:id/takes/:take', (req: Request, res: Response) =>
    takes(res, () => deleteBuildTake(String(req.params.id), authenticate(req)!, String(req.params.take), Number(req.query.revision))));

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

registerCoverWorkflows();
export default createYue2CoverRouter();
