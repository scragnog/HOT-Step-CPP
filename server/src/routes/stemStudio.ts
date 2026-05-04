// stemStudio.ts — Stem Studio stem separation route
//
// Server-side orchestration for two modes:
//   1. Extract (DiT) — generative stem extraction via sequential /synth calls
//   2. SuperSep (ONNX) — neural network separation via ace-server's supersep pipeline
//
// Both modes persist results to data/stems/<jobId>/ for a unified
// mixer/download/history experience.

import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { randomUUID } from 'crypto';
import archiver from 'archiver';
import { aceClient, type AceRequest } from '../services/aceClient.js';
import { ensureEngineFormat } from '../services/audioConvert.js';
import { config } from '../config.js';
import { startGenerationLog, logGeneration, logGenerationParams, finishGenerationLog, failGenerationLog } from '../services/logger.js';

const router = Router();
const ACE_URL = config.aceServer.url;

// ── Constants ────────────────────────────────────────────────────────────

const VALID_TRACKS = [
  'vocals', 'backing_vocals', 'drums', 'bass', 'guitar', 'keyboard',
  'percussion', 'strings', 'synth', 'fx', 'brass', 'woodwinds',
];

const stemsBaseDir = path.join(config.data.dir, 'stems');
fs.mkdirSync(stemsBaseDir, { recursive: true });

// ── Job State ────────────────────────────────────────────────────────────

interface StemJob {
  id: string;
  type: 'extract' | 'supersep';
  status: 'pending' | 'extracting' | 'separating' | 'saving' | 'done' | 'failed' | 'cancelled';
  sourceAudioUrl: string;
  sourceFileName: string;
  tracks: string[];
  currentTrackIndex: number;
  currentTrackName: string;
  currentAceJobId?: string;
  completedStems: string[];
  error?: string;
  warning?: string;
  createdAt: number;
  // SuperSep-specific
  sepLevel?: number;
  aceSupersepJobId?: string;
  sepProgress?: number;      // 0-100 progress from ace-server during separation
  sepMessage?: string;       // status message from ace-server
  savingTotal?: number;       // total stems to save
  savingCurrent?: number;     // current stem being saved
}

const jobs = new Map<string, StemJob>();

// ── Helpers ──────────────────────────────────────────────────────────────

/** Resolve a URL-style audio path to an absolute filesystem path */
function resolveAudioPath(audioUrl: string): string {
  if (audioUrl.startsWith('/references/')) {
    return path.join(config.data.dir, 'references', path.basename(audioUrl));
  }
  if (audioUrl.startsWith('/audio/')) {
    return path.join(config.data.audioDir, path.basename(audioUrl));
  }
  if (path.isAbsolute(audioUrl)) {
    return audioUrl;
  }
  return path.join(config.data.dir, 'references', path.basename(audioUrl));
}

/** Poll ace-server job until completion */
async function pollAceJob(aceJobId: string, job: StemJob): Promise<void> {
  const MAX_POLLS = 7200; // 60 minutes max per stem
  for (let i = 0; i < MAX_POLLS; i++) {
    if (job.status === 'cancelled') {
      await aceClient.cancelJob(aceJobId);
      throw new Error('Cancelled');
    }
    const status = await aceClient.pollJob(aceJobId);
    if (status.status === 'done') return;
    if (status.status === 'failed') throw new Error(`Extract failed for ${job.currentTrackName}`);
    if (status.status === 'cancelled') throw new Error('Cancelled by engine');
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error('Extract timed out');
}

// ── SuperSep Pipeline ────────────────────────────────────────────────────

/** Sanitize a stem name for use as a filename (no slashes, dots, etc.) */
function sanitizeStemName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
}

/** Run the SuperSep pipeline: separate → save stems to disk */
async function runSupersep(job: StemJob): Promise<void> {
  const jobDir = path.join(stemsBaseDir, job.id);
  fs.mkdirSync(jobDir, { recursive: true });

  try {
    // 1. Read and convert source audio
    const srcPath = resolveAudioPath(job.sourceAudioUrl);
    if (!fs.existsSync(srcPath)) {
      throw new Error(`Source audio not found: ${srcPath}`);
    }
    let srcAudioBuf: Buffer;
    try {
      srcAudioBuf = ensureEngineFormat(srcPath);
    } catch {
      srcAudioBuf = fs.readFileSync(srcPath);
    }

    const level = job.sepLevel ?? 0;
    console.log(`[StemStudio] SuperSep job ${job.id}: level=${level}, file=${path.basename(srcPath)} (${(srcAudioBuf.length / 1024 / 1024).toFixed(1)} MB)`);

    // 2. Send to ace-server SuperSep
    job.status = 'separating';
    job.sepMessage = 'Starting separation...';
    const sepRes = await fetch(`${ACE_URL}/supersep/separate?level=${level}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: srcAudioBuf,
    });
    if (!sepRes.ok) {
      const errText = await sepRes.text();
      throw new Error(`SuperSep engine error: ${errText}`);
    }
    const sepData = await sepRes.json() as { id: string };
    const aceJobId = sepData.id;
    job.aceSupersepJobId = aceJobId;
    console.log(`[StemStudio] SuperSep job ${job.id}: ace-server job ${aceJobId}`);

    // 3. Poll ace-server until separation completes
    const MAX_POLLS = 14400; // 2 hours max
    for (let i = 0; i < MAX_POLLS; i++) {
      if ((job.status as string) === 'cancelled') return;

      const progRes = await fetch(`${ACE_URL}/supersep/progress?id=${aceJobId}`);
      const progData = await progRes.json() as { status: string; progress: number; message: string; error?: string };

      job.sepProgress = progData.progress;
      job.sepMessage = progData.message;

      if (progData.status === 'done') break;
      if (progData.status === 'failed' || progData.status === 'cancelled') {
        throw new Error(progData.error || `Separation ${progData.status}`);
      }
      await new Promise(r => setTimeout(r, 500));
    }

    // 4. Fetch stem list
    const resultRes = await fetch(`${ACE_URL}/supersep/result?id=${aceJobId}`);
    if (!resultRes.ok) throw new Error('Failed to fetch SuperSep result');
    const resultData = await resultRes.json() as { stems: Array<{ name: string; category: string; index: number; stage?: number }> };
    const stemList = resultData.stems;

    console.log(`[StemStudio] SuperSep job ${job.id}: ${stemList.length} stems to save`);

    // 5. Download each stem to disk
    job.status = 'saving';
    job.savingTotal = stemList.length;
    job.savingCurrent = 0;
    job.tracks = stemList.map(s => sanitizeStemName(s.name));

    for (let i = 0; i < stemList.length; i++) {
      if ((job.status as string) === 'cancelled') return;

      const stem = stemList[i];
      const safeName = sanitizeStemName(stem.name);
      job.savingCurrent = i + 1;
      job.currentTrackName = stem.name;

      const stemRes = await fetch(`${ACE_URL}/supersep/serve?id=${aceJobId}&stem=${stem.index}`);
      if (!stemRes.ok) {
        console.warn(`[StemStudio] Failed to fetch stem ${stem.index} (${stem.name}), skipping`);
        continue;
      }

      const stemBuf = Buffer.from(await stemRes.arrayBuffer());
      fs.writeFileSync(path.join(jobDir, `${safeName}.wav`), stemBuf);

      // Also save into stage-N subfolder for raw per-stage debugging
      const stageDir = path.join(jobDir, `stage-${stem.stage ?? 1}`);
      fs.mkdirSync(stageDir, { recursive: true });
      fs.writeFileSync(path.join(stageDir, `${safeName}.wav`), stemBuf);

      job.completedStems.push(safeName);
      console.log(`[StemStudio] SuperSep job ${job.id}: saved ${safeName} [stage ${stem.stage ?? 1}] (${(stemBuf.length / 1024).toFixed(0)} KB)`);
    }

    // 6. Write metadata
    // Build stem metadata for the result endpoint — preserve original names + categories
    const stemMeta = stemList.map((s, idx) => ({
      originalName: s.name,
      safeName: sanitizeStemName(s.name),
      category: s.category,
      index: idx,
      stage: s.stage,
    }));

    fs.writeFileSync(path.join(jobDir, '_meta.json'), JSON.stringify({
      id: job.id,
      type: 'supersep',
      sourceAudioUrl: job.sourceAudioUrl,
      sourceFileName: job.sourceFileName,
      sepLevel: job.sepLevel,
      tracks: job.completedStems,
      completedStems: job.completedStems,
      stemMeta,
      createdAt: new Date(job.createdAt).toISOString(),
    }, null, 2));

    job.status = 'done';
    console.log(`[StemStudio] SuperSep job ${job.id}: complete (${job.completedStems.length} stems saved)`);

  } catch (err: any) {
    if (job.status !== 'cancelled') {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      console.error(`[StemStudio] SuperSep job ${job.id}: FAILED — ${err.message}`);
    }
  }
}

// ── Extract Pipeline ─────────────────────────────────────────────────────

/** Run the full extraction pipeline (async — called after POST returns) */
async function runExtraction(job: StemJob, ditSettings: any, style: string, lyrics: string): Promise<void> {
  const jobDir = path.join(stemsBaseDir, job.id);
  fs.mkdirSync(jobDir, { recursive: true });

  try {
    // Read and convert source audio to WAV
    const srcPath = resolveAudioPath(job.sourceAudioUrl);
    if (!fs.existsSync(srcPath)) {
      throw new Error(`Source audio not found: ${srcPath}`);
    }
    let srcAudioBuf: Buffer;
    try {
      srcAudioBuf = ensureEngineFormat(srcPath);
    } catch {
      // Fall back to raw file if conversion fails
      srcAudioBuf = fs.readFileSync(srcPath);
    }

    console.log(`[StemStudio] Job ${job.id}: extracting ${job.tracks.length} tracks from ${path.basename(srcPath)} (${(srcAudioBuf.length / 1024 / 1024).toFixed(1)} MB)`);

    // Check for turbo model (warn but don't block)
    try {
      const props = await aceClient.props();
      const ditModels = props.models?.dit || [];
      const activeModel = ditSettings?.ditModel || ditModels[0] || '';
      if (activeModel.toLowerCase().includes('turbo')) {
        job.warning = 'Extract requires a base/SFT model. Turbo models produce incoherent output for extraction.';
        console.warn(`[StemStudio] WARNING: Turbo model detected (${activeModel}) — extract quality will be poor`);
      }
    } catch { /* non-fatal */ }

    // Extract each track sequentially
    for (let i = 0; i < job.tracks.length; i++) {
      if (job.status === 'cancelled') return;

      const trackName = job.tracks[i];
      job.currentTrackIndex = i;
      job.currentTrackName = trackName;
      job.status = 'extracting';

      // Log to session generations folder
      const stemLogId = `${job.id}_${trackName}`;
      startGenerationLog(stemLogId, 'extract');
      logGeneration(stemLogId, 'INFO', `Stem extraction: ${trackName} (${i + 1}/${job.tracks.length})`);
      logGeneration(stemLogId, 'INFO', `Source: ${job.sourceFileName}`);

      console.log(`[StemStudio] Job ${job.id}: extracting track ${i + 1}/${job.tracks.length} — ${trackName}`);

      // Build AceRequest for this track
      const aceReq: AceRequest = {
        caption: style || '',
        // Only pass lyrics for the 'vocals' track — feeding them into other
        // tracks (e.g. backing_vocals) forces the model to route lead vocals there
        lyrics: trackName === 'vocals' ? (lyrics || '') : '',
        task_type: 'extract',
        track: trackName,
        audio_cover_strength: 1.0,  // forced — DiT sees full mix
        // Use the client-specified model (forced to base/SFT)
        synth_model: ditSettings?.ditModel,
        // Inherit basic DiT settings (or use engine defaults)
        inference_steps: ditSettings?.inferenceSteps,
        infer_method: ditSettings?.inferMethod,
        scheduler: ditSettings?.scheduler || 'linear',
        guidance_mode: ditSettings?.guidanceMode || 'apg',
        guidance_scale: ditSettings?.guidanceScale,
        shift: ditSettings?.shift,
        // Force-disable adapters for extraction
        adapter: '',
        adapter_scale: 0,
        // Clear metadata — let model infer from source audio
        bpm: 0,
        duration: 0,
        keyscale: '',
        timesignature: '',
        seed: Math.floor(Math.random() * 2_147_483_647),
      };

      logGenerationParams(stemLogId, aceReq);

      // Submit via multipart (same pattern as cover mode)
      const aceJobId = await aceClient.submitSynthMultipart(aceReq, srcAudioBuf, undefined, 'wav16');
      job.currentAceJobId = aceJobId;
      logGeneration(stemLogId, 'INFO', `Engine job submitted: ${aceJobId}`);

      // Poll until done
      await pollAceJob(aceJobId, job);

      // Fetch audio result
      const audioRes = await aceClient.getJobResult(aceJobId);
      const audioBuf = Buffer.from(await audioRes.arrayBuffer());
      const stemPath = path.join(jobDir, `${trackName}.wav`);
      fs.writeFileSync(stemPath, audioBuf);

      job.completedStems.push(trackName);
      logGeneration(stemLogId, 'INFO', `Complete: ${(audioBuf.length / 1024).toFixed(0)} KB`);
      finishGenerationLog(stemLogId, 'extract');
      console.log(`[StemStudio] Job ${job.id}: ${trackName} complete (${(audioBuf.length / 1024).toFixed(0)} KB)`);
    }

    // Write metadata file
    fs.writeFileSync(path.join(jobDir, '_meta.json'), JSON.stringify({
      id: job.id,
      type: 'extract',
      sourceAudioUrl: job.sourceAudioUrl,
      sourceFileName: job.sourceFileName,
      tracks: job.tracks,
      completedStems: job.completedStems,
      createdAt: new Date(job.createdAt).toISOString(),
    }, null, 2));

    job.status = 'done';
    console.log(`[StemStudio] Job ${job.id}: extraction complete (${job.completedStems.length} stems)`);

  } catch (err: any) {
    if (job.status !== 'cancelled') {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      console.error(`[StemStudio] Job ${job.id}: FAILED — ${err.message}`);
    }
  }
}

// ── Routes ───────────────────────────────────────────────────────────────

/**
 * POST /extract — Start a new extraction job
 */
router.post('/extract', (req: Request, res: Response) => {
  const { sourceAudioUrl, sourceFileName, tracks, style, lyrics, ditSettings } = req.body;

  // Validate
  if (!sourceAudioUrl) {
    res.status(400).json({ error: 'sourceAudioUrl is required' });
    return;
  }
  if (!tracks || !Array.isArray(tracks) || tracks.length === 0) {
    res.status(400).json({ error: 'tracks must be a non-empty array' });
    return;
  }
  const invalidTracks = tracks.filter((t: string) => !VALID_TRACKS.includes(t));
  if (invalidTracks.length > 0) {
    res.status(400).json({ error: `Invalid track names: ${invalidTracks.join(', ')}` });
    return;
  }

  // Create job
  const job: StemJob = {
    id: randomUUID(),
    type: 'extract',
    status: 'pending',
    sourceAudioUrl,
    sourceFileName: sourceFileName || 'unknown',
    tracks,
    currentTrackIndex: 0,
    currentTrackName: tracks[0],
    completedStems: [],
    createdAt: Date.now(),
  };
  jobs.set(job.id, job);

  // Start extraction async
  runExtraction(job, ditSettings || {}, style || '', lyrics || '');

  console.log(`[StemStudio] Job ${job.id} created: ${tracks.length} tracks from ${sourceFileName || sourceAudioUrl}`);
  res.json({ id: job.id });
});

/**
 * POST /supersep — Start a new SuperSep separation job
 */
router.post('/supersep', (req: Request, res: Response) => {
  const { sourceAudioUrl, sourceFileName, level } = req.body;

  if (!sourceAudioUrl) {
    res.status(400).json({ error: 'sourceAudioUrl is required' });
    return;
  }

  const sepLevel = parseInt(String(level ?? '0'), 10);

  const job: StemJob = {
    id: randomUUID(),
    type: 'supersep',
    status: 'pending',
    sourceAudioUrl,
    sourceFileName: sourceFileName || 'unknown',
    tracks: [],
    currentTrackIndex: 0,
    currentTrackName: '',
    completedStems: [],
    createdAt: Date.now(),
    sepLevel,
  };
  jobs.set(job.id, job);

  // Start separation async
  runSupersep(job);

  console.log(`[StemStudio] SuperSep job ${job.id} created: level=${sepLevel} from ${sourceFileName || sourceAudioUrl}`);
  res.json({ id: job.id });
});

/**
 * GET /:jobId/progress — Poll job progress (works for both extract and supersep)
 */
router.get('/:jobId/progress', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const job = jobs.get(jobId);
  if (!job) {
    // Check if this is a completed job on disk (server restarted)
    const metaPath = path.join(stemsBaseDir, jobId, '_meta.json');
    if (fs.existsSync(metaPath)) {
      res.json({
        status: 'done',
        progress: 100,
        currentTrack: '',
        completedStems: JSON.parse(fs.readFileSync(metaPath, 'utf-8')).completedStems || [],
        totalTracks: 0,
      });
      return;
    }
    res.status(404).json({ error: 'Job not found' });
    return;
  }

  let progress = 0;
  const totalTracks = job.tracks.length;
  const completedCount = job.completedStems.length;

  if (job.type === 'supersep') {
    // SuperSep progress: separation phase (0-80%) + saving phase (80-100%)
    if (job.status === 'separating') {
      progress = Math.round((job.sepProgress || 0) * 0.8);
    } else if (job.status === 'saving') {
      const saveProgress = job.savingTotal ? (job.savingCurrent || 0) / job.savingTotal : 0;
      progress = Math.round(80 + saveProgress * 20);
    } else if (job.status === 'done') {
      progress = 100;
    }
  } else {
    // Extract progress: each track is an equal share
    if (totalTracks > 0) {
      const perTrack = 100 / totalTracks;
      progress = Math.round(completedCount * perTrack + (job.status === 'extracting' ? perTrack * 0.5 : 0));
    }
    if (job.status === 'done') progress = 100;
  }

  res.json({
    status: job.status,
    progress,
    currentTrack: job.currentTrackName,
    completedStems: job.completedStems,
    totalTracks: job.type === 'supersep' ? (job.savingTotal || 0) : totalTracks,
    warning: job.warning,
    error: job.error,
    // SuperSep-specific extras
    sepMessage: job.sepMessage,
  });
});

/**
 * GET /:jobId/result — Get completed stem metadata
 */
router.get('/:jobId/result', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const jobDir = path.join(stemsBaseDir, jobId);
  const metaPath = path.join(jobDir, '_meta.json');

  if (!fs.existsSync(metaPath)) {
    res.status(404).json({ error: 'Job not found or not complete' });
    return;
  }

  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));

  // SuperSep jobs have stemMeta with original names + categories;
  // Extract jobs just have completedStems (track names == filenames)
  const stemMetaList: Array<{ originalName: string; safeName: string; category: string; index: number; stage?: number }> | undefined = meta.stemMeta;

  const stems = (meta.completedStems || []).map((safeName: string, idx: number) => {
    const stemPath = path.join(jobDir, `${safeName}.wav`);
    const stat = fs.existsSync(stemPath) ? fs.statSync(stemPath) : null;

    // Use stemMeta for display name/category if available (supersep), else use track name
    const sMeta = stemMetaList?.find(m => m.safeName === safeName);

    return {
      trackName: sMeta?.originalName || safeName,
      category: sMeta?.category || undefined,
      audioUrl: `/api/stem-studio/${jobId}/stem/${safeName}`,
      durationSec: 0,
      index: idx,
      sizeBytes: stat?.size || 0,
      stage: sMeta?.stage,
    };
  });

  res.json({ id: jobId, type: meta.type || 'extract', stems });
});

/**
 * GET /:jobId/stem/:trackName — Serve individual stem WAV
 */
router.get('/:jobId/stem/:trackName', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const trackName = req.params.trackName as string;
  const stemPath = path.join(stemsBaseDir, jobId, `${trackName}.wav`);

  if (!fs.existsSync(stemPath)) {
    res.status(404).json({ error: `Stem not found: ${trackName}` });
    return;
  }

  res.setHeader('Content-Type', 'audio/wav');
  res.setHeader('Content-Disposition', `inline; filename="${trackName}.wav"`);
  fs.createReadStream(stemPath).pipe(res);
});

/**
 * GET /:jobId/download-all — Download all stems as ZIP
 */
router.get('/:jobId/download-all', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const jobDir = path.join(stemsBaseDir, jobId);
  const metaPath = path.join(jobDir, '_meta.json');

  if (!fs.existsSync(metaPath)) {
    res.status(404).json({ error: 'Job not found' });
    return;
  }

  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
  const sourceBase = (meta.sourceFileName || 'stems').replace(/\.[^.]+$/, '');

  res.setHeader('Content-Type', 'application/zip');
  res.setHeader('Content-Disposition', `attachment; filename="${sourceBase}-stems.zip"`);

  const archive = archiver('zip', { zlib: { level: 1 } }); // Fast compression for large WAVs
  archive.pipe(res);

  for (const trackName of (meta.completedStems || [])) {
    const stemPath = path.join(jobDir, `${trackName}.wav`);
    if (fs.existsSync(stemPath)) {
      archive.file(stemPath, { name: `${sourceBase}/${trackName}.wav` });
    }
  }

  archive.finalize();
});

/**
 * GET /jobs — List all past extraction jobs
 */
router.get('/jobs', (_req: Request, res: Response) => {
  try {
    if (!fs.existsSync(stemsBaseDir)) {
      res.json([]);
      return;
    }

    const entries = fs.readdirSync(stemsBaseDir, { withFileTypes: true });
    const jobSummaries = [];

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const metaPath = path.join(stemsBaseDir, entry.name, '_meta.json');
      if (!fs.existsSync(metaPath)) continue;

      try {
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
        jobSummaries.push({
          id: meta.id || entry.name,
          type: meta.type || 'extract',
          sourceFileName: meta.sourceFileName || 'unknown',
          tracks: meta.tracks || [],
          completedStems: meta.completedStems || [],
          createdAt: meta.createdAt || '',
          sepLevel: meta.sepLevel,
        });
      } catch { /* skip corrupted meta */ }
    }

    // Sort newest first
    jobSummaries.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    res.json(jobSummaries);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * DELETE /:jobId — Delete a single extraction job
 */
router.delete('/:jobId', (req: Request, res: Response) => {
  const jobId = req.params.jobId as string;
  const jobDir = path.join(stemsBaseDir, jobId);

  // Cancel if still running
  const job = jobs.get(jobId);
  if (job && (job.status === 'pending' || job.status === 'extracting')) {
    job.status = 'cancelled';
  }
  jobs.delete(jobId);

  if (fs.existsSync(jobDir)) {
    fs.rmSync(jobDir, { recursive: true, force: true });
    console.log(`[StemStudio] Deleted job ${jobId}`);
    res.json({ ok: true });
  } else {
    res.status(404).json({ error: 'Job not found' });
  }
});

/**
 * DELETE /all — Delete ALL stem data (used by Settings page)
 */
router.delete('/all', (_req: Request, res: Response) => {
  // Cancel all running jobs
  for (const [, job] of jobs) {
    if (job.status === 'pending' || job.status === 'extracting') {
      job.status = 'cancelled';
    }
  }
  jobs.clear();

  if (fs.existsSync(stemsBaseDir)) {
    fs.rmSync(stemsBaseDir, { recursive: true, force: true });
    fs.mkdirSync(stemsBaseDir, { recursive: true });
    console.log('[StemStudio] All stems cleared');
  }
  res.json({ ok: true });
});

/**
 * GET /stats — Stem storage statistics (for Settings page)
 */
router.get('/stats', (_req: Request, res: Response) => {
  try {
    if (!fs.existsSync(stemsBaseDir)) {
      res.json({ totalBytes: 0, jobCount: 0, stemCount: 0 });
      return;
    }

    let totalBytes = 0;
    let jobCount = 0;
    let stemCount = 0;

    const entries = fs.readdirSync(stemsBaseDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      jobCount++;
      const jobDir = path.join(stemsBaseDir, entry.name);
      const files = fs.readdirSync(jobDir);
      for (const file of files) {
        if (file.endsWith('.wav')) {
          stemCount++;
          totalBytes += fs.statSync(path.join(jobDir, file)).size;
        }
      }
    }

    res.json({ totalBytes, jobCount, stemCount });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
