// stemStudio.ts — Stem Studio extraction route
//
// Server-side orchestration for ACE-Step DiT extract mode.
// Manages extraction jobs: sequential /synth calls per stem,
// progress polling, stem serving, ZIP downloads, and cleanup.

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
  status: 'pending' | 'extracting' | 'done' | 'failed' | 'cancelled';
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
        lyrics: lyrics || '',
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
 * GET /:jobId/progress — Poll extraction progress
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

  const totalTracks = job.tracks.length;
  const completedCount = job.completedStems.length;
  // Progress: each track is an equal share. Current track is estimated at 50% while running.
  let progress = 0;
  if (totalTracks > 0) {
    const perTrack = 100 / totalTracks;
    progress = Math.round(completedCount * perTrack + (job.status === 'extracting' ? perTrack * 0.5 : 0));
  }
  if (job.status === 'done') progress = 100;

  res.json({
    status: job.status,
    progress,
    currentTrack: job.currentTrackName,
    completedStems: job.completedStems,
    totalTracks,
    warning: job.warning,
    error: job.error,
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
  const stems = (meta.completedStems || []).map((trackName: string, idx: number) => {
    const stemPath = path.join(jobDir, `${trackName}.wav`);
    const stat = fs.existsSync(stemPath) ? fs.statSync(stemPath) : null;
    return {
      trackName,
      audioUrl: `/api/stem-studio/${jobId}/stem/${trackName}`,
      durationSec: 0, // Could be computed from WAV header but not critical
      index: idx,
      sizeBytes: stat?.size || 0,
    };
  });

  res.json({ id: jobId, stems });
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
          sourceFileName: meta.sourceFileName || 'unknown',
          tracks: meta.tracks || [],
          completedStems: meta.completedStems || [],
          createdAt: meta.createdAt || '',
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
