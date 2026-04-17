// generate.ts — Generation orchestration route
//
// Orchestrates the two-step generation flow:
//   1. POST /lm → poll → get enriched JSON with audio_codes
//   2. POST /synth → poll → get audio
//   3. Save audio + metadata to SQLite
//
// Maintains an in-memory job map for frontend polling.

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { aceClient, type AceRequest } from '../services/aceClient.js';
import { getDb } from '../db/database.js';
import { config } from '../config.js';
import { getUserId } from './auth.js';

const router = Router();

/** Internal job state */
interface GenerationJob {
  id: string;
  userId: string;
  status: 'pending' | 'lm_running' | 'synth_running' | 'saving' | 'succeeded' | 'failed' | 'cancelled';
  stage?: string;
  progress?: number;
  aceJobId?: string;  // Current ace-server job ID (LM or synth)
  lmResults?: AceRequest[];
  result?: {
    audioUrls: string[];
    songIds: string[];
    bpm?: number;
    duration?: number;
    keyScale?: string;
    timeSignature?: string;
  };
  error?: string;
  params: any;
  createdAt: number;
}

const jobs = new Map<string, GenerationJob>();

/** Translate frontend params to AceRequest format */
function translateParams(params: any): AceRequest {
  const req: AceRequest = {
    caption: params.prompt || params.songDescription || params.caption || '',
  };

  // Lyrics / instrumental
  if (params.instrumental) {
    req.lyrics = '[Instrumental]';
  } else if (params.lyrics) {
    req.lyrics = params.lyrics;
  }

  // Metadata
  if (params.bpm) req.bpm = params.bpm;
  if (params.duration) req.duration = params.duration;
  if (params.keyScale) req.keyscale = params.keyScale;
  if (params.timeSignature) req.timesignature = params.timeSignature;
  if (params.vocalLanguage) req.vocal_language = params.vocalLanguage;

  // Seed
  if (params.seed !== undefined && !params.randomSeed) {
    req.seed = params.seed;
  }

  // Batch
  if (params.batchSize) req.lm_batch_size = params.batchSize;

  // LM params
  if (params.lmTemperature !== undefined) req.lm_temperature = params.lmTemperature;
  if (params.lmCfgScale !== undefined) req.lm_cfg_scale = params.lmCfgScale;
  if (params.lmTopP !== undefined) req.lm_top_p = params.lmTopP;
  if (params.lmTopK !== undefined) req.lm_top_k = params.lmTopK;
  if (params.lmNegativePrompt) req.lm_negative_prompt = params.lmNegativePrompt;

  // DiT params
  if (params.inferenceSteps) req.inference_steps = params.inferenceSteps;
  if (params.guidanceScale !== undefined) req.guidance_scale = params.guidanceScale;
  if (params.shift !== undefined) req.shift = params.shift;
  if (params.inferMethod) req.infer_method = params.inferMethod;

  // Cover/repaint
  if (params.taskType) req.task_type = params.taskType;
  if (params.audioCoverStrength !== undefined) req.audio_cover_strength = params.audioCoverStrength;
  if (params.coverNoiseStrength !== undefined) req.cover_noise_strength = params.coverNoiseStrength;
  if (params.repaintingStart !== undefined) req.repainting_start = params.repaintingStart;
  if (params.repaintingEnd !== undefined) req.repainting_end = params.repaintingEnd;
  if (params.trackName) req.track = params.trackName;

  // CoT
  if (params.useCotCaption !== undefined) req.use_cot_caption = params.useCotCaption;

  // Model routing
  if (params.ditModel) req.synth_model = params.ditModel;
  if (params.lmModel) req.lm_model = params.lmModel;
  if (params.loraPath) req.adapter = params.loraPath;
  if (params.loraScale !== undefined) req.adapter_scale = params.loraScale;

  return req;
}

/** Poll ace-server job until completion */
async function pollUntilDone(aceJobId: string, job: GenerationJob, signal: AbortSignal): Promise<void> {
  const POLL_INTERVAL = 500; // ms
  const MAX_POLLS = 3600; // 30 minutes max

  for (let i = 0; i < MAX_POLLS; i++) {
    if (signal.aborted || job.status === 'cancelled') {
      await aceClient.cancelJob(aceJobId);
      throw new Error('Cancelled');
    }

    const status = await aceClient.pollJob(aceJobId);

    if (status.status === 'done') return;
    if (status.status === 'failed') throw new Error('Generation failed on ace-server');
    if (status.status === 'cancelled') throw new Error('Cancelled by ace-server');

    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
  }

  throw new Error('Generation timed out');
}

/** Run the full generation pipeline */
async function runGeneration(job: GenerationJob): Promise<void> {
  const aceReq = translateParams(job.params);
  const abortController = new AbortController();

  // Store abort controller for cancellation
  (job as any)._abort = abortController;

  try {
    // Determine if we need the LM phase
    const skipLm = job.params.skipLm === true;
    const isCoverTask = ['cover', 'cover-nofsq', 'repaint', 'lego', 'extract'].includes(aceReq.task_type || '');

    // Even when LM is "disabled", we still need it if metadata is set to Auto
    // The LM fills in BPM, duration, key, and time signature
    const hasAutoMetadata = !aceReq.bpm || (aceReq.duration !== undefined && aceReq.duration <= 0)
      || !aceReq.keyscale || !aceReq.timesignature;
    const needsMetadataOnly = skipLm && hasAutoMetadata && !isCoverTask;
    const needsFullLm = !skipLm && !aceReq.audio_codes && !isCoverTask;

    let lmResults: AceRequest[] = [aceReq];

    if (needsFullLm || needsMetadataOnly) {
      // Phase 1: LM generation
      job.status = 'lm_running';
      job.stage = needsMetadataOnly
        ? 'Resolving auto metadata (BPM, duration, key)...'
        : 'Generating lyrics & metadata...';
      job.progress = 10;

      const lmJobId = await aceClient.submitLm(aceReq);
      job.aceJobId = lmJobId;

      await pollUntilDone(lmJobId, job, abortController.signal);

      // Fetch LM results (array of enriched AceRequests)
      const resultRes = await aceClient.getJobResult(lmJobId);
      lmResults = await resultRes.json() as AceRequest[];
      job.lmResults = lmResults;

      // If user disabled LM thinking but we ran it for metadata only,
      // strip audio_codes so synth doesn't use them
      if (needsMetadataOnly) {
        lmResults = lmResults.map(r => {
          const { audio_codes, ...rest } = r;
          return rest;
        });
      }

      job.progress = 40;
      job.stage = needsMetadataOnly
        ? 'Metadata resolved, starting synthesis...'
        : 'LM complete, starting synthesis...';
    }

    // Phase 2: Synth generation
    job.status = 'synth_running';
    job.stage = 'Synthesizing audio...';
    job.progress = 50;

    // Submit all LM results for synthesis
    const synthJobId = await aceClient.submitSynth(lmResults);
    job.aceJobId = synthJobId;

    await pollUntilDone(synthJobId, job, abortController.signal);

    job.progress = 90;
    job.stage = 'Saving audio...';
    job.status = 'saving';

    // Fetch audio result
    const audioRes = await aceClient.getJobResult(synthJobId);
    const audioBuffer = Buffer.from(await audioRes.arrayBuffer());
    const contentType = audioRes.headers.get('content-type') || 'audio/mpeg';
    const ext = contentType.includes('wav') ? 'wav' : 'mp3';

    // Save audio files and create DB entries
    const audioUrls: string[] = [];
    const songIds: string[] = [];

    // Check if multipart (batch) or single track
    if (contentType.includes('multipart')) {
      // TODO: Parse multipart response for batch mode
      // For now, handle as single track
      const filename = `${uuidv4()}.${ext}`;
      const filepath = path.join(config.data.audioDir, filename);
      fs.writeFileSync(filepath, audioBuffer);
      audioUrls.push(`/audio/${filename}`);
    } else {
      const filename = `${uuidv4()}.${ext}`;
      const filepath = path.join(config.data.audioDir, filename);
      fs.writeFileSync(filepath, audioBuffer);
      audioUrls.push(`/audio/${filename}`);
    }

    // Get metadata from LM results
    const firstResult = lmResults[0];
    const title = job.params.title || firstResult.caption?.substring(0, 60) || 'Untitled';
    const lyrics = firstResult.lyrics || job.params.lyrics || '';
    const style = firstResult.caption || job.params.style || '';
    const bpm = firstResult.bpm || 0;
    const duration = firstResult.duration || 0;
    const keyScale = firstResult.keyscale || '';
    const timeSignature = firstResult.timesignature || '';

    // Create song entries in DB
    for (const audioUrl of audioUrls) {
      const songId = uuidv4();
      getDb().prepare(`
        INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                           duration, bpm, key_scale, time_signature, tags, dit_model, generation_params)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        songId, job.userId, title, lyrics, style, firstResult.caption || '',
        audioUrl, duration, bpm, keyScale, timeSignature,
        JSON.stringify([]), aceReq.synth_model || '', JSON.stringify(job.params),
      );
      songIds.push(songId);
    }

    job.status = 'succeeded';
    job.progress = 100;
    job.stage = 'Complete!';
    job.result = {
      audioUrls,
      songIds,
      bpm,
      duration,
      keyScale,
      timeSignature,
    };

  } catch (err: any) {
    if (err.message === 'Cancelled') {
      job.status = 'cancelled';
      job.stage = 'Cancelled';
    } else {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.stage = 'Failed';
      console.error(`[Generate] Job ${job.id} failed:`, err.message);
    }
  }
}

// POST /api/generate — start a generation job
router.post('/', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const job: GenerationJob = {
    id: uuidv4(),
    userId,
    status: 'pending',
    stage: 'Queued',
    progress: 0,
    params: req.body,
    createdAt: Date.now(),
  };

  jobs.set(job.id, job);

  // Start generation in background
  runGeneration(job).catch(err => {
    console.error(`[Generate] Unhandled error in job ${job.id}:`, err);
    job.status = 'failed';
    job.error = err.message;
  });

  res.json({
    jobId: job.id,
    status: job.status,
  });
});

// GET /api/generate/status/:id — poll job status
router.get('/status/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  res.json({
    jobId: job.id,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    result: job.result,
    error: job.error,
  });
});

// POST /api/generate/cancel/:id — cancel a running job
router.post('/cancel/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  job.status = 'cancelled';
  if (job.aceJobId) {
    aceClient.cancelJob(job.aceJobId).catch(() => {});
  }
  if ((job as any)._abort) {
    (job as any)._abort.abort();
  }

  res.json({ success: true, jobId: job.id });
});

// POST /api/generate/cancel-all — cancel all running jobs
router.post('/cancel-all', (req, res) => {
  let cancelled = 0;
  for (const [, job] of jobs) {
    if (job.status === 'pending' || job.status === 'lm_running' || job.status === 'synth_running') {
      job.status = 'cancelled';
      if (job.aceJobId) {
        aceClient.cancelJob(job.aceJobId).catch(() => {});
      }
      if ((job as any)._abort) {
        (job as any)._abort.abort();
      }
      cancelled++;
    }
  }
  res.json({ success: true, cancelled });
});

export default router;
