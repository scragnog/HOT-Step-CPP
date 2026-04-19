// generate.ts — Generation orchestration route
//
// Orchestrates the two-step generation flow:
//   1. POST /lm → poll → get enriched JSON with audio_codes
//   2. POST /synth → poll → get audio
//   3. Save audio + metadata to SQLite
//
// Maintains an in-memory job map for frontend polling.
// LM results are cached by seed+params to skip the LM phase on repeats.

import { Router } from 'express';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { aceClient, type AceRequest } from '../services/aceClient.js';
import { getDb } from '../db/database.js';
import { config } from '../config.js';
import { getUserId } from './auth.js';
import { startGenerationLog, logGeneration, logGenerationParams, finishGenerationLog, failGenerationLog } from '../services/logger.js';
import { runMastering } from './mastering.js';

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

// ── LM Audio Code Cache ─────────────────────────────────────
// Caches LM results keyed by seed + all LM-affecting params.
// Same seed + same params = deterministic output = safe to cache.
const LM_CACHE_MAX = 20;
const lmCache = new Map<string, { results: AceRequest[]; timestamp: number }>();

/** Compute a stable hash key from LM-affecting parameters */
function computeLmCacheKey(req: AceRequest): string {
  const keyObj = {
    seed: req.seed,
    caption: req.caption,
    lyrics: req.lyrics,
    bpm: req.bpm,
    duration: req.duration,
    keyscale: req.keyscale,
    timesignature: req.timesignature,
    vocal_language: req.vocal_language,
    lm_model: req.lm_model,
    lm_batch_size: req.lm_batch_size,
    lm_temperature: req.lm_temperature,
    lm_cfg_scale: req.lm_cfg_scale,
    lm_top_p: req.lm_top_p,
    lm_top_k: req.lm_top_k,
    lm_negative_prompt: req.lm_negative_prompt,
    use_cot_caption: req.use_cot_caption,
  };
  return crypto.createHash('sha256')
    .update(JSON.stringify(keyObj))
    .digest('hex')
    .substring(0, 16);
}

/** Evict oldest entries when cache exceeds max size */
function evictLmCache(): void {
  if (lmCache.size <= LM_CACHE_MAX) return;
  // Sort by timestamp, evict oldest
  const entries = [...lmCache.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp);
  const toRemove = entries.slice(0, lmCache.size - LM_CACHE_MAX);
  for (const [key] of toRemove) {
    lmCache.delete(key);
  }
}

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
  if (params.scheduler) req.scheduler = params.scheduler;
  if (params.guidanceMode) req.guidance_mode = params.guidanceMode;

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
  if (params.vaeModel) req.vae_model = params.vaeModel;
  if (params.loraPath) req.adapter = params.loraPath;
  if (params.loraScale !== undefined) req.adapter_scale = params.loraScale;
  if (params.adapterGroupScales) req.adapter_group_scales = params.adapterGroupScales;
  if (params.adapterMode) req.adapter_mode = params.adapterMode;

  // Trigger word — inject adapter filename into caption
  if (params.triggerWord && params.triggerPlacement && params.loraPath) {
    const tw = params.triggerWord;
    const caption = req.caption || '';
    switch (params.triggerPlacement) {
      case 'prepend': req.caption = caption ? `${tw}, ${caption}` : tw; break;
      case 'append':  req.caption = caption ? `${caption}, ${tw}` : tw; break;
      case 'replace': req.caption = tw; break;
    }
  }

  // Solver sub-parameters
  if (params.storkSubsteps !== undefined) req.stork_substeps = params.storkSubsteps;
  if (params.beatStability !== undefined) req.beat_stability = params.beatStability;
  if (params.frequencyDamping !== undefined) req.frequency_damping = params.frequencyDamping;
  if (params.temporalSmoothing !== undefined) req.temporal_smoothing = params.temporalSmoothing;

  // Guidance sub-parameters
  if (params.apgMomentum !== undefined) req.apg_momentum = params.apgMomentum;
  if (params.apgNormThreshold !== undefined) req.apg_norm_threshold = params.apgNormThreshold;

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
    const needsLm = !skipLm && !aceReq.audio_codes && !isCoverTask;
    const taskType = aceReq.task_type || 'text2music';

    // Start per-generation log
    startGenerationLog(job.id, taskType);
    logGenerationParams(job.id, aceReq);

    let lmResults: AceRequest[] = [aceReq];

    if (skipLm && !isCoverTask) {
      // LM disabled — fill in sensible defaults for any "auto" metadata
      // The ace-server /lm always generates audio codes, so we can't call it for metadata only
      if (!aceReq.bpm) aceReq.bpm = 120;
      if (!aceReq.duration || aceReq.duration <= 0) aceReq.duration = 120;
      if (!aceReq.keyscale) aceReq.keyscale = 'C major';
      if (!aceReq.timesignature) aceReq.timesignature = '4';
      lmResults = [aceReq];
    }

    if (needsLm) {
      const cacheKey = computeLmCacheKey(aceReq);
      const useLmCache = job.params.cacheLmCodes !== false; // default true
      const cached = useLmCache ? lmCache.get(cacheKey) : undefined;

      if (cached) {
        // Cache hit — skip LM entirely
        lmResults = cached.results;
        job.lmResults = lmResults;
        cached.timestamp = Date.now(); // refresh LRU

        logGeneration(job.id, 'INFO', `[LM Phase] Cache HIT (key=${cacheKey}), skipping LM. ${lmResults.length} cached result(s)`);
        job.progress = 40;
        job.stage = 'LM cached, starting synthesis...';
      } else {
        // Cache miss — run LM
        job.status = 'lm_running';
        job.stage = 'Generating lyrics & metadata...';
        job.progress = 10;

        logGeneration(job.id, 'INFO', `[LM Phase] Submitting to ace-server... (cache key=${cacheKey})`);

        const lmJobId = await aceClient.submitLm(aceReq);
        job.aceJobId = lmJobId;

        await pollUntilDone(lmJobId, job, abortController.signal);

        // Fetch LM results (array of enriched AceRequests)
        const resultRes = await aceClient.getJobResult(lmJobId);
        lmResults = await resultRes.json() as AceRequest[];
        job.lmResults = lmResults;

        // Store in cache
        if (useLmCache) {
          lmCache.set(cacheKey, { results: lmResults, timestamp: Date.now() });
          evictLmCache();
          logGeneration(job.id, 'INFO', `[LM Phase] Cached results (key=${cacheKey}, cache size=${lmCache.size})`);
        }

        job.progress = 40;
        job.stage = 'LM complete, starting synthesis...';
        logGeneration(job.id, 'INFO', `[LM Phase] Complete. ${lmResults.length} result(s), bpm=${lmResults[0]?.bpm}, duration=${lmResults[0]?.duration}`);
      }

      // Re-inject fields that the user can change between LM cache hits.
      // The C++ LM serializes full AceRequest — so cached results carry
      // stale values for DiT params and routing fields. Override them
      // from the current request so the synth phase uses current settings.
      for (const result of lmResults) {
        // DiT params (user can change solver, steps, etc. between runs)
        if (aceReq.inference_steps !== undefined) result.inference_steps = aceReq.inference_steps;
        if (aceReq.guidance_scale !== undefined) result.guidance_scale = aceReq.guidance_scale;
        if (aceReq.shift !== undefined) result.shift = aceReq.shift;
        if (aceReq.infer_method !== undefined) result.infer_method = aceReq.infer_method;
        if (aceReq.scheduler !== undefined) result.scheduler = aceReq.scheduler;
        if (aceReq.guidance_mode !== undefined) result.guidance_mode = aceReq.guidance_mode;

        // Routing fields (adapter, model selection)
        if (aceReq.synth_model) result.synth_model = aceReq.synth_model;
        if (aceReq.vae_model) result.vae_model = aceReq.vae_model;
        if (aceReq.adapter) result.adapter = aceReq.adapter;
        if (aceReq.adapter_scale !== undefined) result.adapter_scale = aceReq.adapter_scale;
        if (aceReq.adapter_group_scales) result.adapter_group_scales = aceReq.adapter_group_scales;
        if (aceReq.adapter_mode) result.adapter_mode = aceReq.adapter_mode;

        // Re-inject trigger word — CoT caption replaces the original so the
        // trigger word that was injected into aceReq.caption gets lost.
        if (job.params.triggerWord && job.params.triggerPlacement && job.params.loraPath) {
          const tw = job.params.triggerWord;
          const caption = result.caption || '';
          // Only inject if it's not already present (cache hits may already have it)
          if (!caption.includes(tw)) {
            switch (job.params.triggerPlacement) {
              case 'prepend': result.caption = caption ? `${tw}, ${caption}` : tw; break;
              case 'append':  result.caption = caption ? `${caption}, ${tw}` : tw; break;
              case 'replace': result.caption = tw; break;
            }
          }
        }
      }
    }

    // Phase 2: Synth generation
    job.status = 'synth_running';
    job.stage = 'Synthesizing audio...';
    job.progress = 50;

    logGeneration(job.id, 'INFO', `[Synth Phase] Submitting ${lmResults.length} item(s) to ace-server...`);
    if (aceReq.adapter) {
      logGeneration(job.id, 'INFO', `[Synth Phase] Adapter: ${aceReq.adapter} (scale=${aceReq.adapter_scale ?? 1.0})`);
      if (aceReq.adapter_group_scales) {
        logGeneration(job.id, 'INFO', `[Synth Phase] Group scales: ${JSON.stringify(aceReq.adapter_group_scales)}`);
      }
    }

    // Submit all LM results for synthesis
    const coResident = job.params.coResident === true;

    // Timbre reference: if enabled, read the mastering reference WAV and pass
    // it as ref_audio to the C++ engine's timbre conditioning pipeline.
    let refAudioBuf: Buffer | undefined;
    if (job.params.timbreReference && job.params.masteringReference) {
      const refsDir = path.join(config.data.dir, 'references');
      const refPath = path.join(refsDir, job.params.masteringReference);
      if (fs.existsSync(refPath)) {
        refAudioBuf = fs.readFileSync(refPath);
        logGeneration(job.id, 'INFO', `[Synth Phase] Timbre reference: ${job.params.masteringReference} (${(refAudioBuf.length / 1024 / 1024).toFixed(1)} MB)`);
      } else {
        logGeneration(job.id, 'WARNING', `[Synth Phase] Timbre reference file not found: ${refPath}`);
      }
    }

    let synthJobId: string;
    if (refAudioBuf) {
      synthJobId = await aceClient.submitSynthMultipart(lmResults, undefined, refAudioBuf, 'wav16');
    } else {
      synthJobId = await aceClient.submitSynth(lmResults, 'wav16', coResident);
    }
    job.aceJobId = synthJobId;
    if (coResident) {
      logGeneration(job.id, 'INFO', '[Synth Phase] Co-resident mode: DiT+VAE will stay in VRAM');
    }

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

    // ── Post-generation mastering ──
    let masteredAudioUrl = '';
    const masteringRef = job.params.masteringReference;
    if (masteringRef && job.params.masteringEnabled) {
      try {
        job.stage = 'Mastering...';
        job.progress = 92;
        logGeneration(job.id, 'INFO', `[Mastering] Applying reference mastering: ${masteringRef}`);

        const refsDir = path.join(config.data.dir, 'references');
        const refPath = path.join(refsDir, masteringRef);

        for (const audioUrl of audioUrls) {
          const audioFilename = path.basename(audioUrl);
          const targetPath = path.join(config.data.audioDir, audioFilename);
          const ext2 = path.extname(audioFilename);
          const base2 = path.basename(audioFilename, ext2);
          const masteredFilename = `${base2}_mastered${ext2}`;
          const masteredPath = path.join(config.data.audioDir, masteredFilename);

          await runMastering(targetPath, refPath, masteredPath);
          masteredAudioUrl = `/audio/${masteredFilename}`;
          logGeneration(job.id, 'INFO', `[Mastering] Done → ${masteredAudioUrl}`);
        }
      } catch (masterErr: any) {
        logGeneration(job.id, 'WARNING', `[Mastering] Failed (non-fatal): ${masterErr.message}`);
        console.warn(`[Mastering] Non-fatal mastering error:`, masterErr.message);
        // Don't fail the whole generation
      }
    }

    // Create song entries in DB
    for (const audioUrl of audioUrls) {
      const songId = uuidv4();
      getDb().prepare(`
        INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                           duration, bpm, key_scale, time_signature, tags, dit_model,
                           generation_params, mastered_audio_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        songId, job.userId, title, lyrics, style, firstResult.caption || '',
        audioUrl, duration, bpm, keyScale, timeSignature,
        JSON.stringify([]), aceReq.synth_model || '', JSON.stringify(job.params),
        masteredAudioUrl,
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

    logGeneration(job.id, 'INFO', `[Result] ${audioUrls.length} audio file(s) saved, ${songIds.length} song(s) created`);
    logGeneration(job.id, 'INFO', `[Result] Duration: ${duration}s, BPM: ${bpm}, Key: ${keyScale}`);
    finishGenerationLog(job.id, aceReq.task_type || 'text2music');

  } catch (err: any) {
    if (err.message === 'Cancelled') {
      job.status = 'cancelled';
      job.stage = 'Cancelled';
      failGenerationLog(job.id, 'Cancelled by user', aceReq.task_type || 'text2music');
    } else {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.stage = 'Failed';
      console.error(`[Generate] Job ${job.id} failed:`, err.message);
      failGenerationLog(job.id, err.message || 'Unknown error', aceReq.task_type || 'text2music');
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
