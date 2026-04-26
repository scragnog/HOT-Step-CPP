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
import { applyVstChain } from './vst.js';
import { runSpectralLifter } from '../services/spectralLifter.js';
import { subscribeLines, pushLog } from './logs.js';

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
    masteredAudioUrl?: string;
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

  // DCW (Differential Correction in Wavelet domain)
  if (params.dcwEnabled !== undefined) req.dcw_enabled = params.dcwEnabled;
  if (params.dcwMode) req.dcw_mode = params.dcwMode;
  if (params.dcwScaler !== undefined) req.dcw_scaler = params.dcwScaler;
  if (params.dcwHighScaler !== undefined) req.dcw_high_scaler = params.dcwHighScaler;

  // Latent post-processing
  if (params.latentShift !== undefined) req.latent_shift = params.latentShift;
  if (params.latentRescale !== undefined) req.latent_rescale = params.latentRescale;
  if (params.customTimesteps) req.custom_timesteps = params.customTimesteps;

  // Post-VAE spectral denoiser (HOT-Step)
  if (params.denoiseStrength !== undefined) req.denoise_strength = params.denoiseStrength;
  if (params.denoiseSmoothing !== undefined) req.denoise_smoothing = params.denoiseSmoothing;
  if (params.denoiseMix !== undefined) req.denoise_mix = params.denoiseMix;

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
  // Bail out if cancelled while waiting in the queue
  if (job.status === 'cancelled') return;

  const aceReq = translateParams(job.params);
  console.log(`[Generate] Job ${job.id} — ditModel=${job.params.ditModel || '(none)'}, synth_model=${aceReq.synth_model || '(none)'}, source=${job.params.source || 'create'}`);
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
        job.stage = 'Generating lyrics & audio codes...';
        job.progress = 10;

        // Subscribe to engine logs for LM progress
        const unsubLm = subscribeLines((line) => {
          if (line.source !== 'engine') return;
          const lm1 = line.text.match(/\[LM-Phase1\] Step (\d+).*?([\d.]+) tok\/s/);
          if (lm1) {
            job.stage = `LM Phase 1: Step ${lm1[1]} (${lm1[2]} tok/s)`;
            return;
          }
          const lm2 = line.text.match(/\[LM-Phase2\] Step (\d+).*?(\d+) total codes.*?([\d.]+) tok\/s/);
          if (lm2) {
            job.stage = `Audio codes: Step ${lm2[1]} (${lm2[2]} codes, ${lm2[3]} tok/s)`;
            job.progress = 20;
            return;
          }
          if (line.text.includes('[LM-Phase1] Prefill')) {
            job.stage = 'LM: Prefilling prompt...';
          } else if (line.text.includes('[LM-Phase2] Prefill')) {
            job.stage = 'LM: Generating audio codes...';
            job.progress = 15;
          } else if (line.text.includes('[Adapter]') && line.text.includes('Merge')) {
            job.stage = 'Loading adapter...';
          }
        });

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
        job.stage = 'LM complete, preparing synthesis...';
        logGeneration(job.id, 'INFO', `[LM Phase] Complete. ${lmResults.length} result(s), bpm=${lmResults[0]?.bpm}, duration=${lmResults[0]?.duration}`);

        // Unsubscribe LM progress watcher
        unsubLm();
      }

      // Re-inject fields that the user can change between LM cache hits.
      // The C++ LM serializes full AceRequest — so cached results carry
      // stale values for DiT params and routing fields. Override them
      // from the current request so the synth phase uses current settings.
      for (const result of lmResults) {
        // Seed: if randomSeed, set seed=-1 so the engine randomizes.
        // Cached LM results carry the original fixed seed otherwise.
        if (job.params.randomSeed) {
          result.seed = -1;
        } else if (aceReq.seed !== undefined) {
          result.seed = aceReq.seed;
        }

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

        // DCW params (user can toggle between runs)
        if (aceReq.dcw_enabled !== undefined) result.dcw_enabled = aceReq.dcw_enabled;
        if (aceReq.dcw_mode) result.dcw_mode = aceReq.dcw_mode;
        if (aceReq.dcw_scaler !== undefined) result.dcw_scaler = aceReq.dcw_scaler;
        if (aceReq.dcw_high_scaler !== undefined) result.dcw_high_scaler = aceReq.dcw_high_scaler;

        // Latent post-processing params (user can change between runs)
        if (aceReq.latent_shift !== undefined) result.latent_shift = aceReq.latent_shift;
        if (aceReq.latent_rescale !== undefined) result.latent_rescale = aceReq.latent_rescale;
        if (aceReq.custom_timesteps !== undefined) result.custom_timesteps = aceReq.custom_timesteps;

        // Denoise params (user can change between runs)
        if (aceReq.denoise_strength !== undefined) result.denoise_strength = aceReq.denoise_strength;
        if (aceReq.denoise_smoothing !== undefined) result.denoise_smoothing = aceReq.denoise_smoothing;
        if (aceReq.denoise_mix !== undefined) result.denoise_mix = aceReq.denoise_mix;

        // Cover/repaint params — ALWAYS override. Cached LM results may carry
        // stale audio_cover_strength from previous runs (e.g. 0.5 from old Lyric
        // Studio code), causing the engine to enter cover mode unexpectedly.
        // When undefined in the current request, delete from result so the
        // engine uses its default (1.0 = no cover switching).
        result.audio_cover_strength = aceReq.audio_cover_strength;
        result.cover_noise_strength = aceReq.cover_noise_strength;
        result.task_type = aceReq.task_type;

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
    job.stage = 'Loading models for synthesis...';
    job.progress = 45;

    // Subscribe to engine logs for synth progress (DiT steps, VAE, etc.)
    const totalSteps = aceReq.inference_steps || 20;
    const unsubSynth = subscribeLines((line) => {
      if (line.source !== 'engine') return;
      const dit = line.text.match(/\[DiT\] Step (\d+)\/(\d+)\s+t=[\d.]+\s+\[(.+?)\]/);
      if (dit) {
        const step = parseInt(dit[1], 10);
        const total = parseInt(dit[2], 10);
        job.stage = `DiT: Step ${step}/${total} (${dit[3]})`;
        job.progress = 50 + Math.round((step / total) * 35);
        return;
      }
      // Fallback DiT pattern without solver info
      const ditSimple = line.text.match(/\[DiT\] Step (\d+)\/(\d+)/);
      if (ditSimple) {
        const step = parseInt(ditSimple[1], 10);
        const total = parseInt(ditSimple[2], 10);
        job.stage = `DiT: Step ${step}/${total}`;
        job.progress = 50 + Math.round((step / total) * 35);
        return;
      }
      if (line.text.includes('[VAE]') || line.text.includes('vae_decode')) {
        job.stage = 'Decoding audio (VAE)...';
        job.progress = 87;
      } else if (line.text.includes('[Adapter]') && line.text.includes('Merge')) {
        job.stage = 'Loading adapter...';
      } else if (line.text.includes('Loading synth') || line.text.includes('ensure_synth')) {
        job.stage = 'Loading DiT model...';
      } else if (line.text.includes('[FSQ]') || line.text.includes('fsq_detokenize')) {
        job.stage = 'Decoding audio tokens (FSQ)...';
        job.progress = 86;
      } else if (line.text.includes('[DiT]') && line.text.includes('batch')) {
        job.stage = 'Preparing DiT batch...';
        job.progress = 48;
      }
    });

    logGeneration(job.id, 'INFO', `[Synth Phase] Submitting ${lmResults.length} item(s) to ace-server...`);
    if (aceReq.adapter) {
      logGeneration(job.id, 'INFO', `[Synth Phase] Adapter: ${aceReq.adapter} (scale=${aceReq.adapter_scale ?? 1.0})`);
      if (aceReq.adapter_group_scales) {
        logGeneration(job.id, 'INFO', `[Synth Phase] Group scales: ${JSON.stringify(aceReq.adapter_group_scales)}`);
      }
    }

    // Submit all LM results for synthesis
    const coResident = job.params.coResident === true;

    // Timbre reference: if enabled, read the source audio and pass
    // it as ref_audio to the C++ engine's timbre conditioning pipeline.
    // sourceAudioUrl can be an absolute path (from album presets) or a relative
    // reference name (from the Create panel's uploaded references).
    let refAudioBuf: Buffer | undefined;
    const masteringRef = job.params.masteringReference;
    // timbreReference can be:
    //   - a string path (from Lyric Studio / queue with album presets)
    //   - boolean true  (from Create panel checkbox — "also use mastering ref as timbre")
    // When it's boolean true, resolve to the mastering reference path.
    const rawTimbre = job.params.sourceAudioUrl || job.params.timbreReference;
    const timbreRef = (rawTimbre === true && typeof masteringRef === 'string')
      ? masteringRef
      : (typeof rawTimbre === 'string' ? rawTimbre : undefined);
    logGeneration(job.id, 'DEBUG', `[Synth Phase] timbreRef=${timbreRef}, masteringRef=${masteringRef}`);
    if (timbreRef) {
      // Resolve path: absolute paths used directly, relative names looked up in references dir
      const refPath = path.isAbsolute(timbreRef)
        ? timbreRef
        : path.join(config.data.dir, 'references', timbreRef);
      logGeneration(job.id, 'DEBUG', `[Synth Phase] Looking for timbre ref at: ${refPath}`);
      if (fs.existsSync(refPath)) {
        refAudioBuf = fs.readFileSync(refPath);
        logGeneration(job.id, 'INFO', `[Synth Phase] Timbre reference: ${refPath} (${(refAudioBuf.length / 1024 / 1024).toFixed(1)} MB)`);
      } else {
        logGeneration(job.id, 'WARNING', `[Synth Phase] Timbre reference file not found: ${refPath}`);
      }
    }

    // When mastering is enabled, request wav32 (float) from the engine.
    // wav32 skips audio_normalize() which would push the audio to ~0dBFS
    // BEFORE mastering — causing "overcooking" (double normalization).
    // The mastering algorithm handles its own internal level matching.
    const synthFormat = (job.params.masteringEnabled && job.params.masteringReference) ? 'wav32' : 'wav16';
    if (synthFormat === 'wav32') {
      logGeneration(job.id, 'INFO', '[Synth Phase] Using wav32 (raw float) for mastering input — normalization deferred to mastering');
    }

    let synthJobId: string;
    if (refAudioBuf) {
      logGeneration(job.id, 'INFO', `[Synth Phase] Using MULTIPART submission with timbre ref (${refAudioBuf.length} bytes)`);
      synthJobId = await aceClient.submitSynthMultipart(lmResults, undefined, refAudioBuf, synthFormat, coResident);
    } else {
      logGeneration(job.id, 'INFO', `[Synth Phase] Using plain JSON submission (no timbre ref)`);
      synthJobId = await aceClient.submitSynth(lmResults, synthFormat, coResident);
    }
    job.aceJobId = synthJobId;
    if (coResident) {
      logGeneration(job.id, 'INFO', '[Synth Phase] Co-resident mode: DiT+VAE will stay in VRAM');
    }

    await pollUntilDone(synthJobId, job, abortController.signal);

    // Unsubscribe synth progress watcher
    unsubSynth();

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
    const rawTitle = job.params.title || firstResult.caption?.substring(0, 60) || 'Untitled';
    // Format title as "Artist - Song Title" when artist is provided
    const title = job.params.artist
      ? `${job.params.artist} - ${rawTitle}`
      : rawTitle;
    const lyrics = firstResult.lyrics || job.params.lyrics || '';
    // Use subject for style/description when available, otherwise fall back to caption
    const style = job.params.subject || firstResult.caption || job.params.style || '';
    const bpm = firstResult.bpm || 0;
    const duration = firstResult.duration || 0;
    const keyScale = firstResult.keyscale || '';
    const timeSignature = firstResult.timesignature || '';

    // ── Post-processing chain ─────────────────────────────────
    // Raw WAV (audio_url) is NEVER modified. Post-processing runs on a copy.
    // "mastered" = any/all of: Spectral Lifter, VST Chain, Matchering Mastering.
    const spectralLifterOn = !!job.params.spectralLifterEnabled;
    // VST chain self-gates via applyVstChain() — returns false if no plugins active
    const masteringOn = !!masteringRef && !!job.params.masteringEnabled;

    let masteredAudioUrl = '';

    try {
      for (const audioUrl of audioUrls) {
        const audioFilename = path.basename(audioUrl);
        const rawWavPath = path.join(config.data.audioDir, audioFilename);

        if (!rawWavPath.endsWith('.wav')) continue; // Post-processing only on WAV

        const ext2 = path.extname(audioFilename);
        const base2 = path.basename(audioFilename, ext2);
        const processedFilename = `${base2}_mastered${ext2}`;
        const processedPath = path.join(config.data.audioDir, processedFilename);

        // Start with a copy of the raw WAV — original stays pristine
        fs.copyFileSync(rawWavPath, processedPath);
        let anyStageRan = false;

        // Stage 1: Spectral Lifter
        if (spectralLifterOn) {
          job.stage = 'Spectral Lifter...';
          job.progress = 91;
          try {
            const tempLifted = processedPath + '.lifted.wav';
            await runSpectralLifter(processedPath, tempLifted);
            fs.renameSync(tempLifted, processedPath);
            anyStageRan = true;
            logGeneration(job.id, 'INFO', `[Spectral Lifter] Applied to ${audioFilename}`);
          } catch (slErr: any) {
            logGeneration(job.id, 'WARNING', `[Spectral Lifter] Failed (non-fatal): ${slErr.message}`);
            console.warn(`[Spectral Lifter] Non-fatal error:`, slErr.message);
          }
        }

        // Stage 2: VST Chain
        job.stage = 'Applying VST effects...';
        job.progress = 93;
        try {
          const applied = await applyVstChain(processedPath);
          if (applied) {
            anyStageRan = true;
            logGeneration(job.id, 'INFO', `[VST] Chain applied to ${processedFilename}`);
          }
        } catch (vstErr: any) {
          logGeneration(job.id, 'WARNING', `[VST] Chain failed (non-fatal): ${vstErr.message}`);
          console.warn(`[VST] Non-fatal chain error:`, vstErr.message);
        }

        // Stage 3: Matchering Mastering
        if (masteringOn) {
          job.stage = 'Mastering...';
          job.progress = 95;
          try {
            const refPath = path.isAbsolute(masteringRef)
              ? masteringRef
              : path.join(config.data.dir, 'references', masteringRef);
            const tempMastered = processedPath + '.mastered.wav';
            await runMastering(processedPath, refPath, tempMastered);
            fs.renameSync(tempMastered, processedPath);
            anyStageRan = true;
            logGeneration(job.id, 'INFO', `[Mastering] Applied to ${processedFilename}`);
          } catch (masterErr: any) {
            logGeneration(job.id, 'WARNING', `[Mastering] Failed (non-fatal): ${masterErr.message}`);
            console.warn(`[Mastering] Non-fatal mastering error:`, masterErr.message);
          }
        }

        // Only set mastered URL if at least one stage actually ran
        if (anyStageRan) {
          masteredAudioUrl = `/audio/${processedFilename}`;
        } else {
          // No post-processing ran — clean up the copy
          try { fs.unlinkSync(processedPath); } catch {}
        }
      }
    } catch (err: any) {
      logGeneration(job.id, 'WARNING', `[Post-Processing] Chain failed: ${err.message}`);
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
      masteredAudioUrl: masteredAudioUrl || undefined,
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

// ── Async generation queue ────────────────────────────────────────────
// Serializes runGeneration calls so only one job runs at a time.
// The C++ engine is single-GPU — concurrent runGeneration calls cause
// log subscription callbacks to leak progress from one job into another
// because subscribeLines() is a global pub/sub with no job tagging.
const pendingQueue: (() => void)[] = [];
let generationRunning = false;

function enqueueGeneration(job: GenerationJob): void {
  const execute = async () => {
    generationRunning = true;
    try {
      await runGeneration(job);
    } catch (err: any) {
      console.error(`[Generate] Unhandled error in job ${job.id}:`, err);
      job.status = 'failed';
      job.error = err.message;
    } finally {
      generationRunning = false;
      const next = pendingQueue.shift();
      if (next) next();
    }
  };

  if (generationRunning) {
    console.log(`[Generate] Job ${job.id} queued (${pendingQueue.length + 1} waiting)`);
    pendingQueue.push(execute);
  } else {
    execute();
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

  // Enqueue — runs immediately if nothing else is generating,
  // otherwise waits until the current job finishes.
  enqueueGeneration(job);

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
