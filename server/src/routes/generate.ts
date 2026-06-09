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
import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';
import { aceClient, type AceRequest } from '../services/aceClient.js';
import { getDb } from '../db/database.js';
import { config } from '../config.js';
import { getUserId } from './auth.js';
import { startGenerationLog, logGeneration, logGenerationParams, finishGenerationLog, failGenerationLog } from '../services/logger.js';
import { engineReady, engineBootStatus } from '../engineState.js';
import { autoTrimSilence } from '../services/autoTrim.js';
import { writeHslat, latentFrameCount, latentDuration, type HslatMetadata } from '../services/latentFormat.js';
import { subscribeLines, pushLog } from './logs.js';
import { translateParams } from '../services/generation/translateParams.js';
import { computeLmCacheKey, getLmCache, setLmCache, getLmCacheSize, type LmCacheEntry } from '../services/generation/lmCache.js';
import { loadSourceAudio, loadSourceLatent, applyTempoAndPitch, loadTimbreReference } from '../services/generation/sourceAudio.js';
import { runPostProcessingChain } from '../services/generation/postProcessing.js';
import { getCachedLatent, saveCachedLatent } from '../services/generation/sourceLatentCache.js';

const router = Router();

/** Internal job state */
/** Timing data for a single pipeline stage. */
interface StageTiming {
  name: string;
  ms: number;
}

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
    timing?: StageTiming[];
    totalMs?: number;
  };
  error?: string;
  params: any;
  createdAt: number;
  /** Stream preview WAV files emitted by the DEMON-style ring buffer */
  streamPreviews?: Array<{
    path: string;
    step: number;
    totalSteps: number;
    slot: number;
    timestamp: number;
  }>;
}

const jobs = new Map<string, GenerationJob>();

// TTL cleanup: prune terminal jobs older than 1 hour every 10 minutes.
// Prevents unbounded memory growth during long batch sessions.
const JOB_TTL_MS = 60 * 60 * 1000; // 1 hour
setInterval(() => {
  const now = Date.now();
  let pruned = 0;
  for (const [id, job] of jobs) {
    if (['succeeded', 'failed', 'cancelled'].includes(job.status) && now - job.createdAt > JOB_TTL_MS) {
      jobs.delete(id);
      pruned++;
    }
  }
  if (pruned > 0) console.log(`[Generate] Pruned ${pruned} terminal job(s) from memory`);
}, 10 * 60 * 1000).unref();

// translateParams is now imported from ../services/generation/translateParams.ts

/** Poll ace-server job until completion, with stall detection watchdog.
 *  If job.stage/progress don't change for STALE_TIMEOUT_MS, the job is
 *  considered stalled and is cancelled + failed. This prevents a single
 *  wedged generation from blocking the entire queue for 45 minutes. */
async function pollUntilDone(aceJobId: string, job: GenerationJob, signal: AbortSignal, timeoutMinutes?: number): Promise<void> {
  const POLL_INTERVAL = 500;          // ms — 500ms is tight enough for UI, avoids hammering
  // User-configurable wall-clock timeout, clamped to [5, 120] min. Default 45 min.
  const clampedTimeout = Math.max(5, Math.min(120, timeoutMinutes || 45));
  const MAX_WALL_MS = clampedTimeout * 60 * 1000;
  const STALE_TIMEOUT_MS = 120_000;   // 2 min with no progress = stalled
  const startedAt = Date.now();
  let lastProgressAt = Date.now();
  let lastStage = job.stage;
  let lastProgress = job.progress;

  while (true) {
    if (signal.aborted || job.status === 'cancelled') {
      await aceClient.cancelJob(aceJobId).catch(() => {});
      throw new Error('Cancelled');
    }

    // Detect progress changes (set by subscribeLines callbacks in runGeneration)
    if (job.stage !== lastStage || job.progress !== lastProgress) {
      lastProgressAt = Date.now();
      lastStage = job.stage;
      lastProgress = job.progress;
    }

    // Stall detection: no progress update for STALE_TIMEOUT_MS
    const stalledFor = Date.now() - lastProgressAt;
    if (stalledFor > STALE_TIMEOUT_MS) {
      await aceClient.cancelJob(aceJobId).catch(() => {});
      throw new Error(
        `Generation stalled — no progress for ${Math.round(stalledFor / 1000)}s ` +
        `(last stage: "${lastStage}")`
      );
    }

    // Absolute wall-clock timeout
    if (Date.now() - startedAt > MAX_WALL_MS) {
      await aceClient.cancelJob(aceJobId).catch(() => {});
      throw new Error(`Generation timed out (${clampedTimeout} min limit)`);
    }

    // Poll ace-server — wrap in try-catch so transient HTTP timeouts
    // (e.g. ace-server busy mid-DiT-step) don't kill the loop
    try {
      const status = await aceClient.pollJob(aceJobId);
      if (status.status === 'done') return;
      if (status.status === 'failed') throw new Error('Generation failed on ace-server');
      if (status.status === 'cancelled') throw new Error('Cancelled by ace-server');
    } catch (pollErr: any) {
      // Re-throw non-transient errors (actual generation failures)
      if (pollErr.message?.includes('Generation failed') ||
          pollErr.message?.includes('Cancelled')) {
        throw pollErr;
      }
      // Transient poll error (timeout, connection refused) — log and retry
      console.warn(`[Generate] Poll error for job ${aceJobId}: ${pollErr.message} (will retry)`);
    }

    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
  }
}

/** Run the full generation pipeline */
async function runGeneration(job: GenerationJob): Promise<void> {
  const pipelineStart = performance.now();
  const timing: StageTiming[] = [];

  // User-configurable timeout from settings (passed via request body)
  const timeoutMinutes: number | undefined = job.params.generationTimeoutMinutes;

  /** Time a synchronous or async block and record its duration. */
  async function timed<T>(name: string, fn: () => T | Promise<T>): Promise<T> {
    const t0 = performance.now();
    const result = await fn();
    timing.push({ name, ms: Math.round(performance.now() - t0) });
    return result;
  }

  // Bail out if cancelled while waiting in the queue
  if (job.status === 'cancelled') return;

  const aceReq = translateParams(job.params);

  // Write the resolved seed back into job.params so the DB stores the actual
  // seed used — critical for reproducibility when randomSeed is true.
  if (aceReq.seed !== undefined) {
    job.params.seed = aceReq.seed;
  }

  console.log(`[Generate] Job ${job.id} — ditModel=${job.params.ditModel || '(none)'}, synth_model=${aceReq.synth_model || '(none)'}, emb_model=${aceReq.emb_model || '(auto)'}, seed=${aceReq.seed ?? '(engine default)'}, source=${job.params.source || 'create'}`);
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
      const cached = useLmCache ? getLmCache(cacheKey) : undefined;

      if (cached) {
        // Cache hit — reconstruct full AceRequests from current params
        // + cached LM output. Only LM-generated fields come from cache;
        // everything else (DiT, adapter, DCW, etc.) uses current aceReq.
        lmResults = cached.lmOutputs.map(lmOut => ({
          ...aceReq,
          audio_codes: lmOut.audio_codes,
          caption: lmOut.caption,
          lyrics: lmOut.lyrics,
          bpm: lmOut.bpm,
          duration: lmOut.duration,
          keyscale: lmOut.keyscale,
          timesignature: lmOut.timesignature,
        }));
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

        const lmStart = performance.now();
        const lmModelLoads: Array<{ type: string; ms: number; sizeMb?: number }> = [];

        // Subscribe to engine logs for LM progress
        const unsubLm = subscribeLines((line) => {
          if (line.source !== 'engine') return;
          // Capture model load times from [Store] Load <TYPE>: <N> ms
          const storeLoad = line.text.match(/\[Store\] Load (.+?):\s+(\d+)\s*ms/);
          if (storeLoad) {
            lmModelLoads.push({ type: storeLoad[1], ms: parseInt(storeLoad[2], 10) });
            job.stage = `Loading ${storeLoad[1]}...`;
            return;
          }
          // Capture unload sizes for context
          const storeUnload = line.text.match(/\[Store\] Unload (.+?) \(([\d.]+) MB\)/);
          if (storeUnload) {
            const last = lmModelLoads.findLast(m => m.type === storeUnload[1]);
            if (last) last.sizeMb = parseFloat(storeUnload[2]);
            return;
          }
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

        const coResident = job.params.coResident === true;
        const lmJobId = await aceClient.submitLm(aceReq, undefined, coResident);
        job.aceJobId = lmJobId;

        await pollUntilDone(lmJobId, job, abortController.signal, timeoutMinutes);

        // Fetch LM results (array of enriched AceRequests)
        const resultRes = await aceClient.getJobResult(lmJobId);
        lmResults = await resultRes.json() as AceRequest[];

        // The LM engine response only contains LM-relevant fields (caption,
        // lyrics, audio_codes, bpm, etc.). Synth-side sideband fields
        // (adapter, group scales, DCW, solver, scheduler, etc.) must be
        // re-injected from the current request so they reach the /synth
        // endpoint. Without this, adapter_group_scales is missing from the
        // synth JSON, causing the C++ engine to use default scales.
        const synthFields: Partial<AceRequest> = {
          synth_model: aceReq.synth_model,
          vae_model: aceReq.vae_model,
          emb_model: aceReq.emb_model,
          adapter: aceReq.adapter,
          adapter_scale: aceReq.adapter_scale,
          adapter_group_scales: aceReq.adapter_group_scales,
          adapter_mode: aceReq.adapter_mode,
          infer_method: aceReq.infer_method,
          scheduler: aceReq.scheduler,
          guidance_mode: aceReq.guidance_mode,
          guidance_scale: aceReq.guidance_scale,
          dcw_enabled: aceReq.dcw_enabled,
          dcw_mode: aceReq.dcw_mode,
          dcw_scaler: aceReq.dcw_scaler,
          dcw_high_scaler: aceReq.dcw_high_scaler,
          latent_shift: aceReq.latent_shift,
          latent_rescale: aceReq.latent_rescale,
          custom_timesteps: aceReq.custom_timesteps,
          cfg_cutoff_ratio: aceReq.cfg_cutoff_ratio,
          cache_ratio: aceReq.cache_ratio,
          use_cot_caption: aceReq.use_cot_caption,
          negative_prompt: aceReq.negative_prompt,
          use_ort_vae: aceReq.use_ort_vae,
        };
        for (const result of lmResults) {
          Object.assign(result, synthFields);
        }
        job.lmResults = lmResults;

        // Store only LM-generated fields in cache (never DiT/adapter/DCW/etc.)
        if (useLmCache) {
          const lmOutputs: LmCacheEntry[] = lmResults.map(r => ({
            audio_codes: r.audio_codes || '',
            caption: r.caption || '',
            lyrics: r.lyrics || '',
            bpm: r.bpm || 0,
            duration: r.duration || 0,
            keyscale: r.keyscale || '',
            timesignature: r.timesignature || '',
          }));
          setLmCache(cacheKey, lmOutputs);
          logGeneration(job.id, 'INFO', `[LM Phase] Cached LM outputs (key=${cacheKey}, cache size=${getLmCacheSize()})`);
        }

        job.progress = 40;
        job.stage = 'LM complete, preparing synthesis...';
        logGeneration(job.id, 'INFO', `[LM Phase] Complete. ${lmResults.length} result(s), bpm=${lmResults[0]?.bpm}, duration=${lmResults[0]?.duration}`);

        // Unsubscribe LM progress watcher
        unsubLm();
        const lmTotalMs = Math.round(performance.now() - lmStart);
        // Report LM model loads as sub-phases
        const lmLoadTotalMs = lmModelLoads.reduce((sum, m) => sum + m.ms, 0);
        if (lmLoadTotalMs > 50) {
          for (const m of lmModelLoads) {
            if (m.ms > 50) {
              const sizeInfo = m.sizeMb ? ` (${m.sizeMb.toFixed(0)} MB)` : '';
              timing.push({ name: `  Load ${m.type}${sizeInfo}`, ms: m.ms });
            }
          }
        }
        timing.push({ name: 'LM Phase', ms: lmTotalMs });
      }

      // Re-inject trigger word into LM results — CoT caption replaces the
      // original, so the trigger word injected by translateParams gets lost.
      // This applies to both cache hits (CoT caption from cache) and fresh
      // LM results (CoT caption from engine).
      if (job.params.triggerWord && job.params.triggerPlacement && job.params.loraPath) {
        for (const result of lmResults) {
          const tw = job.params.triggerWord;
          const caption = result.caption || '';
          if (!caption.includes(tw)) {
            const before = caption.substring(0, 80);
            switch (job.params.triggerPlacement) {
              case 'prepend': result.caption = caption ? `${tw}, ${caption}` : tw; break;
              case 'append':  result.caption = caption ? `${caption}, ${tw}` : tw; break;
              case 'replace': result.caption = tw; break;
            }
            logGeneration(job.id, 'INFO', `[Trigger] Re-injected "${tw}" (${job.params.triggerPlacement}) — before: "${before}…" → after: "${(result.caption || '').substring(0, 80)}…"`);
          } else {
            logGeneration(job.id, 'INFO', `[Trigger] Already present: "${tw}" found in caption, skipping re-injection`);
          }
        }
      } else if (job.params.triggerWord) {
        // Trigger word configured but condition incomplete — log why
        logGeneration(job.id, 'WARNING', `[Trigger] triggerWord="${job.params.triggerWord}" but placement=${job.params.triggerPlacement}, loraPath=${job.params.loraPath ? 'set' : 'MISSING'} — skipping re-injection`);
      }
    }

    // ── Batch expansion ──────────────────────────────────────
    // When the LM was skipped (skipLm, audio_codes pre-filled, cover task),
    // lmResults has only 1 entry. If the user requested batchSize > 1 we
    // clone the template with unique seeds so the DiT produces N distinct
    // tracks from the same audio codes.
    const requestedBatch = job.params.batchSize || 1;
    if (lmResults.length < requestedBatch) {
      const template = lmResults[0];
      while (lmResults.length < requestedBatch) {
        lmResults.push({
          ...template,
          seed: Math.floor(Math.random() * 2_147_483_647),
        });
      }
      logGeneration(job.id, 'INFO',
        `[Batch] Expanded to ${lmResults.length} track(s) (LM skipped, varying DiT seed)`);
    }

    // Phase 2: Synth generation — one track at a time
    // When batchSize > 1, lmResults has N items. We synth each individually
    // so we get N clean audio files (avoids multipart parsing).
    job.status = 'synth_running';
    job.stage = 'Loading models for synthesis...';
    job.progress = 45;

    const totalTracks = lmResults.length;

    logGeneration(job.id, 'INFO', `[Synth Phase] Synthesizing ${totalTracks} track(s)...`);
    if (aceReq.adapter) {
      logGeneration(job.id, 'INFO', `[Synth Phase] Adapter: ${aceReq.adapter} (scale=${aceReq.adapter_scale ?? 1.0})`);
      if (aceReq.adapter_group_scales) {
        logGeneration(job.id, 'INFO', `[Synth Phase] Group scales: ${JSON.stringify(aceReq.adapter_group_scales)}`);
      }
    }

    const coResident = job.params.coResident === true;

    // ── Source audio (for cover/repaint/lego/extract tasks) ──
    const log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => logGeneration(job.id, level, msg);
    const srcPrepStart = performance.now();
    let srcAudioBuf: Buffer | undefined;
    let srcAudioPath: string | undefined;  // filesystem path for cache key
    if (isCoverTask && job.params.sourceAudioUrl) {
      srcAudioBuf = loadSourceAudio(job.params.sourceAudioUrl, job.id, log);
      // Resolve URL to filesystem path (mirrors loadSourceAudio resolution)
      const u = job.params.sourceAudioUrl;
      srcAudioPath = u.startsWith('/references/')
        ? path.join(config.data.dir, 'references', u.replace('/references/', ''))
        : u.startsWith('/audio/')
          ? path.join(config.data.audioDir, u.replace('/audio/', ''))
          : path.isAbsolute(u) ? u : path.join(config.data.dir, u);
    }

    // ── Source latent (alternative to source audio — skips VAE encode) ──
    let srcLatentBuf: Buffer | undefined;
    if (job.params.sourceLatentUrl) {
      srcLatentBuf = loadSourceLatent(job.params.sourceLatentUrl as string, log);
    }

    // ── Tempo/pitch pre-processing (cover source audio) ──────
    if (srcAudioBuf) {
      srcAudioBuf = applyTempoAndPitch(srcAudioBuf, job.params.tempoScale, job.params.pitchShift, log);
    }

    // ── Auto-cache source latent (VAE encode via /vae endpoint) ──
    if (srcAudioBuf && !srcLatentBuf && srcAudioPath) {
      const tempo = job.params.tempoScale as number | undefined;
      const pitch = job.params.pitchShift as number | undefined;
      srcLatentBuf = getCachedLatent(srcAudioPath, tempo, pitch, aceReq.vae_model);
      if (!srcLatentBuf) {
        try {
          job.stage = 'Encoding source audio (VAE)...';
          logGeneration(job.id, 'INFO', '[Latent Cache] Source cache MISS — VAE-encoding source audio...');
          srcLatentBuf = await aceClient.vaeEncode(srcAudioBuf, aceReq.vae_model);
          saveCachedLatent(srcAudioPath, srcLatentBuf, tempo, pitch, aceReq.vae_model);
        } catch (err) {
          logGeneration(job.id, 'WARNING', `[Latent Cache] VAE encode failed, proceeding with raw audio: ${err}`);
          srcLatentBuf = undefined;
        }
      } else {
        logGeneration(job.id, 'INFO', '[Latent Cache] Source cache HIT — VAE encode will be skipped');
      }
    }

    // ── Timbre reference ──
    let refAudioBuf: Buffer | undefined;
    let refLatentBuf: Buffer | undefined;
    const masteringRef = job.params.masteringReference;
    refAudioBuf = await loadTimbreReference(job.params, masteringRef, aceReq.seed, job.id, log);

    // ── Auto-cache timbre latent (VAE encode via /vae endpoint) ──
    if (refAudioBuf) {
      // Resolve timbre ref path for cache key
      const rawTimbre = job.params.timbreReference;
      const timbreRef = (rawTimbre === true && typeof masteringRef === 'string')
        ? masteringRef
        : (typeof rawTimbre === 'string' ? rawTimbre : undefined);
      let refPath: string | undefined;
      if (timbreRef) {
        refPath = timbreRef.startsWith('/references/')
          ? path.join(config.data.dir, 'references', timbreRef.replace('/references/', ''))
          : path.isAbsolute(timbreRef)
            ? timbreRef
            : path.join(config.data.dir, 'references', timbreRef);
      }

      if (refPath) {
        refLatentBuf = getCachedLatent(refPath, undefined, undefined, aceReq.vae_model);
        if (!refLatentBuf) {
          try {
            job.stage = 'Encoding timbre reference (VAE)...';
            logGeneration(job.id, 'INFO', '[Latent Cache] Timbre cache MISS — VAE-encoding timbre reference...');
            refLatentBuf = await aceClient.vaeEncode(refAudioBuf, aceReq.vae_model);
            saveCachedLatent(refPath, refLatentBuf, undefined, undefined, aceReq.vae_model);
          } catch (err) {
            logGeneration(job.id, 'WARNING', `[Latent Cache] Timbre VAE encode failed, proceeding with raw audio: ${err}`);
            refLatentBuf = undefined;
          }
        } else {
          logGeneration(job.id, 'INFO', '[Latent Cache] Timbre cache HIT — VAE encode will be skipped');
        }
      }
    }
    const srcPrepMs = Math.round(performance.now() - srcPrepStart);
    if (srcPrepMs > 50) timing.push({ name: 'Source/Timbre Prep', ms: srcPrepMs });

    // When any post-processing is enabled, request wav32 (float) from the engine.
    // wav16 applies peak normalization to 0 dBFS + hard clip — any downstream gain
    // (PP-VAE, Spectral Lifter, Ozone VST) will push samples over and cause clipping.
    // wav32 skips normalization entirely, preserving natural headroom for PP stages.
    const ppEnabled = job.params.postProcessingEnabled !== false;
    const anyPpActive = ppEnabled && (
      !!job.params.ppVaeReencode ||
      !!job.params.spectralLifterEnabled ||
      !!job.params.masteringEnabled
    );
    const synthFormat = (anyPpActive || (job.params.masteringEnabled && job.params.masteringReference)) ? 'wav32' : 'wav16';
    if (synthFormat === 'wav32') {
      logGeneration(job.id, 'INFO', '[Synth Phase] Using wav32 (raw float) for post-processing — normalization deferred');
    }

    // LRC: auto-enable synchronized lyric timestamps for non-instrumental tracks
    // (unless user explicitly disabled via the skipLrc toggle)
    const skipLrc = job.params.skipLrc === true;
    const hasLyrics = lmResults.some(r => r.lyrics && r.lyrics !== '[Instrumental]');
    if (hasLyrics && !skipLrc) {
      for (const r of lmResults) {
        if (r.lyrics && r.lyrics !== '[Instrumental]') {
          (r as any).get_lrc = true;
        }
      }
      logGeneration(job.id, 'INFO', '[Synth Phase] LRC generation enabled (non-instrumental lyrics detected)');
    } else if (hasLyrics && skipLrc) {
      logGeneration(job.id, 'INFO', '[Synth Phase] LRC generation skipped (disabled by user)');
    }

    if (coResident) {
      logGeneration(job.id, 'INFO', '[Synth Phase] Co-resident mode: DiT+VAE will stay in VRAM');
    }

    // Save audio files and create DB entries
    const audioUrls: string[] = [];
    const songIds: string[] = [];
    // Deferred parallel tasks (whisper, cover art) — collected during pipeline, awaited at end
    const deferredTasks: Promise<void>[] = [];
    // Per-track mastered URLs (parallel array to audioUrls)
    const masteredUrls: string[] = [];
    // Per-track latent URLs (parallel array to audioUrls)
    const latentUrls: string[] = [];

    // ── Parallel Cover Art: launch right after LM (earliest possible) ──
    // At this point we have title/style/lyrics/subject from LM results.
    // Image generation (GPU) starts now and overlaps with the entire synth phase.
    // The cheap DB link happens after DB insert provides songIds.
    let coverArtResults: Array<{ coverUrl: string }> = [];
    if (job.params.parallelCoverArt && job.params.coverArtEnabled) {
      const coverArtTask = async () => {
        const coverArtStart = performance.now();
        try {
          const { generateCoverImage, getCoverArtReadiness } = await import('../services/coverArt/coverArtService.js');
          const readiness = getCoverArtReadiness();
          if (!readiness.installed) {
            logGeneration(job.id, 'DEBUG', `[CoverArt] Skipped — not installed (missing: ${readiness.missingFiles.join(', ')})`);
            return;
          }
          // Generate one cover per track
          for (let i = 0; i < totalTracks; i++) {
            const trackResult = lmResults[i] || lmResults[0];
            try {
              const result = await generateCoverImage({
                title: job.params.title || trackResult.caption?.substring(0, 60) || 'Untitled',
                style: job.params.caption || job.params.style || '',
                lyrics: trackResult.lyrics || '',
                subject: job.params.coverArtSubject || job.params.subject || '',
              });
              coverArtResults.push({ coverUrl: result.coverUrl });
              logGeneration(job.id, 'INFO', `[CoverArt] Image generated (${(result.durationMs / 1000).toFixed(1)}s)`);
            } catch (coverTrackErr: any) {
              logGeneration(job.id, 'WARNING', `[CoverArt] Image generation failed (non-fatal): ${coverTrackErr.message}`);
            }
          }
        } catch (coverErr: any) {
          logGeneration(job.id, 'WARNING', `[CoverArt] Failed (non-fatal): ${coverErr.message}`);
        }
        const coverArtMs = Math.round(performance.now() - coverArtStart);
        if (coverArtMs > 50) timing.push({ name: 'Cover Art', ms: coverArtMs });
      };
      deferredTasks.push(coverArtTask());
      logGeneration(job.id, 'INFO', '[CoverArt] Launched in parallel (overlapping with synth)');
    }

    // ── Per-track synth loop ──────────────────────────────────
    // Each lmResult becomes a separate /synth call → separate audio file.
    // Progress: each track gets an equal share of the 45→88% range.
    const SYNTH_PROGRESS_START = 45;
    const SYNTH_PROGRESS_END = 88;
    const progressPerTrack = (SYNTH_PROGRESS_END - SYNTH_PROGRESS_START) / totalTracks;

    for (let trackIdx = 0; trackIdx < totalTracks; trackIdx++) {
      const synthReq = lmResults[trackIdx];
      const trackLabel = totalTracks > 1 ? ` (track ${trackIdx + 1}/${totalTracks})` : '';
      const trackProgressBase = SYNTH_PROGRESS_START + trackIdx * progressPerTrack;

      // Vary DiT seed per track for additional variation
      if (job.params.randomSeed && trackIdx > 0) {
        synthReq.seed = Math.floor(Math.random() * 2_147_483_647);
      }

      // Log the synth caption to verify trigger word presence
      const synthCaptionPreview = (synthReq.caption || '').substring(0, 150);
      console.log(`[Synth] Track ${trackIdx + 1} caption: ${synthCaptionPreview}`);
      logGeneration(job.id, 'DEBUG', `[Synth Phase] Track ${trackIdx + 1} caption: "${synthCaptionPreview}"`);

      job.stage = `Synthesizing${trackLabel}...`;
      job.progress = Math.round(trackProgressBase);


      // Sub-phase timing: capture when each engine phase starts/ends
      let ditFirstStepAt = 0;
      let ditLastStepAt = 0;
      let vaeStartAt = 0;
      let vaeEndAt = 0;           // [VAE-Decode Batch0] Decode: marks actual decode end
      let adapterMergeAt = 0;
      let ditLoadAt = 0;          // [DiT-TRT] Load + refit complete
      let ditLoadCompleteAt = 0;  // when DiT model load finished
      let fsqStartAt = 0;
      let textEncStartAt = 0;
      let textEncEndAt = 0;
      let firstEngineLogAt = 0;  // first log line from engine = job started
      let ditLastStepEndAt = 0;  // timestamp of last DiT step log (not vae)
      let resolveParamsAt = 0;   // [Resolve-T] or [Resolve-Params] marks job setup
      const synthModelLoads: Array<{ type: string; ms: number; sizeMb?: number }> = [];

      const unsubSynth = subscribeLines((line) => {
        if (line.source !== 'engine') return;
        const now = performance.now();
        if (!firstEngineLogAt) firstEngineLogAt = now;
        // Capture model load times from [Store] Load <TYPE>: <N> ms
        const storeLoad = line.text.match(/\[Store\] Load (.+?):\s+(\d+)\s*ms/);
        if (storeLoad) {
          synthModelLoads.push({ type: storeLoad[1], ms: parseInt(storeLoad[2], 10) });
          job.stage = `Loading ${storeLoad[1]}${trackLabel}...`;
          return;
        }
        // Capture unload sizes for context
        const storeUnload = line.text.match(/\[Store\] (?:Unload|Evict) (.+?) \(([\d.]+) MB\)/);
        if (storeUnload) {
          const last = synthModelLoads.findLast(m => m.type === storeUnload[1]);
          if (last && !last.sizeMb) last.sizeMb = parseFloat(storeUnload[2]);
          return;
        }
        const dit = line.text.match(/\[DiT(?:-TRT)?\] Step (\d+)\/(\d+)\s+t=[\d.]+\s+\[(.+?)\]/);
        if (dit) {
          if (!ditFirstStepAt) ditFirstStepAt = now;
          ditLastStepAt = now;
          const step = parseInt(dit[1], 10);
          const total = parseInt(dit[2], 10);
          job.stage = `DiT${trackLabel}: Step ${step}/${total} (${dit[3]})`;
          job.progress = Math.round(trackProgressBase + (step / total) * progressPerTrack * 0.8);
          return;
        }
        const ditSimple = line.text.match(/\[DiT(?:-TRT)?\] Step (\d+)\/(\d+)/);
        if (ditSimple) {
          if (!ditFirstStepAt) ditFirstStepAt = now;
          ditLastStepAt = now;
          const step = parseInt(ditSimple[1], 10);
          const total = parseInt(ditSimple[2], 10);
          job.stage = `DiT${trackLabel}: Step ${step}/${total}`;
          job.progress = Math.round(trackProgressBase + (step / total) * progressPerTrack * 0.8);
          return;
        }
        if (line.text.includes('[VAE-Decode]') ||
            line.text.includes('[VAE-ORT] Tiled decode') ||
            line.text.includes('[VAE] Tiled decode') ||
            line.text.includes('[VAE] Graph:')) {
          // Only trigger on actual decode start, not VAE model loading
          // [VAE] alone fires during model load (e.g. "[VAE] Loaded: 5 blocks")
          if (!vaeStartAt) vaeStartAt = now;
          job.stage = `Decoding audio (VAE)${trackLabel}...`;
          job.progress = Math.round(trackProgressBase + progressPerTrack * 0.9);
        } else if (line.text.includes('[VAE-Decode Batch') && line.text.includes('Decode:')) {
          // End of actual VAE decode (e.g. "[VAE-Decode Batch0] Decode: 442.0 ms (ORT)")
          vaeEndAt = now;
        } else if (line.text.includes('[VAE]') && (line.text.includes('Loaded') || line.text.includes('Backend'))) {
          // VAE model loading — update stage but don't set vaeStartAt
          job.stage = `Loading VAE model${trackLabel}...`;
        } else if (
          (line.text.includes('[Adapter]') && line.text.includes('Merge')) ||
          (line.text.includes('[Adapter-TRT]') && (line.text.includes('Applying') || line.text.includes('Loading')))
        ) {
          if (!adapterMergeAt) adapterMergeAt = now;
          job.stage = `Loading adapter${trackLabel}...`;
        } else if (line.text.includes('[DiT-TRT]') && line.text.includes('Load + refit complete')) {
          if (!ditLoadCompleteAt) ditLoadCompleteAt = now;
        } else if (line.text.includes('[DiT-Generate] Building TRT engine') ||
                   (line.text.includes('[DiT-TRT]') && (line.text.includes('STRONGLY_TYPED') ||
                    line.text.includes('kREFIT') || line.text.includes('This will take')))) {
          // TRT engine compilation phase — update stage to prevent stall detection
          if (!ditLoadAt) ditLoadAt = now;
          job.stage = `Building TRT engine${trackLabel} (first run only, ~5-10 min)...`;
        } else if (line.text.includes('[TRT-WARN]') || line.text.includes('[TRT-ERROR]') ||
                   line.text.includes('[DiT-TRT] Engine build in progress')) {
          // TRT emits warnings during engine build + our heartbeat thread
          // Append elapsed time to stage string so stall detector sees a change
          if (ditLoadAt) {
            const elapsed = Math.round((now - ditLoadAt) / 1000);
            job.stage = `Building TRT engine${trackLabel} (${elapsed}s elapsed)...`;
          }
        } else if (line.text.includes('[DiT-Generate] Loading cached TRT engine')) {
          if (!ditLoadAt) ditLoadAt = now;
          job.stage = `Loading TRT engine${trackLabel}...`;
        } else if (line.text.includes('[Encode-Text') && !line.text.includes('Batch')) {
          // First [Encode-Text] log that isn't a per-batch sub-line
          if (!textEncStartAt) textEncStartAt = now;
        } else if (line.text.includes('[Encode-Text') && line.text.includes('enc_S=')) {
          // Last text encoder output line
          textEncEndAt = now;
        } else if (line.text.includes('Loading synth') || line.text.includes('ensure_synth') ||
                   (line.text.includes('[DiT-TRT]') && line.text.includes('Building'))) {
          if (!ditLoadAt) ditLoadAt = now;
          job.stage = `Loading DiT model${trackLabel}...`;
        } else if (line.text.includes('[FSQ]') || line.text.includes('fsq_detokenize')) {
          if (!fsqStartAt) fsqStartAt = now;
          job.stage = `Decoding audio tokens (FSQ)${trackLabel}...`;
        } else if (line.text.includes('[DiT]') && line.text.includes('batch') || line.text.includes('[DiT-TRT]') && line.text.includes('Batch')) {
          job.stage = `Preparing DiT${trackLabel}...`;
        } else if (line.text.includes('[Resolve-T]') || line.text.includes('[Resolve-Params]')) {
          if (!resolveParamsAt) resolveParamsAt = now;
        }

        // ── Streaming pipeline markers ──────────────────────────────
        // [Stream] tick N step M/S — ring buffer progress
        const streamTick = line.text.match(/\[Stream\] tick (\d+) step (\d+)\/(\d+)/);
        if (streamTick) {
          if (!ditFirstStepAt) ditFirstStepAt = now;
          ditLastStepAt = now;
          const step = parseInt(streamTick[2], 10);
          const total = parseInt(streamTick[3], 10);
          job.stage = `Streaming${trackLabel}: Step ${step}/${total}`;
          job.progress = Math.round(trackProgressBase + (step / total) * progressPerTrack * 0.8);
        }
        // [STREAM_PREVIEW] path=<file> step=N/M slot=K
        const preview = line.text.match(/\[STREAM_PREVIEW\] path=(.+?) step=(\d+)\/(\d+) slot=(\d+)/);
        if (preview) {
          if (!job.streamPreviews) job.streamPreviews = [];
          job.streamPreviews.push({
            path: preview[1],
            step: parseInt(preview[2], 10),
            totalSteps: parseInt(preview[3], 10),
            slot: parseInt(preview[4], 10),
            timestamp: Date.now(),
          });
        }
      });

      // Submit single request to /synth
      const synthTrackStart = performance.now();
      let synthJobId: string;
      if (srcAudioBuf || refAudioBuf || srcLatentBuf || refLatentBuf) {
        const parts = [
          srcAudioBuf ? 'src_audio' : '',
          refAudioBuf ? 'timbre_ref' : '',
          srcLatentBuf ? 'src_latents' : '',
          refLatentBuf ? 'ref_latents' : '',
        ].filter(Boolean).join('+');
        logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: MULTIPART submission (${parts})`);
        synthJobId = await aceClient.submitSynthMultipart(synthReq, srcAudioBuf, refAudioBuf, srcLatentBuf, refLatentBuf, synthFormat, coResident);
      } else {
        logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: plain JSON submission`);
        synthJobId = await aceClient.submitSynth(synthReq, synthFormat, coResident);
      }
      job.aceJobId = synthJobId;

      await pollUntilDone(synthJobId, job, abortController.signal, timeoutMinutes);
      unsubSynth();

      // Fetch single-track audio result
      const audioRes = await aceClient.getJobResult(synthJobId);
      const audioBuffer = Buffer.from(await audioRes.arrayBuffer());
      const contentType = audioRes.headers.get('content-type') || 'audio/mpeg';
      const ext = contentType.includes('wav') ? 'wav' : 'mp3';

      const filename = `${uuidv4()}.${ext}`;
      const filepath = path.join(config.data.audioDir, filename);
      fs.writeFileSync(filepath, audioBuffer);
      audioUrls.push(`/audio/${filename}`);

      // Record sub-phase timing for this track
      const synthEndAt = performance.now();
      const synthTrackMs = Math.round(synthEndAt - synthTrackStart);
      const trackSuffix = totalTracks > 1 ? ` Track ${trackIdx + 1}` : '';

      // ── Model Loading breakdown (from [Store] Load/Unload engine logs) ──
      const synthLoadTotalMs = synthModelLoads.reduce((sum, m) => sum + m.ms, 0);
      if (synthLoadTotalMs > 50) {
        for (const m of synthModelLoads) {
          if (m.ms > 50) {
            const sizeInfo = m.sizeMb ? ` (${m.sizeMb.toFixed(0)} MB)` : '';
            timing.push({ name: `  📦 Load ${m.type}${sizeInfo}${trackSuffix}`, ms: m.ms });
          }
        }
        timing.push({ name: `  📦 Model Loading Total${trackSuffix}`, ms: synthLoadTotalMs });
      }

      // Sub-phase breakdown (indented with leading spaces for visual hierarchy)
      if (fsqStartAt && ditFirstStepAt) {
        const fsqMs = Math.round(ditFirstStepAt - fsqStartAt);
        if (fsqMs > 50) timing.push({ name: `  FSQ Detokenize${trackSuffix}`, ms: fsqMs });
      }
      if (adapterMergeAt && ditFirstStepAt) {
        const adapterLabel = ditLoadCompleteAt ? 'Adapter Refit' : 'Adapter Merge';
        const adapterMs = Math.round(ditFirstStepAt - adapterMergeAt);
        if (adapterMs > 50) timing.push({ name: `  ${adapterLabel}${trackSuffix}`, ms: adapterMs });
      }
      if (ditFirstStepAt && ditLastStepAt) {
        timing.push({ name: `  DiT Denoising${trackSuffix}`, ms: Math.round(ditLastStepAt - ditFirstStepAt) });
      }
      if (vaeStartAt) {
        const vaeEnd = vaeEndAt || synthEndAt;  // fallback if end marker wasn't captured
        timing.push({ name: `  VAE Decode${trackSuffix}`, ms: Math.round(vaeEnd - vaeStartAt) });
      }
      if (textEncStartAt && textEncEndAt) {
        const textEncMs = Math.round(textEncEndAt - textEncStartAt);
        if (textEncMs > 50) timing.push({ name: `  Text Encoding${trackSuffix}`, ms: textEncMs });
      }
      // Gap analysis: break 'overhead' into specific gaps
      const gaps: Array<{ name: string; ms: number }> = [];
      // Gap 1: HTTP submit → first engine log (request latency + job queue)
      if (firstEngineLogAt) {
        gaps.push({ name: 'HTTP→Engine', ms: Math.round(firstEngineLogAt - synthTrackStart) });
      }
      // Gap 2: Text encoding end → model load → adapter → DiT start
      // Break into sub-gaps for better visibility
      if (textEncEndAt && ditLoadCompleteAt) {
        // TRT path: separate DiT model load from adapter refit
        gaps.push({ name: 'DiT Model Load', ms: Math.round(ditLoadCompleteAt - textEncEndAt) });
      } else if (textEncEndAt && adapterMergeAt) {
        gaps.push({ name: 'TextEnc→Adapter', ms: Math.round(adapterMergeAt - textEncEndAt) });
      } else if (textEncEndAt && ditFirstStepAt) {
        gaps.push({ name: 'TextEnc→DiT', ms: Math.round(ditFirstStepAt - textEncEndAt) });
      }
      // Gap 3: DiT last step → VAE start (VAE model loading)
      if (ditLastStepAt && vaeStartAt) {
        gaps.push({ name: 'DiT→VAE', ms: Math.round(vaeStartAt - ditLastStepAt) });
      }
      // Gap 4: VAE/synth done → result fetched + file written (HTTP response + I/O)
      const totalAccountedMs =
        (firstEngineLogAt ? firstEngineLogAt - synthTrackStart : 0)
        + (textEncStartAt && textEncEndAt ? textEncEndAt - (resolveParamsAt || firstEngineLogAt || textEncStartAt) : 0)
        + (ditLoadCompleteAt && textEncEndAt ? ditLoadCompleteAt - textEncEndAt : 0)
        + (adapterMergeAt && ditFirstStepAt ? ditFirstStepAt - adapterMergeAt : 0)
        + (textEncEndAt && !ditLoadCompleteAt && adapterMergeAt ? adapterMergeAt - textEncEndAt : 0)
        + (ditFirstStepAt && ditLastStepAt ? ditLastStepAt - ditFirstStepAt : 0)
        + (ditLastStepAt && vaeStartAt ? vaeStartAt - ditLastStepAt : 0)
        + (vaeStartAt ? (vaeEndAt || synthEndAt) - vaeStartAt : 0);
      const unmeasuredMs = synthTrackMs - Math.round(totalAccountedMs);
      // Show individual gaps that are significant
      for (const g of gaps) {
        if (g.ms > 200) timing.push({ name: `  ⏳ ${g.name}${trackSuffix}`, ms: g.ms });
      }
      if (unmeasuredMs > 500) timing.push({ name: `  Synth Overhead${trackSuffix}`, ms: unmeasuredMs });

      // Parent total
      timing.push({ name: `Synth Total${trackSuffix}`, ms: synthTrackMs });
      logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: saved ${filename} (${(audioBuffer.length / 1024).toFixed(0)} KB, ${(synthTrackMs / 1000).toFixed(1)}s)`);

      // Save companion LRC file if engine returned alignment data
      const lrcHeader = audioRes.headers.get('x-lrc-text');
      if (lrcHeader) {
        try {
          const lrcDecoded = Buffer.from(lrcHeader, 'base64').toString('utf-8');
          const lrcFilename = filename.replace(/\.[^.]+$/, '.lrc');
          const lrcPath = path.join(config.data.audioDir, lrcFilename);
          fs.writeFileSync(lrcPath, lrcDecoded);
          logGeneration(job.id, 'INFO', `[LRC] Track ${trackIdx + 1}: saved ${lrcFilename} (${lrcDecoded.length} bytes)`);
        } catch (err) {
          logGeneration(job.id, 'WARNING', `[LRC] Track ${trackIdx + 1}: failed to save LRC: ${err}`);
        }
      }

      // ── Whisper Lyrics Transcription (optional) ──
      // Can run in parallel with post-processing (CPU-only, no VRAM impact)
      const runWhisperForTrack = async (trackNum: number, wavPath: string, trackLyrics: string, wavFilename: string) => {
        const whisperTimingStart = performance.now();
        try {
          const { ensureWhisperCli, findWhisperModel, transcribeWithWhisper } = await import('../services/whisperTranscribe.js');
          const { reconcileLyrics } = await import('../services/lyricsReconcile.js');

          const whisperReady = await ensureWhisperCli();
          if (!whisperReady) {
            logGeneration(job.id, 'WARNING', '[Whisper] whisper-cli not available and auto-download failed — skipping');
            return;
          }
          if (!findWhisperModel(job.params.whisperModel)) {
            logGeneration(job.id, 'WARNING', '[Whisper] No Whisper model found — skipping transcription');
            return;
          }
          const whisperStart = Date.now();
          logGeneration(job.id, 'INFO', `[Whisper] Track ${trackNum}: starting transcription...`);

          const whisperResult = await transcribeWithWhisper(wavPath, trackLyrics, {
            model: job.params.whisperModel,
            language: job.params.whisperLanguage || 'auto',
            beamSize: job.params.whisperBeamSize || 5,
          });

          if (whisperResult && whisperResult.segments?.length > 0) {
            const modelName = job.params.whisperModel || 'auto';
            const lyricsJson = reconcileLyrics(whisperResult, trackLyrics, modelName, false);

            const lyricsJsonFilename = wavFilename.replace(/\.[^.]+$/, '.lyrics.json');
            const lyricsJsonPath = path.join(config.data.audioDir, lyricsJsonFilename);
            fs.writeFileSync(lyricsJsonPath, JSON.stringify(lyricsJson, null, 2));

            const elapsed = Date.now() - whisperStart;
            const wordCount = lyricsJson.lines.reduce((n: number, l: any) => n + l.words.length, 0);
            logGeneration(job.id, 'INFO',
              `[Whisper] Track ${trackNum}: saved ${lyricsJsonFilename} (${lyricsJson.lines.length} lines, ${wordCount} words, ${elapsed}ms)`
            );
          } else {
            logGeneration(job.id, 'WARNING', `[Whisper] Track ${trackNum}: no segments returned`);
          }
        } catch (err: any) {
          logGeneration(job.id, 'WARNING', `[Whisper] Track ${trackNum}: failed: ${err.message}`);
        }
        const whisperMs = Math.round(performance.now() - whisperTimingStart);
        if (whisperMs > 50) timing.push({ name: `Whisper Track ${trackNum}`, ms: whisperMs });
      };

      if (job.params.whisperLyricsEnabled) {
        const whisperPromise = runWhisperForTrack(trackIdx + 1, filepath, synthReq.lyrics || '', filename);
        if (job.params.parallelWhisper) {
          // Deferred — will be awaited after post-processing
          deferredTasks.push(whisperPromise);
          logGeneration(job.id, 'INFO', `[Whisper] Track ${trackIdx + 1}: launched in parallel`);
        } else {
          await whisperPromise;
        }
      }

      // Fetch and save companion latent file (post-DiT neural representation)
      let latentUrl = '';
      try {
        const rawLatent = await aceClient.getJobLatent(synthJobId);
        if (rawLatent && rawLatent.length > 0) {
          const latentMeta: HslatMetadata = {
            duration: latentDuration(rawLatent),
            bpm: aceReq.bpm,
            key_scale: aceReq.keyscale,
            time_signature: aceReq.timesignature,
            caption: aceReq.caption,
            lyrics: aceReq.lyrics,
            seed: aceReq.seed,
            inference_steps: aceReq.inference_steps,
            guidance_scale: aceReq.guidance_scale,
            shift: aceReq.shift,
            task_type: aceReq.task_type,
            adapter: aceReq.adapter,
            adapter_scale: aceReq.adapter_scale,
            dit_model: aceReq.synth_model,
            vae_model: aceReq.vae_model,
            emb_model: aceReq.emb_model,
            created_at: new Date().toISOString(),
          };
          const hslatBuf = writeHslat(rawLatent, latentMeta);
          const latentFilename = filename.replace(/\.[^.]+$/, '.latent');
          const latentPath = path.join(config.data.audioDir, latentFilename);
          fs.writeFileSync(latentPath, hslatBuf);
          latentUrl = `/audio/${latentFilename}`;
          logGeneration(job.id, 'INFO',
            `[Latent] Track ${trackIdx + 1}: saved ${latentFilename} (${latentFrameCount(rawLatent)} frames, ${(hslatBuf.length / 1024).toFixed(0)} KB HSLAT)`);
        }
      } catch (latErr: any) {
        logGeneration(job.id, 'DEBUG', `[Latent] Track ${trackIdx + 1}: capture skipped: ${latErr.message}`);
      }

      // Store latent URL for DB insert
      latentUrls.push(latentUrl);
    } // end per-track synth loop

    // Collect deferred parallel tasks (whisper, cover art) to await before completion
    // This array was populated inside the per-track loop above
    // and will be joined before DB insert.

    // Get metadata from LM results
    const firstResult = lmResults[0];
    const title = job.params.title || firstResult.caption?.substring(0, 60) || 'Untitled';
    const lyrics = firstResult.lyrics || job.params.lyrics || '';
    // Store user's original style input — NOT the AI-generated caption (which has its own column).
    // job.params.caption = the "Style Description" field from CreatePanel.
    const style = job.params.caption || job.params.style || '';
    const bpm = firstResult.bpm || 0;
    let duration = firstResult.duration || 0;
    const keyScale = firstResult.keyscale || '';
    const timeSignature = firstResult.timesignature || '';

    // ── Auto-trim (silence detection) ─────────────────────────
    // If the user enabled auto-trim, scan the WAV from the end for the
    // natural song ending and trim there. This must happen BEFORE post-
    // processing so Spectral Lifter and mastering operate on the trimmed audio.
    const autoTrimOn = !!job.params.autoTrimEnabled && !!job.params.durationBuffer;
    // job.params.duration is the user's ORIGINAL requested duration (e.g., 215s).
    // The buffer was added only to the engine request (req.duration = 215 + 15 = 230),
    // NOT to job.params.duration. So no subtraction needed.
    const originalDuration = (autoTrimOn && job.params.duration)
      ? job.params.duration
      : 0;

    const autoTrimStart = performance.now();
    if (autoTrimOn && originalDuration > 0) {
      for (const audioUrl of audioUrls) {
        const audioFilename = path.basename(audioUrl);
        const rawWavPath = path.join(config.data.audioDir, audioFilename);
        if (!rawWavPath.endsWith('.wav')) continue;
        try {
          const fadeMs = job.params.autoTrimFadeMs || 2000;
          const result = autoTrimSilence(rawWavPath, originalDuration, fadeMs);
          if (result.trimmed) {
            // Update the duration metadata to reflect the trimmed length
            duration = Math.round(result.trimmedDurationSec);
            logGeneration(job.id, 'INFO',
              `[Auto-Trim] Trimmed ${audioFilename}: ${result.originalDurationSec.toFixed(1)}s → ${result.trimmedDurationSec.toFixed(1)}s (trim at ${result.trimPointSec.toFixed(1)}s)`);
          } else {
            // No trim — but still correct the duration to the original (un-buffered) value
            duration = originalDuration;
            logGeneration(job.id, 'INFO',
              `[Auto-Trim] No trim needed for ${audioFilename} (${result.originalDurationSec.toFixed(1)}s)`);
          }
        } catch (trimErr: any) {
          // Trim failed — fall back to original duration
          duration = originalDuration;
          logGeneration(job.id, 'WARNING', `[Auto-Trim] Failed (non-fatal): ${trimErr.message}`);
          console.warn('[Auto-Trim] Non-fatal error:', trimErr.message);
        }
      }
    }
    const autoTrimMs = Math.round(performance.now() - autoTrimStart);
    if (autoTrimMs > 50) timing.push({ name: 'Auto-Trim', ms: autoTrimMs });


    // ── Post-processing chain ─────────────────────────────────
    // Raw WAV (audio_url) is NEVER modified. Post-processing runs on a copy.
    job.progress = 89;
    job.stage = 'Post-processing...';

    // Pass parallel flags to the PP chain
    const ppParams = {
      ...job.params,
      parallelQualityEval: !!job.params.parallelQualityEval,
    };

    let ppQualityScores: Array<{ unmastered?: any; mastered?: any }> = [];
    try {
      const ppResult = await runPostProcessingChain(
        audioUrls, ppParams, totalTracks, job.id,
        log, (stage) => { job.stage = stage; }
      );
      masteredUrls.push(...ppResult.masteredUrls);
      if (ppResult.timing && ppResult.timing.length > 0) {
        timing.push(...ppResult.timing);
      }
      ppQualityScores = ppResult.qualityScores;
    } catch (err: any) {
      logGeneration(job.id, 'WARNING', `[Post-Processing] Chain failed: ${err.message}`);
    }

    // Create song entries in DB — one per track
    for (let i = 0; i < audioUrls.length; i++) {
      const audioUrl = audioUrls[i];
      const trackMastered = masteredUrls[i] || '';
      const trackLatent = latentUrls[i] || '';
      const trackQualityScores = ppQualityScores[i] || {};
      // Use per-track LM result for metadata when available
      const trackResult = lmResults[i] || firstResult;
      const trackLyrics = trackResult.lyrics || job.params.lyrics || '';
      const trackCaption = trackResult.caption || '';

      // Serialize quality scores (only if evaluator was enabled)
      const qualityJson = (trackQualityScores.unmastered || trackQualityScores.mastered)
        ? JSON.stringify(trackQualityScores)
        : '';

      const songId = uuidv4();
      getDb().prepare(`
        INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                           duration, bpm, key_scale, time_signature, tags, dit_model,
                           generation_params, mastered_audio_url, latent_url, quality_scores)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        songId, job.userId, title, trackLyrics, style, trackCaption,
        audioUrl, duration, bpm, keyScale, timeSignature,
        JSON.stringify([]), aceReq.synth_model || '', JSON.stringify(job.params),
        trackMastered, trackLatent, qualityJson,
      );
      songIds.push(songId);

      // Persist cover art subject for future "Regenerate Cover" calls
      if (job.params.coverArtSubject) {
        getDb().prepare('UPDATE songs SET cover_art_subject = ? WHERE id = ?')
          .run(job.params.coverArtSubject, songId);
      }
    }

    // ── Cover Art: link parallel results or run sequential ────
    if (job.params.parallelCoverArt && coverArtResults.length > 0) {
      // Parallel path: image was generated during synth. Link to DB now.
      const { linkCoverToSong } = await import('../services/coverArt/coverArtService.js');
      for (let i = 0; i < songIds.length; i++) {
        const cover = coverArtResults[i];
        if (cover) {
          linkCoverToSong(cover.coverUrl, songIds[i]);
          logGeneration(job.id, 'INFO', `[CoverArt] Linked cover to song ${songIds[i]}`);
        }
      }
    } else if (!job.params.parallelCoverArt && job.params.coverArtEnabled) {
      // Sequential path: generate + link in one call (original behavior)
      const coverArtStart = performance.now();
      try {
        const { generateCoverArt, getCoverArtReadiness } = await import('../services/coverArt/coverArtService.js');
        const readiness = getCoverArtReadiness();
        if (readiness.installed) {
          job.stage = 'Generating cover art...';
          job.progress = 95;
          for (let i = 0; i < songIds.length; i++) {
            const trackResult = lmResults[i] || firstResult;
            try {
              await generateCoverArt({
                songId: songIds[i],
                title,
                style,
                lyrics: trackResult.lyrics || '',
                subject: job.params.coverArtSubject || job.params.subject || '',
              });
              logGeneration(job.id, 'INFO', `[CoverArt] Generated cover for song ${songIds[i]}`);
            } catch (coverTrackErr: any) {
              logGeneration(job.id, 'WARNING', `[CoverArt] Failed for song ${songIds[i]} (non-fatal): ${coverTrackErr.message}`);
            }
          }
        } else {
          logGeneration(job.id, 'DEBUG', `[CoverArt] Skipped — not installed (missing: ${readiness.missingFiles.join(', ')})`);
        }
      } catch (coverErr: any) {
        logGeneration(job.id, 'WARNING', `[CoverArt] Failed (non-fatal): ${coverErr.message}`);
      }
      const coverArtMs = Math.round(performance.now() - coverArtStart);
      if (coverArtMs > 50) timing.push({ name: 'Cover Art', ms: coverArtMs });
    }

    // ── Await all deferred parallel tasks (with timeout) ─────
    if (deferredTasks.length > 0) {
      logGeneration(job.id, 'INFO', `[Parallel] Awaiting ${deferredTasks.length} deferred task(s)...`);
      const TIMEOUT_MS = 60_000; // 60s safety timeout
      const withTimeout = deferredTasks.map(p =>
        Promise.race([
          p,
          new Promise<void>((_, reject) =>
            setTimeout(() => reject(new Error('Deferred task timed out after 60s')), TIMEOUT_MS)
          ),
        ]).catch(err => {
          logGeneration(job.id, 'WARNING', `[Parallel] Task failed/timed out: ${err.message}`);
        })
      );
      await Promise.allSettled(withTimeout);
      logGeneration(job.id, 'INFO', '[Parallel] All deferred tasks completed');
    }

    const totalMs = Math.round(performance.now() - pipelineStart);
    timing.push({ name: 'TOTAL', ms: totalMs });

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
      masteredAudioUrl: masteredUrls.find(u => !!u) || undefined,
      timing,
      totalMs,
    };

    logGeneration(job.id, 'INFO', `[Result] ${audioUrls.length} audio file(s) saved, ${songIds.length} song(s) created`);
    logGeneration(job.id, 'INFO', `[Result] Duration: ${duration}s, BPM: ${bpm}, Key: ${keyScale}`);

    // ── Timing summary table ──
    const maxName = Math.max(...timing.map(t => t.name.length), 6);
    logGeneration(job.id, 'INFO', `[Timing] ── Pipeline Breakdown ──`);
    for (const t of timing) {
      const pct = totalMs > 0 ? ((t.ms / totalMs) * 100).toFixed(1) : '0.0';
      const secs = (t.ms / 1000).toFixed(2);
      const bar = '█'.repeat(Math.round((t.ms / totalMs) * 30));
      if (t.name === 'TOTAL') {
        logGeneration(job.id, 'INFO', `[Timing] ${'─'.repeat(maxName + 30)}`);
      }
      logGeneration(job.id, 'INFO', `[Timing] ${t.name.padEnd(maxName)}  ${secs.padStart(7)}s  ${pct.padStart(5)}%  ${bar}`);
    }
    console.log(`[Generate] Job ${job.id} completed in ${(totalMs / 1000).toFixed(1)}s`);

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

const MAX_RETRIES = 1; // retry once on transient failures

function enqueueGeneration(job: GenerationJob): void {
  const execute = async () => {
    generationRunning = true;
    let attempts = 0;

    while (attempts <= MAX_RETRIES) {
      try {
        await runGeneration(job);
        break; // success — exit retry loop
      } catch (err: any) {
        const msg = err.message || '';
        const isRetryable = !msg.includes('Cancelled')
          && !msg.includes('Unauthorized')
          && job.status !== 'cancelled';

        if (isRetryable && attempts < MAX_RETRIES) {
          attempts++;
          console.log(`[Generate] Job ${job.id} failed (attempt ${attempts}), retrying: ${msg}`);
          logGeneration(job.id, 'WARNING', `[Retry] Attempt ${attempts} failed: ${msg} — retrying with new seed...`);

          // Reset job state for retry
          job.status = 'pending';
          job.stage = `Retrying (attempt ${attempts + 1})...`;
          job.progress = 0;
          job.error = undefined;
          job.aceJobId = undefined;

          // Randomize seed on retry — bad LM output (same seed) may have caused the stall
          job.params.seed = Math.floor(Math.random() * 2_147_483_647);
          job.params.randomSeed = true;

          // Brief pause before retry
          await new Promise(r => setTimeout(r, 2000));
        } else {
          // Final failure — no more retries
          job.status = 'failed';
          job.error = msg;
          job.stage = 'Failed';
          console.error(`[Generate] Job ${job.id} failed permanently${attempts > 0 ? ` after ${attempts + 1} attempt(s)` : ''}: ${msg}`);
          failGenerationLog(job.id, msg, 'unknown');
          break;
        }
      }
    }

    generationRunning = false;
    const next = pendingQueue.shift();
    if (next) next();
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
  // Reject requests while engine is still bootstrapping (downloading DLLs, etc.)
  if (!engineReady) {
    res.status(503).json({
      error: `Engine not ready: ${engineBootStatus}`,
      detail: 'The CUDA runtime is still being set up. Please wait a moment and try again.',
    });
    return;
  }

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

// GET /api/generate/queue — queue health / status inspection
router.get('/queue', (_req, res) => {
  const activeJob = Array.from(jobs.values()).find(j =>
    j.status === 'lm_running' || j.status === 'synth_running' || j.status === 'saving'
  );

  // Count all non-terminal jobs in the jobs Map (includes pending jobs waiting in queue)
  const depth = Array.from(jobs.values()).filter(j =>
    ['pending', 'lm_running', 'synth_running', 'saving'].includes(j.status)
  ).length;

  res.json({
    depth,
    running: generationRunning,
    current: activeJob ? {
      id: activeJob.id,
      status: activeJob.status,
      stage: activeJob.stage,
      progress: activeJob.progress,
      age: Math.round((Date.now() - activeJob.createdAt) / 1000),
      aceJobId: activeJob.aceJobId,
    } : null,
    pending: pendingQueue.length,
  });
});

// POST /api/generate/reset-queue — force-reset: cancel everything, drain queue
router.post('/reset-queue', (_req, res) => {
  let cancelled = 0;

  // Cancel all non-terminal jobs in the jobs Map
  for (const [, job] of jobs) {
    if (['pending', 'lm_running', 'synth_running', 'saving'].includes(job.status)) {
      job.status = 'failed';
      job.error = 'Queue reset by user';
      job.stage = 'Reset';
      if (job.aceJobId) {
        aceClient.cancelJob(job.aceJobId).catch(() => {});
      }
      if ((job as any)._abort) {
        (job as any)._abort.abort();
      }
      cancelled++;
    }
  }

  // Drain the pending execution queue
  const drained = pendingQueue.length;
  pendingQueue.length = 0;
  generationRunning = false;

  console.log(`[Generate] Queue reset: ${cancelled} job(s) cancelled, ${drained} pending drained`);

  res.json({
    success: true,
    cancelled,
    drained,
  });
});

// GET /api/generate/stream/:id — SSE endpoint for streaming preview audio
// Frontend connects via EventSource. Receives:
//   event: status   — job status/stage/progress updates
//   event: preview  — new preview WAV file available for playback
//   event: done     — generation complete, final audio URL
//   event: error    — generation failed
router.get('/stream/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  // SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');  // nginx compatibility
  res.flushHeaders();

  let lastPreviewIdx = 0;
  let lastStage = '';
  let lastProgress = -1;
  let closed = false;

  const sendSSE = (event: string, data: unknown) => {
    if (closed) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  // Poll interval — check for new previews and status changes
  const interval = setInterval(() => {
    if (closed) return;

    // Send status updates when changed
    if (job.stage !== lastStage || job.progress !== lastProgress) {
      lastStage = job.stage || '';
      lastProgress = job.progress || 0;
      sendSSE('status', {
        status: job.status,
        stage: job.stage,
        progress: job.progress,
      });
    }

    // Send new preview events
    const previews = job.streamPreviews;
    if (previews && previews.length > lastPreviewIdx) {
      for (let i = lastPreviewIdx; i < previews.length; i++) {
        // Convert filesystem path to a URL the frontend can fetch
        // Preview WAVs are written to the stream_chunk_dir, which should be
        // inside data/audio/ or a temp dir served by the static middleware
        const p = previews[i];
        sendSSE('preview', {
          url: p.path,  // absolute path — frontend will need a serving route
          step: p.step,
          totalSteps: p.totalSteps,
          slot: p.slot,
        });
      }
      lastPreviewIdx = previews.length;
    }

    // Terminal states — send final event and close
    if (job.status === 'succeeded') {
      sendSSE('done', {
        result: job.result,
      });
      cleanup();
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      sendSSE('error', {
        status: job.status,
        error: job.error,
      });
      cleanup();
    }
  }, 250);  // 4Hz polling — fast enough for audio preview updates

  const cleanup = () => {
    closed = true;
    clearInterval(interval);
    res.end();
  };

  // Client disconnect
  req.on('close', cleanup);
});

export default router;
