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

// translateParams is now imported from ../services/generation/translateParams.ts

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

  // Write the resolved seed back into job.params so the DB stores the actual
  // seed used — critical for reproducibility when randomSeed is true.
  if (aceReq.seed !== undefined) {
    job.params.seed = aceReq.seed;
  }

  console.log(`[Generate] Job ${job.id} — ditModel=${job.params.ditModel || '(none)'}, synth_model=${aceReq.synth_model || '(none)'}, seed=${aceReq.seed ?? '(engine default)'}, source=${job.params.source || 'create'}`);
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

        // The LM engine response only contains LM-relevant fields (caption,
        // lyrics, audio_codes, bpm, etc.). Synth-side sideband fields
        // (adapter, group scales, DCW, solver, scheduler, etc.) must be
        // re-injected from the current request so they reach the /synth
        // endpoint. Without this, adapter_group_scales is missing from the
        // synth JSON, causing the C++ engine to use default scales.
        const synthFields: Partial<AceRequest> = {
          synth_model: aceReq.synth_model,
          vae_model: aceReq.vae_model,
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
          use_cot_caption: aceReq.use_cot_caption,
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
            switch (job.params.triggerPlacement) {
              case 'prepend': result.caption = caption ? `${tw}, ${caption}` : tw; break;
              case 'append':  result.caption = caption ? `${caption}, ${tw}` : tw; break;
              case 'replace': result.caption = tw; break;
            }
          }
        }
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
    let srcAudioBuf: Buffer | undefined;
    if (isCoverTask && job.params.sourceAudioUrl) {
      srcAudioBuf = loadSourceAudio(job.params.sourceAudioUrl, job.id, log);
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

    // ── Timbre reference ──
    let refAudioBuf: Buffer | undefined;
    const masteringRef = job.params.masteringReference;
    refAudioBuf = await loadTimbreReference(job.params, masteringRef, aceReq.seed, job.id, log);

    // When mastering is enabled, request wav32 (float) from the engine.
    const synthFormat = (job.params.masteringEnabled && job.params.masteringReference) ? 'wav32' : 'wav16';
    if (synthFormat === 'wav32') {
      logGeneration(job.id, 'INFO', '[Synth Phase] Using wav32 (raw float) for mastering input — normalization deferred to mastering');
    }

    // LRC: auto-enable synchronized lyric timestamps for non-instrumental tracks
    const hasLyrics = lmResults.some(r => r.lyrics && r.lyrics !== '[Instrumental]');
    if (hasLyrics) {
      for (const r of lmResults) {
        if (r.lyrics && r.lyrics !== '[Instrumental]') {
          (r as any).get_lrc = true;
        }
      }
      logGeneration(job.id, 'INFO', '[Synth Phase] LRC generation enabled (non-instrumental lyrics detected)');
    }

    if (coResident) {
      logGeneration(job.id, 'INFO', '[Synth Phase] Co-resident mode: DiT+VAE will stay in VRAM');
    }

    // Save audio files and create DB entries
    const audioUrls: string[] = [];
    const songIds: string[] = [];
    // Per-track mastered URLs (parallel array to audioUrls)
    const masteredUrls: string[] = [];
    // Per-track latent URLs (parallel array to audioUrls)
    const latentUrls: string[] = [];

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

      job.stage = `Synthesizing${trackLabel}...`;
      job.progress = Math.round(trackProgressBase);

      // Subscribe to engine logs for this track's DiT progress
      const unsubSynth = subscribeLines((line) => {
        if (line.source !== 'engine') return;
        const dit = line.text.match(/\[DiT\] Step (\d+)\/(\d+)\s+t=[\d.]+\s+\[(.+?)\]/);
        if (dit) {
          const step = parseInt(dit[1], 10);
          const total = parseInt(dit[2], 10);
          job.stage = `DiT${trackLabel}: Step ${step}/${total} (${dit[3]})`;
          job.progress = Math.round(trackProgressBase + (step / total) * progressPerTrack * 0.8);
          return;
        }
        const ditSimple = line.text.match(/\[DiT\] Step (\d+)\/(\d+)/);
        if (ditSimple) {
          const step = parseInt(ditSimple[1], 10);
          const total = parseInt(ditSimple[2], 10);
          job.stage = `DiT${trackLabel}: Step ${step}/${total}`;
          job.progress = Math.round(trackProgressBase + (step / total) * progressPerTrack * 0.8);
          return;
        }
        if (line.text.includes('[VAE]') || line.text.includes('vae_decode')) {
          job.stage = `Decoding audio (VAE)${trackLabel}...`;
          job.progress = Math.round(trackProgressBase + progressPerTrack * 0.9);
        } else if (line.text.includes('[Adapter]') && line.text.includes('Merge')) {
          job.stage = `Loading adapter${trackLabel}...`;
        } else if (line.text.includes('Loading synth') || line.text.includes('ensure_synth')) {
          job.stage = `Loading DiT model${trackLabel}...`;
        } else if (line.text.includes('[FSQ]') || line.text.includes('fsq_detokenize')) {
          job.stage = `Decoding audio tokens (FSQ)${trackLabel}...`;
        } else if (line.text.includes('[DiT]') && line.text.includes('batch')) {
          job.stage = `Preparing DiT${trackLabel}...`;
        }
      });

      // Submit single request to /synth
      let synthJobId: string;
      if (srcAudioBuf || refAudioBuf || srcLatentBuf) {
        const parts = [srcAudioBuf ? 'src_audio' : '', refAudioBuf ? 'timbre_ref' : '', srcLatentBuf ? 'src_latents' : ''].filter(Boolean).join('+');
        logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: MULTIPART submission (${parts})`);
        synthJobId = await aceClient.submitSynthMultipart(synthReq, srcAudioBuf, refAudioBuf, srcLatentBuf, synthFormat, coResident);
      } else {
        logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: plain JSON submission`);
        synthJobId = await aceClient.submitSynth(synthReq, synthFormat, coResident);
      }
      job.aceJobId = synthJobId;

      await pollUntilDone(synthJobId, job, abortController.signal);
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

      logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: saved ${filename} (${(audioBuffer.length / 1024).toFixed(0)} KB)`);

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

    // ── Post-processing chain ─────────────────────────────────
    // Raw WAV (audio_url) is NEVER modified. Post-processing runs on a copy.
    job.progress = 89;
    job.stage = 'Post-processing...';

    try {
      const ppResults = await runPostProcessingChain(
        audioUrls, job.params, totalTracks, job.id,
        log, (stage) => { job.stage = stage; }
      );
      masteredUrls.push(...ppResults);
    } catch (err: any) {
      logGeneration(job.id, 'WARNING', `[Post-Processing] Chain failed: ${err.message}`);
    }

    // Create song entries in DB — one per track
    for (let i = 0; i < audioUrls.length; i++) {
      const audioUrl = audioUrls[i];
      const trackMastered = masteredUrls[i] || '';
      const trackLatent = latentUrls[i] || '';
      // Use per-track LM result for metadata when available
      const trackResult = lmResults[i] || firstResult;
      const trackLyrics = trackResult.lyrics || job.params.lyrics || '';
      const trackCaption = trackResult.caption || '';

      const songId = uuidv4();
      getDb().prepare(`
        INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                           duration, bpm, key_scale, time_signature, tags, dit_model,
                           generation_params, mastered_audio_url, latent_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        songId, job.userId, title, trackLyrics, style, trackCaption,
        audioUrl, duration, bpm, keyScale, timeSignature,
        JSON.stringify([]), aceReq.synth_model || '', JSON.stringify(job.params),
        trackMastered, trackLatent,
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
      masteredAudioUrl: masteredUrls.find(u => !!u) || undefined,
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

export default router;
