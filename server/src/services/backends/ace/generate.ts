import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';
import { aceClient, type AceRequest } from '../../aceClient.js';
import { getDb } from '../../../db/database.js';
import { config } from '../../../config.js';
import { precomputePeaks } from '../../audio/peaks.js';
import { startGenerationLog, logGeneration, logGenerationParams, finishGenerationLog, failGenerationLog } from '../../logger.js';
import { autoTrimSilence } from '../../autoTrim.js';
import { wavDurationSec } from '../../audioCrop.js';
import { writeHslat, latentFrameCount, latentDuration, type HslatMetadata } from '../../latentFormat.js';
import { subscribeLines } from '../../../routes/logs.js';
import { translateParams } from '../../generation/translateParams.js';
import { applyTriggers, resolveAdapterTriggers, resolveTriggerSpecs } from '../../generation/triggerWords.js';
import { readAdapterTrigger } from '../../adapters/stMetadata.js';
import { computeLmCacheKey, getLmCache, setLmCache, getLmCacheSize, type LmCacheEntry } from '../../generation/lmCache.js';
import { degeneratePlanReason } from '../../generation/planGuard.js';
import { loadSourceAudio, loadSourceLatent, applyTempoAndPitch, loadTimbreReference } from '../../generation/sourceAudio.js';
import { runPostProcessingChain, normalizePpParams } from '../../generation/postProcessing.js';
import { getCachedLatent, saveCachedLatent } from '../../generation/sourceLatentCache.js';
import { releaseMinimaxVramForAce } from '../minimax/generate.js';
import { pollUntilDone as importedPoller } from '../../generation/pollUntilDone.js';
import { type GenerationJob, type StageTiming } from '../../generation/jobTypes.js';

export async function runAceGeneration(
  job: GenerationJob,
  deps: { pollUntilDone: typeof importedPoller; signal: AbortSignal },
): Promise<void> {
  const { pollUntilDone } = deps;
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


  // Defensive residency arbitration (plan §4.4): an MM3 run may have left ~13 GB
  // parked. Idempotent and cheap when cold; short-fused so a hung engine can
  // never stall an ACE generation.
  await releaseMinimaxVramForAce();

  const aceReq = translateParams(job.params);

  // Write the resolved seeds back into job.params so the DB stores the actual
  // values used — critical for reproducibility when randomSeed is true, or
  // when lmSeedFollowsDit is on (lm_seed isn't sent to the engine at all in
  // that case — the engine ties it to seed itself — so mirror that here
  // rather than leaving a stale/unused "fixed" value in job.params).
  if (aceReq.seed !== undefined) {
    job.params.seed = aceReq.seed;
  }
  if (aceReq.lm_seed !== undefined) {
    job.params.lmSeed = aceReq.lm_seed;
  } else if (job.params.lmSeedFollowsDit !== false) {
    job.params.lmSeed = aceReq.seed;
  }

  console.log(`[Generate] Job ${job.id} — ditModel=${job.params.ditModel || '(none)'}, synth_model=${aceReq.synth_model || '(none)'}, emb_model=${aceReq.emb_model || '(auto)'}, seed=${aceReq.seed ?? '(engine default)'}, lm_seed=${aceReq.lm_seed ?? `(tied to seed: ${aceReq.seed ?? 'engine default'})`}, source=${job.params.source || 'create'}`);
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
          // Resolved per-output LM seed (base + batch index) from the run
          // that produced these codes — see the cache-miss rebuild below.
          lm_seed: lmOut.lm_seed,
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
        try {

        logGeneration(job.id, 'INFO', `[LM Phase] Submitting to ace-server... (cache key=${cacheKey})`);

        const coResident = job.params.coResident === true;
        const lmJobId = await aceClient.submitLm(aceReq, undefined, coResident);
        job.aceJobId = lmJobId;

        await pollUntilDone(lmJobId, job, deps.signal, timeoutMinutes);

        // Fetch LM results (array of enriched AceRequests)
        const resultRes = await aceClient.getJobResult(lmJobId);
        lmResults = await resultRes.json() as AceRequest[];

        // A COMPLETED LM job with zero parseable results is a failure, not an
        // empty batch. Before this check (2026-08-11), an engine-side
        // serialization failure (invalid UTF-8 in LM output made
        // request_to_json ship "[]" while logging success) slid into the
        // "LM skipped" batch path, which then submitted a synth request with
        // an EMPTY caption and died on a confusing 400 two phases later.
        if (!Array.isArray(lmResults) || lmResults.length === 0) {
          throw new Error(
            'LM job completed but returned no parseable results — see ace_engine.log '
            + '(a request_to_json serialization failure logs there; the engine now sanitizes '
            + 'invalid UTF-8 in LM output, so hitting this again means something new)');
        }

        // The LM echo is the C++ AceRequest with LM-generated fields filled
        // in — ServerFields-only sideband (adapter_runtime_quant, alignment
        // timing, rebase, plugin_params, …) is not part of the C++ struct and
        // never survives the round trip. Rebuild each result from the CURRENT
        // request and take only the LM-generated fields from the echo (the
        // same reconstruction the cache-hit path uses above), so every
        // sideband field reaches /synth regardless of cache state.
        lmResults = lmResults.map(lmOut => ({
          ...aceReq,
          audio_codes: lmOut.audio_codes,
          caption: lmOut.caption,
          lyrics: lmOut.lyrics,
          bpm: lmOut.bpm,
          duration: lmOut.duration,
          keyscale: lmOut.keyscale,
          timesignature: lmOut.timesignature,
          // The engine gives each LM batch output its own seed (lm_seed + b)
          // and echoes the resolved value — keep it so per-track
          // generation_params record the seed that actually produced this
          // track's codes, not the request-level base.
          lm_seed: lmOut.lm_seed,
        }));
        job.lmResults = lmResults;

        // ── Plan guard (Rob, 2026-08-29) — adapter-led runs only ──────────
        // Deep LM adapters are loop-fragile per RENDER, not per adapter: the
        // same adapter emits clean plans on most seeds and a stuck loop on
        // others. The degeneracy is fully visible in the raw code statistics,
        // so catch it here — before the DiT burns minutes rendering a click
        // track — and resample with a shifted seed + a firmer repetition
        // penalty. Runs before the cache write so a degenerate plan is never
        // cached. Base-planner runs skip the guard: base has never looped, and
        // legitimately repetitive genres shouldn't fight a watchdog.
        if (aceReq.lm_adapter) {
          const requestedSec = Number(aceReq.duration) || 0;
          for (let attempt = 1; attempt <= 2; attempt++) {
            const bad = lmResults
              .map(r => degeneratePlanReason(r.audio_codes || '', requestedSec))
              .map((reason, idx) => ({ reason, idx }))
              .filter(x => x.reason);
            if (!bad.length) break;
            const penalty = Math.max(Number(aceReq.lm_rep_penalty) || 1.0, 1.0) + 0.05 * attempt;
            const seed = (Number(aceReq.lm_seed) || 0) + 1009 * attempt;
            for (const b of bad) {
              logGeneration(job.id, 'WARNING',
                `[Plan Guard] Degenerate plan (result ${b.idx}): ${b.reason} — retry ${attempt}/2 `
                + `(seed ${seed}, rep penalty ${penalty.toFixed(2)})`);
            }
            job.stage = `Plan looked stuck — resampling (retry ${attempt}/2)...`;
            const retryReq = { ...aceReq, lm_seed: seed, lm_rep_penalty: penalty };
            const retryJobId = await aceClient.submitLm(retryReq, undefined, coResident);
            job.aceJobId = retryJobId;
            await pollUntilDone(retryJobId, job, deps.signal, timeoutMinutes);
            const retryRes = await aceClient.getJobResult(retryJobId);
            const retryOut = await retryRes.json() as AceRequest[];
            if (!Array.isArray(retryOut) || retryOut.length === 0) break;   // keep what we have
            // Replace only the degenerate slots; keep healthy plans untouched.
            for (const b of bad) {
              const lmOut = retryOut[Math.min(b.idx, retryOut.length - 1)];
              lmResults[b.idx] = {
                ...aceReq,
                audio_codes: lmOut.audio_codes,
                caption: lmOut.caption,
                lyrics: lmOut.lyrics,
                bpm: lmOut.bpm,
                duration: lmOut.duration,
                keyscale: lmOut.keyscale,
                timesignature: lmOut.timesignature,
                lm_seed: lmOut.lm_seed,
              };
            }
            job.lmResults = lmResults;
          }
          const still = lmResults
            .map(r => degeneratePlanReason(r.audio_codes || '', requestedSec))
            .filter(Boolean);
          if (still.length) {
            // Never hard-fail a generation over the guard: render the best we
            // have and say so loudly — the user judges by ear anyway.
            logGeneration(job.id, 'WARNING',
              `[Plan Guard] Plan still degenerate after retries (${still[0]}) — rendering it anyway`);
          }
        }

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
            lm_seed: r.lm_seed,
          }));
          setLmCache(cacheKey, lmOutputs);
          logGeneration(job.id, 'INFO', `[LM Phase] Cached LM outputs (key=${cacheKey}, cache size=${getLmCacheSize()})`);
        }

        job.progress = 40;
        job.stage = 'LM complete, preparing synthesis...';
        logGeneration(job.id, 'INFO', `[LM Phase] Complete. ${lmResults.length} result(s), bpm=${lmResults[0]?.bpm}, duration=${lmResults[0]?.duration}`);

        // Unsubscribe LM progress watcher
        } finally {
          unsubLm();
        }
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
      // Resolve exactly the way translateParams did, so the two passes cannot
      // disagree about which word goes where.
      const adapterPaths: string[] = [
        ...(Array.isArray(job.params.loraStack) ? job.params.loraStack.map((e: { path: string }) => e.path) : []),
        ...(job.params.loraPath ? [job.params.loraPath] : []),
        ...(job.params.lmAdapter ? [job.params.lmAdapter] : []),
      ].filter((p, i, a) => p && a.indexOf(p) === i);
      const triggerSpecs = resolveAdapterTriggers(adapterPaths, resolveTriggerSpecs(job.params), readAdapterTrigger);
      if (triggerSpecs.length && adapterPaths.length) {
        for (const result of lmResults) {
          const before = (result.caption || '').substring(0, 80);
          // skipPresent: the planner often keeps the trigger it was trained on,
          // and re-adding it would duplicate the token.
          const r = applyTriggers(result.caption || '', triggerSpecs, { skipPresent: true });
          result.caption = r.caption;
          if (r.applied.length) {
            logGeneration(job.id, 'INFO', `[Trigger] Re-injected "${r.applied.join(', ')}" — before: "${before}…" → after: "${r.caption.substring(0, 80)}…"`);
          } else {
            logGeneration(job.id, 'INFO', `[Trigger] Already present: "${r.skipped.join(', ')}" found in caption, skipping re-injection`);
          }
        }
      } else if (triggerSpecs.length) {
        // Triggers configured but no adapter loaded — log why nothing happened.
        logGeneration(job.id, 'WARNING', `[Trigger] "${triggerSpecs.map(s => s.word).join(', ')}" configured but no adapter is loaded — skipping re-injection`);
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

    // ── Structural seed latent (Song Builder repeated sections) ──
    // Slice the tail (seedSeconds) of the seed section's cumulative latent —
    // that tail IS the original section's own content. The engine biases the
    // repaint region's init noise toward it (seed_strength) so the new section
    // follows the original's harmonic shape.
    let seedLatentBuf: Buffer | undefined;
    if (job.params.seedLatentUrl && Number(job.params.seedStrength) > 0) {
      const full = loadSourceLatent(job.params.seedLatentUrl as string, log);
      if (full) {
        const secs = Number(job.params.seedSeconds) || 0;
        const frames = Math.max(1, Math.round(secs * 25));
        const bytes = frames * 256;
        seedLatentBuf = (secs > 0 && bytes < full.length) ? full.subarray(full.length - bytes) : full;
        log('INFO', `[Seed] Seed latent: ${Math.round(seedLatentBuf.length / 256)} frames (strength=${job.params.seedStrength})`);
      }
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
      !!(job.params.stableStepOn ?? job.params.stableStep) ||
      !!job.params.spectralLifterEnabled ||
      !!job.params.masteringEnabled
    );
    // Song Builder: force wav32 so the engine SKIPS peak-normalization. wav16
    // peak-normalizes the whole canvas to ~0 dBFS each pass, so appending a
    // louder section scales the entire (bit-exact preserved) canvas down — older
    // sections get progressively quieter. wav32 keeps absolute levels stable.
    const isBuilder = job.params.source === 'builder';
    const synthFormat = (anyPpActive || isBuilder || (job.params.masteringEnabled && job.params.masteringReference)) ? 'wav32' : 'wav16';
    if (synthFormat === 'wav32') {
      logGeneration(job.id, 'INFO', `[Synth Phase] Using wav32 (raw float) — normalization deferred${isBuilder ? ' (builder: stable levels across sections)' : ''}`);
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

    // ── Whisper Lyrics Transcription (shared by full-mix and stem modes) ──
    // CPU-only (whisper-cli), no VRAM impact — safe to overlap GPU work.
    const runWhisperForTrack = async (trackNum: number, wavPath: string, trackLyrics: string, wavFilename: string) => {
      const whisperTimingStart = performance.now();
      try {
        const { ensureWhisperCli, findWhisperModel, transcribeWithWhisper } = await import('../../whisperTranscribe.js');
        const { reconcileLyrics } = await import('../../lyricsReconcile.js');

        const whisperReady = await ensureWhisperCli();
        if (!whisperReady) {
          logGeneration(job.id, 'WARNING', '[Whisper] whisper-cli unavailable — skipping');
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

    // Stem-mode Whisper: when StableStep is on (its split is free to reuse) or
    // the "Isolate vocals first" toggle is set, transcription is deferred to the
    // post-processing chain, which hands us the isolated vocal stem via the
    // onVocalStem callback. Keyed by trackIdx; entries still present after PP
    // fall back to full-mix transcription.
    const whisperStemMode = !!job.params.whisperLyricsEnabled
      && !job.params.instrumental
      && !!(job.params.whisperIsolateVocals || job.params.stableStepOn || job.params.stableStep);
    const pendingStemWhisper = new Map<number, { wavPath: string; lyrics: string; filename: string }>();

    // ── Parallel Cover Art: launch right after LM (earliest possible) ──
    // At this point we have title/style/lyrics/subject from LM results.
    // Image generation (GPU) starts now and overlaps with the entire synth phase.
    // The cheap DB link happens after DB insert provides songIds.
    let coverArtResults: Array<{ coverUrl: string }> = [];
    if (job.params.parallelCoverArt && job.params.coverArtEnabled) {
      const coverArtTask = async () => {
        const coverArtStart = performance.now();
        try {
          const { generateCoverImage, getCoverArtReadiness } = await import('../../coverArt/coverArtService.js');
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

      // Auto-set stream_chunk_dir inside data/audio/stream/ so previews are
      // served by the static /audio middleware. Create the directory if needed.
      if (synthReq.stream_mode && !synthReq.stream_chunk_dir) {
        const streamDir = path.join(config.data.audioDir, 'stream');
        fs.mkdirSync(streamDir, { recursive: true });
        synthReq.stream_chunk_dir = streamDir;
      }
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
      // Tile counters for the decode stage label. The stall watchdog only sees
      // progress when job.stage or job.progress CHANGES, and every VAE line
      // used to set the identical string, so a decode that legitimately took
      // longer than the 120 s stale window was cancelled as wedged (#96).
      let vaeTileTotal = 0;
      let vaeTileDone = 0;
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
          if (!vaeStartAt) { vaeStartAt = now; vaeTileDone = 0; vaeTileTotal = 0; }

          // "[VAE] Tiled decode: 5 tiles (chunk=1024, ...)" announces the plan.
          const tilePlan = line.text.match(/Tiled decode:\s*(\d+)\s+tiles/);
          if (tilePlan) vaeTileTotal = parseInt(tilePlan[1], 10);
          // "[VAE] Graph: 539 nodes, T_latent=960" fires once per tile.
          if (line.text.includes('[VAE] Graph:')) vaeTileDone++;

          // The label MUST differ between tiles. Setting the same string every
          // time is what made a live decode look wedged to the watchdog (#96):
          // an 80-step cover on Apple Silicon spends over two minutes in here
          // and was cancelled at 120 s, after which the engine finished the
          // track anyway and the user could not reach it.
          job.stage = vaeTileTotal > 0
            ? `Decoding audio (VAE)${trackLabel}: tile ${Math.min(vaeTileDone, vaeTileTotal)}/${vaeTileTotal}...`
            : `Decoding audio (VAE)${trackLabel}...`;
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

      let synthTrackStart = 0;
      let synthJobId: string;
      try {
      // Submit single request to /synth
      synthTrackStart = performance.now();
      if (srcAudioBuf || refAudioBuf || srcLatentBuf || refLatentBuf || seedLatentBuf) {
        const parts = [
          srcAudioBuf ? 'src_audio' : '',
          refAudioBuf ? 'timbre_ref' : '',
          srcLatentBuf ? 'src_latents' : '',
          refLatentBuf ? 'ref_latents' : '',
          seedLatentBuf ? 'seed_latents' : '',
        ].filter(Boolean).join('+');
        logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: MULTIPART submission (${parts})`);
        synthJobId = await aceClient.submitSynthMultipart(synthReq, srcAudioBuf, refAudioBuf, srcLatentBuf, refLatentBuf, synthFormat, coResident, seedLatentBuf);
      } else {
        logGeneration(job.id, 'INFO', `[Synth Phase] Track ${trackIdx + 1}: plain JSON submission`);
        synthJobId = await aceClient.submitSynth(synthReq, synthFormat, coResident);
      }
      job.aceJobId = synthJobId;

      await pollUntilDone(synthJobId, job, deps.signal, timeoutMinutes);
      } finally {
        unsubSynth();
      }

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
      // Full-mix mode runs here (optionally in parallel with post-processing).
      // Stem mode defers to the PP chain's shared SuperSep split — see the
      // onVocalStem callback at the runPostProcessingChain call site.
      if (job.params.whisperLyricsEnabled) {
        if (whisperStemMode) {
          pendingStemWhisper.set(trackIdx, { wavPath: filepath, lyrics: synthReq.lyrics || '', filename });
          logGeneration(job.id, 'INFO', `[Whisper] Track ${trackIdx + 1}: deferred — will transcribe isolated vocal stem`);
        } else {
          const whisperPromise = runWhisperForTrack(trackIdx + 1, filepath, synthReq.lyrics || '', filename);
          if (job.params.parallelWhisper) {
            // Deferred — will be awaited after post-processing
            deferredTasks.push(whisperPromise);
            logGeneration(job.id, 'INFO', `[Whisper] Track ${trackIdx + 1}: launched in parallel`);
          } else {
            await whisperPromise;
          }
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

    // ── No-adapter reference render (optional 3rd output) ─────
    // Re-synth each track at a low step count on the BARE DiT: identical LM
    // output (caption/lyrics/audio codes — an LM adapter's influence is kept)
    // and identical seed, but every DiT adapter field stripped. Runs AFTER the
    // main loop so the bare DiT is loaded once for all tracks. The output is
    // deliberately raw: no auto-trim, no post-processing — it exists purely so
    // the user can hear what the song sounds like without the DiT adapter.
    // Per-track parallel array like masteredUrls; '' = no reference render.
    const noAdapterUrls: string[] = [];
    const NO_ADAPTER_STEPS = 20;
    const anyDitAdapter = lmResults.some(r => (r.adapters && r.adapters.length > 0) || r.adapter);
    if (job.params.noAdapterRender && anyDitAdapter) {
      for (let trackIdx = 0; trackIdx < audioUrls.length; trackIdx++) {
        const baseReq = lmResults[trackIdx];
        const trackLabel = audioUrls.length > 1 ? ` (track ${trackIdx + 1}/${audioUrls.length})` : '';
        let refUrl = '';
        const refStart = performance.now();
        // Feed the stall watchdog during the bare-DiT reload + denoise steps
        // (the main loop's line subscriber is gone by now).
        const unsubRef = subscribeLines((line) => {
          if (line.source !== 'engine') return;
          const step = line.text.match(/\[DiT(?:-TRT)?\] Step (\d+)\/(\d+)/);
          if (step) {
            job.stage = `No-adapter reference${trackLabel}: Step ${step[1]}/${step[2]}`;
            return;
          }
          const load = line.text.match(/\[Store\] Load (.+?):\s+\d+\s*ms/);
          if (load) job.stage = `No-adapter reference${trackLabel}: loading ${load[1]}...`;
        });
        try {
          const refReq = { ...baseReq };
          delete refReq.adapter;
          delete refReq.adapter_scale;
          delete refReq.adapters;
          delete refReq.adapter_sections;
          delete refReq.adapter_section_align_at;
          delete refReq.adapter_section_isolation;
          delete refReq.adapter_group_scales;
          delete refReq.adapter_mode;
          delete refReq.adapter_runtime_quant;
          delete refReq.adapter_merge_lowvram;
          delete refReq.rebase_source;
          delete refReq.rebase_beta;
          delete (refReq as any).get_lrc;
          refReq.inference_steps = NO_ADAPTER_STEPS;

          job.stage = `No-adapter reference${trackLabel}...`;
          let refJobId: string;
          if (srcAudioBuf || refAudioBuf || srcLatentBuf || refLatentBuf || seedLatentBuf) {
            refJobId = await aceClient.submitSynthMultipart(refReq, srcAudioBuf, refAudioBuf, srcLatentBuf, refLatentBuf, 'wav16', coResident, seedLatentBuf);
          } else {
            refJobId = await aceClient.submitSynth(refReq, 'wav16', coResident);
          }
          job.aceJobId = refJobId;
          await pollUntilDone(refJobId, job, deps.signal, timeoutMinutes);

          const refRes = await aceClient.getJobResult(refJobId);
          const refBuffer = Buffer.from(await refRes.arrayBuffer());
          const refContentType = refRes.headers.get('content-type') || 'audio/mpeg';
          const refExt = refContentType.includes('wav') ? 'wav' : 'mp3';
          const rawBase = path.basename(audioUrls[trackIdx]).replace(/\.[^.]+$/, '');
          const refFilename = `${rawBase}_noadapter.${refExt}`;
          fs.writeFileSync(path.join(config.data.audioDir, refFilename), refBuffer);
          refUrl = `/audio/${refFilename}`;
          logGeneration(job.id, 'INFO',
            `[No-Adapter Ref] Track ${trackIdx + 1}: saved ${refFilename} (${NO_ADAPTER_STEPS} steps, ${(refBuffer.length / 1024).toFixed(0)} KB, ${((performance.now() - refStart) / 1000).toFixed(1)}s)`);
        } catch (refErr: any) {
          if (refErr.message === 'Cancelled') throw refErr;
          logGeneration(job.id, 'WARNING', `[No-Adapter Ref] Track ${trackIdx + 1}: failed (non-fatal): ${refErr.message}`);
        } finally {
          unsubRef();
        }
        noAdapterUrls.push(refUrl);
        const refMs = Math.round(performance.now() - refStart);
        if (refMs > 50) timing.push({ name: `No-Adapter Ref${audioUrls.length > 1 ? ` Track ${trackIdx + 1}` : ''}`, ms: refMs });
      }
    }

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

    // Normalized by the same helper the after-the-fact re-run uses, so both
    // entry points into the chain agree on defaults and flag aliases.
    const ppParams = normalizePpParams(
      job.params,
      audioUrls.map((_, ti) =>
        (lmResults[ti]?.caption || firstResult.caption || job.params.caption || '') as string),
    );

    // Stem-mode Whisper: fired by the PP chain as soon as a track's vocal stem
    // exists (before the SA3 refine) so CPU transcription overlaps GPU work.
    // stemPath null = no stem produced → transcribe the full mix instead.
    const onVocalStem = pendingStemWhisper.size > 0
      ? (trackIdx: number, stemPath: string | null) => {
          const pend = pendingStemWhisper.get(trackIdx);
          if (!pend) return;
          pendingStemWhisper.delete(trackIdx);
          if (stemPath) {
            logGeneration(job.id, 'INFO', `[Whisper] Track ${trackIdx + 1}: transcribing isolated vocal stem`);
          } else {
            logGeneration(job.id, 'WARNING', `[Whisper] Track ${trackIdx + 1}: no vocal stem available — falling back to full mix`);
          }
          const whisperPromise = runWhisperForTrack(
            trackIdx + 1, stemPath ?? pend.wavPath, pend.lyrics, pend.filename
          ).finally(() => {
            if (stemPath) { try { fs.unlinkSync(stemPath); } catch {} }
          });
          deferredTasks.push(whisperPromise);
        }
      : undefined;

    let ppQualityScores: Array<{ unmastered?: any; mastered?: any }> = [];
    try {
      const ppResult = await runPostProcessingChain(
        audioUrls, ppParams, totalTracks, job.id,
        log, (stage) => { job.stage = stage; },
        onVocalStem
      );
      masteredUrls.push(...ppResult.masteredUrls);
      if (ppResult.timing && ppResult.timing.length > 0) {
        timing.push(...ppResult.timing);
      }
      ppQualityScores = ppResult.qualityScores;
    } catch (err: any) {
      logGeneration(job.id, 'WARNING', `[Post-Processing] Chain failed: ${err.message}`);
    }

    // Stem-mode Whisper fallback: any deferred track whose callback never fired
    // (PP chain threw, non-WAV track, etc.) still gets a full-mix transcription.
    if (pendingStemWhisper.size > 0) {
      for (const [ti, pend] of pendingStemWhisper) {
        logGeneration(job.id, 'WARNING', `[Whisper] Track ${ti + 1}: stem split never ran — falling back to full mix`);
        deferredTasks.push(runWhisperForTrack(ti + 1, pend.wavPath, pend.lyrics, pend.filename));
      }
      pendingStemWhisper.clear();
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

      // Duration: the LM provides one for text2music, but repaint/cover skip the
      // LM (firstResult.duration = 0). Backfill from the actual output WAV so the
      // song has a real length — downstream features (e.g. Song Builder clip
      // points) rely on it. Falls back to 0 for non-WAV / read failures.
      let trackDuration = duration;
      if (!(trackDuration > 0)) {
        const wavPath = path.join(config.data.audioDir, path.basename(audioUrl));
        const measured = wavDurationSec(wavPath);
        if (measured > 0) trackDuration = Math.round(measured);
        // The backfill only ever reached the song row; the job result and the
        // "[Result] Duration" line kept the LM's 0 for cover and repaint (#124).
        if (!(duration > 0) && trackDuration > 0) duration = trackDuration;
      }

      // Post-processing is done and this file will not change again, so build
      // its waveform now. The player reads peaks instead of downloading and
      // decoding the audio, and doing it here means the first listen never
      // waits for it.
      precomputePeaks(path.join(config.data.audioDir, path.basename(audioUrl)));

      // Per-track generation_params. job.params.seed/lmSeed only reflect the
      // job-level (first-track) value — when randomSeed varies the DiT seed
      // per track (or the engine varies lm_seed per output in a live LM
      // batch), later tracks would otherwise all show track 1's seed in the
      // DB/UI even though a different one was actually used to synthesize
      // them, making reproduction impossible. trackResult (== lmResults[i],
      // mutated in place by the synth loop) holds the real per-track values.
      const trackParams = {
        ...job.params,
        seed: trackResult.seed !== undefined ? trackResult.seed : job.params.seed,
        lmSeed: trackResult.lm_seed !== undefined ? trackResult.lm_seed : job.params.lmSeed,
      };

      const songId = uuidv4();
      getDb().prepare(`
        INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                           duration, bpm, key_scale, time_signature, tags, dit_model,
                           generation_params, mastered_audio_url, latent_url, quality_scores,
                           noadapter_audio_url, backend)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        songId, job.userId, title, trackLyrics, style, trackCaption,
        audioUrl, trackDuration, bpm, keyScale, timeSignature,
        JSON.stringify([]), aceReq.synth_model || '', JSON.stringify(trackParams),
        trackMastered, trackLatent, qualityJson,
        noAdapterUrls[i] || '', 'ace',
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
      const { linkCoverToSong } = await import('../../coverArt/coverArtService.js');
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
        const { generateCoverArt, getCoverArtReadiness } = await import('../../coverArt/coverArtService.js');
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
      noAdapterAudioUrl: noAdapterUrls.find(u => !!u) || undefined,
      // Index-aligned with audioUrls, so a caller holding track i can find
      // track i's master rather than track 0's.
      masteredAudioUrls: audioUrls.map((_, i) => masteredUrls[i] || ''),
      noAdapterAudioUrls: audioUrls.map((_, i) => noAdapterUrls[i] || ''),
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
