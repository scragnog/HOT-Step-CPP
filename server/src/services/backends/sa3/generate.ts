// backends/sa3/generate.ts — Stable Audio 3 primary-backend generation path
//
// HOT-Step (MDMAchine / A&E Concepts). Runs text-to-audio through /sa3-synth
// or full-source Cover Studio transforms through /sa3-refine, and saves in
// the library exactly the way backends/minimax/generate.ts does: WAV to
// data.audioDir, model-agnostic post-processing, optional Whisper, a songs row,
// job.result. Anything that pattern-matches MM3's file is deliberate — the
// library, the queue and the UI expect one completion contract.
//
// Deliberately SIMPLER than MM3's runner. /sa3-synth is synchronous (it
// returns the WAV in the response body) so there is no job polling, no
// progress stream, no takes, no LRC, no planner LM, no AR cache. The whole
// render is one awaited fetch. Covers preserve the source rate and length.
//
// Prompt: the caption is tokenised with the same T5Gemma tokenizer StableStep
// uses (sa3Tokenizer.ts). NOT buildStableStepPrompt() — that strips vocal
// descriptors and appends "Instrumental only", which is right for a refine
// of an ACE instrumental bed and wrong for a primary render of whatever the
// user asked for. The caption goes through as written; covers append length.

/** Stage shown from submit until the render returns. /sa3-synth is one
 *  synchronous call with no progress stream and no load phase, so model load
 *  and render are indistinguishable from here. Both stages start with
 *  "Loading" so the DAW facade (/^loading/i) reports phase loading_model
 *  and eta_s null instead of a made-up eta; "saving audio" replaces it. */
export const SA3_LOADING_STAGE = 'Loading SA3 model...';
export function sa3RenderStage(durationSec: number, steps: number): string {
  return `SA3: rendering ${Math.round(durationSec)}s (${steps} steps)...`;
}

import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';
import { parseBuffer } from 'music-metadata';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

import { config, getFFmpegPath } from '../../../config.js';
import { getDb } from '../../../db/database.js';
import { aceClient } from '../../aceClient.js';
import { wavDurationSec } from '../../audioCrop.js';
import { tokenizeForSa3 } from '../../sa3Tokenizer.js';
import { runPostProcessingChain } from '../../generation/postProcessing.js';
import type { PostProcessParams } from '../../generation/postProcessing.js';
import { normalizeLufs } from '../../generation/lufsNormalize.js';
import { resolveSourceAudioPath } from '../../generation/sourceAudio.js';
import {
  startGenerationLog, logGeneration, logGenerationParams,
  finishGenerationLog, failGenerationLog,
} from '../../logger.js';
import { releaseMinimaxVramForAce } from '../minimax/generate.js';
import type { GenerationJob, StageTiming } from '../../generation/jobTypes.js';
import { SA3_MAX_DURATION_SEC } from './index.js';
import { SA3_DEFAULT_DIT } from './models.js';

const TASK = 'sa3-text2music';

/** /sa3-synth is one synchronous fetch — no engine job id to poll, so this
 *  only needs the shared lane's abort signal (same deps shape as
 *  minimax/generate.ts and yue2/generate.ts, minus pollUntilDone). */
interface Sa3GenerationDeps {
  signal: AbortSignal;
}

/** The wire request, as sent. Stored in generation_params so a reproduce
 *  re-submits the real inputs. */
interface Sa3SynthWire {
  synthModel: string;
  resolvedSynthModel?: string;
  caption: string;
  prompt: string;         // what was actually tokenised (== caption today)
  nTokens: number;
  durationSec: number;
  steps: number;
  sampler: 'pingpong' | 'euler';
  seed: number;
  adapters: Array<{ name: string; scale: number }>;
  solver?: string;
  scheduler?: string;
  guidanceMode?: string;
  guidanceScale?: number;
  pluginParams?: Record<string, string | number>;
  postprocessPlugin?: string;
  sourceAudioUrl?: string;
  strength?: number;
  sampleRate?: number;
}

/** Default SA3 step count (ear-tested 2026-10-02: 8 beats 12 and 24). */
export const SA3_DEFAULT_STEPS = 8;

/** Lua solver used when sa3Sampler is 'md_pingpong_blend' (the default). */
export const SA3_DEFAULT_SOLVER = 'md_pingpong_blend';

/**
 * Resolve the SA3 sampler pick. 'pingpong' and 'euler' are the native loops;
 * 'md_pingpong_blend' (default when unset) runs the Lua solver of that name
 * on top of the native pingpong wire value. A solver chosen through the
 * Sampler Plugins pickers always wins over this default.
 */
export function resolveSa3Sampler(params: any, plugins: { solver?: string }):
    { sampler: 'pingpong' | 'euler'; solver?: string } {
  const pick = params.sa3Sampler ?? SA3_DEFAULT_SOLVER;
  if (pick === 'euler') return { sampler: 'euler' };
  if (pick === SA3_DEFAULT_SOLVER && !plugins.solver) {
    return { sampler: 'pingpong', solver: SA3_DEFAULT_SOLVER };
  }
  return { sampler: 'pingpong' };
}

/** Seed-breeding blend used when seed B is set but no blend is given (ear test b33/b34). */
export const SA3_DEFAULT_SEED_BLEND = 0.5;

/** Blend for SA3 seed breeding. Unset or blank -> 0.5; explicit values clamp to [0, 1]. */
export function resolveSa3SeedBlend(v: unknown): number {
  if (v === undefined || v === null || v === '') return SA3_DEFAULT_SEED_BLEND;
  const n = Number(v);
  if (!Number.isFinite(n)) return SA3_DEFAULT_SEED_BLEND;
  return Math.min(1, Math.max(0, n));
}

/** Same pattern as minimax's mapMinimaxSamplerPlugins: the solver / scheduler /
 *  guidance pickers are SHARED global UI state, so they only reach the wire
 *  when the user has opted in (params.samplerPluginsEnabled). Otherwise the
 *  render stays on the parity-tested native pingpong path.
 *
 *  Exported so the Storm slot builder can reuse it without duplicating the
 *  plugin-filtering logic. */
export function mapSa3SamplerPlugins(params: any): Pick<Sa3SynthWire,
  'solver' | 'scheduler' | 'guidanceMode' | 'guidanceScale' | 'pluginParams' | 'postprocessPlugin'> {
  if (!params.samplerPluginsEnabled) return {};
  const out: ReturnType<typeof mapSa3SamplerPlugins> = {};
  if (params.inferMethod)  out.solver       = params.inferMethod;
  if (params.scheduler)    out.scheduler    = params.scheduler;
  if (params.guidanceMode) out.guidanceMode = params.guidanceMode;
  // Postprocess plugin: replaces the built-in latent->audio decode. Same UI
  // key ACE's backend uses (params.postprocessPlugin), gated behind the same
  // opt-in toggle as solver/scheduler/guidance.
  if (params.postprocessPlugin) out.postprocessPlugin = params.postprocessPlugin;
  const gs = Number(params.guidanceScale);
  if (Number.isFinite(gs) && gs > 1) out.guidanceScale = gs;
  // The shared pluginParams store holds every "<plugin>:<key>" the user has
  // ever touched, across every plugin -- easily 150+ entries. /sa3-synth takes
  // them on the query string (same wire as /sa3-refine), so sending the lot
  // is an HTTP 414. Forward only the keys that belong to the plugins actually
  // selected; the rest can't reach anything on this render anyway.
  if (params.pluginParams && Object.keys(params.pluginParams).length > 0) {
    const wanted = [out.solver, out.scheduler, out.guidanceMode, out.postprocessPlugin]
      .filter(Boolean) as string[];
    const pp: Record<string, string | number> = {};
    for (const [k, v] of Object.entries(params.pluginParams as Record<string, string | number>)) {
      const colon = k.indexOf(':');
      if (colon <= 0) continue;
      const plugin = k.slice(0, colon);
      if (wanted.includes(plugin)) pp[k] = v;
    }
    if (Object.keys(pp).length > 0) out.pluginParams = pp;
  }
  return out;
}

/** StableStep adapters, reused as-is: the same models/sa3-adapters/*.gguf,
 *  the same merge-at-load path. The UI stores them under the StableStep keys
 *  (stableStepAdapters: [{name, scale}]) — read those so one adapter picker
 *  serves both the refine and the primary render.
 *
 *  Exported so the Storm slot builder can reuse it without duplicating the
 *  adapter-filtering logic. */
export function mapSa3Adapters(params: any): Array<{ name: string; scale: number }> {
  const raw = params.sa3Adapters ?? params.stableStepAdapters;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((a: any) => a && typeof a.name === 'string' && a.name.trim())
    .map((a: any) => ({
      name: String(a.name).trim(),
      scale: Number.isFinite(Number(a.scale)) ? Number(a.scale) : 1.0,
    }));
}

/** Keep the complete source and its rate; the engine handles resampling and tiling. */
export async function prepareSa3Cover(params: any, signal?: AbortSignal) {
  if (typeof params.sourceAudioUrl !== 'string' || !params.sourceAudioUrl) {
    throw new Error('SA3 cover needs source audio');
  }
  const sourcePath = resolveSourceAudioPath(params.sourceAudioUrl);
  let source = fs.readFileSync(sourcePath);
  const { format } = await parseBuffer(source, undefined, { duration: true });
  const durationSec = format.duration, sampleRate = format.sampleRate;
  if (!durationSec || !Number.isFinite(durationSec) || !sampleRate) {
    throw new Error('SA3 cover source needs a readable duration and sample rate');
  }
  // The engine decodes WAV/MP3. Decode other uploads without changing rate
  // or trimming the track, unlike the shared 48 kHz ACE conversion helper.
  if (!['.wav', '.mp3'].includes(path.extname(sourcePath).toLowerCase())) {
    const ffmpeg = getFFmpegPath();
    if (!ffmpeg) throw new Error('SA3 cover needs ffmpeg to decode this source format');
    fs.mkdirSync(config.data.audioDir, { recursive: true });
    const temp = fs.mkdtempSync(path.join(config.data.audioDir, 'sa3-cover-source-'));
    try {
      const decoded = path.join(temp, 'source.wav');
      await promisify(execFile)(ffmpeg, ['-y', '-i', sourcePath, '-vn', '-ac', '2',
        '-c:a', 'pcm_s16le', decoded], { timeout: 120_000, windowsHide: true, signal });
      source = fs.readFileSync(decoded);
    } finally { fs.rmSync(temp, { recursive: true, force: true }); }
  }
  const strength = params.sa3Cover?.transform ?? 0.3;
  if (typeof strength !== 'number' || !Number.isFinite(strength) || strength < 0.1 || strength > 0.9) {
    throw new Error('SA3 cover transform must be between 0.1 and 0.9');
  }
  const raw = params.sa3Cover?.adapters ?? '';
  if (typeof raw !== 'string') throw new Error('SA3 cover adapters must be name:scale pairs');
  const adapters = raw.trim() ? raw.split(',').map(pair => {
    const match = pair.trim().match(/^([^:,]+):([^:,]+)$/);
    const scale = match && match[2].trim() ? Number(match[2]) : NaN;
    if (!match || !match[1].trim() || /[/\\]|\.\./.test(match[1]) || !Number.isFinite(scale)) {
      throw new Error('SA3 cover adapters must be name:scale pairs');
    }
    return { name: match[1].trim(), scale };
  }) : [];
  return { source, durationSec, sampleRate, strength, adapters };
}

export async function runSa3Generation(job: GenerationJob, deps: Sa3GenerationDeps): Promise<void> {
  const log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) =>
    logGeneration(job.id, level, msg);
  const timing: StageTiming[] = [];
  const pipelineStart = performance.now();

  // Defensive residency arbitration (plan §4.4): an MM3 run may have left ~13 GB
  // parked, and SA3 needs the card as much as ACE does. Idempotent and cheap
  // when cold; short-fused so a hung engine can never stall an SA3 render.
  await releaseMinimaxVramForAce();

  // ── Map params ────────────────────────────────────────────────────────
  const isCover = job.params.taskType === 'cover';
  const rawCaption = String(job.params.caption ?? job.params.style ?? '');
  const caption = isCover ? rawCaption : rawCaption.trim();
  if (!caption.trim()) {
    job.status = 'failed';
    job.error  = 'Stable Audio 3 needs a caption (Style Description)';
    job.stage  = 'Failed';
    return;
  }

  let durationSec = Number(job.params.duration);
  if (!Number.isFinite(durationSec) || durationSec < 1) durationSec = 30;
  if (durationSec > SA3_MAX_DURATION_SEC) durationSec = SA3_MAX_DURATION_SEC;

  let steps = Number(job.params.sa3Steps);
  if (!Number.isFinite(steps) || steps < 1) steps = SA3_DEFAULT_STEPS;
  if (steps > 64) steps = 64;


  // Seed: honour a fixed one, otherwise draw and WRITE IT BACK so the row is
  // reproducible — same discipline as the ACE and MM3 paths.
  let seed = Number(job.params.seed);
  if (job.params.randomSeed || !Number.isFinite(seed) || seed < 0) {
    seed = Math.floor(Math.random() * 0x7fffffff);
  }
  seed = Math.floor(seed);
  job.params.seed = seed;

  const adapters = mapSa3Adapters(job.params);
  const plugins  = mapSa3SamplerPlugins(job.params);
  const { sampler, solver: defaultSolver } = resolveSa3Sampler(job.params, plugins);
  if (defaultSolver) plugins.solver = defaultSolver;

  // /sa3-synth is one long synchronous fetch — the shared lane's abort signal
  // (deps.signal) is what makes a user cancel actually stop it rather than
  // just flip job.status while the render keeps running server-side.
  const { signal } = deps;

  startGenerationLog(job.id, TASK);

  try {
    const cover = isCover ? await prepareSa3Cover(job.params, signal) : undefined;
    if (cover) durationSec = cover.durationSec;
    const renderSteps = cover ? Math.floor(steps) : steps;
    const renderSampler = cover ? 'pingpong' as const : sampler;
    const renderAdapters = cover ? cover.adapters : adapters;
    const prompt = cover ? `${caption}\nLength: ${Math.round(durationSec)} seconds` : caption;
    // ── Tokenise ──────────────────────────────────────────────────────────
    job.status   = 'synth_running';
    job.stage    = SA3_LOADING_STAGE;
    job.progress = 2;
    const tokStart = performance.now();
    const { ids, nTokens } = await tokenizeForSa3(prompt);
    timing.push({ name: 'Tokenise', ms: Math.round(performance.now() - tokStart) });
    log('INFO', `[SA3] Prompt: ${nTokens}/256 tokens`);

    const wire: Sa3SynthWire = {
      synthModel: job.envelope.models.dit || SA3_DEFAULT_DIT,
      caption, prompt, nTokens, durationSec, steps: renderSteps, sampler: renderSampler, seed,
      adapters: renderAdapters, ...(cover ? { sourceAudioUrl: job.params.sourceAudioUrl,
        strength: cover.strength, sampleRate: cover.sampleRate } : plugins),
    };
    logGenerationParams(job.id, wire as unknown as Record<string, unknown>);
    if (wire.solver || wire.scheduler || wire.guidanceMode) {
      log('INFO', `[SA3] Plugins: solver=${wire.solver ?? '(pingpong)'} `
        + `scheduler=${wire.scheduler ?? '(sa3 logsnr)'} `
        + `guidance=${wire.guidanceMode ?? '(none)'}`);
    }
    if (renderAdapters.length > 0) {
      log('INFO', `[SA3] Adapters: ${renderAdapters.map(a => `${a.name}:${a.scale}`).join(', ')}`);
    }

    // ── Synth (one synchronous call) ──────────────────────────────────────
    job.stage    = sa3RenderStage(durationSec, renderSteps);
    job.progress = 5;
    if (signal.aborted || (job.status as string) === 'cancelled') throw new Error('Cancelled');
    const synthStart = performance.now();
    const wavBuf = cover ? await aceClient.submitSa3Refine(cover.source, {
      tokens: ids, nTokens, strength: cover.strength, seed, adapters: cover.adapters,
      steps: renderSteps, sampler: renderSampler, outSr: cover.sampleRate, outFmt: 'f32', signal,
    }) : await aceClient.submitSa3Synth({
      synthModel: wire.synthModel,
      onModelResolved(filename) {
        wire.resolvedSynthModel = filename;
        if (filename !== wire.synthModel) {
          log('WARNING', `[SA3] DiT fallback: requested ${wire.synthModel}, loaded ${filename}`);
        }
      },
      tokens: ids,
      nTokens,
      durationSec,
      steps,
      sampler,
      seed,
      // Seed breeding: slerp the initial noise toward seedB (engine /sa3-synth).
      ...(job.params.seedB !== undefined && job.params.seedB !== null && job.params.seedB !== ''
          && Number(job.params.seedB) >= 0
        ? { seedB: Math.floor(Number(job.params.seedB)),
            // SA3 default: a shared seedB with no explicit blend breeds at 0.5
            // (ACE omits the blend instead). See docs/user/backends.md.
            seedBlend: resolveSa3SeedBlend(job.params.seedBlend) }
        : {}),
      adapters,
      outFmt: 'f32',   // keep peaks; the post chain owns the ceiling
      // Diagnostic. Set params.sa3DumpLatent to a bare filename to get the
      // final latent on disk + its RMS in the engine log.
      ...(job.params.sa3DumpLatent ? { dumpLatent: String(job.params.sa3DumpLatent) } : {}),
      ...plugins,
      signal,
    });
    const synthMs = Math.round(performance.now() - synthStart);
    timing.push({ name: cover ? 'SA3 cover refine' : 'SA3 synth', ms: synthMs });
    if (signal.aborted || (job.status as string) === 'cancelled') throw new Error('Cancelled');
    if (!wavBuf || wavBuf.length === 0) throw new Error('Stable Audio 3 returned an empty audio body');

    // ── Save ──────────────────────────────────────────────────────────────
    job.stage    = 'Stable Audio 3: saving audio...';
    job.progress = 95;
    const saveStart = performance.now();
    const filename = `${uuidv4()}.wav`;
    const filepath = path.join(config.data.audioDir, filename);
    fs.writeFileSync(filepath, wavBuf);
    const audioUrl = `/audio/${filename}`;
    const measured = wavDurationSec(filepath);
    const duration = measured > 0 ? Math.round(measured) : Math.round(durationSec);
    log('INFO', `[SA3] Rendered ${duration}s in ${synthMs} ms → ${filename} (${(wavBuf.length / 1024).toFixed(0)} KB)`);

    // ── Optional loudness normalization ───────────────────────────────────
    // Off by default. When enabled, normalizes the WAV in place to the target
    // integrated LUFS with a peak ceiling of -1 dBFS before post-processing.
    // The raw SA3 output level swings ~18 dB across captions; this collapses
    // that variance at the cost of one extra read+write pass on the file.
    if (job.params.sa3LufsEnabled === true) {
      const lufsTarget = Number(job.params.sa3LufsTarget);
      const target = Number.isFinite(lufsTarget) && lufsTarget >= -40 && lufsTarget <= -3
        ? lufsTarget : -14;
      try {
        const lufsStart = performance.now();
        const lufsResult = normalizeLufs(filepath, target, -1.0, { outFormat: 'f32' });
        const lufsMs = Math.round(performance.now() - lufsStart);
        log('INFO', `[SA3] Loudness norm: measured ${lufsResult.measuredLufs.toFixed(1)} LUFS `
          + `→ target ${target} LUFS, gain ${lufsResult.appliedGainDb.toFixed(2)} dB, `
          + `peak after ${lufsResult.peakAfter.toFixed(4)}, limiter ${lufsResult.limiterActive ? 'active' : 'off'} `
          + `(${lufsMs} ms)`);
        timing.push({ name: 'LUFS normalize', ms: lufsMs });
      } catch (lufsErr: any) {
        log('WARNING', `[SA3] Loudness normalization failed (non-fatal): ${lufsErr?.message || lufsErr}`);
      }
    }
    timing.push({ name: 'Save', ms: Math.round(performance.now() - saveStart) });

    // ── Persist params ────────────────────────────────────────────────────
    const captionLine = caption.split('\n').map(s => s.trim()).find(s => s.length > 0) || '';
    const title: string = job.params.title || captionLine.substring(0, 60) || 'Untitled';
    const trackParams = {
      ...job.params,
      backend: 'sa3',
      seed,
      duration: durationSec,
      sa3Request: wire,
      sa3DitModel: wire.resolvedSynthModel ?? wire.synthModel,
      sa3: { steps: renderSteps, sampler: renderSampler,
        sample_rate: cover?.sampleRate ?? 44100, prompt_tokens: nTokens },
    };

    // ── Post-processing (model-agnostic subset, same as MM3) ─────────────
    // SA3 is natively 44.1 kHz. VST + mastering read the rate from the WAV.
    // PP-VAE and Spectral Lifter are ACE-coupled and stay off. StableStep
    // (refine with SA3) is off in this backend's manifest.
    let masteredUrl = '';
    try {
      const ppParams: PostProcessParams = {
        ...job.params,
        ppVaeReencode: false,
        spectralLifterEnabled: false,
        stableStepOn: false,
        stableStep: false,
        // No vocal stem to protect; SA3 renders whatever the caption asked for
        // and the post chain must not try to split it apart.
        instrumental: true,
      };
      if (ppParams.postProcessingEnabled !== false) {
        const ppStart = performance.now();
        const ppResult = await runPostProcessingChain(
          [audioUrl], ppParams, 1, job.id,
          log, (stage) => { job.stage = stage; },
        );
        masteredUrl = ppResult.masteredUrls?.[0] || '';
        const ppMs = Math.round(performance.now() - ppStart);
        if (masteredUrl) log('INFO', `[SA3] Post-processing produced ${masteredUrl} in ${ppMs} ms`);
        timing.push({ name: 'Post-processing', ms: ppMs });
      }
    } catch (ppErr: any) {
      log('WARNING', `[SA3] Post-processing chain failed (non-fatal): ${ppErr?.message || ppErr}`);
    }

    // ── Songs row ─────────────────────────────────────────────────────────
    const songId = uuidv4();
    getDb().prepare(`
      INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                         duration, bpm, key_scale, time_signature, tags, dit_model,
                         generation_params, mastered_audio_url, latent_url, quality_scores,
                         noadapter_audio_url, backend)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      songId, job.userId, title, '', caption, caption,
      audioUrl, duration, 0, '', '',
      JSON.stringify([]), 'sa3', JSON.stringify(trackParams),
      masteredUrl, '', '',
      '', 'sa3',
    );

    // ── Done ──────────────────────────────────────────────────────────────
    const totalMs = Math.round(performance.now() - pipelineStart);
    timing.push({ name: 'TOTAL', ms: totalMs });
    job.status   = 'succeeded';
    job.progress = 100;
    job.stage    = 'Complete!';
    job.result   = { audioUrls: [audioUrl], songIds: [songId], duration, timing, totalMs };

    log('INFO', `[Result] 1 audio file saved, 1 song created (backend=sa3)`);
    console.log(`[Generate] Job ${job.id} (sa3) completed in ${(totalMs / 1000).toFixed(1)}s`);
    finishGenerationLog(job.id, TASK);

  } catch (err: any) {
    if (err.message === 'Cancelled' || err?.name === 'AbortError' || (job.status as string) === 'cancelled') {
      job.status = 'cancelled';
      job.stage  = 'Cancelled';
      failGenerationLog(job.id, 'Cancelled by user', TASK);
    } else {
      job.status = 'failed';
      job.error  = err.message || 'Unknown error';
      job.stage  = 'Failed';
      console.error(`[Generate] Job ${job.id} (sa3) failed:`, err.message);
      failGenerationLog(job.id, err.message || 'Unknown error', TASK);
    }
  }
}
