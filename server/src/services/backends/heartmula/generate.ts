import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { config } from '../../../config.js';
import { getDb } from '../../../db/database.js';
import { aceClient } from '../../aceClient.js';
import { wavDurationSec } from '../../audioCrop.js';
import { runPostProcessingChain } from '../../generation/postProcessing.js';
import { startGenerationLog, logGeneration, logGenerationParams, finishGenerationLog, failGenerationLog } from '../../logger.js';
import { heartmulaClient } from './client.js';
import type { HeartmulaCoverRequest, HeartmulaRequest } from './client.js';
import type { HeartmulaCoverRolls } from '../../../contracts/studioWorkflows.js';
import type { GenerationJob } from '../../generation/jobTypes.js';
import type { GenerationContext, GenerationOutcome, BackendModelSelection } from '../types.js';
import { heartmulaAdapterInfo, heartmulaPick, withHeartmulaTrigger } from './models.js';
import { heartmulaStageText } from './progress.js';

function number(value: unknown, name: string, fallback: number, min: number, max: number, integer = false): number {
  if (value === undefined) return fallback;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max
      || integer && !Number.isInteger(value)) throw new Error(`${name} must be ${integer ? 'an integer' : 'a number'} in ${min}..${max}`);
  return value;
}

function seed(value: unknown, name: string): number | string | undefined {
  if (value === undefined) return undefined;
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return value;
  if (typeof value === 'string' && /^\d+$/.test(value) && BigInt(value) <= 0xffffffffffffffffn) return value;
  throw new Error(`${name} must be a non-negative safe integer or uint64 decimal string`);
}

/** The shared solver/scheduler/guidance picks, only while this backend's toggle is on: the
 *  pickers always hold ACE's defaults (euler + APG), so forwarding them unasked would move
 *  every render off the native decode. */
function samplerPlugins(params: Readonly<Record<string, unknown>>): Partial<HeartmulaRequest> {
  if (params.samplerPluginsEnabled !== true) return {};
  const out: Partial<HeartmulaRequest> = {};
  for (const [from, to] of [['inferMethod', 'infer_method'], ['scheduler', 'scheduler'], ['guidanceMode', 'guidance_mode']] as const) {
    if (typeof params[from] === 'string' && params[from]) out[to] = params[from] as string;
  }
  if (!Object.keys(out).length) return {};
  out.flow_shift = number(params.heartmulaFlowShift, 'schedule shift', 1, 0.01, 20);
  const apg = Number(params.apgNormThreshold);
  if (params.apgNormThreshold !== undefined && Number.isFinite(apg) && apg >= 0 && apg <= 100) out.apg_norm_threshold = apg;
  if (params.pluginParams && typeof params.pluginParams === 'object' && Object.keys(params.pluginParams).length) {
    out.plugin_params = params.pluginParams as Record<string, unknown>;
  }
  return out;
}

/** MuLaCover's duration limit in the engine (/heartmula/props cover_max_duration_s). */
export const HEARTMULA_COVER_MAX_DURATION_S = 300;

/** Validate at enqueue time, before the job can enter the shared GPU lane. A cover
 *  (Cover Studio's rolls) goes to /heartmula/cover on the MuLaCover variant, never
 *  to plain synth; the engine checks the roll sizes and values. */
export function mapHeartmulaRequest(params: Readonly<Record<string, unknown>>, models: BackendModelSelection): HeartmulaRequest | HeartmulaCoverRequest {
  const cover = params.taskType === 'cover' || params.heartmulaCover !== undefined;
  const rolls = params.heartmulaCover as HeartmulaCoverRolls | undefined;
  if (cover && (!rolls || typeof rolls !== 'object' || rolls.version !== 1)) {
    throw new Error('A HeartMuLa cover needs the lead-sheet rolls Cover Studio builds from the approved score');
  }
  const tags = params.heartmulaTags ?? params.tags ?? params.caption ?? params.prompt ?? params.songDescription ?? params.style;
  if (typeof tags !== 'string' || !tags.trim() || Buffer.byteLength(tags) > 4096) throw new Error('tags must be non-empty text of at most 4096 bytes');
  const lyrics = params.instrumental === true ? '' : params.lyrics ?? '';
  if (typeof lyrics !== 'string' || Buffer.byteLength(lyrics) > 32768) throw new Error('lyrics must be text of at most 32768 bytes');
  if (params.randomSeed !== undefined && typeof params.randomSeed !== 'boolean') throw new Error('randomSeed must be boolean');
  if (params.batchSize !== undefined && params.batchSize !== 1) throw new Error('HeartMuLa supports one song per request');
  for (const key of ['variant', 'global', 'local', 'flow', 'scalar_decoder', 'lmAdapter']) {
    if (typeof models[key] !== 'string') throw new Error(`Invalid HeartMuLa model selection: ${key}`);
  }
  // MuLaCover has its own LM pair and takes no HeartMuLa adapter; the codec pick is shared.
  const pick = cover ? { variant: 'MuLaCover', global_type: '', local_type: '', flow_type: models.flow || '',
    scalar_decoder_type: models.scalar_decoder || '', adapter: '', adapter_scale: 1 }
    : heartmulaPick(models, number(params.heartmulaAdapterStrength, 'adapter strength', 1, 0, 2));
  // The adapter learned its trigger at the head of every sample's tags; lead with it here too.
  const tagged = pick.adapter ? withHeartmulaTrigger(tags, heartmulaAdapterInfo(pick.adapter).trigger) : tags;
  if (Buffer.byteLength(tagged) > 4096) throw new Error('tags with the adapter trigger exceed 4096 bytes');
  // The group dials only mean something with an adapter, so a base render's wire stays as it was.
  const groups = pick.adapter ? {
    adapter_scale_attn: number(params.heartmulaAdapterScaleAttn, 'adapter attention scale', 1, 0, 2),
    adapter_scale_mlp: number(params.heartmulaAdapterScaleMlp, 'adapter MLP scale', 1, 0, 2),
    adapter_scale_early: number(params.heartmulaAdapterScaleEarly, 'adapter early scale', 1, 0, 2),
    adapter_scale_mid: number(params.heartmulaAdapterScaleMid, 'adapter mid scale', 1, 0, 2),
    adapter_scale_late: number(params.heartmulaAdapterScaleLate, 'adapter late scale', 1, 0, 2),
    adapter_rebase: number(params.heartmulaAdapterRebase, 'adapter re-base strength', 0.75, 0, 1),
  } : {};
  const mainSeed = seed(params.seed, 'seed');
  const noiseSeed = seed(params.heartmulaNoiseSeed, 'noise seed');
  let coverDuration = HEARTMULA_COVER_MAX_DURATION_S;
  if (cover) {
    let rows = number(rolls!.length, 'lead-sheet length', 0, 1, 5000, true);
    const bpm = number(rolls!.bpm, 'lead-sheet bpm', 0, 1e-6, 1000);
    if (!rows || !bpm || !Array.isArray(rolls!.context_mask) || rolls!.context_mask.length !== rows
        || rolls!.context_mask.some(value => value !== 0 && value !== 1)) {
      throw new Error('Invalid lead-sheet length, bpm or context mask');
    }
    // Only trailing padding shortens the sheet; masked rows inside it retain their time.
    while (rows > 0 && rolls!.context_mask[rows - 1] === 1) rows--;
    coverDuration = Math.max(1, Math.min(HEARTMULA_COVER_MAX_DURATION_S, rows * 15 / bpm));
  }
  return {
    ...pick, ...groups, tags: tagged, lyrics,
    // Create's Auto duration uses -1. Let audio EOS end the song within
    // HeartMuLa's maximum duration, while an omitted duration keeps 120 s.
    // Covers stop at the unpadded sheet length or a shorter caller limit.
    duration_s: cover
      ? Math.min(coverDuration, number(params.duration === -1 ? HEARTMULA_COVER_MAX_DURATION_S : params.duration, 'duration',
        HEARTMULA_COVER_MAX_DURATION_S, 1, HEARTMULA_COVER_MAX_DURATION_S))
      : number(params.duration === -1 ? 240 : params.duration, 'duration', 120, 1, 240),
    ...(!params.randomSeed && mainSeed !== undefined ? { seed: mainSeed } : {}),
    ...(noiseSeed === undefined ? {} : { noise_seed: noiseSeed }),
    temperature: number(params.heartmulaTemperature, 'temperature', 1, Number.MIN_VALUE, 4),
    topk: number(params.heartmulaTopK, 'topk', cover ? 250 : 50, 1, 8197, true),
    cfg_scale: number(params.heartmulaCfgScale, 'cfg scale', 1.5, 1, 10),
    codec_steps: number(params.heartmulaCodecSteps, 'codec steps', 10, 1, 100, true),
    codec_guidance: number(params.heartmulaCodecGuidance, 'codec guidance', 1.25, 1, 10),
    ...samplerPlugins(params),
    ...(cover ? { rolls: rolls! } : {}),
  };
}

export async function runHeartmulaGeneration(job: GenerationJob, ctx: GenerationContext): Promise<GenerationOutcome> {
  const started = performance.now();
  const stages: GenerationOutcome['stages'] = [];
  const artifacts: GenerationOutcome['artifacts'] = [];
  const log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', text: string) => logGeneration(job.id, level, text);
  const checkCancelled = () => {
    if (ctx.signal.aborted || job.status === 'cancelled') throw new Error('Cancelled');
  };
  const task = job.params.taskType === 'cover' ? 'heartmula-cover' : 'heartmula-text2music';
  startGenerationLog(job.id, task);
  try {
    checkCancelled();
    const wire = { ...ctx.envelope.options.heartmula } as unknown as HeartmulaRequest | HeartmulaCoverRequest;
    // The retry loop may have drawn a new seed. Models and all other inputs
    // remain the immutable enqueue-time snapshot.
    if (ctx.attempt.reseeded) wire.seed = job.params.seed;
    if (wire.seed === undefined) wire.seed = Math.floor(Math.random() * 0x7fffffff);
    job.params.seed = wire.seed;
    ctx.attempt.effective.seed = wire.seed;
    ctx.attempt.effective.models = { ...ctx.envelope.models };
    // The rolls run to megabytes: log and store their size, not their values.
    const { rolls, ...recorded } = wire as Partial<HeartmulaCoverRequest> & HeartmulaRequest;
    logGenerationParams(job.id, recorded as unknown as Record<string, unknown>);
    if (rolls) log('INFO', `[HeartMuLa] Cover lead sheet: ${rolls.length} sixteenth-note rows at ${rolls.bpm} BPM`);
    // What the engine is conditioned on, in plain text (the engine log repeats it).
    log('INFO', `[HeartMuLa] Tags sent to engine: ${wire.tags}`);
    if (wire.adapter) log('INFO', `[HeartMuLa] Adapter: ${path.basename(wire.adapter)} (strength ${wire.adapter_scale})`);
    if (!wire.lyrics) log('INFO', '[HeartMuLa] Lyrics sent to engine: (none)');
    else {
      log('INFO', '[HeartMuLa] Lyrics sent to engine:');
      for (const line of wire.lyrics.split('\n')) log('INFO', `[HeartMuLa]   ${line}`);
    }
    job.status = 'synth_running';
    job.stage = 'Loading HeartMuLa models...';
    job.progress = 2;
    const engineId = await heartmulaClient.synth(wire);
    // Register before observing abort: cancelling a submit must still cancel
    // the engine job that was accepted while the HTTP response was in flight.
    job.aceJobId = engineId;
    ctx.attempt.engineJobIds.push(engineId);
    ctx.hooks.onEngineJob(engineId);
    checkCancelled();
    // GET /job (pollUntilDone) only carries the shared phase; /heartmula/job has the
    // stage and its counters, so the queue card shows where the render is.
    const ticker = setInterval(() => {
      void heartmulaClient.job(engineId).then(d => {
        const text = job.status === 'cancelled' ? null : heartmulaStageText(d);
        if (!text) return;
        job.stage = text.stage;
        job.progress = Math.max(job.progress ?? 0, text.progress);
      }).catch(() => {});
    }, 750);
    ticker.unref?.();
    try {
      await ctx.pollUntilDone(engineId, job, ctx.signal, ctx.envelope.policy.timeoutMinutes);
    } finally {
      clearInterval(ticker);
    }
    checkCancelled();
    const detail = await heartmulaClient.job(engineId);
    if (detail.status !== 'done') throw new Error(detail.error?.message || `HeartMuLa job ended with ${detail.status}`);
    ctx.attempt.effective.seed = detail.seed;
    ctx.attempt.effective.noiseSeed = detail.noise_seed;
    ctx.attempt.effective.heartmulaPick = detail.pick;
    ctx.attempt.effective.endReason = detail.end_reason;
    const response = await aceClient.getJobResult(engineId);
    if (!response.ok) throw new Error(`HeartMuLa audio result HTTP ${response.status}`);
    const audio = Buffer.from(await response.arrayBuffer());
    checkCancelled();
    if (audio.length < 44 || audio.toString('ascii', 0, 4) !== 'RIFF' || audio.toString('ascii', 8, 12) !== 'WAVE') {
      throw new Error('HeartMuLa returned an invalid WAV result');
    }
    stages.push({ name: 'HeartMuLa synth', ms: Math.round(performance.now() - started) });
    const filename = `${randomUUID()}.wav`;
    const filepath = path.join(config.data.audioDir, filename);
    fs.writeFileSync(filepath, audio);
    const audioUrl = `/audio/${filename}`;
    const duration = wavDurationSec(filepath) || wire.duration_s;
    artifacts.push({ kind: 'audio', trackIndex: 0, url: audioUrl });
    ctx.hooks.onArtifact(artifacts[0]);
    job.status = 'saving';
    job.stage = 'HeartMuLa: saving audio...';
    job.progress = 95;
    let masteredUrl = '';
    if (job.params.postProcessingEnabled !== false) {
      try {
        const pp = await runPostProcessingChain([audioUrl], {
          ...job.params, ppVaeReencode: false, spectralLifterEnabled: false,
        }, 1, job.id, log, stage => { job.stage = stage; });
        masteredUrl = pp.masteredUrls?.[0] || '';
      } catch (err) { checkCancelled(); log('WARNING', `Post-processing failed: ${err instanceof Error ? err.message : String(err)}`); }
    }
    checkCancelled();
    const songId = randomUUID();
    const { heartmulaCover: _rolls, ...params } = job.params as Record<string, unknown>;
    const trackParams = {
      ...params, backend: 'heartmula', seed: detail.seed,
      heartmulaRequest: recorded, heartmulaResult: detail,
    };
    getDb().prepare(`INSERT INTO songs
      (id, user_id, title, lyrics, style, caption, audio_url, duration, tags,
       dit_model, generation_params, mastered_audio_url, backend)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      songId, job.userId, job.params.title || wire.tags.slice(0, 60), wire.lyrics,
      wire.tags, wire.tags, audioUrl, duration, JSON.stringify([]), 'heartmula',
      JSON.stringify(trackParams), masteredUrl, 'heartmula',
    );
    if (masteredUrl) {
      const mastered = { kind: 'mastered', trackIndex: 0, url: masteredUrl };
      artifacts.push(mastered);
      ctx.hooks.onArtifact(mastered);
    }
    const totalMs = Math.round(performance.now() - started);
    stages.push({ name: 'TOTAL', ms: totalMs });
    job.result = { audioUrls: [audioUrl], songIds: [songId], duration, timing: stages, totalMs,
      ...(masteredUrl ? { masteredAudioUrl: masteredUrl } : {}) };
    job.status = 'succeeded';
    job.stage = 'Complete!';
    job.progress = 100;
    finishGenerationLog(job.id, task);
    return { endReason: 'completed', stages, artifacts, songIds: [songId], result: job.result };
  } catch (err) {
    const cancelled = ctx.signal.aborted || job.status === 'cancelled' || err instanceof Error && err.message.startsWith('Cancelled');
    // Cancel uses the shared job route. The engine serialises cancellation,
    // unload and the next synth on its single worker.
    if (job.aceJobId) await aceClient.cancelJob(job.aceJobId).catch(() => {});
    job.status = cancelled ? 'cancelled' : 'failed';
    job.stage = cancelled ? 'Cancelled' : 'Failed';
    job.error = err instanceof Error ? err.message : String(err);
    if (!cancelled && job.aceJobId) {
      const detail = await heartmulaClient.job(job.aceJobId).catch(() => null);
      if (detail?.error) job.error = `${detail.error.code}: ${detail.error.message}`;
      if (detail?.admission) ctx.attempt.effective.admission = detail.admission;
    }
    failGenerationLog(job.id, job.error, task);
    return { endReason: cancelled ? 'cancelled' : 'failed', stages, artifacts: [], songIds: [], error: job.error };
  }
}
