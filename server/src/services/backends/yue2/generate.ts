// backends/yue2/generate.ts — the YuE2 generation path
//
// Runs a dequeued generation job against the /yue2/* engine family. Modeled
// directly on backends/minimax/generate.ts's finalisation (song row insert,
// WAV write, post-processing chain, cover art, whisper) MINUS the MM3-only
// pieces that have no YuE2 analog in v1: AR plank/hiddens replay, streaming,
// ensemble takes, LRC (YuE2's DiT has no lyric-alignment head yet — matches
// capabilities().features.lyricTimestamps = false).
//
// WHAT IS SHARED WITH ACE/MM3 (deliberately):
//   • the serial job queue + retry wrapper (routes/generate.ts:enqueueGeneration)
//   • the in-memory job map, /status/:id, /cancel/:id, the SSE progress shape
//   • pollUntilDone's watchdog (injected) — YuE2 jobs ride the SAME shared
//     /job endpoints as ACE and MM3 (aceClient.pollJob/getJobResult/cancelJob)
//   • the audio dir + /audio/<uuid>.wav URL convention, and the songs INSERT
//   • cover art, whisper (both backend-agnostic already)
//   • the model-agnostic post-processing chain (VST, StableStep, mastering) —
//     unlike MM3, YuE2 renders NATIVE 48 kHz stereo (yue2vae.sample_rate),
//     so none of MM3's rate-audit reasoning for skipping the chain applies.
//     ppVaeReencode and the Spectral Lifter stay excluded regardless of rate:
//     both round-trip through or were tuned against the ACE VAE specifically.
//
// WHAT IS NOT REUSED (and why):
//   • translateParams / AceRequest — YuE2's wire contract is the small
//     style/lyrics/cot/cfg_scale/ode_steps/vae_variant/seed set (yue2/client.ts).
//   • the LM phase, LM cache, LM-echo rebuild — no two-request seam here.
//   • adapters, latents, source audio, task modes (v1 operations: text2music only).
//   • MM3's plank/hiddens/streaming/takes/LRC machinery — no YuE2 analog yet.

import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';

import { config } from '../../../config.js';
import { getDb } from '../../../db/database.js';
import { aceClient } from '../../aceClient.js';
import { wavDurationSec } from '../../audioCrop.js';
import { runPostProcessingChain } from '../../generation/postProcessing.js';
import type { PostProcessParams } from '../../generation/postProcessing.js';
import {
  startGenerationLog, logGeneration, logGenerationParams,
  finishGenerationLog, failGenerationLog,
} from '../../logger.js';
import { readSafetensorsMeta } from '../../training/yue2Runs.js';
import { jointRunForAdapter } from '../../training/yue2AitkRuns.js';
import { yue2AdapterTrigger } from './jointAdapterContext.js';
import type { Yue2AdapterScales, Yue2FinalDetail } from './client.js';
import { yue2Align, yue2Synth, yue2FinalDetail, yue2Props, yue2PropsCached, type Yue2SynthRequest, type Yue2TrackDetail } from './client.js';
import { yue2LyricsJson } from './align.js';
import { classifyYue2Score, type Yue2ScoreHealth, yue2PlanUsable } from './scoreHealth.js';
import { yue2PersistedSelection } from './index.js';
import { applyYue2StyleTemplate, splitYue2Tail, type Yue2StyleTemplate } from './style.js';
import type { GenerationJob, StageTiming } from '../../generation/jobTypes.js';
import type { GenerationAttempt } from '../types.js';

/** Injected so this module never imports back into routes/generate.ts at
 *  runtime, matching MinimaxGenerationDeps' own shape exactly. */
export interface Yue2GenerationDeps {
  attempt: GenerationAttempt;
  signal: AbortSignal;
  pollUntilDone(
    aceJobId: string, job: GenerationJob, signal: AbortSignal, timeoutMinutes?: number,
  ): Promise<void>;
  /** YuE2 jobs still waiting behind this one, oldest first, for coalescing.
   *  Absent = never coalesce. */
  pendingJobs?(): GenerationJob[];
  /** Lane handoff for the engine's NAR lane (yue2-job.h): once this job is
   *  composed and rendering, the next YuE2 job may start composing. */
  releaseLane?(): boolean;
  nextLaneFamily?(): string | undefined;
  runOnLane?<T>(fn: () => Promise<T>): Promise<T>;
}

/** Mirrors the capability manifest's core.duration (max 360, auto: true). */
const YUE2_MAX_DURATION_SEC = 360;

const DETAIL_POLL_MS = 1_500;

/** How long a YuE2 job with an empty queue behind it waits for siblings before
 *  rendering alone (queue coalescing). */
const YUE2_COALESCE_WAIT_MS = 750;

export interface Yue2ParamMapping {
  req: Yue2SynthRequest;
  notes: string[];
  /** What the user typed, before the adapter's style template wrapped it.
   *  req.style is the prompt the MODEL needs; this is the one a human wrote,
   *  and it is what the song row, the title and StableStep's own SA3 prompt
   *  should show — none of them want "albumA2, in the style of albumA2." */
  caption: string;
  /** The two merged halves and the trigger each was trained on, for the log. */
  halves: Yue2StyleHalves;
}

/**
 * Compose the style prompt the selected LM adapter was actually trained under.
 *
 * Both YuE2 trainers only ever showed the model a trigger INSIDE the style
 * sentence (see style.ts), so sending the caption box verbatim is
 * off-distribution — the reason adapters capped and sang weakly until the
 * template x adapter x seed grid of 2026-09-14 pinned it. Every good render in
 * that campaign used a hand-composed string; this composes the same one.
 *
 * The adapter is ENGINE STATE on this backend rather than a request field
 * (index.ts, lmAdapterSelectable), so where mapMinimaxParams reads
 * params.mm3LmAdapter this reads the persisted pick the picker writes and
 * reconcileSelection replays — the one source that says what is merged into
 * the resident weights.
 */
/** What each merged half is addressed by, for the run log. One render sends ONE
 *  style sentence, so these are not two prompts — they are the two triggers the
 *  two adapters were TRAINED on, and the log exists so a disagreement between
 *  them is visible at a glance instead of after a listening round. */
export interface Yue2StyleHalves {
  ar: { path: string; trigger: string; scales: Yue2AdapterScales };
  nar: { path: string; trigger: string; scales: Yue2AdapterScales };
}

function yue2StyleForAdapter(
  caption: string, params: any,
): { style: string; notes: string[]; halves: Yue2StyleHalves; trainedCot: string } {
  const notes: string[] = [];
  const picked = yue2PersistedSelection().adapters;
  const halfOf = (slot: { path: string; scales: Yue2AdapterScales }) => ({
    path: slot.path,
    trigger: yue2AdapterTrigger(slot.path).trigger,
    scales: slot.scales,
  });
  const halves: Yue2StyleHalves = { ar: halfOf(picked.ar), nar: halfOf(picked.nar) };

  // TWO slots, ONE style sentence. The template composes a single trigger, so
  // when both halves are loaded one of them has to drive it, and that is the
  // NAR pick: it is the half the single-slot picker always meant, and it is the
  // half whose trigger every render in the 2026-09-14 campaign was addressed
  // with. The AR pick still merges and still works — it just does not get to
  // rewrite the prompt. When the two were trained on the same dataset they
  // share a trigger and the question does not arise; when they differ, say so
  // rather than silently addressing one of them.
  const adapter = picked.nar.path || picked.ar.path;
  if (!adapter) return { style: caption, notes, halves, trainedCot: '' };
  const other = adapter === picked.nar.path ? picked.ar.path : '';

  const name = path.basename(adapter);
  const meta = readSafetensorsMeta(adapter);
  // Legacy exports record cot directly. Joint exports currently omit it, but
  // their prepared dataset records which sources could draw full with the
  // native trainer's 50% ABC dropout. Do not call a joint run "off only"
  // merely because its safetensors header lacks the legacy field.
  const jointRun = meta?.cot ? undefined : jointRunForAdapter(adapter);
  let trainedCot = meta?.cot ?? (jointRun ? '' : 'off');
  if (jointRun) {
    const manifest = jointRun.options.dataset;
    if (typeof manifest === 'string') {
      try {
        if (fs.statSync(manifest).size <= 16 * 1024 * 1024) {
          const data = JSON.parse(fs.readFileSync(manifest, 'utf8')) as { items?: Array<{ abc_ids?: unknown[] }> };
          if (Array.isArray(data.items)) trainedCot = data.items.some(item => Array.isArray(item.abc_ids) && item.abc_ids.length > 0)
            ? 'off,full' : 'off';
        }
      } catch { /* missing prepared data leaves joint CoT capability unknown */ }
    }
  }
  if (other) {
    const otherTrigger = yue2AdapterTrigger(other).trigger;
    const thisTrigger = yue2AdapterTrigger(adapter).trigger;
    if (otherTrigger && otherTrigger !== thisTrigger) {
      notes.push(`The AR adapter is addressed by "${otherTrigger}", which is NOT in the style prompt: `
        + 'one sentence can only carry one trigger, and the NAR pick has it. Put the AR trigger in the '
        + 'caption yourself if you want both.');
    }
  }
  // NFC for the same reason the caption gets it: the header's trigger and the
  // caption have to be the same normal form or the already-composed check
  // below compares two spellings of one word and adds a second copy.
  const { trigger, inferred } = yue2AdapterTrigger(adapter);
  if (!trigger) {
    notes.push(`LM adapter "${name}" records no trigger — the style prompt is sent exactly as typed.`);
    return { style: caption, notes, halves, trainedCot };
  }

  // The same opt-out MM3 gives (mm3LmAdapterTrigger): composing is a default,
  // not a cage, and anyone deliberately testing an off-template prompt says so.
  if (params.yue2LmAdapterTrigger === false) {
    notes.push(inferred
      ? `Inferred dataset tag "${trigger}" NOT composed in (yue2LmAdapterTrigger: false); this checkpoint was trained without a trigger.`
      : `LM adapter trigger "${trigger}" NOT composed in (yue2LmAdapterTrigger: false) — this adapter only ever saw its trigger inside the training style sentence, so likeness will be weak.`);
    return { style: caption, notes, halves, trainedCot };
  }

  // Absent means a file exported before --style-template existed, and `bare`
  // is what reproduces that build's behaviour. Both exporters write the field
  // now, so absence is the file's age and not a default to fill in with
  // `upstream`.
  const template: Yue2StyleTemplate = inferred || meta?.styleTemplate === 'upstream' ? 'upstream' : 'bare';
  // The genre/BPM/key tail is part of the trained sentence, not an extra this
  // backend lacks: yue2_style_string() (engine/src/train/yue2-sidecar.h) built
  // "<genre>, <bpm> BPM, key of <key>." from the dataset's sidecar on every
  // artist row, so the model has seen it throughout. There is no wire field
  // for them — the tail IS the channel — so the user's Song Info values are
  // composed into it here.
  //
  // Only when actually set: tail() drops blanks, so a render that leaves BPM
  // at 0 and Key on Auto composes the identical string it did before.
  const userBpm = Number(params.bpm) > 0 ? String(Math.round(Number(params.bpm))) : '';
  const userKey = String(params.keyScale || '').trim();

  // A caption from the dataset ("automatic" caption source) usually ENDS with
  // a tail of its own, because the labeller wrote it in the trained format.
  // Appending a second one gave "…, 178 BPM 168 BPM, key of D minor." So when
  // the user supplies a value, peel the caption's tail off first and let their
  // value win per field, with the caption's own filling anything they left
  // unset. When they supply neither, the caption is not touched at all and the
  // composed string is byte-identical to what it was before any of this.
  const supplied = !!(userBpm || userKey);
  const split = supplied ? splitYue2Tail(caption) : { caption, bpm: '', key: '' };
  const bpm = userBpm || split.bpm;
  const key = userKey || split.key;
  const style = applyYue2StyleTemplate({ trigger, caption: split.caption, template, bpm, key });
  if (supplied) {
    const replaced = [
      userBpm && split.bpm && split.bpm !== userBpm ? `BPM ${split.bpm} -> ${userBpm}` : '',
      userKey && split.key && split.key !== userKey ? `key ${split.key} -> ${userKey}` : '',
    ].filter(Boolean).join(', ');
    notes.push('Style tail: '
      + [bpm ? `${bpm} BPM` : '', key ? `key of ${key}` : ''].filter(Boolean).join(', ')
      + ' — composed into the style sentence, which is the only slot YuE2 has for them.'
      + (replaced ? ` Replaced the caption's own ${replaced}.` : ''));
  }

  if (inferred) {
    notes.push(`Joint adapter trigger "${trigger}" was inferred from the dataset setting; this checkpoint did not record or train with the trigger phrase. Prompt injection is experimental for this run.`);
  }
  if (!caption.trim()) {
    // A caption-dropout run trained artist rows with the caption removed, so
    // the trigger standing alone is a sequence this adapter has really seen.
    notes.push(meta?.captionDropout
      ? `Empty Style Description — rendering from the trigger "${trigger}" alone, which this adapter `
        + `trained for (caption dropout ${meta.captionDropout}).`
      : `Empty Style Description — sending the trigger "${trigger}" alone, which this adapter never `
        + 'trained on (its header records no caption dropout).');
  } else if (style === caption) {
    notes.push(`Style prompt already carried the "${trigger}" ${template} template — left as typed.`);
  } else {
    notes.push(`LM adapter trigger "${trigger}" composed into the style prompt (${template} template).`);
  }
  return { style, notes, halves, trainedCot };
}

/**
 * UI params -> /yue2/synth request.
 *
 * The caption/lyrics field list mirrors ACE's and MM3's own reads
 * (translateParams.ts / mapMinimaxParams) on purpose: one "what to generate"
 * text field, read the same way regardless of active backend.
 */
export function mapYue2Params(params: any): Yue2ParamMapping {
  const notes: string[] = [];

  const rawCaption: string =
    params.prompt || params.songDescription || params.caption || params.style || '';
  // NFC normalization: the caption/lyrics text this backend's tokenizer sees
  // must match what the engine-side BPE wrapper expects
  // (docs/plans/yue2/06-engine-port-plan.md §2 — NFC before pre-tokenization).
  const caption = rawCaption.normalize('NFC');
  const styled = yue2StyleForAdapter(caption, params);
  const style = styled.style;
  notes.push(...styled.notes);

  const lyricsRaw: string = params.instrumental ? '' : (params.lyrics || '');
  const lyrics = lyricsRaw.normalize('NFC');

  const cotExplicit = typeof params.yue2Cot === 'string';
  const cotRaw = cotExplicit ? params.yue2Cot : 'full';
  let cot: Yue2SynthRequest['cot'] =
    cotRaw === 'off' || cotRaw === 'melody' || cotRaw === 'full' ? cotRaw : 'full';
  if (cotRaw !== cot) notes.push(`Invalid yue2Cot "${cotRaw}" — falling back to "full"`);

  // doc 19 decision 4 / phase 5: default the request's mode to one the
  // selected adapter actually trained, rather than always "full" — "off" is
  // NOT the fast mode (its default CFG 1.01 doubles the semantic decode, see
  // index.ts's Chain of Thought hint), so silently sending an untrained
  // adapter into "full" is exactly as wrong as silently sending it into
  // "off" would be. `styled.trainedCot` is '' only when no adapter is
  // selected, in which case there is nothing to check against.
  if (styled.trainedCot) {
    const trainedModes = styled.trainedCot.split(',').map(s => s.trim()).filter(Boolean);
    if (trainedModes.length && !trainedModes.includes(cot)) {
      if (!cotExplicit) {
        const fallbackRaw = trainedModes[0];
        const fallback: Yue2SynthRequest['cot'] =
          fallbackRaw === 'off' || fallbackRaw === 'melody' || fallbackRaw === 'full' ? fallbackRaw : cot;
        notes.push(`Chain of Thought defaulted to "${fallback}" — the selected LM adapter trained `
          + `${trainedModes.join('/')}, not "${cot}".`);
        cot = fallback;
      } else {
        notes.push(`Chain of Thought "${cot}" requested, but the selected LM adapter trained only `
          + `${trainedModes.join('/')} — rendering "${cot}" anyway since it was set explicitly.`);
      }
    }
  }

  // Blank text field = engine default (mode-dependent: 1.01 for cot=off,
  // 1.0 otherwise, per protocol.py's own SongRequest.guidance property) —
  // same blank-means-omit convention as MM3's mm3ArSeed field.
  const cfgRaw = String(params.yue2CfgScale ?? '').trim();
  const cfgNum = cfgRaw === '' ? NaN : Number(cfgRaw);
  const cfg_scale = Number.isFinite(cfgNum) && cfgNum > 0 ? cfgNum : undefined;
  if (cfgRaw !== '' && cfg_scale === undefined) {
    notes.push(`Invalid yue2CfgScale "${cfgRaw}" — using the engine default for cot="${cot}"`);
  }

  const odeRaw = Number(params.yue2OdeSteps);
  const ode_steps = Number.isFinite(odeRaw) && odeRaw > 0 ? Math.round(odeRaw) : 32;

  const narSolver = params.yue2NarSolver === 'wasserstein' ? 'md_wasserstein_yue2' : undefined;
  const narScheduler = params.yue2NarScheduler === 'ht_v3' ? 'md_ht_scheduler V3' : undefined;
  const plugin_params: NonNullable<Yue2SynthRequest['plugin_params']> = {};
  const copyNumbers = (plugin: string, fields: Record<string, string>) => {
    for (const [field, key] of Object.entries(fields)) {
      if (params[field] === undefined) continue;
      const value = Number(params[field]);
      if (Number.isFinite(value)) plugin_params[`${plugin}:${key}`] = value;
      else notes.push(`Invalid ${field} — using the plugin default`);
    }
  };
  if (narSolver) {
    copyNumbers(narSolver, {
      yue2WassTau: 'tau', yue2WassSpectral: 'spectral_weight',
      yue2WassRmsWeight: 'rms_weight', yue2WassRmsTarget: 'rms_target',
      yue2WassGate: 'sigma_gate', yue2WassIterations: 'prox_iterations',
      yue2WassLatentRms: 'latent_rms',
    });
    if (typeof params.yue2WassProject === 'boolean') {
      plugin_params[`${narSolver}:orthogonal_proj`] = params.yue2WassProject;
    }
  }
  if (narScheduler) {
    copyNumbers(narScheduler, {
      yue2HtKinetic: 'kinetic_energy', yue2HtDamping: 'damping_friction',
      yue2HtCritical: 'critical_temp', yue2HtIntensity: 'phase_intensity',
      yue2HtWell: 'well_width', yue2HtFloor: 'density_floor',
      yue2HtPoly: 'poly_slope', yue2HtBlend: 'uniform_blend',
      yue2HtSmooth: 'smooth_window', yue2HtDense: 'dense_steps',
      yue2HtShift: 'shift',
    });
    if (typeof params.yue2HtSnr === 'boolean') {
      plugin_params[`${narScheduler}:snr_space`] = params.yue2HtSnr;
    }
  }
  if (params.yue2NarSolver && params.yue2NarSolver !== 'stock' && !narSolver) {
    notes.push(`Unknown YuE2 NAR solver "${params.yue2NarSolver}" — using midpoint`);
  }
  if (params.yue2NarScheduler && params.yue2NarScheduler !== 'stock' && !narScheduler) {
    notes.push(`Unknown YuE2 NAR scheduler "${params.yue2NarScheduler}" — using uniform steps`);
  }
  // Step-level velocity caching for the NAR midpoint solver (see
  // yue2-nar-graph.h). 0/unset = off, every step computed for real -- the
  // UI slider (index.ts's yue2NarCacheRatio extension) clamps to [0, 0.9],
  // this just guards a directly-posted value too.
  // UI slider (index.ts's yue2ComposeRetries extension) clamps to [0, 10].
  const retriesRaw = Number(params.yue2ComposeRetries);
  const semantic_retries = Number.isInteger(retriesRaw) && retriesRaw >= 0 && retriesRaw <= 10 ? retriesRaw : 2;
  const narCacheRaw = Number(params.yue2NarCacheRatio);
  const nar_cache_ratio = Number.isFinite(narCacheRaw) && narCacheRaw > 0
    ? Math.min(narCacheRaw, 0.9)
    : undefined;

  // UI slider (index.ts's yue2NarChunkSeconds extension): 0 = one chunk.
  const chunkRaw = Number(params.yue2NarChunkSeconds);
  const nar_chunk_frames = Number.isFinite(chunkRaw) && chunkRaw > 0 ? Math.round(Math.min(chunkRaw, 600) * 25) : undefined;

  const vaeRaw = typeof params.yue2VaeVariant === 'string' ? params.yue2VaeVariant : 'standard';
  const vae_variant: Yue2SynthRequest['vae_variant'] = vaeRaw === 'legacy' ? 'legacy' : 'standard';

  // ── Duration: model-ended, same story as MM3 ──────────────────────────────
  // core.duration.auto = true / editable = false: the AR plan/semantic stages
  // end on their own terminator (or the frame cap), so a requested length is
  // never sent — there is no wire slot for it (06-engine-port-plan.md §7's
  // request field list has no `duration`/`max_frames`).
  const requestedDuration = Number(params.duration);
  if (requestedDuration > 0) {
    notes.push(
      `duration: model-ended — the requested ${Math.round(requestedDuration)}s was ignored `
      + `(YuE2 has no length field on the wire; the plan/semantic stages end on their own `
      + `terminator, capped at roughly ${YUE2_MAX_DURATION_SEC}s).`,
    );
  }

  // -1 tells the engine to draw one; it echoes the resolved value back so the
  // render stays reproducible — same convention mapMinimaxParams uses.
  const seed: number = params.randomSeed
    ? -1
    : (typeof params.seed === 'number' && params.seed >= 0 ? params.seed : -1);

  // A previewed-and-approved lead sheet (score preview flow): the engine
  // renders this score instead of planning one. Meaningless under cot=off,
  // which has no plan stage — dropped with a note rather than sent.
  const abcRaw = typeof params.yue2Abc === 'string' ? params.yue2Abc.trim() : '';
  const abc = abcRaw && cot !== 'off' ? abcRaw : undefined;
  if (abcRaw && !abc) notes.push('A previewed score was supplied but Chain of Thought is "off" — the score was ignored.');

  // Batching (docs/plans/yue2/30-upstream-backports.md #6): the shared Batch
  // Size control is songs (own plan, own seed); yue2Variations is NAR noise
  // variations per song. Both omitted at 1 so a plain render's wire request
  // is unchanged. The engine's caps come from /yue2/props via the manifest;
  // clamp here too so a stale UI cannot 400 the request.
  const propsNow = yue2PropsCached();
  const maxSongs = Math.max(1, Number(propsNow?.max_lm_batch) || 1);
  const maxVars = Math.max(1, Number(propsNow?.max_synth_batch) || 1);
  // yue2BatchSize is the YuE2 panel's own control; batchSize is the ACE
  // Generation dropdown's, kept as a fallback for API callers.
  const askedSongs = Math.max(1, Math.round(Number(params.yue2BatchSize ?? params.batchSize) || 1));
  const askedVars = Math.max(1, Math.round(Number(params.yue2Variations) || 1));
  const lm_batch_size = Math.min(askedSongs, maxSongs);
  const synth_batch_size = Math.min(askedVars, maxVars);
  if (askedSongs > maxSongs) notes.push(`batchSize ${askedSongs} requested — this engine renders at most ${maxSongs} song(s) per job`);
  if (askedVars > maxVars) notes.push(`yue2Variations ${askedVars} requested — this engine renders at most ${maxVars} variation(s) per song`);
  const noiseSeedRaw = Number(params.yue2NoiseSeed);
  const noise_seed = Number.isFinite(noiseSeedRaw) && noiseSeedRaw >= 0 ? Math.floor(noiseSeedRaw) : undefined;

  // LM tab: forward only what the UI actually sent (an untouched control is
  // absent), so the engine keeps its checkpoint value otherwise.
  const lmFields: Array<[string, keyof Yue2SynthRequest]> = [
    ['yue2PlanTemperature', 'plan_temperature'], ['yue2PlanTopP', 'plan_top_p'], ['yue2PlanTopK', 'plan_top_k'],
    ['yue2PlanRepPenalty', 'plan_repetition_penalty'], ['yue2PlanRepWindow', 'plan_penalty_window'],
    ['yue2PlanMaxTokens', 'plan_max_tokens'],
    ['yue2SemTemperature', 'semantic_temperature'], ['yue2SemTopP', 'semantic_top_p'], ['yue2SemTopK', 'semantic_top_k'],
    ['yue2SemRepPenalty', 'semantic_repetition_penalty'], ['yue2SemRepWindow', 'semantic_penalty_window'],
    ['yue2SemMinTokens', 'semantic_min_tokens'], ['yue2SemMaxTokens', 'semantic_max_tokens'],
    ['yue2EndThreshold', 'end_threshold'], ['yue2EndBias', 'end_bias'],
    ['yue2EndBiasFrom', 'end_bias_from_sec'], ['yue2EndBiasRamp', 'end_bias_ramp_sec'],
  ];
  const lmOverrides: Record<string, number> = {};
  for (const [uiKey, wireKey] of lmFields) {
    if (params[uiKey] === undefined || params[uiKey] === null || params[uiKey] === '') continue;
    const v = Number(params[uiKey]);
    if (Number.isFinite(v)) lmOverrides[wireKey] = v;
    else notes.push(`Invalid ${uiKey} — using the checkpoint default`);
  }
  if (Object.keys(lmOverrides).length) notes.push(`LM overrides: ${Object.entries(lmOverrides).map(([k, v]) => `${k}=${v}`).join(', ')}`);

  const req: Yue2SynthRequest = {
    style,
    lyrics: lyrics || undefined,
    cot,
    ...(lmOverrides as Partial<Yue2SynthRequest>),
    ...(lm_batch_size > 1 ? { lm_batch_size } : {}),
    ...(synth_batch_size > 1 ? { synth_batch_size } : {}),
    ...(noise_seed !== undefined ? { noise_seed } : {}),
    ...(abc ? { abc } : {}),
    ...(cfg_scale !== undefined ? { cfg_scale } : {}),
    ode_steps,
    ...(nar_cache_ratio !== undefined ? { nar_cache_ratio } : {}),
    ...(nar_chunk_frames !== undefined ? { nar_chunk_frames } : {}),
    ...(semantic_retries > 0 ? { semantic_retries } : {}),
    ode_method: 'midpoint',
    ...(narSolver ? { infer_method: narSolver } : {}),
    ...(narScheduler ? { scheduler: narScheduler } : {}),
    ...(Object.keys(plugin_params).length ? { plugin_params } : {}),
    vae_variant,
    ...(seed >= 0 ? { seed } : {}),
  };

  return { req, notes, caption, halves: styled.halves };
}

/** First non-empty line of the caption, for StableStep's own SA3 prompt —
 *  same shape as MM3's mm3CaptionToSa3Style but without depending on that
 *  sibling backend module (YuE2's caption has no Structured-Caption headings
 *  to strip in the first place). */
function yue2CaptionToStyle(caption: string): string {
  return caption.split('\n').map(s => s.trim()).find(s => s.length > 0) || '';
}

function yue2StageText(phase: string | undefined, step: number, total: number): { stage: string; progress: number } {
  // KEYED ON THE WIRE STRINGS job_phase_str() emits (hot-step-server.cpp:327),
  // not on the Yue2Stage enum names. `vae` was the enum's spelling and never
  // matched anything: the VAE stage arrives as `vae_decode`, so the whole
  // decode showed as "YuE2: Working". Both spellings are accepted now.
  //
  // "Decoding audio (VAE)" is load-bearing text, not a label — pollUntilDone
  // matches it with startsWith() to grant the decode its 15-minute quiet window
  // instead of the 2-minute one. Do not reword that entry.
  //
  // The other three are worded for what is HAPPENING rather than for which
  // model is running. `semantic` is the AR writing the song's tokens and is the
  // longest stage of a render by far; calling it "Planning (semantic)" was
  // accurate and read as still-getting-ready, so a whole generation looked
  // stuck in planning with no rendering stage ever named.
  const names: Record<string, string> = {
    plan: 'YuE2: Planning (ABC)',
    semantic: 'YuE2: Composing',
    nar: 'YuE2: Rendering',
    vae_decode: 'Decoding audio (VAE)',
    vae: 'Decoding audio (VAE)',
  };
  const label = (phase && names[phase]) || 'YuE2: Working';
  // ": Step N/M" is the ticking-stage pattern pollUntilDone's stall watchdog
  // recognizes (services/generation/pollUntilDone.ts) — matching it here
  // gets the tight 2 min stall window on stages that are really progressing,
  // and "Decoding audio (VAE)" gets its own generous 15 min window the same
  // way ACE's VAE tiling does.
  const stage = total > 0 ? `${label}: Step ${step}/${total}` : `${label}...`;
  // Coarse cross-stage progress estimate — four stages, evenly weighted; no
  // richer per-stage detail is available without a /yue2/job route, which
  // the plan explicitly says not to add.
  const order = ['plan', 'semantic', 'nar', 'vae_decode'];
  const idx = phase ? order.indexOf(phase) : -1;
  const base = idx >= 0 ? idx / order.length : 0;
  const within = total > 0 ? (step / total) / order.length : 0;
  const progress = Math.min(99, Math.round((base + within) * 100));
  return { stage, progress };
}

/** Split a `multipart/mixed` body into its raw parts (the engine's batch
 *  result: one WAV per part, no per-part headers worth reading). */
export function splitMultipartMixed(body: Buffer, contentType: string): Buffer[] {
  const m = /boundary=([^;]+)/.exec(contentType);
  if (!m) return [body];
  const boundary = Buffer.from(`--${m[1].trim()}`);
  const parts: Buffer[] = [];
  let pos = body.indexOf(boundary);
  while (pos !== -1) {
    const lineEnd = pos + boundary.length;
    if (body[lineEnd] === 0x2d && body[lineEnd + 1] === 0x2d) break;  // closing "--boundary--"
    const headerEnd = body.indexOf('\r\n\r\n', lineEnd);
    if (headerEnd === -1) break;
    const dataStart = headerEnd + 4;
    const next = body.indexOf(boundary, dataStart);
    if (next === -1) break;
    // Each part's data is followed by "\r\n" before the next boundary line.
    parts.push(body.subarray(dataStart, next - 2));
    pos = next;
  }
  return parts;
}

type Yue2Log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => void;
type Yue2AutoReplan = { attempts: Array<{ seed: number; verdict: string; reason: string }>; accepted: boolean };

/** A job mapped, logged and (when on) auto-replanned: everything that happens
 *  before its request goes to the engine. Several of these can share one
 *  engine call (see runYue2Generation). */
interface Yue2PreparedJob {
  job: GenerationJob;
  req: Yue2SynthRequest;
  caption: string;
  halves: Yue2StyleHalves;
  autoReplan?: Yue2AutoReplan;
  timing: StageTiming[];
  pipelineStart: number;
  log: Yue2Log;
  /** Set on the lead only; a follower's attempt is filled in when its own
   *  lane turn returns the already-finished job (index.ts generate()). */
  attempt?: GenerationAttempt;
}

async function prepareYue2Job(job: GenerationJob, attempt?: GenerationAttempt): Promise<Yue2PreparedJob> {
  const pipelineStart = performance.now();
  const timing: StageTiming[] = [];
  const log: Yue2Log = (level, msg) => logGeneration(job.id, level, msg);

  const { req, notes, caption, halves } = mapYue2Params(job.params);

  startGenerationLog(job.id, 'yue2-text2music');
  logGenerationParams(job.id, req as unknown as Record<string, unknown>);
  for (const n of notes) log('WARNING', `[YuE2] ${n}`);

  console.log(`[Generate] Job ${job.id} — backend=yue2, cot=${req.cot}, seed=${req.seed ?? -1}, `
    + `caption=${req.style.length} chars, lyrics=${req.lyrics ? `${req.lyrics.length} chars` : '(instrumental)'}`);

  // THE PROMPT THAT ACTUALLY WENT OUT, in full, next to the trigger each merged
  // half was trained on. A style that misses its adapter's trigger costs a whole
  // listening round to notice by ear and is obvious here in one line, which is
  // the entire reason this is logged rather than counted in characters.
  const half = (h: { path: string; trigger: string }) =>
    (h.path ? `${path.basename(h.path)} trigger="${h.trigger || '(none recorded)'}"` : '(none)');
  log('INFO', `[YuE2] AR  ${half(halves.ar)}`);
  log('INFO', `[YuE2] NAR ${half(halves.nar)}`);
  if (halves.ar.path && halves.nar.path && halves.ar.trigger !== halves.nar.trigger) {
    log('WARNING', '[YuE2] The two halves were trained on DIFFERENT triggers — one style sentence '
      + 'carries one trigger, so the half whose trigger is absent is being prompted with a string it '
      + 'never saw.');
  }
  log('INFO', `[YuE2] Style sent: ${req.style}`);
  if (req.lyrics) log('DEBUG', `[YuE2] Lyrics sent (${req.lyrics.length} chars):\n${req.lyrics}`);

  if (!req.style.trim()) {
    throw new Error('YuE2 needs a caption — the Style Description field is empty');
  }

  // ── Auto-replan ──
  // A runaway plan (normal sections, then an outro that never ends) is
  // seed-dependent and costs a six-minute render to discover. Plan first
  // (seconds), classify, redraw the seed on a runaway verdict, then render
  // exactly the approved score. Skipped when the user already approved a
  // score in the preview modal, under cot=off (no plan stage), or when the
  // toggle is off.
  let autoReplan: Yue2AutoReplan | undefined;
  if (job.params.yue2AutoReplan !== false && req.cot !== 'off' && !req.abc) {
    autoReplan = { attempts: [], accepted: false };
    // UI slider (index.ts's yue2ReplanAttempts extension) clamps to [1, 10].
    const attemptsRaw = Number(job.params.yue2ReplanAttempts);
    const maxAttempts = Number.isInteger(attemptsRaw) && attemptsRaw >= 1 && attemptsRaw <= 10 ? attemptsRaw : 3;
    let chosen: { abc: string; seed: number } | undefined;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      if ((job.status as string) === 'cancelled') break;
      job.status = 'running';
      job.stage = attempt === 1 ? 'YuE2: planning the score...' : `YuE2: re-planning (attempt ${attempt} of ${maxAttempts})...`;
      job.progress = 1;
      let plan: Awaited<ReturnType<typeof runYue2PlanPreview>>;
      try {
        plan = await runYue2PlanPreview(attempt === 1 ? job.params : { ...job.params, randomSeed: true, seed: -1 });
      } catch (err: any) {
        log('WARNING', `[YuE2] Auto-replan attempt ${attempt} failed (${err?.message || err}); rendering with the engine's own plan`);
        autoReplan = undefined;
        break;
      }
      autoReplan.attempts.push({ seed: plan.seed, verdict: plan.health.verdict, reason: plan.health.reason });
      log('INFO', `[YuE2] Plan attempt ${attempt}: seed ${plan.seed}, ${plan.health.verdict} — ${plan.health.reason}`);
      chosen = { abc: plan.abc, seed: plan.seed };
      if (yue2PlanUsable(plan.health.verdict, job.params.instrumental === true || !req.lyrics)) { autoReplan.accepted = true; break; }
    }
    if (autoReplan && chosen) {
      req.abc = chosen.abc;
      req.seed = chosen.seed;
      if (!autoReplan.accepted) log('WARNING', `[YuE2] Every plan attempt was a runaway, had no vocal line or ran to its cap; rendering the last one (seed ${chosen.seed})`);
      else if (autoReplan.attempts.length > 1) log('INFO', `[YuE2] Bad plan replaced after ${autoReplan.attempts.length} attempts`);
    }
  }

  return { job, req, caption, halves, autoReplan, timing, pipelineStart, log, attempt };
}

/** Everything about a request except the song itself. Two queued jobs whose
 *  keys match can share one AR batch: same cot, guidance, samplers, NAR
 *  settings and adapters, differing only in prompt, score and seeds. */
function yue2CoalesceKey(job: GenerationJob, req: Yue2SynthRequest): string {
  const shared: Record<string, unknown> = { ...req };
  for (const k of ['style', 'lyrics', 'abc', 'seed', 'noise_seed', 'lm_batch_size', 'songs']) delete shared[k];
  const ordered = Object.fromEntries(Object.keys(shared).sort().map(k => [k, shared[k]]));
  return JSON.stringify([job.userId, job.envelope.models, ordered]);
}

/** The "songs" entries one prepared job contributes: its lm_batch_size takes
 *  of the same prompt, seeded seed + i exactly as the plain path would. */
function yue2SongEntries(p: Yue2PreparedJob): NonNullable<Yue2SynthRequest['songs']> {
  const n = Math.max(1, p.req.lm_batch_size ?? 1);
  return Array.from({ length: n }, (_, i) => ({
    style: p.req.style,
    ...(p.req.lyrics ? { lyrics: p.req.lyrics } : {}),
    ...(p.req.abc ? { abc: p.req.abc } : {}),
    ...(typeof p.req.seed === 'number' ? { seed: p.req.seed + i } : {}),
    ...(typeof p.req.noise_seed === 'number' ? { noise_seed: p.req.noise_seed + i } : {}),
  }));
}

function yue2TracksPerJob(p: Yue2PreparedJob, req: Yue2SynthRequest): number {
  return Math.max(1, p.req.lm_batch_size ?? 1) * Math.max(1, req.synth_batch_size ?? 1);
}

function failYue2Job(job: GenerationJob, err: any): void {
  // `as string`: POST /api/generate/cancel/:id mutates job.status from
  // outside this function, which TS's control-flow narrowing can't see.
  if (err?.message === 'Cancelled' || (job.status as string) === 'cancelled') {
    job.status = 'cancelled';
    job.stage = 'Cancelled';
    failGenerationLog(job.id, 'Cancelled by user', 'yue2-text2music');
  } else {
    job.status = 'failed';
    job.error = err?.message || 'Unknown error';
    job.stage = 'Failed';
    console.error(`[Generate] Job ${job.id} (yue2) failed:`, err?.message);
    failGenerationLog(job.id, err?.message || 'Unknown error', 'yue2-text2music');
  }
}

/** Queue coalescing. The job holding the GPU lane pulls compatible YuE2 jobs
 *  still waiting behind it into the same engine call, one "songs" entry each,
 *  so the AR stages decode every song in one batch. Throughput, not latency:
 *  each song finishes when the batch does. A follower's own lane turn comes
 *  later and finds it already finished (index.ts generate() returns the job
 *  as-is when `coalescedInto` is set). A batch that fails or is cancelled
 *  hands its followers back to the queue untouched, so they render alone. */
export async function runYue2Generation(job: GenerationJob, deps: Yue2GenerationDeps): Promise<void> {
  if (job.coalescedInto) return;   // rendered inside another job's batch
  if (job.status === 'cancelled') return;

  let lead: Yue2PreparedJob;
  try {
    lead = await prepareYue2Job(job, deps.attempt);
  } catch (err: any) {
    failYue2Job(job, err);
    return;
  }
  const { log } = lead;
  const timeoutMinutes: number | undefined = job.params.generationTimeoutMinutes;

  // ── Coalesce ──
  const members: Yue2PreparedJob[] = [lead];
  // The ceiling comes from the engine manifest; after a server restart nothing
  // may have fetched it yet, and a missing manifest must not read as "1".
  const props = yue2PropsCached() ?? (await yue2Props()).props;
  const maxSongs = Math.max(1, Number(props?.max_lm_batch) || 1);
  let songsSoFar = Math.max(1, lead.req.lm_batch_size ?? 1);
  if (job.params.yue2Coalesce !== false && deps.pendingJobs && songsSoFar < maxSongs && (job.status as string) !== 'cancelled') {
    const key = yue2CoalesceKey(job, lead.req);
    // Songs queued together arrive milliseconds apart, and with auto-replan
    // off this job reaches here before its siblings have been posted. A short
    // wait when nothing is queued yet costs nothing against a render.
    if (deps.pendingJobs().length === 0) await new Promise(r => setTimeout(r, YUE2_COALESCE_WAIT_MS));
    for (const cand of deps.pendingJobs()) {
      if (songsSoFar >= maxSongs) break;
      if (cand.status !== 'pending' || cand.coalescedInto || cand.params?.yue2Coalesce === false) continue;
      let mapped: Yue2ParamMapping;
      try { mapped = mapYue2Params(cand.params); } catch { continue; }
      if (yue2CoalesceKey(cand, mapped.req) !== key) continue;
      const need = Math.max(1, mapped.req.lm_batch_size ?? 1);
      if (songsSoFar + need > maxSongs) continue;
      cand.coalescedInto = job.id;
      cand.status = 'running';
      cand.stage = 'YuE2: joining a batch...';
      cand.progress = 1;
      try {
        const prepared = await prepareYue2Job(cand);
        if ((cand.status as string) === 'cancelled') { failYue2Job(cand, new Error('Cancelled')); continue; }
        prepared.log('INFO', `[YuE2] Rendering in one batch with job ${job.id}`);
        members.push(prepared);
        songsSoFar += need;
      } catch (err: any) {
        failYue2Job(cand, err);
      }
    }
  }
  const releaseFollowers = (reason: string) => {
    for (const m of members.slice(1)) {
      if ((m.job.status as string) === 'cancelled') { failYue2Job(m.job, new Error('Cancelled')); continue; }
      m.log('WARNING', `[YuE2] ${reason}; this song goes back to the queue to render on its own`);
      m.job.coalescedInto = undefined;
      m.job.status = 'pending';
      m.job.stage = 'Queued';
      m.job.progress = 0;
    }
    members.length = 1;
  };

  // ── Submit ──
  const req: Yue2SynthRequest = members.length === 1 ? lead.req : (() => {
    const shared: Yue2SynthRequest = { ...lead.req };
    delete shared.lyrics; delete shared.abc; delete shared.seed; delete shared.noise_seed; delete shared.lm_batch_size;
    return { ...shared, songs: members.flatMap(yue2SongEntries) };
  })();
  if (members.length > 1) {
    job.coalescedMembers = members.map(m => m.job.id);
    log('INFO', `[YuE2] Batch of ${members.length} queued jobs (${songsSoFar} songs): ${members.map(m => m.job.id).join(', ')}`);
    console.log(`[Generate] Job ${job.id} — YuE2 batch of ${members.length} jobs, ${songsSoFar} songs`);
  }

  let detailTimer: NodeJS.Timeout | undefined;
  let laneReleased = false;
  const setAll = (fn: (j: GenerationJob) => void) => { for (const m of members) if ((m.job.status as string) !== 'cancelled') fn(m.job); };
  let sub: Awaited<ReturnType<typeof yue2Synth>>;
  let finalDetail: Yue2FinalDetail;
  let parts: Buffer[];
  let trackDetails: Yue2TrackDetail[];
  try {
    setAll(j => { j.status = 'running'; j.stage = 'YuE2: submitting...'; j.progress = 2; });
    const submitStart = performance.now();
    sub = await yue2Synth(req);
    // /yue2/synth answers with the job id and nothing else (#177). The seed and
    // the instrumental flag come from what we sent; a random seed (-1) is only
    // known once the job reports its tracks, and is recorded then.
    const reqSeed = lead.req.seed ?? -1;
    job.aceJobId = sub.job_id;   // standard /job id — /cancel/:id reaches it unchanged
    if (reqSeed >= 0) {
      deps.attempt.effective.seed = reqSeed;
      job.params.seed = reqSeed;
      job.params.randomSeed = false;
    }
    log('INFO', `[YuE2] Job ${sub.job_id} submitted — cot=${req.cot}, ode_steps=${req.ode_steps}, `
      + `cfg=${req.cfg_scale ?? '(default)'}, vae=${req.vae_variant}, `
      + `seed ${reqSeed >= 0 ? reqSeed : 'random'}${lead.req.lyrics ? '' : ', instrumental'}`);

    // ── Progress ticker ──
    // No /yue2/job route exists (docs/plans/yue2/06-engine-port-plan.md §7
    // deliberately does not add one) — poll the SAME shared GET /job status
    // every ACE/MM3 job already uses, and translate its generic
    // phase/phase_step/phase_total into YuE2's own stage names.
    detailTimer = setInterval(() => {
      void (async () => {
        if (job.status === 'cancelled') return;
        try {
          const status = await aceClient.pollJob(sub.job_id);
          const phase = status.phase as string | undefined;
          const step = status.phase_step ?? 0;
          const total = status.phase_total ?? 0;
          const { stage, progress } = yue2StageText(phase, step, total);
          setAll(j => {
            j.stage = stage;
            j.progress = progress;
            j.acePhase = phase;
            if (total > 0) j.acePhaseProgress = `step ${step}/${total}`;
          });
          // Composed. The engine renders this song on its own NAR lane
          // (yue2-job.h), so the next YuE2 job can start composing now; any
          // other family stays behind us, since it would evict our weights.
          // The GPU work left here (post-processing) takes the lane again.
          if (!laneReleased && deps.releaseLane && (phase === 'nar' || phase === 'vae_decode')
              && deps.nextLaneFamily?.() === 'yue2') {
            laneReleased = deps.releaseLane();
            if (laneReleased) log('INFO', '[YuE2] Composed; the GPU lane goes to the next YuE2 job while this one renders');
          }
        } catch { /* transient poll failure — pollUntilDone owns the real watchdog */ }
      })();
    }, DETAIL_POLL_MS);
    detailTimer.unref?.();

    await deps.pollUntilDone(sub.job_id, job, deps.signal, timeoutMinutes);
    clearInterval(detailTimer);
    detailTimer = undefined;
    const generateMs = Math.round(performance.now() - submitStart);
    for (const m of members) m.timing.push({ name: 'YuE2 Generate', ms: generateMs });

    setAll(j => { j.stage = 'YuE2: saving audio...'; j.progress = 95; });

    // Additive end_reason/stage_end_reasons/artifacts — see the ASSUMPTION
    // note in client.ts. Every field here is optional; an engine build that
    // doesn't populate them yet just yields an empty object.
    finalDetail = await yue2FinalDetail(sub.job_id);

    const audioRes = await aceClient.getJobResult(sub.job_id);
    const audioBuffer = Buffer.from(await audioRes.arrayBuffer());
    if (audioBuffer.length === 0) {
      throw new Error('YuE2 returned an empty audio body');
    }
    const contentType = audioRes.headers.get('content-type') || 'audio/wav';
    // One WAV, or multipart/mixed with one WAV part per track, song-major
    // (yue2-job.h) — the same shape ACE's batch path emits.
    parts = contentType.startsWith('multipart/mixed')
      ? splitMultipartMixed(audioBuffer, contentType)
      : [audioBuffer];
    if (parts.length === 0) throw new Error('YuE2 returned a multipart body with no parts');
    const perSong = Math.max(1, req.synth_batch_size ?? 1);
    trackDetails = finalDetail.tracks && finalDetail.tracks.length === parts.length
      ? finalDetail.tracks
      : parts.map((_, i) => ({ song: Math.floor(i / perSong), variation: i % perSong, seed: reqSeed, noise_seed: reqSeed }));
  } catch (err: any) {
    if (detailTimer) clearInterval(detailTimer);
    releaseFollowers(`The batch render failed (${err?.message || err})`);
    failYue2Job(job, err);
    return;
  }

  // ── Finish each member from its own slice of the tracks ──
  // Song-major, so a member's tracks are contiguous: lm_batch_size songs
  // times synth_batch_size variations. Song numbers are renumbered per job.
  const finishOne = async (m: Yue2PreparedJob, offset: number) => {
    const count = yue2TracksPerJob(m, req);
    const slice = parts.slice(offset, offset + count);
    const songBase = trackDetails[offset]?.song ?? 0;
    const details = trackDetails.slice(offset, offset + count).map(td => ({ ...td, song: td.song - songBase }));
    if ((m.job.status as string) === 'cancelled') { failYue2Job(m.job, new Error('Cancelled')); return; }
    if (slice.length !== count) {
      failYue2Job(m.job, new Error(`YuE2 returned ${parts.length} track(s) for a batch that expected ${offset + count}`));
      return;
    }
    // A member's own view of the result: its first track's score and reasons
    // stand where a solo render's job-level fields would.
    const memberDetail: Yue2FinalDetail = members.length === 1 ? finalDetail : {
      ...finalDetail,
      abc: details[0]?.abc,
      end_reason: details.some(td => td.end_reason === 'limit_hit') ? 'limit_hit' : (details[0]?.end_reason ?? finalDetail.end_reason),
      stage_end_reasons: details[0]?.stage_end_reasons,
      semantic_ids: undefined,
      tracks: details,
    };
    try {
      await finishYue2Job(m, slice, details, memberDetail);
    } catch (err: any) {
      failYue2Job(m.job, err);
    }
  };
  const offsets = new Map<Yue2PreparedJob, number>();
  {
    let offset = 0;
    for (const m of members) { offsets.set(m, offset); offset += yue2TracksPerJob(m, req); }
  }
  // Only some of the finishing work touches the GPU: StableStep, Whisper, the
  // forced aligner and cover art. A VST chain, mastering and the normaliser
  // are CPU, so those members finish at once, side by side, and never wait
  // for the lane. Members with GPU work wait their turn and run one at a time.
  const cpuOnly = members.filter(m => !yue2FinishNeedsGpu(m.job.params));
  const gpu = members.filter(m => yue2FinishNeedsGpu(m.job.params));
  await Promise.all(cpuOnly.map(m => finishOne(m, offsets.get(m)!)));
  if (gpu.length) {
    const finishGpu = async () => { for (const m of gpu) await finishOne(m, offsets.get(m)!); };
    if (laneReleased && deps.runOnLane) {
      for (const m of gpu) if ((m.job.status as string) !== 'cancelled') m.job.stage = 'YuE2: waiting for the GPU to finish...';
      await deps.runOnLane(finishGpu);
    } else {
      await finishGpu();
    }
  }
}

/** Does this job's post-render work need the GPU? VST, mastering and the
 *  normaliser are CPU; these four are not. */
function yue2FinishNeedsGpu(params: any): boolean {
  const pp = params.postProcessingEnabled !== false;
  return (pp && !!(params.stableStepOn ?? params.stableStep))
    || !!params.whisperLyricsEnabled || !!params.yue2AlignLyrics || !!params.coverArtEnabled;
}

/** Save, post-process, transcribe, align and persist one job's tracks, then
 *  mark it succeeded: the tail of the old single-job path, unchanged. */
async function finishYue2Job(
  p: Yue2PreparedJob, parts: Buffer[], trackDetails: Yue2TrackDetail[], finalDetail: Yue2FinalDetail,
): Promise<void> {
  const { job, req, caption, halves, autoReplan, timing, pipelineStart, log } = p;
  const instrumental = !req.lyrics;
  const saveStart = performance.now();
  {
    // Healthy-but-long vs runaway: the score says which, when there is one
    // (cot=off renders have no plan stage and no score to read).
    const scoreHealth = finalDetail.abc ? classifyYue2Score(finalDetail.abc, finalDetail.end_reason) : undefined;
    if (scoreHealth) {
      log(scoreHealth.verdict === 'runaway' ? 'WARNING' : 'INFO',
        `[YuE2] Score: ${scoreHealth.verdict} — ${scoreHealth.reason}`);
    }
    if (finalDetail.end_reason === 'limit_hit') {
      log('WARNING', scoreHealth?.verdict === 'long'
        ? '[YuE2] Render hit the frame cap, but the score itself is healthy — the song is longer than '
          + 'the cap, not broken. Shorten the lyric or raise the cap.'
        : '[YuE2] Render hit its frame/token limit before reaching a natural ending '
          + '(end_reason: limit_hit) — the song may cut off rather than resolve.');
    } else if (finalDetail.stage_end_reasons) {
      const eosStages = Object.entries(finalDetail.stage_end_reasons)
        .filter(([, r]) => r === 'eos').map(([s]) => s);
      if (eosStages.length) log('INFO', `[YuE2] Stages reaching their own terminator: ${eosStages.join(', ')}`);
    }
    const resolvedSeed = trackDetails[0]?.seed;
    if (typeof resolvedSeed === 'number' && resolvedSeed >= 0) {
      if (p.attempt) p.attempt.effective.seed = resolvedSeed;
      job.params.seed = resolvedSeed;
      job.params.randomSeed = false;
    }

    const captionLine = yue2CaptionToStyle(caption);
    const title: string = job.params.title || captionLine.substring(0, 60) || 'Untitled';
    const style: string = job.params.caption || job.params.style || '';

    const audioUrls: string[] = [];
    const filepaths: string[] = [];
    const durations: number[] = [];
    for (let i = 0; i < parts.length; i++) {
      const filename = `${uuidv4()}.wav`;
      const filepath = path.join(config.data.audioDir, filename);
      fs.writeFileSync(filepath, parts[i]);
      audioUrls.push(`/audio/${filename}`);
      filepaths.push(filepath);
      const td = trackDetails[i];

      // Artifacts: score.abc (decoded ABC text) and semantic ids, saved beside
      // the WAV as sidecars when the engine returned them. Never fatal.
      const abcText = td.abc ?? (parts.length === 1 ? finalDetail.abc : undefined);
      if (abcText) {
        try {
          fs.writeFileSync(path.join(config.data.audioDir, filename.replace(/\.[^.]+$/, '.score.abc')), abcText);
        } catch (e: any) {
          log('WARNING', `[YuE2] Failed to save score.abc sidecar (non-fatal): ${e?.message ?? e}`);
        }
      }
      if (parts.length === 1 && finalDetail.semantic_ids?.length) {
        try {
          fs.writeFileSync(
            path.join(config.data.audioDir, filename.replace(/\.[^.]+$/, '.semantic.json')),
            JSON.stringify(finalDetail.semantic_ids),
          );
        } catch (e: any) {
          log('WARNING', `[YuE2] Failed to save semantic-ids sidecar (non-fatal): ${e?.message ?? e}`);
        }
      }

      // 48 kHz native (yue2vae.sample_rate) — read from the WAV header rather
      // than assumed, same as every other backend.
      const measured = wavDurationSec(filepath);
      durations.push(measured);
      log('INFO', `[YuE2] Saved ${filename} (${(parts[i].length / 1024).toFixed(0)} KB, ${Math.round(measured)}s)`
        + (parts.length > 1 ? ` — song ${td.song + 1}, variation ${td.variation + 1}, seed ${td.seed}, noise ${td.noise_seed}` : ''));
    }
    timing.push({ name: 'Save', ms: Math.round(performance.now() - saveStart) });

    // ── Post-processing (the model-agnostic chain, in full) ────────────────
    // YuE2 renders native 48 kHz stereo, so unlike MM3's v1 (44.1 kHz,
    // whole-chain skip) nothing here needs a rate audit. ppVaeReencode and the
    // Spectral Lifter stay excluded regardless — both are ACE-VAE-coupled,
    // not rate-coupled.
    let masteredUrls: string[] = [];
    try {
      const ppParams: PostProcessParams = {
        ...job.params,
        ppVaeReencode: false,
        spectralLifterEnabled: false,
        instrumental: instrumental,
        stableStepCaptions: [captionLine],
      };
      if (ppParams.postProcessingEnabled !== false) {
        const ppStart = performance.now();
        const ppResult = await runPostProcessingChain(
          audioUrls, ppParams, audioUrls.length, job.id,
          log, (stage) => { job.stage = stage; },
        );
        masteredUrls = ppResult.masteredUrls ?? [];
        const ppMs = Math.round(performance.now() - ppStart);
        if (masteredUrls.some(Boolean)) log('INFO', `[YuE2] Post-processing produced ${masteredUrls.filter(Boolean).length} file(s) in ${ppMs} ms`);
        timing.push({ name: 'Post-processing', ms: ppMs });
      }
    } catch (ppErr: any) {
      log('WARNING', `[YuE2] Post-processing chain failed (non-fatal): ${ppErr?.message || ppErr}`);
    }

    // ── Whisper transcription ────────────────────────────────────────────
    // Backend-agnostic (whisper-cli takes a file path and resamples
    // internally). YuE2's DiT has no lyric-alignment head
    // (capabilities().features.lyricTimestamps = false), so there is no LRC
    // path the way ACE/MM3 have one — but this is no longer the only route to
    // word timings: the forced aligner below is the accurate one, and it
    // writes the same file. Whisper still earns its place on a render whose
    // lyrics you do not have (a cover, an import) or to hear what was really
    // sung rather than what was asked for.
    if (job.params.whisperLyricsEnabled && !instrumental) {
      const wStart = performance.now();
      try {
        const { ensureWhisperCli, findWhisperModel, transcribeWithWhisper } =
          await import('../../whisperTranscribe.js');
        const { reconcileLyrics } = await import('../../lyricsReconcile.js');

        if (!(await ensureWhisperCli())) {
          log('WARNING', '[Whisper] whisper-cli unavailable and auto-download failed — skipping');
        } else if (!findWhisperModel(job.params.whisperModel)) {
          log('WARNING', '[Whisper] no Whisper model installed — skipping');
        } else {
          for (let i = 0; i < filepaths.length; i++) {
            log('INFO', `[Whisper] transcribing YuE2 render${filepaths.length > 1 ? ` ${i + 1}/${filepaths.length}` : ''}...`);
            const wr = await transcribeWithWhisper(filepaths[i], req.lyrics || '', {
              model: job.params.whisperModel,
              language: job.params.whisperLanguage || 'auto',
              beamSize: job.params.whisperBeamSize || 5,
            });
            if (wr && wr.segments?.length > 0) {
              const lyricsJson = reconcileLyrics(wr, req.lyrics || '', job.params.whisperModel || 'auto', false);
              const lyricsPath = filepaths[i].replace(/\.[^.]+$/, '.lyrics.json');
              fs.writeFileSync(lyricsPath, JSON.stringify(lyricsJson, null, 2));
              const words = lyricsJson.lines.reduce((n: number, l: any) => n + l.words.length, 0);
              log('INFO', `[Whisper] saved ${path.basename(lyricsPath)} `
                + `(${lyricsJson.lines.length} lines, ${words} words)`);
            } else {
              log('WARNING', '[Whisper] no segments returned');
            }
          }
        }
      } catch (wErr: any) {
        log('WARNING', `[Whisper] failed (non-fatal): ${wErr?.message || wErr}`);
      }
      timing.push({ name: 'Whisper', ms: Math.round(performance.now() - wStart) });
    }

    // ── Forced alignment (opt-in) ─────────────────────────────────────────
    // YuE2's answer to "where is each word". It runs LAST, after Whisper, so
    // that with both switched on the better source wins the `.lyrics.json`:
    // this one is scored against the lyrics the user actually supplied and
    // cannot invent a word, where a transcriber can and does.
    //
    // The spans come back as codepoint offsets into the lyrics we sent, so
    // align.ts maps every word to its line and [Section] by lookup — no
    // reconciliation, and no chance of a line landing under the wrong header.
    if (job.params.yue2AlignLyrics && !instrumental && req.lyrics) {
      const alStart = performance.now();
      const prevStage = job.stage;
      try {
        // ONE string, sent and mapped. multipart/form-data normalises a text
        // part's newlines to CRLF, so a `\n` lyric reaches the engine as
        // `\r\n` and every codepoint offset it returns runs one ahead per
        // preceding line. Normalising here (and stripping \r engine-side) means
        // both ends count the same characters. The song row keeps req.lyrics.
        const alignText = req.lyrics.replace(/\r\n?/g, '\n');
        for (let i = 0; i < filepaths.length; i++) {
          job.stage = `Aligning lyrics${filepaths.length > 1 ? ` ${i + 1}/${filepaths.length}` : ''}`;
          const aligned = await yue2Align(fs.readFileSync(filepaths[i]), alignText);
          const lyricsJson = yue2LyricsJson(alignText, aligned.words);
          const lyricsPath = filepaths[i].replace(/\.[^.]+$/, '.lyrics.json');
          fs.writeFileSync(lyricsPath, JSON.stringify(lyricsJson, null, 2));
          const words = lyricsJson.lines.reduce((n, l) => n + l.words.length, 0);
          log('INFO', `[Align] saved ${path.basename(lyricsPath)} `
            + `(${lyricsJson.lines.length} lines, ${words} words, ${aligned.model})`);
        }
      } catch (alErr: any) {
        // Never fatal, for the reason Whisper's own failure is not: the song
        // is rendered and saved, and a missing karaoke bar is not a lost take.
        log('WARNING', `[Align] forced alignment failed (non-fatal): ${alErr?.message || alErr}`);
      }
      job.stage = prevStage;
      timing.push({ name: 'Align', ms: Math.round(performance.now() - alStart) });
    }

    // ── Persist: one song row per track ──
    const songIds: string[] = [];
    for (let i = 0; i < parts.length; i++) {
      const td = trackDetails[i];
      const measured = durations[i];
      const duration = measured > 0 ? Math.round(measured) : 0;
      const trackParams = {
        ...job.params,
        backend: 'yue2',
        seed: td.seed,
        // The AR/NAR pick is engine state rather than a request field, so
        // without this the row records nothing about which adapter sang —
        // and the gp.lmAdapter sitting beside it belongs to ACE-Step's 4B
        // planner, which is what the details panel used to show.
        ...(halves.ar.path || halves.nar.path ? { yue2Adapters: halves } : {}),
        yue2Request: { ...req, seed: td.seed, noise_seed: td.noise_seed, lm_batch_size: undefined, synth_batch_size: undefined },
        yue2: {
          ode_steps: req.ode_steps,
          cfg_scale: req.cfg_scale,
          vae_variant: req.vae_variant,
          instrumental: instrumental,
          end_reason: td.end_reason ?? finalDetail.end_reason,
          stage_end_reasons: td.stage_end_reasons ?? finalDetail.stage_end_reasons,
          duration_s: measured > 0 ? Math.round(measured * 10) / 10 : undefined,
          abc_supplied: !!req.abc,
          ...(parts.length > 1 ? { song: td.song, variation: td.variation, noise_seed: td.noise_seed } : {}),
          ...(scoreHealth ? { score_health: scoreHealth } : {}),
          ...(autoReplan ? { auto_replan: autoReplan } : {}),
        },
      };
      const songId = uuidv4();
      const trackTitle = parts.length > 1 ? `${title} (${td.song + 1}.${td.variation + 1})` : title;
      getDb().prepare(`
        INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                           duration, bpm, key_scale, time_signature, tags, dit_model,
                           generation_params, mastered_audio_url, latent_url, quality_scores,
                           noadapter_audio_url, backend)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        songId, job.userId, trackTitle, req.lyrics || '', style, req.style,
        audioUrls[i], duration, 0, '', '',
        JSON.stringify([]), 'yue2', JSON.stringify(trackParams),
        masteredUrls[i] || '', '', '',
        '', 'yue2',
      );
      songIds.push(songId);
    }

    // ── Cover art (backend-agnostic): one image for the batch, linked to every row ──
    if (job.params.coverArtEnabled) {
      const coverStart = performance.now();
      try {
        const { generateCoverArt, getCoverArtReadiness, linkCoverToSong } = await import('../../coverArt/coverArtService.js');
        const readiness = getCoverArtReadiness();
        if (readiness.installed) {
          job.stage = 'Generating cover art...';
          job.progress = 97;
          const cover: any = await generateCoverArt({
            songId: songIds[0],
            title,
            style: style || captionLine,
            lyrics: req.lyrics || '',
            subject: job.params.coverArtSubject || job.params.subject || '',
          });
          log('INFO', `[CoverArt] Generated cover for song ${songIds[0]}`);
          if (cover?.coverUrl) {
            for (let i = 1; i < songIds.length; i++) linkCoverToSong(cover.coverUrl, songIds[i]);
          }
          if (job.params.coverArtSubject) {
            for (const id of songIds) {
              getDb().prepare('UPDATE songs SET cover_art_subject = ? WHERE id = ?')
                .run(job.params.coverArtSubject, id);
            }
          }
        } else {
          log('DEBUG', `[CoverArt] Skipped — not installed (missing: ${readiness.missingFiles.join(', ')})`);
        }
      } catch (coverErr: any) {
        log('WARNING', `[CoverArt] Failed (non-fatal): ${coverErr.message}`);
      }
      const coverMs = Math.round(performance.now() - coverStart);
      if (coverMs > 50) timing.push({ name: 'Cover Art', ms: coverMs });
    }

    const totalMs = Math.round(performance.now() - pipelineStart);
    timing.push({ name: 'TOTAL', ms: totalMs });

    job.status = 'succeeded';
    job.progress = 100;
    job.stage = 'Complete!';
    job.result = {
      // Raw renders, with the masters index-aligned beside them — the ACE shape.
      // Folding the master into audioUrls left the playbar's unmastered switch
      // pointing at the mastered file.
      audioUrls,
      masteredAudioUrl: masteredUrls.find(u => !!u) || undefined,
      masteredAudioUrls: audioUrls.map((_, i) => masteredUrls[i] || ''),
      songIds,
      duration: durations[0] > 0 ? Math.round(durations[0]) : 0,
      timing,
      totalMs,
    };

    log('INFO', `[Result] ${audioUrls.length} audio file(s) saved, ${songIds.length} song(s) created (backend=yue2)`);
    console.log(`[Generate] Job ${job.id} (yue2) completed in ${(totalMs / 1000).toFixed(1)}s`);
    finishGenerationLog(job.id, 'yue2-text2music');

  }
}

// ── Score preview ──────────────────────────────────────────────────────────
//
// Plan only: the same request mapping the render uses (adapter trigger,
// caption template, cot), sent with `plan_only` so the engine stops after the
// lead sheet. Seconds rather than minutes, and the classifier says whether the
// plan is a song or a runaway before any audio exists. The caller then
// re-submits an ordinary generation with `yue2Abc` set to the approved score
// and the seed pinned to the one echoed here.

export interface Yue2PlanPreview {
  abc: string;
  seed: number;
  end_reason: string;
  stage_end_reasons?: Yue2FinalDetail['stage_end_reasons'];
  health: Yue2ScoreHealth;
  notes: string[];
  /** Only with params.semantic: the planner's raw codec id stream. */
  semantic_ids?: number[];
}

export async function runYue2PlanPreview(params: any, signal?: AbortSignal): Promise<Yue2PlanPreview> {
  const { req, notes } = mapYue2Params(params);
  if (!req.style.trim()) throw new Error('YuE2 needs a caption — the Style Description field is empty');
  if (req.cot === 'off') throw new Error('Score preview needs Chain of Thought "melody" or "full" — cot=off has no lead sheet to preview');
  // params.semantic: run the semantic stage too and return its codec ids
  // (about a minute) instead of stopping at the lead sheet (seconds).
  const semantic = params?.semantic === true;
  const planReq: Yue2SynthRequest = semantic ? { ...req, semantic_only: true } : { ...req, plan_only: true };
  delete planReq.abc;

  const sub = await yue2Synth(planReq);
  const started = Date.now();
  // ponytail: 10-minute ceiling on a stage that takes seconds; the shared
  // pollUntilDone watchdog is built around a GenerationJob this call has none of.
  for (;;) {
    if (signal?.aborted) {
      await aceClient.cancelJob(sub.job_id).catch(() => {});
      throw new Error('Score preview cancelled');
    }
    const status = await aceClient.pollJob(sub.job_id);
    if (status.status === 'done') break;
    if (status.status === 'failed' || status.status === 'cancelled') {
      const detail = (status as { error?: string }).error;
      throw new Error(`YuE2 plan ${status.status}${detail ? `: ${detail}` : ''}`);
    }
    if (Date.now() - started > 10 * 60_000) {
      await aceClient.cancelJob(sub.job_id).catch(() => {});
      throw new Error('YuE2 plan stage timed out after 10 minutes');
    }
    await new Promise(r => setTimeout(r, 750));
  }
  const detail = await yue2FinalDetail(sub.job_id);
  const abc = (detail.abc ?? '').trim();
  if (!abc) throw new Error('YuE2 returned no score for the plan stage');
  const end_reason = detail.end_reason ?? 'completed';
  let semantic_ids: number[] | undefined;
  if (semantic) {
    const body = await aceClient.getJobResult(sub.job_id);
    if (!body.ok) throw new Error(`YuE2 semantic result fetch failed (${body.status})`);
    semantic_ids = await body.json() as number[];
  }
  return { abc, seed: detail.tracks?.[0]?.seed ?? planReq.seed ?? -1, end_reason, stage_end_reasons: detail.stage_end_reasons, health: classifyYue2Score(abc, end_reason), notes, ...(semantic_ids ? { semantic_ids } : {}) };
}
