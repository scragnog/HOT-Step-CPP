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
import { yue2Align, yue2Synth, yue2FinalDetail, yue2Props, yue2PropsCached, splitMultipartMixed, yue2AdapterWire, type Yue2SynthRequest, type Yue2TrackDetail } from './client.js';
import { yue2LyricsJson } from './align.js';
import { buildYue2LyricSchedule } from './lyricSchedule.js';
import { classifyYue2Score, type Yue2ScoreHealth, yue2PlanUsable, yue2PickPlan, yue2PlanDraws, readYue2StyleNorms, type Yue2StyleNorms } from './scoreHealth.js';
import {
  yue2PickFromModels, yue2ResolvePick, yue2StackFrom, yue2CoverFromSubmission,
  type Yue2PersistedSelection, type Yue2CoverSubmission,
} from './index.js';

/** The pick a queued job renders with (#204): captured at submit into
 *  envelope.models, never the live picker. */
function jobPick(job: GenerationJob): Yue2PersistedSelection {
  return yue2PickFromModels(job.envelope.models as Record<string, string>);
}

/** A cover job's provenance and rendered score, captured into
 *  envelope.options.yue2 at submit (index.ts's resolveRequest already
 *  rejected a blank ABC or cot=off there). The only source the render and
 *  its metadata read for these fields — job.params is the raw request body
 *  and is never read for them. undefined for an ordinary (non-cover) job. */
export function jobCover(job: GenerationJob): { cover: Yue2CoverSubmission; abc: string } | undefined {
  const opts = job.envelope.options?.[job.envelope.backendId] as Record<string, unknown> | undefined;
  if (!opts) return undefined;
  const cover = yue2CoverFromSubmission(opts);
  if (!cover) return undefined;
  const abc = typeof opts.yue2Abc === 'string' ? opts.yue2Abc : '';
  return { cover, abc };
}
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
 *  rendering alone (queue coalescing). The browser queue already holds an
 *  album's songs for a few seconds after the last click and posts them in one
 *  wave (audioGenQueueStore.ts), so this only covers the wave's own spread. */
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
 * The adapters are the JOB's pick (#204), captured when it was submitted and
 * sent to the engine with the request, which merges them before this job
 * renders. The live picker only sets the default for jobs submitted later.
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
  caption: string, params: any, picked: Yue2PersistedSelection['adapters'],
): { style: string; notes: string[]; halves: Yue2StyleHalves; trainedCot: string } {
  const notes: string[] = [];
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
  // The adapter's own header is the only source. A file that does not record
  // its CoT modes (joint exports do not yet) is unknown, never guessed as "off".
  const trainedCot = meta?.cot ?? '';
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
/** `pick` is the job's own (jobPick); callers without a job (the score
 *  preview) resolve one from the params exactly as a submit would. */
export function mapYue2Params(params: any, pick: Yue2PersistedSelection = yue2ResolvePick(params)): Yue2ParamMapping {
  const notes: string[] = [];

  const rawCaption: string =
    params.prompt || params.songDescription || params.caption || params.style || '';
  // NFC normalization: the caption/lyrics text this backend's tokenizer sees
  // must match what the engine-side BPE wrapper expects
  // (docs/plans/yue2/06-engine-port-plan.md §2 — NFC before pre-tokenization).
  const caption = rawCaption.normalize('NFC');
  const styled = yue2StyleForAdapter(caption, params, pick.adapters);
  const style = styled.style;
  notes.push(...styled.notes);

  const lyricsRaw: string = params.instrumental ? '' : (params.lyrics || '');
  const lyrics = lyricsRaw.normalize('NFC');

  const cotRaw = typeof params.yue2Cot === 'string' ? params.yue2Cot : 'full';
  const cot: Yue2SynthRequest['cot'] =
    cotRaw === 'off' || cotRaw === 'melody' || cotRaw === 'full' ? cotRaw : 'full';
  if (cotRaw !== cot) notes.push(`Invalid yue2Cot "${cotRaw}" — falling back to "full"`);

  // The mode shown in the picker is the mode rendered: an unset yue2Cot is the
  // picker's "full" default, never rewritten to suit the adapter. A mismatch
  // with what the adapter's header says it trained is only reported.
  if (styled.trainedCot) {
    const trainedModes = styled.trainedCot.split(',').map(s => s.trim()).filter(Boolean);
    if (trainedModes.length && !trainedModes.includes(cot)) {
      notes.push(`Chain of Thought "${cot}" requested, but the selected LM adapter trained only `
        + `${trainedModes.join('/')} — rendering "${cot}" anyway.`);
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

  // From the job's pick (#204), which already folded in the per-request
  // yue2VaeVariant select at submit. '' (never chosen) renders standard.
  const vae_variant: Yue2SynthRequest['vae_variant'] = pick.vae_variant === 'legacy' ? 'legacy' : 'standard';

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

  // C6 lyric schedule: ties each sung lyric block to its score section's
  // timing. Soft (bias, -4) is the default for a render with an approved
  // score — an ear test went from 2/11 on-time sections to 11/11 with it
  // (RESEARCH/YUE2_ALIGNMENT_GATE_SET.md:1064). Hard (mask) stays selectable
  // but is never the default. An explicit 'off' (current or saved) always
  // wins. A schedule that cannot be built (no tempo, unlabelled sections, no
  // matching timed lyric block, ...) must never fail the render — it logs
  // why and falls back to the unscheduled baseline instead.
  const scheduleMode = params.yue2LyricSchedule === 'off' ? undefined
    : params.yue2LyricSchedule === 'mask' ? 'mask'
    : params.yue2LyricSchedule === 'bias' || params.yue2LyricSchedule === undefined ? 'bias'
    : undefined;
  // The engine only schedules the no-guidance set (yue2-pipeline.h: "needs
  // cfg_scale 1 (no rule for the guidance set)"); cot is never 'off' here
  // (abc requires it), so the engine's own default CFG is 1.0 unless
  // overridden above.
  const effectiveCfg = cfg_scale ?? 1.0;
  let lyric_schedule: Yue2SynthRequest['lyric_schedule'];
  if (scheduleMode) {
    if (!abc || !lyrics) {
      notes.push(`Lyric schedule (${scheduleMode}) needs a supplied score and lyrics — rendering without it.`);
    } else if (effectiveCfg !== 1) {
      notes.push(`Lyric schedule (${scheduleMode}) skipped — CFG ${effectiveCfg} is not 1 `
        + '(the engine only schedules the no-guidance set); rendering without it at the requested CFG.');
    } else {
      const num = (value: unknown, fallback: number) =>
        value === undefined || value === null || value === '' ? fallback : Number(value);
      try {
        const built = buildYue2LyricSchedule(abc, lyrics, {
          mode: scheduleMode, bias: num(params.yue2LyricScheduleBias, -4),
          abc: params.yue2LyricScheduleAbc === true, leadSec: num(params.yue2LyricScheduleLeadSec, 0),
          behind: num(params.yue2LyricScheduleBehind, -1),
        });
        lyric_schedule = built.wire;
        notes.push(`Lyric schedule: ${scheduleMode}${scheduleMode === 'bias' ? ` ${built.wire.bias}` : ''}, `
          + `${built.wire.sections.length} timed section(s)${built.wire.abc ? ', score lines too' : ''}`
          + (built.untimed.length ? `; never hidden (no timed section): ${built.untimed.join(', ')}` : ''));
      } catch (err: any) {
        notes.push(`Lyric schedule (${scheduleMode}) not built — rendering without it: ${err?.message || err}`);
      }
    }
  }

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
    // The pick travels with the request; the engine applies it when the job
    // starts. null clears: no adapter in the pick means the base model.
    lm_type: pick.lm,
    lm_adapter: yue2AdapterWire(yue2StackFrom(pick.adapters)),
    style,
    lyrics: lyrics || undefined,
    cot,
    ...(lmOverrides as Partial<Yue2SynthRequest>),
    ...(lm_batch_size > 1 ? { lm_batch_size } : {}),
    ...(synth_batch_size > 1 ? { synth_batch_size } : {}),
    ...(noise_seed !== undefined ? { noise_seed } : {}),
    ...(abc ? { abc } : {}),
    ...(lyric_schedule ? { lyric_schedule } : {}),
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


type Yue2Log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => void;
type Yue2AutoReplan = { attempts: Array<{ seed: number; verdict: string; reason: string; flags?: string[] }>; accepted: boolean;
  /** The rendered plan passed the judge with no legibility flags (only set
   *  when flag re-plans are on). false = the least-flagged was rendered. */
  clean?: boolean };

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
  /** Cover provenance captured into the envelope at submit, undefined for an
   *  ordinary job. */
  cover?: Yue2CoverSubmission;
}

async function prepareYue2Job(job: GenerationJob, attempt?: GenerationAttempt): Promise<Yue2PreparedJob> {
  const pipelineStart = performance.now();
  const timing: StageTiming[] = [];
  const log: Yue2Log = (level, msg) => logGeneration(job.id, level, msg);

  // A cover job's ABC and identity come from the envelope captured at
  // submit, never from job.params (the raw request body) — index.ts's
  // resolveRequest already validated it there.
  const captured = jobCover(job);
  const params = captured ? { ...job.params, yue2Abc: captured.abc,
    ...(captured.cover.cfgScale !== undefined ? { yue2CfgScale: captured.cover.cfgScale } : {}) } : job.params;
  const { req, notes, caption, halves } = mapYue2Params(params, jobPick(job));

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

  return { job, req, caption, halves, timing, pipelineStart, log, attempt, cover: captured?.cover };
}

// ── Auto-replan ──
// A runaway plan (normal sections, then an outro that never ends) is
// seed-dependent and costs a six-minute render to discover. Plan first
// (seconds), classify, redraw the seed on a runaway verdict, then render
// exactly the approved score. Skipped when the user already approved a
// score in the preview modal, under cot=off (no plan stage), or when the
// toggle is off.
//
// Every member of a batch plans together, and each draws several seeds per
// pass: the plan stage decodes its songs in lockstep, so eight scores cost
// about what one does (2026-09-28; one at a time, ten tries of a weak
// adapter took three minutes before the batch could start composing).
/** The style norms of the selected planner adapter, read from its own header
 *  (style_norms, written by the joint trainer), or null for the base model or
 *  an adapter with none. Checks the training sheets themselves fail are not
 *  held against its plans. */
function activeStyleNorms(pick: Yue2PersistedSelection): Yue2StyleNorms | null {
  const { ar, nar } = pick.adapters;
  const adapter = ar.path || nar.path;
  if (!adapter) return null;
  const fromHeader = readSafetensorsMeta(adapter)?.raw.style_norms;
  if (fromHeader) {
    try { const j = JSON.parse(fromHeader) as Yue2StyleNorms; if (Array.isArray(j.exempt)) return j; } catch { /* fall through */ }
  }
  // Adapters exported before the header carried them: the run folder's copy.
  return readYue2StyleNorms(jointRunForAdapter(adapter)?.output);
}

async function planYue2Jobs(members: Yue2PreparedJob[]): Promise<void> {
  type Pending = {
    m: Yue2PreparedJob; maxAttempts: number; replanFlags: boolean; instrumental: boolean;
    sheets: string[]; done: boolean;
  };
  const pending: Pending[] = [];
  for (const m of members) {
    const { job, req } = m;
    if (job.params.yue2AutoReplan === false || req.cot === 'off' || req.abc) continue;
    m.autoReplan = { attempts: [], accepted: false };
    // UI slider (index.ts's yue2ReplanAttempts extension) clamps to [1, 10].
    const attemptsRaw = Number(job.params.yue2ReplanAttempts);
    const maxAttempts = Number.isInteger(attemptsRaw) && attemptsRaw >= 1 && attemptsRaw <= 10 ? attemptsRaw : 3;
    // Plan flags (2026-09-27, Rob): a plan the judge passes can still be
    // illegible (one chord for the whole song, a looped riff, a melody on two
    // pitches). On by default like the runaway re-plans; the same tries.
    const replanFlags = job.params.yue2ReplanFlags !== false;
    const instrumental = job.params.instrumental === true || !req.lyrics;
    pending.push({ m, maxAttempts, replanFlags, instrumental, sheets: [], done: false });
  }
  if (!pending.length) return;

  // Members of one batch share one pick (it is part of the coalescing key).
  const norms = activeStyleNorms(jobPick(pending[0].m.job));
  const props = yue2PropsCached() ?? (await yue2Props()).props;
  const slots = Math.max(1, Number(props?.max_plan_batch ?? props?.max_lm_batch) || 1);
  const randomSeed = () => Math.floor(Math.random() * 2 ** 32);

  for (let round = 1; ; round++) {
    const live = pending.filter(p => !p.done && (p.m.job.status as string) !== 'cancelled'
      && p.m.autoReplan!.attempts.length < p.maxAttempts);
    if (!live.length) break;
    // A round lasts as long as its longest plan, and a runaway runs to the
    // 4096-token cap (12.3 ms/step at 8 wide, 5.8 at 1: ~50 s vs ~24 s), so
    // round one draws a single plan per song (yue2PlanDraws).
    const order = yue2PlanDraws(live.map(p => ({ used: p.m.autoReplan!.attempts.length, max: p.maxAttempts })), slots, round === 1);
    // A pinned seed is the first attempt's (round one's); every re-plan draws its own.
    const draws = order.map(i => {
      const p = live[i];
      return { p, seed: round === 1 && typeof p.m.req.seed === 'number' ? p.m.req.seed : randomSeed() };
    });
    for (const p of live) {
      const n = draws.filter(d => d.p === p).length;
      const from = p.m.autoReplan!.attempts.length + 1;
      p.m.job.status = 'running';
      p.m.job.stage = round === 1 ? 'YuE2: planning the score...'
        : `YuE2: re-planning (attempts ${from}-${from + n - 1} of ${p.maxAttempts}, drawn together)...`;
      p.m.job.progress = 1;
    }
    for (const p of pending) {
      if (live.includes(p) || (p.m.job.status as string) === 'cancelled') continue;
      p.m.job.stage = `YuE2: Plan ready, waiting for ${live.length} other song${live.length === 1 ? '' : 's'} in the batch to finish planning...`;
    }

    let plans: Array<{ abc: string; end_reason: string }>;
    try {
      plans = await runYue2PlanBatch(live[0].m.req, draws.map(d => ({
        style: d.p.m.req.style, ...(d.p.m.req.lyrics ? { lyrics: d.p.m.req.lyrics } : {}), seed: d.seed,
      })));
    } catch (err: any) {
      for (const p of live) {
        p.m.log('WARNING', `[YuE2] Auto-replan round ${round} failed (${err?.message || err}); `
          + (p.m.autoReplan!.attempts.length ? 'picking from the plans already drawn' : "rendering with the engine's own plan"));
        if (!p.m.autoReplan!.attempts.length) p.m.autoReplan = undefined;
        p.done = true;
      }
      break;
    }
    draws.forEach((d, i) => {
      const { p } = d;
      const health = classifyYue2Score(plans[i].abc, plans[i].end_reason, p.m.req.lyrics ?? '', norms);
      const flags = health.legibility?.flags ?? [];
      const attempts = p.m.autoReplan!.attempts;
      attempts.push({ seed: d.seed, verdict: health.verdict, reason: health.reason, ...(flags.length ? { flags } : {}) });
      p.sheets.push(plans[i].abc);
      p.m.log('INFO', `[YuE2] Plan attempt ${attempts.length}: seed ${d.seed}, ${health.verdict} — ${health.reason}${flags.length ? ` — flags: ${flags.join('; ')}` : ''}`);
      if (yue2PlanUsable(health.verdict, p.instrumental) && (!p.replanFlags || !flags.length)) p.done = true;
    });
  }

  for (const p of pending) {
    const { m } = p;
    const autoReplan = m.autoReplan;
    if (!autoReplan) continue;
    const picked = yue2PickPlan(autoReplan.attempts, p.instrumental, p.replanFlags);
    if (!picked) { m.autoReplan = undefined; continue; }
    const seed = picked.pick.seed;
    autoReplan.accepted = yue2PlanUsable(picked.pick.verdict, p.instrumental);
    if (p.replanFlags) autoReplan.clean = picked.clean;
    m.req.abc = p.sheets[picked.index];
    m.req.seed = seed;
    if (!autoReplan.accepted) m.log('WARNING', `[YuE2] Every plan attempt was a runaway, had no vocal line or ran to its cap; rendering the last one (seed ${seed})`);
    else if (p.replanFlags && !picked.clean) m.log('WARNING', `[YuE2] No clean plan in ${autoReplan.attempts.length} attempts: every one carried plan flags. Rendering the least-flagged (seed ${seed}: ${(picked.pick.flags ?? []).join('; ')}). This adapter/checkpoint is not writing good plans.`);
    else if (picked.index > 0) m.log('INFO', `[YuE2] Bad plan replaced: rendering attempt ${picked.index + 1} of ${autoReplan.attempts.length}`);
  }
}

/** Everything about a request except the song itself. Two queued jobs whose
 *  keys match can share one AR batch: same cot, guidance, samplers, NAR
 *  settings and adapters, differing only in prompt, score and seeds. */
export function yue2CoalesceKey(job: GenerationJob, req: Yue2SynthRequest): string {
  const shared: Record<string, unknown> = { ...req };
  // A lyric schedule maps spans of one exact prompt, so scheduled jobs share
  // a batch only when style, lyrics and score are identical too.
  const own = req.lyric_schedule ? ['seed', 'noise_seed', 'lm_batch_size', 'songs']
    : ['style', 'lyrics', 'abc', 'seed', 'noise_seed', 'lm_batch_size', 'songs'];
  for (const k of own) delete shared[k];
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

/** Engine batches in flight, by member job id: the queue's X on one song of
 *  a batch reaches the engine through here (dropYue2BatchMember). */
const yue2RunningBatches = new Map<string, { engineJobId: string; members: Array<{ job: GenerationJob; bits: number }> }>();

/** The queue's X on one song of a running YuE2 batch. Before 2026-09-28 it
 *  only marked the job cancelled: the engine kept recomposing that song and
 *  the rest of the batch waited on it, and an X on the lead cancelled the
 *  whole engine job. Now the engine drops the song and the rest carry on.
 *  Call after setting job.status = 'cancelled'. True = handled; false = the
 *  job is not in a running batch, or it was the last live member, so the
 *  caller cancels it the ordinary way. */
export function dropYue2BatchMember(job: GenerationJob): boolean {
  const batch = yue2RunningBatches.get(job.id);
  const me = batch?.members.find(m => m.job === job);
  if (!batch || !me) return false;
  if (batch.members.every(m => (m.job.status as string) === 'cancelled')) {
    aceClient.cancelJob(batch.engineJobId).catch(() => {});
    return false;
  }
  aceClient.dropSongs(batch.engineJobId, me.bits).catch(err =>
    console.warn(`[YuE2] Could not drop job ${job.id} from engine batch ${batch.engineJobId}: ${err?.message || err}`));
  job.stage = 'Cancelled';
  console.log(`[Generate] Job ${job.id} left YuE2 batch ${batch.engineJobId}; the rest carry on`);
  return true;
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
    // Claim every compatible follower first, then prepare them, then plan
    // them all together. The queue UI boxes jobs by the lead's member list:
    // claiming one at a time showed the lead alone, a "batch" of the
    // followers prepared so far, and the rest still "Queued".
    const claimed: Array<{ cand: GenerationJob; need: number }> = [];
    for (const cand of deps.pendingJobs()) {
      if (songsSoFar >= maxSongs) break;
      if (cand.status !== 'pending' || cand.coalescedInto || cand.params?.yue2Coalesce === false) continue;
      let mapped: Yue2ParamMapping;
      try { mapped = mapYue2Params(cand.params, jobPick(cand)); } catch { continue; }
      if (yue2CoalesceKey(cand, mapped.req) !== key) continue;
      const need = Math.max(1, mapped.req.lm_batch_size ?? 1);
      if (songsSoFar + need > maxSongs) continue;
      cand.coalescedInto = job.id;
      cand.status = 'running';
      cand.stage = 'YuE2: joining a batch...';
      cand.progress = 1;
      claimed.push({ cand, need });
      songsSoFar += need;
    }
    if (claimed.length) job.coalescedMembers = [job.id, ...claimed.map(c => c.cand.id)];
    for (const { cand, need } of claimed) {
      try {
        const prepared = await prepareYue2Job(cand);
        if ((cand.status as string) === 'cancelled') { failYue2Job(cand, new Error('Cancelled')); songsSoFar -= need; continue; }
        members.push(prepared);
      } catch (err: any) {
        failYue2Job(cand, err);
        songsSoFar -= need;
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

  // ── Plan ── every member at once (planYue2Jobs)
  await planYue2Jobs(members);
  if ((job.status as string) === 'cancelled') {
    releaseFollowers('The batch lead was cancelled while planning');
    failYue2Job(job, new Error('Cancelled'));
    return;
  }
  for (let i = members.length - 1; i >= 1; i--) {
    const m = members[i];
    if ((m.job.status as string) !== 'cancelled') continue;
    failYue2Job(m.job, new Error('Cancelled'));
    songsSoFar -= Math.max(1, m.req.lm_batch_size ?? 1);
    members.splice(i, 1);
  }
  for (const m of members.slice(1)) m.log('INFO', `[YuE2] Rendering in one batch with job ${job.id}`);

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
  } else {
    job.coalescedMembers = undefined;   // every claimed follower failed to prepare
  }

  // Each member's songs as a bitmask over the batch (song-major, contiguous),
  // to read the engine's per-song songs_done against.
  const songBits: number[] = [];
  const songStart = new Map<Yue2PreparedJob, number>();
  let songCount = 0;
  for (const m of members) {
    const n = Math.max(1, m.req.lm_batch_size ?? 1);
    songBits.push(((1 << n) - 1) << songCount);
    songStart.set(m, songCount);
    songCount += n;
  }
  const allCancelled = () => members.every(m => (m.job.status as string) === 'cancelled');
  // What the poll watchdog watches for a batch: cancelled only once every
  // member is (one member's X drops its songs, dropYue2BatchMember), and the
  // stage/progress of a member still in it.
  const pollView: GenerationJob = members.length === 1 ? job : {
    get status() { return allCancelled() ? 'cancelled' : 'running'; },
    get stage() { return (members.find(m => (m.job.status as string) !== 'cancelled') ?? lead).job.stage; },
    get progress() { return (members.find(m => (m.job.status as string) !== 'cancelled') ?? lead).job.progress; },
    set acePhase(_v: unknown) { /* the ticker sets each member's */ },
    set acePhaseProgress(_v: unknown) { /* the ticker sets each member's */ },
  } as unknown as GenerationJob;

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
    if (members.length > 1) {
      const entry = { engineJobId: sub.job_id, members: members.map((m, i) => ({ job: m.job, bits: songBits[i] })) };
      for (const m of members) yue2RunningBatches.set(m.job.id, entry);
      // An X that landed between planning and here reached no engine job yet.
      const early = entry.members.filter(e => (e.job.status as string) === 'cancelled').reduce((a, e) => a | e.bits, 0);
      if (early && !allCancelled()) await aceClient.dropSongs(sub.job_id, early).catch(() => {});
    }
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
        if (allCancelled()) return;
        try {
          const status = await aceClient.pollJob(sub.job_id);
          const phase = status.phase as string | undefined;
          const step = status.phase_step ?? 0;
          const total = status.phase_total ?? 0;
          const { stage, progress } = yue2StageText(phase, step, total);
          // A song that has finished composing waits on the rest of the
          // batch; say so rather than show it still "Composing". Keeps the
          // ": Step n/total" tail the stall watchdog reads.
          const songsDone = phase === 'semantic' ? status.songs_done ?? 0 : 0;
          let doneCount = 0;
          for (let i = 0; i < songCount; i++) doneCount += (songsDone >>> i) & 1;
          for (const m of members) {
            if ((m.job.status as string) === 'cancelled') continue;
            const bits = songBits[members.indexOf(m)];
            const waiting = songsDone && (songsDone & bits) === bits ? songCount - doneCount : 0;
            m.job.stage = waiting > 0
              ? `YuE2: Composed, waiting for ${waiting} other song${waiting === 1 ? '' : 's'} in the batch: Step ${step}/${total}`
              : stage;
            m.job.progress = progress;
            m.job.acePhase = phase;
            if (total > 0) m.job.acePhaseProgress = `step ${step}/${total}`;
          }
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

    await deps.pollUntilDone(sub.job_id, pollView, deps.signal, timeoutMinutes);
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
    for (const m of members) yue2RunningBatches.delete(m.job.id);
    releaseFollowers(`The batch render failed (${err?.message || err})`);
    failYue2Job(job, err);
    return;
  }
  for (const m of members) yue2RunningBatches.delete(m.job.id);

  // ── Finish each member from its own slice of the tracks ──
  // Song-major, so a member's tracks are contiguous: lm_batch_size songs
  // times synth_batch_size variations. Song numbers are renumbered per job.
  // Tracks are picked by the batch song index the engine reports: a song
  // dropped from the batch (dropYue2BatchMember) is not rendered, so the
  // tracks after it are not where their position would say.
  const finishOne = async (m: Yue2PreparedJob) => {
    const count = yue2TracksPerJob(m, req);
    const first = songStart.get(m)!;
    const n = Math.max(1, m.req.lm_batch_size ?? 1);
    const mine = trackDetails.flatMap((td, i) => (td.song >= first && td.song < first + n ? [i] : []));
    const slice = mine.map(i => parts[i]);
    const details = mine.map(i => ({ ...trackDetails[i], song: trackDetails[i].song - first }));
    if ((m.job.status as string) === 'cancelled') { failYue2Job(m.job, new Error('Cancelled')); return; }
    if (slice.length !== count) {
      failYue2Job(m.job, new Error(`YuE2 returned ${slice.length} track(s) for this song where ${count} were expected`));
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
  // Only some of the finishing work touches the GPU: StableStep, Whisper, the
  // forced aligner and cover art. A VST chain, mastering and the normaliser
  // are CPU, so those members finish at once, side by side, and never wait
  // for the lane. Members with GPU work wait their turn and run one at a time.
  const cpuOnly = members.filter(m => !yue2FinishNeedsGpu(m.job.params));
  const gpu = members.filter(m => yue2FinishNeedsGpu(m.job.params));
  await Promise.all(cpuOnly.map(m => finishOne(m)));
  if (gpu.length) {
    const finishGpu = async () => { for (const m of gpu) await finishOne(m); };
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
  const { job, req, caption, halves, autoReplan, timing, pipelineStart, log, cover } = p;
  const instrumental = !req.lyrics;
  const saveStart = performance.now();
  {
    // Healthy-but-long vs runaway: the score says which, when there is one
    // (cot=off renders have no plan stage and no score to read).
    const scoreHealth = finalDetail.abc ? classifyYue2Score(finalDetail.abc, finalDetail.end_reason, req.lyrics ?? '', activeStyleNorms(jobPick(job))) : undefined;
    if (scoreHealth) {
      log(scoreHealth.verdict === 'runaway' ? 'WARNING' : 'INFO',
        `[YuE2] Score: ${scoreHealth.verdict} — ${scoreHealth.reason}`);
      if (scoreHealth.legibility?.flags?.length) log('WARNING', `[YuE2] Plan flags: ${scoreHealth.legibility.flags.join('; ')}`);
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
          log('WARNING', '[Whisper] whisper-cli unavailable — skipping');
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
        // Overwrites whatever job.params.yue2Cover/yue2Abc held: req.abc is
        // what was actually rendered (resolved from the envelope capture in
        // prepareYue2Job), and cover's provenance is the envelope's own copy
        // — the only sources the metadata reads for these fields.
        ...(cover ? { yue2Cover: cover } : {}),
        ...(req.abc ? { yue2Abc: req.abc } : {}),
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
  const pick = yue2ResolvePick(params);
  const { req, notes } = mapYue2Params(params, pick);
  if (!req.style.trim()) throw new Error('YuE2 needs a caption — the Style Description field is empty');
  if (req.cot === 'off') throw new Error('Score preview needs Chain of Thought "melody" or "full" — cot=off has no lead sheet to preview');
  // params.semantic: run the semantic stage too and return its codec ids
  // (about a minute) instead of stopping at the lead sheet (seconds).
  const semantic = params?.semantic === true;
  const planReq: Yue2SynthRequest = semantic ? { ...req, semantic_only: true } : { ...req, plan_only: true };
  delete planReq.abc;

  const sub = await yue2Synth(planReq);
  await waitYue2PlanJob(sub.job_id, signal);
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
  return { abc, seed: detail.tracks?.[0]?.seed ?? planReq.seed ?? -1, end_reason, stage_end_reasons: detail.stage_end_reasons, health: classifyYue2Score(abc, end_reason, planReq.lyrics ?? '', activeStyleNorms(pick)), notes, ...(semantic_ids ? { semantic_ids } : {}) };
}

async function waitYue2PlanJob(jobId: string, signal?: AbortSignal): Promise<void> {
  const started = Date.now();
  // ponytail: 10-minute ceiling on a stage that takes seconds; the shared
  // pollUntilDone watchdog is built around a GenerationJob this call has none of.
  for (;;) {
    if (signal?.aborted) {
      await aceClient.cancelJob(jobId).catch(() => {});
      throw new Error('Score preview cancelled');
    }
    const status = await aceClient.pollJob(jobId);
    if (status.status === 'done') return;
    if (status.status === 'failed' || status.status === 'cancelled') {
      const detail = (status as { error?: string }).error;
      throw new Error(`YuE2 plan ${status.status}${detail ? `: ${detail}` : ''}`);
    }
    if (Date.now() - started > 10 * 60_000) {
      await aceClient.cancelJob(jobId).catch(() => {});
      throw new Error('YuE2 plan stage timed out after 10 minutes');
    }
    await new Promise(r => setTimeout(r, 750));
  }
}

/** One plan_only pass over several songs (up to props.max_plan_batch), each
 *  with its own prompt and seed, sharing `req`'s settings. Scores come back
 *  in entry order. */
async function runYue2PlanBatch(req: Yue2SynthRequest, songs: NonNullable<Yue2SynthRequest['songs']>): Promise<Array<{ abc: string; end_reason: string }>> {
  const planReq: Yue2SynthRequest = { ...req, plan_only: true, songs };
  for (const k of ['lyrics', 'abc', 'seed', 'noise_seed', 'lm_batch_size', 'synth_batch_size'] as const) delete planReq[k];
  const sub = await yue2Synth(planReq);
  await waitYue2PlanJob(sub.job_id);
  const tracks = (await yue2FinalDetail(sub.job_id)).tracks ?? [];
  if (tracks.length !== songs.length) throw new Error(`YuE2 returned ${tracks.length} plan(s) for ${songs.length} song(s)`);
  return tracks.map(t => {
    const abc = (t.abc ?? '').trim();
    if (!abc) throw new Error('YuE2 returned no score for the plan stage');
    return { abc, end_reason: t.end_reason ?? 'completed' };
  });
}
