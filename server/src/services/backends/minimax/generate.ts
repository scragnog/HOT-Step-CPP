// backends/minimax/generate.ts — the MiniMax-Music3 generation path
//
// Runs a dequeued generation job against the /mm3/* engine family. Called from
// routes/generate.ts's runGeneration() when the active backend is
// 'minimax-m3'; everything MM3-specific lives here so that diff stays four
// lines (plan §2: shared code must not grow backend branches).
//
// WHAT IS SHARED WITH THE ACE PATH (deliberately, not by accident):
//   • the serial job queue + retry wrapper (routes/generate.ts:enqueueGeneration)
//   • the in-memory job map, /status/:id, /cancel/:id and the SSE progress shape
//   • pollUntilDone's watchdog (injected — MM3 jobs ride the SAME /job endpoints)
//   • the audio dir + /audio/<uuid>.wav URL convention, and the songs INSERT
//   • cover art (a text→image service with no engine coupling)
//
// WHAT IS NOT REUSED (and why — see the post-processing note further down):
//   • the LM phase, LM cache, LM-echo rebuild — MM3's planner is internal to
//     the engine pipeline; there is no two-request seam to cache or rebuild.
//   • translateParams / AceRequest — MM3's wire contract is 8 fields.
//   • adapters, latents, source audio, task modes, streaming, no-adapter render.
//   • the post-processing chain (see AUDIT below).
//
// POST-PROCESSING AUDIT (v1 decision: SKIP the whole chain, store the raw WAV).
// MM3 renders 44.1 kHz stereo; the shared chain is 48 kHz-minded and
// ACE-model-coupled at several stages:
//   • StableStep / SA3 refine — hardcodes `outSr: 48000`
//     (services/generation/postProcessing.ts:386) and needs ACE's SA3 GGUFs.
//   • PP-VAE re-encode — round-trips audio through the ACE VAE (wrong model,
//     wrong rate).
//   • Spectral Lifter / SuperSep / mastering-with-reference — engine endpoints
//     built and tuned for the ACE 48 kHz output.
// A resample or per-stage rate audit is real work with audible consequences,
// so v1 does none of it: correctness over polish. The unprocessed WAV is what
// the model produced, which is also the honest baseline for judging the port.
// Auto-trim is skipped for the same reason it doesn't apply: MM3's duration is
// exact frames, there is no ACE-style duration buffer to trim back off.
// Whisper transcription and LRC are skipped for v1 (no engine-side alignment,
// and the stem-mode path is entangled with the PP chain above).

import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';

import { config } from '../../../config.js';
import { getDb } from '../../../db/database.js';
import { aceClient } from '../../aceClient.js';
import { wavDurationSec } from '../../audioCrop.js';
import { MM3_PLANK_EXT, mm3PlankDir, readMm3Plank } from './plank.js';
import { MM3_LM_ADAPTER_DEFAULT_SCALES, resolveMm3LmAdapterPath } from './lmAdapter.js';
import { runPostProcessingChain } from '../../generation/postProcessing.js';
import type { PostProcessParams } from '../../generation/postProcessing.js';
import {
  startGenerationLog, logGeneration, logGenerationParams,
  finishGenerationLog, failGenerationLog,
} from '../../logger.js';
import {
  mm3Synth, mm3JobDetail, mm3TokenizeCheck, mm3Unload,
  type Mm3SynthRequest, type Mm3JobDetail,
} from './client.js';
import type { GenerationJob, StageTiming } from '../../../routes/generate.js';

/** Injected so this module never imports back into routes/generate.ts at
 *  runtime (type-only import above is erased). pollUntilDone owns the stall
 *  watchdog, the wall-clock timeout and cancel checks — MM3 jobs use the same
 *  engine /job endpoints, so it applies unchanged. */
export interface MinimaxGenerationDeps {
  pollUntilDone(
    aceJobId: string, job: GenerationJob, signal: AbortSignal, timeoutMinutes?: number,
  ): Promise<void>;
}

/** MM3's own duration ceiling (mirrors the capability manifest). */
const MM3_MAX_DURATION_SEC = 300;
const MM3_DEFAULT_DURATION_SEC = 60;

// ── Low-step schedule compensation ───────────────────────────────────────────
//
// MM3's native schedule is UNIFORM with shift=1 (engine mm3-dit-graph.h:744),
// faithfully reproducing upstream — and upstream's checkpoint declares
// `mm3.flow.steps = 30`. There is no low-step-count compensation anywhere in
// the reference, so dropping `steps` alone degrades in a specific, measurable
// way rather than gracefully.
//
// Why it degrades: the vocoder decodes latent channels 0..63 as LEFT and
// 64..127 as RIGHT in two INDEPENDENT passes (mm3-vocoder-graph.h:596), and the
// initial noise is i.i.d. across all 128 channels (mm3-pipeline.h:462). So the
// two halves start completely uncorrelated and every bit of stereo coherence
// has to be manufactured by the DiT along the trajectory. Coarse Euler steps
// leave that work unfinished. Measured on a matched 10-vs-30-step pair:
//
//   L/R correlation   -0.07  vs  +0.77      (anti-phase mids, 160 Hz - 2.6 kHz)
//   side/mid ratio    +0.5 dB vs -9.5 dB
//   mid spectrum      -8 dB @ 60 Hz, tapering to 0 dB above 2 kHz
//
// i.e. thin and phasey ("tinny"), NOT dull — the HF is already at the correct
// absolute level. That is the tell that the LATE trajectory is converged and
// only the EARLY (high-noise) part is starved, which is exactly what a shifted
// schedule buys back.
//
// The law: the shift warp t' = shift*t/(1+(shift-1)*t) turns the uniform grid
// u = i/steps into sigma(u) = u / (shift - (shift-1)*u), whose FIRST step is
// 1/(shift*(steps-1)+1). Setting that equal to the 30-step native first step of
// 1/30 and solving gives shift = 29/(steps-1) — which returns exactly 1.0 at
// 30 steps, so the curve is continuous at the boundary and this can never
// perturb a default render.
//
// VALIDATED BY EAR at 10 steps (shift 3.2). 8..29 is interpolation on a curve
// anchored at both ends. Below 8 it is EXTRAPOLATION: the first-step match is
// bought with an ever-larger final leap to clean (0.54 at 6 steps, 0.86 at 2),
// and there is a point where that must break down. If a very-low-step render
// comes out muddy/smeared rather than thin, the shift is too high for that
// budget — that is the signature to tune against.
const MM3_REFERENCE_STEPS = 30;
const MM3_MIN_STEPS = 2;
const MM3_MAX_STEPS = 60;
const MM3_FLOW_SHIFT_MAX = 20;  // engine-side ceiling, mm3-request.h:986

/** Schedule shift that gives `steps` the same high-noise resolution as 30 steps. */
export function mm3LowStepShift(steps: number): number {
  if (!Number.isFinite(steps) || steps <= 1) return MM3_FLOW_SHIFT_MAX;
  const raw = (MM3_REFERENCE_STEPS - 1) / (steps - 1);
  return Math.min(MM3_FLOW_SHIFT_MAX, Math.max(1, Math.round(raw * 100) / 100));
}
/** Progress ticker interval — /mm3/job never takes the engine's MM3 mutex. */
const DETAIL_POLL_MS = 1_500;

// ── Structured Caption repair ────────────────────────────────────────────────

/**
 * The three heading lines of an MM3 Structured Caption, each paired with the
 * first labelled field that belongs under it.
 *
 * All 1,000 of MiniMax's reference captions carry all three headings as bare
 * lines, and Lyric Studio writes them (every one of the 506 captions in the DB
 * starts "Global Metadata"). A caption can still reach here with one missing —
 * measured 2026-08-21: three MM3 renders went out byte-identical to their
 * stored caption MINUS its first line, so the model saw an unlabelled Global
 * Metadata block sitting above two labelled ones. Nothing in this repo strips
 * it, which leaves the caption box itself — a paste that began on line 2 looks
 * exactly like this — so the fix belongs on the way OUT, where every path is
 * covered regardless of how the text got into the box.
 */
const MM3_HEADINGS: ReadonlyArray<{ heading: string; firstLabel: RegExp }> = [
  { heading: 'Global Metadata', firstLabel: /^Basic Attributes\s*:/i },
  { heading: 'Vocal Details', firstLabel: /^Vocal Gender & Timbre\s*:/i },
  { heading: 'Arrangement', firstLabel: /^Instrument Lifecycle Description/i },
];

/**
 * Re-insert any heading line missing from an otherwise well-formed Structured
 * Caption. Returns the caption unchanged, and reports nothing, when the text is
 * not a Structured Caption at all — a bare "Energetic synthwave, 120 BPM" is a
 * legal MM3 caption (it is what the reference pipeline's own fixture sends) and
 * must never grow headings it did not ask for.
 */
export function repairMm3CaptionHeadings(caption: string): { caption: string; restored: string[] } {
  const lines = caption.split('\n');
  // Anchor on the labels, not the headings: a caption with no labels at all is
  // a plain description and is left alone.
  if (!MM3_HEADINGS.some(h => lines.some(l => h.firstLabel.test(l)))) {
    return { caption, restored: [] };
  }

  const restored: string[] = [];
  // Back to front, so an insertion never shifts an index still to be used.
  for (let i = MM3_HEADINGS.length - 1; i >= 0; i--) {
    const { heading, firstLabel } = MM3_HEADINGS[i];
    if (lines.some(l => l.trim().toLowerCase() === heading.toLowerCase())) continue;
    const at = lines.findIndex(l => firstLabel.test(l));
    if (at < 0) continue;   // that whole section is absent — not ours to invent
    lines.splice(at, 0, heading);
    restored.unshift(heading);
  }

  return { caption: restored.length ? lines.join('\n') : caption, restored };
}

// ── Param mapping ────────────────────────────────────────────────────────────

export interface MinimaxParamMapping {
  req: Mm3SynthRequest;
  /** Non-fatal notes (clamps, ignored knobs) surfaced to the generation log. */
  notes: string[];
}

/**
 * UI params → /mm3/synth request.
 *
 * The caption field list is IDENTICAL to translateParams.ts:49 on purpose: the
 * UI has one "what to generate" text field and both backends must read the
 * same one, or switching backends would silently change which box matters.
 * Everything ACE-specific (adapters, LM knobs, task modes, latents,
 * bpm/keyscale/timesignature, negative prompt) is ignored — MM3 has no wire
 * slot for any of it.
 *
 * The solver/scheduler/guidance picks are the exception, and only since the
 * sampler-plugin bridge landed: the SAME Lua plugins now drive both DiTs, so
 * those three fields DO travel — but only on explicit opt-in. See
 * mapMinimaxSamplerPlugins below for why that gate exists.
 */
export function mapMinimaxParams(params: any): MinimaxParamMapping {
  const notes: string[] = [];

  const rawCaption: string = params.prompt || params.songDescription || params.caption || params.style || '';
  const repaired = repairMm3CaptionHeadings(rawCaption);
  const caption = repaired.caption;
  if (repaired.restored.length) {
    notes.push(
      `Structured Caption was missing its ${repaired.restored.map(h => `"${h}"`).join(' and ')} `
      + `heading line — restored before sending. All 1,000 of MiniMax's reference captions carry all three.`,
    );
  }

  // Instrumental: MM3's contract is blank lyrics → the engine substitutes its
  // own instrumental token. Do NOT send ACE's '[Instrumental]' sentinel; that
  // would be tokenized as literal lyric text.
  const lyrics: string = params.instrumental ? '' : (params.lyrics || '');

  let duration = Number(params.duration);
  if (!(duration > 0)) {
    duration = MM3_DEFAULT_DURATION_SEC;
    notes.push(`no duration supplied — defaulting to ${MM3_DEFAULT_DURATION_SEC}s`);
  }
  // The ACE duration buffer (autoTrimEnabled + durationBuffer) is deliberately
  // NOT added: nothing trims it back off on this path.
  if (duration > MM3_MAX_DURATION_SEC) {
    notes.push(`duration ${Math.round(duration)}s exceeds the MiniMax-Music3 limit — clamped to ${MM3_MAX_DURATION_SEC}s`);
    duration = MM3_MAX_DURATION_SEC;
  }

  // -1 tells the engine to draw one; it echoes the resolved value back so the
  // take stays reproducible.
  const seed: number = params.randomSeed
    ? -1
    : (typeof params.seed === 'number' && params.seed >= 0 ? params.seed : -1);

  // AR-stage seed. -1 (the default) ties it to `seed`, which is MM3's native
  // behaviour. Setting it explicitly pins the PLAN while the flow noise still
  // follows `seed` — the only way to reroll the noise and still hit the AR
  // cache. Range-checked here so a junk value never reaches the engine.
  // Blank/absent means "tied". Number('') is 0, not NaN, so the blank check
  // has to come first or an untouched text field would silently pin the
  // planner to seed 0.
  const arSeedRaw = String(params.mm3ArSeed ?? '').trim();
  const arSeedNum = arSeedRaw === '' ? NaN : Number(arSeedRaw);
  const arSeed: number = Number.isFinite(arSeedNum) && arSeedNum >= 0 ? Math.round(arSeedNum) : -1;

  // The one combination that cannot work, called out rather than left to look
  // like a dead toggle: a fresh random seed each render means a fresh plan each
  // render, so there is nothing for the cache to match.
  if (params.mm3ReuseAr !== false && seed < 0 && arSeed < 0) {
    notes.push(
      'AR cache is on but the seed is random — every render plans afresh, so it cannot hit. '
      + 'Fix the seed, or set an explicit AR seed to keep the plan while the flow noise rerolls.',
    );
  }

  const requestedBatch = Number(params.batchSize) || 1;
  if (requestedBatch > 1) {
    notes.push(`batchSize ${requestedBatch} requested — MiniMax-Music3 v1 generates 1 track per job (capability manifest: batch.max = 1)`);
  }

  // Hoisted above the sampler-plugin mapping: low-step compensation has to know
  // the step count before it can decide whether to engage.
  const rawSteps = Number(params.mm3Steps);
  const resolvedSteps =
    Number.isFinite(rawSteps) && rawSteps >= MM3_MIN_STEPS && rawSteps <= MM3_MAX_STEPS
      ? Math.round(rawSteps)
      : undefined;

  const samplerPlugins = mapMinimaxSamplerPlugins(params, resolvedSteps, notes);

  return {
    req: {
      caption,
      lyrics,
      duration: Math.round(duration * 1000) / 1000,
      seed,
      // Surfaced as capability `extensions` (backends/minimax/index.ts) rather
      // than left checkpoint-fixed: the flow stage is the majority of wall time
      // on a full-length render and `steps` is a linear dial on it. Both are
      // omitted from the wire when unset so the engine keeps applying the
      // checkpoint defaults (30 / 1.7) — an out-of-range value is dropped here
      // rather than sent for the engine to reject mid-job.
      ...(resolvedSteps !== undefined ? { steps: resolvedSteps } : {}),
      ...(Number.isFinite(Number(params.mm3CfgFlow)) && Number(params.mm3CfgFlow) >= 1.0 &&
          Number(params.mm3CfgFlow) <= 5.0
            ? { cfg_flow: Number(params.mm3CfgFlow) }
            : {}),
      get_wav_bits: 16,
      // LRC timestamps ride the same toggle ACE uses (skipLrc inverted).
      // Instrumentals are filtered engine-side, so no check is needed here.
      get_lrc: params.skipLrc !== true,
      // MM3 Plank capture — opt-in, zero cost when off.
      ...(params.mm3SaveArCodes === true ? { get_ar_codes: true } : {}),
      // ── AR cache ──
      //
      // `!== false`, not `=== true`, and that is not a typo: the UI only writes
      // a backend-declared param into the request once the user TOUCHES it
      // (BackendGenerationDropdown reads `backendParams[key] ?? p.default`).
      // For a manifest default of true, an untouched control shows ON while
      // sending nothing — so an absent value has to mean the declared default
      // or the toggle would lie. Same idiom as get_lrc/skipLrc above.
      //
      // The engine owns the "did anything upstream change?" decision; the
      // server never has to model it. See engine/src/minimax/mm3-ar-cache.h.
      ...(params.mm3ReuseAr !== false ? { reuse_ar: true } : {}),
      ...(arSeed >= 0 ? { ar_seed: arSeed } : {}),
      // ── Streaming ──
      // `=== true`, not `!== false`: the manifest default is OFF, so an
      // untouched control must resolve to off. (The mirror-image of the
      // mm3ReuseAr comment above — the idiom follows the declared default,
      // it is not a house style.)
      ...(params.mm3Stream === true ? { stream: true } : {}),
      // Ensemble takes: N different songs from this one prompt in a single
      // batched AR pass. Omitted at 1 so an ordinary render's wire request is
      // byte-for-byte what it always was.
      ...(Number(params.mm3Takes) > 1 ? { takes: Math.min(8, Math.max(1, Math.round(Number(params.mm3Takes)))) } : {}),
      // MM3 Plank replay. A plank that will not load is a note, not a failure:
      // the render proceeds with a normal AR pass.
      ...(params.mm3PlankPath ? (() => {
        const codes = readMm3Plank(String(params.mm3PlankPath));
        if (!codes) {
          notes.push(`AR plank not found or unreadable (${params.mm3PlankPath}) — running a normal AR pass`);
          return {};
        }
        // Entry 0 is the un-emitted iteration, so I codes render I-1 frames.
        // The engine derives max_frames/duration from the array itself
        // (mm3-request.h), so nothing needs overriding here.
        notes.push(
          `AR replay from ${path.basename(String(params.mm3PlankPath))} `
          + `(${codes.forced_semantic.length - 1} frames — codes pinned, AR compute still runs)`,
        );
        return {
          forced_semantic: codes.forced_semantic,
          forced_acoustic: codes.forced_acoustic,
        };
      })() : {}),
      ...samplerPlugins,
      // Runtime LM LoRA (engine mm3-lm-adapter.h). A reference that fails
      // containment is a NOTE + base-model render, never a silent partial —
      // but once the path is sent, a load failure fails the JOB engine-side
      // (an adapter that silently didn't load is indistinguishable from "the
      // adapter does nothing").
      ...(params.mm3LmAdapter ? (() => {
        const resolved = resolveMm3LmAdapterPath(String(params.mm3LmAdapter));
        if (!resolved || !fs.existsSync(resolved)) {
          notes.push(`LM adapter not found (${params.mm3LmAdapter}) — rendering with the base model`);
          return {};
        }
        const dial = (key: string, fallback: number): number => {
          const v = Number((params as Record<string, unknown>)[key]);
          return Number.isFinite(v) && v >= -4 && v <= 4 ? v : fallback;
        };
        const d = MM3_LM_ADAPTER_DEFAULT_SCALES;
        // Application mode: "runtime" (default; low-rank deltas in-graph, live
        // dials, ~+28%/step at r256) or "merge" (folded into the resident
        // weights once — zero per-step cost; scale changes re-merge).
        const mode = params.mm3LmAdapterMode === 'merge' ? 'merge' : 'runtime';
        notes.push(`LM adapter: ${path.basename(resolved)} (${mode})`);
        return {
          lm_adapter: resolved,
          lm_adapter_mode: mode,
          lm_adapter_scale: dial('mm3LmAdapterScale', d.scale),
          lm_adapter_scale_attn: dial('mm3LmAdapterScaleAttn', d.scaleAttn),
          lm_adapter_scale_mlp: dial('mm3LmAdapterScaleMlp', d.scaleMlp),
          lm_adapter_scale_early: dial('mm3LmAdapterScaleEarly', d.scaleEarly),
          lm_adapter_scale_mid: dial('mm3LmAdapterScaleMid', d.scaleMid),
          lm_adapter_scale_late: dial('mm3LmAdapterScaleLate', d.scaleLate),
        };
      })() : {}),
    },
    notes,
  };
}

/**
 * Sampler-plugin fields for the /mm3/synth request, or {} when the user has
 * not opted in.
 *
 * OPT-IN IS DELIBERATE, and it is the whole reason this is a separate function
 * rather than three lines in the mapping above. The solver/guidance/scheduler
 * pickers are shared global state: the UI always holds a value for them (ACE
 * defaults to euler + apg), so forwarding them unconditionally would move EVERY
 * MiniMax-Music3 render off the native flow loop the moment this shipped —
 * silently, and for users who never asked for it. `guidance_mode: "apg"` in
 * particular is not MM3's plain CFG; it is a different algorithm.
 *
 * So the picks only travel when `samplerPluginsEnabled` is on. Off (the default)
 * is byte-for-byte today's behaviour. The key is backend-neutral on purpose:
 * it is a declared extension knob, and BackendGenerationDropdown reads it to
 * decide whether to show the pickers at all — that component must not learn any
 * MM3-specific names.
 *
 * Names are ACE's throughout — see translateParams.ts:128-130, 273, 300. The
 * engine takes the same spellings, so this is a pass-through, not a mapping.
 */
function mapMinimaxSamplerPlugins(
  params: any,
  steps: number | undefined,
  notes: string[],
): Partial<Mm3SynthRequest> {
  const autoOn  = params.mm3AutoLowStep !== false;   // opt-OUT, see index.ts
  const lowStep = steps !== undefined && steps < MM3_REFERENCE_STEPS;
  const autoShift = lowStep ? mm3LowStepShift(steps as number) : 1;

  // The user's own shift wins, but only if they actually moved the slider —
  // 1.0 IS the default, so it cannot be distinguished from "untouched". Anyone
  // who genuinely wants the raw native schedule at a low step count turns
  // Low-Step Compensation off, which is unambiguous.
  const rawShift = Number(params.mm3FlowShift);
  const userShift = Number.isFinite(rawShift) && rawShift > 0
      && rawShift <= MM3_FLOW_SHIFT_MAX && rawShift !== 1
    ? rawShift
    : undefined;

  if (!params.samplerPluginsEnabled) {
    if (!autoOn || !lowStep) return {};
    // AUTOMATIC PATH. Deliberately narrow: scheduler + shift and nothing else.
    // `inferMethod` / `guidanceMode` / `apgNormThreshold` / `pluginParams` are
    // shared global pickers that hold ACE's defaults (euler + APG) whether or
    // not this backend was ever considered — forwarding them here would swap
    // MM3 onto a different guidance ALGORITHM as a side effect of lowering a
    // step count. That is precisely the failure the gate below exists to
    // prevent, and engaging automatically makes it worse, not better.
    const shift = userShift ?? autoShift;
    notes.push(
      `low-step compensation: ${steps} steps is under the checkpoint's ${MM3_REFERENCE_STEPS}, so the flow ran `
      + `scheduler=linear at flow_shift=${shift}`
      + (userShift !== undefined
          ? ' (your Schedule Shift)'
          : autoShift >= MM3_FLOW_SHIFT_MAX
            ? ` (29/(steps-1) wanted ${Math.round(((MM3_REFERENCE_STEPS - 1) / ((steps as number) - 1)) * 100) / 100}, clamped to the engine ceiling — under ~8 steps this law is extrapolating)`
            : ' (29/(steps-1), matching 30-step first-step resolution)')
      + '. Solver and guidance stay native. Turn off "Low-Step Compensation" for the raw native schedule.',
    );
    return { scheduler: 'linear', flow_shift: shift };
  }

  const out: Partial<Mm3SynthRequest> = {};
  if (params.inferMethod) out.infer_method = params.inferMethod;
  if (params.scheduler) out.scheduler = params.scheduler;
  if (params.guidanceMode) out.guidance_mode = params.guidanceMode;

  if (userShift !== undefined) {
    out.flow_shift = userShift;
  } else if (autoOn && lowStep) {
    // Plugins on, shift left at default, steps under 30: still compensate.
    // Their scheduler pick stands if they made one — shift warps whichever
    // curve is selected, so this stays additive rather than overriding.
    out.flow_shift = autoShift;
    if (!out.scheduler) out.scheduler = 'linear';
    notes.push(
      `low-step compensation: flow_shift=${autoShift} applied for ${steps} steps `
      + `(29/(steps-1)) on scheduler=${out.scheduler}`,
    );
  }
  if (Number.isFinite(Number(params.apgNormThreshold))
      && Number(params.apgNormThreshold) >= 0 && Number(params.apgNormThreshold) <= 100) {
    out.apg_norm_threshold = Number(params.apgNormThreshold);
  }
  if (params.pluginParams && Object.keys(params.pluginParams).length > 0) {
    out.plugin_params = params.pluginParams;
  }

  if (Object.keys(out).length === 0) {
    notes.push('sampler plugins enabled but nothing selected — running the native MiniMax-Music3 flow loop');
    return {};
  }
  notes.push(
    `sampler plugins: solver=${out.infer_method ?? '(native euler)'}, `
    + `scheduler=${out.scheduler ?? '(native)'}, guidance=${out.guidance_mode ?? '(native cfg)'}`
    + ' — this leaves the parity-proven default path',
  );
  return out;
}

// ── Progress ─────────────────────────────────────────────────────────────────

/** MM3 stage → user-facing text + progress %.
 *
 *  `elapsedSec` is appended ONLY for stages the engine cannot report progress
 *  inside (weights load, arbitration, stitch/encode). That keeps the shared
 *  stall watchdog fed through a legitimately long, silent phase — a 24 GB cold
 *  load can exceed its 120 s limit — while leaving the stepped stages (ar /
 *  cond / flow / vocode) genuinely stall-detectable. */
export function minimaxStageText(d: Mm3JobDetail, elapsedSec: number): { stage: string; progress: number } {
  const frac = (a: number, b: number) => (b > 0 ? Math.max(0, Math.min(1, a / b)) : 0);
  const silent = (text: string, progress: number) => ({ stage: `${text} (${elapsedSec}s)`, progress });

  switch (d.stage) {
    case 'queued':
      return silent('MiniMax-Music3: queued', 2);
    case 'arbitrating':
      return silent('MiniMax-Music3: making room in VRAM', 4);
    case 'warming':
      return silent('MiniMax-Music3: loading weights', 6);
    case 'warm':
      return silent('MiniMax-Music3: weights resident', 8);
    case 'ar':
      // The autoregressive planner — MM3's analogue of the ACE LM phase.
      return { stage: `MiniMax-Music3: planning (frame ${d.step}/${d.n_steps})`, progress: 10 + Math.round(frac(d.step, d.n_steps) * 25) };
    case 'cond':
      return { stage: `MiniMax-Music3: conditioning (window ${d.window + 1}/${d.n_windows})`, progress: 36 + Math.round(frac(d.window + 1, d.n_windows) * 4) };
    case 'flow': {
      // Fraction across ALL windows, not within one. Per-window it sawtoothed
      // 40->85 once per window; that was survivable when vocoding came after
      // every window, and stops being so with streaming on, where the vocoder
      // reports in BETWEEN flow passes.
      const overall = d.n_windows > 0 ? (d.window + frac(d.step, d.n_steps)) / d.n_windows : frac(d.step, d.n_steps);
      return {
        stage: `MiniMax-Music3: flow step ${d.step}/${d.n_steps}${d.n_windows > 1 ? ` (window ${d.window + 1}/${d.n_windows})` : ''}`,
        progress: 40 + Math.round(Math.max(0, Math.min(1, overall)) * 45),
      };
    }
    case 'stream': {
      // Interleaved streaming: the planner and the flow stage take turns many
      // times a second, so their own bands (10-35 and 40-85) would make the bar
      // oscillate for the whole render. One axis instead — audio actually
      // produced — with partial credit for the window currently being planned
      // so it does not sit at zero through the first window.
      //   window/n_windows = windows emitted, step/n_steps = frames planned.
      const framesPerWindow = d.n_windows > 0 ? d.n_steps / d.n_windows : d.n_steps;
      const nextFrac = framesPerWindow > 0
        ? Math.max(0, Math.min(1, d.step / framesPerWindow - d.window))
        : 0;
      const overall = d.n_windows > 0 ? (d.window + nextFrac) / d.n_windows : 0;
      return {
        stage: `MiniMax-Music3: planning + rendering (${d.window}/${d.n_windows} windows, frame ${d.step}/${d.n_steps})`,
        progress: 10 + Math.round(Math.max(0, Math.min(1, overall)) * 80),
      };
    }
    case 'vocode': {
      // Two shapes, because the vocoder runs at two different points. Serial:
      // one pass after every window is denoised, in its own 85-91 band.
      // Streaming: inline after each window's flow, so it belongs at exactly
      // the position that window occupies in the 40-85 band — reporting it at
      // 85 would run the bar forward and then snap it back, once per window.
      const stage = `MiniMax-Music3: vocoding (window ${d.window + 1}/${d.n_windows})`;
      return d.streaming
        ? { stage, progress: 40 + Math.round(frac(d.window + 1, d.n_windows) * 45) }
        : { stage, progress: 85 + Math.round(frac(d.window + 1, d.n_windows) * 6) };
    }
    case 'stitch':
      return silent('MiniMax-Music3: stitching windows', 92);
    case 'encoding':
      return silent('MiniMax-Music3: encoding WAV', 93);
    case 'done':
      return { stage: 'MiniMax-Music3: fetching audio', progress: 94 };
    default:
      return silent(`MiniMax-Music3: ${d.stage}`, 5);
  }
}

// ── The run ──────────────────────────────────────────────────────────────────

export async function runMinimaxGeneration(job: GenerationJob, deps: MinimaxGenerationDeps): Promise<void> {
  const pipelineStart = performance.now();
  const timing: StageTiming[] = [];
  const timeoutMinutes: number | undefined = job.params.generationTimeoutMinutes;
  const log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => logGeneration(job.id, level, msg);

  if (job.status === 'cancelled') return;

  const { req, notes } = mapMinimaxParams(job.params);

  const abortController = new AbortController();
  (job as any)._abort = abortController;

  startGenerationLog(job.id, 'mm3-text2music');
  logGenerationParams(job.id, req as unknown as Record<string, unknown>);
  for (const n of notes) log('WARNING', `[MM3] ${n}`);

  // caption length is the SEND-side half of the caption echo: the engine prints
  // the cleaned caption itself ([MM3-Job] <id> caption ...), so a mismatch
  // between these two lines localises a drop to the wire rather than the UI.
  console.log(`[Generate] Job ${job.id} — backend=minimax-m3, duration=${req.duration}s, seed=${req.seed}, caption=${req.caption.length} chars, lyrics=${req.lyrics ? `${req.lyrics.length} chars` : '(instrumental)'}`);

  let detailTimer: NodeJS.Timeout | undefined;

  try {
    if (!req.caption.trim()) {
      throw new Error('MiniMax-Music3 needs a caption — the Style Description field is empty');
    }

    // ── Defensive arbitration is NOT needed here ──
    // The MM3 job evicts the ACE side itself, on the GPU worker thread, only
    // when the weights would not otherwise fit (mm3-job.h:183 mm3_arbitrate_vram).
    // Unloading ACE from Node would fight that logic and slow down the common
    // case where both fit. The reverse direction (ACE defensively unloading
    // MM3) IS done in routes/generate.ts, because the ACE path has no
    // equivalent engine-side arbitration.

    // ── Pre-flight: prompt token budget ──
    // Cold-capable, so this catches an over-long caption+lyrics before any GPU
    // work. POST /mm3/synth re-checks and would 400 anyway; doing it here buys
    // a clear, actionable job error instead of a raw HTTP failure.
    job.status = 'synth_running';
    job.stage = 'MiniMax-Music3: checking prompt length...';
    job.progress = 1;
    const tokStart = performance.now();
    try {
      const tok = await mm3TokenizeCheck(req.caption, req.lyrics || '');
      log('INFO', `[MM3] Prompt: ${tok.tokens}/${tok.limit} tokens${tok.instrumental ? ' (instrumental)' : ''}`);
      if (!tok.ok) {
        throw new Error(
          `Prompt is too long for MiniMax-Music3: ${tok.tokens} tokens vs a ${tok.limit}-token limit. `
          + 'Shorten the caption (the Arrangement section is usually the longest part) or the lyrics.',
        );
      }
    } catch (tokErr: any) {
      // Only a genuine over-limit answer is fatal; a failed probe (engine cold,
      // busy, tokenizer unavailable) must not block the run — /mm3/synth
      // enforces the same limit synchronously.
      if (tokErr?.message?.startsWith('Prompt is too long')) throw tokErr;
      log('WARNING', `[MM3] Token pre-flight unavailable (continuing — /mm3/synth re-checks): ${tokErr.message}`);
    }
    const tokMs = Math.round(performance.now() - tokStart);
    if (tokMs > 50) timing.push({ name: 'Token pre-flight', ms: tokMs });

    // ── Submit ──
    job.stage = 'MiniMax-Music3: submitting...';
    job.progress = 2;
    const submitStart = performance.now();
    const sub = await mm3Synth(req);
    job.aceJobId = sub.job_id;   // standard /job id — /cancel/:id reaches it unchanged
    // Published on the job so GET /api/generate/status/:id can tell the browser
    // whether there is anything to listen to. The ENGINE's answer, not the
    // request's: it is allowed to decline, and when it does the UI must behave
    // exactly as it does today rather than opening a stream that never fills.
    job.mm3Streaming = req.stream === true && sub.streaming !== false;
    // The render's resolved length, known the instant the job is accepted. The
    // browser needs it BEFORE any audio arrives, so a streaming card can show
    // how much of the track is finished instead of an indeterminate spinner.
    job.mm3Duration = sub.duration;
    // How many songs this render will produce, published the instant the engine
    // accepts the job. The browser stands up one queue entry and one card per
    // take off this — waiting until the takes finish would put them all on
    // screen at the end, which is the one thing streaming exists to avoid.
    //
    // Seeds go over as DECIMAL STRINGS: they are uint64, and
    // 18226392072674864222 with its two successors all collapse onto the same
    // float64 — which is exactly what made three distinct takes report one
    // seed and become individually unreproducible.
    job.mm3Takes = Math.max(1, Number(sub.takes ?? req.takes ?? 1));
    if (job.mm3Takes > 1) {
      // seed_str, never seed: the number has already lost the low digits by the
      // time it reaches JS, so basing the takes on it would give three seeds
      // that are all wrong and all identical.
      const base = BigInt(sub.seed_str ?? String(sub.seed ?? 0));
      job.mm3TakeSeeds = Array.from({ length: job.mm3Takes }, (_, t) => (base + BigInt(t)).toString());
      log('INFO', `[MM3] Ensemble: ${job.mm3Takes} takes from one prompt, seeds `
        + `${job.mm3TakeSeeds[0]}..${job.mm3TakeSeeds[job.mm3Takes - 1]}`);
    }
    if (req.stream && !job.mm3Streaming) {
      log('WARNING', '[MM3] Streaming was requested but the engine declined it — this render is not streamable');
    } else if (job.mm3Streaming) {
      log('INFO', '[MM3] Streaming on — windows play as they finish; the complete WAV is still saved as usual');
    }

    // Persist the RESOLVED seed so a randomized take is reproducible.
    job.params.seed = sub.seed;
    job.params.randomSeed = false;
    log('INFO',
      `[MM3] Job ${sub.job_id} submitted — ${sub.prompt_tokens} prompt tokens, ${sub.max_frames} frames `
      + `(${sub.duration.toFixed(1)}s), seed ${sub.seed}, ${sub.steps} steps, cfg ${sub.cfg_flow}`
      + `${sub.instrumental ? ', instrumental' : ''}`);
    // Echoed by the engine only when a plugin was actually selected — the
    // difference between "I picked a solver" and "a solver ran".
    if (sub.sampler_plugins) {
      const sp = sub.sampler_plugins;
      log('INFO',
        `[MM3] Sampler plugins in force — solver=${sp.solver || '(native euler)'}, `
        + `scheduler=${sp.scheduler || '(native)'}, guidance=${sp.guidance || '(native cfg)'}, `
        + `shift=${sp.shift}, ${sp.n_params} declared param(s)`);
    }

    // ── Progress ticker ──
    // GET /job (polled by pollUntilDone) gives the ACE-phase mapping; this adds
    // MM3's own vocabulary and keeps the stall watchdog fed.
    const runStart = Date.now();
    detailTimer = setInterval(() => {
      void (async () => {
        const d = await mm3JobDetail(sub.job_id);
        if (!d || job.status === 'cancelled') return;
        const elapsed = Math.round((Date.now() - runStart) / 1000);
        const { stage, progress } = minimaxStageText(d, elapsed);
        job.stage = stage;
        job.progress = progress;
        // MM3's stage is finer-grained than GET /job's phase mapping; surface
        // it through the existing ace_phase_progress channel.
        if (d.n_steps > 0) job.acePhaseProgress = `step ${d.step}/${d.n_steps}`;
        // Whether the engine is dispatching windows DURING planning. Only
        // knowable once the worker has run its VRAM check, and it is the
        // difference between "audio in seconds" and "audio after the plan", so
        // it is logged once and published for the player to be honest about.
        if (d.streaming && d.stream_interleaved !== undefined && job.mm3Interleaved === undefined) {
          job.mm3Interleaved = d.stream_interleaved === true;
          log('INFO', d.stream_interleaved
            ? '[MM3] Streaming is INTERLEAVED — windows render while the planner runs, so audio starts in seconds'
            : '[MM3] Streaming is SERIAL — the two model stacks would not co-reside, so audio starts once planning finishes');
        }
      })();
    }, DETAIL_POLL_MS);
    detailTimer.unref?.();

    await deps.pollUntilDone(sub.job_id, job, abortController.signal, timeoutMinutes);
    clearInterval(detailTimer);
    detailTimer = undefined;
    timing.push({ name: 'MM3 Generate', ms: Math.round(performance.now() - submitStart) });

    // ── Result ──
    job.stage = 'MiniMax-Music3: saving audio...';
    job.progress = 95;
    const saveStart = performance.now();
    // Final MM3 detail: shape + per-stage timings, straight from the engine.
    const finalDetail = await mm3JobDetail(sub.job_id);
    if (finalDetail?.result) {
      const r = finalDetail.result;
      log('INFO',
        `[MM3] Rendered ${r.frames} frames → ${r.duration_sec.toFixed(1)}s @ ${r.sample_rate} Hz, `
        + `rms ${r.rms.toFixed(4)}, peak ${r.peak.toFixed(3)}${r.eos ? ', EOS hit' : ''}${r.has_nan ? ' — WARNING: NaNs present' : ''}`);
      // AR cache outcome. Worth a line of its own: a hit changes the shape of
      // the run (no LM load, no stage 1) and "why was that render so fast?" /
      // "why did it re-plan?" are the two questions this feature raises.
      if (finalDetail.ar_cached) {
        log('INFO', '[MM3] AR cache HIT — planner skipped, this render was flow-stage only');
      } else if (req.reuse_ar) {
        log('INFO', '[MM3] AR cache miss — planned fresh; the result is held for the next render');
      }
      if (r.ms) {
        timing.push({ name: finalDetail.ar_cached ? '  AR planner (cached)' : '  AR planner', ms: Math.round(r.ms.ar) });
        timing.push({ name: '  Condition encoder', ms: Math.round(r.ms.cond) });
        timing.push({ name: '  Flow (DiT)', ms: Math.round(r.ms.flow) });
        timing.push({ name: '  Vocoder', ms: Math.round(r.ms.voc) });
      }
      if (finalDetail.evicted_modules) {
        log('INFO', `[MM3] VRAM arbitration evicted ${finalDetail.evicted_modules} ACE module(s) (${(finalDetail.evicted_mb ?? 0).toFixed(0)} MB)`);
      }

      // MM3 Plank capture. Entirely non-fatal: the audio is already rendered
      // and saved by this point, so a failure here costs the plank, not the
      // song. /mm3/job?ar=1 is an ENGINE endpoint — it must go to
      // config.aceServer.url, not back into this Node server.
      if (req.get_ar_codes && r.ar_codes_available) {
        try {
          const arRes = await fetch(
            `${config.aceServer.url}/mm3/job?id=${encodeURIComponent(sub.job_id)}&ar=1`,
            { signal: AbortSignal.timeout(60_000) },
          );
          if (!arRes.ok) {
            log('WARNING', `[MM3 Plank] AR code fetch failed (HTTP ${arRes.status}) — nothing saved`);
          } else {
            const blob = Buffer.from(await arRes.arrayBuffer());
            const dir = mm3PlankDir();
            fs.mkdirSync(dir, { recursive: true });
            const plankId = uuidv4();
            fs.writeFileSync(path.join(dir, `${plankId}${MM3_PLANK_EXT}`), blob);
            fs.writeFileSync(path.join(dir, `${plankId}${MM3_PLANK_EXT}.json`), JSON.stringify({
              id: plankId,
              jobId: sub.job_id,
              created: new Date().toISOString(),
              caption: req.caption,
              lyrics: req.lyrics || '',
              duration: sub.duration,
              seed: sub.seed,
              frames: r.frames,
              sizeBytes: blob.length,
            }, null, 2));
            job.params.mm3LastPlankPath = `${plankId}${MM3_PLANK_EXT}`;
            log('INFO', `[MM3 Plank] Saved ${plankId}${MM3_PLANK_EXT} `
              + `(${(blob.length / 1024).toFixed(0)} KB, ${r.frames} frames)`);
          }
        } catch (plankErr: any) {
          log('WARNING', `[MM3 Plank] AR code save failed (non-fatal): ${plankErr?.message ?? plankErr}`);
        }
      }
    }

    // ── One song per take ──────────────────────────────────────────────────
    //
    // An ensemble render produced N different songs from this one prompt, and
    // each becomes its own library track: its own file, its own LRC, its own
    // post-processing, its own row. `takes` is what the ENGINE actually
    // rendered (it clamps to the checkpoint's row budget), never what was
    // asked for — a UI that requested 8 on a 4-take checkpoint must save 4
    // songs, not iterate over four missing ones.
    const nTakes = Math.max(1, Number(finalDetail?.takes ?? 1));
    const takeDetail: any[] = Array.isArray((finalDetail as any)?.take_detail)
      ? (finalDetail as any).take_detail : [];
    if (nTakes > 1) {
      log('INFO', `[MM3] Ensemble render: ${nTakes} takes -> ${nTakes} songs`);
    }
    const audioUrls: string[] = [];
    const songIds: string[] = [];

    for (let takeIdx = 0; takeIdx < nTakes; takeIdx++) {
      // Decimal string throughout — a uint64 seed does not survive float64, and
      // a song row that cannot reproduce itself is worse than no seed at all.
      const takeSeed = String(
        takeDetail[takeIdx]?.seed_str
        ?? (job.mm3TakeSeeds?.[takeIdx])
        ?? (sub.seed_str ? (BigInt(sub.seed_str) + BigInt(takeIdx)).toString() : String(sub.seed)),
      );
      // Take 0's bytes are ALSO on the shared /job?id=&result=1, but
      // /mm3/take serves every take through one uniform URL — so there is no
      // "first one is special" branch to get wrong.
      const takeRes = await fetch(
        `${config.aceServer.url}/mm3/take?id=${encodeURIComponent(sub.job_id)}&take=${takeIdx}`,
        { signal: AbortSignal.timeout(120_000) },
      );
      if (!takeRes.ok) {
        throw new Error(`Failed to fetch MiniMax-Music3 take ${takeIdx} (HTTP ${takeRes.status})`);
      }
      const audioBuffer = Buffer.from(await takeRes.arrayBuffer());
      if (audioBuffer.length === 0) {
        throw new Error(`MiniMax-Music3 returned an empty audio body for take ${takeIdx}`);
      }
      const contentType = takeRes.headers.get('content-type') || 'audio/wav';
      const ext = contentType.includes('wav') ? 'wav' : 'bin';
      const filename = `${uuidv4()}.${ext}`;
      const filepath = path.join(config.data.audioDir, filename);
      fs.writeFileSync(filepath, audioBuffer);
      const audioUrl = `/audio/${filename}`;

      // LRC lyric timestamps, if the engine produced them for THIS take — the
      // takes sing the same words at different moments, so a shared LRC would
      // be wrong for every take but one.
      try {
        const lrcRes = await fetch(
          `${config.aceServer.url}/mm3/take?id=${encodeURIComponent(sub.job_id)}&take=${takeIdx}&lrc=1`,
          { signal: AbortSignal.timeout(30_000) },
        );
        if (lrcRes.ok) {
          const lrcText = await lrcRes.text();
          if (lrcText.trim()) {
            const lrcPath = path.join(config.data.audioDir, filename.replace(/\.[^.]+$/, '.lrc'));
            fs.writeFileSync(lrcPath, lrcText);
            log('INFO', `[LRC] saved ${path.basename(lrcPath)} (${lrcText.length} bytes)`);
          }
        }
      } catch (lErr: any) {
        log('WARNING', `[LRC] failed to save: ${lErr?.message || lErr}`);
      }
      // Duration from the actual WAV header (MM3 renders 44.1 kHz — the parser
      // reads the real rate, so no 48 kHz assumption leaks in here).
      const measured = wavDurationSec(filepath);
      const duration = measured > 0
        ? Math.round(measured)
        : Math.round(finalDetail?.result?.duration_sec || sub.duration || 0);

      log('INFO', `[MM3] Saved ${filename} (${(audioBuffer.length / 1024).toFixed(0)} KB, ${duration}s)`);
      timing.push({ name: 'Save', ms: Math.round(performance.now() - saveStart) });

      // ── Persist ──
      const captionLine = req.caption.split('\n').map(s => s.trim()).find(s => s.length > 0) || '';
      const baseTitle: string = job.params.title || captionLine.substring(0, 60) || 'Untitled';
      // Takes are siblings, not versions, so they are numbered rather than
      // left silently identical — four rows with the same name in the library
      // is indistinguishable from a bug.
      const title: string = nTakes > 1 ? `${baseTitle} (take ${takeIdx + 1})` : baseTitle;
      const style: string = job.params.caption || job.params.style || '';
      const trackParams = {
        ...job.params,
        backend: 'minimax-m3',
        // THIS take's seed, not the base one. Take t was drawn from seed + t,
        // and stamping the base on every row would leave all but the first
        // song unreproducible.
        seed: takeSeed,
        ...(nTakes > 1 ? { mm3Take: takeIdx, mm3Takes: nTakes, mm3BaseSeed: sub.seed } : {}),
        duration: sub.duration,
        // The exact wire request, so a reproduce/A-B flow has the real inputs
        // rather than an ACE-shaped approximation of them.
        mm3Request: req,
        mm3: {
          max_frames: sub.max_frames,
          steps: sub.steps,
          cfg_flow: sub.cfg_flow,
          prompt_tokens: sub.prompt_tokens,
          instrumental: sub.instrumental,
          sample_rate: finalDetail?.result?.sample_rate ?? 44100,
        },
      };

      // ── Post-processing (the model-agnostic subset) ──────────────────────
      //
      // v1 skipped the whole chain because it is 48 kHz-minded and ACE-coupled.
      // Re-auditing it stage by stage, that is only true of some of it:
      //
      //   VST chain     — rate-agnostic. vst-host reads the rate from the WAV and
      //                   calls setupProcessing with it; the hardcoded 48000 is
      //                   in the GUI command, not the processing path.
      //   StableStep    — SA3 is NATIVELY 44.1 kHz (sa3-refine.h SA3_SR 44100)
      //                   and out_sr defaults to the input rate, so a 44.1 kHz
      //                   mix is a better match here than ACE's 48 kHz. The
      //                   48000 the old audit cited belongs to the vocal-split
      //                   branch only — disableStemSplit keeps us off it.
      //   Mastering     — reads and writes the file's own sample rate.
      //
      // Still excluded, deliberately: PP-VAE re-encode (round-trips through the
      // ACE VAE — wrong model and rate) and Spectral Lifter (tuned for the ACE
      // 48 kHz output). Those are model-coupled, not merely rate-coupled.
      let masteredUrl = '';
      try {
        const ppParams: PostProcessParams = {
          ...job.params,
          // Model-coupled stages: off regardless of what the UI persisted.
          ppVaeReencode: false,
          spectralLifterEnabled: false,
          // The vocal split MUST run for a vocal track: SA3 refines instrumental
          // only, and handing it a full mix makes it hallucinate replacement
          // vocals. SuperSep is a plain audio separator with no ACE coupling, and
          // at 44.1 kHz the whole StableStep path is resample-free (see the
          // native-rate branch in postProcessing.ts), so nothing here needs the
          // 48 kHz plumbing.
          instrumental: sub.instrumental,
        };
        // No "is anything actually on?" pre-check needed: the chain copies the
        // WAV, runs whatever is enabled, and if NOTHING ran it deletes the copy
        // and returns '' — so an all-off pass is a cheap no-op that also covers
        // the VST case (applyVstChain returns false when the chain is empty).
        if (ppParams.postProcessingEnabled !== false) {
          const ppStart = performance.now();
          const ppResult = await runPostProcessingChain(
            [audioUrl], ppParams, 1, job.id,
            log, (stage) => { job.stage = stage; },
          );
          masteredUrl = ppResult.masteredUrls?.[0] || '';
          const ppMs = Math.round(performance.now() - ppStart);
          if (masteredUrl) {
            log('INFO', `[MM3] Post-processing produced ${masteredUrl} in ${ppMs} ms`);
          }
        }
      } catch (ppErr: any) {
        // Non-fatal, exactly as on the ACE side: a failed post stage must never
        // lose the render that already succeeded.
        log('WARNING', `[MM3] Post-processing chain failed (non-fatal): ${ppErr?.message || ppErr}`);
      }

      // ── Whisper transcription ────────────────────────────────────────────
      //
      // Backend-agnostic: whisper-cli takes a file path and resamples internally,
      // so it needs nothing from ACE. Runs on the RAW render rather than the
      // post-processed copy — VST/mastering colour the mix without adding any
      // information, and StableStep remixes the original vocal stem untouched, so
      // the raw file is the cleanest input for ASR.
      //
      // This is also, for now, MM3's only route to word timings: the LRC path
      // reads ACE's DiT lyric cross-attention, which MM3's DiT does not have.
      if (job.params.whisperLyricsEnabled && !sub.instrumental) {
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
            log('INFO', '[Whisper] transcribing MiniMax-Music3 render...');
            const wr = await transcribeWithWhisper(filepath, req.lyrics || '', {
              model: job.params.whisperModel,
              language: job.params.whisperLanguage || 'auto',
              beamSize: job.params.whisperBeamSize || 5,
            });
            if (wr && wr.segments?.length > 0) {
              const lyricsJson = reconcileLyrics(wr, req.lyrics || '', job.params.whisperModel || 'auto', false);
              const lyricsPath = path.join(config.data.audioDir,
                                           filename.replace(/\.[^.]+$/, '.lyrics.json'));
              fs.writeFileSync(lyricsPath, JSON.stringify(lyricsJson, null, 2));
              const words = lyricsJson.lines.reduce((n: number, l: any) => n + l.words.length, 0);
              log('INFO', `[Whisper] saved ${path.basename(lyricsPath)} `
                + `(${lyricsJson.lines.length} lines, ${words} words, ${Math.round(performance.now() - wStart)} ms)`);
            } else {
              log('WARNING', '[Whisper] no segments returned');
            }
          }
        } catch (wErr: any) {
          // Non-fatal, like every other post stage: a failed transcription must
          // not cost the render.
          log('WARNING', `[Whisper] failed (non-fatal): ${wErr?.message || wErr}`);
        }
        timing.push({ name: 'Whisper', ms: Math.round(performance.now() - wStart) });
      }

      const songId = uuidv4();
      getDb().prepare(`
        INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                           duration, bpm, key_scale, time_signature, tags, dit_model,
                           generation_params, mastered_audio_url, latent_url, quality_scores,
                           noadapter_audio_url, backend)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        songId, job.userId, title, req.lyrics || '', style, req.caption,
        audioUrl, duration, 0, '', '',
        JSON.stringify([]), 'mm3', JSON.stringify(trackParams),
        masteredUrl, '', '',
        '', 'minimax-m3',
      );

      // ── Cover art (backend-agnostic: a text→image service, no engine coupling) ──
      if (job.params.coverArtEnabled) {
        const coverStart = performance.now();
        try {
          const { generateCoverArt, getCoverArtReadiness } = await import('../../coverArt/coverArtService.js');
          const readiness = getCoverArtReadiness();
          if (readiness.installed) {
            job.stage = 'Generating cover art...';
            job.progress = 97;
            await generateCoverArt({
              songId,
              title,
              style: style || captionLine,
              lyrics: req.lyrics || '',
              subject: job.params.coverArtSubject || job.params.subject || '',
            });
            log('INFO', `[CoverArt] Generated cover for song ${songId}`);
            if (job.params.coverArtSubject) {
              getDb().prepare('UPDATE songs SET cover_art_subject = ? WHERE id = ?')
                .run(job.params.coverArtSubject, songId);
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


      audioUrls.push(masteredUrl || audioUrl);
      songIds.push(songId);
    }
    // Every take is the same requested length unless one hit EOS early; report
    // the longest, which is the render as a whole.
    const duration = Math.max(
      1,
      ...(takeDetail.length
        ? takeDetail.map((d: any) => Math.round(Number(d?.duration_s) || 0))
        : [Math.round(finalDetail?.result?.duration_sec || sub.duration || 0)]),
    );

    const totalMs = Math.round(performance.now() - pipelineStart);
    timing.push({ name: 'TOTAL', ms: totalMs });

    job.status = 'succeeded';
    job.progress = 100;
    job.stage = 'Complete!';
    job.result = {
      audioUrls,
      songIds,
      duration,
      timing,
      totalMs,
    };

    log('INFO', `[Result] ${audioUrls.length} audio file(s) saved, ${songIds.length} song(s) created `
      + `(backend=minimax-m3${songIds.length > 1 ? `, ${songIds.length} takes` : ''})`);
    console.log(`[Generate] Job ${job.id} (minimax-m3) completed in ${(totalMs / 1000).toFixed(1)}s`);
    finishGenerationLog(job.id, 'mm3-text2music');

  } catch (err: any) {
    if (detailTimer) clearInterval(detailTimer);
    // `as string`: POST /api/generate/cancel/:id mutates job.status from
    // outside this function, which TS's control-flow narrowing can't see.
    if (err.message === 'Cancelled' || (job.status as string) === 'cancelled') {
      job.status = 'cancelled';
      job.stage = 'Cancelled';
      failGenerationLog(job.id, 'Cancelled by user', 'mm3-text2music');
    } else {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.stage = 'Failed';
      console.error(`[Generate] Job ${job.id} (minimax-m3) failed:`, err.message);
      failGenerationLog(job.id, err.message || 'Unknown error', 'mm3-text2music');
    }
  } finally {
    if (detailTimer) clearInterval(detailTimer);
  }
}

/** Best-effort MM3 eviction for the ACE path (plan §4.4). Cheap when cold,
 *  short-fused so a hung engine can never stall an ACE generation.
 *
 *  Keeps the engine's AR slot: this is eviction the USER did not ask for, and
 *  a trip to ACE and back should not cost a cached plan (~130 s of replan on a
 *  155 s song). Safe because the AR key pins the LM + depth model identity and
 *  a real model change drops the slot on its own path — see the note on
 *  mm3_handle_unload (mm3-server.h). */
export async function releaseMinimaxVramForAce(): Promise<void> {
  try {
    const r = await mm3Unload(3_000, /*keepArCache=*/true);
    if (r?.unloaded) {
      console.log(`[Generate] Freed MiniMax-Music3 VRAM before ACE work (${(r.freed_mb ?? 0).toFixed(0)} MB)`
        + `${r.ar_cache_kept ? ' — AR plan cache kept' : ''}`);
    }
  } catch { /* never blocks ACE */ }
}
