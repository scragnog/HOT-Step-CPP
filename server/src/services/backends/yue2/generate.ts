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
import { yue2Synth, yue2FinalDetail, type Yue2SynthRequest } from './client.js';
import { classifyYue2Score, type Yue2ScoreHealth } from './scoreHealth.js';
import { yue2PersistedSelection } from './index.js';
import { applyYue2StyleTemplate, type Yue2StyleTemplate } from './style.js';
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
}

/** Mirrors the capability manifest's core.duration (max 360, auto: true). */
const YUE2_MAX_DURATION_SEC = 360;

const DETAIL_POLL_MS = 1_500;

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
  ar: { path: string; trigger: string };
  nar: { path: string; trigger: string };
}

function yue2StyleForAdapter(
  caption: string, params: any,
): { style: string; notes: string[]; halves: Yue2StyleHalves; trainedCot: string } {
  const notes: string[] = [];
  const picked = yue2PersistedSelection().adapters;
  const halfOf = (p: string) => ({
    path: p,
    trigger: yue2AdapterTrigger(p).trigger,
  });
  const halves: Yue2StyleHalves = { ar: halfOf(picked.ar.path), nar: halfOf(picked.nar.path) };

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
  // No genre/BPM/key tail: at training time those came from the dataset's own
  // sidecars, and this backend exposes no such fields (capabilities: bpm
  // false, keyscale false). Whatever belongs in the tail the user writes into
  // the caption, which is exactly where it lands.
  const style = applyYue2StyleTemplate({ trigger, caption, template });

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

  const req: Yue2SynthRequest = {
    style,
    lyrics: lyrics || undefined,
    cot,
    ...(abc ? { abc } : {}),
    ...(cfg_scale !== undefined ? { cfg_scale } : {}),
    ode_steps,
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

export async function runYue2Generation(job: GenerationJob, deps: Yue2GenerationDeps): Promise<void> {
  const pipelineStart = performance.now();
  const timing: StageTiming[] = [];
  const timeoutMinutes: number | undefined = job.params.generationTimeoutMinutes;
  const log = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => logGeneration(job.id, level, msg);

  if (job.status === 'cancelled') return;

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
  if (req.lyrics) log('DEBUG', `[YuE2] Lyrics sent (${req.lyrics.length} chars):
${req.lyrics}`);

  let detailTimer: NodeJS.Timeout | undefined;

  try {
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
    let autoReplan: { attempts: Array<{ seed: number; verdict: string; reason: string }>; accepted: boolean } | undefined;
    if (job.params.yue2AutoReplan !== false && req.cot !== 'off' && !req.abc) {
      autoReplan = { attempts: [], accepted: false };
      const maxAttempts = 3;
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
        if (plan.health.verdict !== 'runaway') { autoReplan.accepted = true; break; }
      }
      if (autoReplan && chosen) {
        req.abc = chosen.abc;
        req.seed = chosen.seed;
        if (!autoReplan.accepted) log('WARNING', `[YuE2] Every plan attempt was a runaway; rendering the last one (seed ${chosen.seed})`);
        else if (autoReplan.attempts.length > 1) log('INFO', `[YuE2] Runaway plan replaced after ${autoReplan.attempts.length} attempts`);
      }
    }

    // ── Submit ──
    job.status = 'running';
    job.stage = 'YuE2: submitting...';
    job.progress = 2;
    const submitStart = performance.now();
    const sub = await yue2Synth(req);
    deps.attempt.effective.seed = sub.seed_str ?? sub.seed;
    job.aceJobId = sub.job_id;   // standard /job id — /cancel/:id reaches it unchanged
    job.params.seed = sub.seed;
    job.params.randomSeed = false;
    log('INFO', `[YuE2] Job ${sub.job_id} submitted — cot=${req.cot}, ode_steps=${sub.ode_steps ?? req.ode_steps}, `
      + `cfg=${sub.cfg_scale ?? req.cfg_scale ?? '(default)'}, vae=${sub.vae_variant ?? req.vae_variant}, `
      + `seed ${sub.seed}${sub.instrumental ? ', instrumental' : ''}`);

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
          job.stage = stage;
          job.progress = progress;
          job.acePhase = phase;
          if (total > 0) job.acePhaseProgress = `step ${step}/${total}`;
        } catch { /* transient poll failure — pollUntilDone owns the real watchdog */ }
      })();
    }, DETAIL_POLL_MS);
    detailTimer.unref?.();

    await deps.pollUntilDone(sub.job_id, job, deps.signal, timeoutMinutes);
    clearInterval(detailTimer);
    detailTimer = undefined;
    timing.push({ name: 'YuE2 Generate', ms: Math.round(performance.now() - submitStart) });

    // ── Result ──
    job.stage = 'YuE2: saving audio...';
    job.progress = 95;
    const saveStart = performance.now();

    // Additive end_reason/stage_end_reasons/artifacts — see the ASSUMPTION
    // note in client.ts. Every field here is optional; an engine build that
    // doesn't populate them yet just yields an empty object.
    const finalDetail = await yue2FinalDetail(sub.job_id);
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

    const audioRes = await aceClient.getJobResult(sub.job_id);
    const audioBuffer = Buffer.from(await audioRes.arrayBuffer());
    if (audioBuffer.length === 0) {
      throw new Error('YuE2 returned an empty audio body');
    }
    const contentType = audioRes.headers.get('content-type') || 'audio/wav';
    const ext = contentType.includes('wav') ? 'wav' : 'bin';
    const filename = `${uuidv4()}.${ext}`;
    const filepath = path.join(config.data.audioDir, filename);
    fs.writeFileSync(filepath, audioBuffer);
    const audioUrl = `/audio/${filename}`;

    // Artifacts: score.abc (decoded ABC text) and semantic ids, saved beside
    // the WAV as sidecars when the engine returned them. Never fatal.
    if (finalDetail.abc) {
      try {
        fs.writeFileSync(path.join(config.data.audioDir, filename.replace(/\.[^.]+$/, '.score.abc')), finalDetail.abc);
      } catch (e: any) {
        log('WARNING', `[YuE2] Failed to save score.abc sidecar (non-fatal): ${e?.message ?? e}`);
      }
    }
    if (finalDetail.semantic_ids?.length) {
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
    const duration = measured > 0 ? Math.round(measured) : 0;

    log('INFO', `[YuE2] Saved ${filename} (${(audioBuffer.length / 1024).toFixed(0)} KB, ${duration}s)`);
    timing.push({ name: 'Save', ms: Math.round(performance.now() - saveStart) });

    // ── Persist ──
    const captionLine = yue2CaptionToStyle(caption);
    const title: string = job.params.title || captionLine.substring(0, 60) || 'Untitled';
    const style: string = job.params.caption || job.params.style || '';
    const trackParams = {
      ...job.params,
      backend: 'yue2',
      seed: sub.seed_str ?? sub.seed,
      yue2Request: req,
      yue2: {
        ode_steps: sub.ode_steps ?? req.ode_steps,
        cfg_scale: sub.cfg_scale ?? req.cfg_scale,
        vae_variant: sub.vae_variant ?? req.vae_variant,
        instrumental: sub.instrumental,
        end_reason: finalDetail.end_reason,
        stage_end_reasons: finalDetail.stage_end_reasons,
        duration_s: measured > 0 ? Math.round(measured * 10) / 10 : undefined,
        abc_supplied: !!req.abc,
        ...(scoreHealth ? { score_health: scoreHealth } : {}),
        ...(autoReplan ? { auto_replan: autoReplan } : {}),
      },
    };

    // ── Post-processing (the model-agnostic chain, in full) ────────────────
    // YuE2 renders native 48 kHz stereo, so unlike MM3's v1 (44.1 kHz,
    // whole-chain skip) nothing here needs a rate audit. ppVaeReencode and the
    // Spectral Lifter stay excluded regardless — both are ACE-VAE-coupled,
    // not rate-coupled.
    let masteredUrl = '';
    try {
      const ppParams: PostProcessParams = {
        ...job.params,
        ppVaeReencode: false,
        spectralLifterEnabled: false,
        instrumental: sub.instrumental,
        stableStepCaptions: [captionLine],
      };
      if (ppParams.postProcessingEnabled !== false) {
        const ppStart = performance.now();
        const ppResult = await runPostProcessingChain(
          [audioUrl], ppParams, 1, job.id,
          log, (stage) => { job.stage = stage; },
        );
        masteredUrl = ppResult.masteredUrls?.[0] || '';
        const ppMs = Math.round(performance.now() - ppStart);
        if (masteredUrl) log('INFO', `[YuE2] Post-processing produced ${masteredUrl} in ${ppMs} ms`);
        timing.push({ name: 'Post-processing', ms: ppMs });
      }
    } catch (ppErr: any) {
      log('WARNING', `[YuE2] Post-processing chain failed (non-fatal): ${ppErr?.message || ppErr}`);
    }

    // ── Whisper transcription ────────────────────────────────────────────
    // Backend-agnostic (whisper-cli takes a file path and resamples
    // internally). This is YuE2's only route to word timings in v1 — its DiT
    // has no lyric-alignment head (capabilities().features.lyricTimestamps =
    // false), so there is no LRC path the way ACE/MM3 have one.
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
          log('INFO', '[Whisper] transcribing YuE2 render...');
          const wr = await transcribeWithWhisper(filepath, req.lyrics || '', {
            model: job.params.whisperModel,
            language: job.params.whisperLanguage || 'auto',
            beamSize: job.params.whisperBeamSize || 5,
          });
          if (wr && wr.segments?.length > 0) {
            const lyricsJson = reconcileLyrics(wr, req.lyrics || '', job.params.whisperModel || 'auto', false);
            const lyricsPath = path.join(config.data.audioDir, filename.replace(/\.[^.]+$/, '.lyrics.json'));
            fs.writeFileSync(lyricsPath, JSON.stringify(lyricsJson, null, 2));
            const words = lyricsJson.lines.reduce((n: number, l: any) => n + l.words.length, 0);
            log('INFO', `[Whisper] saved ${path.basename(lyricsPath)} `
              + `(${lyricsJson.lines.length} lines, ${words} words, ${Math.round(performance.now() - wStart)} ms)`);
          } else {
            log('WARNING', '[Whisper] no segments returned');
          }
        }
      } catch (wErr: any) {
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
      songId, job.userId, title, req.lyrics || '', style, req.style,
      audioUrl, duration, 0, '', '',
      JSON.stringify([]), 'yue2', JSON.stringify(trackParams),
      masteredUrl, '', '',
      '', 'yue2',
    );

    // ── Cover art (backend-agnostic) ──
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

    const totalMs = Math.round(performance.now() - pipelineStart);
    timing.push({ name: 'TOTAL', ms: totalMs });

    job.status = 'succeeded';
    job.progress = 100;
    job.stage = 'Complete!';
    job.result = {
      audioUrls: [masteredUrl || audioUrl],
      songIds: [songId],
      duration,
      timing,
      totalMs,
    };

    log('INFO', `[Result] 1 audio file saved, 1 song created (backend=yue2)`);
    console.log(`[Generate] Job ${job.id} (yue2) completed in ${(totalMs / 1000).toFixed(1)}s`);
    finishGenerationLog(job.id, 'yue2-text2music');

  } catch (err: any) {
    if (detailTimer) clearInterval(detailTimer);
    // `as string`: POST /api/generate/cancel/:id mutates job.status from
    // outside this function, which TS's control-flow narrowing can't see.
    if (err.message === 'Cancelled' || (job.status as string) === 'cancelled') {
      job.status = 'cancelled';
      job.stage = 'Cancelled';
      failGenerationLog(job.id, 'Cancelled by user', 'yue2-text2music');
    } else {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.stage = 'Failed';
      console.error(`[Generate] Job ${job.id} (yue2) failed:`, err.message);
      failGenerationLog(job.id, err.message || 'Unknown error', 'yue2-text2music');
    }
  } finally {
    if (detailTimer) clearInterval(detailTimer);
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
  health: Yue2ScoreHealth;
  notes: string[];
}

export async function runYue2PlanPreview(params: any, signal?: AbortSignal): Promise<Yue2PlanPreview> {
  const { req, notes } = mapYue2Params(params);
  if (!req.style.trim()) throw new Error('YuE2 needs a caption — the Style Description field is empty');
  if (req.cot === 'off') throw new Error('Score preview needs Chain of Thought "melody" or "full" — cot=off has no lead sheet to preview');
  const planReq: Yue2SynthRequest = { ...req, plan_only: true };
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
  return { abc, seed: sub.seed, end_reason, health: classifyYue2Score(abc, end_reason), notes };
}
