import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, Check, Download, Loader2, Play, RotateCcw, Upload, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { YUE2_JOINT_PRESETS_KEY, type Yue2JointPreset } from './yue2JointPresets';
import { TrainingChart } from './TrainingChart';
import { StyledSelect } from '../shared/StyledSelect';
import { Yue2LadderReview } from './Yue2LadderReview';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import {
  cancelJob,
  captionMissingYue2,
  clearPreparedData,
  getPreparedData,
  getJob,
  getYue2AitkPrepare,
  listYue2AitkRuns,
  linkYue2JointCheckpointPreset,
  listYue2JointPreviews,
  listJobs,
  jobStreamUrl,
  startYue2AitkPrepare,
  startYue2JointTrain,
  type Yue2AitkPrepareRequest,
  type TrainingJobSummary,
  type TrainingMetricEvent,
  type TrainingStreamEvent,
  type Yue2AitkCheckpointRecord,
  type Yue2AitkRunRecord,
  type Yue2JointTrainRequest,
  type Yue2JointBaseInfo,
  type Yue2JointPreviewOptions,
  type Yue2JointPreviewRecord,
  type PreparedCache,
} from '../../services/trainingApi';
import { useBackendStore } from '../../stores/backendStore';
import { yue2AdapterHalfBytes, formatMB } from '../../utils/yue2AdapterSize';
import { useTrainingStore } from '../../stores/trainingStore';
import { descentRate, formatDurationMs } from '../../utils/trainingEta';

const JOB_KEY = 'hs-yue2-aitk-job:';
const FORM_KEY = 'hs-yue2-aitk-form:';
const PREP_KEY = 'hs-yue2-aitk-prepare:';
const METRIC_CAP = 2000;
// `loss` is absent once the planner is frozen: the trainer then reports only
// the decoder's terms and the composite has no AR part. Those steps still
// count for progress, pace and the gradient norm.
type JointStepPoint = { step: number; loss?: number; ep: number; arKl?: number; narMse?: number; narRecon?: number; frozen?: boolean; gradNorm?: number; stepMs?: number; elapsedMs?: number; ma5?: number; ma20?: number };
type JointMilestone = { epoch: number; loss: number; path: string };
function jointLossRate(points: JointStepPoint[]): number | null {
  const means = points.map(p => p.ma20).filter((v): v is number => typeof v === 'number');
  return means.length >= 9 ? descentRate(means) : null;
}
/** Steps in the KL trend fit; the engine's kKlTrendWindow (yue2-aitk-runtime.cpp). */
const KL_TREND_WINDOW = 30;
/** What the engine's KL stop reads at the last of `kls`: a least-squares line
 *  through the last 30 read at its end (trend), or the 20-step mean. null until
 *  the window is full, as in the engine. `slope` is per step (trend only). */
function klStopReading(kls: number[], mode: 'mean' | 'trend'): { value: number; slope?: number } | null {
  const n = mode === 'trend' ? KL_TREND_WINDOW : 20;
  if (kls.length < n) return null;
  const v = kls.slice(-n);
  const ym = v.reduce((s, x) => s + x, 0) / n;
  if (mode === 'mean') return { value: ym };
  const xm = (n - 1) / 2;
  let sxy = 0, sxx = 0;
  v.forEach((y, i) => { sxy += (i - xm) * (y - ym); sxx += (i - xm) ** 2; });
  const slope = sxy / sxx;
  return { value: ym + slope * (n - 1 - xm), slope };
}
function jointEta(points: JointStepPoint[], form: Yue2JointTrainRequest): string {
  const last = points[points.length - 1];
  const durations = points.map(p => p.stepMs).filter((ms): ms is number => typeof ms === 'number' && ms > 0).slice(-20);
  if (!last || !durations.length) return '';
  const pace = durations.reduce((sum, ms) => sum + ms, 0) / durations.length;
  const remaining = Math.max(0, form.steps - last.step);
  if (form.stopMode === 'kl' && form.targetKl && form.targetKl > 0) {
    const all = points.map(p => p.arKl).filter((v): v is number => typeof v === 'number');
    const reading = klStopReading(all, form.targetKlMode ?? 'mean');
    if (!reading) return `AR KL warming up · cap ${formatDurationMs(remaining * pace)}`;
    const mean = reading.value;
    if (mean >= form.targetKl) return `KL target reached · cap ${formatDurationMs(remaining * pace)}`;
    const older = points.slice(-40, -20).map(p => p.arKl).filter((v): v is number => typeof v === 'number');
    const rate = reading.slope ?? (older.length === 20 ? (mean - older.reduce((s, v) => s + v, 0) / 20) / 20 : 0);
    if (!(rate > 0)) return `AR KL ${mean.toFixed(2)} of ${form.targetKl} · cap ${formatDurationMs(remaining * pace)}`;
    const stepsToTarget = (form.targetKl - mean) / rate;
    return stepsToTarget > remaining
      ? `KL ${mean.toFixed(2)} · target unlikely before cap · cap ${formatDurationMs(remaining * pace)}`
      : `KL ${mean.toFixed(2)} · target ETA ${formatDurationMs(Math.max(1, stepsToTarget) * pace)}`;
  }
  if (form.stopMode !== 'loss' || !(form.targetLoss && form.targetLoss > 0))
    return `cap ETA ${formatDurationMs(remaining * pace)}`;
  const current = last.ma20;
  const rate = jointLossRate(points);
  if (rate === null || current === undefined) return `target ETA estimating · cap ${formatDurationMs(remaining * pace)}`;
  if (current <= form.targetLoss) return `target reached · cap ${formatDurationMs(remaining * pace)}`;
  if (!(rate > 0)) return `target trend stalled · cap ${formatDurationMs(remaining * pace)}`;
  const stepsToTarget = (current - form.targetLoss) / rate;
  if (stepsToTarget > remaining) return `target unlikely before cap · cap ${formatDurationMs(remaining * pace)}`;
  return `target ETA ${formatDurationMs(Math.max(1, stepsToTarget) * pace)}`;
}
/** A named snapshot of the training settings. Per-run and per-machine values
 *  (dataset/checkpoint/output paths, resume record) are deliberately not part
 *  of a preset: a preset answers "how do I train", never "against which run". */
const PRESET_EXCLUDED_KEYS: ReadonlySet<keyof Yue2JointTrainRequest> = new Set([
  'trainingMethod', 'autoPrepare', 'preparation', 'checkpoint', 'dataset', 'output', 'resume', 'alignmentEnabled',
]);
function snapshotPresetSettings(form: Yue2JointTrainRequest, lyricTiming: boolean): Partial<Yue2JointTrainRequest> {
  const settings: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(form)) {
    if (PRESET_EXCLUDED_KEYS.has(key as keyof Yue2JointTrainRequest) || value === undefined) continue;
    settings[key] = value;
  }
  settings.lyricTiming = lyricTiming;
  return settings as Partial<Yue2JointTrainRequest>;
}
// 2026-09-27: every checkpoint of a run is a rung of its ladder, rendered in
// parallel with training (everySteps 0 = no pauses): one 300 s draft take
// (12 decoder steps), the same settings the Refine tab's ladders used.
const LADDER_PREVIEW: Yue2JointPreviewOptions = { enabled: true, everySteps: 0, parallel: true, takes: 1, odeSteps: 12, narCacheRatio: 0, seconds: 300, seed: 424242,
  previewMaxFrames: 7500, baseline: false, control: false };
function defaultPreview(_everySteps: number): Yue2JointPreviewOptions { return { ...LADDER_PREVIEW }; }
const DEFAULT_FORM: Yue2JointTrainRequest = {
  trainingMethod: 'aitk', checkpoint: '', dataset: '', output: '',
  // NAR budget recipe (2026-09-21): the run still ends on the planner's KL to
  // base, but the decoder gets twice the learning rate on the way there.
  // lr 2e-4 with the planner at 0.3x leaves the planner's ABSOLUTE rate at
  // 6e-5 — the same 1e-4 x 0.6 Recipe A used — so the AR walks an unchanged
  // path to KL 1.4 while the decoder, which is where likeness lives, moves
  // twice as far per step. The two halves are independent: the NAR's
  // conditioning prefix is a detached recompute and NAR backward only ever
  // uploads to the NAR adapters (yue2-aitk-joint-step.h), so the decoder's
  // rate cannot perturb the planner's KL. Gen-time NAR strength 2.0 was
  // beating 1.0 by ear; this is that, trained in rather than dialled in.
  // 750 is a cap, not a target — an artist that has not reached KL 1.4 by
  // then is not going to.
  //
  // Prodigy by default (2026-09-21, later the same day): once its weight
  // update was bias-corrected (lm-optim.h) it beat this AdamW recipe by ear
  // AND by clock on the same album — KL 1.4 in 214 steps against AdamW's
  // 308, d settling at ~1e-3 where AdamW had been told 2e-4. `lr` is ignored
  // under Prodigy (base_lr is gamma = 1.0); the planner scale still applies,
  // now on top of d. The pre-fix Prodigy history — "learns too fast,
  // corrupts the AR" — was the missing bias correction, not the optimizer.
  // 2026-09-27: the base-matched recipe replaced all of the above (Rob's ear
  // test on Dookie: "better in every way"). Balanced = 100 updates of 4 songs.
  // The server forces the recipe's fixed parts whatever this form says
  // (applyBaseMatchedRecipe); the lines below that name tuned-only knobs are
  // history the migration in readStoredForm clears from stored forms.
  // device '' = the server picks this build's first GPU (CUDA0, Vulkan0, MTL0).
  method: 'base-matched', steps: 100, saveEvery: 10, gradAccum: 4, seed: 42, device: '', base: '', lyricTiming: false, cursorWeight: 0,
  optimizer: 'adamw-lm', cautious: false, prodigyD0: 1e-6, muonLrScale: 1, muonNsSteps: 5,
  // LoKr 64/4/256 (2026-09-22, Rob's pick after the size sweep): scale
  // alpha/dim = 4, all four sites factorized, ~106 MB for the AR+NAR pair.
  // rank stays 64 so switching back to LoRA restores the LoRA recipe.
  rank: 64, alpha: 256, adapterType: 'lokr', lokrDim: 64, lokrFactor: 4,
  // LoKr under Prodigy (Rob's ear tests, 2026-09-22): LoRA's KL 1.4 overcooks
  // both halves; KL 1.0 with the planner at 0.6 and the decoder at 1.0 is the
  // tested recipe. LoRA keeps KL 1.4 with the planner at 0.3 (LORA_STOP).
  // Trend (2026-09-22): the 20-step mean lagged the KL trend by ~10 steps.
  // Presets (2026-09-24, Rob): Balanced is the default. The KL target stops
  // the planner; the decoder trains on to the step cap.
  stopMode: 'steps',
  // Planner freeze (2026-09-23): the KL target used to end the whole run, so
  // the decoder, which carries timbre, stopped wherever the planner did. The
  // checkpoint-mix ear test (AR200+NAR150 over AR200+NAR100) said the decoder
  // wants more. Now the planner freezes at its KL and the decoder trains 100
  // more steps; the KL checkpoint is still saved, so the old stop point is
  // one of the rungs. Caption dropout 0.5: the measured recipe, so a new
  // caption lands on the artist rather than beside one memorised track.
  narExtraSteps: 0, captionDropout: 0,
  autoRefine: false,
  // The ladder's previews render while training runs, so the engine stays up.
  preview: LADDER_PREVIEW, stopEngine: false,
};
// Tuned-recipe knobs the server forces under base-matched; cleared from
// stored forms so a saved value from before 2026-09-27 cannot linger.
const TUNED_KEYS = ['targetKl', 'targetLoss', 'targetKlMode', 'narExtraSteps', 'klWeight', 'captionDropout', 'plannerLrScale', 'narLrScale',
  'spikeFactor', 'spikeStop', 'spikeStopWindow', 'reconStop', 'reconStopWindow', 'reconTarget', 'lrSchedule', 'lrFloor', 'lrDecaySteps',
  'lrDecayShape', 'klOvershootMargin', 'lrCycleSteps', 'lrCycleMult', 'klCheckpointEvery', 'refineWarmup', 'rungAdaptiveLr'] as const;
const LORA_STOP = { targetKl: 1.4, plannerLrScale: 0.3, narLrScale: undefined };
const LOKR_STOP = { targetKl: 1.0, plannerLrScale: 0.6, narLrScale: 1 };
// Presets (2026-09-27 ear test on a full album). All three are the
// base-matched recipe (the server applies its fixed parts); they differ in
// updates and songs per update. Fast and Balanced are the 4-song run
// (likeness 4.3 at update 50, 5.0 at 100, 28 s an update on the 5090);
// Thorough is the 8-song run (5.0 from update 90, 60 s an update, 3.3 h).
const PRESETS = [
  { key: 'fast', label: 'Fast', steps: 50, gradAccum: 4 },
  { key: 'balanced', label: 'Balanced', steps: 100, gradAccum: 4 },
  { key: 'thorough', label: 'Thorough', steps: 200, gradAccum: 8 },
] as const;
const presetValues = (p: typeof PRESETS[number]) => ({ stopMode: 'steps' as const, steps: p.steps, gradAccum: p.gradAccum, saveEvery: 10 });
const activePreset = (f: Yue2JointTrainRequest) => PRESETS.find(p => f.steps === p.steps && (f.gradAccum ?? 4) === p.gradAccum)?.key;
/** Run time scales with songs processed (updates × songs per update), so the
 *  presets show their cost relative to Balanced rather than wall-clock times
 *  that only hold for one GPU and one dataset. */
const presetTime = (p: { steps: number; gradAccum: number }) => {
  const r = (p.steps * p.gradAccum) / (100 * 4);
  return r === 0.5 ? '½×' : `${Number(r.toFixed(1))}×`;
};
/** Mirror of BASE_MATCHED_DEFAULTS in server/src/services/training/yue2JointTrainRunner.ts,
 *  shown as each blank field's placeholder. The server fills blanks from its own copy. */
const BASE_MATCHED_DEFAULTS = {
  lr: 1e-4, weightDecay: 0.1, beta2: 0.95, abcDropout: 0.5, narCropFrames: 0, arLossWeight: 0.25,
  gradAccum: 4, textDropout: 0.1, lyricDropout: 0.1, bothDropout: 0.1, warmupFraction: 0.03,
} as const;
type PrepareForm = Yue2AitkPrepareRequest;



function readStored<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) as T : fallback;
  } catch { return fallback; }
}

/** A form saved before a field existed is missing that key, so loading merges
 *  the stored values over the defaults: the user's saved values win, new
 *  fields fill in with their defaults. The persistence effect writes the
 *  merged form back, so this self-heals on the first load after an upgrade. */
function readStoredForm(datasetId: string): Yue2JointTrainRequest {
  const stored = readStored<Partial<Yue2JointTrainRequest>>(`${FORM_KEY}${datasetId}`, {});
  // CUDA0 was the stored default before Vulkan training (2026-09-28); on a
  // Vulkan build it would name a device that does not exist.
  if (stored.device === 'CUDA0') stored.device = '';
  const migration = `${FORM_KEY}${datasetId}:defaults-64`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(migration)) {
    // Move values that match the former defaults; retain deliberate custom values.
    if (stored.steps === 3000) stored.steps = 400;
    if (stored.saveEvery === 250) stored.saveEvery = 50;
    if (stored.rank === 32) stored.rank = 64;
    if (stored.alpha === 32) stored.alpha = 64;
    window.localStorage.setItem(migration, '1');
  }
  // Recipe A (2026-09-20): values still sitting on the old defaults move to
  // the new ones; anything the user changed on purpose stays.
  const recipeA = `${FORM_KEY}${datasetId}:defaults-recipe-a`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(recipeA)) {
    if (stored.rank === 64) stored.rank = 32;
    if (stored.alpha === 64) stored.alpha = 32;
    if (stored.steps === 400) stored.steps = 500;
    if (stored.optimizer === 'prodigy') stored.optimizer = 'adamw';
    if (stored.stopMode === undefined || stored.stopMode === 'steps') { stored.stopMode = 'kl'; stored.targetKl = 1.4; }
    if (stored.lr === undefined) stored.lr = 1e-4;
    if (stored.plannerLrScale === undefined) stored.plannerLrScale = 0.6;
    if (stored.preview?.enabled) stored.preview = { ...stored.preview, enabled: false };
    window.localStorage.setItem(recipeA, '1');
  }
  // NAR budget (2026-09-21): forms still sitting on Recipe A's values move to
  // the new defaults; anything the user set deliberately stays put.
  const narBudget = `${FORM_KEY}${datasetId}:defaults-nar-budget`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(narBudget)) {
    if (stored.rank === 32) stored.rank = 64;
    if (stored.alpha === 32) stored.alpha = 64;
    if (stored.steps === 500) stored.steps = 750;
    if (stored.lr === 1e-4) stored.lr = 2e-4;
    if (stored.plannerLrScale === 0.6) stored.plannerLrScale = 0.3;
    window.localStorage.setItem(narBudget, '1');
  }
  // Prodigy default (2026-09-21): Recipe A had moved everyone to adamw, so a
  // stored adamw is the old default, not a choice, and moves with it.
  const prodigy = `${FORM_KEY}${datasetId}:defaults-prodigy`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(prodigy)) {
    if (stored.optimizer === 'adamw') stored.optimizer = 'prodigy';
    window.localStorage.setItem(prodigy, '1');
  }
  // LoKr default (2026-09-22): a form that never chose an adapter type was on
  // the LoRA default, so it moves; its alpha moves only if it was still the
  // LoRA default 64. A form that picked LoRA or LoKr deliberately stays.
  const lokrDefault = `${FORM_KEY}${datasetId}:defaults-lokr-64-4-256`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(lokrDefault)) {
    if (stored.adapterType === undefined) {
      stored.adapterType = 'lokr'; stored.lokrDim = 64; stored.lokrFactor = 4;
      if (stored.alpha === undefined || stored.alpha === 64) stored.alpha = 256;
    }
    window.localStorage.setItem(lokrDefault, '1');
  }
  // LoKr stop recipe (2026-09-22): a LoKr form still on LoRA's KL 1.4 moves to
  // the LoKr pair; a deliberately edited target stays.
  const lokrStop = `${FORM_KEY}${datasetId}:defaults-lokr-stop`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(lokrStop)) {
    if (stored.adapterType === 'lokr') {
      if (stored.targetKl === undefined || stored.targetKl === 1.4) stored.targetKl = LOKR_STOP.targetKl;
      if (stored.narLrScale === undefined) stored.narLrScale = LOKR_STOP.narLrScale;
    }
    window.localStorage.setItem(lokrStop, '1');
  }
  // LoKr recipe 2 (2026-09-22): planner 0.3 -> 0.6, decoder 0.5 -> 1.0, for
  // LoKr forms still on the first recipe's values.
  const lokrRecipe2 = `${FORM_KEY}${datasetId}:defaults-lokr-recipe-2`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(lokrRecipe2)) {
    if (stored.adapterType === 'lokr') {
      if (stored.plannerLrScale === undefined || stored.plannerLrScale === 0.3) stored.plannerLrScale = LOKR_STOP.plannerLrScale;
      if (stored.narLrScale === undefined || stored.narLrScale === 0.5) stored.narLrScale = LOKR_STOP.narLrScale;
    }
    window.localStorage.setItem(lokrRecipe2, '1');
  }
  // LoKr recipe 3 (2026-09-22): target KL 0.9 -> 1.0.
  const lokrRecipe3 = `${FORM_KEY}${datasetId}:defaults-lokr-recipe-3`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(lokrRecipe3)) {
    if (stored.adapterType === 'lokr' && stored.targetKl === 0.9) stored.targetKl = LOKR_STOP.targetKl;
    window.localStorage.setItem(lokrRecipe3, '1');
  }
  // Preview 90 s (2026-09-22): a form still on the old 40 s default moves.
  const preview90 = `${FORM_KEY}${datasetId}:defaults-preview-90`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(preview90)) {
    if (stored.preview && stored.preview.seconds === 40) stored.preview = { ...stored.preview, seconds: 90, previewMaxFrames: 2250 };
    window.localStorage.setItem(preview90, '1');
  }
  // 2026-09-23 defaults: save every 25, cautious on, LoKr KL 1.1. Values still
  // on the previous defaults move; deliberate ones stay.
  const d0923 = `${FORM_KEY}${datasetId}:defaults-2026-09-23`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(d0923)) {
    if (stored.saveEvery === 50) stored.saveEvery = 25;
    if (stored.optimizer !== 'adamw' && (stored.cautious === undefined || stored.cautious === false)) stored.cautious = true;
    if (stored.adapterType === 'lokr' && stored.targetKl === 1.0) stored.targetKl = LOKR_STOP.targetKl;
    window.localStorage.setItem(d0923, '1');
  }
  // Planner freeze + caption dropout (2026-09-23): fields a form never set
  // take the new defaults; deliberate values stay.
  const freeze = `${FORM_KEY}${datasetId}:defaults-planner-freeze`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(freeze)) {
    if (stored.narExtraSteps === undefined) stored.narExtraSteps = DEFAULT_FORM.narExtraSteps;
    if (stored.captionDropout === undefined) stored.captionDropout = DEFAULT_FORM.captionDropout;
    window.localStorage.setItem(freeze, '1');
  }
  // Presets (2026-09-24): stored forms move to Balanced once.
  // 2026-09-24: presets end at the KL; the decoder trains on during refinement.
  const klEnd = `${FORM_KEY}${datasetId}:defaults-kl-end`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(klEnd)) {
    if (stored.stopMode === 'kl' && stored.narExtraSteps === stored.steps) stored.narExtraSteps = 0;
    window.localStorage.setItem(klEnd, '1');
  }
  const presets = `${FORM_KEY}${datasetId}:defaults-presets`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(presets)) {
    Object.assign(stored, presetValues(PRESETS[1]));
    window.localStorage.setItem(presets, '1');
  }
  const autoRefine = `${FORM_KEY}${datasetId}:defaults-auto-refine`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(autoRefine)) {
    if (stored.autoRefine === undefined) stored.autoRefine = DEFAULT_FORM.autoRefine;
    window.localStorage.setItem(autoRefine, '1');
  }
  const recon = `${FORM_KEY}${datasetId}:defaults-recon-stop`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(recon)) {
    if (stored.reconStop === undefined) { stored.reconStop = DEFAULT_FORM.reconStop; stored.reconStopWindow = DEFAULT_FORM.reconStopWindow; }
    // The old two-point knee shipped with 3; the fitted knee needs 10 points.
    if (stored.reconStopWindow === 3) stored.reconStopWindow = DEFAULT_FORM.reconStopWindow;
    window.localStorage.setItem(recon, '1');
  }
  const spike = `${FORM_KEY}${datasetId}:defaults-spike-guard`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(spike)) {
    if (stored.spikeFactor === undefined) { stored.spikeFactor = DEFAULT_FORM.spikeFactor; stored.spikeStop = DEFAULT_FORM.spikeStop; stored.spikeStopWindow = DEFAULT_FORM.spikeStopWindow; }
    window.localStorage.setItem(spike, '1');
  }
  // 2026-09-25 defaults: wsd schedule, AR KL target 1.0. A form that never
  // touched the schedule, or is still sitting on the old KL 1.2, moves;
  // anything set deliberately stays.
  const d0925 = `${FORM_KEY}${datasetId}:defaults-2026-09-25`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(d0925)) {
    if (stored.lrSchedule === undefined) stored.lrSchedule = 'wsd';
    if (stored.targetKl === 1.2) stored.targetKl = 1.0;
    window.localStorage.setItem(d0925, '1');
  }
  // 2026-09-25 (Rob): checkpoint previews are the Refine tab's job; a primary
  // run renders none by default. Turned off once; tick it again to keep it.
  const previewOff = `${FORM_KEY}${datasetId}:defaults-2026-09-25-preview-off`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(previewOff)) {
    if (stored.preview?.enabled) stored.preview = { ...stored.preview, enabled: false };
    window.localStorage.setItem(previewOff, '1');
  }
  // 2026-09-27 (Rob): the base-matched recipe replaces the tuned one. Every
  // stored form moves to it once, on the Balanced preset; the tuned-only
  // knobs are cleared (the server forces them anyway).
  const baseMatched = `${FORM_KEY}${datasetId}:defaults-base-matched-2026-09-27`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(baseMatched)) {
    for (const key of TUNED_KEYS) delete (stored as Record<string, unknown>)[key];
    stored.method = 'base-matched'; stored.stopMode = 'steps'; stored.optimizer = 'adamw-lm'; stored.cautious = false;
    stored.autoRefine = false; stored.cursorWeight = 0;
    Object.assign(stored, presetValues(PRESETS[1]));
    window.localStorage.setItem(baseMatched, '1');
  }
  // 2026-09-27 (Rob): the run's own ladder is previewed and scored on this
  // page, so previews are on and the engine stays up. Turned on once.
  const ladderPreviews = `${FORM_KEY}${datasetId}:defaults-ladder-previews-2026-09-27`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(ladderPreviews)) {
    stored.preview = { ...LADDER_PREVIEW, ...(stored.preview?.caption ? { caption: stored.preview.caption } : {}), ...(stored.preview?.lyrics ? { lyrics: stored.preview.lyrics } : {}) };
    stored.stopEngine = false;
    window.localStorage.setItem(ladderPreviews, '1');
  }
  return { ...DEFAULT_FORM, ...stored, method: 'base-matched' };
}
function writeStored(key: string, value: unknown): void {
  if (typeof window === 'undefined') return;
  try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage may be full or unavailable */ }
}

function isJointJob(job: TrainingJobSummary, datasetId: string): boolean {
  return job.datasetId === datasetId && job.kind === 'yue2-joint-train';
}
function isPrepareJob(job: TrainingJobSummary, datasetId: string): boolean {
  return job.datasetId === datasetId && job.kind === 'yue2-prepare-aitk';
}

export const Yue2AitkTrainCard: React.FC<{ datasetId: string; legacyManifest?: string; cursorReady?: boolean; lyricTiming: boolean; onLyricTimingChange: (value: boolean) => void; onTimingLockedChange?: (locked: boolean) => void; exposeStart?: (fn: () => Promise<string | null>) => void }> = ({ datasetId, legacyManifest, cursorReady = false, lyricTiming, onLyricTimingChange, onTimingLockedChange, exposeStart }) => {
  const { t } = useTranslation();
  const [form, setForm] = useState<Yue2JointTrainRequest>(() => readStoredForm(datasetId));
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  // Tracks with no .yue2.txt are re-captioned from the audio before training
  // (on by default): otherwise they train on the long ACE caption. Gemini's
  // model list is the live one the Label panel uses.
  const caps = useTrainingStore(s => s.capabilities);
  const mossOk = !!caps?.moss.available;
  const gemini = caps?.llm.providers.find(p => p.id === 'gemini' && p.available);
  const captionDefault: 'gemini' | 'moss' | null = gemini ? 'gemini' : mossOk ? 'moss' : null;
  const savedCaption = form.autoCaption || undefined;
  const captionProvider = savedCaption && ((savedCaption.provider === 'gemini' && gemini) || (savedCaption.provider === 'moss' && mossOk))
    ? savedCaption.provider : captionDefault;
  const autoCaption = form.autoCaption === false || !captionProvider ? null
    : { provider: captionProvider, ...(captionProvider === 'gemini' && savedCaption?.model ? { model: savedCaption.model } : {}) };
  const [captioning, setCaptioning] = useState<TrainingJobSummary | null>(null);
  const captionMissing = async () => {
    if (!autoCaption) return;
    const started = await captionMissingYue2(datasetId, autoCaption);
    if (!started.jobId) return;
    let j = await getJob(started.jobId);
    setCaptioning(j);
    while (j.status === 'queued' || j.status === 'running') {
      await new Promise(r => setTimeout(r, 1500));
      j = await getJob(started.jobId);
      setCaptioning(j);
    }
    setCaptioning(null);
    if (j.status !== 'done') throw new Error(`Captioning failed: ${j.error || j.status}`);
    const left = (await captionMissingYue2(datasetId, { checkOnly: true })).missing ?? 0;
    if (left) throw new Error(`${left} track(s) still have no YuE2 caption after captioning; see the Label log on the Dataset page. Not training on the ACE captions.`);
  };
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [presets, setPresets] = useState<Yue2JointPreset[]>(() => readStored<Yue2JointPreset[]>(YUE2_JOINT_PRESETS_KEY, []));
  const [presetName, setPresetName] = useState('');
  const [presetError, setPresetError] = useState('');
  // The saved form, with the dataset's own manifest over whatever was saved:
  // a saved path that differs came from another dataset and trained its album.
  const readPrepare = (): PrepareForm => {
    const saved = readStored<PrepareForm>(`${PREP_KEY}${datasetId}`, {
      legacyManifest: legacyManifest ?? '', checkpoint: '', tokenizer: '', output: '',
      models: { vae: '', semantic: '', sheetsage: '' },
    });
    return legacyManifest ? { ...saved, legacyManifest } : saved;
  };
  const [prepare, setPrepare] = useState<PrepareForm>(readPrepare);
  const [bases, setBases] = useState<Yue2JointBaseInfo[]>([]);
  const [defaultBase, setDefaultBase] = useState('');
  const [defaultDevice, setDefaultDevice] = useState('');
  const [prepareJob, setPrepareJob] = useState<TrainingJobSummary | null>(null);
  const [prepareManifest, setPrepareManifest] = useState('');
  const [appliedPrepareJobId, setAppliedPrepareJobId] = useState(() => readStored<string>(`${PREP_KEY}${datasetId}:applied`, ''));
  const [defaultsAvailable, setDefaultsAvailable] = useState<boolean | null>(null);
  const [missingDefaults, setMissingDefaults] = useState<string[]>([]);
  const [defaultsRevision, setDefaultsRevision] = useState(0);
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const selectModels = useBackendStore(s => s.selectModels);
  const yue2RunAllActive = useTrainingStore(s => s.yue2RunAllActive);
  const batchDraft = useTrainingStore(s => s.yue2BatchDraft);
  const datasets = useTrainingStore(s => s.datasets);
  const setBatchDraft = useTrainingStore(s => s.setYue2BatchDraft);
  const startBatch = useTrainingStore(s => s.startYue2Batch);
  const setPhase = useTrainingStore(s => s.setPhase);
  const [batchStarting, setBatchStarting] = useState(false);
  const [batchClearCache, setBatchClearCache] = useState(false);
  // A batch draft turns this card into the recipe editor for N datasets: the
  // form is the same, Start sends it to the server-side batch instead of one
  // run, and the per-dataset paths are resolved per item by the runner.
  const runBatch = async () => {
    if (!batchDraft?.length) return;
    setBatchStarting(true); setError('');
    try {
      const timingWeight = lyricTiming
        ? (typeof form.cursorWeight === 'number' && Number.isFinite(form.cursorWeight) ? form.cursorWeight : 0.08) : 0;
      const recipe = { ...form, autoCaption: autoCaption ?? (false as const), cursorWeight: timingWeight, dataset: '', output: '', resume: '',
        ...(form.preview ? { preview: { ...defaultPreview(form.saveEvery), ...form.preview,
          everySteps: form.preview.parallel ? 0 : form.saveEvery, previewMaxFrames: Math.max(8, Math.min(360, form.preview.seconds || 300)) * 25 } } : {}) };
      await startBatch({ datasetIds: batchDraft, lyricTiming, clearCache: batchClearCache, recipe });
      setPhase('train');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { setBatchStarting(false); }
  };
  const [aitkRuns, setAitkRuns] = useState<Yue2AitkRunRecord[]>([]);
  const [ladderNonce, setLadderNonce] = useState(0);
  const [resumeChoice, setResumeChoice] = useState('');
  const [clearing, setClearing] = useState(false);
  const [cacheInfo, setCacheInfo] = useState<{ slug: string; caches: PreparedCache[]; busy: boolean } | null>(null);
  const [clearNote, setClearNote] = useState('');
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [applyingCheckpoint, setApplyingCheckpoint] = useState(false);
  const [applyNote, setApplyNote] = useState('');
  const [linkingPreset, setLinkingPreset] = useState(false);
  const [presetLinkNote, setPresetLinkNote] = useState('');
  const [runsError, setRunsError] = useState('');
  const [liveMetric, setLiveMetric] = useState<TrainingMetricEvent | null>(null);
  // AITK's joint runner emits step metrics without an epoch stream. Keep the
  // history here in the same step-domain shape used by the legacy YuE2 chart.
  // The job SSE endpoint replays its buffer, so this also reconstructs the
  // curve after a reload or EventSource reconnect.
  const [stepHistory, setStepHistory] = useState<JointStepPoint[]>([]);
  // The KL stop reading at every step, for the chart: the same computation the
  // engine makes, so the line crosses the target where the run stops.
  const klMode = form.targetKlMode ?? 'mean';
  const chartSteps = useMemo(() => {
    const kls: number[] = [];
    return stepHistory.map(point => {
      if (typeof point.arKl === 'number') kls.push(point.arKl);
      const reading = typeof point.arKl === 'number' ? klStopReading(kls, klMode) : null;
      const p = { ...point, loss: point.loss ?? Number.NaN };
      return reading ? { ...p, klStop: reading.value } : p;
    });
  }, [stepHistory, klMode]);
  // The planner freezes at its KL target: from then on steps carry no loss.
  const frozenAt = useMemo(() => {
    const first = stepHistory.findIndex(p => p.frozen || p.loss === undefined);
    return first > 0 && !stepHistory[first - 1].frozen && stepHistory[first - 1].loss !== undefined ? stepHistory[first - 1].step : undefined;
  }, [stepHistory]);
  const [milestones, setMilestones] = useState<JointMilestone[]>([]);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [showJobLogs, setShowJobLogs] = useState(false);
  const [jointPreviews, setJointPreviews] = useState<Yue2JointPreviewRecord[]>([]);

  useEffect(() => {
    // The dataset's own manifest always wins, including when it arrives late.
    if (legacyManifest && prepare.legacyManifest !== legacyManifest) {
      setPrepare(previous => ({ ...previous, legacyManifest }));
    }
  }, [legacyManifest, prepare.legacyManifest]);

  useEffect(() => {
    let cancelled = false;
    void getPreparedData(datasetId).then(value => { if (!cancelled) setCacheInfo(value); })
      .catch(() => { if (!cancelled) setCacheInfo(null); });
    return () => { cancelled = true; };
  }, [datasetId]);

  useEffect(() => {
    let cancelled = false;
    void getYue2AitkPrepare(datasetId).then(result => {
      const defaults = result.defaults;
      if (cancelled) return;
      setBases(result.bases ?? []);
      setDefaultBase(result.defaultBase ?? '');
      setDefaultDevice(result.defaultDevice ?? '');
      setDefaultsAvailable(!!defaults);
      setMissingDefaults(result.missing ?? []);
      if (!defaults) return;
      setPrepare(previous => ({
        ...previous,
        ...Object.fromEntries(Object.entries(defaults).filter(([key, value]) => key !== 'models' && !previous[key as keyof PrepareForm] && typeof value === 'string')),
        models: { ...previous.models, ...Object.fromEntries(Object.entries(defaults.models ?? {}).filter(([key, value]) => !previous.models[key as keyof PrepareForm['models']] && typeof value === 'string')) },
      }));
    }).catch(() => { /* Defaults are advisory; manual paths remain available. */ });
    return () => { cancelled = true; };
  }, [datasetId, defaultsRevision]);

  useEffect(() => {
    let cancelled = false;
    setRunsError('');
    const refresh = () => listYue2AitkRuns(datasetId).then(result => {
      if (cancelled) return;
      setRunsError('');
      setAitkRuns(result.runs);
      const available = result.runs.flatMap(run => run.checkpoints)
        .filter(checkpoint => checkpoint.arPath && checkpoint.narPath);
      setSelectedCheckpoint(previous => previous && available.some(checkpoint => checkpoint.dir === previous)
        ? previous : (available[0]?.dir ?? ''));
    }).catch(err => {
      if (!cancelled) setRunsError(err instanceof Error ? err.message : String(err));
    });
    void refresh();
    const running = job?.status === 'queued' || job?.status === 'running';
    const timer = running ? window.setInterval(refresh, 5000) : undefined;
    return () => { cancelled = true; if (timer !== undefined) window.clearInterval(timer); };
  }, [datasetId, job?.id, job?.status, ladderNonce]);
  // The ladder shown under the run: the run this card's job made, else the
  // live one, else the newest. Its previews are fetched for that run.
  const ladderRunRec = aitkRuns.find(r => r.jobId === job?.id) ?? aitkRuns.find(r => r.live) ?? [...aitkRuns].sort((a, b) => b.createdAt - a.createdAt)[0];
  const ladderRunId = ladderRunRec?.jobId;

  useEffect(() => {
    let cancelled = false;
    if (!ladderRunId) { setJointPreviews([]); return; }
    const refresh = () => listYue2JointPreviews(datasetId, ladderRunId)
      .then(result => { if (!cancelled) setJointPreviews(result.previews); })
      .catch(() => { if (!cancelled) setJointPreviews([]); });
    void refresh();
    // Previews land while the run trains and for a while after it ends
    // (the parallel chain drains), so keep polling a live or just-finished run.
    const fresh = typeof job?.finishedAt === 'number' && Date.now() - job.finishedAt < 30 * 60_000;
    const timer = job?.status === 'queued' || job?.status === 'running' || fresh || ladderRunRec?.live
      ? window.setInterval(refresh, 5000) : undefined;
    return () => { cancelled = true; if (timer !== undefined) window.clearInterval(timer); };
  }, [datasetId, ladderRunId, job?.status, ladderNonce]);

  useEffect(() => {
    setLiveMetric(null);
    setStepHistory([]);
    setMilestones([]);
    setJobLogs([]);
    setShowJobLogs(false);
    if (!job?.id) return;
    const metricKey = `${JOB_KEY}${datasetId}:metrics:${job.id}`;
    const saved = readStored<{ steps?: JointStepPoint[]; milestones?: JointMilestone[] }>(metricKey, {});
    if (saved.steps?.length) setStepHistory(saved.steps);
    const savedMilestones = readStored<JointMilestone[]>(`${metricKey}:milestones`, saved.milestones ?? []);
    if (savedMilestones.length) setMilestones(savedMilestones);
    const stream = new EventSource(jobStreamUrl(job.id));
    stream.onmessage = event => {
      try {
        const item = JSON.parse(event.data) as TrainingStreamEvent;
        if (item.type === 'metric' && item.metric === 'step') {
          setLiveMetric(item);
          if (typeof item.step === 'number' && Number.isFinite(item.step)) {
            setStepHistory(previous => {
              const existing = previous.find(point => point.step === item.step);
              const prior = previous.filter(point => point.step !== item.step);
              const stepMs = typeof item.stepMs === 'number' ? item.stepMs : undefined;
              if (typeof item.narRecon === 'number' && item.loss === undefined && item.stepMs === undefined) {
                // Checkpoint meters: attach to the step's point.
                const merged = { ...(existing ?? { step: item.step!, ep: item.step! }), narRecon: item.narRecon };
                const withRecon = [...prior, merged].sort((a, b) => a.step - b.step).slice(-METRIC_CAP);
                writeStored(metricKey, { steps: withRecon });
                return withRecon;
              }
              const next = [...prior, { ...(existing?.narRecon !== undefined ? { narRecon: existing.narRecon } : {}), step: item.step!, ep: item.step!,
                ...(typeof item.loss === 'number' && Number.isFinite(item.loss) ? { loss: item.loss } : {}),
                ...(typeof item.narMse === 'number' ? { narMse: item.narMse } : {}),
                ...(item.plannerFrozen ? { frozen: true } : {}),
                ...(typeof item.arKl === 'number' ? { arKl: item.arKl } : {}),
                ...(typeof item.gradNorm === 'number' ? { gradNorm: item.gradNorm } : {}),
                // Cumulative training time from the server (survives preview
                // pauses and resumes); wall clock since start only as a fallback.
                ...(typeof item.trainMs === 'number' ? { elapsedMs: item.trainMs }
                  : typeof job.startedAt === 'number' && typeof item.ts === 'number'
                    ? { elapsedMs: Math.max(0, item.ts - job.startedAt) } : {}),
                ...(stepMs !== undefined ? { stepMs } : {}) }];
              next.sort((a, b) => a.step - b.step);
              const capped = next.slice(-METRIC_CAP);
              const mean = (pts: JointStepPoint[]) => { const v = pts.map(p => p.loss).filter((x): x is number => typeof x === 'number'); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : undefined; };
              const withMean = capped.map((point, index) => point.loss === undefined ? point : { ...point,
                ma5: mean(capped.slice(Math.max(0, index - 4), index + 1)),
                ...(index >= 19 ? { ma20: mean(capped.slice(index - 19, index + 1)) } : {}) });
              writeStored(metricKey, { steps: withMean });
              return withMean;
            });
          }
        } else if (item.type === 'metric' && item.metric === 'milestone'
          && typeof item.step === 'number' && typeof item.path === 'string') {
          setMilestones(previous => {
            const next = [...previous.filter(point => point.path !== item.path), { epoch: item.step!, loss: item.loss ?? 0, path: item.path! }];
            writeStored(`${metricKey}:milestones`, next);
            return next;
          });
        } else if (item.type === 'log') {
          const stamp = new Date(item.ts).toLocaleTimeString();
          setJobLogs(previous => [...previous, `${stamp} ${item.level}: ${item.message}`].slice(-100));
        } else if (item.type === 'status' && !['queued', 'running'].includes(item.status)) {
          stream.close();
        }
      } catch { /* Ignore malformed replay frames; polling remains authoritative. */ }
    };
    stream.onerror = () => { /* EventSource reconnects; job polling handles terminal state. */ };
    return () => stream.close();
  }, [job?.id, job?.startedAt]);

  useEffect(() => {
    setForm(readStoredForm(datasetId));
    setResumeChoice('');
    setPrepare(readPrepare());
    setPrepareManifest(readStored<string>(`${PREP_KEY}${datasetId}:manifest`, ''));
    setJob(null);
    setPrepareJob(null);
    setAppliedPrepareJobId(readStored<string>(`${PREP_KEY}${datasetId}:applied`, ''));
    setError('');
    let cancelled = false;
    const storedId = readStored<string>(`${JOB_KEY}${datasetId}`, '');
    const storedPrepareId = readStored<string>(`${PREP_KEY}${datasetId}:job`, '');
    const restore = async () => {
      if (storedPrepareId) {
        try {
          const restored = await getJob(storedPrepareId);
          if (!cancelled && isPrepareJob(restored, datasetId)) setPrepareJob(restored);
        } catch { /* Fall back to the job list below. */ }
      }
      if (storedId) {
        try {
          const restored = await getJob(storedId);
          if (!cancelled && isJointJob(restored, datasetId)) {
            setJob(restored);
            return;
          }
        } catch { /* Fall back to the dataset job list. */ }
      }
      try {
        const result = await listJobs(datasetId);
        const latestPrepare = result
          .filter(item => isPrepareJob(item, datasetId))
          .sort((a, b) => b.createdAt - a.createdAt)[0];
        if (!cancelled && latestPrepare) {
          setPrepareJob(latestPrepare);
          window.localStorage.setItem(`${PREP_KEY}${datasetId}:job`, JSON.stringify(latestPrepare.id));
        }
        const latest = result
          .filter(item => isJointJob(item, datasetId))
          .sort((a, b) => b.createdAt - a.createdAt)[0];
        if (!cancelled && latest) {
          setJob(latest);
          window.localStorage.setItem(`${JOB_KEY}${datasetId}`, JSON.stringify(latest.id));
        }
      } catch { /* The form remains usable when history is unavailable. */ }
    };
    void restore();
    return () => { cancelled = true; };
  }, [datasetId]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(`${FORM_KEY}${datasetId}`, JSON.stringify(form));
    }
  }, [datasetId, form]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(`${PREP_KEY}${datasetId}`, JSON.stringify(prepare));
      if (prepareJob) window.localStorage.setItem(`${PREP_KEY}${datasetId}:job`, JSON.stringify(prepareJob.id));
      if (prepareManifest) window.localStorage.setItem(`${PREP_KEY}${datasetId}:manifest`, JSON.stringify(prepareManifest));
      window.localStorage.setItem(`${PREP_KEY}${datasetId}:applied`, JSON.stringify(appliedPrepareJobId));
    }
  }, [datasetId, prepare, prepareJob, prepareManifest, appliedPrepareJobId]);

  useEffect(() => {
    if (typeof window !== 'undefined') window.localStorage.setItem(YUE2_JOINT_PRESETS_KEY, JSON.stringify(presets));
  }, [presets]);

  // "Perform all stages" runs in the store and can outlive this card: after a
  // page remount it starts training through the old instance, so the job never
  // reaches this one. Adopt any joint job the store reports for this dataset.
  const storeJob = useTrainingStore(s => s.activeJob);
  useEffect(() => {
    if (!storeJob || !isJointJob(storeJob, datasetId) || storeJob.id === job?.id) return;
    setJob(storeJob);
    window.localStorage.setItem(`${JOB_KEY}${datasetId}`, JSON.stringify(storeJob.id));
  }, [datasetId, storeJob?.id]);

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    const id = job.id;
    const timer = window.setInterval(() => {
      void getJob(id).then(next => {
        if (next.datasetId === datasetId) setJob(next);
      }).catch(err => setError(err instanceof Error ? err.message : String(err)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [datasetId, job?.id, job?.status]);

  useEffect(() => {
    if (!prepareJob || !['queued', 'running'].includes(prepareJob.status)) return;
    const id = prepareJob.id;
    const timer = window.setInterval(() => {
      void getJob(id).then(next => {
        if (next.datasetId === datasetId) {
          setPrepareJob(next);
          if (next.status === 'done' && prepareManifest && appliedPrepareJobId !== next.id) {
            setForm(previous => ({ ...previous, checkpoint: prepare.checkpoint, dataset: prepareManifest }));
            setAppliedPrepareJobId(next.id);
          }
        }
      }).catch(err => setError(err instanceof Error ? err.message : String(err)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [datasetId, prepareJob?.id, prepareJob?.status, prepareManifest, prepare.checkpoint, appliedPrepareJobId]);

  useEffect(() => {
    if (prepareJob?.status === 'done' && prepareManifest && appliedPrepareJobId !== prepareJob.id) {
      setForm(previous => ({ ...previous, checkpoint: prepare.checkpoint, dataset: prepareManifest }));
      setAppliedPrepareJobId(prepareJob.id);
    }
  }, [prepareJob?.id, prepareJob?.status, prepareManifest, prepare.checkpoint, appliedPrepareJobId]);

  const set = <K extends keyof Yue2JointTrainRequest>(key: K, value: Yue2JointTrainRequest[K]) =>
    setForm(previous => ({ ...previous, [key]: value }));
  const savePreset = () => {
    const name = presetName.trim();
    if (!name) { setPresetError(t('trainingStudio.yue2.method.presetNameRequired', 'Give the preset a name first.')); return; }
    // Saving under an existing name updates that preset, after asking (#203).
    const existing = presets.find(preset => preset.name.toLowerCase() === name.toLowerCase());
    if (existing && !window.confirm(t('trainingStudio.yue2.method.presetOverwrite', 'Replace the preset "{{name}}" with the current settings?', { name: existing.name }))) return;
    setPresetError('');
    const saved: Yue2JointPreset = { version: 2, name: existing?.name ?? name, settings: snapshotPresetSettings(form, lyricTiming) };
    setPresets(previous => existing ? previous.map(preset => preset === existing ? saved : preset) : [...previous, saved]);
    setPresetName('');
  };
  const exportPreset = (preset: Yue2JointPreset) => {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ kind: 'hot-step-yue2-joint-preset', ...preset }, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${preset.name.replace(/[^a-zA-Z0-9 _-]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const presetFileRef = useRef<HTMLInputElement>(null);
  const importPresets = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text()) as unknown;
      const incoming = (Array.isArray(parsed) ? parsed : [parsed]).filter((item): item is Yue2JointPreset =>
        !!item && typeof item === 'object' && typeof (item as Yue2JointPreset).name === 'string'
        && !!(item as Yue2JointPreset).settings && typeof (item as Yue2JointPreset).settings === 'object');
      if (!incoming.length) throw new Error('no preset in that file');
      setPresets(previous => {
        const next = [...previous];
        for (const item of incoming) {
          // A shared file must not carry someone else's paths into this machine's run.
          const settings = Object.fromEntries(Object.entries(item.settings)
            .filter(([key]) => !PRESET_EXCLUDED_KEYS.has(key as keyof Yue2JointTrainRequest))) as Partial<Yue2JointTrainRequest>;
          let name = item.name.trim() || 'Imported preset';
          for (let n = 2; next.some(p => p.name.toLowerCase() === name.toLowerCase()); n++) name = `${item.name.trim()} (${n})`;
          next.push({ version: 2, name, settings });
        }
        return next;
      });
      setPresetError('');
    } catch (err) {
      setPresetError(t('trainingStudio.yue2.method.presetImportFailed', 'Could not import that file: {{error}}', { error: err instanceof Error ? err.message : String(err) }));
    }
  };
  // Settings that differ from the recipe defaults, per-run paths excluded
  // (#200). Each one can be reset on its own, or all at once.
  const changedSettings = Object.entries(snapshotPresetSettings(form, lyricTiming))
    .filter(([key, value]) => JSON.stringify(value) !== JSON.stringify((DEFAULT_FORM as unknown as Record<string, unknown>)[key]));
  const resetSetting = (key: string) => {
    if (key === 'lyricTiming') { onLyricTimingChange(false); return; }
    setForm(previous => ({ ...previous, [key]: (DEFAULT_FORM as unknown as Record<string, unknown>)[key] }));
  };
  const resetAllSettings = () => {
    setForm(previous => {
      const next: Record<string, unknown> = { ...DEFAULT_FORM };
      for (const key of PRESET_EXCLUDED_KEYS) next[key] = previous[key];
      return next as unknown as Yue2JointTrainRequest;
    });
    if (lyricTiming) onLyricTimingChange(false);
  };
  const loadPreset = (preset: Yue2JointPreset) => {
    // A preset saved before adapter types existed was a LoRA recipe; without
    // this it would load its LoRA alpha onto the LoKr default.
    setForm(previous => ({ ...previous, adapterType: 'lora', ...LORA_STOP, cautious: false, ...preset.settings }));
    if (preset.version === 2 && typeof preset.settings.lyricTiming === 'boolean') onLyricTimingChange(preset.settings.lyricTiming);
  };
  const removePreset = (name: string) => {
    setPresets(previous => previous.filter(preset => preset.name !== name));
  };
  const run = async (): Promise<string | null> => {
    setStarting(true); setError('');
    try {
      const timingWeight = lyricTiming
        ? (typeof form.cursorWeight === 'number' && Number.isFinite(form.cursorWeight) ? form.cursorWeight : 0.08)
        : 0;
      const [resumeRunId, resumeStepText] = resumeChoice.split('|');
      const selectedResume = resumeRunId && resumeStepText ? { resumeRunId, resumeStep: Number(resumeStepText) } : {};
      if (!resumeChoice && !form.resume?.trim()) await captionMissing();
      const request = { ...form, autoCaption: undefined, lyricTiming, alignmentEnabled: lyricTiming, cursorWeight: timingWeight,
        autoPrepare: !resumeChoice && !form.resume?.trim(), preparation: prepare,
        checkpoint: '', output: '',
        ...(form.preview ? { preview: { ...defaultPreview(form.saveEvery), ...form.preview,
          everySteps: form.preview.parallel ? 0 : form.saveEvery, previewMaxFrames: Math.max(8, Math.min(360, form.preview.seconds || 300)) * 25 } } : {}),
        ...(form.resume?.trim() && !resumeChoice ? { resume: form.resume.trim() } : {}),
        ...selectedResume };
      const result = await startYue2JointTrain(datasetId, request);
      if (typeof window !== 'undefined') window.localStorage.setItem(`${JOB_KEY}${datasetId}`, JSON.stringify(result.jobId));
      setJob(await getJob(result.jobId));
      return result.jobId;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally { setStarting(false); }
  };
  // "Perform all stages" (Yue2TrainStages) drives its final training stage
  // through this card's own start so it always trains with the form the user
  // sees. The card hands out the freshest `run` after every render.
  const startRef = useRef<(() => Promise<string | null>) | null>(null);
  useEffect(() => {
    startRef.current = run;
    if (exposeStart) exposeStart(() => startRef.current ? startRef.current() : Promise.resolve(null));
  });
  const prepareDataset = async () => {
    setStarting(true); setError('');
    try {
      const result = await startYue2AitkPrepare(datasetId, { ...prepare, lyricTiming });
      setPrepareManifest(result.manifest);
      setAppliedPrepareJobId('');
      setPrepareJob(await getJob(result.jobId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { setStarting(false); }
  };
  const stop = async () => {
    if (!job) return;
    setError('');
    try {
      await cancelJob(job.id);
      setJob(await getJob(job.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const stopPrepare = async () => {
    if (!prepareJob) return;
    setError('');
    try {
      await cancelJob(prepareJob.id);
      setPrepareJob(await getJob(prepareJob.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const availableCheckpoints: Yue2AitkCheckpointRecord[] = aitkRuns.flatMap(run => run.checkpoints)
    .filter(checkpoint => checkpoint.arPath && checkpoint.narPath);
  const applyCheckpoint = async () => {
    const checkpoint = availableCheckpoints.find(item => item.dir === selectedCheckpoint);
    if (!checkpoint?.arPath || !checkpoint.narPath || activeBackendId !== 'yue2') return;
    setApplyingCheckpoint(true);
    setApplyNote('');
    const unity = {
      lmAdapterArScale: '1', lmAdapterArScaleAttn: '1', lmAdapterArScaleMlp: '1',
      lmAdapterArScaleEarly: '1', lmAdapterArScaleMid: '1', lmAdapterArScaleLate: '1',
      lmAdapterNarScale: '1', lmAdapterNarScaleAttn: '1', lmAdapterNarScaleMlp: '1',
      lmAdapterNarScaleEarly: '1', lmAdapterNarScaleMid: '1', lmAdapterNarScaleLate: '1',
    };
    const ok = await selectModels({ lmAdapterAr: checkpoint.arPath, lmAdapterNar: checkpoint.narPath, ...unity }, activeBackendId);
    setApplyingCheckpoint(false);
    setApplyNote(ok
      ? t('trainingStudio.yue2.method.checkpointApplied', 'AR and NAR adapters applied for the next generation.')
      : t('trainingStudio.yue2.method.checkpointApplyFailed', 'Could not apply this checkpoint. The previous model selection is still active.'));
  };
  const selectResume = (value: string) => {
    setResumeChoice(value);
    if (!value) { setForm(previous => ({ ...previous, resume: '', dataset: '' })); return; }
    const [jobId, stepText] = value.split('|');
    const source = aitkRuns.find(item => item.jobId === jobId);
    const step = Number(stepText);
    if (!source) return;
    const saved = source.options as Partial<Yue2JointTrainRequest> & { alignment?: { enabled?: boolean; cursorWeight?: number } };
    setForm(previous => ({ ...previous, ...saved, trainingMethod: 'aitk',
      checkpoint: '', output: '', resume: '',
      steps: Math.max(previous.steps, step + 400), saveEvery: saved.saveEvery ?? previous.saveEvery,
      dataset: typeof saved.dataset === 'string' ? saved.dataset : '',
      lyricTiming: saved.alignment?.enabled ?? previous.lyricTiming,
      cursorWeight: saved.alignment?.cursorWeight ?? previous.cursorWeight }));
    if (saved.alignment) onLyricTimingChange(saved.alignment.enabled === true);
  };
  const clearCaches = async () => {
    if (!cacheInfo || clearing) return;
    const summary = cacheInfo.caches.map(item => `${item.name}: ${item.files} files, ${(item.bytes / 1048576).toFixed(1)} MiB`).join('\n');
    if (!window.confirm(`Clear all prepared data for ${cacheInfo.slug}?\n\n${summary || 'No generated caches found.'}\n\nSource tracks, sidecars, labels and adapters will remain.`)) return;
    setClearing(true); setClearNote('');
    try {
      await clearPreparedData(datasetId, cacheInfo.slug);
      setResumeChoice('');
      setForm(previous => ({ ...previous, resume: '', dataset: '' }));
      window.localStorage.setItem(`${FORM_KEY}${datasetId}`, JSON.stringify({ ...form, resume: '', dataset: '' }));
      window.localStorage.removeItem(`${PREP_KEY}${datasetId}:manifest`);
      window.localStorage.removeItem(`${PREP_KEY}${datasetId}:applied`);
      window.location.reload();
    } catch (err) {
      setClearNote(err instanceof Error ? err.message : String(err));
      void getPreparedData(datasetId).then(setCacheInfo).catch(() => {});
    } finally { setClearing(false); }
  };
  const linkCheckpointPreset = async () => {
    if (!selectedCheckpoint) return;
    setLinkingPreset(true);
    setPresetLinkNote('');
    try {
      const result = await linkYue2JointCheckpointPreset(datasetId, selectedCheckpoint);
      setPresetLinkNote(result.updated > 0
        ? `Linked this AR/NAR pair to ${result.updated} Lyric Studio album preset${result.updated === 1 ? '' : 's'}.`
        : 'No Lyric Studio album preset is linked to this dataset yet. Export the dataset to Lyric Studio first.');
    } catch (err) {
      setPresetLinkNote(err instanceof Error ? err.message : String(err));
    } finally {
      setLinkingPreset(false);
    }
  };
  const active = job?.status === 'queued' || job?.status === 'running';
  const preparing = prepareJob?.status === 'queued' || prepareJob?.status === 'running';
  const input = 'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 disabled:opacity-50';
  const field = (label: string, key: string, type = 'text', source: unknown = form, update?: (value: string) => void, info?: string, meta?: string) => (
    <label className="flex flex-col gap-1">
      <ParamLabel label={label} info={info} meta={meta} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" />
      <input className={input} type={type} value={String((source as Record<string, unknown>)[key] ?? '')} disabled={(!!resumeChoice && ['seed', 'device', 'rank', 'alpha', 'adapterType', 'lokrDim', 'lokrFactor', 'saveEvery', 'cursorWeight'].includes(key)) || active || starting || preparing || yue2RunAllActive}
        onChange={event => update ? update(event.target.value) : set(key as keyof Yue2JointTrainRequest, type === 'number' ? Number(event.target.value) : event.target.value as never)} />
    </label>
  );
  // The lyric timing toggle lives in its own card on the stages page; it is
  // locked while this card is resuming, preparing or training.
  const timingLocked = !!resumeChoice || active || preparing || starting;
  useEffect(() => { onTimingLockedChange?.(timingLocked); }, [timingLocked, onTimingLockedChange]);
  const progress = job && job.total > 0 ? ` · ${job.done}/${job.total}` : '';
  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
      <h3 className="text-sm font-semibold text-amber-700 dark:text-amber-300">
        {t('trainingStudio.yue2.method.aitkTitle', 'Joint Training')}
      </h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2 leading-relaxed">
        {t('trainingStudio.yue2.method.autoTrainHint', 'Start training prepares the dataset automatically, then trains the planner (AR) and decoder (NAR) adapters together with the YuE2 report recipe. Unchanged prepared data is reused.')}
      </p>
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
        <div>
          <label className="flex flex-col gap-1">
            <ParamLabel
              rootClassName="flex items-center h-4"
              label={t('trainingStudio.yue2.method.resumePrevious', 'Resume a previous run')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.resumePreviousInfo', 'Continues an earlier run from a saved optimizer checkpoint, with the original dataset, base, optimizer and adapter settings restored. Pick "Start a new run" to train from scratch instead; a run with no saved optimizer state, or one still running, cannot be resumed.')}
            />
            <StyledSelect
              accent="amber"
              value={resumeChoice}
              disabled={active || preparing || starting || yue2RunAllActive}
              onChange={selectResume}
              placeholder={t('trainingStudio.yue2.method.resumeStartNew', 'Start a new run')}
              className="w-full"
              options={[
                { value: '', label: t('trainingStudio.yue2.method.resumeStartNew', 'Start a new run') },
                ...aitkRuns.flatMap(run => run.checkpoints.filter(checkpoint => !!checkpoint.optimizerPath).map(checkpoint => ({
                  value: `${run.jobId}|${checkpoint.step}`,
                  label: `${new Date(run.createdAt).toLocaleString()} · step ${checkpoint.step} · ${run.status}${run.resumeError ? ` — ${run.resumeError}` : ''}${run.live ? ' — running' : ''}`,
                  disabled: !!run.resumeError || run.live,
                }))),
                ...aitkRuns.filter(run => !run.checkpoints.some(checkpoint => !!checkpoint.optimizerPath)).map(run => ({
                  value: `unavailable:${run.jobId}`,
                  label: `${new Date(run.createdAt).toLocaleString()} · ${run.resumeError || 'No saved optimizer checkpoint'}`,
                  disabled: true,
                })),
              ]}
            />
          </label>
          {!resumeChoice && <input
            className="mt-2 w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-500 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 disabled:opacity-50"
            value={form.resume ?? ''}
            disabled={active || starting || preparing || yue2RunAllActive}
            onChange={event => set('resume', event.target.value)}
            placeholder={t('trainingStudio.yue2.method.resumeManual', 'Or paste a resume record path (optional)')}
            title={t('trainingStudio.yue2.method.resumeInfo', 'The saved resume record of a previous run, to continue it without picking it from the Resume list above. Leave blank to start a new run.')}
          />}
          {resumeChoice && <p className="mt-1 text-[11px] text-zinc-500">The server restores the original dataset, base, optimizer and adapter settings. Set Steps to the total step you want to reach.</p>}
        </div>
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 h-4">
            <Toggle
              size="sm"
              accent="amber"
              checked={!!autoCaption}
              disabled={!captionDefault || !!resumeChoice || active || preparing || starting || yue2RunAllActive}
              onChange={checked => setForm(previous => ({ ...previous, autoCaption: checked ? { provider: captionProvider ?? 'gemini' } : false }))}
              aria-label={t('trainingStudio.yue2.method.autoCaption', 'Caption tracks that have no YuE2 caption')}
            />
            <ParamLabel
              label={t('trainingStudio.yue2.method.autoCaption', 'Caption tracks that have no YuE2 caption')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={captionDefault
                ? t('trainingStudio.yue2.method.autoCaptionHint', 'Before training, tracks without a .yue2.txt are re-captioned from the audio (ACE, MM3 and YuE2 captions; lyrics and BPM are left alone). Skipped when every track has one. Off: trains on the long ACE caption for those tracks instead.')
                : t('trainingStudio.yue2.method.autoCaptionNone', 'No captioner available: add a Gemini key in Settings → AI Services or install MOSS. Tracks without a .yue2.txt train on the long ACE caption.')}
            />
          </div>
          {autoCaption && <div className="flex flex-col gap-2">
            <StyledSelect
              accent="amber"
              value={autoCaption.provider}
              disabled={active || preparing || starting || yue2RunAllActive}
              onChange={value => setForm(previous => ({ ...previous, autoCaption: { provider: value } }))}
              options={[
                ...(gemini ? [{ value: 'gemini' as const, label: t('trainingStudio.yue2.method.captionGemini', 'Gemini (cloud, hears the audio)') }] : []),
                ...(mossOk ? [{ value: 'moss' as const, label: t('trainingStudio.yue2.method.captionMoss', 'MOSS (local, hears the audio)') }] : []),
              ]}
              className="w-full"
            />
            {autoCaption.provider === 'gemini' && gemini && gemini.models.length > 0 && <StyledSelect
              accent="amber"
              value={autoCaption.model || gemini.defaultModel}
              disabled={active || preparing || starting || yue2RunAllActive}
              onChange={value => setForm(previous => ({ ...previous, autoCaption: { provider: 'gemini', model: value } }))}
              options={gemini.models.map(m => ({ value: m, label: m }))}
              className="w-full"
            />}
          </div>}
        </div>
      </div>
      {captioning && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300 flex items-center gap-1.5"><Loader2 size={12} className="animate-spin" />
        {t('trainingStudio.yue2.method.captioning', 'Captioning tracks without a YuE2 caption: {{done}} / {{total}}', { done: captioning.done, total: captioning.total })}</p>}
      {lyricTiming && !cursorReady && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.lyricTimingNeedsAlignment', 'Run vocal stems and lyric alignment above before starting with timing supervision enabled.')}</p>}
      <details className="mt-4 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-3">
        <summary className="cursor-pointer text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.autoPrepareAdvanced', 'Advanced: dataset preparation and model paths')}</summary>
        <p className="text-[11px] text-zinc-500 mt-1">{t('trainingStudio.yue2.method.autoPrepareHint', 'Preparation runs automatically at the start of training. These controls are only needed for custom paths, manual preparation or resuming a run.')}</p>
        {(defaultsAvailable === false || missingDefaults.length > 0) && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.defaultsMissing', 'Model paths were not found automatically. Install the YuE2 training assets in Model Manager or enter their paths here.')} {missingDefaults.join('; ')}</p>}
        <details className="mt-3">
          <summary className="cursor-pointer text-[11px] font-medium text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.method.advancedPaths', 'Advanced paths and provenance')}</summary>
        <button type="button" disabled={active || preparing || starting || yue2RunAllActive} onClick={() => setDefaultsRevision(value => value + 1)} className="mt-2 text-xs text-amber-700 dark:text-amber-300 hover:underline">{t('trainingStudio.yue2.method.refreshPaths', 'Check installed assets again')}</button>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          {field(t('trainingStudio.yue2.method.legacyManifest', 'Existing YuE2 manifest'), 'legacyManifest', 'text', prepare, value => setPrepare(previous => ({ ...previous, legacyManifest: value })),
            t('trainingStudio.yue2.method.legacyManifestInfo', 'On-disk path to the legacy YuE2 manifest that native preparation converts from. Usually filled in automatically from the dataset; set it by hand only to prepare from a different manifest.'))}
          {field(t('trainingStudio.yue2.method.tokenizer', 'Tokenizer path'), 'tokenizer', 'text', prepare, value => setPrepare(previous => ({ ...previous, tokenizer: value })),
            t('trainingStudio.yue2.method.tokenizerInfo', 'On-disk path to the YuE2 text/audio tokenizer used to prepare the dataset. Usually filled in automatically once the YuE2 training assets are installed.'))}
          {field(t('trainingStudio.yue2.method.prepareOutput', 'New prepared output directory'), 'output', 'text', prepare, value => setPrepare(previous => ({ ...previous, output: value })),
            t('trainingStudio.yue2.method.prepareOutputInfo', 'Where native preparation writes the converted dataset (latents, codes, lead sheets). Leave the default unless you want a second prepared copy of this dataset.'))}
          {field(t('trainingStudio.yue2.method.vae', 'VAE model'), 'vae', 'text', prepare.models, value => setPrepare(previous => ({ ...previous, models: { ...previous.models, vae: value } })),
            t('trainingStudio.yue2.method.vaeInfo', 'On-disk path to the YuE2 VAE checkpoint used to encode audio during preparation. Usually filled in automatically once installed in Model Manager.'))}
          {field(t('trainingStudio.yue2.method.semantic', 'Semantic tokenizer'), 'semantic', 'text', prepare.models, value => setPrepare(previous => ({ ...previous, models: { ...previous.models, semantic: value } })),
            t('trainingStudio.yue2.method.semanticInfo', 'On-disk path to the semantic tokenizer model used during preparation. Usually filled in automatically once installed in Model Manager.'))}
          {field(t('trainingStudio.yue2.method.sheetsage', 'SheetSage model'), 'sheetsage', 'text', prepare.models, value => setPrepare(previous => ({ ...previous, models: { ...previous.models, sheetsage: value } })),
            t('trainingStudio.yue2.method.sheetsageInfo', 'On-disk path to the SheetSage lead-sheet model used during preparation. Usually filled in automatically once installed in Model Manager.'))}
        </div>
        </details>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          {form.resume?.trim() && field(t('trainingStudio.yue2.method.resumeDataset', 'Prepared manifest for resume'), 'dataset', 'text', form, undefined,
            t('trainingStudio.yue2.method.resumeDatasetInfo', 'The prepared dataset manifest that matches the run being resumed by record below. Filled in automatically when a resume record is set; only needed if you are pointing at a different prepared copy.'))}
        </div>
        <button type="button" onClick={() => void prepareDataset()} disabled={preparing || active || starting || yue2RunAllActive || !prepare.legacyManifest || !prepare.checkpoint || !prepare.tokenizer || !prepare.output || !prepare.models.vae || !prepare.models.semantic || !prepare.models.sheetsage}
          className="mt-3 px-3 py-1.5 rounded-lg text-xs font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40">
          {preparing ? t('trainingStudio.yue2.method.preparing', 'Preparing dataset…') : t('trainingStudio.yue2.method.prepare', 'Prepare native dataset')}
        </button>
        {prepareJob && <span className="ml-3 text-[11px] text-zinc-600 dark:text-zinc-400">{prepareJob.status} · {prepareJob.phase || 'waiting'}</span>}
        {preparing && <button type="button" onClick={() => void stopPrepare()} className="ml-3 text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.yue2.method.cancel', 'Stop')}</button>}
        {prepareJob?.error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{prepareJob.error}</div>}
      </details>
      <div className="mt-4">
        <div className="flex items-center gap-2 mb-2">
          <ParamLabel label={t('trainingStudio.yue2.method.preset', 'Preset')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.presetInfo', 'All three are the same recipe; they differ in how many updates run and how many songs each update averages. Training time grows with the songs processed, so each preset shows its time relative to Balanced. For scale, Balanced took about 50 minutes on a 15-track album on an RTX 5090; a slower GPU or a longer album takes proportionally longer.')} />
          {!activePreset(form) && <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.presetCustom', 'custom')}</span>}
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {PRESETS.map(p => <button key={p.key} type="button" disabled={active || starting || preparing || yue2RunAllActive}
            onClick={() => setForm(previous => ({ ...previous, ...presetValues(p) }))}
            className={`flex flex-col items-center gap-1 px-4 py-3 rounded-xl border-2 transition-colors disabled:opacity-40 ${activePreset(form) === p.key
              ? 'border-blue-500 bg-blue-500/15 text-blue-700 dark:text-blue-300'
              : 'border-zinc-300 dark:border-white/15 text-zinc-800 dark:text-zinc-100 hover:border-blue-500/50 hover:bg-blue-500/5'}`}>
            <span className="text-base font-bold">{t(`trainingStudio.yue2.method.preset_${p.key}`, p.label)}</span>
            <span className="text-xs text-zinc-500">{p.steps} × {p.gradAccum} songs · {p.key === 'balanced' ? t('trainingStudio.yue2.method.presetBaseline', '1× time (baseline)') : t('trainingStudio.yue2.method.presetRelative', '{{ratio}} the time', { ratio: presetTime(p) })}</span>
          </button>)}
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.method.base', 'Base model')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.baseInfo', 'The frozen model the adapters train against. ConvRot (int8) is the checkpoint the recipe was tuned and ear-tested on, and needs a CUDA build. A GGUF base runs on any GPU: bf16 is full precision, and the quantized files trade a little accuracy for a lot of memory (Q4_K_M holds the base in about 2 GB). The adapter works with every base at generation time.')}
            meta={t('trainingStudio.yue2.method.baseMeta', 'default {{base}}', { base: defaultBase || 'convrot' })}
          />
          <StyledSelect
            accent="amber"
            value={form.base || defaultBase}
            disabled={!!resumeChoice || active || starting || preparing || yue2RunAllActive || bases.length === 0}
            className="w-full"
            onChange={value => set('base', value)}
            options={bases.filter(b => b.runnable).map(b => ({
              value: b.id,
              label: `${b.convrot ? t('trainingStudio.yue2.method.baseConvrot', 'ConvRot int8') : `GGUF ${b.id}`} · ${(b.bytes / 1e9).toFixed(1)} GB`,
            }))}
          />
        </label>
        {field(t('trainingStudio.yue2.method.device', 'Device'), 'device', 'text', form, undefined,
          t('trainingStudio.yue2.method.deviceInfo', 'Which GPU trains this run, for a machine with more than one: CUDA0, CUDA1 on an NVIDIA build, Vulkan0, Vulkan1 on a Vulkan build. Leave empty for the first GPU.'),
          t('trainingStudio.yue2.method.deviceMeta', 'default {{device}}', { device: defaultDevice || 'CUDA0' }))}
        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.method.adapterType', 'Adapter type')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.adapterTypeInfo', 'The adapter parameterization trained. LoRA is the standard low-rank pair. LoKr trains a Kronecker-factored delta per site instead, at a similar file size for more capacity, but is experimental. Switching resets rank/alpha (or dim/factor/alpha) and the KL stop to that type\'s tested defaults.')}
          />
          <StyledSelect
            accent="amber"
            value={form.adapterType ?? 'lora'}
            disabled={!!resumeChoice || active || starting || preparing || yue2RunAllActive}
            className="w-full"
            onChange={value => {
              const adapterType = value === 'lokr' ? 'lokr' : 'lora';
              // LoKr's scale is alpha/dim, so alpha follows the dim by default
              // (LyCORIS scale 1); LoRA gets its rank/alpha defaults back.
              setForm(previous => adapterType === 'lokr'
                ? { ...previous, adapterType, lokrDim: previous.lokrDim ?? 64, lokrFactor: previous.lokrFactor ?? 4, alpha: 4 * (previous.lokrDim ?? 64), ...LOKR_STOP }
                : { ...previous, adapterType, alpha: previous.rank ?? 64, ...LORA_STOP });
            }}
            options={[
              { value: 'lora' as const, label: t('trainingStudio.yue2.method.adapterLora', 'LoRA') },
              { value: 'lokr' as const, label: t('trainingStudio.yue2.method.adapterLokr', 'LoKr (experimental)') },
            ]}
          />
        </label>
      </div>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-3">
        {field(t('trainingStudio.yue2.method.steps', 'Updates'), 'steps', 'number', form, undefined,
          t('trainingStudio.yue2.method.stepsInfo', 'How many optimizer updates to train. Each update averages "songs per update" songs, so Fast (50 × 4) sees 200 songs and Thorough (200 × 8) sees 1600. The run ends here and every tenth update is a checkpoint; pick one by ear.'),
          t('trainingStudio.yue2.method.stepsMeta', 'Balanced 100'))}
        {field(t('trainingStudio.yue2.method.saveEvery', 'Save every'), 'saveEvery', 'number', form, undefined,
          t('trainingStudio.yue2.method.saveEveryInfo', 'How many steps between saved checkpoints. Lower gives more rungs to pick from (and more previews, if enabled) at the cost of disk space and time; higher saves less often.'),
          t('trainingStudio.yue2.method.saveEveryMeta', 'default 25'))}
        {field(t('trainingStudio.yue2.method.seed', 'Seed'), 'seed', 'number', form, undefined,
          t('trainingStudio.yue2.method.seedInfo', 'The random seed for training (batch order, dropout, initial noise). Changing it gives a different run on the same data; keeping it fixed makes a rerun reproducible.'),
          t('trainingStudio.yue2.method.seedMeta', 'default 42'))}
        {(form.adapterType ?? 'lora') === 'lokr' ? <>
          {field(t('trainingStudio.yue2.method.lokrDim', 'LoKr dim'), 'lokrDim', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrDimInfo', 'The size of the Kronecker-factored delta. Higher gives the adapter more capacity, at a larger file and more VRAM; the tested recipe keeps alpha at 4x this value.'),
            t('trainingStudio.yue2.method.lokrDimMeta', 'default 64'))}
          {field(t('trainingStudio.yue2.method.lokrFactor', 'LoKr factor'), 'lokrFactor', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrFactorInfo', 'How each weight is split into two Kronecker factors: the smaller factor is at most this size. 4 is tested; 8 also fits. At factor 4, stay below dim 256, where some sites stop factorizing and ignore alpha.'),
            t('trainingStudio.yue2.method.lokrFactorMeta', 'default 4'))}
          {field(t('trainingStudio.yue2.method.lokrAlpha', 'LoKr alpha'), 'alpha', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrAlphaInfo', 'The LoKr strength, alpha / dim. The tested default is 4x the dim; for more capacity raise dim and keep alpha at 4x dim rather than raising alpha alone.'),
            t('trainingStudio.yue2.method.lokrAlphaMeta', 'default 256 (4x dim 64)'))}
        </> : <>
          {field(t('trainingStudio.yue2.method.rank', 'LoRA rank'), 'rank', 'number', form, undefined,
            t('trainingStudio.yue2.method.rankInfo', 'The size of the low-rank adapter pair. Higher gives more capacity to learn the artist, at a larger file and more VRAM; lower trains a smaller, less expressive adapter.'),
            t('trainingStudio.yue2.method.rankMeta', 'default 64'))}
          {field(t('trainingStudio.yue2.method.alpha', 'LoRA alpha'), 'alpha', 'number', form, undefined,
            t('trainingStudio.yue2.method.alphaInfo', 'The LoRA scale. This card keeps it equal to rank (scale 1); raising alpha above rank strengthens the adapter\'s effect without changing its size.'),
            t('trainingStudio.yue2.method.alphaMeta', 'default = rank'))}
        </>}
      
      </div>
      {(() => {
        const lokr = (form.adapterType ?? 'lora') === 'lokr';
        const half = yue2AdapterHalfBytes(lokr
          ? { type: 'lokr', dim: Number(form.lokrDim ?? 64), factor: Number(form.lokrFactor ?? 4) }
          : { type: 'lora', rank: Number(form.rank ?? 64) });
        const loraAlt = lokr ? yue2AdapterHalfBytes({ type: 'lora', rank: Number(form.lokrDim ?? 64) }) : null;
        return <p className={`text-[11px] mt-2 ${half === null ? 'text-amber-700 dark:text-amber-300' : 'text-zinc-500'}`}>
          {half === null
            ? t('trainingStudio.yue2.method.sizeBadFactor', 'This LoKr factor does not split the attention and MLP outputs on their boundaries, so training will refuse it. Try 4 or 8.')
            : t('trainingStudio.yue2.method.sizeEstimate', 'Expected adapter size: {{total}} for both halves ({{half}} each for the planner and the decoder).', { total: formatMB(2 * half), half: formatMB(half) })}
          {half !== null && loraAlt !== null && ` ${t('trainingStudio.yue2.method.sizeLoraCompare', 'A rank-{{rank}} LoRA would be {{lora}}.', { rank: Number(form.lokrDim ?? 64), lora: formatMB(2 * loraAlt) })}`}
        </p>;
      })()}
      <details className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/40 dark:bg-black/5 p-3">
        <summary className="cursor-pointer text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.baseMatchedKnobs', 'Base-matched settings')}</summary>
        <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.baseMatchedKnobsHint', 'Blank uses the default shown in grey. The report does not state a fine-tuning learning rate or the dropout rates, so those defaults are agreed guesses; the rest are the report\'s own values.')}</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
          {([
            ['lr', t('trainingStudio.yue2.method.lr', 'Learning rate'), t('trainingStudio.yue2.method.bmLrInfo', 'AdamW peak learning rate for both halves. The report gives full-model rates only (3e-4 joint, annealed to 3e-5), which do not transfer to an adapter; 1e-4 is the agreed default.'), 'default 1e-4'],
            ['warmup', t('trainingStudio.yue2.method.warmup', 'Warmup steps'), t('trainingStudio.yue2.method.warmupInfo', 'Linear warmup to the peak rate, then cosine decay to a 0.1x floor at the step count. Blank = 3% of the steps, the base\'s own joint-phase warmup share.'), 'default 3% of steps'],
            ['weightDecay', t('trainingStudio.yue2.method.weightDecay', 'Weight decay'), t('trainingStudio.yue2.method.bmWeightDecayInfo', 'Decoupled weight decay, as the base\'s joint phase used.'), 'default 0.1 (report)'],
            ['gradAccum', t('trainingStudio.yue2.method.gradAccum', 'Songs per update'), t('trainingStudio.yue2.method.gradAccumInfo', 'Gradients of this many songs are averaged before each optimizer update, so one step sees more than one song, as the base\'s batch of 256 did. Each update takes this many times longer. Fast and Balanced use 4 (28 s an update on the 5090), Thorough 8 (60 s). 1 updates on every song.'), 'default 4'],
            ['arLossWeight', t('trainingStudio.yue2.method.arLossWeight', 'Planner loss weight'), t('trainingStudio.yue2.method.arLossWeightInfo', 'The planner\'s cross-entropy is weighted by this against the decoder\'s flow loss before the shared gradient clip, so it sets how much of the clip the planner takes. The report trains the base at 0.25; 1.0 is the tuned recipe.'), 'default 0.25 (report)'],
            ['beta2', t('trainingStudio.yue2.method.beta2', 'Adam beta2'), t('trainingStudio.yue2.method.beta2Info', 'How long the optimizer\'s second moment remembers. The report\'s joint phase used 0.95; torch\'s 0.999 is what the tuned recipe uses.'), 'default 0.95 (report)'],
            ['textDropout', t('trainingStudio.yue2.method.textDropout', 'Text dropout'), t('trainingStudio.yue2.method.textDropoutInfo', 'Share of steps trained with the style text (trigger included) removed and the lyrics kept. The report drops text and lyrics separately or together for guidance but gives no rates; 0.1 is the agreed default.'), 'default 0.1'],
            ['lyricDropout', t('trainingStudio.yue2.method.lyricDropout', 'Lyric dropout'), t('trainingStudio.yue2.method.lyricDropoutInfo', 'Share of steps trained with the lyrics removed and the style kept, written the way the official instrumental tooling sends a lyric-free request.'), 'default 0.1'],
            ['bothDropout', t('trainingStudio.yue2.method.bothDropout', 'Unconditional dropout'), t('trainingStudio.yue2.method.bothDropoutInfo', 'Share of steps trained on the bare instruction with no style or lyrics: exactly the prompt the runtime\'s guidance uses as its unconditional branch.'), 'default 0.1'],
            ['abcDropout', t('trainingStudio.yue2.method.abcDropout', 'ABC dropout'), t('trainingStudio.yue2.method.bmAbcDropoutInfo', 'Share of steps trained without the lead sheet, matching the report\'s balanced mix of tasks with and without a score.'), 'default 0.5 (report)'],
            ['narCropFrames', t('trainingStudio.yue2.method.narCropFrames', 'Decoder crop (frames)'), t('trainingStudio.yue2.method.bmNarCropInfo', 'The decoder trains on this many frames per song (25 per second). 0 trains on the whole song, as the base did (about 12 GB of VRAM and 11-15 s a step on a 4-minute song); 1500 is the tuned recipe\'s 60 s crop.'), 'default 0 (whole song, report)'],
          ] as const).map(([key, label, info, meta]) => (
            <label key={key} className="flex flex-col gap-1">
              <ParamLabel label={label} info={info} meta={meta} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" />
              <input className={`${input} placeholder:text-zinc-500`} type="number" step="any"
                placeholder={String(key === 'warmup' ? Math.max(1, Math.round((Number(form.steps) || 0) * BASE_MATCHED_DEFAULTS.warmupFraction)) : BASE_MATCHED_DEFAULTS[key])}
                value={form[key] ?? ''} disabled={active || starting || preparing || yue2RunAllActive}
                onChange={event => set(key, event.target.value === '' ? undefined : Number(event.target.value))} />
            </label>
          ))}
        </div>
      </details>
      <div className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/40 dark:bg-black/5 p-3">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.presets', 'Training presets')}</span>
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.presetHint', 'A preset captures the training settings (steps, save cadence, seed, device, optimizer, rank/alpha and stop target) — never the dataset, checkpoint or output paths. Presets are stored in this browser.')}</span>
        </div>
        <div className="mt-2 flex items-center gap-2 flex-wrap">
          <input className={`${input} min-w-40 flex-1`} placeholder={t('trainingStudio.yue2.method.presetNamePlaceholder', 'New preset name')}
            value={presetName} disabled={active || preparing || starting || yue2RunAllActive}
            onChange={event => { setPresetName(event.target.value); setPresetError(''); }}
            onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); void savePreset(); } }} />
          <button type="button" onClick={() => void savePreset()} disabled={active || preparing || starting || yue2RunAllActive}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40">
            {t('trainingStudio.yue2.method.presetSave', 'Save current settings')}
          </button>
          <button type="button" onClick={() => presetFileRef.current?.click()} disabled={active || preparing || starting || yue2RunAllActive}
            title={t('trainingStudio.yue2.method.presetImportTitle', 'Add presets from a .json file exported here or by someone else')}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10 disabled:opacity-40">
            <Upload size={12} />{t('trainingStudio.yue2.method.presetImport', 'Import')}
          </button>
          <input ref={presetFileRef} type="file" accept=".json,application/json" className="hidden"
            onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void importPresets(file); }} />
        </div>
        {presetError && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{presetError}</p>}
        {presets.length > 0 && <div className="mt-2 flex flex-wrap gap-2">
          {presets.map(preset => (
            <span key={preset.name} className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-300 dark:border-white/10 bg-white/60 dark:bg-zinc-900/40 pl-2.5 pr-1 py-1 text-[11px] text-zinc-700 dark:text-zinc-300">
              <button type="button" onClick={() => loadPreset(preset)} disabled={active || preparing || starting || yue2RunAllActive}
                title={t('trainingStudio.yue2.method.presetLoad', 'Load this preset into the form')}
                className="font-medium hover:underline disabled:no-underline disabled:opacity-40">
                {preset.name}{preset.settings.steps !== undefined && preset.settings.saveEvery !== undefined
                  ? ` · ${preset.settings.steps} steps, save every ${preset.settings.saveEvery}` : ''}
              </button>
              <button type="button" onClick={() => exportPreset(preset)}
                title={t('trainingStudio.yue2.method.presetExport', 'Download this preset as a .json file to share or back up')}
                className="rounded p-0.5 text-zinc-400 hover:text-amber-600 dark:hover:text-amber-400">
                <Download size={12} />
              </button>
              <button type="button" onClick={() => removePreset(preset.name)}
                title={t('trainingStudio.yue2.method.presetRemove', 'Remove this preset')}
                className="rounded p-0.5 text-zinc-400 hover:text-red-600 dark:hover:text-red-400">
                <X size={12} />
              </button>
            </span>
          ))}
        </div>}
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
            {t('trainingStudio.yue2.method.changedSettings', 'Changed from defaults')}
          </span>
          {changedSettings.length === 0
            ? <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.noChangedSettings', 'none: this form is the recipe defaults')}</span>
            : <>
              {changedSettings.map(([key, value]) => (
                <span key={key} className="inline-flex items-center gap-1 rounded-lg border border-sky-500/40 bg-sky-500/10 pl-2 pr-1 py-0.5 text-[11px] text-sky-700 dark:text-sky-300">
                  {key}{typeof value === 'object' ? '' : `: ${String(value)}`}
                  <button type="button" onClick={() => resetSetting(key)} disabled={active || preparing || starting || yue2RunAllActive}
                    title={t('trainingStudio.yue2.method.resetSetting', 'Reset this setting to its default')}
                    className="rounded p-0.5 hover:text-sky-900 dark:hover:text-white disabled:opacity-40">
                    <RotateCcw size={11} />
                  </button>
                </span>
              ))}
              <button type="button" onClick={resetAllSettings} disabled={active || preparing || starting || yue2RunAllActive}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-[11px] font-semibold border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10 disabled:opacity-40">
                <RotateCcw size={11} />{t('trainingStudio.yue2.method.resetAll', 'Reset all to defaults')}
              </button>
            </>}
        </div>
      </div>
      <Toggle
        accent="amber"
        className="mt-3"
        checked={form.stopEngine !== false}
        disabled={active || preparing || starting || yue2RunAllActive}
        onChange={checked => setForm(previous => ({ ...previous, stopEngine: checked }))}
        label={t('trainingStudio.yue2.method.stopEngine', 'Stop the engine during training')}
        info={t('trainingStudio.yue2.method.stopEngineHelp', 'Off by default since the ladder previews render while training runs and need the engine up (about 14 GB for the trainer plus 10-12 GB for the engine and a render). On: the trainer gets the whole GPU, generation is unavailable, and no previews render until the run ends; use it on a smaller card and render the ladder afterwards.')}
      />
      <details className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/30 dark:bg-black/10 p-3">
        <summary className="cursor-pointer text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">
          {t('trainingStudio.yue2.method.previewTitle', 'Checkpoint previews (optional)')}
        </summary>
        <Toggle
          accent="amber"
          className="mt-2"
          checked={form.preview?.enabled ?? false}
          disabled={active || preparing || starting || yue2RunAllActive}
          onChange={checked => setForm(previous => ({ ...previous, preview: { ...(previous.preview ?? defaultPreview(form.saveEvery)), enabled: checked } }))}
          label={t('trainingStudio.yue2.method.previewEnable', 'Render a preview at every saved checkpoint')}
          info={t('trainingStudio.yue2.method.previewManual', 'On by default: each checkpoint gets one draft take (12 decoder steps, 300 s) rendered while training continues, so the ladder is ready to score when the run ends. Off: render rungs by hand from the ladder below.')}
        />
        {form.preview?.enabled && <Toggle
          accent="amber"
          size="sm"
          className="mt-2"
          checked={form.preview.parallel !== false}
          disabled={active || preparing || starting || yue2RunAllActive}
          onChange={checked => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, parallel: checked }, ...(checked ? { stopEngine: false } : {}) }))}
          label={t('trainingStudio.yue2.method.previewParallel', 'In parallel with training')}
          info={t('trainingStudio.yue2.method.previewParallelInfo', 'Render each checkpoint\'s preview while training continues; needs the engine up (this also turns "Stop the engine during training" off). Off: training pauses at each checkpoint, renders, and resumes, which fits a smaller card but takes longer.')}
        />}
        {form.preview?.enabled && <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          {field(t('trainingStudio.yue2.method.previewSeconds', 'Preview seconds'), 'seconds', 'number', form.preview, value => setForm(previous => {
            const seconds = Math.max(8, Math.min(360, Number(value) || 300));
            return { ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seconds, previewMaxFrames: seconds * 25 } };
          }), t('trainingStudio.yue2.method.previewSecondsInfo', 'How long the rendered preview sample is, from 8 to 120 seconds. Longer previews show more of the song but take longer to render at every checkpoint.'), t('trainingStudio.yue2.method.previewSecondsMeta', 'default 90'))}
          {field(t('trainingStudio.yue2.method.previewSeed', 'Preview seed'), 'seed', 'number', form.preview, value => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seed: Number(value) } })),
            t('trainingStudio.yue2.method.previewSeedInfo', 'The random seed used for every preview render, so previews across checkpoints are directly comparable rather than each landing on a different random take.'), t('trainingStudio.yue2.method.previewSeedMeta', 'default 424242'))}
          <p className="text-[11px] text-zinc-500 md:col-span-2">{t('trainingStudio.yue2.method.previewSongHint', 'The first track in this dataset is used for the preview. Caption and lyrics overrides below are optional.')}</p>
          <Toggle
            accent="amber"
            size="sm"
            checked={form.preview.baseline}
            disabled={active || preparing || starting || yue2RunAllActive}
            onChange={checked => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, baseline: checked } }))}
            label={t('trainingStudio.yue2.method.previewBaseline', 'Include baseline')}
            info={t('trainingStudio.yue2.method.previewBaselineInfo', 'Also renders a take from the unmodified base model alongside the adapter, so you can hear what the adapter changed.')}
          />
          <Toggle
            accent="amber"
            size="sm"
            checked={form.preview.control}
            disabled={active || preparing || starting || yue2RunAllActive}
            onChange={checked => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, control: checked } }))}
            label={t('trainingStudio.yue2.method.previewControl', 'Include control')}
            info={t('trainingStudio.yue2.method.previewControlInfo', 'Also renders a fixed reference take at every checkpoint, for a stable point of comparison as the adapter trains.')}
          />
          <label className="md:col-span-2 flex flex-col gap-1">
            <ParamLabel
              label={t('trainingStudio.yue2.method.previewCaption', 'Caption override (optional)')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.previewCaptionInfo', 'Replaces the preview track\'s own caption for the rendered sample. Leave blank to use the track\'s caption as-is.')}
            />
            <textarea className={`${input} min-h-16 resize-y`} value={form.preview.caption ?? ''} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, caption: event.target.value } }))} />
          </label>
          <label className="md:col-span-2 flex flex-col gap-1">
            <ParamLabel
              label={t('trainingStudio.yue2.method.previewLyrics', 'Lyrics override (optional)')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.previewLyricsInfo', 'Replaces the preview track\'s own lyrics for the rendered sample. Leave blank to use the track\'s lyrics as-is.')}
            />
            <textarea className={`${input} min-h-20 resize-y`} value={form.preview.lyrics ?? ''} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, lyrics: event.target.value } }))} />
          </label>
        </div>}
      </details>
      <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.hardware', 'Joint training requires a CUDA build and an NVIDIA GPU with BF16 support (Ampere or newer).')}</p>
      <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.autoOutput', 'Adapters are saved in your global adapters folder under yue2-joint-adapters/triggerword_date_time.')}</p>
      <div className="mt-4 rounded-lg border border-red-500/20 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">Prepared data</p>
        <p className="mt-1 text-[11px] text-zinc-500">Clear generated latents, codes, lead sheets, alignment, stems, MM3 caches and ACE tensors for this dataset. Source files, labels and adapters are kept. Older runs may then be unavailable to resume.</p>
        <button type="button" onClick={() => void clearCaches()}
          disabled={!cacheInfo?.caches.length || cacheInfo.busy || active || preparing || starting || clearing || yue2RunAllActive}
          className="mt-2 px-3 py-1.5 rounded-lg text-xs font-semibold border border-red-500/50 text-red-600 dark:text-red-400 hover:bg-red-500/10 disabled:opacity-40">
          {clearing ? 'Clearing…' : 'Clear all prepared data'}
        </button>
        {cacheInfo && <span className="ml-2 text-[11px] text-zinc-500">{cacheInfo.caches.length} cache folders · {(cacheInfo.caches.reduce((sum, item) => sum + item.bytes, 0) / 1048576).toFixed(1)} MiB</span>}
        {clearNote && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{clearNote}</p>}
      </div>
      {error && <div className="mt-3 flex items-start gap-2 text-xs text-red-600 dark:text-red-400"><AlertTriangle size={14} className="mt-0.5 shrink-0" />{error}</div>}
      {job?.error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{job.error}</div>}
      {batchDraft && batchDraft.length > 0 && <div className="mt-4 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
        <p className="text-xs font-semibold text-amber-700 dark:text-amber-300">
          {t('trainingStudio.yue2.aitkBatch.draftTitle', 'Batch: these settings apply to {{count}} dataset(s)', { count: batchDraft.length })}
        </p>
        <p className="mt-1 text-[11px] text-zinc-600 dark:text-zinc-400 break-words">
          {batchDraft.map(id => datasets.find(d => d.id === id)?.name ?? id).join(' · ')}
        </p>
        <p className="mt-1 text-[11px] text-zinc-500">
          {t('trainingStudio.yue2.aitkBatch.draftHint', 'Each dataset runs its caches, lead sheets, timing (when enabled), preparation and joint training in order with this recipe. The batch runs on the server; the page follows the dataset being trained.')}
        </p>
        <Toggle
          accent="amber"
          className="mt-2"
          checked={batchClearCache}
          disabled={batchStarting || yue2RunAllActive}
          onChange={setBatchClearCache}
          label={t('trainingStudio.yue2.aitkBatch.clearCache', 'Clear out all cached data')}
          info={t('trainingStudio.yue2.aitkBatch.clearCacheHint', "Deletes each dataset's YuE2 caches (latents, codes, lead sheets, vocal stems, lyric timing, prepared datasets) before it starts, so everything is rebuilt from the audio. Source audio, captions and trained adapters are not touched. Each album then takes as long as a first-time preparation.")}
        />
        <div className="mt-2 flex items-center gap-3">
          <button type="button" onClick={() => void runBatch()} disabled={batchStarting || yue2RunAllActive}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
            {batchStarting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {t('trainingStudio.yue2.aitkBatch.start', 'Start batch ({{count}})', { count: batchDraft.length })}
          </button>
          <button type="button" onClick={() => setBatchDraft(null)} className="text-xs text-zinc-500 hover:underline">{t('trainingStudio.yue2.aitkBatch.discard', 'Discard batch')}</button>
        </div>
      </div>}
      <div className="mt-4 flex items-center gap-3 flex-wrap">
        <button type="button" onClick={() => void run()} disabled={!!batchDraft?.length || active || preparing || starting || yue2RunAllActive || (resumeChoice || form.resume?.trim() ? !form.dataset : (!prepare.legacyManifest || !prepare.tokenizer)) || (!resumeChoice && lyricTiming && !cursorReady)}
          className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
          {starting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          {active ? (job?.phase === 'preparing' ? t('trainingStudio.yue2.method.preparing', 'Preparing dataset…') : t('trainingStudio.yue2.method.running', 'Joint training is running')) : t('trainingStudio.yue2.method.start', 'Start joint training')}
        </button>
        {active && <button type="button" onClick={() => void stop()} className="text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.yue2.method.cancel', 'Stop')}</button>}
        {job && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{job.status} · {job.phase || 'waiting'}{progress}
          {liveMetric?.step !== undefined && ` · step ${liveMetric.step}${liveMetric.loss !== undefined ? ` · loss ${liveMetric.loss.toFixed(4)}` : ''}`}
        </span>}
      </div>
      {stepHistory.length > 1 && (
        <div className="mt-3">
          <TrainingChart
            epochs={[]}
            steps={chartSteps}
            milestones={milestones}
            target={form.stopMode === 'loss' ? (form.targetLoss ?? 0) : 0}
            klTarget={form.stopMode === 'kl' ? (form.targetKl ?? 0) : 0}
            klStopLabel={(form.targetKlMode ?? 'mean') === 'trend' ? 'KL trend (stop reading)' : 'KL 20-step mean (stop reading)'}
            maxEpochs={form.steps}
          />
          {frozenAt !== undefined && <div className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">
            {t('trainingStudio.yue2.method.plannerFrozen', 'Planner frozen at step {{step}} (KL target reached). The decoder is still training, faster: the loss and KL lines end here, the step counter and pace below keep moving.', { step: frozenAt })}
          </div>}
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] tabular-nums text-zinc-500">
            {frozenAt === undefined && <span>MA5 {stepHistory[stepHistory.length - 1].ma5?.toFixed(4) ?? '—'}</span>}
            {frozenAt === undefined && <span>20-step stop mean {stepHistory[stepHistory.length - 1].ma20?.toFixed(4) ?? '—'}</span>}
            {jointLossRate(stepHistory) !== null && <span>loss rate {jointLossRate(stepHistory)!.toFixed(5)}/step</span>}
            <span>{stepHistory[stepHistory.length - 1].step} / {form.steps} steps</span>
            {stepHistory[stepHistory.length - 1].elapsedMs !== undefined && <span>elapsed {formatDurationMs(stepHistory[stepHistory.length - 1].elapsedMs!)}</span>}
            {stepHistory[stepHistory.length - 1].stepMs !== undefined && <span>pace {(stepHistory.slice(-20).reduce((sum, point) => sum + (point.stepMs ?? 0), 0) / Math.max(1, stepHistory.slice(-20).filter(point => point.stepMs !== undefined).length) / 1000).toFixed(2)}s/step</span>}
            {job?.status === 'running' && <span>{jointEta(stepHistory, form)}</span>}
          </div>
        </div>
      )}
      {jobLogs.length > 0 && <details className="mt-2" open={showJobLogs} onToggle={event => setShowJobLogs(event.currentTarget.open)}>
        <summary className="cursor-pointer text-[11px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.method.showLogs', 'Show training log (last 100 lines)')}</summary>
        <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-zinc-950 p-2 text-[10px] leading-4 text-zinc-300 whitespace-pre-wrap">{jobLogs.join('\n')}</pre>
      </details>}
      {job?.status === 'done' && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{t('trainingStudio.yue2.method.checkpointWritten', 'Joint checkpoints are in the selected output directory.')}</p>}
      {(aitkRuns.length > 0 || runsError) && <div className="mt-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.auditionTitle', 'Audition a joint checkpoint')}</p>
        {activeBackendId !== 'yue2' && <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.selectYue2', 'Select the YuE2 backend to use these adapters for generation.')}</p>}
        {runsError && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{runsError}</p>}
        {availableCheckpoints.length > 0 && <div className="mt-2 flex items-center gap-2 flex-wrap">
          {/* One group per run, folded into each option's label: every run for
              this dataset is listed, and bare "step N" rows from two runs read
              as stale duplicates (#182). StyledSelect has no optgroup, so the
              run's timestamp and status prefix each checkpoint's row instead. */}
          <StyledSelect
            accent="amber"
            value={selectedCheckpoint}
            onChange={value => { setSelectedCheckpoint(value); setApplyNote(''); setPresetLinkNote(''); }}
            className="w-auto min-w-[220px]"
            options={aitkRuns.filter(run => run.checkpoints.some(checkpoint => checkpoint.arPath && checkpoint.narPath))
              .flatMap(run => run.checkpoints.filter(checkpoint => checkpoint.arPath && checkpoint.narPath).map(checkpoint => ({
                value: checkpoint.dir,
                label: `${new Date(run.createdAt).toLocaleString()} · ${run.live ? 'running' : run.status} · step ${checkpoint.step}${checkpoint.dir === availableCheckpoints[0]?.dir ? ' (latest)' : ''}`,
              })))}
          />
          <button type="button" onClick={() => void applyCheckpoint()} disabled={activeBackendId !== 'yue2' || active || preparing || applyingCheckpoint || yue2RunAllActive || !selectedCheckpoint}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/50 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-40 flex items-center gap-1.5">
            {applyingCheckpoint ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
            {t('trainingStudio.yue2.method.useCheckpoint', 'Use for generation')}
          </button>
          <button type="button" onClick={() => void linkCheckpointPreset()} disabled={active || preparing || linkingPreset || yue2RunAllActive || !selectedCheckpoint}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/50 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-40 flex items-center gap-1.5">
            {linkingPreset ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
            {t('trainingStudio.yue2.method.linkAlbumPreset', 'Use in Lyric Studio album preset')}
          </button>
        </div>}
        {!runsError && availableCheckpoints.length === 0 && <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.noAuditionCheckpoint', 'No complete AR/NAR checkpoint is available yet.')}</p>}
        {applyNote && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{applyNote}</p>}
        {presetLinkNote && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{presetLinkNote}</p>}
      </div>}
      {ladderRunRec && <div className="mt-3 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.ladderTitle', 'Checkpoint ladder')} · {new Date(ladderRunRec.createdAt).toLocaleString()}{ladderRunRec.live ? ` · ${t('trainingStudio.yue2.method.ladderLive', 'training')}` : ''}</p>
        <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.ladderHint', 'Every saved checkpoint is a rung. Previews render while the run trains (one 300 s draft take on the dataset\'s first sung track); a rung with none yet says so, and Render adds a take. Score likeness and corruption 1-5, then press Use this rung on your pick: it becomes the dataset\'s adapter and you can clean up the rest. Scores also feed the Review page and "Finish scored" for batches.')}</p>
        <Yue2LadderReview datasetId={datasetId} datasetName={datasets.find(d => d.id === datasetId)?.name} run={ladderRunRec} previews={jointPreviews}
          renderOpts={{ seconds: form.preview?.seconds ?? 300, takes: 1, draft: (form.preview?.odeSteps ?? 12) > 0 && (form.preview?.odeSteps ?? 12) < 32 }}
          onChanged={() => setLadderNonce(n => n + 1)} onError={setError} idPrefix="train-rung" />
      </div>}
    </div>
  );
};

