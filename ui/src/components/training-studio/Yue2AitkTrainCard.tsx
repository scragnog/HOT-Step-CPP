import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { yue2SamplerSnapshot } from '../../stores/globalParamsStore';
import { AlertTriangle, Loader2, Play } from 'lucide-react';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { YUE2_JOINT_BASE_MATCHED_DEFAULTS, YUE2_JOINT_BASE_MATCHED_VALUES, YUE2_JOINT_FIXED_RECIPE_DROPOUTS, YUE2_JOINT_FORM_DEFAULTS, YUE2_JOINT_LADDER_PREVIEW, YUE2_JOINT_LEGACY_VALUES, YUE2_JOINT_LOKR_STOP, YUE2_JOINT_LORA_STOP, YUE2_JOINT_PRESETS, YUE2_JOINT_TUNED_KEYS } from '../../../../server/src/contracts/trainingRecipes';
import { pollWhile } from '../../queries/contract';
import { trainingQueries } from '../../queries/training-data/queries';
import { useJobSelection, useReload, useRunEvents, useScopedId } from '../../queries/training-runs/hooks';
import { activeJob as isActiveJob, runQueries, trainingWritePlan } from '../../queries/training-runs/queries';
import { waitForQuery } from '../../queries/wait';

import { preferencesApi } from '../../services/preferencesApi';
import { subscribeEventSource } from '../../services/sharedEventSource';
import {
  cancelJob,
  captionMissingYue2,
  dispatchYue2Batch,
  jobStreamUrl,
  pickLadderRun,
  startYue2AitkPrepare,
  updateYue2JointIntervals,
  type TrainingJobSummary,
  type TrainingMetricEvent,
  type TrainingStreamEvent,
  type Yue2AitkPrepareRequest,
  type Yue2JointPreviewOptions,
  type Yue2JointTrainRequest
} from '../../services/trainingApi';
import { currentWorkerRef } from '../../services/trainingOperations';
import * as preparationApi from '../../services/trainingPreparationApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { formatDurationMs } from '../../utils/trainingEta';
import { formatMB, yue2AdapterHalfBytes } from '../../utils/yue2AdapterSize';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { PreviewField, TrainingPreviewSettings, TrainingCheckpointLadder, TrainingRunPicker, TrainingPresetChoice } from './TrainingPreviewPanel';
import { TrainingRunPanel } from './Yue2StageCard';
import { Yue2LadderReview } from './Yue2LadderReview';
import { Yue2OptimizerFields } from './Yue2OptimizerFields';
import { Yue2SplitChart } from './Yue2SplitChart';
import { trainingRunEta } from './trainingRunEta';
import { lossDescentRate, lossTargetEta } from './trainingTargetEta';
import { TrainingEtaLine } from './TrainingEtaLine';
import { reduceYue2Metric, type Yue2Point } from './yue2ChartData';
import { attachHoldout, captureJointOverrides, submitJointStart } from './yue2JointCommand';
import { YUE2_JOINT_PRESETS_KEY, type Yue2JointPreset } from './yue2JointPresets';
import { ChangedSettingsRow, readStored, useNamedPresets, UserPresetRow, writeStored } from './trainingUserPresets';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';

const JOB_KEY = 'hs-yue2-aitk-job:';
export const FORM_KEY = 'hs-yue2-aitk-form:';
export const PREP_KEY = 'hs-yue2-aitk-prepare:';
// These dataset-scoped keys are local editing drafts; the accepted operation
// carries its own captured form and source revisions.
const METRIC_CAP = 2000;
// The card prunes other jobs' chart caches when selecting a job. Preserve
// the displayed job across page loads so terminal history needs no replay.
// `loss` is absent once the planner is frozen: the trainer then reports only
// the decoder's terms and the composite has no AR part. Those steps still
// count for progress, pace and the gradient norm.
type JointStepPoint = Yue2Point;
type JointMilestone = { epoch: number; loss: number; path: string; decoderOnly?: boolean };
function jointLossRate(points: JointStepPoint[]): number | null {
  return lossDescentRate(points.map(p => p.ma20).filter((v): v is number => typeof v === 'number'));
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
function jointCapEta(points: JointStepPoint[], form: Yue2JointTrainRequest) {
  const last = points[points.length - 1];
  const durations = points.map(p => p.stepMs).filter((ms): ms is number => typeof ms === 'number' && ms > 0).slice(-20);
  if (!last || !durations.length) return null;
  // Previews render in parallel with training here, so none are left to add.
  return trainingRunEta({ step: last.step, totalSteps: form.steps, msPerStep: durations.reduce((s, ms) => s + ms, 0) / durations.length, everySteps: 0, previews: [] });
}
/** The stop-mode branches' text; null when the run just stops at its step cap, which the shared ETA line states. */
function jointEta(points: JointStepPoint[], form: Yue2JointTrainRequest): string | null {
  const last = points[points.length - 1];
  const durations = points.map(p => p.stepMs).filter((ms): ms is number => typeof ms === 'number' && ms > 0).slice(-20);
  if (!last || !durations.length) return '';
  const pace = durations.reduce((sum, ms) => sum + ms, 0) / durations.length;
  const remaining = Math.max(0, form.steps - last.step);
  const capMs = jointCapEta(points, form)?.ms ?? 0;
  if (form.stopMode === 'kl' && form.targetKl && form.targetKl > 0) {
    const all = points.map(p => p.arKl).filter((v): v is number => typeof v === 'number');
    const reading = klStopReading(all, form.targetKlMode ?? 'mean');
    if (!reading) return `AR KL warming up · cap ${formatDurationMs(capMs)}`;
    const mean = reading.value;
    if (mean >= form.targetKl) return `KL target reached · cap ${formatDurationMs(capMs)}`;
    const older = points.slice(-40, -20).map(p => p.arKl).filter((v): v is number => typeof v === 'number');
    const rate = reading.slope ?? (older.length === 20 ? (mean - older.reduce((s, v) => s + v, 0) / 20) / 20 : 0);
    if (!(rate > 0)) return `AR KL ${mean.toFixed(2)} of ${form.targetKl} · cap ${formatDurationMs(capMs)}`;
    const stepsToTarget = (form.targetKl - mean) / rate;
    return stepsToTarget > remaining
      ? `KL ${mean.toFixed(2)} · target unlikely before cap · cap ${formatDurationMs(capMs)}`
      : `KL ${mean.toFixed(2)} · target ETA ${formatDurationMs(Math.max(1, stepsToTarget) * pace)}`;
  }
  if (form.stopMode !== 'loss' || !(form.targetLoss && form.targetLoss > 0))
    return null;
  return lossTargetEta({ current: last.ma20, rate: jointLossRate(points), target: form.targetLoss, remaining, paceMs: pace, capMs });
}
/** A named snapshot of the training settings. Per-run and per-machine values
 *  (dataset/checkpoint/output paths, resume record) are deliberately not part
 *  of a preset: a preset answers "how do I train", never "against which run". */
const PRESET_EXCLUDED_KEYS: ReadonlySet<keyof Yue2JointTrainRequest> = new Set([
  'trainingMethod', 'autoPrepare', 'preparation', 'checkpoint', 'dataset', 'output', 'resume', 'alignmentEnabled', 'gpuUuid',
]);
const FIXED_RECIPE_SCHEDULE = { steps: 1500, seed: 42, saveEvery: 200 } as const;
/** The recipe default for the chosen adapter type: picking LoRA/DoRA fills
 *  alpha and the KL stop, which are that type's defaults, not edits (#318).
 *  The adapter type itself still shows as changed. */
export function baselineFor(f: Pick<Yue2JointTrainRequest, 'adapterType' | 'rank' | 'method'>): Record<string, unknown> {
  return { ...DEFAULT_FORM, ...((f.adapterType ?? 'lora') === 'lokr' ? LOKR_STOP : { alpha: f.rank ?? 64, ...LORA_STOP }),
    ...(f.method === 'fixed-recipe' ? FIXED_RECIPE_SCHEDULE : {}) } as unknown as Record<string, unknown>;
}
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
// parallel with training (everySteps 0 = no pauses): 300 s draft takes
// (12 decoder steps), the same settings the Refine tab's ladders used.
// 2026-09-30: two takes, one on the fixed seed and one on a fresh random seed.
const LADDER_PREVIEW: Yue2JointPreviewOptions = YUE2_JOINT_LADDER_PREVIEW as Yue2JointPreviewOptions;
function defaultPreview(_everySteps: number): Yue2JointPreviewOptions { return { ...LADDER_PREVIEW }; }
const DEFAULT_FORM: Yue2JointTrainRequest = YUE2_JOINT_FORM_DEFAULTS as Yue2JointTrainRequest;
// Tuned-recipe knobs the server forces under base-matched; cleared from
// stored forms so a saved value from before 2026-09-27 cannot linger.
const TUNED_KEYS = YUE2_JOINT_TUNED_KEYS;
const LORA_STOP = YUE2_JOINT_LORA_STOP;
const LOKR_STOP = YUE2_JOINT_LOKR_STOP;
// Presets (2026-09-27 ear test on a full album). All three are the
// base-matched recipe (the server applies its fixed parts); they differ in
// updates, songs per update, decoder crop and checkpoint spacing; all three
// train a dim-128 LoKr (2026-09-29, Rob, after an overnight batch at these
// settings). All three train the decoder on 60 s crops: a blind test scored
// crops level with whole songs at about 40% less time, and Rob hears no gain
// from whole songs. Fast 100 x 4, Balanced 200 x 4, Thorough 300 x 8; each
// saves ten rungs. minutes = the measured 17 s a cropped 4-song update on a
// 5090 (x2 for 8 songs) x updates, for the relative-time label.
// Legacy (2026-09-29, Rob): the tuned recipe the card shipped before
// base-matched (v1.3.4's defaults), for anyone who wants its speed back.
// One deliberate gap: lyric timing stays off (it needs stems + alignment
// first).
// It stops at the planner's KL target, so steps is a cap; minutes assumes
// the ~250 updates of 1 song a Prodigy run took to KL 1.2. Target 1.0
// since 2026-10-06: past ~1 the planner lost the lead sheet's tempo late in a take.
const LEGACY_VALUES: Partial<Yue2JointTrainRequest> = YUE2_JOINT_LEGACY_VALUES as Partial<Yue2JointTrainRequest>;
// Leaving Legacy: the tuned-only knobs go (the server forces them anyway)
// and the optimizer returns to the base-matched presets' pick.
const BASE_MATCHED_VALUES: Partial<Yue2JointTrainRequest> = YUE2_JOINT_BASE_MATCHED_VALUES as Partial<Yue2JointTrainRequest>;
const PRESETS = YUE2_JOINT_PRESETS;
const presetValues = (p: typeof PRESETS[number]): Partial<Yue2JointTrainRequest> => ({
  ...(p.key === 'legacy' ? LEGACY_VALUES : { ...BASE_MATCHED_VALUES, stopMode: 'steps' as const }),
  steps: p.steps, gradAccum: p.gradAccum, narCropFrames: p.narCropFrames, saveEvery: p.saveEvery, lokrDim: p.lokrDim });
const activePreset = (f: Yue2JointTrainRequest) => f.method === 'fixed-recipe' ? undefined : PRESETS.find(p => (f.method === 'tuned') === (p.key === 'legacy')
  && (p.key !== 'legacy' || f.targetKl === LEGACY_VALUES.targetKl) && f.steps === p.steps && (f.gradAccum ?? 4) === p.gradAccum
  && (f.narCropFrames ?? 0) === p.narCropFrames && f.saveEvery === p.saveEvery && (f.adapterType !== 'lokr' || f.lokrDim === p.lokrDim))?.key;
/** Presets show their cost relative to Balanced rather than wall-clock times
 *  that only hold for one GPU and one dataset. */
const presetTime = (p: { minutes: number }) => `${Number((p.minutes / PRESETS[2].minutes).toFixed(1))}×`;
/** Mirror of BASE_MATCHED_DEFAULTS in server/src/services/training/yue2JointTrainRunner.ts,
 *  shown as each blank field's placeholder. The server fills blanks from its own copy. */
const BASE_MATCHED_DEFAULTS = YUE2_JOINT_BASE_MATCHED_DEFAULTS;
type PrepareForm = Yue2AitkPrepareRequest;



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
    writeStored(migration, 1);
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
    writeStored(recipeA, 1);
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
    writeStored(narBudget, 1);
  }
  // Prodigy default (2026-09-21): Recipe A had moved everyone to adamw, so a
  // stored adamw is the old default, not a choice, and moves with it.
  const prodigy = `${FORM_KEY}${datasetId}:defaults-prodigy`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(prodigy)) {
    if (stored.optimizer === 'adamw') stored.optimizer = 'prodigy';
    writeStored(prodigy, 1);
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
    writeStored(lokrDefault, 1);
  }
  // LoKr stop recipe (2026-09-22): a LoKr form still on LoRA's KL 1.4 moves to
  // the LoKr pair; a deliberately edited target stays.
  const lokrStop = `${FORM_KEY}${datasetId}:defaults-lokr-stop`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(lokrStop)) {
    if (stored.adapterType === 'lokr') {
      if (stored.targetKl === undefined || stored.targetKl === 1.4) stored.targetKl = LOKR_STOP.targetKl;
      if (stored.narLrScale === undefined) stored.narLrScale = LOKR_STOP.narLrScale;
    }
    writeStored(lokrStop, 1);
  }
  // LoKr recipe 2 (2026-09-22): planner 0.3 -> 0.6, decoder 0.5 -> 1.0, for
  // LoKr forms still on the first recipe's values.
  const lokrRecipe2 = `${FORM_KEY}${datasetId}:defaults-lokr-recipe-2`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(lokrRecipe2)) {
    if (stored.adapterType === 'lokr') {
      if (stored.plannerLrScale === undefined || stored.plannerLrScale === 0.3) stored.plannerLrScale = LOKR_STOP.plannerLrScale;
      if (stored.narLrScale === undefined || stored.narLrScale === 0.5) stored.narLrScale = LOKR_STOP.narLrScale;
    }
    writeStored(lokrRecipe2, 1);
  }
  // LoKr recipe 3 (2026-09-22): target KL 0.9 -> 1.0.
  const lokrRecipe3 = `${FORM_KEY}${datasetId}:defaults-lokr-recipe-3`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(lokrRecipe3)) {
    if (stored.adapterType === 'lokr' && stored.targetKl === 0.9) stored.targetKl = LOKR_STOP.targetKl;
    writeStored(lokrRecipe3, 1);
  }
  // Preview 90 s (2026-09-22): a form still on the old 40 s default moves.
  const preview90 = `${FORM_KEY}${datasetId}:defaults-preview-90`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(preview90)) {
    if (stored.preview && stored.preview.seconds === 40) stored.preview = { ...stored.preview, seconds: 90, previewMaxFrames: 2250 };
    writeStored(preview90, 1);
  }
  // 2026-09-23 defaults: save every 25, cautious on, LoKr KL 1.1. Values still
  // on the previous defaults move; deliberate ones stay.
  const d0923 = `${FORM_KEY}${datasetId}:defaults-2026-09-23`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(d0923)) {
    if (stored.saveEvery === 50) stored.saveEvery = 25;
    if (stored.optimizer !== 'adamw' && (stored.cautious === undefined || stored.cautious === false)) stored.cautious = true;
    if (stored.adapterType === 'lokr' && stored.targetKl === 1.0) stored.targetKl = LOKR_STOP.targetKl;
    writeStored(d0923, 1);
  }
  // Planner freeze + caption dropout (2026-09-23): fields a form never set
  // take the new defaults; deliberate values stay.
  const freeze = `${FORM_KEY}${datasetId}:defaults-planner-freeze`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(freeze)) {
    if (stored.narExtraSteps === undefined) stored.narExtraSteps = DEFAULT_FORM.narExtraSteps;
    if (stored.captionDropout === undefined) stored.captionDropout = DEFAULT_FORM.captionDropout;
    writeStored(freeze, 1);
  }
  // Presets (2026-09-24): stored forms move to Balanced once.
  // 2026-09-24: presets end at the KL; the decoder trains on during refinement.
  const klEnd = `${FORM_KEY}${datasetId}:defaults-kl-end`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(klEnd)) {
    if (stored.stopMode === 'kl' && stored.narExtraSteps === stored.steps) stored.narExtraSteps = 0;
    writeStored(klEnd, 1);
  }
  const presets = `${FORM_KEY}${datasetId}:defaults-presets`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(presets)) {
    Object.assign(stored, presetValues(PRESETS[2]));
    writeStored(presets, 1);
  }
  const autoRefine = `${FORM_KEY}${datasetId}:defaults-auto-refine`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(autoRefine)) {
    if (stored.autoRefine === undefined) stored.autoRefine = DEFAULT_FORM.autoRefine;
    writeStored(autoRefine, 1);
  }
  const recon = `${FORM_KEY}${datasetId}:defaults-recon-stop`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(recon)) {
    if (stored.reconStop === undefined) { stored.reconStop = DEFAULT_FORM.reconStop; stored.reconStopWindow = DEFAULT_FORM.reconStopWindow; }
    // The old two-point knee shipped with 3; the fitted knee needs 10 points.
    if (stored.reconStopWindow === 3) stored.reconStopWindow = DEFAULT_FORM.reconStopWindow;
    writeStored(recon, 1);
  }
  const spike = `${FORM_KEY}${datasetId}:defaults-spike-guard`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(spike)) {
    if (stored.spikeFactor === undefined) { stored.spikeFactor = DEFAULT_FORM.spikeFactor; stored.spikeStop = DEFAULT_FORM.spikeStop; stored.spikeStopWindow = DEFAULT_FORM.spikeStopWindow; }
    writeStored(spike, 1);
  }
  // 2026-09-25 defaults: wsd schedule, AR KL target 1.0. A form that never
  // touched the schedule, or is still sitting on the old KL 1.2, moves;
  // anything set deliberately stays.
  const d0925 = `${FORM_KEY}${datasetId}:defaults-2026-09-25`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(d0925)) {
    if (stored.lrSchedule === undefined) stored.lrSchedule = 'wsd';
    if (stored.targetKl === 1.2) stored.targetKl = 1.0;
    writeStored(d0925, 1);
  }
  // 2026-09-25 (Rob): checkpoint previews are the Refine tab's job; a primary
  // run renders none by default. Turned off once; tick it again to keep it.
  const previewOff = `${FORM_KEY}${datasetId}:defaults-2026-09-25-preview-off`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(previewOff)) {
    if (stored.preview?.enabled) stored.preview = { ...stored.preview, enabled: false };
    writeStored(previewOff, 1);
  }
  // 2026-09-27 (Rob): the base-matched recipe replaces the tuned one. Every
  // stored form moves to it once, on the Balanced preset; the tuned-only
  // knobs are cleared (the server forces them anyway).
  const baseMatched = `${FORM_KEY}${datasetId}:defaults-base-matched-2026-09-27`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(baseMatched)) {
    for (const key of TUNED_KEYS) delete (stored as Record<string, unknown>)[key];
    stored.method = 'base-matched'; stored.stopMode = 'steps'; stored.optimizer = 'adamw-lm'; stored.cautious = false;
    stored.autoRefine = false; stored.cursorWeight = 0;
    Object.assign(stored, presetValues(PRESETS[2]));
    writeStored(baseMatched, 1);
  }
  // 2026-09-27 (Rob): the run's own ladder is previewed and scored on this
  // page, so previews are on and the engine stays up. Turned on once.
  const ladderPreviews = `${FORM_KEY}${datasetId}:defaults-ladder-previews-2026-09-27`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(ladderPreviews)) {
    stored.preview = { ...LADDER_PREVIEW, ...(stored.preview?.caption ? { caption: stored.preview.caption } : {}), ...(stored.preview?.lyrics ? { lyrics: stored.preview.lyrics } : {}) };
    stored.stopEngine = false;
    writeStored(ladderPreviews, 1);
  }
  // 2026-09-28 reset: the chain above runs on every new dataset's empty form
  // too, and Recipe A (lr undefined -> 1e-4) plus NAR budget (1e-4 -> 2e-4)
  // left a tuned-era 2e-4 that the base-matched step never cleared, so runs
  // trained at twice the recipe's rate. Every stored form goes back to the
  // recipe defaults once; only the per-run paths survive.
  const resetAll = `${FORM_KEY}${datasetId}:defaults-reset-2026-09-28`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(resetAll)) {
    for (const key of Object.keys(stored) as (keyof Yue2JointTrainRequest)[]) {
      if (!PRESET_EXCLUDED_KEYS.has(key)) delete stored[key];
    }
    writeStored(`${FORM_KEY}${datasetId}`, stored);
    writeStored(resetAll, 1);
  }
  // 2026-10-15: one take per rung by default (was two since 2026-09-30); the
  // count is now a field in the previews section.
  const oneTake = `${FORM_KEY}${datasetId}:defaults-ladder-one-take-2026-10-15`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(oneTake)) {
    stored.preview = { ...LADDER_PREVIEW, ...stored.preview, takes: 1 };
    writeStored(`${FORM_KEY}${datasetId}`, stored);
    writeStored(oneTake, 1);
  }
  // Legacy is the one preset that runs the tuned recipe; anything else is base-matched.
  return { ...DEFAULT_FORM, ...stored, preview: { ...LADDER_PREVIEW, ...stored.preview }, method: stored.method === 'tuned' || stored.method === 'fixed-recipe' ? stored.method : 'base-matched' };
}
function isJointJob(job: TrainingJobSummary, datasetId: string): boolean {
  return job.datasetId === datasetId && job.kind === 'yue2-joint-train';
}
function isPrepareJob(job: TrainingJobSummary, datasetId: string): boolean {
  return job.datasetId === datasetId && job.kind === 'yue2-prepare-aitk';
}

export function createPresetLoadGate() {
  let pending = false;
  return {
    get pending() { return pending; },
    begin() { if (pending) return false; pending = true; return true; },
    release() { pending = false; },
    capture<T>(fn: () => T): T {
      if (pending) throw new Error('Wait for the preset to finish loading before starting training.');
      return fn();
    },
  };
}

/** GH #307: change the checkpoint and preview interval of the run training
 *  now. Blank fields stay as they are; the trainer takes the rest up between
 *  steps. Preview every shows only on a run that pauses for previews. */
const LiveIntervals: React.FC<{ datasetId: string; jobId: string; worker?: string | null; options?: Record<string, unknown> }> = ({ datasetId, jobId, worker, options }) => {
  const { t } = useTranslation();
  const preview = options?.preview as Yue2JointPreviewOptions | undefined;
  const [saveEvery, setSaveEvery] = useState('');
  const [previewEvery, setPreviewEvery] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const current = { saveEvery: typeof options?.saveEvery === 'number' ? options.saveEvery : undefined, previewEvery: preview?.enabled && preview.everySteps > 0 ? preview.everySteps : undefined };
  const apply = async () => {
    setBusy(true); setNote('');
    try {
      const done = await updateYue2JointIntervals(datasetId, jobId, {
        ...(saveEvery ? { saveEvery: Number(saveEvery) } : {}), ...(previewEvery ? { previewEvery: Number(previewEvery) } : {}) }, { worker });
      setSaveEvery(''); setPreviewEvery('');
      setNote(t('trainingStudio.yue2.method.intervalsApplied', 'Now: checkpoint every {{save}}', { save: done.saveEvery })
        + (done.previewEvery ? t('trainingStudio.yue2.method.intervalsAppliedPreview', ', preview every {{preview}}', { preview: done.previewEvery }) : ''));
    } catch (err) { setNote(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };
  const field = 'w-24 px-2 py-1 rounded-lg text-xs bg-white/70 dark:bg-black/20 border border-zinc-300/70 dark:border-white/10 text-zinc-800 dark:text-zinc-100';
  return (
    <div className="mt-2 flex items-end gap-3 flex-wrap">
      <label className="flex flex-col gap-1">
        <ParamLabel label={t('trainingStudio.yue2.method.liveSaveEvery', 'Checkpoint every')} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
          info={t('trainingStudio.yue2.method.liveSaveEveryInfo', 'Change how often this run saves a checkpoint, without stopping it. Counted from step 0: 50 saves at 50, 100, 150 and so on. Takes effect after the step in progress.')} />
        <input className={field} type="number" min={1} value={saveEvery} placeholder={current.saveEvery === undefined ? '' : String(current.saveEvery)} onChange={e => setSaveEvery(e.target.value)} />
      </label>
      {current.previewEvery !== undefined && <label className="flex flex-col gap-1">
        <ParamLabel label={t('trainingStudio.yue2.method.livePreviewEvery', 'Preview every')} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
          info={t('trainingStudio.yue2.method.livePreviewEveryInfo', 'Change how many steps this run trains between preview pauses, counted from the last pause. If the new pause is already behind the current step, the run pauses for a preview after the step in progress.')} />
        <input className={field} type="number" min={1} value={previewEvery} placeholder={String(current.previewEvery)} onChange={e => setPreviewEvery(e.target.value)} />
      </label>}
      <button type="button" onClick={() => void apply()} disabled={busy || (!saveEvery && !previewEvery)}
        className="px-3 py-1 rounded-lg text-xs font-semibold bg-amber-500/80 text-black hover:bg-amber-400 disabled:opacity-40">
        {t('trainingStudio.yue2.method.liveApply', 'Apply to this run')}
      </button>
      {note && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{note}</span>}
    </div>
  );
};

export const Yue2AitkTrainCard: React.FC<{ datasetId: string; legacyManifest?: string; cursorReady?: boolean; lyricTiming: boolean; onLyricTimingChange: (value: boolean) => void; onTimingLockedChange?: (locked: boolean) => void; exposeStart?: (fn: () => Promise<Record<string, unknown>>) => void }> = ({ datasetId, legacyManifest, cursorReady = false, lyricTiming, onLyricTimingChange, onTimingLockedChange, exposeStart }) => {
  const { t } = useTranslation();
  const [storedForm, setForm] = useState<Yue2JointTrainRequest>(() => readStoredForm(datasetId));
  // Saved forms, imported presets and resume records can contain old overrides.
  // Display, persist and submit the same schedule the server will run.
  const form = useMemo(() => storedForm.method === 'fixed-recipe'
    ? { ...storedForm, ...FIXED_RECIPE_SCHEDULE } : storedForm, [storedForm]);
  const latestForm = useRef(form);
  latestForm.current = form;
  const latestLyricTiming = useRef(lyricTiming);
  latestLyricTiming.current = lyricTiming;
  const presetLoadRequest = useRef(0);
  const trainingWorker = useTrainingStore(s => s.trainingWorker);
  const client = useQueryClient();
  // Keep the canonical Query owner, with the persisted terminal chart also
  // gating the bridge if the HTTP snapshot briefly lags completion.
  const [jobId, selectJob] = useScopedId(trainingWorker, datasetId);
  const jobQuery = useQuery({ ...trainingQueries.job(trainingWorker, jobId), enabled: !!jobId,
    refetchInterval: pollWhile<TrainingJobSummary>(isActiveJob) });
  const job = jobQuery.data ?? null;
  const cachedTerminal = useMemo(() => {
    const saved = readStored<{ status?: TrainingJobSummary['status'] }>(`${JOB_KEY}${datasetId}:metrics:${jobId}`, {});
    return !!saved.status && !isActiveJob({ status: saved.status });
  }, [datasetId, trainingWorker, jobId]);
  useRunEvents(job && isActiveJob(job) && !cachedTerminal ? job.id : null, trainingWorker);
  const setJob = (selected: TrainingJobSummary | null) => {
    if (selected) client.setQueryData(trainingQueries.job(trainingWorker, selected.id).queryKey, selected);
    selectJob(selected?.id ?? '');
  };
  const [jointPipelineId, setJointPipelineId] = useScopedId(trainingWorker, datasetId);
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
  const [captionJobId, setCaptionJobId] = useState<string | null>(null);
  // The progress line renders the one job cache entry. The waiter below polls
  // that same entry, so this observer only reads it — a second cadence here
  // would fetch the job twice.
  const captionQuery = useQuery({
    ...trainingQueries.job(trainingWorker, captionJobId ?? ''),
    enabled: !!captionJobId,
    refetchInterval: false,
  });
  const captioning = captionJobId ? captionQuery.data ?? null : null;
  // No onError: the caller's catch (run, or the run-all orchestrator) sets the
  // error line, as before; the service is silent so that line is sole notifier.
  const captionStartMut = useMutation({
    mutationFn: async (cap: { provider: 'gemini' | 'moss'; model?: string }) => {
      const started = await captionMissingYue2(datasetId, cap, { worker: trainingWorker });
      // The stages list shows the captioning job as soon as it starts.
      await trainingWritePlan(client, trainingWorker);
      if (!started.jobId) return;
      setCaptionJobId(started.jobId);
      // The one non-React waiter owns the cadence (1.5 s active, backing off
      // while failing) instead of a hand-rolled loop with its own timer.
      const j = await waitForQuery<TrainingJobSummary>(
        trainingQueries.job(trainingWorker, started.jobId),
        job => isActiveJob(job),
      );
      setCaptionJobId(null);
      if (j.status !== 'done') throw new Error(`Captioning failed: ${j.error || j.status}`);
      const left = (await captionMissingYue2(datasetId, { checkOnly: true }, { worker: trainingWorker })).missing ?? 0;
      if (left) throw new Error(`${left} track(s) still have no YuE2 caption after captioning; see the Label log on the Dataset page. Not training on the ACE captions.`);
    },
  });
  const captionMissing = () => { if (!autoCaption) return; return captionStartMut.mutateAsync(autoCaption); };
  const [commandError, setError] = useState('');
  const [starting, setStarting] = useState(false);
  // Saved presets keep their names and settings; only the leftover tuned-era
  // learning rate (see the 2026-09-28 reset) is dropped from them, once.
  const presetState = useNamedPresets<Yue2JointTrainRequest>({
    storageKey: YUE2_JOINT_PRESETS_KEY, fileKind: 'hot-step-yue2-joint-preset',
    snapshot: () => snapshotPresetSettings(form, lyricTiming), excluded: PRESET_EXCLUDED_KEYS as ReadonlySet<string>,
    initial: () => {
      const list = readStored<Yue2JointPreset[]>(YUE2_JOINT_PRESETS_KEY, []);
      const flag = `${YUE2_JOINT_PRESETS_KEY}:lr-reset-2026-09-28`;
      if (typeof window === 'undefined' || window.localStorage.getItem(flag)) return list;
      const cleaned = list.map(p => {
        if (p.settings.lr !== 2e-4 && p.settings.lr !== 1e-4) return p;
        const { lr: _lr, ...settings } = p.settings;
        return { ...p, settings };
      });
      writeStored(YUE2_JOINT_PRESETS_KEY, cleaned);
      writeStored(flag, 1);
      return cleaned;
    },
  });
  const { presets, setError: setPresetError } = presetState;
  const [presetLoading, setPresetLoading] = useState(false);
  const presetLoadGate = useRef(createPresetLoadGate());
  const pendingPresetApplication = useRef<{ form: Yue2JointTrainRequest; lyricTiming: boolean } | null>(null);
  useEffect(() => {
    const pending = pendingPresetApplication.current;
    // Match the applied state object; the fixed-recipe display is a normalized copy.
    if (pending && storedForm === pending.form && lyricTiming === pending.lyricTiming) {
      pendingPresetApplication.current = null;
      presetLoadGate.current.release();
      setPresetLoading(false);
    }
  }, [storedForm, lyricTiming]);
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
  const prepareQuery = useQuery(runQueries.prepare(trainingWorker, datasetId));
  const bases = prepareQuery.data?.bases ?? [];
  const defaultBase = prepareQuery.data?.defaultBase ?? '';
  const defaultDevice = prepareQuery.data?.defaultDevice ?? '';
  const gpuQuery = useQuery({ ...runQueries.gpus(), enabled: defaultDevice === 'CUDA0' });
  const nvidiaGpus = gpuQuery.data?.gpus ?? [];
  const [prepareJob, setPrepareJob] = useJobSelection(trainingWorker, datasetId);
  const [prepareManifest, setPrepareManifest] = useState('');
  const [appliedPrepareJobId, setAppliedPrepareJobId] = useState(() => readStored<string>(`${PREP_KEY}${datasetId}:applied`, ''));
  const defaultsAvailable = prepareQuery.data ? !!prepareQuery.data.defaults : null;
  const missingDefaults = prepareQuery.data?.missing ?? [];
  const refreshDefaults = () => { void prepareQuery.refetch(); };
  const yue2RunAllActive = useTrainingStore(s => s.yue2RunAllActive);
  const batchDraft = useTrainingStore(s => s.yue2BatchDraft);
  const datasets = useTrainingStore(s => s.datasets);
  const setBatchDraft = useTrainingStore(s => s.setYue2BatchDraft);
  const startBatch = useTrainingStore(s => s.startYue2Batch);
  const setPhase = useTrainingStore(s => s.setPhase);
  const [batchStarting, setBatchStarting] = useState(false);
  const [batchClearCache, setBatchClearCache] = useState(false);
  const setTrainingWorker = useTrainingStore(s => s.setTrainingWorker);
  const workersQuery = useQuery({ ...trainingQueries.workers(), enabled: !!batchDraft?.length && !trainingWorker });
  const workers = workersQuery.data ?? [];
  const [runOn, setRunOn] = useState('');
  // A batch draft turns this card into the recipe editor for N datasets: the
  // form is the same, Start sends it to the server-side batch instead of one
  // run, and the per-dataset paths are resolved per item by the runner.
  // The batch write targets runOn, so its invalidation plans for runOn too —
  // planning the local trainingWorker left the worker's queues stale. No
  // onError: runBatch's catch sets the error line (sole notifier).
  const dispatchMut = useMutation({
    mutationFn: async ({ worker, ...input }: { worker: string } & Parameters<typeof dispatchYue2Batch>[1]) => {
      await dispatchYue2Batch(worker, input);
      return worker;
    },
    onSuccess: worker => { void trainingWritePlan(client, worker); },
  });
  const runBatch = async () => {
    if (!batchDraft?.length || presetLoadGate.current.pending) return;
    setBatchStarting(true); setError('');
    try {
      const batchTiming = form.method === 'fixed-recipe' ? false : lyricTiming;
      const timingWeight = batchTiming
        ? (typeof form.cursorWeight === 'number' && Number.isFinite(form.cursorWeight) ? form.cursorWeight : 0.08) : 0;
      const recipe = { ...form, gpuUuid: runOn ? undefined : form.gpuUuid,
        device: runOn ? '' : defaultDevice === 'CUDA0' ? 'CUDA0' : form.device,
        autoCaption: autoCaption ?? (false as const), cursorWeight: timingWeight, dataset: '', output: '', resume: '',
        ...(form.preview ? { preview: { ...defaultPreview(form.saveEvery), ...form.preview, sampler: yue2SamplerSnapshot(),
          everySteps: form.preview.parallel ? 0 : form.saveEvery, previewMaxFrames: Math.max(8, Math.min(360, form.preview.seconds || 300)) * 25 } } : {}) };
      if (runOn) {
        await dispatchMut.mutateAsync({ worker: runOn, datasetIds: batchDraft, lyricTiming: batchTiming, clearCache: batchClearCache, recipe });
        setBatchDraft(null);
        await setTrainingWorker(runOn);
        return;
      }
      await startBatch({ datasetIds: batchDraft, lyricTiming: batchTiming, clearCache: batchClearCache, recipe });
      setPhase('train');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { setBatchStarting(false); }
  };
  const runsQuery = useQuery({ ...runQueries.jointRuns(datasetId),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof import('../../services/trainingApi').listYue2AitkRuns>>>(data =>
      !data || data.runs.some(run => run.live) || job?.status === 'queued' || job?.status === 'running', 5000) });
  const aitkRuns = runsQuery.data?.runs ?? [];
  const [resumeChoice, setResumeChoice] = useState('');
  const [liveMetric, setLiveMetric] = useState<TrainingMetricEvent | null>(null);
  const [gateReading, setGateReading] = useState<TrainingMetricEvent['plannerGate']>();
  const gate = form.plannerGate;
  // On a resume the run's own gate and split apply; the preview is for new runs.
  const gateResuming = !!resumeChoice || !!form.resume?.trim();
  const holdoutQuery = useQuery({ ...runQueries.holdout(trainingWorker, datasetId, gate?.evalSeed ?? 42), enabled: !!gate && !gateResuming });
  // What Decoder steps held before the gate raised it, so switching the gate off puts it back.
  const narBeforeGate = useRef<number | undefined>(undefined);
  const [chartVram, setChartVram] = useState<number>();
  // AITK's joint runner emits step metrics without an epoch stream. Keep the
  // history here in the same step-domain shape used by the legacy YuE2 chart.
  // The job SSE endpoint replays its buffer, so this also reconstructs the
  // curve after a reload or EventSource reconnect.
  const [stepHistory, setStepHistory] = useState<JointStepPoint[]>([]);
  const [historyUnavailable, setHistoryUnavailable] = useState(false);
  // The KL stop reading at every step, for the chart: the same computation the
  // engine makes, so the line crosses the target where the run stops.
  // The planner freezes at its KL target: from then on steps carry no loss.
  const frozenAt = useMemo(() => {
    const first = stepHistory.findIndex(p => p.frozen);
    return first > 0 && !stepHistory[first - 1].frozen && stepHistory[first - 1].loss !== undefined ? stepHistory[first - 1].step : undefined;
  }, [stepHistory]);
  const [milestones, setMilestones] = useState<JointMilestone[]>([]);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [showJobLogs, setShowJobLogs] = useState(false);


  useEffect(() => {
    // The dataset's own manifest always wins, including when it arrives late.
    if (legacyManifest && prepare.legacyManifest !== legacyManifest) {
      setPrepare(previous => ({ ...previous, legacyManifest }));
    }
  }, [legacyManifest, prepare.legacyManifest]);

  useEffect(() => {
      const defaults = prepareQuery.data?.defaults;
      if (!defaults) return;
      // A saved path with the default's file name in another folder is the same
      // model from before a data-folder move, so the installed default wins.
      const fileName = (file: string) => file.split(/[\\/]/).pop()?.toLowerCase() ?? '';
      const takeDefault = (saved: unknown, value: unknown) => typeof value === 'string'
        && (!saved || (typeof saved === 'string' && saved !== value && fileName(saved) === fileName(value)));
      setPrepare(previous => ({
        ...previous,
        ...Object.fromEntries(Object.entries(defaults).filter(([key, value]) => key !== 'models' && takeDefault(previous[key as keyof PrepareForm], value))),
        models: { ...previous.models, ...Object.fromEntries(Object.entries(defaults.models ?? {}).filter(([key, value]) => takeDefault(previous.models[key as keyof PrepareForm['models']], value))) },
      }));
  }, [datasetId, trainingWorker, prepareQuery.data]);

  useReload(runQueries.jointRuns(datasetId).queryKey, `${job?.id ?? ''}:${job?.status ?? ''}`);
  // The ladder shown under the run: the run picked (on the Review page or in
  // the ladder's run list), else the run this card's job made, else the live
  // one, else the newest. Its previews are fetched for that run.
  const pickedLadderRun = useTrainingStore(s => s.refineLadderRun);
  const setPickedLadderRun = useTrainingStore(s => s.setRefineLadderRun);
  // A job training on a worker has no local run record until the mirror's
  // first pass lands it. While that's true, an explicit pick still wins, but
  // there's no history to fall back to: an older local run must not silently
  // stand in for the live one underway.
  const activeJob = job && isJointJob(job, datasetId) && (job.status === 'queued' || job.status === 'running') ? job : undefined;
  const ladderRunRec = pickLadderRun(aitkRuns, pickedLadderRun, activeJob?.id);
  const ladderRunId = ladderRunRec?.jobId;
  const awaitingMirror = !ladderRunRec && !!activeJob && !!trainingWorker;

  const previewsQuery = useQuery({ ...runQueries.previews(datasetId, ladderRunId ?? ''), enabled: !!ladderRunId,
    refetchInterval: pollWhile(() => !!ladderRunRec?.live || job?.status === 'queued' || job?.status === 'running'
      || typeof job?.finishedAt === 'number' && Date.now() - job.finishedAt < 30 * 60_000, 5000) });
  const jointPreviews = previewsQuery.data?.previews ?? [];

  useEffect(() => {
    setLiveMetric(null);
    setGateReading(undefined);
    setChartVram(undefined);
    setStepHistory([]);
    setHistoryUnavailable(false);
    setMilestones([]);
    setJobLogs([]);
    setShowJobLogs(false);
    if (!job?.id) return;
    const metricKey = `${JOB_KEY}${datasetId}:metrics:${job.id}`;
    // Only the job on screen keeps a cached chart.
    try {
      for (const key of Object.keys(window.localStorage)) {
        if (key.startsWith(JOB_KEY) && key.includes(':metrics:') && !key.startsWith(metricKey)) window.localStorage.removeItem(key);
      }
    } catch { /* storage unavailable */ }
    const saved = readStored<{ steps?: JointStepPoint[]; milestones?: JointMilestone[]; status?: TrainingJobSummary['status']; historyUnavailable?: boolean }>(metricKey, {});
    if (saved.steps?.length) setStepHistory(saved.steps);
    const savedMilestones = readStored<JointMilestone[]>(`${metricKey}:milestones`, saved.milestones ?? []);
    if (savedMilestones.length) setMilestones(savedMilestones);
    // An uncached finished job needs one replay. Legacy series lack status,
    // so retain their history when the job itself is known terminal.
    const complete = !saved.historyUnavailable && (saved.status ? !isActiveJob({ status: saved.status })
      : !!saved.steps?.length && !isActiveJob(job));
    if (complete) return;
    let steps = saved.steps ?? [];
    let points = savedMilestones;
    let status: TrainingJobSummary['status'] = 'running';
    let unavailable = false;
    let dirty = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const persist = () => {
      clearTimeout(timer);
      timer = undefined;
      if (!dirty || unavailable) return;
      writeStored(metricKey, { steps, status });
      writeStored(`${metricKey}:milestones`, points);
      dirty = false;
    };
    const schedulePersist = () => {
      dirty = true;
      timer ??= setTimeout(persist, 1000);
    };
    const stop = subscribeEventSource(jobStreamUrl(job.id, { worker: trainingWorker }), { message: data => {
      try {
        const item = JSON.parse(data) as TrainingStreamEvent;
        if (item.type === 'metric' && item.metric === 'vram') {
          setChartVram(item.usedMb ?? item.vramMb ?? (item.totalMb !== undefined && item.freeMb !== undefined ? item.totalMb - item.freeMb : undefined));
        } else if (item.type === 'metric' && item.metric === 'step') {
          setLiveMetric(item);
          if (typeof item.step === 'number' && Number.isFinite(item.step)) {
            steps = reduceYue2Metric(steps, { ...item,
              // Older trainers lack cumulative time; retain the job-clock fallback.
              trainMs: item.trainMs ?? (typeof job.startedAt === 'number' && typeof item.ts === 'number'
                ? Math.max(0, item.ts - job.startedAt) : undefined),
            }, METRIC_CAP);
            setStepHistory(steps);
            schedulePersist();
          }
        } else if (item.type === 'metric' && item.metric === 'eval' && item.plannerGate) {
          setGateReading(item.plannerGate);
        } else if (item.type === 'metric' && item.metric === 'milestone'
          && typeof item.step === 'number' && typeof item.path === 'string') {
          points = [...points.filter(point => point.path !== item.path), { epoch: item.step, loss: item.loss ?? 0, path: item.path,
            ...(item.decoderOnly ? { decoderOnly: true } : {}) }];
          setMilestones(points);
          schedulePersist();
        } else if (item.type === 'log') {
          const stamp = new Date(item.ts).toLocaleTimeString();
          setJobLogs(previous => [...previous, `${stamp} ${item.level}: ${item.message}`].slice(-100));
        } else if (item.type === 'status' && !['queued', 'running'].includes(item.status)) {
          status = item.status;
          dirty = true;
          persist();
          stop();
        }
      } catch { /* Ignore malformed replay frames; polling remains authoritative. */ }
    }, error: () => {
      // A failed terminal replay is final, and cannot complete a chart cache.
      if (!isActiveJob(job)) {
        stop();
        unavailable = true;
        dirty = false;
        clearTimeout(timer);
        try {
          window.localStorage.removeItem(metricKey);
          window.localStorage.removeItem(`${metricKey}:milestones`);
        } catch { /* storage unavailable */ }
        setHistoryUnavailable(true);
      }
      persist();
    } }, { replayOnly: !isActiveJob(job), isTerminal: data => {
      try {
        const item = JSON.parse(data) as TrainingStreamEvent;
        return item.type === 'status' && !['queued', 'running'].includes(item.status);
      } catch { return false; }
    } });
    return () => { stop(); persist(); };
  }, [datasetId, trainingWorker, job?.id, job?.startedAt]);

  useEffect(() => {
    setForm(readStoredForm(datasetId));
    setResumeChoice('');
    setPrepare(readPrepare());
    setPrepareManifest(readStored<string>(`${PREP_KEY}${datasetId}:manifest`, ''));
    setJob(null);
    setJointPipelineId('');
    setPrepareJob(null);
    setAppliedPrepareJobId(readStored<string>(`${PREP_KEY}${datasetId}:applied`, ''));
    setError('');
    let cancelled = false;
    const storedId = readStored<string>(`${JOB_KEY}${datasetId}`, '');
    const storedPrepareId = readStored<string>(`${PREP_KEY}${datasetId}:job`, '');
    const restore = async () => {
      if (storedPrepareId) {
        try {
          const restored = await client.query(trainingQueries.job(trainingWorker, storedPrepareId));
          if (!cancelled && isPrepareJob(restored, datasetId)) setPrepareJob(restored);
        } catch { /* Fall back to the job list below. */ }
      }
      if (storedId) {
        try {
          const restored = await client.query(trainingQueries.job(trainingWorker, storedId));
          if (!cancelled && isJointJob(restored, datasetId)) {
            setJob(restored);
            return;
          }
        } catch { /* Fall back to the dataset job list. */ }
      }
      try {
        const result = await client.query(runQueries.jobs(trainingWorker, datasetId));
        const latestPrepare = result
          .filter(item => isPrepareJob(item, datasetId))
          .sort((a, b) => b.createdAt - a.createdAt)[0];
        if (!cancelled && latestPrepare) {
          setPrepareJob(latestPrepare);
          writeStored(`${PREP_KEY}${datasetId}:job`, latestPrepare.id);
        }
        const latest = result
          .filter(item => isJointJob(item, datasetId))
          .sort((a, b) => b.createdAt - a.createdAt)[0];
        if (!cancelled && latest) {
          setJob(latest);
          writeStored(`${JOB_KEY}${datasetId}`, latest.id);
        }
      } catch { /* The form remains usable when history is unavailable. */ }
    };
    void restore();
    return () => { cancelled = true; };
  }, [datasetId, trainingWorker]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      writeStored(`${FORM_KEY}${datasetId}`, form);
    }
  }, [datasetId, form]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      writeStored(`${PREP_KEY}${datasetId}`, prepare);
      if (prepareJob) writeStored(`${PREP_KEY}${datasetId}:job`, prepareJob.id);
      if (prepareManifest) writeStored(`${PREP_KEY}${datasetId}:manifest`, prepareManifest);
      writeStored(`${PREP_KEY}${datasetId}:applied`, appliedPrepareJobId);
    }
  }, [datasetId, prepare, prepareJob, prepareManifest, appliedPrepareJobId]);

  // "Perform all stages" runs in the store and can outlive this card: after a
  // page remount it starts training through the old instance, so the job never
  // reaches this one. Adopt any joint job the store reports for this dataset.
  const storeJob = useActiveTrainingJob();
  useEffect(() => {
    if (!storeJob || !isJointJob(storeJob, datasetId) || storeJob.id === job?.id) return;
    setJob(storeJob);
    writeStored(`${JOB_KEY}${datasetId}`, storeJob.id);
  }, [datasetId, storeJob?.id]);



  const pipelineQuery = useQuery({ ...runQueries.preparation(jointPipelineId), enabled: !!jointPipelineId });
  const pipelineJobQuery = useQuery({ ...runQueries.preparationJob(jointPipelineId), enabled: !!jointPipelineId,
    refetchInterval: pollWhile<TrainingJobSummary | null>(data => !!jointPipelineId && isActiveJob(data)) });
  useEffect(() => {
    const current = pipelineJobQuery.data;
    if (current && isJointJob(current, datasetId)) {
      setJob(current);
      writeStored(`${JOB_KEY}${datasetId}`, current.id);
    }
    const pipeline = pipelineQuery.data;
    if (pipeline && ['done', 'failed', 'cancelled', 'interrupted'].includes(pipeline.status)) {
      if (pipeline.error) setError(pipeline.error);
      setJointPipelineId('');
    }
  }, [datasetId, jointPipelineId, pipelineQuery.data, pipelineJobQuery.data]);




  useEffect(() => {
    if (prepareJob?.status === 'done' && prepareManifest && appliedPrepareJobId !== prepareJob.id) {
      setForm(previous => ({ ...previous, checkpoint: previous.method === 'fixed-recipe' ? '' : prepare.checkpoint, dataset: prepareManifest }));
      setAppliedPrepareJobId(prepareJob.id);
    }
  }, [prepareJob?.id, prepareJob?.status, prepareManifest, prepare.checkpoint, appliedPrepareJobId]);

  const set = <K extends keyof Yue2JointTrainRequest>(key: K, value: Yue2JointTrainRequest[K]) =>
    setForm(previous => ({ ...previous, [key]: value }));
  // Settings that differ from the recipe defaults, per-run paths excluded
  // (#200). Each one can be reset on its own, or all at once.
  const changedSettings = Object.entries(snapshotPresetSettings(form, lyricTiming))
    .filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(baselineFor(form)[key]));
  const resetSetting = (key: string) => {
    if (key === 'lyricTiming') { onLyricTimingChange(false); return; }
    setForm(previous => ({ ...previous, [key]: baselineFor(previous)[key] }));
  };
  const resetAllSettings = () => {
    setForm(previous => {
      const next: Record<string, unknown> = { ...DEFAULT_FORM };
      for (const key of PRESET_EXCLUDED_KEYS) next[key] = previous[key];
      return next as unknown as Yue2JointTrainRequest;
    });
    if (lyricTiming) onLyricTimingChange(false);
  };
  const loadPreset = async (preset: Yue2JointPreset) => {
    if (!presetLoadGate.current.begin()) return;
    const request = ++presetLoadRequest.current;
    setPresetLoading(true);
    setPresetError('');
    let applying = false;
    try {
      const { result } = await preferencesApi.presets.resolveYue2Joint(preset, form, lyricTiming);
      if (request !== presetLoadRequest.current) return;
      if (latestForm.current !== form || latestLyricTiming.current !== lyricTiming) {
        throw new Error(t('trainingStudio.yue2.method.presetFormChanged',
          'The form changed while the preset was loading. Choose the preset again.'));
      }
      pendingPresetApplication.current = { form: result.effectiveForm, lyricTiming: result.lyricTiming };
      applying = true;
      setForm(result.effectiveForm);
      onLyricTimingChange(result.lyricTiming);
    } catch (err) {
      if (request !== presetLoadRequest.current) return;
      setPresetError(t('trainingStudio.yue2.method.presetLoadFailed', 'Could not load preset: {{error}}',
        { error: err instanceof Error ? err.message : String(err) }));
    } finally {
      if (!applying && request === presetLoadRequest.current) {
        presetLoadGate.current.release();
        setPresetLoading(false);
      }
    }
  };
  // A saved preset is "on" when every setting it stores matches the form.
  const current = snapshotPresetSettings(form, lyricTiming) as Record<string, unknown>;
  const userPresetActive = (preset: Yue2JointPreset) => Object.entries(preset.settings)
    .every(([key, value]) => JSON.stringify(value) === JSON.stringify(current[key]));
  const captureJointRequest = async (): Promise<Record<string, unknown>> => {
    const request = presetLoadGate.current.capture(() =>
      captureJointOverrides(form, prepare, lyricTiming, defaultDevice, resumeChoice));
    // The songs shown on the card are the songs held out: the ids go with the request.
    if (request.plannerGate && !request.resume && !resumeChoice) {
      const seed = form.plannerGate?.evalSeed ?? 42;
      const pick = await client.fetchQuery({ ...runQueries.holdout(trainingWorker, datasetId, seed), staleTime: 0 });
      if (pick.error) throw new Error(pick.error);
      attachHoldout(request, pick.holdoutIds);
    }
    if (!resumeChoice && !form.resume?.trim()) await captionMissing();
    return request;
  };
  // No onError: a failed start rejects into run's catch, which sets the
  // error line (sole notifier), as before.
  const prepareStartMut = useMutation({
    mutationFn: (snapshot: Parameters<typeof preparationApi.startYue2Preparation>[0]) => preparationApi.startYue2Preparation(snapshot),
    onSuccess: () => trainingWritePlan(client, trainingWorker),
  });
  const run = async (): Promise<string | null> => {
    if (presetLoadGate.current.pending) return null;
    const worker = currentWorkerRef();
    setStarting(true); setError('');
    try {
      const request = await captureJointRequest();
      const pipeline = await submitJointStart(datasetId, request, {
        getContext: preparationApi.getYue2PreparationContext,
        start: snapshot => prepareStartMut.mutateAsync(snapshot),
      }, undefined, worker);
      // A new run takes the ladder over from whatever was picked before.
      setPickedLadderRun('');
      setJointPipelineId(pipeline.id);
      return pipeline.id;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally { setStarting(false); }
  };
  // The preparation command captures this form before any stage starts.
  const startRef = useRef<(() => Promise<Record<string, unknown>>) | null>(null);
  useEffect(() => {
    startRef.current = captureJointRequest;
    if (exposeStart) exposeStart(() => startRef.current ? startRef.current() : Promise.reject(new Error('Joint form is unavailable')));
  });
  const prepareMut = useMutation({
    mutationFn: async () => {
      const result = await startYue2AitkPrepare(datasetId, { ...prepare,
        lyricTiming: form.method === 'fixed-recipe' ? false : lyricTiming,
        ...(form.method === 'fixed-recipe' ? { captionGrammar: 'fixed-recipe' as const, base: form.base } : {}) }, { worker: trainingWorker });
      return { result, job: await client.query(trainingQueries.job(trainingWorker, result.jobId)) };
    },
    onSuccess: r => { setPrepareManifest(r.result.manifest); setAppliedPrepareJobId(''); setPrepareJob(r.job); void trainingWritePlan(client, trainingWorker); },
    onError: err => setError(err instanceof Error ? err.message : String(err)),
  });
  const prepareDataset = () => {
    if (presetLoadGate.current.pending) return;
    setError('');
    void prepareMut.mutateAsync().catch(() => null);
  };
  const pipelineCancelMut = useMutation({
    mutationFn: (pipelineId: string) => preparationApi.commandYue2Preparation(pipelineId, 'cancel'),
    onSuccess: () => { void trainingWritePlan(client, trainingWorker); },
    onError: err => setError(err instanceof Error ? err.message : String(err)),
  });
  // Shared by Stop (train job) and the prepare Stop: silent is threaded through
  // the target because this card's stop error line is the sole notifier.
  const cancelMut = useMutation({
    mutationFn: async (v: { jobId: string; worker: string | null }) => {
      await cancelJob(v.jobId, { worker: v.worker, silent: true });
      return client.query(trainingQueries.job(v.worker, v.jobId));
    },
    onSuccess: (_measured, v) => { void trainingWritePlan(client, v.worker); },
    onError: err => setError(err instanceof Error ? err.message : String(err)),
  });
  const stop = () => {
    if (!job && !jointPipelineId) return;
    setError('');
    if (jointPipelineId) void pipelineCancelMut.mutateAsync(jointPipelineId).catch(() => null);
    else if (job) void cancelMut.mutateAsync({ jobId: job.id, worker: trainingWorker }).then(measured => setJob(measured)).catch(() => null);
  };
  const stopPrepare = () => {
    if (!prepareJob) return;
    setError('');
    void cancelMut.mutateAsync({ jobId: prepareJob.id, worker: trainingWorker }).then(measured => setPrepareJob(measured)).catch(() => null);
  };
  const error = commandError || prepareQuery.error?.message || runsQuery.error?.message || previewsQuery.error?.message || pipelineQuery.error?.message || pipelineJobQuery.error?.message || '';
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
  const active = !!jointPipelineId || job?.status === 'queued' || job?.status === 'running';
  const preparing = prepareJob?.status === 'queued' || prepareJob?.status === 'running';
  const input = 'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 disabled:opacity-50';
  // Shared with field()'s own input below — a reset needs the same lock its
  // input has, and 'seed' is also the resume-restored field name the preview
  // sub-object's own seed key collides with (field() has no notion of which
  // object it is reading from), so resumeChoice locks it too.
  const fieldLocked = (key: string, source: unknown = form) =>
    (source === form && form.method === 'fixed-recipe' && Object.hasOwn(FIXED_RECIPE_SCHEDULE, key))
    || (!!resumeChoice && ['seed', 'device', 'rank', 'alpha', 'adapterType', 'lokrDim', 'lokrFactor', 'saveEvery', 'cursorWeight'].includes(key))
    || active || starting || preparing || yue2RunAllActive;
  const field = (label: string, key: string, type = 'text', source: unknown = form, update?: (value: string) => void, info?: string, meta?: string, onReset?: () => void) => (
    <PreviewField label={label} info={info} meta={meta} onReset={onReset}>
      <input className={input} type={type} step={key === 'arScale' ? 0.1 : undefined} min={key === 'arScale' ? 0 : undefined} value={String((source as Record<string, unknown>)[key] ?? '')} disabled={fieldLocked(key, source)}
        onChange={event => update ? update(event.target.value) : set(key as keyof Yue2JointTrainRequest, type === 'number' ? Number(event.target.value) : event.target.value as never)} />
    </PreviewField>
  );
  // The lyric timing toggle lives in its own card on the stages page; it is
  // locked while this card is resuming, preparing or training.
  const busy = active || preparing || starting || prepareMut.isPending || yue2RunAllActive;
  const timingLocked = !!resumeChoice || active || preparing || starting;
  useEffect(() => { onTimingLockedChange?.(timingLocked); }, [timingLocked, onTimingLockedChange]);
  const progress = job && job.total > 0 ? ` · ${job.done}/${job.total}` : '';
  return (
    <TrainingRunPanel title={t('trainingStudio.yue2.method.aitkTitle', 'Joint Training')}
      description={t('trainingStudio.yue2.method.autoTrainHint', 'Start training prepares the dataset automatically, then trains the planner (AR) and decoder (NAR) adapters together with the YuE2 report recipe. Unchanged prepared data is reused.')}>
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
                  label: `${new Date(run.createdAt).toLocaleString()} · step ${checkpoint.step}${checkpoint.loss !== undefined ? ` · 20-step ${checkpoint.decoderOnly ? 'decoder ' : ''}mean loss ${checkpoint.loss.toFixed(4)}` : ''} · ${run.status}${run.resumeError ? ` — ${run.resumeError}` : ''}${run.live ? ' — running' : ''}`,
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
          {form.method === 'base-matched' && <div className="flex items-center gap-2 h-4">
            <Toggle
              size="sm"
              accent="amber"
              checked={form.calibrated === true}
              disabled={!!resumeChoice || active || preparing || starting || yue2RunAllActive}
              onChange={checked => setForm(previous => ({ ...previous, calibrated: checked }))}
              aria-label={t('trainingStudio.yue2.method.calibrated', 'Dataset-Calibrated Training')}
            />
            <ParamLabel
              label={t('trainingStudio.yue2.method.calibrated', 'Dataset-Calibrated Training')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.calibratedInfo', 'Experimental. Sizes the run to the album instead of using the preset as it is: the preset\'s updates and save interval are scaled by the album\'s minutes of audio against a 45-minute album (between 0.6× and 2×), so the ladder keeps the same number of rungs. A 53-minute album on a 200-update preset trains for 240 updates. The rule came from ear-scored ladders of the earlier recipe, where longer albums took longer to reach full likeness; runs with this on test whether it holds for this one. The training log says what it chose, and the run records it. Off: the preset runs exactly as shown. Applies to new runs, single and batch.')}
            />
          </div>}
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
        <button type="button" disabled={active || preparing || starting || yue2RunAllActive} onClick={() => refreshDefaults()} className="mt-2 text-xs text-amber-700 dark:text-amber-300 hover:underline">{t('trainingStudio.yue2.method.refreshPaths', 'Check installed assets again')}</button>
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
        <button type="button" onClick={prepareDataset} disabled={presetLoading || preparing || active || starting || prepareMut.isPending || yue2RunAllActive || !prepare.legacyManifest || !prepare.checkpoint || !prepare.tokenizer || !prepare.output || !prepare.models.vae || !prepare.models.semantic || !prepare.models.sheetsage}
          className="mt-3 px-3 py-1.5 rounded-lg text-xs font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40">
          {preparing ? t('trainingStudio.yue2.method.preparing', 'Preparing dataset…') : t('trainingStudio.yue2.method.prepare', 'Prepare native dataset')}
        </button>
        {prepareJob && <span className="ml-3 text-[11px] text-zinc-600 dark:text-zinc-400">{prepareJob.status} · {prepareJob.phase || 'waiting'}</span>}
        {preparing && <button type="button" disabled={cancelMut.isPending} onClick={stopPrepare} className="ml-3 text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.yue2.method.cancel', 'Stop')}</button>}
        {prepareJob?.error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{prepareJob.error}</div>}
      </details>
      <div className="mt-4">
        <div className="flex items-center gap-2 mb-2">
          <ParamLabel label={t('trainingStudio.yue2.method.preset', 'Preset')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.presetInfo', 'Fast, Balanced and Thorough are the same recipe; they differ in how many updates run, how many songs each update averages, and how often a checkpoint is saved (each saves ten). All three train the decoder on 60 s crops. Legacy is the previous recipe: one song per update, stopping once the planner has moved a set distance from the base model, so it can finish sooner, but its adapters scored lower by ear than the three new presets. Each preset shows its time relative to Balanced. For scale, Balanced takes about an hour on a 15-track album on an RTX 5090; a slower GPU or a longer album takes proportionally longer.')} />
          {form.method !== 'fixed-recipe' && !activePreset(form) && !presets.some(userPresetActive) && <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.presetCustom', 'custom')}</span>}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <TrainingPresetChoice selected={form.method === 'fixed-recipe'} disabled={busy || presetLoading} onClick={() => { onLyricTimingChange(false); setForm(previous => ({ ...previous,
            method: 'fixed-recipe', base: 'bf16', checkpoint: '', adapterType: 'lora', rank: 64, alpha: 64,
            ...FIXED_RECIPE_SCHEDULE,
            optimizer: 'adamw-lm', cautious: false, lr: 1e-4, weightDecay: 0.01,
            beta1: 0.9, beta2: 0.999, lrSchedule: 'constant', warmup: 0,
            stopMode: 'steps', arLossWeight: 0.5, klWeight: 0.2, plannerLrScale: 0.5,
            gradAccum: 1, arCropFrames: 0, narCropFrames: 1500,
            ...YUE2_JOINT_FIXED_RECIPE_DROPOUTS,
            lyricTiming: false, cursorWeight: 0, calibrated: false, autoRefine: false,
          })); }}>
            <span className="text-base font-bold">{t('trainingStudio.yue2.method.preset_fixed-recipe', 'HOT-Approved')}</span>
            <span className="text-xs text-zinc-500">1500 × 1 {t('trainingStudio.yue2.method.presetFixedSong', 'song')}</span>
          </TrainingPresetChoice>
          {PRESETS.map(p => <TrainingPresetChoice key={p.key} selected={activePreset(form) === p.key} disabled={active || starting || preparing || yue2RunAllActive}
            onClick={() => setForm(previous => ({ ...previous,
              ...(previous.method === 'fixed-recipe' ? DEFAULT_FORM : {}), ...presetValues(p) }))}>
            <span className="text-base font-bold">{t(`trainingStudio.yue2.method.preset_${p.key}`, p.label)}</span>
            <span className="text-xs text-zinc-500">{p.key === 'legacy' ? t('trainingStudio.yue2.method.presetLegacySteps', 'up to {{steps}} × 1 song', { steps: p.steps }) : `${p.steps} × ${p.gradAccum} songs`} · {p.key === 'balanced' ? t('trainingStudio.yue2.method.presetBaseline', '1× time (baseline)') : t('trainingStudio.yue2.method.presetRelative', '{{ratio}} the time', { ratio: presetTime(p) })}</span>
          </TrainingPresetChoice>)}
        </div>
        {form.method === 'fixed-recipe' && <p className="mt-2 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.fixedRecipeNote', 'Fixed recipe: 1500 steps, seed 42 and a checkpoint every 200 steps. These fields are locked. The server also fixes the adapter, optimizer, loss, crop and timing settings. The base defaults to bf16 and can be changed below.')}</p>}
        {form.method === 'fixed-recipe' && (form.base || 'bf16') !== 'bf16' && <p className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.fixedRecipeBaseNote', 'Not ear-tested on this base: HOT-Approved was tuned and scored on bf16. A quantized base uses less VRAM.')}</p>}
        {activePreset(form) === 'legacy' && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.presetLegacyNote', 'Legacy runs the previous recipe (Prodigy, one song per update, stops at planner KL 1.0, LoKr 256). It may train faster, but quality will not be as good as the new presets: they fixed the endings, structure and late-run degradation this recipe has. Lyric timing is off and the audio cache is not re-cut for it, so it is close to the old recipe rather than exact.')}</p>}
        <UserPresetRow state={presetState} isActive={userPresetActive} onLoad={preset => void loadPreset(preset)} busy={busy} loading={presetLoading}
          info={t('trainingStudio.yue2.method.presetHint', 'A preset captures the training settings (steps, save cadence, seed, device, optimizer, rank/alpha and stop target), never the dataset, checkpoint or output paths. Click a name to load it; the save icon overwrites it with the current settings, the download icon exports it to share.')} />
        <ChangedSettingsRow changed={changedSettings} onReset={resetSetting} onResetAll={resetAllSettings}
          disabled={active || preparing || starting || yue2RunAllActive} />
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
        {defaultDevice === 'CUDA0' ? <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.method.trainingGpu', 'Training GPU')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.trainingGpuInfo', 'Choose the NVIDIA card for this run. Settings uses the GPU selected in the Settings page. CUDA names the chosen card CUDA0 inside the trainer.')}
          />
          <StyledSelect
            accent="amber"
            value={nvidiaGpus.find(gpu => gpu.uuid.toLowerCase() === form.gpuUuid?.toLowerCase())?.uuid ?? form.gpuUuid ?? ''}
            disabled={!!resumeChoice || active || starting || preparing || yue2RunAllActive}
            className="w-full"
            onChange={value => set('gpuUuid', value)}
            options={[
              { value: '', label: t('trainingStudio.yue2.method.trainingGpuSettings', 'Settings GPU') },
              ...nvidiaGpus.map(gpu => ({ value: gpu.uuid, label: `GPU ${gpu.index}: ${gpu.name} (${Math.round(gpu.memoryMB / 1024)} GB)` })),
              ...(form.gpuUuid && !nvidiaGpus.some(gpu => gpu.uuid.toLowerCase() === form.gpuUuid?.toLowerCase())
                ? [{ value: form.gpuUuid, label: t('trainingStudio.yue2.method.trainingGpuUnavailable', 'Unavailable GPU: {{uuid}}', { uuid: form.gpuUuid }) }]
                : []),
            ]}
          />
        </label> : field(t('trainingStudio.yue2.method.device', 'Device'), 'device', 'text', form, undefined,
          t('trainingStudio.yue2.method.deviceInfo', 'The ggml device for this run, such as Vulkan0 or Vulkan1. Leave empty for the first GPU.'),
          t('trainingStudio.yue2.method.deviceMeta', 'default {{device}}', { device: defaultDevice || 'CUDA0' }))}
        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.method.adapterType', 'Adapter type')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.adapterTypeInfo', 'The adapter parameterization trained. LoRA is the standard low-rank pair. LoKr trains a Kronecker-factored delta per site instead, at a similar file size for more capacity; it is the default and what every preset was tested with. DoRA is a LoRA pair plus a learned magnitude per output row, so it uses the LoRA rank and alpha; it is untested on this model and needs a little more VRAM and time per update. Switching resets rank/alpha (or dim/factor/alpha) and the KL stop to that type\'s tested defaults.')}
          />
          <StyledSelect
            accent="amber"
            value={form.adapterType ?? 'lora'}
            disabled={!!resumeChoice || active || starting || preparing || yue2RunAllActive}
            className="w-full"
            onChange={value => {
              const adapterType = value === 'lokr' || value === 'dora' ? value : 'lora';
              // LoKr's scale is alpha/dim, so alpha follows the dim by default
              // (LyCORIS scale 1); LoRA gets its rank/alpha defaults back.
              setForm(previous => adapterType === 'lokr'
                ? { ...previous, adapterType, lokrDim: previous.lokrDim ?? 128, lokrFactor: previous.lokrFactor ?? 4, alpha: 256, ...LOKR_STOP }
                : { ...previous, adapterType, alpha: previous.rank ?? 64, ...LORA_STOP });
            }}
            options={[
              { value: 'lora' as const, label: t('trainingStudio.yue2.method.adapterLora', 'LoRA') },
              { value: 'lokr' as const, label: t('trainingStudio.yue2.method.adapterLokr', 'LoKr') },
              { value: 'dora' as const, label: t('trainingStudio.yue2.method.adapterDora', 'DoRA') },
            ]}
          />
        </label>
      </div>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-3">
        {field(t('trainingStudio.yue2.method.steps', 'Updates'), 'steps', 'number', form, undefined,
          t('trainingStudio.yue2.method.stepsInfo', 'How many optimizer updates to train. Each update averages "songs per update" songs, so Fast (100 × 4) sees 400 songs, Balanced (200 × 4) 800 and Thorough (300 × 8) 2400. The run ends here; pick a checkpoint by ear.'),
          t('trainingStudio.yue2.method.stepsMeta', 'Balanced 200'))}
        {form.method === 'tuned' && form.stopMode === 'kl' ? (
          // Legacy: save by steps or by planner KL. In KL mode saveEvery stays
          // the step cadence for the decoder phase, where the KL is frozen.
          <label className="flex flex-col gap-1">
            <ParamLabel label={t('trainingStudio.yue2.method.saveEvery', 'Save every')}
              info={t('trainingStudio.yue2.method.saveEveryKlInfo', 'Steps saves a checkpoint every N steps. AR KL saves a rung each time the planner KL crosses the next multiple of the value, up to the KL target, with previews on each. Once the planner freezes at the target the KL stops moving, so the decoder phase saves (and previews) every {{steps}} steps.', { steps: form.saveEvery })}
              meta={form.klCheckpointEvery! > 0 ? t('trainingStudio.yue2.method.saveEveryKlMeta', 'default 0.15') : t('trainingStudio.yue2.method.saveEveryStepsMeta', 'default 25')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" />
            <StyledSelect accent="amber" className="w-full" value={form.klCheckpointEvery! > 0 ? 'kl' : 'steps'} disabled={fieldLocked('saveEvery')}
              onChange={value => setForm(previous => ({ ...previous, klCheckpointEvery: value === 'kl' ? 0.15 : undefined }))}
              options={[
                { value: 'steps' as const, label: t('trainingStudio.yue2.method.saveEverySteps', 'Steps') },
                { value: 'kl' as const, label: t('trainingStudio.yue2.method.saveEveryKl', 'AR KL') },
              ]} />
            {form.klCheckpointEvery! > 0
              ? <input className={input} type="number" step="0.05" min="0.01" max="1" value={form.klCheckpointEvery} disabled={fieldLocked('saveEvery')}
                  onChange={event => set('klCheckpointEvery', Number(event.target.value))} />
              : <input className={input} type="number" value={String(form.saveEvery ?? '')} disabled={fieldLocked('saveEvery')}
                  onChange={event => set('saveEvery', Number(event.target.value))} />}
          </label>
        ) : field(t('trainingStudio.yue2.method.saveEvery', 'Save every'), 'saveEvery', 'number', form, undefined,
          t('trainingStudio.yue2.method.saveEveryInfo', 'How many steps between saved checkpoints. Lower gives more rungs to pick from (and more previews, if enabled) at the cost of disk space and time; higher saves less often. The presets save ten rungs: every 10 on Fast, 20 on Balanced, 30 on Thorough.'),
          t('trainingStudio.yue2.method.saveEveryMeta', 'Balanced 20'))}
        {field(t('trainingStudio.yue2.method.seed', 'Seed'), 'seed', 'number', form, undefined,
          t('trainingStudio.yue2.method.seedInfo', 'The random seed for training (batch order, dropout, initial noise). Changing it gives a different run on the same data; keeping it fixed makes a rerun reproducible.'),
          t('trainingStudio.yue2.method.seedMeta', 'default 42'))}
        {(form.adapterType ?? 'lora') === 'lokr' ? <>
          {field(t('trainingStudio.yue2.method.lokrDim', 'LoKr dim'), 'lokrDim', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrDimInfo', 'The size of the Kronecker-factored delta. Higher gives the adapter more capacity, at a larger file and more VRAM. The presets use 128 with alpha 256 (about 213 MB for both halves; 64 is about 106 MB).'),
            t('trainingStudio.yue2.method.lokrDimMeta', 'default 128'))}
          {field(t('trainingStudio.yue2.method.lokrFactor', 'LoKr factor'), 'lokrFactor', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrFactorInfo', 'How each weight is split into two Kronecker factors: the smaller factor is at most this size. 4 is tested; 8 also fits. At factor 4, stay below dim 256, where some sites stop factorizing and ignore alpha.'),
            t('trainingStudio.yue2.method.lokrFactorMeta', 'default 4'))}
          {field(t('trainingStudio.yue2.method.lokrAlpha', 'LoKr alpha'), 'alpha', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrAlphaInfo', 'The LoKr strength, alpha / dim. The presets use 256 at dim 128 (a scale of 2); the earlier dim-64 recipe used 256 at dim 64. Raise dim for more capacity rather than raising alpha alone.'),
            t('trainingStudio.yue2.method.lokrAlphaMeta', 'default 256'))}
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
          ? { type: 'lokr', dim: Number(form.lokrDim ?? 128), factor: Number(form.lokrFactor ?? 4) }
          : { type: 'lora', rank: Number(form.rank ?? 64) });
        const loraAlt = lokr ? yue2AdapterHalfBytes({ type: 'lora', rank: Number(form.lokrDim ?? 128) }) : null;
        return <p className={`text-[11px] mt-2 ${half === null ? 'text-amber-700 dark:text-amber-300' : 'text-zinc-500'}`}>
          {half === null
            ? t('trainingStudio.yue2.method.sizeBadFactor', 'This LoKr factor does not split the attention and MLP outputs on their boundaries, so training will refuse it. Try 4 or 8.')
            : t('trainingStudio.yue2.method.sizeEstimate', 'Expected adapter size: {{total}} for both halves ({{half}} each for the planner and the decoder).', { total: formatMB(2 * half), half: formatMB(half) })}
          {half !== null && loraAlt !== null && ` ${t('trainingStudio.yue2.method.sizeLoraCompare', 'A rank-{{rank}} LoRA would be {{lora}}.', { rank: Number(form.lokrDim ?? 128), lora: formatMB(2 * loraAlt) })}`}
        </p>;
      })()}
      <details className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/40 dark:bg-black/5 p-3">
        <summary className="cursor-pointer text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.advancedSettings', 'Advanced')}</summary>
        <div className="mt-2 mb-3 max-w-md">
          {resumeChoice ? <p className="text-xs text-zinc-500">{t('trainingStudio.yue2.method.optimizerRestored', 'Optimizer: {{optimizer}} (restored from the selected run)', { optimizer: form.optimizer ?? 'adamw-lm' })}</p>
            : <Yue2OptimizerFields joint value={{ optimizer: form.optimizer ?? 'adamw-lm', prodigyD0: form.prodigyD0 ?? 1e-6, muonLrScale: form.muonLrScale ?? 1,
                muonNsSteps: form.muonNsSteps ?? 5, cautious: form.cautious === true }}
                onChange={patch => setForm(previous => ({ ...previous, ...patch }))} />}
        </div>
        {form.method === 'tuned' && <p className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.legacyKnobsHint', "Legacy runs the previous recipe: a blank field below uses that recipe's own value, not the grey one.")}</p>}
        <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.baseMatchedKnobsHint', 'Blank uses the default shown in grey. The report does not state a fine-tuning learning rate or the dropout rates, so those defaults are agreed guesses; the rest are the report\'s own values.')}</p>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-2">
          {([
            ['lr', t('trainingStudio.yue2.method.lr', 'Learning rate'), t('trainingStudio.yue2.method.bmLrInfo', 'AdamW peak learning rate for both halves. The report gives full-model rates only (3e-4 joint, annealed to 3e-5), which do not transfer to an adapter; 1e-4 is the agreed default.'), 'default 1e-4'],
            ['warmup', t('trainingStudio.yue2.method.warmup', 'Warmup steps'), t('trainingStudio.yue2.method.warmupInfo', 'Linear warmup to the peak rate, then cosine decay to a 0.1x floor at the step count. Blank = 3% of the steps, the base\'s own joint-phase warmup share.'), 'default 3% of steps'],
            ['weightDecay', t('trainingStudio.yue2.method.weightDecay', 'Weight decay'), t('trainingStudio.yue2.method.bmWeightDecayInfo', 'Decoupled weight decay, as the base\'s joint phase used.'), 'default 0.1 (report)'],
            ['gradAccum', t('trainingStudio.yue2.method.gradAccum', 'Songs per update'), t('trainingStudio.yue2.method.gradAccumInfo', 'Gradients of this many songs are averaged before each optimizer update, so one step sees more than one song, as the base\'s batch of 256 did. Each update takes this many times longer. Fast and Balanced use 4 (about 17 s an update on the 5090 with 60 s decoder crops, 28 s with whole songs), Thorough 8 (about twice that). 1 updates on every song.'), 'default 4'],
            ['arLossWeight', t('trainingStudio.yue2.method.arLossWeight', 'Planner loss weight'), t('trainingStudio.yue2.method.arLossWeightInfo', 'The planner\'s cross-entropy is weighted by this against the decoder\'s flow loss before the shared gradient clip, so it sets how much of the clip the planner takes. The report trains the base at 0.25; 1.0 is the tuned recipe.'), 'default 0.25 (report)'],
            ['beta2', t('trainingStudio.yue2.method.beta2', 'Adam beta2'), t('trainingStudio.yue2.method.beta2Info', 'How long the optimizer\'s second moment remembers. The report\'s joint phase used 0.95; torch\'s 0.999 is what the tuned recipe uses.'), 'default 0.95 (report)'],
            ['textDropout', t('trainingStudio.yue2.method.textDropout', 'Text dropout'), t('trainingStudio.yue2.method.textDropoutInfo', 'Share of steps trained with the style text (trigger included) removed and the lyrics kept. The report drops text and lyrics separately or together for guidance but gives no rates; 0.1 is the agreed default.'), 'default 0.1'],
            ['lyricDropout', t('trainingStudio.yue2.method.lyricDropout', 'Lyric dropout'), t('trainingStudio.yue2.method.lyricDropoutInfo', 'Share of steps trained with the lyrics removed and the style kept, written the way the official instrumental tooling sends a lyric-free request.'), 'default 0.1'],
            ['bothDropout', t('trainingStudio.yue2.method.bothDropout', 'Uncond. dropout'), t('trainingStudio.yue2.method.bothDropoutInfo', 'Share of steps trained on the bare instruction with no style or lyrics: exactly the prompt the runtime\'s guidance uses as its unconditional branch.'), 'default 0.1'],
            ['abcDropout', t('trainingStudio.yue2.method.abcDropout', 'ABC dropout'), t('trainingStudio.yue2.method.bmAbcDropoutInfo', 'Share of steps trained without the lead sheet, matching the report\'s balanced mix of tasks with and without a score.'), 'default 0.5 (report)'],
            ['narCropFrames', t('trainingStudio.yue2.method.narCropFrames', 'Decoder crop (frames)'), t('trainingStudio.yue2.method.bmNarCropInfo', 'The decoder trains on this many frames per song (25 per second). 0 trains on the whole song, as the base did (about 12 GB of VRAM and 11-15 s a step on a 4-minute song). Every preset uses 1500, a 60 s crop: about 40% faster, and it scored as well as whole songs in a blind test.'), 'presets 1500'],
            ['arCropFrames', t('trainingStudio.yue2.method.arCropFrames', 'Planner crop (frames)'), t('trainingStudio.yue2.method.arCropInfo', 'The planner trains on only the first this-many frames of each song (25 per second) instead of the whole song. The planner backward is the largest cost of an update, so this is the biggest speed lever, but the planner then never trains on the rest of the song or its ending, and whole songs are what fixed endings and structure in this recipe. It only applies while the decoder crop is on (not 0), and a song shorter than the decoder crop still trains whole. Untested by ear. 0 or blank trains whole songs.'), 'default 0 (whole song)'],
          ] as const).map(([key, label, info, meta]) => (
            <label key={key} className="flex flex-col gap-1">
              <ParamLabel label={label} info={info} meta={meta} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
                onReset={busy ? undefined
                  // gradAccum and narCropFrames are NOT blank-default knobs — DEFAULT_FORM
                  // populates both (4, 1500), so an untouched Balanced form already has a
                  // value here and a blanket reset-to-undefined would both show an arrow
                  // on a form nobody touched and, for narCropFrames, flip the preset to
                  // Custom by clearing the decoder crop (activePreset checks it against
                  // 0). The other nine fields are genuinely blank by default.
                  : (key === 'gradAccum' || key === 'narCropFrames')
                    ? (form[key] !== DEFAULT_FORM[key] ? () => set(key, DEFAULT_FORM[key]) : undefined)
                    : (form[key] !== undefined ? () => set(key, undefined) : undefined)} />
              <input className={`${input} placeholder:text-zinc-500`} type="number" step="any"
                placeholder={String(key === 'warmup' ? Math.max(1, Math.round((Number(form.steps) || 0) * BASE_MATCHED_DEFAULTS.warmupFraction)) : BASE_MATCHED_DEFAULTS[key])}
                value={form[key] ?? ''} disabled={active || starting || preparing || yue2RunAllActive}
                onChange={event => set(key, event.target.value === '' ? undefined : Number(event.target.value))} />
            </label>
          ))}
        </div>
        {/* Legacy (tuned) recipe knobs. The server forces its own values under
            base-matched, so the block stays visible but locked there. */}
        {(() => {
          const off = form.method !== 'tuned' || busy;
          const kl = (form.stopMode ?? 'steps') === 'kl';
          const label = 'text-[10px] font-medium text-zinc-500 uppercase tracking-wider';
          const num = (key: keyof Yue2JointTrainRequest, name: string, info: string, meta: string) => (
            <label key={key} className="flex flex-col gap-1">
              <ParamLabel label={name} info={info} meta={meta || undefined} className={label}
                onReset={off || form[key] === undefined ? undefined : () => set(key, undefined)} />
              <input className={`${input} placeholder:text-zinc-500`} type="number" step="any" placeholder="recipe default"
                value={String(form[key] ?? '')} disabled={off}
                onChange={event => set(key, (event.target.value === '' ? undefined : Number(event.target.value)) as never)} />
            </label>
          );
          const decoderOn = kl && (form.narExtraSteps ?? 0) > 0;
          return <div className={`mt-4 pt-3 border-t border-zinc-300/70 dark:border-white/10 ${form.method !== 'tuned' ? 'opacity-50' : ''}`}>
            <span className={label}>{t('trainingStudio.yue2.method.legacyTitle', 'Legacy recipe')}</span>
            <p className="mt-1 text-[11px] text-zinc-500">{form.method === 'tuned'
              ? t('trainingStudio.yue2.method.legacyHint', "The knobs only the Legacy recipe reads. Blank uses the recipe's own value.")
              : t('trainingStudio.yue2.method.legacyLockedHint', 'Only the Legacy preset uses these; base-matched fixes its own values.')}</p>
            <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-2">
              <label className="flex flex-col gap-1">
                <ParamLabel label={t('trainingStudio.yue2.method.stopMode', 'Train until')} className={label}
                  info={t('trainingStudio.yue2.method.stopModeInfo', 'What ends the run: an AR KL target (the planner has moved a set distance from the base model), a plain step count, or a target loss. Legacy uses the KL target, the tested stop. Updates is the cap either way.')} />
                <StyledSelect accent="amber" value={form.stopMode ?? 'steps'} disabled={off} className="w-full"
                  onChange={value => set('stopMode', value)}
                  options={[
                    { value: 'kl' as const, label: t('trainingStudio.yue2.method.stopKl', 'AR KL target') },
                    { value: 'steps' as const, label: t('trainingStudio.yue2.method.stopSteps', 'Step count') },
                    { value: 'loss' as const, label: t('trainingStudio.yue2.method.stopLoss', 'Target loss') },
                  ]} />
              </label>
              {kl && num('targetKl', t('trainingStudio.yue2.method.targetKl', 'AR KL target'),
                t('trainingStudio.yue2.method.targetKlFieldInfo', 'How far the planner may move from the base model before the run stops (or the planner freezes, with the decoder phase below on). Higher trains a stronger likeness but risks planner damage (looping outros); lower stays safer but weaker.'),
                t('trainingStudio.yue2.method.targetKlLegacyMeta', 'Legacy 1.0'))}
              {kl && <label className="flex flex-col gap-1">
                <ParamLabel label={t('trainingStudio.yue2.method.targetKlMode', 'KL reading')} className={label}
                  info={t('trainingStudio.yue2.method.targetKlModeInfo', 'How the KL target is read off the training curve. The 30-step trend line reacts at once to where the curve is heading; the 20-step mean is smoother but lags about 10 steps, so the run trains a little past the target before it notices.')} />
                <StyledSelect accent="amber" value={form.targetKlMode ?? 'mean'} disabled={off} className="w-full"
                  onChange={value => set('targetKlMode', value)}
                  options={[
                    { value: 'trend' as const, label: t('trainingStudio.yue2.method.targetKlTrend', '30-step trend') },
                    { value: 'mean' as const, label: t('trainingStudio.yue2.method.targetKlMean', '20-step mean') },
                  ]} />
              </label>}
              {(form.stopMode ?? 'steps') === 'loss' && num('targetLoss', t('trainingStudio.yue2.method.targetLoss', 'Target loss'),
                t('trainingStudio.yue2.method.targetLossInfo', 'Stops the run once the trailing 20-step mean of the composite loss (AR CE + KL weight × AR KL + NAR flow MSE) is at or below this. Lower trains longer.'), '')}
              {num('plannerLrScale', t('trainingStudio.yue2.method.plannerLrScale', 'Planner LR scale'),
                t('trainingStudio.yue2.method.plannerLrScaleInfo', 'The planner (AR) half trains at the learning rate × this. Higher reaches the KL target sooner; lower is gentler and less prone to planner damage.'), 'Legacy 0.6')}
              {num('narLrScale', t('trainingStudio.yue2.method.narLrScale', 'Decoder LR scale'),
                t('trainingStudio.yue2.method.narLrScaleInfo', 'The decoder (NAR, where timbre and likeness live) half trains at the learning rate × this. Lower it if renders garble words or lose audio quality.'), 'Legacy 1')}
              {num('klWeight', t('trainingStudio.yue2.method.klWeight', 'KL anchor'),
                t('trainingStudio.yue2.method.klWeightInfo', 'How strongly the planner loss is pulled back toward the base model each step. Higher keeps the planner closer to the base (safer, less likeness); lower lets it drift further per step.'), 'default 0.2')}
              {num('captionDropout', t('trainingStudio.yue2.method.captionDropout', 'Caption dropout'),
                t('trainingStudio.yue2.method.captionDropoutInfo', "The share of steps trained on the trigger word alone instead of the song's full caption, so a new caption at generation time still lands on the artist. 0 trains on the caption every step."), 'Legacy 0.5')}
              {num('spikeFactor', t('trainingStudio.yue2.method.spikeFactor', 'Spike guard'),
                t('trainingStudio.yue2.method.spikeFactorInfo', 'Skip any update whose gradient norm is over this many times the recent median, so one bad step cannot corrupt the adapter. 0 turns the guard off.'), 'Legacy 5')}
              {num('spikeStop', t('trainingStudio.yue2.method.spikeStop', 'Stop after spikes'),
                t('trainingStudio.yue2.method.spikeStopInfo', 'End the run when this many updates are skipped within the window, keeping the last pre-spike weights. 0 never stops on spikes alone.'), 'Legacy 3')}
              {num('spikeStopWindow', t('trainingStudio.yue2.method.spikeStopWindow', 'Spike window'),
                t('trainingStudio.yue2.method.spikeStopWindowInfo', 'How many steps the skipped updates must fall within to end the run.'), 'Legacy 20')}
              <label className="flex flex-col gap-1">
                <ParamLabel label={t('trainingStudio.yue2.method.lrSchedule', 'LR schedule')} className={label}
                  info={t('trainingStudio.yue2.method.lrScheduleInfo', 'How the learning rate changes over the run. cosine and linear decay to zero at the step cap, so a run that stops early on its KL keeps the weights mid-decay. wsd stays flat until a stop is near, then decays for a set number of steps so the kept weights are annealed. constant stays flat. sgdr cycles the rate in growing loops.')} />
                <StyledSelect accent="amber" value={form.lrSchedule ?? 'cosine'} disabled={off} className="w-full"
                  onChange={value => set('lrSchedule', value)}
                  options={[
                    { value: 'cosine' as const, label: 'cosine' },
                    { value: 'cosine-floor' as const, label: 'cosine to a floor' },
                    { value: 'constant' as const, label: 'constant' },
                    { value: 'linear' as const, label: 'linear' },
                    { value: 'wsd' as const, label: 'wsd' },
                    { value: 'sgdr' as const, label: 'sgdr' },
                  ]} />
              </label>
              {form.lrSchedule === 'cosine-floor' && num('lrFloor', t('trainingStudio.yue2.method.lrFloor', 'LR floor'),
                t('trainingStudio.yue2.method.lrFloorInfo', 'Where the cosine decay ends, as a fraction of the learning rate, instead of zero.'), '')}
              {form.lrSchedule === 'wsd' && num('lrDecaySteps', t('trainingStudio.yue2.method.lrDecaySteps', 'Decay steps'),
                t('trainingStudio.yue2.method.lrDecayStepsInfo', 'How many steps the wsd decay lasts once a stop triggers it. Longer is a gentler anneal.'), '')}
              {form.lrSchedule === 'wsd' && kl && num('klOvershootMargin', t('trainingStudio.yue2.method.klOvershootMargin', 'KL overshoot'),
                t('trainingStudio.yue2.method.klOvershootMarginInfo', 'How far the KL may pass the target during the decay before the stop acts at once instead of waiting for the decay to finish.'), 'default 0.1')}
              {form.lrSchedule === 'wsd' && <label className="flex flex-col gap-1">
                <ParamLabel label={t('trainingStudio.yue2.method.lrDecayShape', 'Decay shape')} className={label}
                  info={t('trainingStudio.yue2.method.lrDecayShapeInfo', 'The curve of the wsd decay: linear steps down evenly; cosine eases in and out.')} />
                <StyledSelect accent="amber" value={form.lrDecayShape ?? 'linear'} disabled={off} className="w-full"
                  onChange={value => set('lrDecayShape', value)}
                  options={[{ value: 'linear' as const, label: 'linear' }, { value: 'cosine' as const, label: 'cosine' }]} />
              </label>}
              {form.lrSchedule === 'sgdr' && num('lrCycleSteps', t('trainingStudio.yue2.method.lrCycleSteps', 'First cycle'),
                t('trainingStudio.yue2.method.lrCycleStepsInfo', 'The length of the first sgdr cycle, in steps.'), '')}
              {form.lrSchedule === 'sgdr' && num('lrCycleMult', t('trainingStudio.yue2.method.lrCycleMult', 'Cycle growth'),
                t('trainingStudio.yue2.method.lrCycleMultInfo', 'How much longer each sgdr cycle is than the last. 1 keeps them equal.'), '')}
            </div>
            <Toggle
              accent="amber"
              className="mt-3"
              checked={decoderOn}
              disabled={off || !kl}
              onChange={checked => setForm(previous => checked
                ? { ...previous, narExtraSteps: 250, reconStop: previous.reconStop ?? 0.005, reconStopWindow: previous.reconStopWindow ?? 10, reconKeepDelta: previous.reconKeepDelta ?? 0.003 }
                : { ...previous, narExtraSteps: 0 })}
              label={t('trainingStudio.yue2.method.decoderPhase', 'Train the decoder on after the KL stop')}
              info={t('trainingStudio.yue2.method.decoderPhaseInfo', 'At the KL target the planner freezes instead of the run ending, and the decoder (timbre, where likeness lives) trains alone until its reconstruction meter flattens: the gain over the last few checkpoints, read off a fitted line, drops under the minimum below. It is the plateau test the Refine page used. A decoder checkpoint whose reconstruction did not beat the best so far by the keep threshold is deleted, so the ladder keeps only real gains. Updates is still the cap. Needs "Train until" on the AR KL target.')}
            />
            {decoderOn && <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-2">
              {num('narExtraSteps', t('trainingStudio.yue2.method.narExtraSteps', 'Decoder budget'),
                t('trainingStudio.yue2.method.narExtraStepsInfo', 'The most steps the decoder trains alone after the freeze. The plateau stop usually ends it sooner; Updates caps it too.'), 'default 250')}
              {num('reconStop', t('trainingStudio.yue2.method.reconStop', 'Plateau: min gain'),
                t('trainingStudio.yue2.method.reconStopInfo', 'Stop once the reconstruction meter improves by less than this fraction over the window. 0 trains to the budget.'), 'default 0.005')}
              {num('reconStopWindow', t('trainingStudio.yue2.method.reconStopWindow', 'Plateau window'),
                t('trainingStudio.yue2.method.reconStopWindowInfo', 'How many checkpoints the gain line is fitted over. Wider smooths noise but reacts later.'), 'default 10')}
              {num('reconKeepDelta', t('trainingStudio.yue2.method.reconKeepDelta', 'Keep threshold'),
                t('trainingStudio.yue2.method.reconKeepDeltaInfo', 'A decoder-phase checkpoint is kept only if its reconstruction beats the best so far by at least this much. 0 keeps every checkpoint.'), 'default 0.003')}
              {num('reconTarget', t('trainingStudio.yue2.method.reconTarget', 'Recon target'),
                t('trainingStudio.yue2.method.reconTargetInfo', 'Also stop once the reconstruction meter is at or under this. Blank: the plateau alone decides.'), 'off')}
            </div>}
          </div>;
        })()}
        {(() => {
          const label = 'text-[10px] font-medium text-zinc-500 uppercase tracking-wider';
          const locked = busy || gateResuming;
          const gateNum = (key: 'every' | 'minEpochs' | 'patience' | 'epsilon' | 'evalSeed', name: string, info: string, meta: string) => (
            <label key={key} className="flex flex-col gap-1">
              <ParamLabel label={name} info={info} meta={meta} className={label} />
              <input className={input} type="number" step="any" placeholder={meta.replace('default ', '')} value={String(gate?.[key] ?? '')} disabled={locked}
                onChange={event => setForm(previous => ({ ...previous, plannerGate: { ...previous.plannerGate, [key]: event.target.value === '' ? undefined : Number(event.target.value) } }))} />
            </label>
          );
          const pick = holdoutQuery.data;
          return <div className="mt-4 pt-3 border-t border-zinc-300/70 dark:border-white/10">
            <span className={label}>{t('trainingStudio.yue2.method.gateTitle', 'Held-out planner gate')}</span>
            <Toggle
              accent="amber"
              className="mt-2"
              checked={!!gate}
              disabled={locked}
              onChange={checked => {
                if (checked) {
                  narBeforeGate.current = form.narExtraSteps;
                  setForm(previous => ({ ...previous, plannerGate: {}, narExtraSteps: (previous.narExtraSteps ?? 0) > 0 ? previous.narExtraSteps : 250 }));
                } else setForm(previous => ({ ...previous, plannerGate: undefined, narExtraSteps: narBeforeGate.current }));
              }}
              label={t('trainingStudio.yue2.method.gateEnable', 'Freeze the planner when held-out loss stops improving')}
              info={t('trainingStudio.yue2.method.gateEnableInfo', 'Holds a few songs out of training and scores the planner on them every few updates. Once the minimum epochs have passed, a run of readings that do not beat the best by the epsilon freezes the planner where it is (no rollback); the decoder then trains on for its own budget. Off by default.')}
            />
            {gate && <>
              <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.gateStarting', 'These are starting settings, not validated ones.')} {t('trainingStudio.yue2.method.gateCost', 'Each reading takes about 4 s for 5 held-out songs; checking every 10 updates adds about 25% to the joint phase.')}</p>
              <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-2">
                {gateNum('every', t('trainingStudio.yue2.method.gateEvery', 'Check every'),
                  t('trainingStudio.yue2.method.gateEveryInfo', 'Optimiser updates between held-out readings. Lower reacts sooner and costs more time.'), 'default 10')}
                {gateNum('minEpochs', t('trainingStudio.yue2.method.gateMinEpochs', 'Min epochs'),
                  t('trainingStudio.yue2.method.gateMinEpochsInfo', 'The planner cannot freeze before this many passes over the training songs (the held-out ones do not count).'), 'default 5')}
                {gateNum('patience', t('trainingStudio.yue2.method.gatePatience', 'Patience'),
                  t('trainingStudio.yue2.method.gatePatienceInfo', 'How many readings in a row may fail to beat the best before the planner freezes.'), 'default 3')}
                {gateNum('epsilon', t('trainingStudio.yue2.method.gateEpsilon', 'Epsilon'),
                  t('trainingStudio.yue2.method.gateEpsilonInfo', 'A reading counts as better only if it is below the best by more than this, in absolute cross-entropy.'), 'default 0.001')}
                {gateNum('evalSeed', t('trainingStudio.yue2.method.gateSeed', 'Eval seed'),
                  t('trainingStudio.yue2.method.gateSeedInfo', 'Seeds both the choice of held-out songs and the readings, so every reading scores the same draws.'), 'default 42')}
                <label className="flex flex-col gap-1">
                  <ParamLabel label={t('trainingStudio.yue2.method.gateNarSteps', 'Decoder steps after freeze')} className={label}
                    info={t('trainingStudio.yue2.method.gateNarStepsInfo', 'The most steps the decoder trains alone after the planner freezes. Must be above 0. Updates caps the run as well.')} meta="default 250" />
                  <input className={input} type="number" step="1" value={String(form.narExtraSteps ?? '')} disabled={locked}
                    onChange={event => set('narExtraSteps', event.target.value === '' ? undefined : Number(event.target.value))} />
                </label>
              </div>
              {gateResuming
                ? <p className="mt-2 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.gateResume', "Resuming keeps the run's own gate settings, held-out songs and decoder budget; no new songs are held out.")}</p>
                : holdoutQuery.isError ? <p className="mt-2 text-[11px] text-red-600 dark:text-red-400">{holdoutQuery.error instanceof Error ? holdoutQuery.error.message : String(holdoutQuery.error)}</p>
                : pick?.error ? <p className="mt-2 text-[11px] text-red-600 dark:text-red-400">{pick.error}</p>
                : pick ? <p className="mt-2 text-[11px] text-zinc-600 dark:text-zinc-400">
                  {t('trainingStudio.yue2.method.gateHoldout', 'Holding out {{count}} of {{songs}} songs ({{training}} left to train on): {{ids}}',
                    { count: pick.holdoutIds.length, songs: pick.songs, training: pick.training, ids: pick.holdoutIds.join(', ') })}</p>
                : <p className="mt-2 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.gateLoading', 'Choosing held-out songs…')}</p>}
              {(form.narExtraSteps ?? 0) <= 0 && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{t('trainingStudio.yue2.method.gateNeedsNar', "Set Decoder steps after freeze above 0: the gate needs the decoder's budget.")}</p>}
            </>}
          </div>;
        })()}
      </details>
      <TrainingPreviewSettings
        title={t('trainingStudio.yue2.method.previewTitle', 'Checkpoint previews (optional)')}
        enabled={form.preview?.enabled ?? false}
        disabled={active || preparing || starting || yue2RunAllActive}
        onEnabledChange={checked => setForm(previous => ({ ...previous, preview: { ...(previous.preview ?? defaultPreview(form.saveEvery)), enabled: checked } }))}
        enableLabel={t('trainingStudio.yue2.method.previewEnable', 'Render a preview at every saved checkpoint')}
        enableInfo={t('trainingStudio.yue2.method.previewManual', 'On by default: each checkpoint gets one draft take (12 decoder steps, 300 s) rendered while training continues, so the ladder is ready to score when the run ends. Off: render rungs by hand from the ladder below.')}
        enableDefault={LADDER_PREVIEW.enabled}
        lead={
          <Toggle
            accent="amber"
            className="mt-2"
            checked={form.stopEngine !== false}
            disabled={active || preparing || starting || yue2RunAllActive}
            onChange={checked => setForm(previous => ({ ...previous, stopEngine: checked }))}
            label={t('trainingStudio.yue2.method.stopEngine', 'Stop the engine during training')}
            info={t('trainingStudio.yue2.method.stopEngineHelp', 'Off by default since the ladder previews render while training runs and need the engine up (about 14 GB for the trainer plus 10-12 GB for the engine and a render). On: the trainer gets the whole GPU, generation is unavailable, and no previews render until the run ends; use it on a smaller card and render the ladder afterwards.')}
          />
        }
        afterToggle={<>
          {form.preview?.enabled && form.stopEngine === false && <Toggle
            accent="amber"
            size="sm"
            className="mt-2"
            checked={form.preview.parallel !== false}
            disabled={active || preparing || starting || yue2RunAllActive}
            onChange={checked => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, parallel: checked }, ...(checked ? { stopEngine: false } : {}) }))}
            label={t('trainingStudio.yue2.method.previewParallel', 'In parallel with training')}
            info={t('trainingStudio.yue2.method.previewParallelInfo', 'Render each checkpoint\'s preview while training continues; needs the engine up (this also turns "Stop the engine during training" off). Off: training pauses at each checkpoint, renders, and resumes, which fits a smaller card but takes longer.')}
            defaultValue={LADDER_PREVIEW.parallel}
          />}
          {form.preview?.enabled && form.stopEngine !== false && <p className="mt-2 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.previewEngineStopped', 'The engine is stopped during training, so only the final checkpoint gets a preview automatically; render the other rungs from the ladder after the run.')}</p>}
        </>}
      >
        {form.preview?.enabled && <>
          {field(t('trainingStudio.yue2.method.previewSeconds', 'Preview seconds'), 'seconds', 'number', form.preview, value => setForm(previous => {
            const seconds = Math.max(8, Math.min(360, Number(value) || 300));
            return { ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seconds, previewMaxFrames: seconds * 25 } };
          }), t('trainingStudio.yue2.method.previewSecondsInfo', 'How long the rendered preview sample is, from 8 to 120 seconds. Longer previews show more of the song but take longer to render at every checkpoint.'), t('trainingStudio.yue2.method.previewSecondsMeta', 'default 300'),
            !busy && form.preview.seconds !== LADDER_PREVIEW.seconds
              ? () => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seconds: LADDER_PREVIEW.seconds, previewMaxFrames: LADDER_PREVIEW.seconds * 25 } }))
              : undefined)}
          {field(t('trainingStudio.yue2.method.previewSeed', 'Preview seed'), 'seed', 'number', form.preview, value => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seed: Number(value) } })),
            t('trainingStudio.yue2.method.previewSeedInfo', 'The random seed used for every preview render, so previews across checkpoints are directly comparable rather than each landing on a different random take.'), t('trainingStudio.yue2.method.previewSeedMeta', 'default 424242'),
            !fieldLocked('seed', form.preview) && form.preview.seed !== LADDER_PREVIEW.seed
              ? () => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seed: LADDER_PREVIEW.seed } }))
              : undefined)}
          {field(t('trainingStudio.yue2.method.previewTakes', 'Takes per checkpoint'), 'takes', 'number', form.preview, value => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, takes: Math.max(1, Math.min(4, Math.round(Number(value)) || 1)) } })),
            t('trainingStudio.yue2.method.previewTakesInfo', 'How many preview tracks each checkpoint renders, 1 to 4. Take 1 uses the preview seed, so it compares like for like across checkpoints; every extra take uses a fresh random seed. Each take adds its render time to every checkpoint.'), t('trainingStudio.yue2.method.previewTakesMeta', 'default 1'),
            !busy && (form.preview.takes ?? 1) !== LADDER_PREVIEW.takes
              ? () => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, takes: LADDER_PREVIEW.takes } }))
              : undefined)}
          {field(t('trainingStudio.yue2.method.previewArScale', 'Preview AR scale'), 'arScale', 'number', form.preview, value => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, arScale: Math.max(0, Number(value) || 0) } })),
            t('trainingStudio.yue2.method.previewArScaleInfo', 'Music adapter strength for checkpoint previews, including manual renders. Default 0.6. The sound (NAR) adapter stays at 1.0.'), t('trainingStudio.yue2.method.previewArScaleMeta', 'default 0.6'),
            !busy && form.preview.arScale !== LADDER_PREVIEW.arScale
              ? () => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, arScale: LADDER_PREVIEW.arScale } }))
              : undefined)}
          <label className="flex flex-col gap-1 md:col-span-2">
            <ParamLabel label={t('trainingStudio.yue2.method.previewLyrics', 'Preview lyrics')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.previewLyricsInfo', 'Lyric Studio generation: the newest lyrics generated for this dataset\'s artist in Lyric Studio, so each rung sings a song it never trained on. Falls back to the dataset\'s lyrics when there are none. Dataset lyrics: the first sung track\'s own lyrics. Either way the caption comes from the dataset, because that is what training saw, and every rung of a run uses the same words.')}
              onReset={!busy && form.preview.lyricsSource !== undefined
                ? () => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, lyricsSource: undefined } }))
                : undefined} />
            <StyledSelect
              accent="amber"
              value={form.preview.lyricsSource ?? 'generated'}
              disabled={busy}
              onChange={value => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, lyricsSource: value === 'dataset' ? 'dataset' : undefined } }))}
              options={[
                { value: 'generated', label: t('trainingStudio.yue2.method.previewLyricsGenerated', 'Lyric Studio generation (falls back to dataset lyrics)') },
                { value: 'dataset', label: t('trainingStudio.yue2.method.previewLyricsDataset', 'Dataset lyrics') },
              ]}
              className="w-full"
            />
          </label>
          <p className="text-[11px] text-zinc-500 md:col-span-2">{t('trainingStudio.yue2.method.previewSongHint', 'The caption comes from the first sung track in this dataset. Caption and lyrics overrides below replace either choice.')}</p>
          <Toggle
            accent="amber"
            size="sm"
            checked={form.preview.baseline}
            disabled={active || preparing || starting || yue2RunAllActive}
            onChange={checked => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, baseline: checked } }))}
            label={t('trainingStudio.yue2.method.previewBaseline', 'Include baseline')}
            info={t('trainingStudio.yue2.method.previewBaselineInfo', 'Also renders a take from the unmodified base model alongside the adapter, so you can hear what the adapter changed.')}
            defaultValue={LADDER_PREVIEW.baseline}
          />
          <Toggle
            accent="amber"
            size="sm"
            checked={form.preview.control}
            disabled={active || preparing || starting || yue2RunAllActive}
            onChange={checked => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, control: checked } }))}
            label={t('trainingStudio.yue2.method.previewControl', 'Include control')}
            info={t('trainingStudio.yue2.method.previewControlInfo', 'Also renders a fixed reference take at every checkpoint, for a stable point of comparison as the adapter trains.')}
            defaultValue={LADDER_PREVIEW.control}
          />
          <label className="md:col-span-2 flex flex-col gap-1">
            <ParamLabel
              label={t('trainingStudio.yue2.method.previewCaption', 'Caption override (optional)')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.previewCaptionInfo', 'Replaces the preview track\'s own caption for the rendered sample. Leave blank to use the track\'s caption as-is.')}
              onReset={!busy && (form.preview.caption ?? '') !== ''
                ? () => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, caption: undefined } }))
                : undefined}
            />
            <textarea className={`${input} min-h-16 resize-y`} value={form.preview.caption ?? ''} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, caption: event.target.value } }))} />
          </label>
          <label className="md:col-span-2 flex flex-col gap-1">
            <ParamLabel
              label={t('trainingStudio.yue2.method.previewLyrics', 'Lyrics override (optional)')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.previewLyricsInfo', 'Replaces the preview track\'s own lyrics for the rendered sample. Leave blank to use the track\'s lyrics as-is.')}
              onReset={!busy && (form.preview.lyrics ?? '') !== ''
                ? () => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, lyrics: undefined } }))
                : undefined}
            />
            <textarea className={`${input} min-h-20 resize-y`} value={form.preview.lyrics ?? ''} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, lyrics: event.target.value } }))} />
          </label>
        </>}
      </TrainingPreviewSettings>
      <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.autoOutput', 'Adapters are saved in your global adapters folder under yue2-joint-adapters/triggerword_date_time.')}</p>
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
        {!trainingWorker && workers.length > 0 && <div className="mt-2 flex items-center gap-3 flex-wrap">
          <ParamLabel
            label={t('trainingStudio.workers.runOn', 'Run on')}
            className="text-[11px] font-semibold text-zinc-700 dark:text-zinc-300"
            info={t('trainingStudio.workers.runOnInfo', 'A training worker trains the batch on its own GPU. Each dataset is captioned here first (Gemini, when captioning is on), its audio and sidecars are sent to the worker (only files it does not already have), and it joins the worker\'s batch. Switch "Train on" at the top to that worker to follow the run. The ladder, its previews and scoring stay here: they open on this run once it lands in this machine\'s own index.')}
          />
          <StyledSelect
            accent="amber"
            value={runOn}
            disabled={batchStarting}
            onChange={value => setRunOn(String(value))}
            options={[{ value: '', label: t('trainingStudio.workers.thisPc', 'This PC') }, ...workers.map(w => ({ value: w.name, label: w.online ? w.name : `${w.name} (offline)`, disabled: !w.online }))]}
            className="w-48"
          />
        </div>}
        <div className="mt-2 flex items-center gap-3">
          <button type="button" onClick={() => void runBatch()} disabled={presetLoading || batchStarting || yue2RunAllActive}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
            {batchStarting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {t('trainingStudio.yue2.aitkBatch.start', 'Start batch ({{count}})', { count: batchDraft.length })}
          </button>
          <button type="button" onClick={() => setBatchDraft(null)} className="text-xs text-zinc-500 hover:underline">{t('trainingStudio.yue2.aitkBatch.discard', 'Discard batch')}</button>
        </div>
      </div>}
      <div className="mt-4 flex items-center gap-3 flex-wrap">
        <button type="button" onClick={() => void run()} disabled={presetLoading || !!batchDraft?.length || active || preparing || starting || !!jointPipelineId || yue2RunAllActive}
          className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
          {starting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          {jointPipelineId && !job ? t('trainingStudio.yue2.method.admitting', 'Starting joint training…') : active ? (job?.phase === 'preparing' ? t('trainingStudio.yue2.method.preparing', 'Preparing dataset…') : t('trainingStudio.yue2.method.running', 'Joint training is running')) : t('trainingStudio.yue2.method.start', 'Start joint training')}
        </button>
        {active && <button type="button" disabled={cancelMut.isPending || pipelineCancelMut.isPending} onClick={stop} className="text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.yue2.method.cancel', 'Stop')}</button>}
        {job && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{job.status} · {job.phase || 'waiting'}{progress}
          {liveMetric?.step !== undefined && ` · step ${liveMetric.step}${liveMetric.loss !== undefined ? ` · ${liveMetric.decoderOnly ? 'decoder loss' : 'loss'} ${liveMetric.loss.toFixed(4)}` : ''}`}
        </span>}
        {job && gateReading && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">
          {t('trainingStudio.yue2.method.gateReading', 'held-out CE {{ce}} · best {{best}} (step {{bestStep}}) · stale {{stale}}/{{patience}}{{armed}}',
            { ce: gateReading.ce.toFixed(4), best: gateReading.best.toFixed(4), bestStep: gateReading.bestStep, stale: gateReading.stale, patience: gateReading.patience,
              armed: gateReading.armed ? '' : ' · not armed yet' })}
        </span>}
      </div>
      {job?.status === 'running' && isJointJob(job, datasetId) && <LiveIntervals key={job.id} datasetId={datasetId} jobId={job.id} worker={trainingWorker}
        options={aitkRuns.find(r => r.jobId === job.id)?.options} />}
      {stepHistory.length > 0 && (
        <div className="mt-3">
          <Yue2SplitChart points={stepHistory} milestones={milestones} totalSteps={job?.total ?? 0} vramMb={chartVram}
            options={aitkRuns.find(r => r.jobId === job?.id)?.options} />
          {frozenAt !== undefined && <div className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">
            {t('trainingStudio.yue2.method.plannerFrozen', "Planner frozen at step {{step}} (KL target reached). The decoder is still training, faster: the planner's CE and KL lines end here, the step counter and pace below keep moving.", { step: frozenAt })}
          </div>}
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] tabular-nums text-zinc-500">
            {jointLossRate(stepHistory) !== null && <span>loss rate {jointLossRate(stepHistory)!.toFixed(5)}/step</span>}
            {stepHistory.at(-1)?.elapsedMs !== undefined && <span>elapsed {formatDurationMs(stepHistory.at(-1)!.elapsedMs!)}</span>}
            {job?.status === 'running' && (jointEta(stepHistory, form) !== null ? <span>{jointEta(stepHistory, form)}</span> : jointCapEta(stepHistory, form) && <TrainingEtaLine eta={jointCapEta(stepHistory, form)!} />)}
          </div>
        </div>
      )}
      {historyUnavailable && <p className="mt-3 text-xs text-zinc-500">
        {t('trainingStudio.yue2.method.historyUnavailable', 'Training history for this run is no longer available.')}
      </p>}
      {jobLogs.length > 0 && <details className="mt-2" open={showJobLogs} onToggle={event => setShowJobLogs(event.currentTarget.open)}>
        <summary className="cursor-pointer text-[11px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.method.showLogs', 'Show training log (last 100 lines)')}</summary>
        <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-zinc-950 p-2 text-[10px] leading-4 text-zinc-300 whitespace-pre-wrap">{jobLogs.join('\n')}</pre>
      </details>}
      {awaitingMirror && <div className="mt-3 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">
        <p className="text-xs text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.awaitingMirror', 'Waiting for the first sync from {{worker}}', { worker: trainingWorker })}</p>
      </div>}
      {ladderRunRec && <TrainingCheckpointLadder>
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300 flex items-center gap-1.5">{t('trainingStudio.yue2.method.ladderTitle', 'Checkpoint ladder')} · {new Date(ladderRunRec.createdAt).toLocaleString()}{ladderRunRec.live ? ` · ${t('trainingStudio.yue2.method.ladderLive', 'training')}` : ''}
          {ladderRunRec.origin && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-violet-500/15 text-violet-700 dark:text-violet-300"
            title={t('trainingStudio.review.originInfo', 'Trained on this worker and mirrored here; previews and scoring run here.') as string}>{ladderRunRec.origin.worker}</span>}</p>
        {aitkRuns.length > 1 && <div className="mt-2 flex flex-col gap-1 max-w-md">
          <TrainingRunPicker label={t('trainingStudio.yue2.method.ladderRun', 'Run')}
            info={t('trainingStudio.yue2.method.ladderRunInfo', 'This dataset has more than one joint run. Pick which run\'s ladder to listen to and score. Each run keeps its own scores.')}
            value={ladderRunRec.jobId} onChange={setPickedLadderRun}
            options={[...aitkRuns].sort((a, b) => b.createdAt - a.createdAt).map(r => ({ value: r.jobId,
              label: `${new Date(r.createdAt).toLocaleString()} · ${r.live ? t('trainingStudio.yue2.method.ladderLive', 'training') : r.status} · ${r.checkpoints.filter(c => c.arPath && c.narPath).length} ${t('trainingStudio.yue2.method.ladderRungs', 'rungs')}${r.origin ? ` · ${r.origin.worker}` : ''}` }))} />
        </div>}
        <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.ladderHint', 'Every saved checkpoint is a rung. Previews render while the run trains: by default one 300 s draft take of the dataset\'s first sung track per rung, on the same seed every rung, so the rungs sing roughly the same song and what changes between them is the training; compare rungs on it. Takes per checkpoint adds more, each on a new random seed: a song no other rung made, which shows how the checkpoint does on a fresh draw and catches failures the fixed seed happens to miss. A rung with no previews yet says so, and Render adds takes. Score likeness and corruption 1-5, then press Use this rung on your pick: it becomes the dataset\'s adapter and you can clean up the rest. Scores also feed the Review page and "Finish scored" for batches.')}</p>
        <Yue2LadderReview datasetId={datasetId} datasetName={datasets.find(d => d.id === datasetId)?.name} run={ladderRunRec} previews={jointPreviews}
          renderOpts={{ seconds: form.preview?.seconds ?? 300, takes: form.preview?.takes ?? 1, arScale: form.preview?.arScale ?? LADDER_PREVIEW.arScale, draft: (form.preview?.odeSteps ?? 12) > 0 && (form.preview?.odeSteps ?? 12) < 32 }}
          onError={setError} idPrefix="train-rung" />
      </TrainingCheckpointLadder>}
    </TrainingRunPanel>
  );
};
