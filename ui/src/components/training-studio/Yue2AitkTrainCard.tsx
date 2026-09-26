import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, Check, Loader2, Play, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Yue2OptimizerFields } from './Yue2OptimizerFields';
import { YUE2_JOINT_PRESETS_KEY, type Yue2JointPreset } from './yue2JointPresets';
import { TrainingChart } from './TrainingChart';
import { StyledSelect } from '../shared/StyledSelect';
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
  type Yue2JointPreviewOptions,
  type Yue2JointPreviewRecord,
  type Yue2OptimOptions,
  type PreparedCache,
} from '../../services/trainingApi';
import { useBackendStore } from '../../stores/backendStore';
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
  steps: 500, saveEvery: 25, seed: 42, device: 'CUDA0', lyricTiming: true, cursorWeight: 0.08,
  // Cautious on (2026-09-23, Rob's call after the limpbizkit A/B).
  optimizer: 'prodigy', cautious: true, prodigyD0: 1e-6, muonLrScale: 1, muonNsSteps: 5,
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
  stopMode: 'kl', targetKl: 1.0, targetKlMode: 'trend', lr: 2e-4, plannerLrScale: 0.6, narLrScale: 1,
  // WSD by default (2026-09-25): the stop-triggered decay anneals the kept
  // weights instead of leaving them mid-cosine on an early KL stop.
  lrSchedule: 'wsd',
  // Planner freeze (2026-09-23): the KL target used to end the whole run, so
  // the decoder, which carries timbre, stopped wherever the planner did. The
  // checkpoint-mix ear test (AR200+NAR150 over AR200+NAR100) said the decoder
  // wants more. Now the planner freezes at its KL and the decoder trains 100
  // more steps; the KL checkpoint is still saved, so the old stop point is
  // one of the rungs. Caption dropout 0.5: the measured recipe, so a new
  // caption lands on the artist rather than beside one memorised track.
  narExtraSteps: 0, captionDropout: 0.5,
  // Spike guard (2026-09-23): an RBF decoder collapsed after two gradient
  // spikes (norm 3 and 7 against a 0.2 median) at step 298. Skip any update
  // above 5x the recent median; three skips within 20 steps ends the run on
  // the last pre-spike weights.
  spikeFactor: 5, spikeStop: 3, spikeStopWindow: 20,
  // Decoder stop (2026-09-24, Rob's ear check on Steel Panther 300/425/500:
  // subtle, diminishing returns): stop once the reconstruction meter gains
  // under 0.5% over three checkpoints. The preset's step cap still applies.
  reconStop: 0.005, reconStopWindow: 10,
  // After the main run: the server starts a planner refinement (Refine tab
  // defaults) and the card moves to the Refine tab. Rob's call, 2026-09-24.
  autoRefine: true,
};
const LORA_STOP = { targetKl: 1.4, plannerLrScale: 0.3, narLrScale: undefined };
const LOKR_STOP = { targetKl: 1.0, plannerLrScale: 0.6, narLrScale: 1 };
// Training presets (2026-09-24). The planner stops at the KL target and
// freezes; the decoder keeps training until the step cap (narExtraSteps =
// cap, so it never ends the run before the cap does).
const PRESETS = [
  { key: 'fast', label: 'Fast', targetKl: 0.8, steps: 300 },
  { key: 'balanced', label: 'Balanced', targetKl: 1.0, steps: 500 },
  { key: 'thorough', label: 'Thorough', targetKl: 1.6, steps: 700 },
] as const;
// 2026-09-24 (Rob): the primary run ends at the KL target; the decoder's
// further training happens on the Refine tab from the rung he picks.
const presetValues = (p: typeof PRESETS[number]) => ({ stopMode: 'kl' as const, targetKl: p.targetKl, steps: p.steps, narExtraSteps: 0 });
const activePreset = (f: Yue2JointTrainRequest) => PRESETS.find(p => (f.stopMode ?? 'steps') === 'kl' && f.targetKl === p.targetKl && f.steps === p.steps && !(f.narExtraSteps ?? 0))?.key;
type PrepareForm = Yue2AitkPrepareRequest;

function defaultPreview(everySteps: number): Yue2JointPreviewOptions {
  // 90 s (2026-09-22): 40 s often ended inside the intro, before any vocal.
  return { enabled: false, everySteps, seconds: 90, seed: 424242,
    previewMaxFrames: 2250, baseline: false, control: false };
}

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
  return { ...DEFAULT_FORM, ...stored };
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

export const Yue2AitkTrainCard: React.FC<{ datasetId: string; legacyManifest?: string; cursorReady?: boolean; lyricTiming: boolean; onLyricTimingChange: (value: boolean) => void; exposeStart?: (fn: () => Promise<string | null>) => void }> = ({ datasetId, legacyManifest, cursorReady = false, lyricTiming, onLyricTimingChange, exposeStart }) => {
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
          everySteps: form.saveEvery, previewMaxFrames: Math.max(8, Math.min(120, form.preview.seconds || 90)) * 25 } } : {}) };
      await startBatch({ datasetIds: batchDraft, lyricTiming, clearCache: batchClearCache, recipe });
      setPhase('train');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { setBatchStarting(false); }
  };
  const [aitkRuns, setAitkRuns] = useState<Yue2AitkRunRecord[]>([]);
  const [resumeChoice, setResumeChoice] = useState('');
  // Auto-refine: when the run this card started reaches done, hand over to
  // the Refine tab, where the server-started refinement shows as the active job.
  const handedOver = useRef<string>('');
  useEffect(() => {
    // Only a run that finished just now: a page reload restores an old done job.
    const fresh = typeof job?.finishedAt === 'number' && Date.now() - job.finishedAt < 120_000;
    if (!job || job.status !== 'done' || !fresh || form.autoRefine === false || resumeChoice || handedOver.current === job.id) return;
    handedOver.current = job.id;
    const timer = window.setTimeout(() => setPhase('refine'), 2500);
    return () => window.clearTimeout(timer);
  }, [job?.id, job?.status]);
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
  }, [datasetId, job?.id, job?.status]);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => listYue2JointPreviews(datasetId, job?.id)
      .then(result => { if (!cancelled) setJointPreviews(result.previews); })
      .catch(() => { if (!cancelled) setJointPreviews([]); });
    void refresh();
    const timer = job?.status === 'queued' || job?.status === 'running'
      ? window.setInterval(refresh, 5000) : undefined;
    return () => { cancelled = true; if (timer !== undefined) window.clearInterval(timer); };
  }, [datasetId, job?.id, job?.status]);

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
  const optimValue: Yue2OptimOptions = {
    optimizer: form.optimizer ?? 'adamw',
    prodigyD0: form.prodigyD0 ?? 1e-6,
    muonLrScale: form.muonLrScale ?? 1,
    muonNsSteps: form.muonNsSteps ?? 5,
    cautious: form.cautious === true,
  };
  const savePreset = () => {
    const name = presetName.trim();
    if (!name) { setPresetError(t('trainingStudio.yue2.method.presetNameRequired', 'Give the preset a name first.')); return; }
    if (presets.some(preset => preset.name.toLowerCase() === name.toLowerCase())) {
      setPresetError(t('trainingStudio.yue2.method.presetExists', 'A preset with that name already exists.'));
      return;
    }
    setPresetError('');
    setPresets(previous => [...previous, { version: 2, name, settings: snapshotPresetSettings(form, lyricTiming) }]);
    setPresetName('');
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
    if ((form.stopMode ?? 'steps') === 'kl' && !(typeof form.targetKl === 'number' && form.targetKl > 0)) {
      setError(t('trainingStudio.yue2.method.targetKlRequired', 'Enter an AR KL target above 0 to train until KL.'));
      setStarting(false);
      return null;
    }
    if ((form.stopMode ?? 'steps') === 'loss' && !(typeof form.targetLoss === 'number' && form.targetLoss > 0)) {
      setError(t('trainingStudio.yue2.method.targetLossRequired', 'Enter a target loss above 0 to train until loss.'));
      setStarting(false);
      return null;
    }
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
          everySteps: form.saveEvery, previewMaxFrames: Math.max(8, Math.min(120, form.preview.seconds || 90)) * 25 } } : {}),
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
  // Refinement: the finished run's decoder trains on, planner frozen, with
  // checkpoints every 10 steps so the reconstruction stop lands near the knee.
  // A new run beside the original; the original is never touched. The
  // planner stays frozen: it is the sensitive half (late-song decay past its
  // KL), the decoder is where likeness keeps improving.
  const [refineRun, setRefineRun] = useState('');
  const [refineBudget, setRefineBudget] = useState(500);
  const refine = async () => {
    const run = aitkRuns.find(r => r.jobId === refineRun);
    const last = run?.checkpoints.filter(c => !!c.optimizerPath).sort((a, b) => b.step - a.step)[0];
    if (!run || !last) return;
    setStarting(true); setError('');
    try {
      const result = await startYue2JointTrain(datasetId, { ...form, trainingMethod: 'aitk', refine: true,
        resumeRunId: run.jobId, resumeStep: last.step, steps: last.step + refineBudget, saveEvery: 10,
        stopMode: 'kl', narExtraSteps: last.step + refineBudget, freezePlannerNow: true,
        reconStop: form.reconStop ?? DEFAULT_FORM.reconStop, reconStopWindow: 10,
        lyricTiming, alignmentEnabled: lyricTiming, autoPrepare: false, checkpoint: '', output: '' } as Yue2JointTrainRequest);
      if (typeof window !== 'undefined') window.localStorage.setItem(`${JOB_KEY}${datasetId}`, JSON.stringify(result.jobId));
      setJob(await getJob(result.jobId));
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
  const input = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-xs text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50';
  const field = (label: string, key: string, type = 'text', source: unknown = form, update?: (value: string) => void, info?: string, meta?: string) => (
    <label className="flex flex-col gap-1">
      <ParamLabel label={label} info={info} meta={meta} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" />
      <input className={input} type={type} value={String((source as Record<string, unknown>)[key] ?? '')} disabled={(!!resumeChoice && ['seed', 'device', 'rank', 'alpha', 'adapterType', 'lokrDim', 'lokrFactor', 'saveEvery', 'cursorWeight'].includes(key)) || active || starting || preparing || yue2RunAllActive}
        onChange={event => update ? update(event.target.value) : set(key as keyof Yue2JointTrainRequest, type === 'number' ? Number(event.target.value) : event.target.value as never)} />
    </label>
  );
  const progress = job && job.total > 0 ? ` · ${job.done}/${job.total}` : '';
  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
      <h3 className="text-sm font-semibold text-amber-700 dark:text-amber-300">
        {t('trainingStudio.yue2.method.aitkTitle', 'Joint Training is selected')}
      </h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2 leading-relaxed">
        {t('trainingStudio.yue2.method.autoTrainHint', 'Start training prepares the dataset automatically, then trains AR and NAR together. Unchanged prepared data is reused.')}
      </p>
      <Toggle
        accent="amber"
        className="mt-3"
        checked={lyricTiming}
        disabled={!!resumeChoice || active || preparing || starting || yue2RunAllActive}
        onChange={onLyricTimingChange}
        label={t('trainingStudio.yue2.method.lyricTiming', 'Lyric timing supervision')}
        info={lyricTiming
          ? t('trainingStudio.yue2.method.lyricTimingOn', 'Uses vocal stems and forced alignment before training so the planner learns where each word lands.')
          : t('trainingStudio.yue2.method.lyricTimingOff', 'Skips stems, alignment, and the optional cursor objective. On: needs vocal stems and lyric alignment run first (below).')}
      />
      <Toggle
        accent="amber"
        className="mt-3"
        checked={!!autoCaption}
        disabled={!captionDefault || !!resumeChoice || active || preparing || starting || yue2RunAllActive}
        onChange={checked => setForm(previous => ({ ...previous, autoCaption: checked ? { provider: captionProvider ?? 'gemini' } : false }))}
        label={t('trainingStudio.yue2.method.autoCaption', 'Caption tracks that have no YuE2 caption')}
        info={captionDefault
          ? t('trainingStudio.yue2.method.autoCaptionHint', 'Before training, tracks without a .yue2.txt are re-captioned from the audio (ACE, MM3 and YuE2 captions; lyrics and BPM are left alone). Skipped when every track has one. Off: trains on the long ACE caption for those tracks instead.')
          : t('trainingStudio.yue2.method.autoCaptionNone', 'No captioner available: add a Gemini key in Settings → AI Services or install MOSS. Tracks without a .yue2.txt train on the long ACE caption.')}
      />
      {autoCaption && <div className="ml-6 mt-1 flex flex-wrap items-center gap-2 text-[11px]">
        <StyledSelect
          accent="amber"
          size="sm"
          value={autoCaption.provider}
          disabled={active || preparing || starting || yue2RunAllActive}
          onChange={value => setForm(previous => ({ ...previous, autoCaption: { provider: value } }))}
          options={[
            ...(gemini ? [{ value: 'gemini' as const, label: t('trainingStudio.yue2.method.captionGemini', 'Gemini (cloud, hears the audio)') }] : []),
            ...(mossOk ? [{ value: 'moss' as const, label: t('trainingStudio.yue2.method.captionMoss', 'MOSS (local, hears the audio)') }] : []),
          ]}
          className="w-auto"
        />
        {autoCaption.provider === 'gemini' && gemini && gemini.models.length > 0 && <StyledSelect
          accent="amber"
          size="sm"
          value={autoCaption.model || gemini.defaultModel}
          disabled={active || preparing || starting || yue2RunAllActive}
          onChange={value => setForm(previous => ({ ...previous, autoCaption: { provider: 'gemini', model: value } }))}
          options={gemini.models.map(m => ({ value: m, label: m }))}
          className="w-auto"
        />}
      </div>}
      {captioning && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300 flex items-center gap-1.5"><Loader2 size={12} className="animate-spin" />
        {t('trainingStudio.yue2.method.captioning', 'Captioning tracks without a YuE2 caption: {{done}} / {{total}}', { done: captioning.done, total: captioning.total })}</p>}
      {lyricTiming && !cursorReady && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.lyricTimingNeedsAlignment', 'Run vocal stems and lyric alignment below before starting with timing supervision enabled.')}</p>}
      <label className="mt-4 flex flex-col gap-1">
        <ParamLabel
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
      {resumeChoice && <p className="mt-1 text-[11px] text-zinc-500">The server restores the original dataset, base, optimizer and adapter settings. Set Steps to the total step you want to reach.</p>}
      <div className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
        <div className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.refineTitle', 'Refine a finished adapter')}</div>
        <p className="text-[11px] text-zinc-500 mt-1">{t('trainingStudio.yue2.method.refineHint', 'Trains the decoder on from where the run ended, planner frozen, with a checkpoint every 10 steps; the reconstruction stop ends it at the knee or at the budget. Writes a new run beside the original.')}</p>
        <div className="mt-2 flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-1 min-w-[260px] flex-1">
            <ParamLabel
              label={t('trainingStudio.yue2.method.refineRun', 'Finished run')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.refineRunInfo', 'The completed run to keep training the decoder from. Only runs with a saved optimizer checkpoint, not currently running, are offered.')}
            />
            <StyledSelect
              accent="amber"
              value={refineRun}
              disabled={active || preparing || starting || yue2RunAllActive}
              onChange={setRefineRun}
              placeholder={t('trainingStudio.yue2.method.refinePick', 'Pick a run')}
              className="w-full"
              options={aitkRuns.filter(run => !run.live && !run.resumeError && run.checkpoints.some(c => !!c.optimizerPath)).map(run => {
                const last = run.checkpoints.filter(c => !!c.optimizerPath).sort((a, b) => b.step - a.step)[0];
                return { value: run.jobId, label: `${new Date(run.createdAt).toLocaleString()} · to step ${last.step} · ${run.status}` };
              })}
            />
          </label>
          <label className="flex flex-col gap-1 w-28">
            <ParamLabel
              label={t('trainingStudio.yue2.method.refineBudget', 'Max extra steps')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.refineBudgetInfo', 'The step budget for the refinement run, on top of the step the finished run reached. The reconstruction stop below usually ends it sooner, at the knee; this is only the cap.')}
              meta={t('trainingStudio.yue2.method.refineBudgetMeta', 'default 500')}
            />
            <input className={input} type="number" min={10} step={10} value={refineBudget} disabled={active || preparing || starting || yue2RunAllActive}
              onChange={event => setRefineBudget(Math.max(10, Math.round(Number(event.target.value) || 0)))} />
          </label>
          <button type="button" onClick={() => void refine()} disabled={!refineRun || active || preparing || starting || yue2RunAllActive}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/50 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-40">
            {t('trainingStudio.yue2.method.refineStart', 'Refine decoder')}
          </button>
        </div>
      </div>
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
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.preset', 'Preset')}</span>
        {PRESETS.map(p => <button key={p.key} type="button" disabled={active || starting || preparing || yue2RunAllActive}
          onClick={() => setForm(previous => ({ ...previous, ...presetValues(p) }))}
          className={`px-3 py-1 rounded-lg text-xs font-semibold border disabled:opacity-40 ${activePreset(form) === p.key
            ? 'border-blue-500 bg-blue-500/15 text-blue-700 dark:text-blue-300'
            : 'border-zinc-300/70 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10'}`}>
          {t(`trainingStudio.yue2.method.preset_${p.key}`, p.label)}
          <span className="ml-1 font-normal text-zinc-500">KL {p.targetKl} · cap {p.steps}</span>
        </button>)}
        {!activePreset(form) && <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.presetCustom', 'custom')}</span>}
        <span className="flex-1" />
        <Toggle
          accent="amber"
          id="yue2-auto-refine"
          className={active || starting || preparing || yue2RunAllActive ? 'opacity-50 pointer-events-none' : ''}
          checked={form.autoRefine !== false}
          onChange={v => set('autoRefine', v)}
          label={t('trainingStudio.yue2.method.autoRefine', 'Automatically proceed to refinement')}
          info={t('trainingStudio.yue2.method.autoRefineHint', 'When the run completes, start a planner refinement of it with the Refine tab defaults (KL rungs to 2.0, previews per rung) and move to the Refine tab. Off: the run stops at done and stays on this tab.')}
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
        {!resumeChoice && <details className="md:col-span-2"><summary className="cursor-pointer text-[11px] text-zinc-500">Manual resume path</summary>{field(t('trainingStudio.yue2.method.resume', 'Resume record (optional)'), 'resume', 'text', form, undefined,
          t('trainingStudio.yue2.method.resumeInfo', 'The saved resume record of a previous run, to continue it without picking it from the Resume list above. Leave blank to start a new run.'))}</details>}
        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.method.stopMode', 'Train until')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.stopModeInfo', 'What ends the run: an AR KL target (the planner has moved a set distance from the base model), a plain step count, or a target loss. The presets above use the KL target, which is the tested stop.')}
          />
          <StyledSelect
            accent="amber"
            value={form.stopMode ?? 'steps'}
            disabled={active || starting || preparing || yue2RunAllActive}
            onChange={value => set('stopMode', value)}
            className="w-full"
            options={[
              { value: 'kl' as const, label: t('trainingStudio.yue2.method.stopKl', 'AR KL target') },
              { value: 'steps' as const, label: t('trainingStudio.yue2.method.stopSteps', 'Step count') },
              { value: 'loss' as const, label: t('trainingStudio.yue2.method.stopLoss', 'Target loss') },
            ]}
          />
        </label>
        {field((form.stopMode ?? 'steps') !== 'steps'
          ? t('trainingStudio.yue2.method.maxSteps', 'Max steps')
          : t('trainingStudio.yue2.method.steps', 'Steps'), 'steps', 'number', form, undefined,
          (form.stopMode ?? 'steps') !== 'steps'
            ? t('trainingStudio.yue2.method.maxStepsInfo', 'The step budget the run cannot exceed, even once the KL or loss target is reached and the decoder keeps training. Raise it to let a slow-converging artist train longer; lower it to cap wall-clock time.')
            : t('trainingStudio.yue2.method.stepsInfo', 'How many steps to train, with no other stop condition. Raise it for a longer, more thorough run; lower it to stop sooner.'),
          t('trainingStudio.yue2.method.stepsMeta', 'default 500'))}
        {(form.stopMode ?? 'steps') === 'loss' && field(t('trainingStudio.yue2.method.targetLoss', 'Target loss (composite, trailing mean)'), 'targetLoss', 'number', form, undefined,
          t('trainingStudio.yue2.method.targetLossInfo', 'Stops the run once the trailing 20-step mean of the composite loss (AR CE + 0.2 × AR KL + NAR flow MSE + timing CE × weight) is at or below this. Lower is a stricter target and trains longer; leave blank to use the step count instead.'))}
        {(form.stopMode ?? 'steps') === 'kl' && field(t('trainingStudio.yue2.method.targetKl', 'AR KL target'), 'targetKl', 'number', form, undefined,
          t('trainingStudio.yue2.method.targetKlFieldInfo', 'How far the planner may move from the base model before it freezes. Higher trains a stronger likeness but risks planner damage (looping outros); lower stays safer but weaker. LoRA ships 1.4, LoKr 1.0, because LoKr moves further per unit of KL.'),
          t('trainingStudio.yue2.method.targetKlFieldMeta', 'default 1.0 (LoKr) / 1.4 (LoRA)'))}
        {(form.stopMode ?? 'steps') === 'kl' && <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.method.targetKlMode', 'KL reading')}
            className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.method.targetKlModeInfo', 'How the KL target is read off the training curve. The 30-step trend line reacts immediately to where the curve is heading; the 20-step mean is smoother but lags about 10 steps behind, so the run trains a little past the target before it notices.')}
          />
          <StyledSelect
            accent="amber"
            value={form.targetKlMode ?? 'mean'}
            disabled={active || starting || preparing || yue2RunAllActive}
            onChange={value => setForm(previous => ({ ...previous, targetKlMode: value }))}
            className="w-full"
            options={[
              { value: 'trend' as const, label: t('trainingStudio.yue2.method.targetKlTrend', '30-step trend line (no lag)') },
              { value: 'mean' as const, label: t('trainingStudio.yue2.method.targetKlMean', '20-step mean (lags ~10 steps)') },
            ]}
          />
        </label>}
        {(form.stopMode ?? 'steps') === 'kl' && field(t('trainingStudio.yue2.method.narExtraSteps', 'Decoder steps after KL'), 'narExtraSteps', 'number', form, undefined,
          t('trainingStudio.yue2.method.narExtraStepsInfo', 'Once the planner freezes at its KL target, the decoder (where likeness lives) keeps training alone for this many more steps. 0 ends the run at the KL, saving that checkpoint; raise it to let the decoder train further before stopping, up to the step cap.'),
          t('trainingStudio.yue2.method.narExtraStepsMeta', 'default 0'))}
        {field(t('trainingStudio.yue2.method.saveEvery', 'Save every'), 'saveEvery', 'number', form, undefined,
          t('trainingStudio.yue2.method.saveEveryInfo', 'How many steps between saved checkpoints. Lower gives more rungs to pick from (and more previews, if enabled) at the cost of disk space and time; higher saves less often.'),
          t('trainingStudio.yue2.method.saveEveryMeta', 'default 25'))}
        {field(t('trainingStudio.yue2.method.seed', 'Seed'), 'seed', 'number', form, undefined,
          t('trainingStudio.yue2.method.seedInfo', 'The random seed for training (batch order, dropout, initial noise). Changing it gives a different run on the same data; keeping it fixed makes a rerun reproducible.'),
          t('trainingStudio.yue2.method.seedMeta', 'default 42'))}
        {field(t('trainingStudio.yue2.method.device', 'CUDA device'), 'device', 'text', form, undefined,
          t('trainingStudio.yue2.method.deviceInfo', 'Which CUDA device trains this run, for a machine with more than one GPU.'),
          t('trainingStudio.yue2.method.deviceMeta', 'default CUDA0'))}
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
        {(form.adapterType ?? 'lora') === 'lokr' ? <>
          {field(t('trainingStudio.yue2.method.lokrDim', 'LoKr dim'), 'lokrDim', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrDimInfo', 'The size of the Kronecker-factored delta. Higher gives the adapter more capacity, at a larger file and more VRAM; the tested recipe keeps alpha at 4x this value.'),
            t('trainingStudio.yue2.method.lokrDimMeta', 'default 64'))}
          {field(t('trainingStudio.yue2.method.lokrFactor', 'LoKr factor'), 'lokrFactor', 'number', form, undefined,
            t('trainingStudio.yue2.method.lokrFactorInfo', 'How many sites are Kronecker-factorized, of the four (attention/MLP pairs). At factor 4, stay below dim 256, where some sites stop factorizing and ignore alpha.'),
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
        {lyricTiming && field(t('trainingStudio.yue2.method.cursorWeight', 'Timing loss weight'), 'cursorWeight', 'number', form, undefined,
          t('trainingStudio.yue2.method.cursorWeightInfo', 'How much the lyric-timing (cursor) objective counts in the composite loss, next to the AR/NAR terms. Higher pushes the planner to track word timing more tightly, at some cost to the other objectives; lower lets timing drift more.'),
          t('trainingStudio.yue2.method.cursorWeightMeta', 'default 0.08'))}
      </div>
      {(form.adapterType ?? 'lora') === 'lokr' && <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.lokrHint', 'LoKr trains a Kronecker-factored delta per site instead of a low-rank pair. Strength is alpha / dim; 4x (64 / 4 / 256, about 106 MB for both halves) is the tested default, against 279 MB for the rank-64 LoRA. For more capacity raise dim and keep alpha at 4x dim; at factor 4 stay below dim 256, where some sites stop factorizing and ignore alpha.')}</p>}
      {(form.stopMode ?? 'steps') === 'kl' && <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.targetKlHint', 'AR KL is how far the planner has moved from the base model, so it means the same for every artist. For LoRA, likeness starts near 1.25 and planner damage (looping outros) near 1.9. LoKr moves further per unit of KL, so it ships 1.0. Once the KL reading reaches the target the planner freezes there. With "Decoder steps after KL" above 0, the decoder (timbre, where likeness lives) keeps training alone for that many steps; 0 ends the run at the KL, as before. The KL checkpoint is saved either way. Max steps is the cap. The presets end the run at the KL target: Fast 0.8, Balanced 1.0, Thorough 1.6 (the step count is the cap). The decoder then trains on during refinement, from the rung you pick, until its reconstruction target.')}</p>}
      {(form.stopMode ?? 'steps') === 'loss' && <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.targetLossHint', 'Composite = AR CE + 0.2 × AR KL + NAR flow MSE + timing CE × weight. Training stops once the trailing 20-step mean is at or below this.')}</p>}
      <div className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/40 dark:bg-black/5 p-3">
        {resumeChoice ? <p className="text-xs text-zinc-500">Optimizer: {form.optimizer ?? 'adamw'} (restored from the selected run)</p>
          : <Yue2OptimizerFields joint value={optimValue} onChange={patch => setForm(previous => ({ ...previous, ...patch }))} />}
      </div>
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
              <button type="button" onClick={() => removePreset(preset.name)}
                title={t('trainingStudio.yue2.method.presetRemove', 'Remove this preset')}
                className="rounded p-0.5 text-zinc-400 hover:text-red-600 dark:hover:text-red-400">
                <X size={12} />
              </button>
            </span>
          ))}
        </div>}
      </div>
      <details className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/30 dark:bg-black/10 p-3">
        <summary className="cursor-pointer text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">
          {t('trainingStudio.yue2.method.advancedTitle', 'Advanced: planner and optimizer')}
        </summary>
        <p className="mt-2 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.advancedHint', 'Blank = the engine default. These are the knobs the reference AITK recipe exposes; the defaults are what every run so far has used.')}</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
          {([
            ['lr', t('trainingStudio.yue2.method.lr', 'Learning rate'), t('trainingStudio.yue2.method.lrInfo', 'The base learning rate, applied to the AR and NAR halves through their own scales below. Only used by AdamW; Prodigy learns its own rate and Muon uses its own scale instead.'), 'default 1e-4 · AdamW only'],
            ['weightDecay', t('trainingStudio.yue2.method.weightDecay', 'Weight decay'), t('trainingStudio.yue2.method.weightDecayInfo', 'L2 penalty on the adapter weights each step. Higher keeps the adapter smaller and more conservative; lower lets it move further to fit the training data.'), 'default 1e-4'],
            ['plannerLrScale', t('trainingStudio.yue2.method.plannerLrScale', 'Planner learning-rate scale'), t('trainingStudio.yue2.method.plannerLrScaleInfo', 'The AR (planner) half trains at learning rate × this. Higher moves the planner faster toward the KL target (or step cap); lower is gentler and less prone to planner damage. This card ships 0.6 for LoKr and 0.3 for LoRA.'), 'default 1.0'],
            ['narLrScale', t('trainingStudio.yue2.method.narLrScale', 'Decoder (NAR) learning-rate scale'), t('trainingStudio.yue2.method.narLrScaleInfo', 'The NAR (decoder, where timbre and likeness live) half trains at learning rate × this. Lower it if renders garble words or lose audio quality before the planner reaches its KL target.'), 'default 1.0'],
            ['klWeight', t('trainingStudio.yue2.method.klWeight', 'KL anchor to base (planner)'), t('trainingStudio.yue2.method.klWeightInfo', 'How strongly the planner loss is pulled back toward the base model each step. Higher keeps the planner closer to the base (safer, less likeness); lower lets it drift further per step of training.'), 'default 0.2'],
            ['abcDropout', t('trainingStudio.yue2.method.abcDropout', 'ABC dropout'), t('trainingStudio.yue2.method.abcDropoutInfo', 'The share of lead-sheet (ABC notation) training examples trained without their sheet, so one adapter serves generation both with and without a lead sheet. Higher trains it to rely on the sheet less; lower makes it expect one more often.'), 'default 0.5'],
            ['captionDropout', t('trainingStudio.yue2.method.captionDropout', 'Caption dropout'), t('trainingStudio.yue2.method.captionDropoutInfo', 'The share of steps trained on the trigger word alone instead of the song\'s full caption. 0.5, this card\'s recipe, stops the adapter binding to each track\'s caption so a new caption at generation time still lands on the artist; 0 trains on the caption every step. Needs a dataset prepared after 2026-09-20.'), 'default 0 (this card ships 0.5)'],
            ['spikeFactor', t('trainingStudio.yue2.method.spikeFactor', 'Spike guard'), t('trainingStudio.yue2.method.spikeFactorInfo', 'Skip any weight update whose gradient norm is over this many times the recent median, to stop one bad step from corrupting the adapter. 0 turns the guard off; lower makes it trigger more readily.'), 'this card ships 5'],
            ['spikeStop', t('trainingStudio.yue2.method.spikeStop', 'Stop after spikes'), t('trainingStudio.yue2.method.spikeStopInfo', 'End the run when this many updates are skipped by the spike guard close together (within the window below), keeping the last pre-spike weights. 0 never stops the run on spikes alone.'), 'this card ships 3'],
            ['spikeStopWindow', t('trainingStudio.yue2.method.spikeStopWindow', 'Spike window (steps)'), t('trainingStudio.yue2.method.spikeStopWindowInfo', 'How many steps the spike-guard skips above must fall within to count as a run-ending cluster. Wider makes the stop easier to trigger; narrower requires the spikes to be closer together.'), 'this card ships 20'],
            ['reconStop', t('trainingStudio.yue2.method.reconStop', 'Decoder stop (min gain)'), t('trainingStudio.yue2.method.reconStopInfo', 'Once the planner is frozen, stop when the decoder reconstruction meter improves by less than this fraction over the window below (the knee of the curve). 0 trains to the step cap instead; lower makes the run keep going for smaller gains.'), 'this card ships 0.005'],
            ['reconStopWindow', t('trainingStudio.yue2.method.reconStopWindow', 'Decoder stop window (checkpoints)'), t('trainingStudio.yue2.method.reconStopWindowInfo', 'How many checkpoints the reconstruction-gain trend above is fitted over. Wider smooths out noise but reacts to the knee more slowly; narrower reacts faster but is noisier.'), 'this card ships 10'],
            ['narCropFrames', t('trainingStudio.yue2.method.narCropFrames', 'Decoder crop (frames)'), t('trainingStudio.yue2.method.narCropFramesInfo', 'The decoder trains on a random window this many frames long (25 frames/s). 0 trains on the whole song, shortened only where the prompt would not fit the context. Longer windows see more of each song per step but cost more VRAM and time.'), 'default 1500 (60 s, the reference recipe)'],
          ] as const).map(([key, label, info, meta]) => (
            <label key={key} className="flex flex-col gap-1">
              <ParamLabel label={label} info={info} meta={meta} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" />
              <input className={input} type="number" step="any" placeholder="engine default"
                value={form[key] ?? ''} disabled={active || starting || preparing || yue2RunAllActive}
                onChange={event => set(key, event.target.value === '' ? undefined : Number(event.target.value))} />
            </label>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          <label className="flex flex-col gap-1">
            <ParamLabel
              label={t('trainingStudio.yue2.method.lrSchedule', 'Learning-rate schedule')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.lrScheduleInfo', 'How the learning rate changes over the run. wsd (warmup, flat, triggered decay) is the default: flat until a stop is near, then a short decay, so the kept weights are annealed rather than caught mid-decay. With a KL target the decay starts when the KL trend says the target is about a decay away, landing on the target rather than past it; if the KL still passes the target by the overshoot margin (default 0.1) during the decay, the stop acts at once. cosine and linear decay to zero (or a floor) at the step cap, so a run that stops early on its KL keeps the weights mid-decay, at a high rate. constant stays flat after warmup, with no annealing. sgdr cycles the rate in growing loops; it is expected to lose, since the restarts shake the planner.')}
            />
            <StyledSelect
              accent="amber"
              value={form.lrSchedule ?? 'wsd'}
              disabled={active || starting || preparing || yue2RunAllActive}
              onChange={value => setForm(previous => ({ ...previous, lrSchedule: value }))}
              className="w-full"
              options={[
                { value: 'cosine' as const, label: 'cosine', hint: 'Decays to zero at the step cap.' },
                { value: 'cosine-floor' as const, label: 'cosine to a floor', hint: 'The same cosine, ending at the floor field below instead of zero.' },
                { value: 'constant' as const, label: 'constant', hint: 'Flat after warmup. No annealing.' },
                { value: 'linear' as const, label: 'linear', hint: 'A straight line to zero at the step cap.' },
                { value: 'wsd' as const, label: 'warmup, flat, triggered decay (wsd, default)', hint: 'Flat until a stop is near, then a short decay so the kept weights are annealed.' },
                { value: 'sgdr' as const, label: 'cosine restarts (sgdr)', hint: 'Cosine cycles, each longer than the last. Expected to lose.' },
              ]}
            />
          </label>
          {form.lrSchedule === 'cosine-floor' && field(t('trainingStudio.yue2.method.lrFloor', 'Floor (fraction of the rate)'), 'lrFloor', 'number', form, value => set('lrFloor', value === '' ? undefined : Number(value)),
            t('trainingStudio.yue2.method.lrFloorInfo', 'Where the cosine decay ends, as a fraction of the base learning rate, instead of decaying to zero. Higher keeps a stronger residual rate at the end of the run; 0 behaves like plain cosine.'))}
          {form.lrSchedule === 'wsd' && field(t('trainingStudio.yue2.method.lrDecaySteps', 'Decay steps'), 'lrDecaySteps', 'number', form, value => set('lrDecaySteps', value === '' ? undefined : Number(value)),
            t('trainingStudio.yue2.method.lrDecayStepsInfo', 'How many steps the wsd schedule\'s decay lasts once triggered. Longer gives a gentler anneal; shorter reaches the low rate faster but more abruptly.'))}
          {form.lrSchedule === 'wsd' && (form.stopMode ?? 'steps') === 'kl' && field(t('trainingStudio.yue2.method.klOvershootMargin', 'KL overshoot margin'), 'klOvershootMargin', 'number', form, value => set('klOvershootMargin', value === '' ? undefined : Number(value)),
            t('trainingStudio.yue2.method.klOvershootMarginInfo', 'How far the KL reading may pass the target during the triggered decay before the stop acts immediately instead of waiting for the decay to finish. Lower stops sooner on an overshoot; higher lets the decay run its course more often.'),
            t('trainingStudio.yue2.method.klOvershootMarginMeta', 'default 0.1'))}
          {form.lrSchedule === 'wsd' && <label className="flex flex-col gap-1">
            <ParamLabel
              label={t('trainingStudio.yue2.method.lrDecayShape', 'Decay shape')}
              className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
              info={t('trainingStudio.yue2.method.lrDecayShapeInfo', 'The curve of the wsd schedule\'s triggered decay. linear steps down evenly; cosine eases in and out, spending more time near the flat rate and the floor and less in between.')}
              meta={t('trainingStudio.yue2.method.lrDecayShapeMeta', 'default linear')}
            />
            <StyledSelect
              accent="amber"
              value={form.lrDecayShape ?? 'linear'}
              disabled={active || starting || preparing || yue2RunAllActive}
              onChange={value => set('lrDecayShape', value)}
              className="w-full"
              options={[
                { value: 'linear' as const, label: 'linear' },
                { value: 'cosine' as const, label: 'cosine' },
              ]}
            />
          </label>}
          {form.lrSchedule === 'sgdr' && field(t('trainingStudio.yue2.method.lrCycleSteps', 'First cycle (steps)'), 'lrCycleSteps', 'number', form, value => set('lrCycleSteps', value === '' ? undefined : Number(value)),
            t('trainingStudio.yue2.method.lrCycleStepsInfo', 'The length of the first sgdr cosine cycle. Each following cycle grows by the multiplier below; a shorter first cycle means more, faster restarts early in the run.'))}
          {form.lrSchedule === 'sgdr' && field(t('trainingStudio.yue2.method.lrCycleMult', 'Cycle growth'), 'lrCycleMult', 'number', form, value => set('lrCycleMult', value === '' ? undefined : Number(value)),
            t('trainingStudio.yue2.method.lrCycleMultInfo', 'How much longer each sgdr cycle is than the last, as a multiplier. Higher spaces the restarts further apart as the run goes on; 1 keeps every cycle the same length.'))}
        </div>
      </details>
      <Toggle
        accent="amber"
        className="mt-3"
        checked={form.stopEngine !== false}
        disabled={active || preparing || starting || yue2RunAllActive}
        onChange={checked => setForm(previous => ({ ...previous, stopEngine: checked }))}
        label={t('trainingStudio.yue2.method.stopEngine', 'Stop the engine during training')}
        info={t('trainingStudio.yue2.method.stopEngineHelp', 'On by default: the trainer gets the whole GPU and generation is unavailable until it finishes. Off: keep generating (and scoring ladders) while it trains. Both then share the GPU and run slower, and if VRAM runs out Windows spills to system memory and everything crawls.')}
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
          label={t('trainingStudio.yue2.method.previewEnable', 'Render one artist sample at saved checkpoints')}
          info={t('trainingStudio.yue2.method.previewManual', 'Off by default. Samples are saved for manual listening; they do not start automatically in the player.')}
        />
        {form.preview?.enabled && <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          {field(t('trainingStudio.yue2.method.previewSeconds', 'Preview seconds'), 'seconds', 'number', form.preview, value => setForm(previous => {
            const seconds = Math.max(8, Math.min(120, Number(value) || 90));
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
      {job?.status === 'done' && form.autoRefine !== false && !resumeChoice && <p className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.autoRefineNote', 'Refinement is starting on the Refine tab.')}</p>}
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
      {jointPreviews.length > 0 && <div className="mt-3 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.previewStrip', 'Checkpoint previews')}</p>
        <div className="mt-2 flex flex-col gap-2">
          {jointPreviews.map(preview => <div key={preview.id} className="flex items-center gap-2 text-[11px] text-zinc-600 dark:text-zinc-400">
            <span className="w-20 shrink-0">step {preview.step} · {preview.kind}</span>
            <span className="flex-1">{preview.status === 'failed' ? preview.error || 'render failed' : preview.endReason === 'preview_limit' ? t('trainingStudio.yue2.method.previewCapped', 'Preview length reached') : preview.status}</span>
            {preview.score && <span title={[preview.score.reason, ...(preview.score.flags ?? [])].join('\n')} className={`shrink-0 font-semibold uppercase tracking-wider ${preview.score.verdict === 'healthy' && !preview.score.flags?.length ? 'text-emerald-500' : preview.score.verdict === 'long' || preview.score.verdict === 'healthy' ? 'text-amber-500' : 'text-red-500'}`}>
              {preview.score.verdict} · {preview.score.bars} bars · {Math.round(preview.score.vocalShare * 100)}% vocal
            </span>}
            {preview.audioUrl && preview.status === 'done' && <audio controls preload="none" src={preview.audioUrl} className="h-7 max-w-[240px]" />}
          </div>)}
        </div>
      </div>}
    </div>
  );
};

