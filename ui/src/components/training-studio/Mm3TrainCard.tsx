import { assertRecipeWorker, resolveTrainingRecipe } from '../../services/trainingRecipesApi';
// Mm3TrainCard.tsx — Training Studio phase 3, MiniMax-Music3 branch.
//
// Rendered instead of the ACE LM/DiT cards when the active backend is
// MiniMax-Music3: codes + the MM3-native captions -> an LM LoRA.
//
// The CODES export is phase 2, not here (Mm3CodesCard) — for MM3, codes are
// what preprocessing means. This card only GATES on them, exactly as the ACE
// train panel gates on a preprocess variant, so there is one place to run the
// export and one place to run training.
//
// Everything below the form is the SHARED job machinery: the same SSE stream,
// the same JobProgress, the same loss chart.
//
// THE DEFAULTS ARE NOT DUPLICATED HERE. They arrive in the `mm3` status
// payload from services/training/mm3Train.ts, which is the single place the
// validated recipe lives. This form seeds itself from that response, so a
// change on the server reaches the UI without an edit here.

import React, { useEffect, useState } from 'react';
import {
  AlertTriangle, ChevronDown, ChevronRight, Loader2, Play, ShieldCheck, XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { dispatchMm3Batch, estimateMm3PeakMb, estimateMm3PrefixMb, mm3FlashVramCalibrated, type Mm3PresetName } from '../../services/trainingApi';
import type { Mm3TrainLmRequest } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { JobProgress } from './JobProgress';
import { Mm3CheckpointLadder } from './Mm3CheckpointLadder';
import { Mm3RunsPanel } from './Mm3RunsPanel';
import { useMm3Status } from './useMm3Status';
import { Yue2SplitChart } from './Yue2SplitChart';
import { TrainingEtaLine } from './TrainingEtaLine';
import { previewsFromRows, trainingRunEta } from './trainingRunEta';
import { TrainingPreviewSettings, TrainingPresetChoice } from './TrainingPreviewPanel';
import { TrainingBatchCard, TrainingRunPanel } from './Yue2StageCard';
import { Yue2AitkBatchWizard } from './Yue2AitkBatchWizard';
import { ChangedSettingsRow, readStored, useNamedPresets, UserPresetRow, writeStored, type NamedPreset } from './trainingUserPresets';
import { lossDescentRate, lossTargetEta } from './trainingTargetEta';
import { formatDurationMs } from '../../utils/trainingEta';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';
import { runQueries, trainingWritePlan } from '../../queries/training-runs/queries';
import { trainingQueries } from '../../queries/training-data/queries';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { BTN_SM, CARD, INPUT } from './yue2CardUi';



interface FormState {
  steps: number;
  stopMode: 'steps' | 'loss';
  targetLoss: number;
  targetLossMetric: 'train' | 'eval';
  targetLossEpochs: number;
  saveEvery: number;
  keepResumeState: boolean;
  updateLinkedPresets: boolean;
  longTracks: 'exclude' | 'crop';
  rank: number;
  alpha: number;
  lr: number;
  maxFrames: number;
  cropMode: 'random' | 'beginning' | 'structured';
  cropStartFrac: number;
  cropEndFrac: number;
  cropStartTiles: number;
  depthLossWeight: number;
  depthLossFrames: number;
  optimizer: 'muon' | 'adamw' | 'prodigy';
  muonLrScale: number;
  adapterType: 'lora' | 'lokr';
  lokrFactor: number;
  // ── Flag-contract parity fields (2026-09-05) ───────────────────────────
  attnBackend: 'exact' | 'flash';
  trainPrecision: 'full' | 'fast' | 'fastest';
  rslora: boolean;
  dora: boolean;
  hira: boolean;
  loha: boolean;
  pissa: boolean;
  hotPizza: boolean;
  hra: boolean;
  loraPlusRatio: number;
  /** Soft prompt (a token + a TRAINABLE prefix, both new to MM3). Distinct
   *  from prefixFrames below, which is the FROZEN real-audio history prefix —
   *  a different engine mechanism (train/mm3-lm-kvprefix.h vs lm-prefix.h). */
  artistTokenOn: boolean;
  artistToken: string;
  artistTokenK: number;
  artistTokenLr: number;
  prefixN: number;
  /** '' = auto-pick (server-side ladder — see mm3Preview.ts). */
  previewSongId: string;
  gradAccum: number;
  seed: number;
  warmup: number;
  lyricsDropout: number;
  lrEndFrac: number;
  trigger: string;
  triggerPrepend: boolean;
  basePrecision: string;
  holdout: number;
  evalEvery: number;
  cropAnchor: 'song' | 'zero';
  prefixFrames: number;
  previewEverySteps: number;
  previewEveryMinutes: number;
  previewSeconds: number;
  previewSeed: number;
  previewCaption: string;
  previewLyrics: string;
  previewControl: boolean;
  previewBaseline: boolean;
  previewScaleMlp: number;
  regDatasetId: string;
  regEvery: number;
  regTopK: number;
}

/** Same shape as TrainDitForm's DitMethod / TrainLmForm's LmMethod (2026-09-05
 *  flag-contract parity). LoKr is its own type; DoRA/HiRA/LoHa/HRA are the
 *  LoRA type with one flag set. */
type Mm3Method = 'lokr' | 'lora' | 'dora' | 'hira' | 'loha' | 'hra';
const methodOf = (s: FormState): Mm3Method =>
  s.adapterType === 'lokr' ? 'lokr' : s.dora ? 'dora' : s.hira ? 'hira' : s.loha ? 'loha' : s.hra ? 'hra' : 'lora';
const MM3_METHOD_KEYS: Record<Mm3Method, { label: string; info: string }> = {
  lokr: { label: 'adapterLokr', info: 'adapterTypeHint' },
  lora: { label: 'adapterLora', info: 'adapterTypeHint' },
  dora: { label: 'dora', info: 'doraInfo' },
  hira: { label: 'hira', info: 'hiraInfo' },
  loha: { label: 'loha', info: 'lohaInfo' },
  hra:  { label: 'hra',  info: 'hraInfo' },
};

/** Per-dataset form edits, so a reload keeps them (as YuE2's card does). */
const FORM_KEY = 'hs-mm3-form:';
/** Named presets, MM3's own list; YuE2's live under their own key. */
const PRESETS_KEY = 'hs-mm3-presets';
/** Per-run or per-dataset values a preset must not carry. */
const PRESET_EXCLUDED: ReadonlySet<string> = new Set(['previewSongId', 'trigger', 'regDatasetId']);
const settingsOf = (f: FormState): Record<string, unknown> =>
  Object.fromEntries(Object.entries(f).filter(([k]) => !PRESET_EXCLUDED.has(k)));
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

const NumField: React.FC<{
  label: string; value: number; onChange: (v: number) => void; step?: number; info?: string; meta?: string;
  disabled?: boolean; min?: number; max?: number;
  /** The server-default value to compare against and show beside the reset
   *  icon; the icon itself calls onReset, not a write of this value. */
  defaultValue?: number;
  onReset?: () => void;
}> = ({ label, value, onChange, step = 1, info, meta, disabled, min, max, defaultValue, onReset }) => (
  <label className={`flex flex-col gap-1${disabled ? ' opacity-50' : ''}`}>
    <ParamLabel
      label={label}
      info={info}
      meta={meta}
      className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
      onReset={defaultValue !== undefined && value !== defaultValue && !disabled ? onReset : undefined}
    />
    <input
      type="number" className={INPUT} value={value} step={step} min={min} max={max} disabled={disabled}
      onChange={e => onChange(Number(e.target.value))}
    />
  </label>
);

export const Mm3TrainCard: React.FC<{ datasetId: string; trigger?: string }> = ({ datasetId, trigger }) => {
  const { t } = useTranslation();
  const activeJob = useActiveTrainingJob();
  const startMm3TrainLm = useTrainingStore(s => s.startMm3TrainLm);
  // A batch draft (Train multiple…) turns this card into the recipe editor for N datasets.
  const batchDraft = useTrainingStore(s => s.yue2BatchDraft);
  const setBatchDraft = useTrainingStore(s => s.setYue2BatchDraft);
  const startMm3Batch = useTrainingStore(s => s.startMm3Batch);
  // Run on: a batch started here can train on a training worker instead
  // (pushed there, then copied back by the mirror), as YuE2's does.
  const client = useQueryClient();
  const trainingWorker = useTrainingStore(s => s.trainingWorker);
  const setTrainingWorker = useTrainingStore(s => s.setTrainingWorker);
  const workersQuery = useQuery({ ...trainingQueries.workers(), enabled: !!batchDraft?.length && !trainingWorker });
  const workers = workersQuery.data ?? [];
  const [runOn, setRunOn] = useState('');
  // Which codes cache this run trains on. Off = the standard cache, exactly as
  // every run before the launder gate existed.
  const [trainLaunder, setTrainLaunder] = useState(false);
  const mm3Live = useTrainingStore(s => s.mm3Live);
  const storeError = useTrainingStore(s => s.error);
  const setPhase = useTrainingStore(s => s.setPhase);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);
  const trainEvalSeries = useTrainingStore(s => s.trainEvalSeries);
  const trainTargetLoss = useTrainingStore(s => s.trainTargetLoss);

  const { status, error: statusError } = useMm3Status(datasetId);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  // The form is DERIVED, not seeded: server defaults underneath, the user's
  // edits on top. No effect copies one into the other, so there is no window
  // where the form holds stale numbers and no cascading render — and the
  // validated recipe still has exactly one home (mm3Train.ts).
  const [edits, setEdits] = useState<Partial<FormState>>(() => readStored<Partial<FormState>>(`${FORM_KEY}${datasetId}`, {}));
  useEffect(() => { writeStored(`${FORM_KEY}${datasetId}`, edits); }, [datasetId, edits]);
  const [batchOpen, setBatchOpen] = useState(false);
  const [batchClearCache, setBatchClearCache] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const datasets = useTrainingStore(s => s.datasets);
  const jobLog = useTrainingStore(s => s.jobLog);

  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobRunning = jobStatus === 'queued' || jobStatus === 'running';
  const mine = jobKind === 'mm3-train-lm';
  // Nothing about the run is editable while it is being started or is going.
  const locked = busy || (mine && jobRunning);
  // The live run's ladder: the run the job is writing, else the one its
  // previews came from once the job has ended.
  const trainPreviews = useTrainingStore(s => s.trainPreviews);
  const runsQuery = useQuery({ ...runQueries.mm3Runs(datasetId), enabled: mine });
  const liveRun = runsQuery.data?.runs.find(r => r.running)?.runName ?? trainPreviews[trainPreviews.length - 1]?.run ?? '';

  // Muon normalises its update, so the schedule LR alone understates what the
  // optimizer applied by the scale factor.
  const lrScale = status?.defaults.optimizer === 'muon' ? (status.defaults.muonLrScale ?? 1) : 1;

  // `base` is what the form shows with no edits — the server's current
  // defaults (or a recommendation, where one overrides the default), resolved
  // fresh on every render. Resetting a field means deleting it from `edits`,
  // not writing a value, so a later server-side default change still reaches
  // the form for anyone who never touched that field.
  const base: FormState | null = status ? {
    steps: status.defaults.steps ?? 800,
    stopMode: status.defaults.stopMode ?? 'steps',
    targetLoss: status.defaults.targetLoss ?? 1.0,
    targetLossMetric: status.defaults.targetLossMetric ?? 'train',
    targetLossEpochs: status.defaults.targetLossEpochs ?? 5,
    saveEvery: status.defaults.saveEvery ?? 100,
    keepResumeState: status.defaults.keepResumeState ?? false,
    updateLinkedPresets: status.defaults.updateLinkedPresets ?? true,
    longTracks: status.defaults.longTracks === 'crop' ? 'crop' : 'exclude',
    // Rank follows the recommendation for the same reason as the base: at the
    // default 256 nothing fits below ~24 GB, so a 16 GB card would open on a
    // red 'will not fit' form with the fix two fields away and unstated.
    rank: status.recommended?.rank ?? status.defaults.rank ?? 256,
    // Alpha keeps the recipe's alpha/rank ratio when the recommender lowers the
    // rank. A fixed alpha 128 over a recommended rank 16 trained every update
    // 8x too hard: the "tube" adapters of HOTSTEP-1033.
    alpha: Math.round((status.defaults.alpha ?? 256)
      * (status.recommended?.rank ?? status.defaults.rank ?? 256) / (status.defaults.rank ?? 256)),
    lr: status.defaults.lr ?? 8e-5,
    maxFrames: status.defaults.maxFrames ?? 1500,
    cropMode: (status.defaults.cropMode as 'random' | 'beginning' | 'structured') ?? 'structured',
    cropStartFrac: status.defaults.cropStartFrac ?? 0.55,
    cropEndFrac: status.defaults.cropEndFrac ?? 0.15,
    cropStartTiles: status.defaults.cropStartTiles ?? 3,
    depthLossWeight: status.defaults.depthLossWeight ?? 1.0,
    depthLossFrames: status.defaults.depthLossFrames ?? 128,
    optimizer: status.defaults.optimizer ?? 'prodigy',
    muonLrScale: status.defaults.muonLrScale ?? 64,
    adapterType: status.defaults.adapterType ?? 'lora',
    lokrFactor: status.defaults.lokrFactor ?? 6,
    // Flag-contract parity (2026-09-05). All off/exact/1 by default — none of
    // these has an MM3-specific measurement yet (see the hover text), and
    // --attn/--dora/--hira/--loha/--pissa/--hra/--prefix-n are landing in the
    // mm3-lm-train engine parser concurrently with this UI, not before it.
    attnBackend: status.defaults.attnBackend ?? 'exact',
    trainPrecision: status.defaults.trainPrecision ?? 'fastest',
    rslora: status.defaults.rslora ?? false,
    dora: status.defaults.dora ?? false,
    hira: status.defaults.hira ?? false,
    loha: status.defaults.loha ?? false,
    pissa: status.defaults.pissa ?? false,
    hotPizza: status.defaults.hotPizza ?? false,
    hra: status.defaults.hra ?? false,
    loraPlusRatio: status.defaults.loraPlusRatio ?? 1,
    artistTokenOn: !!status.defaults.artistToken,
    artistToken: status.defaults.artistToken ?? '',
    artistTokenK: status.defaults.artistTokenK ?? 32,
    artistTokenLr: status.defaults.artistTokenLr ?? 0.005,
    prefixN: status.defaults.prefixN ?? 0,
    previewSongId: '',
    gradAccum: status.defaults.gradAccum ?? 1,
    seed: status.defaults.seed ?? 42,
    warmup: status.defaults.warmup ?? 25,
    lyricsDropout: status.defaults.lyricsDropout ?? 0,
    lrEndFrac: status.defaults.lrEndFrac ?? 0.005,
    trigger: trigger ?? '',
    triggerPrepend: status.defaults.triggerPrepend !== false,
    // The RECOMMENDED base, not the global default: the default was chosen on a
    // 32 GB card and on a 12 GB one it is simply wrong. The server picks the
    // highest-fidelity base that fits THIS GPU, and falls back to the default
    // when it cannot read the card.
    // The configured default WINS over the VRAM recommender's pick. The
    // recommender walks bases by fidelity and falls through to a smaller one
    // whenever the best does not fit its budget, which silently downgraded
    // f16 to q8_0. It still sets `rank` and still raises overBudget, so the
    // user is warned rather than quietly given a different base.
    basePrecision: status.defaults.basePrecision || status.recommended?.base || 'f16',
    holdout: status.defaults.holdout ?? 0.15,
    evalEvery: status.defaults.evalEvery ?? 50,
    cropAnchor: (status.defaults.cropAnchor as 'song' | 'zero') ?? 'song',
    // OFF by default, because nothing trained with a prefix has been heard yet.
    prefixFrames: status.defaults.prefixFrames ?? 0,
    // Previews default OFF. They are the fastest way to learn whether a run is
    // worth finishing, but each one costs about a minute, so opting in is the
    // user's call rather than a surprise on the clock.
    // Cadence follows checkpoints (Rob, 2026-08-25): a preview per checkpoint,
    // no minutes clock. `||` on purpose — a server default of 0 ("off") falls
    // back to the checkpoint cadence rather than disabling previews.
    previewEverySteps: status.defaults.previewEverySteps || (status.defaults.saveEvery ?? 50),
    previewEveryMinutes: 0,
    previewSeconds: 40,
    previewSeed: 424242,
    previewCaption: '',
    previewLyrics: '',
    // Off by default (Rob, 2026-08-25): with a preview at every checkpoint the
    // control and baseline renders would double the pause cost for takes that
    // rarely get listened to. Both remain a checkbox away.
    previewControl: false,
    previewBaseline: false,
    previewScaleMlp: 1.0,
    // Prior preservation is off until a corpus is chosen: it needs a second
    // dataset the user has to nominate, and defaulting it on would silently
    // train a different objective than the form otherwise describes.
    regDatasetId: '',
    regEvery: status.defaults.regularisation?.every ?? 3,
    regTopK: status.defaults.regularisation?.topK ?? 64,
  } : null;
  const form: FormState | null = base ? { ...base, ...edits } : null;
  const presetState = useNamedPresets<FormState>({
    storageKey: PRESETS_KEY, fileKind: 'hot-step-mm3-preset', excluded: PRESET_EXCLUDED,
    snapshot: () => (form ? settingsOf(form) : {}) as Partial<FormState>,
  });
  // A saved preset is "on" when every setting it stores matches the form.
  const userPresetActive = (preset: NamedPreset<FormState>) => !!form
    && Object.entries(preset.settings).every(([k, v]) => same(v, (form as unknown as Record<string, unknown>)[k]));
  // Only what differs from the defaults goes into edits; per-run values stay as they were.
  const loadPreset = (preset: NamedPreset<FormState>) => {
    if (!base) return;
    setEdits(prev => {
      const next: Record<string, unknown> = {};
      for (const k of PRESET_EXCLUDED) if (k in prev) next[k] = (prev as Record<string, unknown>)[k];
      for (const [k, v] of Object.entries(preset.settings)) {
        if (!PRESET_EXCLUDED.has(k) && !same(v, (base as unknown as Record<string, unknown>)[k])) next[k] = v;
      }
      return next as Partial<FormState>;
    });
  };
  const changedSettings = form && base
    ? Object.entries(settingsOf(form)).filter(([k, v]) => !same(v, (base as unknown as Record<string, unknown>)[k])) : [];
  // Time left: steps at the run's mean step time plus the checkpoint previews still to come.
  const mm3Eta = jobKind === 'mm3-train-lm' && jobStatus === 'running' && trainStepSeries.length > 0 ? (() => {
    const ms = trainStepSeries.map(p => p.stepMs).filter((v): v is number => !!v && v > 0);
    return trainingRunEta({ step: trainStepSeries[trainStepSeries.length - 1].step, totalSteps: activeJob?.total ?? 0,
      msPerStep: ms.length ? ms.reduce((a, b) => a + b, 0) / ms.length : null,
      everySteps: form?.previewEverySteps ?? 0, previews: previewsFromRows(trainPreviews) });
  })() : null;

  // Loss-target readings, from the same series the chart draws. Train loss: the trailing mean over
  // `targetLossEpochs` passes of the songs (approximate: the trainer's own window is the one that stops
  // the run). Held-out loss: the last evaluation.
  const stopLossLive = mine && trainTargetLoss > 0;
  const lossMeans = (() => {
    const win = Math.max(1, (form?.targetLossEpochs ?? 5) * Math.max(1, status?.codes ?? 1));
    const out: number[] = []; let sum = 0;
    trainStepSeries.forEach((p, i) => {
      sum += p.loss;
      if (i >= win) sum -= trainStepSeries[i - win].loss;
      if (i >= win - 1) out.push(sum / win);
    });
    return out;
  })();
  const evalLosses = trainEvalSeries.map(e => e.loss);
  const stopMeans = form?.targetLossMetric === 'eval' ? evalLosses : lossMeans;
  const stopReading = stopMeans.at(-1);
  const stepsPerReading = form?.targetLossMetric === 'eval' ? Math.max(1, form.evalEvery) : 1;
  const stepMsList = trainStepSeries.map(p => p.stepMs).filter((v): v is number => !!v && v > 0).slice(-20);
  const paceMs = stepMsList.length ? stepMsList.reduce((a, b) => a + b, 0) / stepMsList.length : 0;
  const lastStep = trainStepSeries.at(-1)?.step ?? 0;
  const descent = lossDescentRate(stopMeans);
  const targetEtaText = stopLossLive && jobStatus === 'running' && paceMs > 0 ? lossTargetEta({
    current: stopReading, rate: descent === null ? null : descent / stepsPerReading, target: trainTargetLoss,
    remaining: Math.max(0, (activeJob?.total ?? 0) - lastStep), paceMs, capMs: mm3Eta?.ms ?? 0,
  }) : null;

  // Peak VRAM for the CURRENT form, not for the defaults — rank is the biggest
  // single term after the base itself, so an estimate pinned to the defaults
  // would be wrong for exactly the users who need it most.
  const chosen = status?.bases?.find(b => b.id === form?.basePrecision);
  const flashCalibrated = status ? (status.flashVramCalibrated ?? mm3FlashVramCalibrated(status.vramModel)) : false;
  // Whether this engine build can run --attn flash AT ALL. The fused
  // attention-training op is CUDA + CPU only, so on a Vulkan (AMD/Intel) or
  // Metal engine ace-train refuses to start — and since flash is the default,
  // that made the DEFAULT recipe unlaunchable with no way to see why (#149).
  // An older server does not report it; treat that as supported, as before.
  const flashSupported = status?.flashSupported !== false;
  // What will ACTUALLY run, which is what the estimate and the checkbox both
  // have to show — the route coerces to exact on an unsupported build.
  const attnEffective: 'exact' | 'flash' =
    form && flashSupported && form.attnBackend === 'flash' ? 'flash' : 'exact';
  const peak = (() => {
    if (!form || !status?.vramModel || !chosen) return null;
    const mb    = estimateMm3PeakMb(chosen.bytes, form.rank, form.maxFrames, status.vramModel,
                                    form.optimizer,
                                    form.cropAnchor === 'song' ? form.prefixFrames : 0,
                                    256, attnEffective)
                + (chosen.extraMb || 0)
                + (form.trainPrecision !== 'full' && chosen.id !== 'bf16' ? status.vramModel.bf16StageMb ?? 0 : 0);
    // Flash mode's own coefficients are not measured yet (MM3_VRAM_MODEL.flash
    // is null server-side), so the number above is silently standing in for
    // exact mode's — say so rather than presenting it as a proven saving.
    const flashCaveat = attnEffective === 'flash' && !flashCalibrated
      ? ` (${t('trainingStudio.mm3.flashVramPending', 'flash: estimate pending measurement')})` : '';
    const total = status.gpuTotalMb || 0;
    const gb    = (mb / 1024).toFixed(1);
    // 0 means the engine could not be read, NOT a card with no memory. Show the
    // estimate without a verdict rather than inventing a scary one.
    if (total <= 0) {
      return { text: t('trainingStudio.mm3.peakUnknown', { gb }) + flashCaveat, tone: 'text-zinc-500' };
    }
    const totalGb = (total / 1024).toFixed(1);
    if (mb + 1536 <= total) {
      return { text: t('trainingStudio.mm3.peakFits', { gb, totalGb }) + flashCaveat, tone: 'text-emerald-500' };
    }
    if (mb <= total) {
      // Fits on paper, with nothing left for the desktop. This is the state that
      // produced 12-14 s/step instead of 3.7 in the f16 A/B, so it is a warning
      // rather than an error.
      return { text: t('trainingStudio.mm3.peakTight', { gb, totalGb }) + flashCaveat, tone: 'text-amber-500' };
    }
    // "Pick a smaller base" is bad advice when the server already established
    // that nothing in the catalogue fits at any rank on the ladder.
    if (status.recommended?.overBudget) {
      return { text: t('trainingStudio.mm3.peakNoFit', { gb, totalGb }) + flashCaveat, tone: 'text-rose-500' };
    }
    return { text: t('trainingStudio.mm3.peakOver', { gb, totalGb }) + flashCaveat, tone: 'text-rose-500' };
  })();

  // The card has no per-type default BUNDLE the way TrainDitForm's pickType
  // does — lokrFactor/rank/alpha already coexist in one flat state, only
  // their visibility changes — so switching method here is just the flag
  // set. PiSSA drops when hopping off plain LoRA; rsLoRA drops for HRA.
  const method: Mm3Method = form ? methodOf(form) : 'lora';
  const pickMethod = (m: Mm3Method) => {
    if (!form || m === method) return;
    if (m === 'lokr') { set('adapterType', 'lokr'); return; }
    setEdits(prev => ({
      ...prev,
      adapterType: 'lora',
      dora: m === 'dora', hira: m === 'hira', loha: m === 'loha', hra: m === 'hra',
      pissa: m === 'lora' ? form.pissa : false,
      hotPizza: m === 'lora' ? form.hotPizza : false,
      rslora: m === 'hra' ? false : form.rslora,
    }));
  };

  const startTrain = async () => {
    if (!form) return;
    setBusy(true);
    try {
      const recipe = await resolveTrainingRecipe<FormState>('mm3-lm', {
        ...form, activePreset, trainLaunder, flashSupported,
      }, activePreset === 'custom' ? undefined : activePreset);
      const body = recipe.execution as Mm3TrainLmRequest;
      assertRecipeWorker(recipe);
      if (batchDraft?.length) {
        // One recipe for every dataset: rank and base are pinned, not re-recommended per dataset.
        // The preview song and the trigger word belong to this dataset, so each run picks its own
        // (the server falls back to each dataset's custom tag when no trigger is sent).
        const { previewSongId: _song, ...preview } = body.preview ?? {};
        const { trigger: _trigger, ...shared } = body;
        const batchBody = { ...shared, ...(body.preview ? { preview } : {}) };
        if (runOn) {
          await dispatchMm3Batch(runOn, { datasetIds: batchDraft, clearCache: batchClearCache, recipe: batchBody });
          void trainingWritePlan(client, runOn);
          setBatchDraft(null);
          await setTrainingWorker(runOn);
          return;
        }
        await startMm3Batch(batchDraft, batchBody, batchClearCache);
        setPhase('train');
        return;
      }
      await startMm3TrainLm(body);
    } catch (err) {
      useTrainingStore.setState({ error: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  };

  if (!status && !statusError) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setEdits(e => ({ ...e, [k]: v }));
  // Deletes the edit rather than writing base[k] back — base is re-derived
  // every render, so a delete keeps tracking it if the server default moves.
  const reset = <K extends keyof FormState>(k: K) =>
    setEdits(e => { const rest = { ...e }; delete rest[k]; return rest; });
  // PiSSA's own onChange forces hotPizza off whenever pissa goes false (HOT-PiZZA
  // needs PiSSA underneath it); resetting pissa back to a false base has to apply
  // that same coupling, or hotPizza could be left on with pissa off.
  const resetPissa = () => {
    if (!base) return;
    setEdits(prev => {
      const rest = { ...prev };
      delete rest.pissa;
      if (!base.pissa) rest.hotPizza = false;
      return rest;
    });
  };
  // HOT-PiZZA's own onChange forces pissa ON whenever hotPizza goes true. A plain
  // delete of hotPizza is not enough when a PISSA OVERRIDE is sitting underneath
  // it (pissa turned off by hand, then HOT-PiZZA reset back toward a shipped
  // recipe where base.hotPizza is true): the displayed value is
  // form.pissa && form.hotPizza, so with pissa still pinned false the icon never
  // clears. Dropping the pissa override too lets it fall back to base.pissa,
  // which is what HOT-PiZZA being on in the recipe implies.
  const resetHotPizza = () => {
    if (!base) return;
    setEdits(prev => {
      const rest = { ...prev };
      delete rest.hotPizza;
      if (base.hotPizza) delete rest.pissa;
      return rest;
    });
  };
  // pickMethod('lokr') writes only adapterType, but every LoRA-family method
  // (m !== 'lokr') also pins dora/hira/loha/hra plus pissa/hotPizza/rslora into
  // edits (see pickMethod above) — a reset that called pickMethod would leave
  // those overrides sitting in edits even after the method itself went back to
  // the server's, pinning them against any future default change. Delete the
  // method-row keys outright instead, then apply the same coupling pickMethod
  // applies, relative to the method the restored base now shows.
  const resetMethod = () => {
    if (!base) return;
    setEdits(prev => {
      const rest = { ...prev };
      delete rest.adapterType;
      delete rest.dora;
      delete rest.hira;
      delete rest.loha;
      delete rest.hra;
      const baseMethod = methodOf(base);
      if (baseMethod !== 'lora') { delete rest.pissa; delete rest.hotPizza; }
      if (baseMethod === 'hra') delete rest.rslora;
      return rest;
    });
  };

  // Presets are DERIVED from the fields, never stored: the row highlights
  // whichever preset the four governing fields currently equal, and shows
  // Custom otherwise. Editing a field therefore leaves the preset row honest
  // without a second piece of state that could disagree with the form.
  const PRESET_ORDER: Mm3PresetName[] = ['fast', 'balanced', 'thorough'];   // default first (Fast since 2026-11-07)
  const presets = status?.presets;
  const activePreset: Mm3PresetName | 'custom' = (() => {
    if (!presets || !form) return 'custom';
    const hit = PRESET_ORDER.find(p => {
      const q = presets[p];
      return q && form.steps === q.steps && form.lr === q.lr
          && form.maxFrames === q.maxFrames && form.prefixFrames === q.prefixFrames;
    });
    return hit ?? 'custom';
  })();
  const applyPreset = (p: Mm3PresetName) => {
    const q = presets?.[p];
    if (!q) return;
    setEdits(e => ({ ...e, steps: q.steps, lr: q.lr, maxFrames: q.maxFrames, prefixFrames: q.prefixFrames,
                     stopMode: 'steps' }));
  };

  // Coupled fields reset through their own handlers so a chip never leaves one half behind.
  const resetSetting = (key: string) => {
    if (key === 'pissa') resetPissa();
    else if (key === 'hotPizza') resetHotPizza();
    else if (['adapterType', 'dora', 'hira', 'loha', 'hra'].includes(key)) resetMethod();
    else reset(key as keyof FormState);
  };
  const resetAllSettings = () => setEdits(prev => Object.fromEntries(Object.entries(prev).filter(([k]) => PRESET_EXCLUDED.has(k))) as Partial<FormState>);

  const hasCodes = (status?.codes ?? 0) > 0;
  const hasLaundered = (status?.codesLaundered ?? 0) > 0;
  const trainBlocked = (status?.missingForTrain.length ?? 0) > 0;

  return (
    <div className="flex flex-col gap-4">
      {(statusError || storeError) && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{statusError || storeError}</span>
        </div>
      )}

      {!batchDraft?.length && (
        <TrainingBatchCard disabled={jobRunning} onOpen={() => setBatchOpen(true)}
          description={t('trainingStudio.mm3.batchCardInfo', 'Choose datasets, then apply the recipe below to each in order: codes, then LoRA training. The queue survives closing the page and server restarts.')} />
      )}
      <Yue2AitkBatchWizard open={batchOpen} family="mm3" onClose={() => setBatchOpen(false)} />

      {/* ── LM LoRA ── */}
      <TrainingRunPanel
        title={t('trainingStudio.mm3.trainTitle', 'LM LoRA training')}
        description={t('trainingStudio.mm3.trainBlurb',
          'Trains the planner LM on this dataset. Checkpoints are written straight to the MiniMax-Music3 '
          + 'adapter folder, so they appear in the generation panel\'s adapter picker as soon as they are '
          + 'saved — there is no install step. The engine is paused for the run.')}>
        {trainBlocked ? (
          <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t('trainingStudio.mm3.missing', 'Missing model files')}: {status?.missingForTrain.join(', ')}
            </span>
          </div>
        ) : !hasCodes && !hasLaundered && !batchDraft?.length ? (   // a batch exports codes itself
          <div className="flex flex-col items-start gap-3">
            <div className="flex items-start gap-2 text-xs text-zinc-500">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.mm3.needsCodes',
                'Export the RVQ codes first — training reads them, not the audio.')}
            </div>
            <button
              onClick={() => setPhase('preprocess')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
            >
              {t('trainingStudio.mm3.goToCodes', 'Go to Codes')}
            </button>
          </div>
        ) : form && base && (
          <>
            {/* -- Codes cache (the launder gate) --------------------------
                Only rendered when a laundered cache exists; a dataset without
                one trains on the standard cache with no extra UI at all. */}
            {hasLaundered && (
              <div className="mb-3">
                <Toggle
                  accent="amber"
                  checked={trainLaunder}
                  onChange={setTrainLaunder}
                  disabled={!hasLaundered}
                  label={t('trainingStudio.mm3.trainLaunderLabel', 'Train on cover-laundered codes')}
                  info={t('trainingStudio.mm3.trainLaunderBlurb',
                    'Trains on the vocal-forward code targets for dense mixes ({{n}} tracks laundered) '
                    + 'instead of the standard cache. Off: the standard cache, exactly as before.',
                    { n: status?.codesLaundered ?? 0 })}
                />
              </div>
            )}
            {/* -- Recipe (2026-09-07) -------------------------------------
                Three recipes that tied blind; they trade minutes, not
                audible quality. The cards reflect the fields, so a hand edit
                shows as custom rather than misreporting a recipe. Saved
                presets and the changed-settings chips sit under them, as on
                YuE2's page. */}
            {presets && (
              <div className="mb-3">
                <div className="flex items-center gap-2 mb-2">
                  <ParamLabel label={t('trainingStudio.mm3.preset', 'Recipe')}
                    className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
                    info={t('trainingStudio.mm3.presetInfo', 'Fast, Balanced and Thorough tied in a blind listen; they trade minutes, not audible quality. Each card shows its steps and its time relative to Balanced (steps times crop length). Editing steps, learning rate, crop or history shows custom.')} />
                  {activePreset === 'custom' && !presetState.presets.some(userPresetActive)
                    && <span className="text-[11px] text-zinc-500">{t('trainingStudio.mm3.preset.custom', 'custom')}</span>}
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {PRESET_ORDER.filter(p => presets[p]).map(p => {
                    const q = presets[p];
                    const bal = presets.balanced;
                    const ratio = bal ? Number(((q.steps * q.maxFrames) / (bal.steps * bal.maxFrames)).toFixed(1)) : 1;
                    return <TrainingPresetChoice key={p} selected={activePreset === p} disabled={locked} onClick={() => applyPreset(p)}>
                      <span className="text-base font-bold">{t(`trainingStudio.mm3.preset.${p}`, p)}</span>
                      <span className="text-xs text-zinc-500">{q.steps} × {form.gradAccum} {t('trainingStudio.mm3.presetSongs', form.gradAccum === 1 ? 'song' : 'songs')} · {ratio === 1
                        ? t('trainingStudio.yue2.method.presetBaseline', '1× time (baseline)')
                        : t('trainingStudio.yue2.method.presetRelative', '{{ratio}} the time', { ratio: `${ratio}×` })}</span>
                    </TrainingPresetChoice>;
                  })}
                </div>
                {activePreset !== 'custom' && <p className="text-[11px] text-zinc-500 mt-2">
                  {t(`trainingStudio.mm3.preset.${activePreset}Info`, '')}
                </p>}
                <UserPresetRow state={presetState} isActive={userPresetActive} onLoad={loadPreset} busy={locked}
                  info={t('trainingStudio.mm3.presetHint', 'A preset captures the training settings (steps, learning rate, crop, rank, optimizer, adapter method, previews), never the dataset, trigger word or preview song. Click a name to load it; the save icon overwrites it with the current settings, the download icon exports it to share. Presets are kept in this browser.')} />
                <ChangedSettingsRow changed={changedSettings} onReset={resetSetting} onResetAll={resetAllSettings} disabled={locked} />
              </div>
            )}
            {/* -- Stopping strategy ---------------------------------------
                Two ways to answer "when is this run done": a step count, or a
                loss to reach. The step field never goes away, because in loss
                mode it is the CAP - a target that is never met still has to end
                the run somewhere. */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
              <label className="flex flex-col gap-1">
                <ParamLabel
                  label={t('trainingStudio.mm3.stopMode', 'Train until')}
                  info={t('trainingStudio.mm3.stopModeInfo',
                    'Whether the run ends on a fixed step count, or as soon as a target loss is '
                    + 'reached. In target-loss mode the step count below still caps the run, so a '
                    + 'target that never arrives still ends somewhere.')}
                  className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                  onReset={form.stopMode !== base.stopMode ? () => reset('stopMode') : undefined}
                />
                <StyledSelect
                  accent="amber"
                  value={form.stopMode}
                  onChange={v => set('stopMode', v)}
                  options={[
                    { value: 'steps', label: t('trainingStudio.mm3.stopSteps', 'Step count') },
                    { value: 'loss', label: t('trainingStudio.mm3.stopLoss', 'Target loss') },
                  ]}
                  className="w-full"
                />
              </label>
              <NumField
                label={form.stopMode === 'loss'
                  ? t('trainingStudio.mm3.maxSteps', 'Max steps')
                  : t('trainingStudio.mm3.steps', 'Steps')}
                value={form.steps}
                onChange={v => set('steps', v)}
                defaultValue={base.steps} onReset={() => reset('steps')}
                meta={t('trainingStudio.mm3.stepsMeta', 'default 500')}
                info={form.stopMode === 'loss'
                  ? t('trainingStudio.mm3.maxStepsHint',
                      'The cap. If the target never arrives, the run ends here.')
                  : t('trainingStudio.mm3.stepsInfo',
                      'How many training steps to run. Higher lets the adapter learn more but takes '
                      + 'longer and risks overfitting a small dataset; lower is faster but may undercook '
                      + 'the likeness.')} />
              {form.stopMode === 'loss' && (
                <>
                  <NumField label={t('trainingStudio.mm3.targetLoss', 'Target loss')}
                    value={form.targetLoss} onChange={v => set('targetLoss', v)} step={0.05}
                    defaultValue={base.targetLoss} onReset={() => reset('targetLoss')}
                    info={t('trainingStudio.mm3.targetLossHint',
                      'Stops as soon as the loss reaches this. Down is NOT automatically '
                      + 'better: under ~0.05 training loss the runs that got there had '
                      + 'memorised the songs.')} />
                  <label className="flex flex-col gap-1">
                    <ParamLabel
                      label={t('trainingStudio.mm3.targetLossMetric', 'Measured on')}
                      info={form.targetLossMetric === 'eval'
                        ? t('trainingStudio.mm3.metricEvalHint',
                            'The number that says the adapter GENERALISES rather than memorised. '
                            + 'It only lands on evaluation steps, so with Evaluate every set at '
                            + '{{n}} it can only fire every {{n}} steps — lower that first, and '
                            + 'keep a hold-out fraction above 0.',
                            { n: form.evalEvery })
                        : t('trainingStudio.mm3.metricTrainHint',
                            'The mean of the last {{n}} full passes over the dataset, because one '
                            + 'step is one crop of one song and swings more than the whole run '
                            + 'does. Whole passes, not a step count: a window that is not a '
                            + 'multiple of your {{songs}} songs weighs some tracks more than '
                            + 'others. Available on every run, but it cannot tell learning from '
                            + 'memorising.',
                            { n: form.targetLossEpochs, songs: status?.codes ?? 0 })}
                      className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                      onReset={form.targetLossMetric !== base.targetLossMetric ? () => reset('targetLossMetric') : undefined}
                    />
                    <StyledSelect
                      accent="amber"
                      value={form.targetLossMetric}
                      onChange={v => set('targetLossMetric', v)}
                      options={[
                        { value: 'train', label: t('trainingStudio.mm3.metricTrain', 'Training loss (trailing mean)') },
                        { value: 'eval', label: t('trainingStudio.mm3.metricEval', 'Held-out loss') },
                      ]}
                      className="w-full"
                    />
                  </label>
                </>
              )}
            </div>
            {form.stopMode === 'loss' && form.targetLossMetric === 'eval'
              && (form.holdout <= 0 || form.evalEvery <= 0) && (
              <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mb-3">
                <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                <span>
                  {t('trainingStudio.mm3.targetNeedsEval',
                    'Targeting the held-out loss needs a hold-out fraction above 0 and Evaluate '
                    + 'every above 0 — both are in Advanced. The run will be refused otherwise.')}
                </span>
              </div>
            )}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField label={t('trainingStudio.mm3.saveEvery', 'Checkpoint every')} value={form.saveEvery}
                onChange={v => set('saveEvery', v)}
                defaultValue={base.saveEvery} onReset={() => reset('saveEvery')}
                meta={t('trainingStudio.mm3.saveEveryMeta', 'default 100 steps')}
                info={t('trainingStudio.mm3.saveEveryInfo',
                  'How many training steps between saved checkpoints. Lower saves more often, at the '
                  + 'cost of disk space and a little time; higher risks losing more progress if the run '
                  + 'is stopped early or crashes.')} />
              <NumField label={t('trainingStudio.mm3.rank', 'Rank')} value={form.rank}
                onChange={v => set('rank', v)}
                defaultValue={base.rank} onReset={() => reset('rank')}
                meta={t('trainingStudio.mm3.rankMeta', 'default 128, follows the VRAM recommendation')}
                info={t('trainingStudio.mm3.rankInfo',
                  'The LoRA\'s capacity — how many parameters the adapter learns. Higher captures more '
                  + 'detail but needs more VRAM and is more prone to overfitting on a small dataset; '
                  + 'lower trains faster and fits smaller cards but may undercook the likeness.')} />
              <NumField label={t('trainingStudio.mm3.maxFrames', 'Crop (frames)')} value={form.maxFrames}
                onChange={v => set('maxFrames', v)} step={50} min={64} max={9000}
                defaultValue={base.maxFrames} onReset={() => reset('maxFrames')}
                meta={t('trainingStudio.mm3.maxFramesMeta', 'default 9000 frames')}
                info={t('trainingStudio.mm3.maxFramesInfo',
                  'How many frames of audio each training crop covers. Longer crops give the model more '
                  + 'context per step but cost VRAM quadratically and slow each step down; shorter crops '
                  + 'are cheaper but see less of the song at once.')} />
              <NumField label={t('trainingStudio.mm3.cropStartFrac', 'Crops at song start')}
                value={form.cropStartFrac} onChange={v => set('cropStartFrac', v)} step={0.05}
                defaultValue={base.cropStartFrac} onReset={() => reset('cropStartFrac')}
                info={t('trainingStudio.mm3.cropStartFracHint',
                  'Share anchored at frame 0 — what teaches songs to OPEN like songs. '
                  + 'Too low and renders jump in mid-flow.') as string} />
              <NumField label={t('trainingStudio.mm3.cropStartTiles', 'Start tiles')}
                value={form.cropStartTiles} onChange={v => set('cropStartTiles', v)} step={1}
                defaultValue={base.cropStartTiles} onReset={() => reset('cropStartTiles')}
                info={t('trainingStudio.mm3.cropStartTilesHint',
                  'Half the start share stays at frame 0; the rest lands on aligned tiles '
                  + 'after it, teaching the intro→build→verse arc under short crops. '
                  + '1 = frame 0 only.') as string} />
              <NumField label={t('trainingStudio.mm3.cropEndFrac', 'Crops at song end')}
                value={form.cropEndFrac} onChange={v => set('cropEndFrac', v)} step={0.05}
                defaultValue={base.cropEndFrac} onReset={() => reset('cropEndFrac')}
                info={t('trainingStudio.mm3.cropEndFracHint',
                  'Share flush to the track end — the only place EOS is taught. The '
                  + 'REMAINDER of these two is the random share (currently '
                  + `${Math.max(0, Math.round((1 - form.cropStartFrac - form.cropEndFrac) * 100))}% mid-song crops).`) as string} />
              <NumField label={t('trainingStudio.mm3.depthLossWeight', 'Acoustic loss weight')}
                value={form.depthLossWeight} onChange={v => set('depthLossWeight', v)} step={0.1}
                defaultValue={base.depthLossWeight} onReset={() => reset('depthLossWeight')}
                info={t('trainingStudio.mm3.depthLossWeightHint',
                  'Trains the adapter to keep vocal timbre intact: acoustic codebooks are '
                  + 'supervised through the frozen depth decoder. 0 disables — renders then '
                  + 'drift into chipmunk/goblin voices. Leave at 1.') as string} />
              <Toggle accent="amber" className="col-span-2" checked={form.keepResumeState}
                onChange={v => set('keepResumeState', v)}
                defaultValue={base.keepResumeState} onReset={() => reset('keepResumeState')}
                label={t('trainingStudio.mm3.keepResumeState', 'Keep resume state after completion')}
                info={t('trainingStudio.mm3.keepResumeStateHint',
                  'The optimizer state (about 4 GB) lets a finished run be continued past its step '
                  + 'count. Off: it is deleted once the run ends. A run that stops early keeps it either way.')} />
              <Toggle accent="amber" className="col-span-2" checked={form.updateLinkedPresets}
                onChange={v => set('updateLinkedPresets', v)}
                defaultValue={base.updateLinkedPresets} onReset={() => reset('updateLinkedPresets')}
                label={t('trainingStudio.mm3.updateLinkedPresets', 'Update linked album presets')}
                info={t('trainingStudio.mm3.updateLinkedPresetsHint',
                  'When the run finishes, Lyric Studio album presets linked to this dataset are pointed at '
                  + 'its final checkpoint. Turn off for test runs so they do not replace a preset you use.')} />
              <Toggle accent="amber" className="col-span-2" checked={form.longTracks === 'exclude'}
                onChange={v => set('longTracks', v ? 'exclude' : 'crop')}
                defaultValue={base.longTracks === 'exclude'} onReset={() => reset('longTracks')}
                label={t('trainingStudio.mm3.longTracksExclude', 'Leave out tracks longer than the window')}
                info={t('trainingStudio.mm3.longTracksHint',
                  'The recipe trains each track as one whole sequence. A track longer than the window '
                  + '(360 s at 9000 frames) cannot be, so it is left out and named in the log. Off: it is '
                  + 'trained in crops of the window instead, which is the pre-2026-09-11 behaviour.')} />
              <NumField label={t('trainingStudio.mm3.depthLossFrames', 'Acoustic frames/step')}
                value={form.depthLossFrames} onChange={v => set('depthLossFrames', v)} step={16}
                defaultValue={base.depthLossFrames} onReset={() => reset('depthLossFrames')}
                info={t('trainingStudio.mm3.depthLossFramesHint',
                  'Frames sampled per step for the acoustic loss.') as string} />
            </div>
            <label className="flex flex-col gap-1 mt-3">
              <ParamLabel
                label={t('trainingStudio.mm3.trigger', 'Trigger word')}
                info={t('trainingStudio.mm3.triggerInfo',
                  'A word associated with this adapter. Typed into a prompt at render time, it invokes '
                  + 'the adapter\'s identity. Leave blank to skip triggering entirely — the adapter still '
                  + 'trains and still loads, it just has no invocation word.')}
                className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
              />
              <input className={INPUT} value={form.trigger} onChange={e => set('trigger', e.target.value)} />
            </label>
            {/* A sibling, not a child: a label nested inside a label is invalid
                and clicking the toggle would focus the text input instead. */}
            <Toggle accent="amber" className="mt-2" checked={form.triggerPrepend}
              onChange={v => set('triggerPrepend', v)}
              disabled={!form.trigger.trim()}
              defaultValue={base.triggerPrepend} onReset={() => reset('triggerPrepend')}
              label={t('trainingStudio.mm3.triggerPrepend', 'Train the trigger')}
              info={t('trainingStudio.mm3.triggerPrependHint',
                'Puts "trigger, " at the front of every training caption, in memory — your files '
                + 'are not touched. Leave this ON. Off: the word is only recorded in the '
                + 'adapter sidecar and never learned, so typing it at render time bolts an unseen '
                + 'token sequence onto your prompt and makes the result WORSE, not better.')} />

            {/* -- Previews ------------------------------------------------
                Above Advanced on purpose: this is the control that decides
                whether the user finds out at step 200 or at step 800 that the
                run is going the wrong way. */}
            <TrainingPreviewSettings
              disabled={locked}
              title={t('trainingStudio.mm3.previewTitle', 'Audio previews during training')}
              enabled={form.previewEverySteps > 0 || form.previewEveryMinutes > 0}
              onEnabledChange={enabled => setEdits(previous => ({ ...previous,
                previewEverySteps: enabled ? base.previewEverySteps : 0,
                previewEveryMinutes: enabled ? base.previewEveryMinutes : 0,
              }))}
              enableLabel={t('trainingStudio.mm3.previewEnabled', 'Render previews during training')}
              enableInfo={t('trainingStudio.mm3.previewEnabledInfo', 'Pauses the trainer to render checkpoint samples, then resumes it. Turn off to set both preview cadences to zero.')}
              enableDefault={base.previewEverySteps > 0 || base.previewEveryMinutes > 0}
              afterToggle={
                <p className="text-[10px] text-zinc-500 leading-snug mt-1">
                  {t('trainingStudio.mm3.previewBlurb',
                    'Renders a sample from the checkpoint while the run is still going, so you can hear '
                    + 'whether it is heading the right way and abort if not. Each preview point pauses '
                    + 'training for about a minute: the trainer has to hand the whole card to the render, '
                    + 'so it saves its optimizer state, exits, renders, and resumes exactly where it was. '
                    + 'Zero in both fields turns previews off.')}
                </p>
              }
            >
              <div className="md:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-3">
                <NumField disabled={locked} label={t('trainingStudio.mm3.previewEverySteps', 'Every N steps')}
                  value={form.previewEverySteps} onChange={v => set('previewEverySteps', v)}
                  defaultValue={base.previewEverySteps} onReset={() => reset('previewEverySteps')}
                  step={50} info={t('trainingStudio.mm3.previewStepsHint', '0 = off') as string} />
                <NumField disabled={locked} label={t('trainingStudio.mm3.previewEveryMinutes', 'Or every N minutes')}
                  value={form.previewEveryMinutes} onChange={v => set('previewEveryMinutes', v)}
                  defaultValue={base.previewEveryMinutes} onReset={() => reset('previewEveryMinutes')}
                  step={5} info={t('trainingStudio.mm3.previewMinutesHint',
                    'Whichever comes first') as string} />
                <NumField disabled={locked} label={t('trainingStudio.mm3.previewSeconds', 'Length (s)')}
                  value={form.previewSeconds} onChange={v => set('previewSeconds', v)}
                  defaultValue={base.previewSeconds} onReset={() => reset('previewSeconds')}
                  step={4} info={t('trainingStudio.mm3.previewSecondsHint',
                    '24 s costs about 16 s of GPU') as string} />
                <NumField disabled={locked} label={t('trainingStudio.mm3.previewSeed', 'Preview seed')}
                  value={form.previewSeed} onChange={v => set('previewSeed', v)}
                  defaultValue={base.previewSeed} onReset={() => reset('previewSeed')}
                  info={t('trainingStudio.mm3.previewSeedHint',
                    'Fixed across the run') as string} />
                <NumField disabled={locked} label={t('trainingStudio.mm3.previewScaleMlp', 'Preview MLP scale')}
                  value={form.previewScaleMlp} onChange={v => set('previewScaleMlp', v)}
                  defaultValue={base.previewScaleMlp} onReset={() => reset('previewScaleMlp')}
                  step={0.05} info={t('trainingStudio.mm3.previewScaleMlpHint',
                    'How hard the adapter’s MLP delta is applied in previews only. '
                    + '1 = full, 0 = attention only.') as string} />
              </div>
              {/* Explicit song pick (2026-09-05): a select over the dataset's
                  usable rows, default = auto. Fixes the auto-pick landing on
                  a noise interlude (an album dataset's "instrumental_2",
                  40s of "[Instrumental]") by letting a user route around
                  whatever the ladder in mm3Preview.ts picks — the ladder
                  itself was also fixed the same day, this is the manual
                  override for when it still isn't the song you want. Wired
                  through the SAME override the caption box below uses:
                  picking a song here only takes effect while that box is
                  blank. */}
              <label className="md:col-span-2 flex flex-col gap-1 mt-3">
                <ParamLabel
                  label={t('trainingStudio.mm3.previewSong', 'Preview song')}
                  info={form.previewCaption.trim()
                    ? t('trainingStudio.mm3.previewSongDisabled', 'Ignored while the caption box below is filled in.')
                    : t('trainingStudio.mm3.previewSongHint',
                        'Which held-out song a preview render uses. Rows marked "likely instrumental" '
                        + 'are exactly what the auto-pick now avoids — pick one anyway if that is '
                        + 'deliberately what you want to hear.')}
                  className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                />
                <StyledSelect
                  accent="amber"
                  value={form.previewSongId}
                  disabled={locked || !!form.previewCaption.trim()}
                  onChange={v => set('previewSongId', v)}
                  options={[
                    { value: '', label: t('trainingStudio.mm3.previewSongAuto', 'Auto (held-out, real lyrics preferred)') },
                    ...(status?.previewSongs ?? []).map(s => ({
                      value: s.id,
                      label: `${s.filename}${s.held ? '' : ` (${t('trainingStudio.mm3.previewSongTraining', 'training')})`}`
                        + `${!s.usableLyrics ? ` — ${t('trainingStudio.mm3.previewSongInstrumental', 'likely instrumental')}` : ''}`
                        + `${s.durationS > 0 ? ` · ${Math.round(s.durationS)}s` : ''}`,
                    })),
                  ]}
                  className="w-full"
                />
              </label>
              <label className="md:col-span-2 flex flex-col gap-1 mt-3">
                <ParamLabel
                  label={t('trainingStudio.mm3.previewCaption', 'Preview caption')}
                  info={t('trainingStudio.mm3.previewCaptionHint',
                    'Leave blank unless you have a reason not to: a held-out caption carries the '
                    + 'artist’s true BPM and tuning, and a wrong one becomes audible as soon as '
                    + 'identity bakes in. The trigger is prepended as "trigger, " on the caption’s '
                    + 'first line, which is the exact shape the training rows use.')}
                  className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                  onReset={form.previewCaption !== base.previewCaption ? () => reset('previewCaption') : undefined}
                />
                <textarea
                  disabled={locked}
                  className={`${INPUT} font-mono text-[11px] leading-snug`} rows={3}
                  placeholder={t('trainingStudio.mm3.previewCaptionPlaceholder',
                    'Blank = the preview-song pick above, or auto (first held-out song with real '
                    + 'lyrics), with the trigger prepended') as string}
                  value={form.previewCaption}
                  onChange={e => set('previewCaption', e.target.value)}
                />
              </label>
              <label className="md:col-span-2 flex flex-col gap-1 mt-3">
                <ParamLabel
                  label={t('trainingStudio.mm3.previewLyrics', 'Preview lyrics')}
                  info={t('trainingStudio.mm3.previewLyricsHint',
                    'Lyrics sung in the preview. Blank uses the lyrics of the picked or auto-chosen song.')}
                  className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                  onReset={form.previewLyrics !== base.previewLyrics ? () => reset('previewLyrics') : undefined}
                />
                <textarea
                  disabled={locked}
                  className={`${INPUT} font-mono text-[11px] leading-snug`} rows={3}
                  value={form.previewLyrics}
                  onChange={e => set('previewLyrics', e.target.value)}
                />
              </label>
              <div className="md:col-span-2 flex flex-col gap-1.5 mt-3">
                <Toggle accent="amber" disabled={locked} checked={form.previewControl}
                  onChange={v => set('previewControl', v)}
                  defaultValue={base.previewControl} onReset={() => reset('previewControl')}
                  label={t('trainingStudio.mm3.previewControl', 'Also render a neutral control caption')}
                  info={t('trainingStudio.mm3.previewControlHint',
                    'An off-genre prompt rendered WITH the adapter. This is the one that catches '
                    + 'the adapter damaging the base planner - on the artist caption a damaged '
                    + 'model and a good one both sound roughly like the artist.')} />
                <Toggle accent="amber" disabled={locked} checked={form.previewBaseline}
                  onChange={v => set('previewBaseline', v)}
                  defaultValue={base.previewBaseline} onReset={() => reset('previewBaseline')}
                  label={t('trainingStudio.mm3.previewBaseline', 'Render a no-adapter reference first')}
                  info={t('trainingStudio.mm3.previewBaselineHint',
                    'Free - the engine is still up before the run starts. Without it there is '
                    + 'nothing to judge "worse than base" against.')} />
              </div>
            </TrainingPreviewSettings>

            {/* -- Prior preservation -------------------------------------- */}
            <div className="mt-3 rounded-lg border border-zinc-200 dark:border-white/10 p-3">
              <div className="flex items-center gap-2">
                <ShieldCheck size={13} className="text-amber-500" />
                <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                  {t('trainingStudio.mm3.regTitle', 'Prior preservation')}
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 leading-snug mt-1">
                {t('trainingStudio.mm3.regBlurb',
                  'Spends some steps on an UNRELATED dataset, scored against what the base model '
                  + 'itself predicted there rather than against that music. The adapter is then '
                  + 'penalised for changing its mind about material that has nothing to do with the '
                  + 'artist, which is the only thing in the objective that separates "learned the '
                  + 'voice" from "rewrote the planner". Leave the corpus unset to turn it off.')}
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
                <label className="flex flex-col gap-1 md:col-span-1">
                  <ParamLabel
                    label={t('trainingStudio.mm3.regDataset', 'Corpus')}
                    info={(status?.regCandidates?.length ?? 0) === 0
                      ? t('trainingStudio.mm3.regNone',
                          'No other dataset has RVQ codes yet — a regularisation corpus needs exactly '
                          + 'what a training corpus needs. Run the codes export on one.')
                      : t('trainingStudio.mm3.regDatasetHint',
                          'Which unrelated dataset supplies the prior-preservation steps. Pick something '
                          + 'with nothing in common with this artist. Its lyrics may be empty; only its '
                          + 'captions and codes are used.')}
                    className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                  />
                  <StyledSelect
                    accent="amber"
                    value={form.regDatasetId}
                    onChange={v => set('regDatasetId', v)}
                    options={[
                      { value: '', label: t('trainingStudio.mm3.regOff', 'Off') },
                      ...(status?.regCandidates ?? []).map(d => ({ value: d.id, label: `${d.name} (${d.songs})` })),
                    ]}
                    className="w-full"
                  />
                </label>
                <NumField label={t('trainingStudio.mm3.regEvery', 'Every N steps')}
                  value={form.regEvery} onChange={v => set('regEvery', v)}
                  defaultValue={base.regEvery} onReset={() => reset('regEvery')}
                  info={t('trainingStudio.mm3.regEveryHint',
                    '3 = one prior step per two style steps') as string} />
                <NumField label={t('trainingStudio.mm3.regTopK', 'Classes kept')}
                  value={form.regTopK} onChange={v => set('regTopK', v)} step={64}
                  defaultValue={base.regTopK} onReset={() => reset('regTopK')}
                  info={t('trainingStudio.mm3.regTopKHint',
                    'Coverage: 64 = 90%, 128 = 94%, 256 = 97%') as string} />
              </div>
              {form.regDatasetId && (
                <div className="mt-2 text-[10px] text-amber-600/90 dark:text-amber-400/90 leading-snug">
                  {t('trainingStudio.mm3.regDilution',
                    'This dilutes style exposure: at every {{n}} steps only {{s}} of your {{total}} '
                    + 'steps train on the artist. Raise Steps to about {{want}} to keep the same '
                    + 'exposure as running without it.',
                    {
                      n: form.regEvery,
                      s: form.steps - Math.floor(form.steps / Math.max(2, form.regEvery)),
                      total: form.steps,
                      want: Math.round(form.steps / (1 - 1 / Math.max(2, form.regEvery))),
                    })}
                </div>
              )}
            </div>

            <button
              onClick={() => setAdvanced(v => !v)}
              className="flex items-center gap-1 mt-3 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {t('trainingStudio.mm3.advanced', 'Advanced')}
            </button>

            {advanced && (
              <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 flex flex-col gap-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <NumField label={t('trainingStudio.mm3.alpha', 'Alpha')} value={form.alpha}
                    onChange={v => set('alpha', v)}
                    defaultValue={base.alpha} onReset={() => reset('alpha')}
                    meta={t('trainingStudio.mm3.alphaMeta', 'default 128')}
                    info={t('trainingStudio.mm3.alphaInfo',
                      'Scales how strongly the LoRA\'s learned update is applied on top of the frozen '
                      + 'weights. Usually left equal to rank; raising it relative to rank strengthens '
                      + 'the adapter\'s effect, lowering it softens it.')} />
                  <NumField label={t('trainingStudio.mm3.lr', 'Learning rate')} value={form.lr}
                    onChange={v => set('lr', v)} step={1e-5}
                    defaultValue={base.lr} onReset={() => reset('lr')}
                    meta={t('trainingStudio.mm3.lrMeta', 'default 8e-5')}
                    info={t('trainingStudio.mm3.lrInfo',
                      'How fast the adapter\'s weights move each step. Higher can learn faster but risks '
                      + 'instability or overfitting; lower is safer but slower. Under the Prodigy '
                      + 'optimizer this becomes a schedule multiplier only — Prodigy picks its own '
                      + 'effective step size.')} />
                  <NumField label={t('trainingStudio.mm3.gradAccum', 'Grad accum')} value={form.gradAccum}
                    onChange={v => set('gradAccum', v)}
                    defaultValue={base.gradAccum} onReset={() => reset('gradAccum')}
                    meta={t('trainingStudio.mm3.gradAccumMeta', 'default 1')}
                    info={t('trainingStudio.mm3.gradAccumInfo',
                      'Number of steps whose gradients are summed before the weights update once, '
                      + 'simulating a larger batch without more VRAM. Raising it smooths the gradient '
                      + 'estimate at the cost of a slower wall-clock per effective step; 1 = no '
                      + 'accumulation.')} />
                  <NumField label={t('trainingStudio.mm3.seed', 'Seed')} value={form.seed}
                    onChange={v => set('seed', v)}
                    defaultValue={base.seed} onReset={() => reset('seed')}
                    meta={t('trainingStudio.mm3.seedMeta', 'default 42')}
                    info={t('trainingStudio.mm3.seedInfo',
                      'The random seed for crop order and initialisation. Changing it gives a different '
                      + 'run outcome for otherwise-identical settings; keeping it fixed makes two runs '
                      + 'comparable.')} />
                  <NumField label={t('trainingStudio.mm3.warmup', 'Warmup steps')} value={form.warmup}
                    onChange={v => set('warmup', v)} step={5} min={0}
                    defaultValue={base.warmup} onReset={() => reset('warmup')}
                    meta={t('trainingStudio.mm3.warmupMeta', 'default 25')}
                    info={t('trainingStudio.mm3.warmupInfo',
                      'Steps over which the learning rate ramps up from zero to its peak before the '
                      + 'decay begins. Step 1 still runs at a rate of 0.')} />
                  <NumField label={t('trainingStudio.mm3.lrEndFrac', 'LR end fraction')} value={form.lrEndFrac}
                    onChange={v => set('lrEndFrac', v)} step={0.001} min={0} max={1}
                    defaultValue={base.lrEndFrac} onReset={() => reset('lrEndFrac')}
                    meta={t('trainingStudio.mm3.lrEndFracMeta', 'default 0.005')}
                    info={t('trainingStudio.mm3.lrEndFracInfo',
                      'Where the learning-rate decay ends, as a fraction of the peak rate. 1 holds the '
                      + 'rate flat after warmup; 0 decays to nothing at the last step.')} />
                  <NumField label={t('trainingStudio.mm3.lyricsDropout', 'Lyric dropout')} value={form.lyricsDropout}
                    onChange={v => set('lyricsDropout', v)} step={0.05} min={0} max={1}
                    defaultValue={base.lyricsDropout} onReset={() => reset('lyricsDropout')}
                    meta={t('trainingStudio.mm3.lyricsDropoutMeta', 'default 0')}
                    info={t('trainingStudio.mm3.lyricsDropoutInfo',
                      'Share of steps trained with the lyrics removed and the caption kept, so the adapter '
                      + 'does not depend on lyrics being present. 0 trains on the lyrics every step.')} />
                </div>
                {/* The dataset-wide caption box lived here until 2026-09-09. One caption for
                    every track replaced the per-track .mm3.txt captions the renders use, and
                    adapters trained that way did not end songs (album B 0/6 vs 4/6). Removed. */}
                {/* ── Method row (2026-09-05) ───────────────────────────────
                    Same shape as TrainDitForm's DitMethod / TrainLmForm's
                    LmMethod: LoKr is its own type, DoRA/HiRA/LoHa/HRA are the
                    LoRA type with one flag set, PiSSA/rsLoRA/LoRA+ sit
                    underneath as checkboxes. None of the five new methods has
                    an MM3-specific measurement — see each button's hover
                    text — and --dora/--hira/--loha/--pissa/--hra are landing
                    in the mm3-lm-train engine parser concurrently with this
                    card, not before it: a run that picks one is refused by an
                    older ace-train, loudly, rather than silently training
                    plain LoRA. */}
                <div className="flex flex-col gap-1.5">
                  <ParamLabel
                    label={t('trainingStudio.mm3.adapterType', 'Adapter type')}
                    info={t('trainingStudio.mm3.adapterTypeHint',
                      'LoRA is the default. LoKr trains a Kronecker pair per slot for a smaller file '
                      + '(about 528 MB at factor 6 against a rank-128 LoRA\'s 1.4 GB) — NOT YET '
                      + 'VALIDATED BY EAR. DoRA/HiRA/LoHa/HRA are new here 2026-09-05 (see each '
                      + 'button\'s hover text); none has an MM3 measurement of its own yet.')}
                    className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                    onReset={method !== methodOf(base) ? resetMethod : undefined}
                  />
                  <div className="flex flex-wrap items-center gap-1.5">
                    {(['lokr', 'lora', 'dora', 'hira', 'loha', 'hra'] as Mm3Method[]).map(m => {
                      const active = method === m;
                      return (
                        <button
                          key={m}
                          type="button"
                          onClick={() => pickMethod(m)}
                          title={t(`trainingStudio.mm3.${MM3_METHOD_KEYS[m].info}`)}
                          className={`px-4 py-1.5 rounded-lg text-xs font-bold border transition-colors ${
                            active
                              ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                              : 'text-zinc-500 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
                          }`}
                        >
                          {t(`trainingStudio.mm3.${MM3_METHOD_KEYS[m].label}`)}
                        </button>
                      );
                    })}
                  </div>
                  {method !== 'lokr' && (
                    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2 mt-1">
                      <Toggle accent="amber" size="sm" checked={form.pissa && form.hotPizza} disabled={method !== 'lora'}
                        onChange={v => setEdits(prev => ({ ...prev, hotPizza: v, ...(v ? { pissa: true } : {}) }))}
                        defaultValue={base.pissa && base.hotPizza}
                        onReset={resetHotPizza}
                        label={t('trainingStudio.mm3.hotPizza', 'HOT-PiZZA')}
                        info={t('trainingStudio.mm3.hotPizzaInfo')} />
                      <Toggle accent="amber" size="sm" checked={form.pissa} disabled={method !== 'lora'}
                        onChange={v => setEdits(prev => ({ ...prev, pissa: v, ...(v ? {} : { hotPizza: false }) }))}
                        defaultValue={base.pissa}
                        onReset={resetPissa}
                        label={t('trainingStudio.mm3.pissa', 'PiSSA init')}
                        info={t('trainingStudio.mm3.pissaInfo')} />
                      <Toggle accent="amber" size="sm" checked={form.rslora} disabled={method === 'hra'}
                        onChange={v => set('rslora', v)}
                        defaultValue={base.rslora}
                        onReset={() => reset('rslora')}
                        label={t('trainingStudio.mm3.rslora', 'rsLoRA')}
                        info={t('trainingStudio.mm3.rsloraInfo')} />
                      <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                        <ParamLabel
                          label={t('trainingStudio.mm3.loraPlusRatio', 'LoRA+ ratio')}
                          info={t('trainingStudio.mm3.loraPlusRatioInfo')}
                          className="text-xs text-zinc-700 dark:text-zinc-300"
                          onReset={form.loraPlusRatio !== base.loraPlusRatio ? () => reset('loraPlusRatio') : undefined}
                        />
                        <input type="number" min={1} max={64} step={1} value={form.loraPlusRatio}
                          onChange={e => set('loraPlusRatio', Math.max(1, Number(e.target.value) || 1))}
                          className={`${INPUT} w-20`} />
                      </label>
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex flex-col gap-1">
                    <ParamLabel
                      label={t('trainingStudio.mm3.optimizer', 'Optimizer')}
                      info={t('trainingStudio.mm3.optimizerHint',
                        'Prodigy estimates its own step size, so the learning rate below becomes a '
                        + 'schedule multiplier only. On album B it converged to 8.19e-5 against the '
                        + '8e-5 tuned by hand. It costs two extra state buffers (about +2.7 GB at rank '
                        + '128) and CANNOT resume, so it is unavailable when mid-training previews are '
                        + 'on. AdamW matches the published SimpleTuner recipe. Muon is NOT recommended: '
                        + 'at the default LR scale of 64 it produced an adapter that rendered digital '
                        + 'silence, because that value was tuned on a different model.')}
                      className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                      onReset={form.optimizer !== base.optimizer ? () => reset('optimizer') : undefined}
                    />
                    <StyledSelect
                      accent="amber"
                      value={form.optimizer}
                      onChange={v => set('optimizer', v)}
                      options={[
                        { value: 'prodigy', label: 'Prodigy (sets its own LR)' },
                        { value: 'adamw', label: 'AdamW' },
                        { value: 'muon', label: 'Muon' },
                      ]}
                      className="w-full"
                    />
                  </label>
                  <Toggle accent="amber" className={`self-end pb-1${flashSupported ? '' : ' opacity-60'}`}
                    checked={attnEffective === 'flash'}
                    disabled={!flashSupported}
                    onChange={v => set('attnBackend', v ? 'flash' : 'exact')}
                    defaultValue={base.attnBackend === 'flash'}
                    onReset={() => reset('attnBackend')}
                    label={t('trainingStudio.mm3.attnBackend', 'Flash attention')}
                    info={!flashSupported
                      ? t('trainingStudio.mm3.attnBackendUnsupported',
                          'Unavailable on this build: the fused attention kernels exist for CUDA '
                          + 'only, and the trainer refuses to start on a {{backend}} engine rather '
                          + 'than run them on the CPU behind your back. Training runs in exact mode.',
                          { backend: status?.engineBackend ?? 'non-CUDA' })
                      : flashCalibrated
                        ? t('trainingStudio.mm3.attnBackendHelp',
                            'Fused attention kernels — see the VRAM estimate above for the measured saving.')
                        : t('trainingStudio.mm3.attnBackendHelpPending',
                            'Fused attention kernels, once mm3-lm-train gains --attn (landing '
                            + 'concurrently). VRAM saving not yet measured — the estimate above stands '
                            + 'in with exact mode\'s number until it is.')} />
                  <label className="flex flex-col gap-1">
                    <ParamLabel
                      label={t('trainingStudio.mm3.trainPrecision', 'Training precision')}
                      meta={t('trainingStudio.mm3.trainPrecisionMeta', 'default Fastest')}
                      info={t('trainingStudio.mm3.trainPrecisionInfo',
                        'How exactly the trainer does its arithmetic. Full keeps every matrix product in '
                        + 'FP32. Fast runs the frozen base weights\' products in BF16 on the tensor cores: '
                        + 'about 1.8x per step on an RTX 5090. Fastest also runs flash attention in BF16: '
                        + 'about 2.6x per step. Fast and Fastest add about 0.4 GB of peak VRAM on a quantized base, for a BF16 copy of one layer at a time; the estimate includes it. The faster choices '
                        + 'change the trained weights slightly; matched adapters showed no audible loss '
                        + 'in an ear test, so Fastest is the default. Its gradients are too coarse for '
                        + 'the trainer\'s finite-difference check, though they match Full within 0.15%. Fastest only '
                        + 'changes attention when Flash attention is on; otherwise it trains as Fast. '
                        + 'Needs a CUDA card; on cards older than the RTX 30 series, Fastest keeps the default '
                        + 'attention precision.')}
                      className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                      onReset={form.trainPrecision !== base.trainPrecision ? () => reset('trainPrecision') : undefined}
                    />
                    <StyledSelect
                      accent="amber"
                      value={form.trainPrecision}
                      onChange={v => set('trainPrecision', v)}
                      options={[
                        { value: 'full', label: t('trainingStudio.mm3.trainPrecisionFull', 'Full (FP32)') },
                        { value: 'fast', label: t('trainingStudio.mm3.trainPrecisionFast', 'Fast (BF16 matmuls)'),
                          hint: t('trainingStudio.mm3.trainPrecisionFastHint', 'About 1.8x per step') },
                        { value: 'fastest', label: t('trainingStudio.mm3.trainPrecisionFastest', 'Fastest (BF16 matmuls + attention, default)'),
                          hint: t('trainingStudio.mm3.trainPrecisionFastestHint', 'About 2.6x per step') },
                      ]}
                      className="w-full"
                    />
                  </label>
                  {form.adapterType === 'lokr' && (
                    <NumField label={t('trainingStudio.mm3.lokrFactor', 'LoKr factor')}
                      value={form.lokrFactor} onChange={v => set('lokrFactor', v)}
                      defaultValue={base.lokrFactor} onReset={() => reset('lokrFactor')}
                      info={t('trainingStudio.mm3.lokrFactorHint',
                        '6 gives ~528 MB and 264M parameters. Higher is smaller and less capable: '
                        + '8 -> 274 MB, 16 -> 109 MB.') as string} />
                  )}
                  {form.optimizer === 'muon' && (
                    <NumField label={t('trainingStudio.mm3.muonLrScale', 'Muon LR scale')}
                      value={form.muonLrScale} onChange={v => set('muonLrScale', v)}
                      defaultValue={base.muonLrScale} onReset={() => reset('muonLrScale')}
                      info={t('trainingStudio.mm3.muonLrScaleHint',
                        '64 is the best of the values measured so far, not a tuned optimum.') as string} />
                  )}
                </div>

                {/* ── Soft prompt (2026-09-05) ──────────────────────────────
                    A trainable token + a trainable K/V prefix, both new to
                    MM3 (the mm3-lm-train ENGINE parser already accepts
                    --artist-token/-k/-lr; --prefix-n is landing concurrently
                    with this card). Kept visually separate from "History
                    before crop" below, which is a DIFFERENT, FROZEN prefix
                    baked from real audio (train/mm3-lm-kvprefix.h) — the two
                    are not the same knob and must not be confused. */}
                <div className="rounded-lg border border-zinc-200 dark:border-white/10 p-3 flex flex-col gap-2">
                  <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                    {t('trainingStudio.mm3.softPromptGroup', 'Soft prompt (trainable, this run)')}
                  </span>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {t('trainingStudio.mm3.softPromptGroupHelp',
                      'A named token adds k learned vectors behind a placeholder in every caption; the '
                      + 'prefix adds n trainable key/value columns per layer. Both train WITH the LoRA '
                      + 'and ship in the same adapter file. Different mechanism from "History before '
                      + 'crop" below, which is FROZEN real-audio context, not a trained parameter.')}
                  </span>
                  <Toggle accent="amber" checked={form.artistTokenOn}
                    onChange={on => {
                      setEdits(prev => ({
                        ...prev, artistTokenOn: on,
                        artistToken: on && !form.artistToken.trim()
                          ? (form.trigger.trim() || 'artist') : form.artistToken,
                      }));
                    }}
                    defaultValue={base.artistTokenOn}
                    onReset={() => reset('artistTokenOn')}
                    label={t('trainingStudio.mm3.artistTokenOn', 'Train an artist token')}
                    info={t('trainingStudio.mm3.artistTokenOnInfo',
                      'Adds a named token whose learned vectors sit behind a placeholder in every '
                      + 'training caption, trained alongside the LoRA. Off: no soft-prompt token is '
                      + 'added — only the LoRA weights carry the identity.')} />
                  {form.artistTokenOn && (
                    <div className="grid grid-cols-3 gap-2">
                      <label className="flex flex-col gap-1">
                        <ParamLabel
                          label={t('trainingStudio.mm3.artistToken', 'Token name')}
                          info={t('trainingStudio.mm3.artistTokenInfo',
                            'The placeholder word the learned vectors attach to in captions. '
                            + 'Defaults to the trigger word if one is set.')}
                          className="text-[10px] text-zinc-500"
                          onReset={form.artistToken !== base.artistToken ? () => reset('artistToken') : undefined}
                        />
                        <input className={INPUT} value={form.artistToken}
                          onChange={e => set('artistToken', e.target.value.replace(/[^A-Za-z0-9_-]/g, ''))} />
                      </label>
                      <NumField label={t('trainingStudio.mm3.artistTokenK', 'Token vectors (k)')}
                        value={form.artistTokenK} onChange={v => set('artistTokenK', v)}
                        defaultValue={base.artistTokenK} onReset={() => reset('artistTokenK')}
                        meta={t('trainingStudio.mm3.artistTokenKMeta', 'default 32')}
                        info={t('trainingStudio.mm3.artistTokenKInfo',
                          'How many learned vectors the artist token adds behind its placeholder. More '
                          + 'vectors give the token more room to encode identity but add trainable '
                          + 'parameters; fewer is cheaper but may undercapture it.')} />
                      <NumField label={t('trainingStudio.mm3.artistTokenLr', 'Token LR')}
                        value={form.artistTokenLr} onChange={v => set('artistTokenLr', v)} step={0.0005}
                        defaultValue={base.artistTokenLr} onReset={() => reset('artistTokenLr')}
                        meta={t('trainingStudio.mm3.artistTokenLrMeta', 'default 0.005')}
                        info={t('trainingStudio.mm3.artistTokenLrInfo',
                          'The learning rate for the artist token and its vectors, separate from the '
                          + 'adapter\'s own learning rate above. Higher moves the token\'s identity in '
                          + 'faster but risks overshooting; lower is safer but slower to bake in.')} />
                    </div>
                  )}
                  <NumField label={t('trainingStudio.mm3.prefixN', 'Prefix columns (trainable)')}
                    value={form.regDatasetId ? 0 : form.prefixN}
                    onChange={v => set('prefixN', Math.max(0, Math.min(64, v)))}
                    disabled={!!form.regDatasetId}
                    defaultValue={base.prefixN} onReset={() => reset('prefixN')}
                    info={(form.regDatasetId
                      ? t('trainingStudio.mm3.prefixNVsReg',
                          'Unavailable with a regularisation dataset: the prior capture needs an inert '
                          + 'model, and a prefix is non-zero from initialisation. The engine refuses '
                          + 'the pair.')
                      : t('trainingStudio.mm3.prefixNInfo',
                          'n trainable key/value columns per layer that every position attends to. 0 = off. '
                          + 'Not the frozen history prefix below — this one trains.')) as string} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex flex-col gap-1">
                    <ParamLabel
                      label={t('trainingStudio.mm3.base', 'Base precision')}
                      info={chosen?.quality === 'poor'
                        ? t('trainingStudio.mm3.basePoor',
                            'This base is too lossy to train against: measured +14% on the first-step loss '
                            + 'and roughly double the gradient norm, i.e. the quantizer injects more error '
                            + 'than the adapter is being asked to learn. Prefer the smallest base marked '
                            + 'good or better that fits.')
                        : t('trainingStudio.mm3.baseHint',
                            'Every base trains the same way — the frozen weights are dequantized in-graph, '
                            + 'so only VRAM and fidelity change, and step time barely moves (within ~5% '
                            + 'across the whole range). Pick the smallest one that comfortably fits your '
                            + 'card, then spend what is left on LoRA rank.')}
                      className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                    />
                    <StyledSelect
                      accent="amber"
                      value={form.basePrecision}
                      onChange={v => set('basePrecision', v)}
                      options={(status?.bases ?? []).map(b => ({
                        value: b.id,
                        label: `${b.id} — ${(b.bytes / 1073741824).toFixed(1)} GB`
                          + `${b.quality === 'poor' ? ' (not recommended)' : ''}`
                          + `${b.id === status?.recommended?.base ? ' ✓' : ''}`,
                      }))}
                      className="w-full"
                    />
                    {peak && (
                      <span className={`text-[10px] leading-snug ${peak.tone}`}>{peak.text}</span>
                    )}
                  </label>
                  <NumField label={t('trainingStudio.mm3.holdout', 'Hold-out fraction')}
                    value={form.holdout} onChange={v => set('holdout', v)} step={0.05}
                    defaultValue={base.holdout} onReset={() => reset('holdout')}
                    info={t('trainingStudio.mm3.holdoutHint',
                      '0 disables evaluation — the training loss then cannot tell learning from '
                      + 'memorising. Ignored below 6 songs.') as string} />
                  <NumField label={t('trainingStudio.mm3.evalEvery', 'Evaluate every')}
                    value={form.evalEvery} onChange={v => set('evalEvery', v)} step={25}
                    defaultValue={base.evalEvery} onReset={() => reset('evalEvery')}
                    info={t('trainingStudio.mm3.evalEveryHint',
                      'Steps between held-out evaluations. 0 = off. This is also the finest '
                      + 'grain a held-out target can stop on.') as string} />
                </div>
                <label className="flex flex-col gap-1">
                  <ParamLabel
                    label={t('trainingStudio.mm3.cropMode', 'Crop mode')}
                    info={form.cropMode === 'random'
                      ? t('trainingStudio.mm3.cropModeRandom',
                          '`random` hands the model a prompt followed straight by mid-song audio with '
                          + 'no history in front of it, and supervises it — so it learns that a song may '
                          + 'begin at any position. It also leaves the ending to chance: EOS is only '
                          + 'supervised by a crop that reaches the track end, which on a 3-minute song '
                          + 'is under 3% of steps. Renders begin mid-flow and never resolve.')
                      : form.cropMode === 'beginning'
                        ? t('trainingStudio.mm3.cropModeIntros',
                            '`beginning` takes every track from frame 0, so every supervised position '
                            + 'has the song’s real history in front of it. What it can never teach is an '
                            + 'ending: EOS is only supervised by a crop that reaches the track end, and '
                            + 'it never does unless the whole song fits in the window.')
                        : t('trainingStudio.mm3.cropModeStructured',
                            '`structured` anchors most crops at frame 0, so every supervised position '
                            + 'carries the song’s real history exactly as it will at generation time, '
                            + 'and pins the rest flush to the track end — the only place EOS is '
                            + 'supervised. This is the default and the two others are each broken at '
                            + 'one end.')}
                    className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                    onReset={form.cropMode !== base.cropMode ? () => reset('cropMode') : undefined}
                  />
                  <StyledSelect
                    accent="amber"
                    value={form.cropMode}
                    onChange={v => set('cropMode', v)}
                    options={[
                      { value: 'structured', label: 'structured' },
                      { value: 'random', label: 'random' },
                      { value: 'beginning', label: 'beginning' },
                    ]}
                    className="w-full"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <ParamLabel
                    label={t('trainingStudio.mm3.cropAnchor', 'Crop position')}
                    info={t('trainingStudio.mm3.cropAnchorHint',
                      'Under "zero" every crop was presented to the model as if it were the opening of '
                      + 'the song, whatever part of the track it came from - while generation always '
                      + 'starts at frame 0. That mismatch teaches the model that a song can begin '
                      + 'anywhere, and shows up as sound arriving instantly at 0:00 and as tempo '
                      + 'drifting mid-track. "song" labels each crop with where it actually is. Runs '
                      + 'trained under the two are not comparable.')}
                    className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                    onReset={form.cropAnchor !== base.cropAnchor ? () => reset('cropAnchor') : undefined}
                  />
                  <StyledSelect
                    accent="amber"
                    value={form.cropAnchor}
                    onChange={v => set('cropAnchor', v)}
                    options={[
                      { value: 'song', label: 'song (true position)' },
                      { value: 'zero', label: 'zero (legacy)' },
                    ]}
                    className="w-full"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <ParamLabel
                    label={t('trainingStudio.mm3.prefixFrames', 'History before crop (frames)')}
                    info={form.cropAnchor === 'zero'
                      ? t('trainingStudio.mm3.prefixNeedsSong',
                          'Needs crop position "song". History placed at positions the crop then '
                          + 're-uses is not a history.')
                      : form.prefixFrames > 0
                        ? t('trainingStudio.mm3.prefixOn',
                            '{{sec}} s of the track before each crop is run through the model first, '
                            + 'with no gradient, so the crop attends back over real history instead of '
                            + 'starting from nothing. Adds about {{gb}} GB and roughly 60% to step '
                            + 'time. UNHEARD: no adapter trained this way has been auditioned.',
                            { sec: (form.prefixFrames / 25).toFixed(1),
                              gb: (estimateMm3PrefixMb(form.prefixFrames, form.maxFrames,
                                                       status!.vramModel) / 1024).toFixed(1) })
                        : t('trainingStudio.mm3.prefixOff2',
                            'Off. Each crop is presented at its true position in the track with an '
                            + 'EMPTY context, so the middle third of the planner learns to produce '
                            + 'late-song behaviour from a 30-second view. That is the band that renders '
                            + 'better with the Middle Third dial at 0. History is LINEAR in VRAM where '
                            + 'crop length is quadratic, so it buys context far more cheaply than a '
                            + 'longer crop does.')}
                    className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                    onReset={form.cropAnchor !== 'zero' && form.prefixFrames !== base.prefixFrames
                      ? () => reset('prefixFrames') : undefined}
                  />
                  <div className="flex gap-2">
                    <input type="number" className={INPUT} step={50} min={0}
                      value={form.prefixFrames}
                      disabled={form.cropAnchor === 'zero'}
                      onChange={e => set('prefixFrames', Math.max(0, Number(e.target.value)))} />
                    <button type="button" className={BTN_SM}
                      disabled={form.cropAnchor === 'zero'}
                      onClick={() => set('prefixFrames',
                        form.prefixFrames > 0 ? 0 : form.maxFrames)}>
                      {/* An ACTION, not a state. It read "Off" while the field
                          still said 4096, which is how at least one user
                          reported having switched the prefix off and trained
                          with it anyway (#142). */}
                      {form.prefixFrames > 0
                        ? t('trainingStudio.mm3.prefixTurnOff', 'Turn off')
                        : t('trainingStudio.mm3.prefixMatch', 'Match crop')}
                    </button>
                  </div>
                </label>
              </div>
            )}

            {!!batchDraft?.length && <div className="mt-4 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
              <p className="text-xs font-semibold text-amber-700 dark:text-amber-300">
                {t('trainingStudio.mm3.batchSelected', 'Recipe applies to {{count}} datasets, in order.', { count: batchDraft.length })}
              </p>
              <p className="mt-1 text-[11px] text-zinc-600 dark:text-zinc-400 break-words">
                {batchDraft.map(id => datasets.find(d => d.id === id)?.name ?? id).join(' · ')}
              </p>
              <Toggle accent="amber" className="mt-2" checked={batchClearCache} disabled={busy} onChange={setBatchClearCache}
                label={t('trainingStudio.yue2.aitkBatch.clearCache', 'Clear out all cached data')}
                info={t('trainingStudio.mm3.batchClearCacheHint', "Deletes each dataset's MM3 caches (RVQ codes) before it starts, so they are rebuilt from the audio. Source audio, captions and trained adapters are not touched. Each dataset then takes as long as a first-time code export.")} />
            </div>}
            {!!batchDraft?.length && !trainingWorker && workers.length > 0 && <div className="mt-4 flex items-center gap-3 flex-wrap">
              <ParamLabel
                label={t('trainingStudio.workers.runOn', 'Run on')}
                className="text-[11px] font-semibold text-zinc-700 dark:text-zinc-300"
                info={t('trainingStudio.workers.runOnInfoMm3', 'A training worker trains the batch on its own GPU. Each dataset\'s audio, .mm3.txt captions and dataset.json are sent to the worker (only files it does not already have), and it joins the worker\'s batch: codes, then training. Switch "Train on" at the top to that worker to follow the run. Each checkpoint and preview is copied back here as it lands, so the run then opens here like one trained on this PC. The recipe is sized for this PC\'s GPU; check it fits the worker\'s.')}
              />
              <StyledSelect
                accent="amber"
                value={runOn}
                disabled={busy}
                onChange={value => setRunOn(String(value))}
                options={[{ value: '', label: t('trainingStudio.workers.thisPc', 'This PC') }, ...workers.map(w => ({ value: w.name, label: w.online ? w.name : `${w.name} (offline)`, disabled: !w.online }))]}
                className="w-48"
              />
            </div>}

            <button
              onClick={() => void startTrain()}
              disabled={busy || (jobRunning && !runOn)}
              className="mt-4 px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 transition-colors flex items-center gap-2"
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              {batchDraft?.length
                ? t('trainingStudio.mm3.startBatch', 'Start batch ({{count}})', { count: batchDraft.length })
                : busy ? t('trainingStudio.mm3.starting', 'Starting…')
                  : mine && jobRunning ? (activeJob?.phase === 'preparing'
                    ? t('trainingStudio.mm3.preparing', 'Preparing…') : t('trainingStudio.mm3.running', 'Training is running'))
                    : t('trainingStudio.mm3.start', 'Start training')}
            </button>
            {!!batchDraft?.length && <button type="button" disabled={busy} onClick={() => setBatchDraft(null)}
              className="mt-2 text-xs text-zinc-500 hover:underline">{t('trainingStudio.yue2.aitkBatch.discard', 'Discard batch')}</button>}
          </>
        )}
      </TrainingRunPanel>

      {/* ── Live run: the shared job machinery, unchanged ── */}
      {mine && activeJob && (
        <div className={CARD}>
          <JobProgress hideEta={jobKind === 'mm3-train-lm'} />
          {targetEtaText ? <div className="mt-1 text-[11px] text-zinc-500">{targetEtaText}</div>
            : mm3Eta && <div className="mt-1 text-[11px] text-zinc-500"><TrainingEtaLine eta={mm3Eta} /></div>}
          {jobKind === 'mm3-train-lm' && mm3Live && (
            // The run-stats row. MM3 has no epochs, so none of the ACE tiles
            // apply — these are the numbers that actually say what the run is
            // doing, and STEP TIME is the one that exposes a VRAM spill (about
            // 4 s when it fits, ~10x that when it pages to host memory).
            <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2">
              {[
                { k: 'loss', v: mm3Live.loss ? mm3Live.loss.toFixed(4) : '—' },
                { k: 'gradNorm', v: mm3Live.gradNorm ? mm3Live.gradNorm.toFixed(3) : '—' },
                // The SCHEDULE value is not what Muon applies: its update is
                // normalised, so the effective rate is lr x muon-lr-scale.
                // Showing 3.4e-5 while the optimizer used 2.2e-3 was a quietly
                // misleading tile.
                {
                  k: 'lr',
                  v: mm3Live.lr
                    ? (lrScale > 1
                      ? `${(mm3Live.lr * lrScale).toExponential(2)}`
                      : mm3Live.lr.toExponential(2))
                    : '—',
                },
                { k: 'stepTime', v: mm3Live.stepMs ? `${(mm3Live.stepMs / 1000).toFixed(1)}s` : '—' },
                {
                  k: 'vram',
                  v: mm3Live.totalMb
                    ? `${Math.round(mm3Live.usedMb / 1024)}/${Math.round(mm3Live.totalMb / 1024)} GB`
                    : '—',
                  warn: mm3Live.totalMb > 0 && mm3Live.usedMb > mm3Live.totalMb - 512,
                },
              ].map(tile => (
                <div key={tile.k}
                  className="rounded-lg border border-zinc-200 dark:border-white/5 px-2.5 py-1.5">
                  <div className="text-[10px] uppercase tracking-wider text-zinc-500">
                    {t(`trainingStudio.mm3.stat.${tile.k}`, tile.k)}
                  </div>
                  <div className={`text-sm font-semibold tabular-nums ${
                    (tile as { warn?: boolean }).warn ? 'text-amber-500' : 'text-zinc-800 dark:text-zinc-200'
                  }`}>{tile.v}</div>
                </div>
              ))}
            </div>
          )}
          {jobKind === 'mm3-train-lm' && liveRun && <Mm3CheckpointLadder datasetId={datasetId} run={liveRun}
            reloadKey={`${trainPreviews.length}:${trainMilestones.length}:${jobStatus}`} />}
          {jobKind === 'mm3-train-lm' && trainStepSeries.length > 1 && (
            <div className="mt-3">
              {/* Same chart as YuE2's: loss, held-out loss, grad norm and
                  learning rate on a step axis, checkpoint markers from the
                  milestones. The target line is drawn only when the run is
                  stopping on one. */}
              <Yue2SplitChart mode="mm3" points={trainStepSeries.map(s => ({ ...s, ep: s.step }))}
                milestones={trainMilestones}
                evals={trainEvalSeries.map(e => ({ ep: e.step, loss: e.loss }))}
                target={trainTargetLoss} vramMb={mm3Live?.usedMb} totalSteps={activeJob.total}
                options={stopLossLive ? { stopMode: 'loss', targetLoss: trainTargetLoss, targetLossMetric: form?.targetLossMetric, stopReading } : undefined} />
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] tabular-nums text-zinc-500">
                {descent !== null && stepsPerReading === 1 && <span>{t('trainingStudio.mm3.lossRate', 'loss rate')} {descent.toFixed(5)}/step</span>}
                {activeJob.startedAt != null && <span>{t('trainingStudio.mm3.elapsed', 'elapsed')} {formatDurationMs(Date.now() - activeJob.startedAt)}</span>}
              </div>
            </div>
          )}
          {jobLog.length > 0 && <details className="mt-2" open={showLog} onToggle={event => setShowLog(event.currentTarget.open)}>
            <summary className="cursor-pointer text-[11px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.method.showLogs', 'Show training log (last 100 lines)')}</summary>
            <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-zinc-950 p-2 text-[10px] leading-4 text-zinc-300 whitespace-pre-wrap">{jobLog.slice(-100).map(l => `${new Date(l.ts).toLocaleTimeString()} ${l.level}: ${l.message}`).join('\n')}</pre>
          </details>}
        </div>
      )}
      <Mm3RunsPanel key={datasetId} datasetId={datasetId} renderOpts={{ seconds: form?.previewSeconds, scaleMlp: form?.previewScaleMlp }} />
    </div>
  );
};

export default Mm3TrainCard;
