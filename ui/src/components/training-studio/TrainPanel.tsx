// TrainPanel.tsx — Training Studio phase 3
//
// Tensor cache → 5 Hz FSQ codes → a Qwen3 LM LoRA in adapters/lm/<name>-<size>.
// The whole run happens in the standalone ace-train binary, which needs the GPU
// to itself: the server stops ace-server for the duration and restarts it
// afterwards, so this panel is explicit about the engine being paused.
//
// Two sibling cards: the planner LoRA (LM) and the sound LoRA (DiT). Both run in
// the same standalone ace-train binary, both need the GPU to themselves, and only
// one job at a time can own a dataset — so the two cards share every gate above
// and differ only in which slice of the store they read.

import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle, Check, Copy, Cpu, Database, FileCode2, Flag, Headphones, Layers, Loader2, PauseCircle, Target, Waves, XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { TrainDitOptions, TrainLmOptions } from '../../services/trainingApi';
import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { formatDurationMs, totalEpochMs } from '../../utils/trainingEta';
import { AuditionCard, type MilestoneAuditionRequest } from './AuditionCard';
import { JobProgress } from './JobProgress';
import { TrainingChart } from './TrainingChart';
import { TrainingRunStats } from './TrainingRunStats';
import { Mm3TrainCard } from './Mm3TrainCard';
import { TRAIN_DIT_LOKR_DEFAULTS, TrainDitForm, type TrainDitFormState } from './TrainDitForm';
import { TRAIN_LM_DEFAULTS, TrainLmForm, type TrainLmFormState } from './TrainLmForm';
import { useTrainingStream } from './useTrainingStream';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const STATUS_RELOAD_DEBOUNCE_MS = 400;

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

export const TrainPanel: React.FC = () => {
  const { t } = useTranslation();
  // WHICH TRAINING, not which flag. ACE trains a DiT and a planner LM from a
  // preprocessed tensor cache; MiniMax-Music3 trains an LM from RVQ codes and
  // has no DiT path and no preprocess variant. Those are two different
  // features that happen to share a phase, not one feature with a capability
  // toggle — so this branches on the active backend, in this one place, rather
  // than inventing a flag that would have to mean "which of two unrelated UIs".
  const backendId = useBackendStore(s => s.activeBackendId);
  const mm3Mode = backendId === 'minimax-m3';
  const capabilities = useTrainingStore(s => s.capabilities);
  const datasets = useTrainingStore(s => s.datasets);
  const detail = useTrainingStore(s => s.detail);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const activeJob = useTrainingStore(s => s.activeJob);
  const error = useTrainingStore(s => s.error);
  const trainLmStatus = useTrainingStore(s => s.trainLmStatus);
  const trainLmLoading = useTrainingStore(s => s.trainLmLoading);
  const trainLmEpochs = useTrainingStore(s => s.trainLmEpochs);
  const trainLmLast = useTrainingStore(s => s.trainLmLast);
  const trainLmSkippedLong = useTrainingStore(s => s.trainLmSkippedLong);
  const trainLmVram = useTrainingStore(s => s.trainLmVram);
  const trainDitStatus = useTrainingStore(s => s.trainDitStatus);
  const trainDitLoading = useTrainingStore(s => s.trainDitLoading);
  const trainDitEpochs = useTrainingStore(s => s.trainDitEpochs);
  const trainDitLast = useTrainingStore(s => s.trainDitLast);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);
  const trainMaxEpochs = useTrainingStore(s => s.trainMaxEpochs);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const loadCapabilities = useTrainingStore(s => s.loadCapabilities);
  const loadTrainLmStatus = useTrainingStore(s => s.loadTrainLmStatus);
  const startTrainLm = useTrainingStore(s => s.startTrainLm);
  const loadTrainDitStatus = useTrainingStore(s => s.loadTrainDitStatus);
  const startTrainDit = useTrainingStore(s => s.startTrainDit);

  const [form, setForm] = useState<TrainLmFormState>(TRAIN_LM_DEFAULTS);
  // K1: LoKR is the form's initial state — Rob's validated default preference.
  const [ditForm, setDitForm] = useState<TrainDitFormState>(TRAIN_DIT_LOKR_DEFAULTS);
  const [starting, setStarting] = useState(false);
  const [ditStarting, setDitStarting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [ditCopied, setDitCopied] = useState(false);
  // True from the instant the status debounce is armed until that read settles.
  // `trainLmLoading` only turns true INSIDE loadTrainLmStatus, i.e. 400 ms after
  // mount, and until then both it and `trainLmStatus` are falsy — the spinner and
  // the needs-preprocess gate below are both skipped and the fully enabled form
  // renders for that window. Seeded true so the hole never exists.
  const [statusPending, setStatusPending] = useState(true);
  // A milestone badge cannot start an audition on its own — the prompt lives in
  // the AuditionCard. Clicking one hands the milestone dir down; the nonce lets
  // a repeat click on the SAME milestone re-trigger.
  const [milestoneRequest, setMilestoneRequest] = useState<MilestoneAuditionRequest | null>(null);
  const auditionMilestone = (path: string, loss: number, epoch: number) => {
    setMilestoneRequest(prev => ({
      path,
      label: `loss ${loss.toFixed(1)} · #${epoch}`,
      nonce: (prev?.nonce ?? 0) + 1,
    }));
  };
  // Same hole on the DiT side, and it is worse there: with trainDitStatus null the
  // card's `variantKey === ''` gate is falsy, so the form rendered fully enabled
  // with an EMPTY base-model chip, and handleStartDit then omits variantKey and
  // lets the server silently pick newestVariantKey. Reachable via the 400 ms
  // debounce, via openDataset() nulling the status on a dataset switch, and
  // durably whenever GET /train-dit fails (loadTrainDitStatus swallows the error).
  const [ditStatusPending, setDitStatusPending] = useState(true);

  const tl = capabilities?.trainLm;
  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobId = activeJob?.id;
  const jobActive = jobStatus === 'queued' || jobStatus === 'running';
  const trainJobActive = jobKind === 'train-lm' && jobActive;
  const ditJobActive = jobKind === 'train-dit' && jobActive;

  // The dataset phase mounts the stream inside DatasetDetail, which is not
  // rendered here — without this the progress bar and loss curve never move.
  useTrainingStream(jobActive && jobId ? jobId : null);

  // The adapter defaults to the dataset's own slug (L12), re-seeded whenever a
  // different dataset is opened.
  const slug = detail?.slug ?? '';
  useEffect(() => {
    if (slug) setForm(f => ({ ...f, adapterName: slug }));
    if (slug) setDitForm(f => ({ ...f, adapterName: slug }));
  }, [slug]);

  // Adapter name and size both change where the run would write, so the status
  // card follows them. Debounced — the name is a text input.
  const { lmSize, adapterName } = form;
  // Read by the terminal-job effect below without joining its dep array.
  const statusQueryRef = useRef({ lmSize, adapterName });
  statusQueryRef.current = { lmSize, adapterName };
  const statusQuery = () => {
    const { lmSize: sz, adapterName: nm } = statusQueryRef.current;
    return { lmSize: sz, ...(nm.trim() ? { adapterName: nm.trim() } : {}) };
  };

  useEffect(() => {
    if (!selectedDatasetId) return;
    setStatusPending(true);
    const handle = window.setTimeout(() => {
      void loadTrainLmStatus({
        lmSize,
        ...(adapterName.trim() ? { adapterName: adapterName.trim() } : {}),
      }).finally(() => setStatusPending(false));
    }, STATUS_RELOAD_DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [selectedDatasetId, lmSize, adapterName, loadTrainLmStatus]);

  // A finished job wrote the adapter dir and un-suspended the engine.
  //
  // lmSize/adapterName are deliberately NOT in the dep array (the proven sibling
  // PreprocessPanel does the same): `activeJob` is never cleared when a job ends,
  // so after any run jobKind/jobStatus stay terminal and every keystroke in the
  // adapter-name box would re-fire this body — an undebounced status GET plus a
  // /capabilities call, and /capabilities does a live /props round-trip to
  // ace-server. They are read through a ref instead.
  useEffect(() => {
    if (jobKind !== 'train-lm' || !jobStatus) return;
    if (jobStatus === 'queued' || jobStatus === 'running') return;
    void loadTrainLmStatus(statusQuery());
    void loadCapabilities();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, jobKind, jobStatus, loadTrainLmStatus, loadCapabilities]);

  // ── The same two effects for the DiT adapter ──────────────────────────────
  // Its status is also what supplies the read-only base-model chip, so it loads
  // even when the DiT card is never expanded.
  const ditAdapterName = ditForm.adapterName;
  const ditQueryRef = useRef(ditAdapterName);
  ditQueryRef.current = ditAdapterName;
  const ditStatusQuery = () => {
    const nm = ditQueryRef.current.trim();
    return nm ? { adapterName: nm } : {};
  };

  useEffect(() => {
    if (!selectedDatasetId) return;
    setDitStatusPending(true);
    const handle = window.setTimeout(() => {
      void loadTrainDitStatus(ditAdapterName.trim() ? { adapterName: ditAdapterName.trim() } : {})
        .finally(() => setDitStatusPending(false));
    }, STATUS_RELOAD_DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [selectedDatasetId, ditAdapterName, loadTrainDitStatus]);

  useEffect(() => {
    if (jobKind !== 'train-dit' || !jobStatus) return;
    if (jobStatus === 'queued' || jobStatus === 'running') return;
    // ditStatusQuery reads the adapter name through a ref, deliberately: it must
    // not join this dep array. `activeJob` is never cleared when a job ends, so
    // after any run jobKind/jobStatus stay terminal and every keystroke in the
    // name box would re-fire an undebounced status GET plus a /capabilities call
    // — and /capabilities does a live /props round-trip to ace-server.
    void loadTrainDitStatus(ditStatusQuery());
    void loadCapabilities();
  }, [jobId, jobKind, jobStatus, loadTrainDitStatus, loadCapabilities]);

  const patchForm = (patch: Partial<TrainLmFormState>) => setForm(f => ({ ...f, ...patch }));
  const patchDitForm = (patch: Partial<TrainDitFormState>) => setDitForm(f => ({ ...f, ...patch }));

  const handleStart = async () => {
    const opts: TrainLmOptions = {
      lmSize: form.lmSize,
      // §2.7: OMITTED means "let the server pick the BF16 base for this size".
      // Sending '' would be forwarded as `--lm ''` and ace-train exits 2.
      ...(form.lmModel ? { lmModel: form.lmModel } : {}),
      ...(trainLmStatus?.variantKey ? { variantKey: trainLmStatus.variantKey } : {}),
      adapterName: form.adapterName.trim(),
      targetLoss: form.targetLoss,
      epochs: form.epochs,
      rank: form.rank,
      alpha: form.alpha,
      learningRate: form.learningRate,
      gradAccum: form.gradAccum,
      gradClip: form.gradClip,
      warmupRatio: form.warmupRatio,
      weightDecay: form.weightDecay,
      maxLen: form.maxLen,
      seed: form.seed,
      lossOnCot: form.lossOnCot,
      order: form.order,
      milestoneStep: form.milestoneStep,
      milestoneKeep: form.milestoneKeep,
      stages: form.stages,
      overwrite: form.overwrite,
      stopEngine: form.stopEngine,
      // Resume + post-training calibration (2026-08-10). 'latest' = the server
      // resolves the newest non-calibrated run of this adapter name.
      //
      // ALWAYS SENT, both ways. Since 2026-08-12 an OMITTED initAdapter means
      // 'latest' on the LM route (so the batch pipeline's empty body resumes),
      // which makes '' the only way to say "from scratch" — unticking the box
      // would otherwise be a no-op.
      initAdapter: form.resumeFromLatest ? 'latest' : '',
      calibrate: form.calibrate,
      calibrateRepoint: form.calibrateRepoint,
      // Speed levers. Sent only when the user moved them off the shipped
      // default, so a normal start posts the same body it always did — and an
      // engine without the flags never sees them.
      ...(form.weights !== 'f32-window' ? { weights: form.weights } : {}),
      ...(form.batch !== 1 ? { batch: form.batch } : {}),
      // ALWAYS sent, unlike the two above: the server default ('mm') is not the
      // CLI default ('outprod'), so omitting it would hide which formulation ran
      // and make the drawer's 'outprod' option indistinguishable from "unset".
      bwd: form.bwd,
      adapterType: form.adapterType,
      optimizer: form.optimizer,
      ...(form.optimizer === 'muon' ? { muonLrScale: form.muonLrScale } : {}),
      ...(form.adapterType === 'lokr'
        ? { lokrDim: form.lokrDim, lokrAlpha: form.lokrAlpha, lokrFactor: form.lokrFactor }
        : {}),
    };
    setStarting(true);
    try { await startTrainLm(opts); } finally { setStarting(false); }
  };

  const handleStartDit = async () => {
    const opts: TrainDitOptions = {
      // The base is NOT sent — the server resolves --dit from the variant's
      // preprocess_meta.json, because the cached latents are that model's output.
      ...(trainDitStatus?.variantKey ? { variantKey: trainDitStatus.variantKey } : {}),
      adapterName: ditForm.adapterName.trim(),
      adapterType: ditForm.adapterType,
      // §2.1: lora trains via rank/alpha, lokr via the four lokr* fields — the
      // server ignores whichever side doesn't match adapterType, but sending
      // only the relevant one keeps the request body honest about what ran.
      ...(ditForm.adapterType === 'lokr'
        ? {
            lokrDim: ditForm.lokrDim,
            lokrAlpha: ditForm.lokrAlpha,
            lokrFactor: ditForm.lokrFactor,
            lokrDecomposeBoth: ditForm.lokrDecomposeBoth,
          }
        : { rank: ditForm.rank, alpha: ditForm.alpha }),
      targetMlp: ditForm.targetMlp,
      layers: ditForm.layers,
      crop: ditForm.crop,
      cropMin: ditForm.cropMin,
      cropMax: ditForm.cropMax,
      targetLoss: ditForm.targetLoss,
      epochs: ditForm.epochs,
      learningRate: ditForm.learningRate,
      gradAccum: ditForm.gradAccum,
      gradClip: ditForm.gradClip,
      warmupRatio: ditForm.warmupRatio,
      weightDecay: ditForm.weightDecay,
      lossWeighting: ditForm.lossWeighting,
      snrGamma: ditForm.snrGamma,
      tBias: ditForm.tBias,
      channelBalance: ditForm.channelBalance,
      timestepMu: ditForm.timestepMu,
      timestepSigma: ditForm.timestepSigma,
      tMin: ditForm.tMin,
      tMax: ditForm.tMax,
      cfgRatio: ditForm.cfgRatio,
      genreRatio: ditForm.genreRatio,
      seed: ditForm.seed,
      order: ditForm.order,
      // Resume + post-training calibration, same shape as train-lm.
      //
      // ALWAYS SENT, both ways. Since 2026-08-13 an OMITTED initAdapter means
      // 'latest' on the DiT route too (so the batch pipeline's empty body
      // resumes), which makes '' the only way to say "from scratch" —
      // unticking the box would otherwise be a no-op.
      initAdapter: ditForm.resumeFromLatest ? 'latest' : '',
      calibrate: ditForm.calibrate,
      calibrateRepoint: ditForm.calibrateRepoint,
      milestoneStep: ditForm.milestoneStep,
      milestoneKeep: ditForm.milestoneKeep,
      vramReserveMb: ditForm.vramReserveMb,
      mirror: ditForm.mirror,
      bwd: ditForm.bwd,
      attnBackend: ditForm.attnBackend,
      // Muon knobs go over the wire only when Muon is selected; the route
      // defaults them anyway, and this keeps an AdamW request byte-identical
      // to what it was before the optimizer field existed.
      optimizer: ditForm.optimizer,
      ...(ditForm.optimizer === 'muon'
        ? { muonLrScale: ditForm.muonLrScale, muonNsSteps: ditForm.muonNsSteps }
        : {}),
      batch: ditForm.batch,
      ckptSegments: ditForm.ckptSegments,
      stages: ditForm.stages,
      overwrite: ditForm.overwrite,
      stopEngine: ditForm.stopEngine,
    };
    setDitStarting(true);
    try { await startTrainDit(opts); } finally { setDitStarting(false); }
  };

  const copyAdapterDir = () => {
    const dir = trainLmStatus?.adapterDir;
    if (!dir) return;
    void navigator.clipboard?.writeText(dir).then(
      () => { setCopied(true); window.setTimeout(() => setCopied(false), 1500); },
      () => { /* clipboard blocked — the path is selectable on screen anyway */ },
    );
  };

  const copyDitAdapterDir = () => {
    const dir = trainDitStatus?.adapterDir;
    if (!dir) return;
    void navigator.clipboard?.writeText(dir).then(
      () => { setDitCopied(true); window.setTimeout(() => setDitCopied(false), 1500); },
      () => { /* clipboard blocked — the path is selectable on screen anyway */ },
    );
  };

  const header = (
    <div className="flex items-start gap-3">
      <Cpu size={18} className="text-amber-500 mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.title')}</h2>
        <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.train.subtitle')}</p>
      </div>
    </div>
  );

  // ── Gate 1: no dataset selected ─────────────────────────────────────────
  if (!selectedDatasetId) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className={`${CARD} flex flex-col gap-3`}>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">{t('trainingStudio.train.pickDataset')}</p>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-2">
            {datasets.map(ds => (
              <button
                key={ds.id}
                onClick={() => void openDataset(ds.id)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg border border-zinc-200 dark:border-white/10 hover:border-amber-500/40 text-left transition-colors"
              >
                <Database size={14} className="text-amber-500 flex-shrink-0" />
                <span className="flex-1 min-w-0">
                  <span className="block text-sm font-semibold text-zinc-900 dark:text-white truncate">{ds.name}</span>
                  <span className="block text-[11px] text-zinc-500 truncate">
                    {ds.builtAt
                      ? t('trainingStudio.list.samples', { count: ds.sampleCount })
                      : t('trainingStudio.list.notBuilt')}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!detail || !capabilities) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  // ── Gate 2: no ace-train binary ─────────────────────────────────────────
  if (!tl?.available) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-5 flex items-start gap-2 text-sm text-red-500 dark:text-red-400">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          {t('trainingStudio.train.noBinary')}
        </div>
      </div>
    );
  }

  // ── Gate 3: not built ───────────────────────────────────────────────────
  if (!detail.builtAt || !detail.datasetJsonPath) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-5 flex flex-col items-start gap-3">
          <div className="flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.train.needsBuild')}
          </div>
          <button
            onClick={() => setPhase('dataset')}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
          >
            {t('trainingStudio.preprocess.goToDataset')}
          </button>
        </div>
      </div>
    );
  }

  // ── MiniMax-Music3 branch ───────────────────────────────────────────────
  // Placed AFTER the built gate (MM3 reads dataset.json too) and BEFORE the
  // preprocess-variant gate below, which is an ACE concept: MM3 trains from
  // RVQ codes and has no tensor cache, so falling through would block it on a
  // prerequisite it does not have.
  if (mm3Mode) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <Mm3TrainCard datasetId={detail.id} trigger={detail.customTag || ''} />
      </div>
    );
  }

  // Status still in flight on first open — deciding gate 4 needs it. Covers the
  // debounce window too (statusPending), not just the request itself.
  if (!trainLmStatus && (trainLmLoading || statusPending)) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  // ── Gate 4: no preprocessed tensors ─────────────────────────────────────
  // Only when the status actually loaded — a failed read must not masquerade
  // as "you never preprocessed".
  if (trainLmStatus && trainLmStatus.variantKey === '') {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-5 flex flex-col items-start gap-3">
          <div className="flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.train.needsPreprocess')}
          </div>
          <button
            onClick={() => setPhase('preprocess')}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
          >
            {t('trainingStudio.train.goToPreprocess')}
          </button>
        </div>
      </div>
    );
  }

  // ── Gate 5: another kind of job owns this dataset ───────────────────────
  // train-dit is NOT "another kind" — it is the second card on this very panel.
  // Neither is 'audition': the AuditionCard lives here too, and bailing out
  // here would unmount the card (and its players) for the whole run.
  if (jobActive && jobKind !== 'train-lm' && jobKind !== 'train-dit' && jobKind !== 'audition') {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
          <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
          {t('trainingStudio.train.otherJobRunning')}
        </div>
        <JobProgress />
      </div>
    );
  }

  const enginePaused = form.stopEngine || !!capabilities.preprocess?.engineSuspended;
  const ditEnginePaused = ditForm.stopEngine || !!capabilities.preprocess?.engineSuspended;
  const status = trainLmStatus;
  const ditStatus = trainDitStatus;
  const ditStatusLoading = trainDitLoading || ditStatusPending;
  // NOTE: trainLmLast.etaMs / trainDitLast.etaMs (the engine's own
  // mean-epoch-ms x epochs-to-CAP) are deliberately NOT rendered any more. They
  // sat under the chart labelled "ETA" while JobProgress showed a loss-curve
  // estimate labelled the same thing, and the two answered different questions.
  // TrainingRunStats above the chart is now the single source (§Task 1b).

  return (
    <div className="flex flex-col gap-4">
      {header}

      <div className="flex items-center gap-2 text-[11px] text-zinc-500">
        <Database size={12} className="text-amber-500 flex-shrink-0" />
        <span className="font-semibold text-zinc-700 dark:text-zinc-300 truncate">{detail.name}</span>
        {status?.tensorsDir && (
          <span className="font-mono truncate" title={status.tensorsDir}>{status.tensorsDir}</span>
        )}
      </div>

      {/* Codes freshness. The server computes these at real cost (every line of
          lm_codes.jsonl plus one statSync per row); without surfacing them the
          user has no signal that the extract stage is about to re-encode the
          whole dataset — or that they should tick "Re-extract codes". */}
      {status?.variantKey && (
        <div className="flex items-center gap-2 flex-wrap text-[11px] text-zinc-500 -mt-2">
          <FileCode2 size={12} className="text-amber-500 flex-shrink-0" />
          <span className="tabular-nums">
            {status.codesExists
              ? t('trainingStudio.train.codes', { count: status.codesCount })
              : t('trainingStudio.train.codesNone')}
          </span>
          {status.codesStale > 0 && (
            <span className="tabular-nums px-1.5 py-0.5 rounded-full border text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20">
              {t('trainingStudio.train.codesStale', { count: status.codesStale })}
            </span>
          )}
          {status.codesMissing > 0 && (
            <span className="tabular-nums px-1.5 py-0.5 rounded-full border text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20">
              {t('trainingStudio.train.codesMissing', { count: status.codesMissing })}
            </span>
          )}
        </div>
      )}

      {error && (
        <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{error}</div>
      )}

      {/* ── LM card ──────────────────────────────────────────────────────── */}
      <div className={`${CARD} flex flex-col gap-4`}>
        <div className="flex items-start gap-2">
          <Cpu size={15} className="text-amber-500 mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.lmTitle')}</h3>
            <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.train.lmSubtitle')}</p>
          </div>
        </div>

        {trainJobActive && enginePaused && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
            <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.preprocess.enginePaused')}
          </div>
        )}

        <TrainLmForm
          capabilities={capabilities}
          value={form}
          vram={trainLmVram}
          onChange={patchForm}
          disabled={jobActive || starting}
          starting={starting}
          onStart={() => void handleStart()}
        />
      </div>

      {/* ── Job area (LM) ────────────────────────────────────────────────── */}
      {/* Gate 5 above already returned for every other kind, so `jobActive`
          here means train-lm or train-dit — the DiT run has its own area. */}
      {jobKind === 'train-lm' && (
        <div className="flex flex-col gap-3">
          <JobProgress />

          {trainLmSkippedLong > 0 && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.train.skippedLong', { count: trainLmSkippedLong })}
            </div>
          )}

          {/* The step layer arrives long before the second epoch does, so the
              chart is worth showing from the first handful of optimizer steps —
              on a long-epoch run that is the only live feedback there is. The
              stats strip renders from the first frame either way: elapsed time
              is useful before any curve exists. */}
          <div className={`${CARD} flex flex-col gap-2`}>
            <TrainingRunStats />
            {(trainLmEpochs.length >= 2 || trainStepSeries.length >= 2) && (
              <TrainingChart
                epochs={trainLmEpochs}
                steps={trainStepSeries}
                milestones={trainMilestones}
                target={form.targetLoss}
                maxEpochs={trainMaxEpochs || form.epochs}
              />
            )}
          </div>

          {/* No ETA tile: the run's only ETA lives in TrainingRunStats above. */}
          {trainLmLast && (
            <div className={`${CARD} grid grid-cols-2 sm:grid-cols-3 gap-3`}>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.epochLoss')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainLmLast.loss.toFixed(4)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.learningRateShort')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainLmLast.lr.toExponential(2)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.gradNorm')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainLmLast.gradNorm.toFixed(3)}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Done state ───────────────────────────────────────────────────── */}
      {status?.adapterExists && (
        <div className={`${CARD} flex flex-col gap-3 border-emerald-500/30`}>
          <div className="flex items-center gap-2">
            <Check size={15} className="text-emerald-500 flex-shrink-0" />
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.done')}</h3>
            {status.stoppedOnTarget && (
              <span className="flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-emerald-500 bg-emerald-500/10 border-emerald-500/20">
                <Target size={10} /> {t('trainingStudio.train.stoppedOnTarget')}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <code className="flex-1 min-w-0 truncate font-mono text-[11px] text-zinc-600 dark:text-zinc-400" title={status.adapterDir}>
              {status.adapterDir}
            </code>
            <button
              onClick={copyAdapterDir}
              className="p-1.5 rounded-lg text-zinc-500 hover:text-amber-500 hover:bg-amber-500/10 transition-colors flex-shrink-0"
              title={status.adapterDir}
            >
              {copied ? <Check size={13} className="text-emerald-500" /> : <Copy size={13} />}
            </button>
          </div>

          <div className="flex items-center gap-3 flex-wrap text-[11px] text-zinc-600 dark:text-zinc-400">
            {status.finalLoss >= 0 && (
              <span className="tabular-nums">
                {t('trainingStudio.train.finalLoss')}: <strong>{status.finalLoss.toFixed(4)}</strong>
              </span>
            )}
            {status.epochsRun > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.epochsRun')}: <strong>{status.epochsRun}</strong>
                </span>
              </>
            )}
            {/* Time taken, summed from the persisted per-epoch durations — the
                job (and its startedAt) is long gone by the time this card is
                read off disk. Train stage only; extract/export are not in it. */}
            {totalEpochMs(status.epochs) > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.timeTaken')}: <strong>{formatDurationMs(totalEpochMs(status.epochs))}</strong>
                </span>
              </>
            )}
            {status.adapterBytes > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">{formatBytes(status.adapterBytes)}</span>
              </>
            )}
            {status.lmSize && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span>{status.lmSize}</span>
              </>
            )}
          </div>

          {/* Done state: only the epoch series survives to disk, so no step
              layer here — TrainingChart degrades to the epoch + MA5 curves. */}
          {status.epochs.length >= 2 && (
            <TrainingChart
              epochs={status.epochs}
              milestones={status.milestones}
              target={status.targetLoss}
              maxEpochs={status.epochsRun}
            />
          )}

          {status.milestones.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.milestones')}</span>
              <div className="flex items-center gap-2 flex-wrap">
                {status.milestones.map(m => (
                  <button
                    key={m.path}
                    onClick={() => auditionMilestone(m.path, m.loss, m.epoch)}
                    title={`${m.path}\n${t('trainingStudio.audition.milestoneButton')}`}
                    className="flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-sky-500 bg-sky-500/10 border-sky-500/20 hover:bg-sky-500/20 transition-colors tabular-nums"
                  >
                    <Flag size={10} /> {m.loss.toFixed(1)} · #{m.epoch}
                    <Headphones size={10} className="text-amber-500" />
                  </button>
                ))}
              </div>
            </div>
          )}

          <p className="text-xs text-emerald-600 dark:text-emerald-400">{t('trainingStudio.train.visibleNow')}</p>
        </div>
      )}

      {/* ── Codes audition ───────────────────────────────────────────────── */}
      {/* Auto-expanded once an adapter exists; collapsed-but-openable before
          that, because an A/B with nothing to compare against is noise (§6.2). */}
      <AuditionCard milestoneRequest={milestoneRequest} />

      {/* ── DiT card ─────────────────────────────────────────────────────── */}
      <div className={`${CARD} flex flex-col gap-4`}>
        <div className="flex items-start gap-2">
          <Waves size={15} className="text-amber-500 mt-0.5 flex-shrink-0" />
          <div className="min-w-0">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.dit.title')}</h3>
            <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.train.dit.subtitle')}</p>
          </div>
        </div>

        {/* Gates, first failing one wins (§5.2). Gates 2 and 4 above already
            cover the common cases for the LM card; these keep the DiT card
            honest on its own capability block and its own variant read. */}
        {!capabilities.trainDit?.available ? (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">
            <XCircle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.train.dit.noBinary')}
          </div>
        ) : !ditStatus ? (
          <div className="flex items-center gap-2 px-3 py-2 text-xs text-zinc-500">
            <Loader2 size={13} className="animate-spin flex-shrink-0" />
            {ditStatusLoading ? '…' : t('trainingStudio.train.dit.statusUnavailable')}
          </div>
        ) : ditStatus.variantKey === '' ? (
          <div className="flex flex-col items-start gap-3">
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.train.dit.needsPreprocess')}
            </div>
            <button
              onClick={() => setPhase('preprocess')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
            >
              {t('trainingStudio.train.goToPreprocess')}
            </button>
          </div>
        ) : jobActive && jobKind !== 'train-dit' ? (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
            <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.train.dit.otherJobRunning')}
          </div>
        ) : null}

        {ditJobActive && ditEnginePaused && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
            <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.preprocess.enginePaused')}
          </div>
        )}

        {/* Pre-flight provenance, the DiT twin of the LM card's codes strip:
            sampleCount says how many cached songs will actually train, and
            channelStats decides whether the default-ON Channel balance switch
            does anything at all (a missing channel_stats.json disables it
            silently, plan §3.1). Neither had a reader before. */}
        {ditStatus && ditStatus.variantKey !== '' && (
          <div className="flex flex-col gap-1">
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
              <span className="tabular-nums font-semibold text-zinc-700 dark:text-zinc-300">
                {ditStatus.sampleCount}
              </span>
              <span>{t('trainingStudio.train.dit.cachedSongs')}</span>
              <span className="text-zinc-400 dark:text-zinc-600">·</span>
              <span>{t('trainingStudio.train.dit.minVram')}</span>
            </div>
            {!ditStatus.channelStats && (
              <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400">
                <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                {t('trainingStudio.train.dit.channelStatsMissing')}
              </div>
            )}
          </div>
        )}

        <TrainDitForm
          value={ditForm}
          onChange={patchDitForm}
          ditModel={ditStatus?.ditModel ?? ''}
          sampleCount={ditStatus?.sampleCount ?? 0}
          disabled={
            jobActive || ditStarting || !capabilities.trainDit?.available || !ditStatus ||
            ditStatus.variantKey === ''
          }
          starting={ditStarting}
          onStart={() => void handleStartDit()}
        />
      </div>

      {/* ── Job area (DiT) ───────────────────────────────────────────────── */}
      {jobKind === 'train-dit' && (
        <div className="flex flex-col gap-3">
          <JobProgress />

          <div className={`${CARD} flex flex-col gap-2`}>
            <TrainingRunStats />
            {(trainDitEpochs.length >= 2 || trainStepSeries.length >= 2) && (
              <TrainingChart
                epochs={trainDitEpochs}
                steps={trainStepSeries}
                milestones={trainMilestones}
                target={ditForm.targetLoss}
                maxEpochs={trainMaxEpochs || ditForm.epochs}
              />
            )}
          </div>

          {trainDitLast && (
            <div className={`${CARD} grid grid-cols-2 sm:grid-cols-4 gap-3`}>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.dit.epochLoss')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainDitLast.loss.toFixed(4)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.dit.ma5')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">
                  {trainDitLast.ma5 > 0 ? trainDitLast.ma5.toFixed(4) : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.learningRateShort')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainDitLast.lr.toExponential(2)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.dit.gradNorm')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">{trainDitLast.gradNorm.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.dit.cropFrames')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">
                  {trainDitLast.crop > 0 ? trainDitLast.crop : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.dit.trainedLayers')}</span>
                <span className="text-sm font-bold tabular-nums text-zinc-900 dark:text-white">
                  {trainDitLast.layers > 0 ? trainDitLast.layers : '—'}
                </span>
              </div>
              {/* No ETA tile — TrainingRunStats above the chart owns it. */}
            </div>
          )}
        </div>
      )}

      {/* ── DiT done state ───────────────────────────────────────────────── */}
      {ditStatus?.adapterExists && (
        <div className={`${CARD} flex flex-col gap-3 border-emerald-500/30`}>
          <div className="flex items-center gap-2 flex-wrap">
            <Check size={15} className="text-emerald-500 flex-shrink-0" />
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.train.dit.done')}</h3>
            {ditStatus.stoppedOnTarget && (
              <span className="flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-emerald-500 bg-emerald-500/10 border-emerald-500/20">
                <Target size={10} /> {t('trainingStudio.train.dit.stoppedOnTarget')}
              </span>
            )}
            {ditStatus.partialDepth && (
              <span
                className="flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20"
                title={t('trainingStudio.train.dit.partialDepth')}
              >
                <Layers size={10} /> {t('trainingStudio.train.dit.partialDepth')}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <code className="flex-1 min-w-0 truncate font-mono text-[11px] text-zinc-600 dark:text-zinc-400" title={ditStatus.adapterDir}>
              {ditStatus.adapterDir}
            </code>
            <button
              onClick={copyDitAdapterDir}
              className="p-1.5 rounded-lg text-zinc-500 hover:text-amber-500 hover:bg-amber-500/10 transition-colors flex-shrink-0"
              title={ditStatus.adapterDir}
            >
              {ditCopied ? <Check size={13} className="text-emerald-500" /> : <Copy size={13} />}
            </button>
          </div>

          <div className="flex items-center gap-3 flex-wrap text-[11px] text-zinc-600 dark:text-zinc-400">
            {ditStatus.finalLoss >= 0 && (
              <span className="tabular-nums">
                {t('trainingStudio.train.dit.finalLoss')}: <strong>{ditStatus.finalLoss.toFixed(4)}</strong>
              </span>
            )}
            {ditStatus.bestLoss >= 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.dit.bestLoss')}: <strong>{ditStatus.bestLoss.toFixed(4)}</strong>
                </span>
              </>
            )}
            {ditStatus.epochsRun > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.dit.epochsRun')}: <strong>{ditStatus.epochsRun}</strong>
                </span>
              </>
            )}
            {totalEpochMs(ditStatus.epochs) > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.timeTaken')}: <strong>{formatDurationMs(totalEpochMs(ditStatus.epochs))}</strong>
                </span>
              </>
            )}
            {ditStatus.crop > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.dit.cropFrames')}: <strong>{ditStatus.crop}</strong>
                </span>
              </>
            )}
            {ditStatus.layers > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">
                  {t('trainingStudio.train.dit.trainedLayers')}: <strong>{ditStatus.layers}</strong>
                </span>
              </>
            )}
            {ditStatus.adapterBytes > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="tabular-nums">{formatBytes(ditStatus.adapterBytes)}</span>
              </>
            )}
          </div>

          {ditStatus.epochs.length >= 2 && (
            <TrainingChart
              epochs={ditStatus.epochs}
              milestones={ditStatus.milestones}
              target={ditStatus.targetLoss}
              maxEpochs={ditStatus.epochsRun}
            />
          )}

          {ditStatus.milestones.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-zinc-500">{t('trainingStudio.train.dit.milestones')}</span>
              <div className="flex items-center gap-2 flex-wrap">
                {ditStatus.milestones.map(m => (
                  <span
                    key={m.path}
                    title={m.path}
                    className="flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-sky-500 bg-sky-500/10 border-sky-500/20 tabular-nums"
                  >
                    <Flag size={10} /> {m.loss.toFixed(1)} · #{m.epoch}
                  </span>
                ))}
              </div>
            </div>
          )}

          <p className="text-xs text-emerald-600 dark:text-emerald-400">{t('trainingStudio.train.dit.visibleNow')}</p>
        </div>
      )}
    </div>
  );
};

export default TrainPanel;
