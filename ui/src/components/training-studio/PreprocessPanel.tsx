// PreprocessPanel.tsx — Training Studio phase 2
//
// dataset.json + a base model → one safetensors tensor cache per song, written
// under data/training/tensors/<slug>/<variant>. The encode runs in the
// standalone ace-train binary, which needs the GPU to itself: the server stops
// ace-server for the duration of the job and restarts it afterwards, so this
// panel is explicit about the engine being paused.

import React, { useEffect, useState } from 'react';
import { AlertTriangle, Database, Layers, Loader2, PauseCircle, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { PreprocessOptions } from '../../services/trainingApi';
import { useTrainingTarget } from './useTrainingTarget';
import { useTrainingStore } from '../../stores/trainingStore';
import { Yue2StageCard } from './Yue2StageCard';
import { JobProgress } from './JobProgress';
import { Mm3CodesCard } from './Mm3CodesCard';
import {
  BF16_RE,
  PREPROCESS_DEFAULTS,
  PreprocessOptionsForm,
  type PreprocessFormState,
} from './PreprocessOptionsForm';
import { PreprocessVariantCard } from './PreprocessVariantCard';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

export const PreprocessPanel: React.FC = () => {
  const { t } = useTranslation();
  // ACE's preprocess produces a tensor cache keyed by the DiT model that made
  // it; MiniMax-Music3 has neither. Showing ACE's variant list while MM3 is
  // active offers caches an MM3 run can never consume — so the phase switches
  // wholesale, it does not merge.
  // The studio's effective target, not the raw backend: a running job pins it.
  const mm3Mode = useTrainingTarget().target === 'minimax-m3';
  const capabilities = useTrainingStore(s => s.capabilities);
  const datasets = useTrainingStore(s => s.datasets);
  const detail = useTrainingStore(s => s.detail);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const activeJob = useActiveTrainingJob();
  const error = useTrainingStore(s => s.error);
  const preprocessStatus = useTrainingStore(s => s.preprocessStatus);
  const preprocessLoading = useTrainingStore(s => s.preprocessLoading);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const loadCapabilities = useTrainingStore(s => s.loadCapabilities);
  const loadPreprocessStatus = useTrainingStore(s => s.loadPreprocessStatus);
  const startPreprocess = useTrainingStore(s => s.startPreprocess);
  const deletePreprocessVariant = useTrainingStore(s => s.deletePreprocessVariant);

  const [form, setForm] = useState<PreprocessFormState>(PREPROCESS_DEFAULTS);
  const [starting, setStarting] = useState(false);

  const pp = capabilities?.preprocess;
  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobId = activeJob?.id;
  const jobActive = jobStatus === 'queued' || jobStatus === 'running';
  const preprocessJobActive = jobKind === 'preprocess' && jobActive;

  // Seed the pickers from the server's recommendation once capabilities land.
  useEffect(() => {
    if (!pp) return;
    setForm((f) => ({
      ...f,
      ditModel: f.ditModel || pp.defaultDit,
      vaeModel: f.vaeModel || pp.defaultVae,
      textEncoder: f.textEncoder || pp.defaultTextEnc,
    }));
  }, [pp]);

  useEffect(() => {
    void loadPreprocessStatus();
  }, [selectedDatasetId, loadPreprocessStatus]);

  // A finished job changes both the cache on disk and the engine's suspended
  // flag — re-read both.
  useEffect(() => {
    if (jobKind !== 'preprocess' || !jobStatus) return;
    if (jobStatus === 'queued' || jobStatus === 'running') return;
    void loadPreprocessStatus();
    void loadCapabilities();
  }, [jobId, jobKind, jobStatus, loadPreprocessStatus, loadCapabilities]);

  const patchForm = (patch: Partial<PreprocessFormState>) => setForm(f => ({ ...f, ...patch }));

  const handleStart = async () => {
    const opts: PreprocessOptions = {
      ditModel: form.ditModel,
      // §2.7: OMITTED means "use the server default". Sending '' would be passed
      // straight through as `--vae ''` by any resolver that uses ?? instead of
      // ||, and ace-train exits 2.
      ...(form.vaeModel ? { vaeModel: form.vaeModel } : {}),
      ...(form.textEncoder ? { textEncoder: form.textEncoder } : {}),
      maxDuration: form.maxDuration,
      normalize: form.normalize,
      targetDb: form.targetDb,
      dtype: form.dtype,
      compat: form.compat,
      maxCaptionTokens: form.maxCaptionTokens,
      maxLyricTokens: form.maxLyricTokens,
      vaeChunk: form.vaeChunk,
      vaeOverlap: form.vaeOverlap,
      overwrite: form.overwrite,
      stopEngine: form.stopEngine,
    };
    setStarting(true);
    try { await startPreprocess(opts); } finally { setStarting(false); }
  };

  const header = (
    <div className="flex items-start gap-3">
      <Layers size={18} className="text-amber-500 mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <h2 className="text-sm font-bold text-zinc-900 dark:text-white">
          {mm3Mode ? t('trainingStudio.mm3.phaseCodes', 'Codes') : t('trainingStudio.preprocess.title')}
        </h2>
        <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">
          {mm3Mode
            ? t('trainingStudio.mm3.phaseCodesSub', 'Audio → RVQ codes, the MiniMax-Music3 equivalent of preprocessing')
            : t('trainingStudio.preprocess.subtitle')}
        </p>
      </div>
    </div>
  );

  // 1. No dataset selected — pick one first.
  if (!selectedDatasetId) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className={`${CARD} flex flex-col gap-3`}>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">{t('trainingStudio.preprocess.pickDataset')}</p>
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

  if (!detail) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  // MM3: codes ARE the preprocess step. Placed before the ACE gates below,
  // which are about a tensor cache and a base model MM3 does not have.
  if (mm3Mode && detail.builtAt && detail.datasetJsonPath) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <Mm3CodesCard datasetId={detail.id} />
      </div>
    );
  }

  // 2. Not built — Preprocess consumes dataset.json, so Build has to run first.
  if (!detail.builtAt || !detail.datasetJsonPath) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-5 flex flex-col items-start gap-3">
          <div className="flex items-start gap-2 text-sm text-amber-600 dark:text-amber-400">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.preprocess.needsBuild')}
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

  // Capabilities still in flight — deciding anything below needs them.
  if (!capabilities) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  // 3. No ace-train binary — nothing can run. A response with NO `preprocess`
  //    block at all (older server) must land here too: falling through to the
  //    idle branch renders three empty selects and a permanently disabled Start
  //    button with no banner and no explanation.
  if (!pp?.available) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-5 flex items-start gap-2 text-sm text-red-500 dark:text-red-400">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          {t('trainingStudio.preprocess.noBinary')}
        </div>
      </div>
    );
  }

  const enginePaused = form.stopEngine || !!pp?.engineSuspended;
  const notBf16 = !!form.ditModel && !BF16_RE.test(form.ditModel);
  const variants = preprocessStatus?.variants ?? [];

  return (
    <div className="flex flex-col gap-4">
      {header}

      <div className="flex items-center gap-2 text-[11px] text-zinc-500">
        <Database size={12} className="text-amber-500 flex-shrink-0" />
        <span className="font-semibold text-zinc-700 dark:text-zinc-300 truncate">{detail.name}</span>
        <span className="font-mono truncate" title={detail.datasetJsonPath}>{detail.datasetJsonPath}</span>
      </div>

      {error && (
        <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{error}</div>
      )}

      {/* 4. A job owns the GPU — show progress instead of the form. */}
      {preprocessJobActive ? (
        <>
          {enginePaused && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
              <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.preprocess.enginePaused')}
            </div>
          )}
          <JobProgress />
        </>
      ) : (
        <>
          {/* A label/enhance/build job for this dataset disables every control
              here. Without this note the user gets a fully greyed-out form, no
              progress card and no reason why. */}
          {jobActive && jobKind !== 'preprocess' && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
              <PauseCircle size={13} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.preprocess.otherJobRunning')}
            </div>
          )}

          <Yue2StageCard icon={<Layers size={15} className="text-amber-500" />}
            title={t('trainingStudio.assets.tensors')} done={variants.length > 0}
            action={
            <button
              onClick={() => void handleStart()}
              disabled={!form.ditModel || jobActive || starting || !pp.available}
              className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {starting ? <Loader2 size={15} className="animate-spin" /> : <Layers size={15} />}
              {t('trainingStudio.preprocess.start')}
            </button>
            }>
            <PreprocessOptionsForm
              capabilities={capabilities}
              value={form}
              onChange={patchForm}
              disabled={jobActive || starting}
            />

            {notBf16 && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
                <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                {t('trainingStudio.preprocess.notBf16')}
              </div>
            )}

          </Yue2StageCard>

          {/* A finished/failed preprocess job stays visible until the next one
              starts; an active job of another kind gets its progress card too,
              so the disabled form above has something to point at. */}
          {(jobKind === 'preprocess' || jobActive) && <JobProgress />}

          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.preprocess.caches')}</h3>
              {preprocessLoading && <Loader2 size={12} className="animate-spin text-zinc-500" />}
            </div>
            {variants.length === 0 ? (
              <div className={`${CARD} text-sm text-zinc-500`}>{t('trainingStudio.preprocess.noCaches')}</div>
            ) : (
              <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-3">
                {variants.map(v => (
                  <PreprocessVariantCard
                    key={v.variantKey}
                    variant={v}
                    busy={jobActive}
                    onDelete={(key) => void deletePreprocessVariant(key)}
                  />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default PreprocessPanel;
