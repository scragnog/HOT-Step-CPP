// Shared phase host. Parameters, status and execution belong to each plug-in.
import { Cpu, Database, Loader2, XCircle, AlertTriangle } from 'lucide-react';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';
import { getTrainingPlugin } from './trainingPlugins';
import { resolveTrainGate } from './trainGates';
import type { TrainingTargetId, TrainingSection } from './trainingPluginTypes';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

export const TrainPanel: React.FC<{ target: TrainingTargetId; section?: TrainingSection }> = ({ target, section = 'train' }) => {
  const { t } = useTranslation();
  const plugin = getTrainingPlugin(target);
  const capabilities = useTrainingStore(s => s.capabilities);
  const datasets = useTrainingStore(s => s.datasets);
  const detail = useTrainingStore(s => s.detail);
  const trainingWorker = useTrainingStore(s => s.trainingWorker);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);

  // ACE owns its status-dependent gates, including the initial dataset picker.
  if (target === 'ace') return <>{plugin.renderTrain?.({ datasetId: detail?.id ?? '', slug: detail?.slug ?? '', trigger: detail?.customTag || '', section })}</>;

  // Review is the plug-in's own page (HeartMuLa's checkpoint previews); it sits here for the job stream above.
  if (section === 'review' && plugin.renderReview) {
    return detail
      ? <div className="flex flex-col gap-4">{plugin.renderReview({ datasetId: detail.id, slug: detail.slug, trigger: detail.customTag || '', section })}</div>
      : <p className="text-xs text-zinc-500">{t('trainingStudio.heartmula.reviewPickDataset', 'Open a dataset to review its checkpoint previews.')}</p>;
  }

  // HeartMuLa owns readiness through its status route, including preparation
  // from source audio. ACE capabilities and its built manifest are unrelated.
  if (target === 'heartmula' && detail) {
    const render = section === 'prepare' ? plugin.renderPrepare : plugin.renderTrain;
    return <React.Fragment key={`${trainingWorker}:${detail.id}:${section}`}>
      {render?.({ datasetId: detail.id, slug: detail.slug, trigger: detail.customTag || '', section })}
    </React.Fragment>;
  }

  const header = (
    <>
    <div className="flex items-start gap-3">
      <Cpu size={18} className="text-amber-500 mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{section === 'prepare' ? t('trainingStudio.phase.prepare', 'Prepare')
          : section === 'optimise' ? t('trainingStudio.phase.optimise', 'Optimise') : t('trainingStudio.train.title')}</h2>
        <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{section === 'prepare'
          ? t('trainingStudio.yue2.prepareSubtitle', 'Turn the album into what YuE2 trains on: the latent cache, the codec codes and the lead sheets.')
          : section === 'optimise'
            ? t('trainingStudio.yue2.optimiseSubtitle', 'Optional measurements of the prepared album before training, saved with the dataset.')
            : t('trainingStudio.train.subtitle')}</p>
      </div>
    </div>
    </>
  );

  const gate = resolveTrainGate({
    target, section, hasSelectedDataset: !!selectedDatasetId, loaded: !!detail && !!capabilities,
    binaryAvailable: !!capabilities?.trainLm?.available, built: !!detail?.builtAt && !!detail.datasetJsonPath,
    statusPending: false, noVariant: false, jobActive: false,
  });

  // ── Gate 1: no dataset selected ─────────────────────────────────────────
  if (gate === 'pick-dataset') {
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

  if (gate === 'loading' || !detail || !capabilities) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  // SA3 scans source audio and prepares its own snapshot before training.
  // It does not require ACE's built manifest or tensor cache.
  if (gate === 'sa3') return <div className="flex flex-col gap-4">
    <React.Fragment key={`${trainingWorker}:${detail.id}`}>
      {plugin.renderTrain?.({ datasetId: detail.id, slug: detail.slug, trigger: detail.customTag || '', section })}
    </React.Fragment>
  </div>;

  // ── Gate 2: no ace-train binary ─────────────────────────────────────────
  if (gate === 'no-binary') {
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

  // YuE2 prepares directly from audio, before the manifest gate.
  if (gate === 'yue2') {
    return (
      <div className="flex flex-col gap-4">
        {header}
        {(section === 'optimise' ? plugin.renderOptimise : section === 'prepare' ? plugin.renderPrepare : plugin.renderTrain)
          ?.({ datasetId: detail.id, slug: detail.slug, trigger: detail.customTag || '', section })}
      </div>
    );
  }

  // ── Gate 3: not built ───────────────────────────────────────────────────
  if (gate === 'needs-build') {
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

  // MM3 needs the built manifest, but never ACE tensors.
  if (gate === 'mm3') {
    return (
      <div className="flex flex-col gap-4">
        {header}
        {plugin.renderTrain?.({ datasetId: detail.id, slug: detail.slug, trigger: detail.customTag || '', section })}
      </div>
    );
  }

  return null;
};

export default TrainPanel;
