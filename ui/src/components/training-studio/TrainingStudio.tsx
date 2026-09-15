// TrainingStudio.tsx — Training Studio orchestrator (Dataset phase)
//
// Phase 1 ships dataset creation: import audio → label locally
// (/understand + Essentia) → optionally enhance in the cloud → review/edit →
// build dataset.json. Phase 2 adds Preprocess (dataset.json → tensor caches).
// Phase 3 adds Train (tensor caches → an LM LoRA). Monitor is still a stepper
// chip only.

import React, { useEffect } from 'react';
import { AlertTriangle, GraduationCap, Layers } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { YUE2_BACKEND_ID } from '../../utils/yue2CaptionSource';
import { CapabilityBanner } from './CapabilityBanner';
import { DatasetDetail } from './DatasetDetail';
import { DatasetList } from './DatasetList';
import { MonitorPanel } from './MonitorPanel';
import { PhaseStepper } from './PhaseStepper';
import { PreprocessPanel } from './PreprocessPanel';
import { TrainPanel } from './TrainPanel';
import { Yue2QueuePanel } from './Yue2QueuePanel';

export const TrainingStudio: React.FC = () => {
  const { t } = useTranslation();
  const phase = useTrainingStore(s => s.phase);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const error = useTrainingStore(s => s.error);
  const detail = useTrainingStore(s => s.detail);
  const loadCapabilities = useTrainingStore(s => s.loadCapabilities);
  const loadDatasets = useTrainingStore(s => s.loadDatasets);
  // WHICH BACKEND THIS STUDIO IS TRAINING FOR. It was invisible, and the phases
  // change shape with it (ACE preprocesses a tensor cache, MM3 exports RVQ
  // codes), so a user could reasonably read ACE variants as MM3's and wonder
  // why nothing matched. The chip below is not decoration.
  const backends = useBackendStore(s => s.backends);
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const fetchBackends = useBackendStore(s => s.fetchBackends);
  const activeBackend = backends.find(b => b.id === activeBackendId);
  const multiBackend = backends.length > 1;

  const setPhase = useTrainingStore(s => s.setPhase);

  useEffect(() => {
    void loadCapabilities();
    void loadDatasets();
    void fetchBackends();
  }, [loadCapabilities, loadDatasets, fetchBackends]);

  // PreprocessPanel is ACE's tensor cache and MM3's codes; YuE2 has neither,
  // and its own latent stage lives on the Train page. PhaseStepper drops the
  // chip, but a user who was standing on the phase when the backend changed
  // would otherwise be left looking at ACE's variant list with no way back to
  // it — so move them on rather than render something YuE2 can never consume.
  useEffect(() => {
    if (activeBackendId === YUE2_BACKEND_ID && phase === 'preprocess') setPhase('train');
  }, [activeBackendId, phase, setPhase]);

  const fatalError = error && !detail;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col gap-6">
        {/* Title */}
        <div className="flex items-start gap-3">
          <GraduationCap size={26} className="text-amber-500 mt-0.5 flex-shrink-0" />
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-xl font-bold text-zinc-900 dark:text-white">{t('trainingStudio.title')}</h1>
              {multiBackend && activeBackend && (
                <span
                  title={t('trainingStudio.backendHint',
                    'Training follows the active generation backend. Switch it with the backend pill in the bar at the top of the window.') as string}
                  className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold border border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400"
                >
                  <Layers size={11} />
                  {activeBackend.displayName}
                </span>
              )}
            </div>
            <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.subtitle')}</p>
            {/* Side-Step is a deeper training suite than this one and the LoKR /
                LoRA formats are interchangeable — worth pointing users at. */}
            <p className="text-[11px] text-zinc-500 dark:text-zinc-500 mt-1.5">
              {t('trainingStudio.sideStepNote')}{' '}
              <a
                href="https://github.com/koda-dernet/Side-Step"
                target="_blank"
                rel="noreferrer"
                className="font-semibold text-amber-600 dark:text-amber-500 hover:underline"
              >
                Side-Step
              </a>{' '}
              {t('trainingStudio.sideStepNoteTail')}
            </p>
          </div>
        </div>

        <CapabilityBanner />
        <PhaseStepper />

        {/* Above the phase content, not inside it: the bulk queue is started
            from the dataset grid, runs datasets that are never opened, and
            must stay visible wherever the user wanders while it works. Renders
            nothing until a queue has been started this session. */}
        <Yue2QueuePanel />

        {fatalError ? (
          <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-5 flex flex-col items-start gap-3">
            <div className="flex items-start gap-2 text-sm text-red-500 dark:text-red-400">
              <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.error.load', { message: error })}
            </div>
            <button
              onClick={() => { void loadCapabilities(); void loadDatasets(); }}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-500/15 border border-red-500/25 text-red-500 hover:bg-red-500/25 transition-colors"
            >
              {t('trainingStudio.error.retry')}
            </button>
          </div>
        ) : phase === 'dataset' ? (
          selectedDatasetId ? <DatasetDetail /> : <DatasetList />
        ) : phase === 'preprocess' ? (
          <PreprocessPanel />
        ) : phase === 'train' ? (
          <TrainPanel />
        ) : (
          <MonitorPanel />
        )}
      </div>
    </div>
  );
};

export default TrainingStudio;
