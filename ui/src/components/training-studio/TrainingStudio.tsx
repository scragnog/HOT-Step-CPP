// TrainingStudio.tsx — Training Studio orchestrator (Dataset phase)
//
// Phase 1 ships dataset creation: import audio → label locally
// (/understand + Essentia) → optionally enhance in the cloud → review/edit →
// build dataset.json. Phase 2 adds Preprocess (dataset.json → tensor caches).
// Phase 3 adds Train (tensor caches → an LM LoRA). Monitor is still a stepper
// chip only.

import React, { useEffect, useRef } from 'react';
import { AlertTriangle, GraduationCap, Layers } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore, type TrainingPhase } from '../../stores/trainingStore';
import { YUE2_BACKEND_ID } from '../../utils/yue2CaptionSource';
import { CapabilityBanner } from './CapabilityBanner';
import { DatasetDetail } from './DatasetDetail';
import { DatasetList } from './DatasetList';
import { MonitorPanel } from './MonitorPanel';
import { PhaseStepper } from './PhaseStepper';
import { RefinePanel } from './RefinePanel';
import { ReviewPanel } from './ReviewPanel';
import { PreprocessPanel } from './PreprocessPanel';
import { TrainPanel } from './TrainPanel';
import { Yue2QueuePanel } from './Yue2QueuePanel';
import { Yue2BatchPanel } from './Yue2BatchPanel';

// ── URL helpers ──────────────────────────────────────────────────────────────
//
// Mirrors Lyric Studio's scheme (LyricStudioV2.tsx): a base path plus deep
// segments for the selected entity and sub-view, restored on load and synced
// on every navigation so refresh/back/forward land in the same place.
//
//   /training-studio                                → list (phase=dataset, no selection)
//   /training-studio/dataset/:id                    → dataset detail (phase=dataset)
//   /training-studio/dataset/:id/preprocess          → preprocess phase
//   /training-studio/dataset/:id/train               → train phase (incl. YuE2 stages)
//   /training-studio/monitor                         → batch pipeline queue (dataset-independent)

const TS_BASE = '/training-studio';

function buildTrainingUrl(datasetId: string | null, phase: TrainingPhase): string {
  if (phase === 'monitor') return `${TS_BASE}/monitor`;
  if (!datasetId) return TS_BASE;
  const id = encodeURIComponent(datasetId);
  if (phase === 'preprocess') return `${TS_BASE}/dataset/${id}/preprocess`;
  if (phase === 'train') return `${TS_BASE}/dataset/${id}/train`;
  if (phase === 'refine') return `${TS_BASE}/dataset/${id}/refine`;
  if (phase === 'review') return `${TS_BASE}/review`;
  return `${TS_BASE}/dataset/${id}`;
}

function parseTrainingUrl(path: string): { datasetId?: string; phase?: 'preprocess' | 'train' | 'refine' | 'review' | 'monitor' } {
  if (path.startsWith(`${TS_BASE}/monitor`)) return { phase: 'monitor' };
  if (path.startsWith(`${TS_BASE}/review`)) return { phase: 'review' };
  const m = path.match(/\/training-studio\/dataset\/([^/]+)(?:\/(preprocess|train|refine))?/);
  if (!m) return {};
  return { datasetId: decodeURIComponent(m[1]), phase: (m[2] as 'preprocess' | 'train' | 'refine') || undefined };
}

export const TrainingStudio: React.FC = () => {
  const { t } = useTranslation();
  const phase = useTrainingStore(s => s.phase);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const error = useTrainingStore(s => s.error);
  const detail = useTrainingStore(s => s.detail);
  const loadCapabilities = useTrainingStore(s => s.loadCapabilities);
  const loadDatasets = useTrainingStore(s => s.loadDatasets);
  const openDataset = useTrainingStore(s => s.openDataset);
  const closeDataset = useTrainingStore(s => s.closeDataset);
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

  // ── URL routing ──
  // True while a URL (initial load or popstate) is being applied to the store,
  // so the sync effect below doesn't fight it with an intermediate pushState
  // while openDataset() is still resolving.
  const isRestoringUrl = useRef(false);

  // Initial load: adopt whatever dataset/phase the URL names.
  useEffect(() => {
    const parsed = parseTrainingUrl(window.location.pathname);
    if (parsed.phase === 'monitor') {
      setPhase('monitor');
      return;
    }
    if (!parsed.datasetId) return;
    isRestoringUrl.current = true;
    (async () => {
      try {
        await openDataset(parsed.datasetId!);
        setPhase(parsed.phase ?? 'dataset');
      } finally {
        isRestoringUrl.current = false;
      }
    })();
    // Runs once on mount only — subsequent navigation is handled by the sync
    // effect and popstate listener below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep the URL in sync with the store (dataset selection + phase).
  useEffect(() => {
    if (isRestoringUrl.current) return;
    const url = buildTrainingUrl(selectedDatasetId, phase);
    if (window.location.pathname !== url) window.history.pushState({}, '', url);
  }, [selectedDatasetId, phase]);

  // Browser back/forward.
  useEffect(() => {
    const handlePopState = () => {
      const parsed = parseTrainingUrl(window.location.pathname);
      isRestoringUrl.current = true;
      (async () => {
        try {
          if (parsed.phase === 'monitor') {
            setPhase('monitor');
          } else if (parsed.datasetId) {
            await openDataset(parsed.datasetId);
            setPhase(parsed.phase ?? 'dataset');
          } else {
            closeDataset();
            setPhase('dataset');
          }
        } finally {
          isRestoringUrl.current = false;
        }
      })();
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [openDataset, closeDataset, setPhase]);

  // PreprocessPanel is ACE's tensor cache and MM3's codes; YuE2 has neither,
  // and its own latent stage lives on the Train page. The Monitor phase is
  // the same story: its batch pipeline only runs the preprocess-based
  // backends. PhaseStepper drops both chips, but a user who was standing on
  // either phase when the backend changed would otherwise be left looking at
  // content YuE2 can never use — so move them on rather than render it.
  useEffect(() => {
    if (activeBackendId === YUE2_BACKEND_ID && (phase === 'preprocess' || phase === 'monitor')) setPhase('train');
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
        <Yue2BatchPanel />

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
        ) : phase === 'refine' ? (
          <RefinePanel />
        ) : phase === 'review' ? (
          <ReviewPanel />
        ) : (
          <MonitorPanel />
        )}
      </div>
    </div>
  );
};

export default TrainingStudio;
