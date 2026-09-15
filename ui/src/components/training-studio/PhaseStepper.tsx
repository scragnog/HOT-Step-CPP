// PhaseStepper.tsx — Dataset · Preprocess · Train · Monitor
//
// All four phases ship. Monitor is the batch-pipeline queue view (see
// MonitorPanel.tsx).
//
// Phase 2 is absent under YuE2: its latent cache is stage 1 of the five-stage
// flow on the Train page (Yue2TrainStages.tsx), which the "Perform all stages"
// button drives. A second entry point onto the same cache would be two UIs free
// to disagree about one artifact.

import React from 'react';
import { Database, Layers, Cpu, Activity, Lock } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { YUE2_BACKEND_ID } from '../../utils/yue2CaptionSource';

type Phase = 'dataset' | 'preprocess' | 'train' | 'monitor';

const PHASES: Array<{ id: Phase; icon: React.ReactNode; labelKey: string; enabled: boolean }> = [
  { id: 'dataset',    icon: <Database size={14} />, labelKey: 'trainingStudio.phase.dataset',    enabled: true },
  { id: 'preprocess', icon: <Layers size={14} />,   labelKey: 'trainingStudio.phase.preprocess', enabled: true },
  { id: 'train',      icon: <Cpu size={14} />,      labelKey: 'trainingStudio.phase.train',      enabled: true },
  { id: 'monitor',    icon: <Activity size={14} />, labelKey: 'trainingStudio.phase.monitor',    enabled: true },
];

export const PhaseStepper: React.FC = () => {
  const { t } = useTranslation();
  const phase = useTrainingStore(s => s.phase);
  const setPhase = useTrainingStore(s => s.setPhase);
  // Phase 2 is a different thing per backend: ACE encodes a tensor cache,
  // MiniMax-Music3 exports RVQ codes. Same slot in the pipeline, so the chip is
  // relabelled rather than a fifth phase being invented.
  const backendId = useBackendStore(s => s.activeBackendId);
  const mm3Mode = backendId === 'minimax-m3';
  // YuE2 owns its preprocess on the Train page, so the slot is dropped rather
  // than relabelled.
  const phases = backendId === YUE2_BACKEND_ID
    ? PHASES.filter(p => p.id !== 'preprocess')
    : PHASES;

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {phases.map((p, i) => {
        const active = phase === p.id;
        return (
          <React.Fragment key={p.id}>
            {i > 0 && <div className="w-4 h-px bg-zinc-300 dark:bg-white/10" />}
            <button
              type="button"
              disabled={!p.enabled}
              onClick={() => p.enabled && setPhase(p.id)}
              title={p.enabled ? undefined : t('trainingStudio.phase.comingSoon')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                active
                  ? 'text-amber-500 bg-amber-500/10 border-amber-500/30'
                  : p.enabled
                    ? 'text-zinc-500 border-transparent hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5'
                    : 'text-zinc-400 dark:text-zinc-600 border-transparent cursor-not-allowed'
              }`}
            >
              {p.enabled ? p.icon : <Lock size={12} />}
              {mm3Mode && p.id === 'preprocess'
                ? t('trainingStudio.mm3.phaseCodes', 'Codes')
                : t(p.labelKey)}
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );
};

export default PhaseStepper;
