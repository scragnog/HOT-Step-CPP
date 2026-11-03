// Yue2NarLiveRun.tsx — the live half of a running NAR job: the shared job
// progress, the per-step stat tiles and the split-loss chart. Renders nothing
// unless the selected job is this trainer's, so the card can mount it blindly.

import React from 'react';
import { useTranslation } from 'react-i18next';
import type { TrainingJobSummary } from '../../services/trainingApi';
import type { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
import { Yue2SplitChart } from './Yue2SplitChart';
import { CARD } from './yue2CardUi';

type StoreState = ReturnType<typeof useTrainingStore.getState>;

export const Yue2NarLiveRun: React.FC<{
  job: TrainingJobSummary | null;
  live: StoreState['yue2Live'];
  stepSeries: StoreState['trainStepSeries'];
  milestones: StoreState['trainMilestones'];
}> = ({ job, live, stepSeries, milestones }) => {
  const { t } = useTranslation();
  if (!job || job.kind !== 'yue2-nar-train') return null;

  return (
    <div className={CARD}>
      <JobProgress />
      {live && (
        <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2">
          {[
            { k: 'loss', v: live.loss ? live.loss.toFixed(4) : '—' },
            { k: 'runMean', v: live.runMean ? live.runMean.toFixed(4) : '—' },
            { k: 'gradNorm', v: live.gradNorm ? live.gradNorm.toFixed(3) : '—' },
            { k: 'stepTime', v: live.stepMs ? `${(live.stepMs / 1000).toFixed(2)}s` : '—' },
            {
              k: 'vram',
              v: live.totalMb
                ? `${Math.round(live.usedMb / 1024)}/${Math.round(live.totalMb / 1024)} GB`
                : '—',
              warn: live.totalMb > 0 && live.usedMb > live.totalMb - 512,
            },
          ].map(tile => (
            <div key={tile.k}
              className="rounded-lg border border-zinc-200 dark:border-white/5 px-2.5 py-1.5">
              <div className="text-[10px] uppercase tracking-wider text-zinc-500">
                {t(`trainingStudio.yue2.stat.${tile.k}`, tile.k)}
              </div>
              <div className={`text-sm font-semibold tabular-nums ${
                (tile as { warn?: boolean }).warn ? 'text-amber-500' : 'text-zinc-800 dark:text-zinc-200'
              }`}>{tile.v}</div>
            </div>
          ))}
        </div>
      )}
      {stepSeries.length > 0 && (
        <div className="mt-3">
          {/* target 0 hides the target line: the YuE2 trainer has no
              stop-on-loss mode, so there is never a line to draw. */}
          <Yue2SplitChart mode="nar" points={stepSeries.map(s => ({ ...s, ep: s.step }))}
            milestones={milestones} vramMb={live?.usedMb} totalSteps={job.total} />
        </div>
      )}
    </div>
  );
};
