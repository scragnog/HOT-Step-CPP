// TrainingRunStats.tsx — the ONE place a training run's ETA is rendered.
//
// Sits directly above the loss curve because the number only means anything
// with the curve next to it: "≈9m to target 0.40" and a violet MA5 line that
// has been flat for forty epochs are a conversation, not a promise.
//
// JobProgress used to render a second, different ETA in its header (and the
// metric tile row a third, the engine's own time-to-cap). Both are gone — see
// the note above selectTrainingEta in trainingStore.ts.

import React, { useEffect, useState } from 'react';
import { Clock, Gauge, Timer } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { selectTrainingEpochs, selectTrainingEta, useTrainingStore } from '../../stores/trainingStore';
import { avgEpochMs, formatDurationMs, formatEpochPace, formatEtaMs } from '../../utils/trainingEta';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';

/** Elapsed is wall-clock, so it has to tick on its own — the SSE metric frames
 *  are minutes apart on a DiT run and would leave the number frozen. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [active]);
  return now;
}

export const TrainingRunStats: React.FC = () => {
  const { t } = useTranslation();
  const job = useActiveTrainingJob();
  const eta = useTrainingStore(s => selectTrainingEta(s, job?.kind));
  const epochs = useTrainingStore(s => selectTrainingEpochs(s, job?.kind));

  const active = job?.status === 'queued' || job?.status === 'running';
  const now = useNow(!!active);

  if (!job) return null;

  // startedAt excludes the queue wait, which is the honest start of the run.
  // It is null until the queue picks the job up; the epoch durations are the
  // only clock before then (and after a reload, where startedAt still arrives
  // from the server but the browser was not watching).
  const startedAt = job.startedAt ?? 0;
  const elapsedMs = startedAt > 0
    ? Math.max(0, (active ? now : (job.finishedAt ?? now)) - startedAt)
    : epochs.reduce((a, e) => a + (e.ms > 0 ? e.ms : 0), 0);
  const perEpochMs = avgEpochMs(epochs);

  const at = eta.kind === 'estimating' || eta.kind === 'none' ? '' : formatEtaMs(eta.etaMs);
  const etaLabel =
    eta.kind === 'target' && at
      ? t('trainingStudio.train.etaToTarget', { eta: at, target: eta.target.toFixed(2) })
      : eta.kind === 'unlikely'
        ? at
          ? t('trainingStudio.train.etaUnlikely', { eta: at, target: eta.target.toFixed(2) })
          : t('trainingStudio.train.etaUnlikelyOpen', { target: eta.target.toFixed(2) })
        : eta.kind === 'stalled'
          ? at
            ? t('trainingStudio.train.etaStalled', { eta: at })
            : t('trainingStudio.train.etaStalledOpen')
          : eta.kind === 'cap' && at
            ? t('trainingStudio.train.etaCap', { eta: at })
            : t('trainingStudio.train.etaEstimating');

  // Amber for the two "this is not going to happen" states, so the strip reads
  // at a glance without anyone parsing the sentence.
  const etaTone = eta.kind === 'unlikely' || eta.kind === 'stalled'
    ? 'text-amber-600 dark:text-amber-400'
    : 'text-zinc-600 dark:text-zinc-300';

  return (
    <div className="flex items-center gap-x-3 gap-y-1 flex-wrap text-[11px] text-zinc-500">
      {active && (
        <span className={`flex items-center gap-1 font-semibold min-w-0 ${etaTone}`} title={etaLabel}>
          <Clock size={12} className="flex-shrink-0" />
          <span className="truncate">{etaLabel}</span>
        </span>
      )}
      {elapsedMs > 0 && (
        <span className="flex items-center gap-1 tabular-nums">
          <Timer size={12} className="flex-shrink-0" />
          {t('trainingStudio.train.elapsed', { time: formatDurationMs(elapsedMs) })}
        </span>
      )}
      {perEpochMs > 0 && (
        <span className="flex items-center gap-1 tabular-nums">
          <Gauge size={12} className="flex-shrink-0" />
          {t('trainingStudio.train.pace', { pace: formatEpochPace(perEpochMs) })}
        </span>
      )}
    </div>
  );
};

export default TrainingRunStats;
