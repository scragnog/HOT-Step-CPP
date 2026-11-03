// Yue2RunsList.tsx — the NAR checkpoint ladder for a YuE2 dataset.
// Read-only on purpose: there is no resume route for YuE2 yet, so the ladder
// is for picking which rung to load into the adapter field.

import { useQuery } from '@tanstack/react-query';
import { History } from 'lucide-react';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useReload } from '../../queries/training-runs/hooks';
import { runQueries } from '../../queries/training-runs/queries';
import type { Yue2RunSummary } from '../../services/trainingApi';
import { formatDateTime } from '../../utils/dateUtils';
import { gb } from './TrainingNumberField';
import { BTN_SM, CARD } from './yue2CardUi';

// ── Previous runs and their checkpoint ladders ──────────────────────────────

const OUTCOME: Record<Yue2RunSummary['outcome'], { label: string; tone: string }> = {
  completed: { label: 'Finished',      tone: 'text-emerald-500' },
  halted:    { label: 'Stopped early', tone: 'text-amber-500' },
  failed:    { label: 'Failed',        tone: 'text-rose-500' },
  unknown:   { label: 'Unknown',       tone: 'text-zinc-500' },
};

/** Compact, and read-only on purpose: there is no resume route for YuE2 yet
 *  (the engine refuses a resume whose rank/alpha/target/trigger/seed differ,
 *  and it is exact only within one machine and build), so a "continue" button
 *  here would be a promise the server cannot keep. What the ladder is FOR is
 *  picking which rung to load — the adapter picker takes the file path. */
export const Yue2RunsList: React.FC<{ datasetId: string; reloadKey: unknown }> = ({ datasetId, reloadKey }) => {
  const { t } = useTranslation();
  const query = useQuery(runQueries.narRuns(datasetId));
  const runs = query.data?.runs ?? null;
  const error = query.error?.message ?? null;
  useReload(runQueries.narRuns(datasetId).queryKey, reloadKey);
  const [open, setOpen] = useState<string | null>(null);
  const [copied, setCopied] = useState('');


  if (error) return <div className={`${CARD} text-xs text-rose-500`}>{error}</div>;
  if (!runs || runs.length === 0) return null;

  const copy = (p: string) => {
    void navigator.clipboard?.writeText(p).then(
      () => { setCopied(p); window.setTimeout(() => setCopied(''), 1500); },
      () => { /* clipboard blocked — the path is selectable on screen anyway */ },
    );
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <History size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.yue2.runsTitle', 'Previous runs')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2.runsBlurb',
          'Every NAR LoRA this dataset has produced. Each snapshot is a plain safetensors file, so the '
          + 'whole ladder is loadable — copy a path into the YuE2 adapter field to hear that rung.')}
      </p>
      <div className="flex flex-col gap-2">
        {runs.map(run => {
          const o = OUTCOME[run.outcome];
          return (
            <div key={run.runName} className="rounded-lg border border-zinc-200 dark:border-white/10 p-2.5">
              <div className="flex items-start gap-2 flex-wrap">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 break-all">
                    {run.runName}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5">
                    <span className={run.running ? 'text-amber-500' : o.tone}>
                      {run.running ? t('trainingStudio.yue2.runsRunning', 'Running now') : o.label}
                    </span>
                    <span>
                      {t('trainingStudio.yue2.runsSteps', 'step {{done}} of {{cap}}',
                        { done: run.lastStep, cap: run.configuredSteps || run.lastStep })}
                    </span>
                    <span>
                      {t('trainingStudio.yue2.runsCkpts', '{{n}} checkpoints',
                        { n: run.checkpoints.length })}
                    </span>
                    {run.trigger && (
                      <span>
                        {t('trainingStudio.yue2.runsTrigger', 'trigger "{{w}}"', { w: run.trigger })}
                      </span>
                    )}
                    {run.rank !== undefined && (
                      <span>
                        {t('trainingStudio.yue2.runsShape', 'rank {{rank}}, {{target}}',
                          { rank: run.rank, target: run.target || '—' })}
                      </span>
                    )}
                    {run.best && (
                      <span>
                        {t('trainingStudio.yue2.runsBest', 'best loss {{loss}} at {{step}}',
                          { loss: run.best.loss.toFixed(4), step: run.best.step })}
                      </span>
                    )}
                    <span>{gb(run.sizeBytes)}</span>
                    <span>{formatDateTime(run.updatedAt)}</span>
                  </div>
                  {run.failure && (
                    <div className="text-[10px] text-rose-500 mt-1 break-words">{run.failure}</div>
                  )}
                  {run.resume && !run.running && (
                    <div className="text-[10px] text-amber-600/90 dark:text-amber-400/90 mt-1 leading-snug">
                      {t('trainingStudio.yue2.runsHalted',
                        'This run still holds its optimizer state, which means it stopped before its clean '
                        + 'finish — the engine deletes that file on export. Continuing a run is not wired '
                        + 'up yet; start a fresh one with the same recipe.')}
                    </div>
                  )}
                </div>
                {run.checkpoints.length > 0 && (
                  <button className={BTN_SM}
                    onClick={() => setOpen(open === run.runName ? null : run.runName)}>
                    {open === run.runName
                      ? t('trainingStudio.yue2.runsHide', 'Hide ladder')
                      : t('trainingStudio.yue2.runsShow', 'Ladder')}
                  </button>
                )}
              </div>

              {open === run.runName && (
                <div className="mt-2.5 pt-2.5 border-t border-zinc-200 dark:border-white/10 flex flex-col gap-1">
                  {run.checkpoints.map(c => (
                    <button key={c.name} onClick={() => copy(c.path)}
                      title={c.path}
                      className="flex items-center gap-2 text-left text-[10px] text-zinc-600 dark:text-zinc-300 hover:text-amber-600 dark:hover:text-amber-400 transition-colors">
                      <span className="font-mono break-all flex-1 min-w-0">{c.name}</span>
                      {c.final && (
                        <span className="text-emerald-500 shrink-0">
                          {t('trainingStudio.yue2.runsFinal', 'final')}
                        </span>
                      )}
                      {c.loss !== undefined && (
                        <span className="shrink-0 tabular-nums">{c.loss.toFixed(4)}</span>
                      )}
                      <span className="shrink-0 text-zinc-500">{gb(c.bytes)}</span>
                      <span className="shrink-0 text-zinc-500">
                        {copied === c.path
                          ? t('trainingStudio.yue2.runsCopied', 'copied')
                          : t('trainingStudio.yue2.runsCopy', 'copy path')}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

