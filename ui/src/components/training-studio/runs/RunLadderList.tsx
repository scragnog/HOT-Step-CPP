import React, { useState } from 'react';
import { History } from 'lucide-react';
import { useCopiedKey } from '@/hooks/useCopyFeedback';
import { copyTextThen } from '@/utils/storage/clipboard';
import { formatDateTime } from '@/utils/format/dateUtils';
import { formatLoss, formatTrainingSize } from '@/utils/format/format';
import { BTN_SM, CARD } from '@/components/training-studio/lib/yue2CardUi';
import { RUN_OUTCOME } from '@/components/training-studio/lib/runOutcome';

export interface LadderCheckpoint {
  name: string;
  path: string;
  step?: number;
  bytes?: number;
  final?: boolean;
  loss?: number;
}

export interface LadderRun {
  runName: string;
  outcome: 'completed' | 'halted' | 'failed' | 'unknown';
  running?: boolean;
  lastStep: number;
  configuredSteps?: number;
  checkpoints: LadderCheckpoint[];
  trigger?: string | null;
  rank?: number;
  target?: string | null;
  styleTemplate?: string | null;
  best?: { loss: number; step: number } | null;
  sizeBytes?: number;
  updatedAt: number | string;
  failure?: string | null;
  resume?: unknown;
}

/** Everything the two YuE2 ladders say, already translated by the caller. */
export interface LadderLabels {
  running: string;
  halted: string;
  show: string;
  hide: string;
  final: string;
  copy: string;
  copied: string;
  suggested?: string;
  steps: (done: number, cap: number) => string;
  ckpts: (count: number) => string;
  trigger: (word: string) => string;
  shape: (rank: number, target: string) => string;
  best: (loss: string, step: number) => string;
  style?: (template: string) => string;
  checkpointLoss: (loss: number) => string;
}

/**
 * The read-only run ladder the YuE2 cards share: each run with its outcome,
 * step progress, size and checkpoints, and a copy-this-path row per rung.
 *
 * Yue2RunsList and Yue2ArRunsList were the same 140 lines twice, differing only
 * in their translation keys, the suggested rung and the loss label.
 */
export const RunLadderList: React.FC<{
  runs: LadderRun[] | null;
  error: string | null;
  title: string;
  blurb: React.ReactNode;
  labels: LadderLabels;
  suggestedStep?: number;
}> = ({ runs, error, title, blurb, labels, suggestedStep }) => {
  const [open, setOpen] = useState<string | null>(null);
  const [copied, markCopied] = useCopiedKey();

  if (error) return <div className={`${CARD} text-xs text-rose-500`}>{error}</div>;
  if (!runs || runs.length === 0) return null;

  const copy = (path: string) => copyTextThen(path, () => markCopied(path));

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <History size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{title}</h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">{blurb}</p>
      <div className="flex flex-col gap-2">
        {runs.map(run => {
          const outcome = RUN_OUTCOME[run.outcome];
          return (
            <div key={run.runName} className="rounded-lg border border-zinc-200 dark:border-white/10 p-2.5">
              <div className="flex items-start gap-2 flex-wrap">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 break-all">
                    {run.runName}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5">
                    <span className={run.running ? 'text-amber-500' : outcome.tone}>
                      {run.running ? labels.running : outcome.label}
                    </span>
                    <span>{labels.steps(run.lastStep, run.configuredSteps || run.lastStep)}</span>
                    <span>{labels.ckpts(run.checkpoints.length)}</span>
                    {run.trigger && <span>{labels.trigger(run.trigger)}</span>}
                    {run.rank !== undefined && <span>{labels.shape(run.rank, run.target || '—')}</span>}
                    {run.styleTemplate && labels.style && <span>{labels.style(run.styleTemplate)}</span>}
                    {run.best && <span>{labels.best(formatLoss(run.best.loss), run.best.step)}</span>}
                    <span>{formatTrainingSize(run.sizeBytes ?? 0)}</span>
                    <span>{formatDateTime(run.updatedAt)}</span>
                  </div>
                  {run.failure && (
                    <div className="text-[10px] text-rose-500 mt-1 break-words">{run.failure}</div>
                  )}
                  {!!run.resume && !run.running && (
                    <div className="text-[10px] text-amber-600/90 dark:text-amber-400/90 mt-1 leading-snug">
                      {labels.halted}
                    </div>
                  )}
                </div>
                {run.checkpoints.length > 0 && (
                  <button className={BTN_SM}
                    onClick={() => setOpen(open === run.runName ? null : run.runName)}>
                    {open === run.runName ? labels.hide : labels.show}
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
                      {suggestedStep !== undefined && c.step === suggestedStep && labels.suggested && (
                        <span className="text-amber-500 shrink-0">{labels.suggested}</span>
                      )}
                      {c.final && <span className="text-emerald-500 shrink-0">{labels.final}</span>}
                      {typeof c.loss === 'number' && (
                        <span className="shrink-0 tabular-nums">{labels.checkpointLoss(c.loss)}</span>
                      )}
                      <span className="shrink-0 text-zinc-500">{formatTrainingSize(c.bytes ?? 0)}</span>
                      <span className="shrink-0 text-zinc-500">
                        {copied === c.path ? labels.copied : labels.copy}
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
