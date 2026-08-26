// Mm3RunsPanel.tsx — previous MM3 LM training runs for this dataset, and the
// control that carries one on.
//
// A run directory has always held everything needed to continue: checkpoints,
// the optimizer state, and the log. What it never had was a way to ASK. This is
// that: the runs that exist, how far each got, whether it stopped on purpose,
// and a "continue" that reuses the recipe the run was trained with rather than
// whatever the form above happens to say now.
//
// TWO THINGS THIS DELIBERATELY SURFACES rather than hiding:
//
//   * `behindBy` — the saved state can sit behind the last step the run
//     reached. Runs finished before the engine learned to save state on a clean
//     exit hold state from their last preview pause, so continuing them
//     retrains the steps in between. Better said than silently repeated.
//   * `optionsSource` — a run with no recorded recipe is continued with what
//     its log states plus today's defaults for what the log does not carry.
//     That is a weaker claim than "the recipe it was trained with" and reads as
//     one.

import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, History, Loader2, PlayCircle, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { listMm3Runs, type Mm3RunSummary } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 '
            + 'dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none '
            + 'focus:border-amber-500/50';

const OUTCOME: Record<Mm3RunSummary['outcome'], { label: string; tone: string }> = {
  'completed':      { label: 'Finished',       tone: 'text-emerald-500' },
  'target-reached': { label: 'Target reached', tone: 'text-emerald-500' },
  'halted':         { label: 'Stopped early',  tone: 'text-amber-500' },
  'failed':         { label: 'Failed',         tone: 'text-rose-500' },
  'unknown':        { label: 'Unknown',        tone: 'text-zinc-500' },
};

function gb(bytes: number): string {
  return bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(1)} GB`
                             : `${Math.round(bytes / 1048576)} MB`;
}

function when(ms?: number): string {
  if (!ms) return '';
  const d = new Date(ms);
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export const Mm3RunsPanel: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const { t } = useTranslation();
  const resumeMm3TrainLm = useTrainingStore(s => s.resumeMm3TrainLm);
  const activeJob = useTrainingStore(s => s.activeJob);
  const jobRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';

  const [runs, setRuns] = useState<Mm3RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [addSteps, setAddSteps] = useState(250);
  const [mode, setMode] = useState<'steps' | 'loss'>('steps');
  const [targetLoss, setTargetLoss] = useState(0.4);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await listMm3Runs(datasetId);
      setRuns(r.runs);
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
      setRuns([]);
    }
  }, [datasetId]);

  // Reloaded when a job ends as well as on mount: the run that just finished is
  // the one most likely to be continued, and it would otherwise be missing from
  // a list fetched before it existed.
  useEffect(() => { void load(); }, [load, jobRunning]);

  if (runs !== null && runs.length === 0 && !error) return null;

  const start = async (run: Mm3RunSummary) => {
    if (!run.resume) return;
    setBusy(true);
    try {
      await resumeMm3TrainLm({
        runName: run.runName,
        addSteps: Math.max(1, addSteps),
        stopMode: mode,
        ...(mode === 'loss' ? { targetLoss } : {}),
      });
      setOpen(null);
    } finally {
      setBusy(false);
      void load();
    }
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <History size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.mm3.runsTitle', 'Previous runs')}
        </h3>
        <button onClick={() => void load()}
          className="ml-auto text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          title={t('trainingStudio.mm3.runsRefresh', 'Refresh') as string}>
          <RefreshCw size={13} />
        </button>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.mm3.runsBlurb',
          'Every run this dataset has produced, with its checkpoints still on disk. A run that saved '
          + 'its optimizer state can be carried on: the weights, the momentum, the shuffled pass and '
          + 'the crop RNG all come back, so it continues rather than starts again. The learning-rate '
          + 'schedule is laid out over the NEW step cap, so raising the cap restarts the tail of the '
          + 'cosine curve rather than continuing it.')}
      </p>

      {error && (
        <div className="text-xs text-rose-500 mb-2">{error}</div>
      )}
      {runs === null && (
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <Loader2 size={13} className="animate-spin" /> …
        </div>
      )}

      <div className="flex flex-col gap-2">
        {(runs ?? []).map(run => {
          const o = OUTCOME[run.outcome];
          const last = run.checkpoints.length
            ? run.checkpoints[run.checkpoints.length - 1] : undefined;
          return (
            <div key={run.runName}
              className="rounded-lg border border-zinc-200 dark:border-white/10 p-2.5">
              <div className="flex items-start gap-2 flex-wrap">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 break-all">
                    {run.runName}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5">
                    <span className={run.running ? 'text-amber-500' : o.tone}>
                      {run.running ? t('trainingStudio.mm3.runsRunning', 'Running now') : o.label}
                    </span>
                    <span>
                      {t('trainingStudio.mm3.runsSteps', 'step {{done}} of {{cap}}',
                        { done: run.lastStep, cap: run.configuredSteps || run.lastStep })}
                    </span>
                    <span>
                      {t('trainingStudio.mm3.runsCkpts', '{{n}} checkpoints', {
                        n: run.checkpoints.length,
                      })}
                    </span>
                    {run.best && (
                      <span>
                        {t('trainingStudio.mm3.runsBest', 'best held-out {{loss}} at {{step}}', {
                          loss: run.best.loss.toFixed(4), step: run.best.step,
                        })}
                      </span>
                    )}
                    {last?.loss !== undefined && !run.best && (
                      <span>
                        {t('trainingStudio.mm3.runsLastLoss', 'last loss {{loss}}',
                          { loss: last.loss.toFixed(4) })}
                      </span>
                    )}
                    <span>{gb(run.sizeBytes)}</span>
                    <span>{when(run.updatedAt)}</span>
                    {run.launches > 1 && (
                      <span>
                        {t('trainingStudio.mm3.runsLaunches', 'continued {{n}}x',
                          { n: run.launches - 1 })}
                      </span>
                    )}
                  </div>
                  {run.failure && (
                    <div className="text-[10px] text-rose-500 mt-1 break-words">{run.failure}</div>
                  )}
                </div>
                {run.resume && !run.running && (
                  <button
                    onClick={() => setOpen(open === run.runName ? null : run.runName)}
                    disabled={jobRunning}
                    className="shrink-0 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5"
                  >
                    <PlayCircle size={12} />
                    {t('trainingStudio.mm3.runsContinue', 'Continue')}
                  </button>
                )}
              </div>

              {!run.resume && !run.running && (
                <div className="text-[10px] text-zinc-500 mt-1.5">
                  {t('trainingStudio.mm3.runsNoState',
                    'No saved optimizer state in this run, so it cannot be continued — only runs that '
                    + 'paused for a preview, or finished after this feature shipped, have one. Its '
                    + 'checkpoints are still usable in the adapter picker.')}
                </div>
              )}

              {open === run.runName && run.resume && (
                <div className="mt-2.5 pt-2.5 border-t border-zinc-200 dark:border-white/10 flex flex-col gap-2">
                  <div className="text-[10px] text-zinc-500 leading-snug">
                    {t('trainingStudio.mm3.runsResumeFrom',
                      'Continuing from step {{step}}, saved {{when}}{{how}}.', {
                        step: run.resume.step,
                        when: when(run.resume.savedAt),
                        how: run.resume.reason === 'final'
                          ? ' on a clean exit' : ' at a preview pause',
                      })}
                    {' '}
                    {t('trainingStudio.mm3.runsResumeShape',
                      'rank {{rank}} {{type}} on {{opt}}, {{songs}} songs (+{{held}} held out).', {
                        rank: run.resume.rank, type: run.resume.adapterType.toUpperCase(),
                        opt: run.resume.optimizer, songs: run.resume.samples,
                        held: run.resume.holdout,
                      })}
                  </div>
                  {run.resume.behindBy > 0 && (
                    <div className="flex items-start gap-1.5 text-[10px] text-amber-600 dark:text-amber-400 leading-snug">
                      <AlertTriangle size={11} className="mt-0.5 flex-shrink-0" />
                      <span>
                        {t('trainingStudio.mm3.runsBehind',
                          'The saved state is {{n}} steps behind where this run got to (step {{last}}), '
                          + 'so those steps get trained again. That is not damage — the run simply '
                          + 'repeats ground it covered — but it is time you pay twice.', {
                            n: run.resume.behindBy, last: run.lastStep,
                          })}
                      </span>
                    </div>
                  )}
                  {run.optionsSource !== 'manifest' && (
                    <div className="flex items-start gap-1.5 text-[10px] text-amber-600 dark:text-amber-400 leading-snug">
                      <AlertTriangle size={11} className="mt-0.5 flex-shrink-0" />
                      <span>
                        {t('trainingStudio.mm3.runsReconstructed',
                          'This run predates recipe recording, so it will be continued with what its '
                          + 'training log states (rank, optimizer, crop policy, the held-out split) '
                          + 'and today’s defaults for the rest — learning rate, seed and trigger '
                          + 'among them. The engine refuses the resume outright if anything it '
                          + 'fingerprints does not match.')}
                      </span>
                    </div>
                  )}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    <label className="flex flex-col gap-1">
                      <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
                        {t('trainingStudio.mm3.runsAddSteps', 'Train this many more')}
                      </span>
                      <input type="number" className={INPUT} step={50} min={1}
                        value={addSteps}
                        onChange={e => setAddSteps(Math.max(1, Number(e.target.value)))} />
                      <span className="text-[10px] text-zinc-500">
                        {t('trainingStudio.mm3.runsNewCap', 'new cap: step {{n}}',
                          { n: run.resume.step + Math.max(1, addSteps) })}
                      </span>
                    </label>
                    <label className="flex flex-col gap-1">
                      <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
                        {t('trainingStudio.mm3.runsStopMode', 'Stop on')}
                      </span>
                      <select className={INPUT} value={mode}
                        onChange={e => setMode(e.target.value as 'steps' | 'loss')}>
                        <option value="steps">
                          {t('trainingStudio.mm3.stopSteps', 'Step count')}
                        </option>
                        <option value="loss">
                          {t('trainingStudio.mm3.stopLoss', 'Target loss')}
                        </option>
                      </select>
                    </label>
                    {mode === 'loss' && (
                      <label className="flex flex-col gap-1">
                        <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
                          {t('trainingStudio.mm3.runsTargetLoss', 'Target loss')}
                        </span>
                        <input type="number" className={INPUT} step={0.05} min={0}
                          value={targetLoss}
                          onChange={e => setTargetLoss(Math.max(0, Number(e.target.value)))} />
                        <span className="text-[10px] text-zinc-500">
                          {t('trainingStudio.mm3.runsTargetHint',
                            'The step count above becomes the cap')}
                        </span>
                      </label>
                    )}
                  </div>
                  <button
                    onClick={() => void start(run)}
                    disabled={busy || jobRunning}
                    className="self-start px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 transition-colors flex items-center gap-1.5"
                  >
                    {busy ? <Loader2 size={12} className="animate-spin" /> : <PlayCircle size={12} />}
                    {t('trainingStudio.mm3.runsGo', 'Continue training')}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Mm3RunsPanel;
