// JobProgress.tsx — live job progress bar, log tail and cancel
//
// Progress arrives over SSE (see useTrainingStream). Understand work sits
// behind the engine's single FIFO GPU worker, so queue depth is surfaced
// rather than pretending labeling is instant.

import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, Clock, Loader2, RotateCcw, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';

const CARD ='rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

const PHASE_KEYS: Record<string, string> = {
  essentia: 'trainingStudio.job.phaseEssentia',
  understand: 'trainingStudio.job.phaseUnderstand',
  'waiting-for-engine': 'trainingStudio.job.phaseUnderstand',
  genius: 'trainingStudio.job.phaseGenius',
  llm: 'trainingStudio.job.phaseLlm',
  build: 'trainingStudio.job.phaseBuild',
  // yue2-stems. Without this the raw phase enum is interpolated into the
  // translated sentence and non-English users read "separating" mid-sentence.
  separating: 'trainingStudio.job.phaseSeparating',
  // Preprocess phases (§2.7). Without these the raw kebab-case enum is
  // interpolated verbatim into the translated "{{done}}/{{total}} — {{phase}}"
  // sentence, so non-English users read "engine-stop" mid-sentence.
  'engine-stop': 'trainingStudio.job.phaseEngineStop',
  'loading-models': 'trainingStudio.job.phaseLoadingModels',
  preprocess: 'trainingStudio.job.phasePreprocess',
  stats: 'trainingStudio.job.phaseStats',
  'engine-restart': 'trainingStudio.job.phaseEngineRestart',
  // train-lm phases (§2.7). 'engine-stop', 'loading-models' and
  // 'engine-restart' above are reused verbatim.
  extract: 'trainingStudio.job.phaseExtract',
  train: 'trainingStudio.job.phaseTrain',
  export: 'trainingStudio.job.phaseExport',
  // train-dit phases (§2.6). The engine reuses the SAME phase strings as
  // train-lm ('train' / 'export'), so a bare phase lookup cannot distinguish
  // them. These composite `<kind>:<phase>` entries are tried first and give the
  // DiT run its own wording; every other phase falls through to the plain key.
  'train-dit:train': 'trainingStudio.job.phaseTrainDit',
  'train-dit:export': 'trainingStudio.job.phaseExportDit',
  // audition phases (codes-preview plan §5.4). Frozen strings; without these
  // the raw kebab-case enum is interpolated into the translated sentence.
  'audition-lm-base': 'trainingStudio.job.phaseAuditionLmBase',
  'audition-lm-adapter': 'trainingStudio.job.phaseAuditionLmAdapter',
  'audition-decode': 'trainingStudio.job.phaseAuditionDecode',
  'audition-render': 'trainingStudio.job.phaseAuditionRender',
  'audition-writing': 'trainingStudio.job.phaseAuditionWriting',
};

function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '';
  const m = Math.round(seconds / 60);
  if (m < 1) return '<1m';
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

/** The kinds whose `done`/`total` is optimizer steps, not files — the per-item
 *  pace ETA below is meaningless for them and the loss-curve ETA replaces it. */
function isTrainingKind(kind: string): boolean {
  return kind === 'train-lm' || kind === 'train-dit';
}

export const JobProgress: React.FC = () => {
  const { t } = useTranslation();
  const job = useTrainingStore(s => s.activeJob);
  const jobLog = useTrainingStore(s => s.jobLog);
  const cancelJob = useTrainingStore(s => s.cancelJob);
  const startLabel = useTrainingStore(s => s.startLabel);
  const startBuild = useTrainingStore(s => s.startBuild);
  const samplesById = useTrainingStore(s => s.samplesById);

  // Rolling per-sample durations, used for the ETA once 3 have completed.
  const marksRef = useRef<{ jobId: string; lastAt: number; lastDone: number; deltas: number[] }>({
    jobId: '', lastAt: 0, lastDone: 0, deltas: [],
  });
  const [eta, setEta] = useState('');

  useEffect(() => {
    if (!job) return;
    // A training job's `done` counts optimizer steps, so its pace says nothing
    // about how long the run has left — the loss-curve ETA below owns that.
    if (isTrainingKind(job.kind)) return;
    const m = marksRef.current;
    if (m.jobId !== job.id) { m.jobId = job.id; m.lastAt = Date.now(); m.lastDone = job.done; m.deltas = []; setEta(''); return; }
    // Don't count time spent waiting in the global job queue as sample time.
    if (job.status === 'queued') { m.lastAt = Date.now(); return; }
    // `done` only increments as the serial per-track work completes (cloud lane
    // when one is enabled, else the Essentia lane), so every increment is a
    // valid pace sample regardless of which phase produced it.
    if (job.done > m.lastDone) {
      const now = Date.now();
      // A burst of increments per tick (parallel Essentia-only lane) still
      // yields one delta per completion by splitting the elapsed time evenly.
      const completed = job.done - m.lastDone;
      if (m.lastAt) {
        const per = (now - m.lastAt) / 1000 / completed;
        for (let i = 0; i < completed; i++) m.deltas.push(per);
      }
      m.lastAt = now;
      m.lastDone = job.done;
      if (m.deltas.length >= 2) {
        const last5 = m.deltas.slice(-5);
        const avg = last5.reduce((a, b) => a + b, 0) / last5.length;
        setEta(formatEta(avg * Math.max(0, job.total - job.done)));
      }
    }
  }, [job]);

  if (!job) return null;

  const pct = job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;
  const active = job.status === 'queued' || job.status === 'running';
  const failedJob = job.status === 'failed';
  const current = job.currentSampleId ? samplesById[job.currentSampleId] : undefined;
  // Not every job walks dataset ROWS. The YuE2 cache stages walk audio files
  // straight off disk and report a filename here, which is not a sample id and
  // so finds nothing in samplesById — leaving the one line that says what is
  // being worked on blank for exactly the jobs that take longest.
  const currentLabel = current?.filename ?? (job.currentSampleId || '');
  const currentTitle = current?.relPath ?? (job.currentSampleId || '');
  const phaseKey = PHASE_KEYS[`${job.kind}:${job.phase}`] ?? PHASE_KEYS[job.phase];
  const phaseLabel = phaseKey ? t(phaseKey) : job.phase;
  const errorLines = jobLog.filter(l => l.level === 'error').slice(-10);

  // ETA. Per-ITEM pace only, and ONLY for the kinds whose done/total counts
  // files. Training runs render theirs once, in TrainingRunStats above the loss
  // curve — a second number here disagreed with it by construction (that one
  // answers "when does the loss reach the target", the tile row's old one
  // answered "when does the epoch cap arrive") and both were labelled "ETA".
  const etaLabel = eta ? t('trainingStudio.job.eta', { eta }) : t('trainingStudio.job.etaEstimating');
  const showEta = active && !isTrainingKind(job.kind);

  // Retry has to re-run the job the card is actually showing. The enhance kinds
  // carry provider/model options this card does not have, so they use the
  // Enhance panel's own buttons instead.
  const retry =
    job.kind === 'label' ? () => void startLabel({ scope: 'unlabeled' })
      : job.kind === 'build' ? () => void startBuild()
        : null;

  return (
    <div className={`${CARD} flex flex-col gap-3 ${failedJob ? 'border-red-500/30' : ''}`}>
      <div className="flex items-center gap-2">
        {active && <Loader2 size={15} className="animate-spin text-amber-500 flex-shrink-0" />}
        {job.status === 'done' && <CheckCircle2 size={15} className="text-emerald-500 flex-shrink-0" />}
        {failedJob && <XCircle size={15} className="text-red-500 flex-shrink-0" />}
        {job.status === 'cancelled' && <AlertTriangle size={15} className="text-amber-500 flex-shrink-0" />}

        <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 flex-1 min-w-0 truncate">
          {job.status === 'queued'
            ? t('trainingStudio.job.queued')
            : job.status === 'running'
              ? t('trainingStudio.job.running', { done: job.done, total: job.total, phase: phaseLabel })
              : job.status === 'done'
                ? t('trainingStudio.job.done', { done: job.done, failed: job.failed })
                : job.status === 'cancelled'
                  ? t('trainingStudio.job.cancelled')
                  : t('trainingStudio.job.failed')}
        </div>

        {showEta && (
          <span className="text-[11px] text-zinc-500 truncate max-w-[55%]" title={etaLabel}>
            {etaLabel}
          </span>
        )}

        {active && (
          <button
            onClick={() => void cancelJob()}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 transition-colors"
          >
            {t('trainingStudio.job.cancel')}
          </button>
        )}
        {failedJob && retry && (
          <button
            onClick={retry}
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-amber-500/10 border border-amber-500/20 text-amber-500 hover:bg-amber-500/20 transition-colors"
          >
            <RotateCcw size={11} /> {t('trainingStudio.job.retry')}
          </button>
        )}
      </div>

      {/* Bar */}
      <div className="h-1.5 rounded-full bg-zinc-200 dark:bg-white/10 overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${failedJob ? 'bg-red-500' : 'bg-amber-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="flex items-center gap-3 text-[11px] text-zinc-500 flex-wrap">
        <span className="tabular-nums">{job.done} / {job.total}</span>
        {job.failed > 0 && (
          <span className="text-red-500">{t('trainingStudio.job.failedCount', { count: job.failed })}</span>
        )}
        {job.engineQueueDepth > 0 && (
          <span className="flex items-center gap-1 text-amber-600 dark:text-amber-400">
            <Clock size={11} /> {t('trainingStudio.job.waitingEngine', { count: job.engineQueueDepth })}
          </span>
        )}
        {currentLabel && <span className="font-mono truncate max-w-[280px]" title={currentTitle}>{currentLabel}</span>}
      </div>

      {job.error && (
        <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">
          {job.error}
        </div>
      )}

      {(failedJob ? errorLines : jobLog.slice(-8)).length > 0 && (
        <div className="max-h-32 overflow-y-auto rounded-lg bg-zinc-100 dark:bg-black/30 px-3 py-2 font-mono text-[10px] leading-relaxed">
          {(failedJob ? errorLines : jobLog.slice(-8)).map((l, i) => (
            <div
              key={`${l.ts}-${i}`}
              className={
                l.level === 'error' ? 'text-red-400'
                  : l.level === 'warn' ? 'text-amber-400'
                    : 'text-zinc-500'
              }
            >
              {l.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default JobProgress;
