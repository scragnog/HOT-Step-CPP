import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { TrainingChart, type ChartStepPoint, type ChartMilestone } from './TrainingChart';
import { klReading, reduceYue2Metric, type Yue2Point } from './yue2ChartData';
import { isTerminalStatus, jobStreamUrl, type TrainingJobSummary, type TrainingStreamEvent } from '../../services/trainingApi';
import { subscribeEventSource } from '../../services/sharedEventSource';
import { useTrainingStore } from '../../stores/trainingStore';

interface Props {
  mode?: 'joint' | 'ar' | 'nar';
  points?: Yue2Point[];
  job?: TrainingJobSummary | null;
  options?: Record<string, unknown>;
  milestones?: ChartMilestone[];
  evals?: Array<{ ep: number; loss: number }>;
  totalSteps?: number;
  vramMb?: number;
}

export const Yue2SplitChart: React.FC<Props> = ({ mode = 'joint', points, job, options, milestones = [], evals = [], totalSteps = 0, vramMb }) => {
  const { t } = useTranslation();
  const [stream, setStream] = useState<Yue2Point[]>([]);
  const [streamVram, setStreamVram] = useState<number>();
  const hasPoints = points !== undefined;
  // The job lives on the worker selected when its chart opened.
  const worker = useTrainingStore(s => s.trainingWorker);
  // A new job starts its own series; a status change must not wipe the points
  // the stream has already collected for this one.
  useEffect(() => {
    setStream([]); setStreamVram(undefined);
  }, [job?.id, hasPoints, worker]);
  // Only a live job needs the stream. A finished one renders the series it
  // already received, and a subscription left open would only reconnect and
  // replay the closed stream's buffer forever.
  useEffect(() => {
    if (hasPoints || !job?.id || isTerminalStatus(job.status)) return;
    return subscribeEventSource(jobStreamUrl(job.id, { worker }), { message: data => {
      try {
        const item = JSON.parse(data) as TrainingStreamEvent;
        if (item.type !== 'metric') return;
        if (item.metric === 'vram') setStreamVram(item.usedMb ?? item.vramMb ?? (item.totalMb !== undefined && item.freeMb !== undefined ? item.totalMb - item.freeMb : undefined));
        else setStream(p => reduceYue2Metric(p, item));
      } catch { /* Ignore malformed replay frames. */ }
    } });
  }, [job?.id, job?.status, hasPoints, worker]);
  const history = points ?? stream;
  const last = history.at(-1);
  const lastStopMean = history.findLast(p => Number.isFinite(p.ma20))?.ma20;
  const frozenIndex = history.findIndex(p => p.frozen);
  const frozen = frozenIndex < 0 ? undefined : history[Math.max(0, frozenIndex - 1)].step;
  const kls: number[] = [];
  const klMode = options?.targetKlMode === 'trend' ? 'trend' : 'mean';
  const window = typeof options?.targetLossWindow === 'number' ? options.targetLossWindow : 20;
  const partial = options?.unfreezePlanner === true || options?.refinePlanner === true;
  const steps: ChartStepPoint[] = history.map(p => {
    if (p.arKl !== undefined) kls.push(p.arKl);
    return { ...p, ep: p.step, loss: p.loss ?? NaN,
      klStop: options && p.arKl !== undefined ? klReading(kls, klMode, window, partial) : undefined };
  });
  const labels = {
    loss: t(mode === 'ar' ? 'trainingStudio.split.ce' : 'trainingStudio.split.decoderLoss', mode === 'ar' ? 'CE' : 'Decoder loss'),
    arCe: t('trainingStudio.split.arCe', 'AR CE'), cursorCe: t('trainingStudio.split.cursorCe', 'Cursor CE'),
    arKl: t('trainingStudio.split.kl', 'KL vs base'), narMse: t('trainingStudio.split.decoderLoss', 'Decoder loss'),
    narRecon: t('trainingStudio.split.recon', 'Reconstruction'), narDrift: t('trainingStudio.split.drift', 'Drift'),
    eval: t('trainingStudio.split.grammar', 'Token-grammar check'),
    klStop: t('trainingStudio.split.klStop', { mode: klMode, window: klMode === 'trend' ? 30 : window, defaultValue: 'KL {{mode}} ({{window}} readings)' }),
    grad: t('trainingStudio.split.grad', 'Grad norm'), lr: t('trainingStudio.chart.mLr', 'Learning rate'),
  };
  const tooltips = {
    arKl: t('trainingStudio.split.klInfo', "How far the planner's token predictions moved from the frozen base. Higher means more change, not better."),
    narDrift: t('trainingStudio.split.driftInfo', "How much the decoder's predictions changed from its starting adapter on fixed probes."),
    narRecon: t('trainingStudio.split.reconInfo', 'How closely one denoising step rebuilds the training latents on fixed probes. Lower is better on those probes, not a listening score.'),
    grad: t('trainingStudio.split.gradInfo', 'Overall gradient size before clipping. Spikes flag an unstable update.'),
    eval: t('trainingStudio.split.grammarInfo', 'Validation on the minted token pack. It should stay roughly flat; rising means token-grammar damage.'),
  };
  const missing = t('trainingStudio.split.missing', 'Not recorded for this run');
  const chart = (side: 'ar' | 'nar') => {
    const inactive = mode !== 'joint' && mode !== side;
    const series = side === 'ar' ? (mode === 'ar' ? ['loss', 'eval', 'grad', 'lr'] : ['arCe', 'arKl', 'klStop', 'cursorCe', 'lr'])
      : (mode === 'nar' ? ['loss', 'grad', 'lr'] : ['narMse', 'narRecon', 'narDrift', 'lr']);
    const defaults = side === 'ar' ? (mode === 'ar' ? ['loss', 'eval', 'grad'] : ['arCe', 'arKl', 'klStop']) : (mode === 'nar' ? ['loss', 'grad'] : ['narMse', 'narRecon']);
    const hasData = steps.some(p => series.some(key => Number.isFinite(p[key === 'grad' ? 'gradNorm' : key as keyof ChartStepPoint]))) || (side === 'ar' && evals.length > 0);
    return <section className="min-w-0 rounded-lg border border-zinc-300/50 dark:border-white/10 p-2">
      <h4 className="text-xs font-semibold">{t(side === 'ar' ? 'trainingStudio.split.ar' : 'trainingStudio.split.nar', side === 'ar' ? 'AR (stage 1)' : 'NAR (stage 2)')}</h4>
      <p className="text-[11px] text-zinc-500 mb-2">{side === 'ar'
        ? t('trainingStudio.split.arBlurb', "Stage 1 writes the song's coarse token plan from lyrics and tags.")
        : t('trainingStudio.split.narBlurb', "Stage 2 turns stage 1's tokens into fine audio detail.")}</p>
      {inactive ? <p className="text-xs text-zinc-500">{t('trainingStudio.split.inactive', { stage: side === 'ar' ? 1 : 2, defaultValue: 'Stage {{stage}} not trained in this run' })}</p>
        : hasData ? <>
          <TrainingChart epochs={[]} steps={steps} milestones={milestones} evals={side === 'ar' ? evals : []} target={0}
            klTarget={side === 'ar' && options?.stopMode === 'kl' && typeof options.targetKl === 'number' ? options.targetKl : 0}
            klStopLabel={labels.klStop} config={{ series, defaults, more: series.filter(k => !defaults.includes(k)), labels, tooltips,
              xMax: last?.step, markers: side === 'ar' && frozen !== undefined ? [{ step: frozen, label: t('trainingStudio.split.frozen', 'Planner frozen') }] : [] }} />
          {defaults.filter(k => k !== 'klStop' && !(k === 'eval' && evals.length) && !steps.some(p => Number.isFinite(p[(k === 'grad' ? 'gradNorm' : k) as keyof ChartStepPoint]))).map(k => <p key={k} className="text-[10px] text-zinc-500">{labels[k as keyof typeof labels]}: {missing}</p>)}
        </> : <p className="text-xs text-zinc-500">{missing}</p>}
    </section>;
  };
  const pace = history.slice(-20).map(p => p.stepMs).filter((v): v is number => v !== undefined);
  const vram = vramMb ?? streamVram;
  return <div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">{chart('ar')}{chart('nar')}</div>
    <div className="mt-1 flex flex-wrap gap-x-3 text-[10px] text-zinc-500 tabular-nums">
      <span>{t('trainingStudio.split.steps', { step: last?.step ?? 0, total: totalSteps || job?.total || 0, defaultValue: 'Step {{step}} / {{total}}' })}</span>
      {mode === 'joint' && last?.gradNorm !== undefined && <span title={tooltips.grad}>{labels.grad} {last.gradNorm.toFixed(3)}</span>}
      {pace.length > 0 && <span>{t('trainingStudio.split.pace', { value: (pace.reduce((a, b) => a + b, 0) / pace.length / 1000).toFixed(2), defaultValue: 'Pace {{value}} s/step' })}</span>}
      {vram !== undefined && <span>{t('trainingStudio.split.vram', { value: (vram / 1024).toFixed(1), defaultValue: 'VRAM {{value}} GB' })}</span>}
      {frozen !== undefined && <span>{t('trainingStudio.split.frozenAt', { step: frozen, defaultValue: 'Planner frozen at step {{step}}' })}</span>}
      {options?.stopMode === 'loss' && <span>{t('trainingStudio.split.compositeStop', { value: lastStopMean?.toFixed(4) ?? missing, target: options.targetLoss, defaultValue: 'Composite loss stop mean {{value}} / target {{target}}' })}</span>}
    </div>
  </div>;
};
