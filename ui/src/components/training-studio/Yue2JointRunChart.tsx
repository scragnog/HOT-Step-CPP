// Yue2JointRunChart.tsx — the live chart and stats line of one YuE2 joint
// training job, fed from its event stream. The Refine tab's view of a run;
// Yue2AitkTrainCard keeps its own richer copy (logs, milestones, previews).
//
// ponytail: a focused copy of the card's stream handling. Fold the card onto
// this component when it next changes shape.
import React, { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { TrainingChart } from './TrainingChart';
import { jobStreamUrl, type TrainingJobSummary, type TrainingStreamEvent } from '../../services/trainingApi';
import { formatDurationMs } from '../../utils/trainingEta';

type Point = { step: number; ep: number; loss?: number; arKl?: number; narMse?: number; narRecon?: number; frozen?: boolean; gradNorm?: number; stepMs?: number; elapsedMs?: number; ma5?: number; klStop?: number };
const CAP = 2000;
const KL_TREND_WINDOW = 30;

function klTrend(kls: number[]): number | null {
  const n = KL_TREND_WINDOW;
  if (kls.length < n) return null;
  const v = kls.slice(-n), ym = v.reduce((s, x) => s + x, 0) / n, xm = (n - 1) / 2;
  let sxy = 0, sxx = 0;
  v.forEach((y, i) => { sxy += (i - xm) * (y - ym); sxx += (i - xm) ** 2; });
  return ym + (sxy / sxx) * (n - 1 - xm);
}

export const Yue2JointRunChart: React.FC<{ job: TrainingJobSummary | null; totalSteps: number; klTarget?: number }> = ({ job, totalSteps, klTarget }) => {
  const { t } = useTranslation();
  const [history, setHistory] = useState<Point[]>([]);
  useEffect(() => {
    setHistory([]);
    if (!job?.id) return;
    const stream = new EventSource(jobStreamUrl(job.id));
    stream.onmessage = event => {
      try {
        const item = JSON.parse(event.data) as TrainingStreamEvent;
        if (item.type !== 'metric' || item.metric !== 'step' || typeof item.step !== 'number') return;
        setHistory(previous => {
          const existing = previous.find(p => p.step === item.step);
          const prior = previous.filter(p => p.step !== item.step);
          if (typeof item.narRecon === 'number' && item.loss === undefined && item.stepMs === undefined) {
            return [...prior, { ...(existing ?? { step: item.step!, ep: item.step! }), narRecon: item.narRecon }].sort((a, b) => a.step - b.step).slice(-CAP);
          }
          const next = [...prior, { ...(existing?.narRecon !== undefined ? { narRecon: existing.narRecon } : {}), step: item.step!, ep: item.step!,
            ...(typeof item.loss === 'number' && Number.isFinite(item.loss) ? { loss: item.loss } : {}),
            ...(typeof item.narMse === 'number' ? { narMse: item.narMse } : {}),
            ...(item.plannerFrozen ? { frozen: true } : {}),
            ...(typeof item.arKl === 'number' ? { arKl: item.arKl } : {}),
            ...(typeof item.gradNorm === 'number' ? { gradNorm: item.gradNorm } : {}),
            ...(typeof item.trainMs === 'number' ? { elapsedMs: item.trainMs } : {}),
            ...(typeof item.stepMs === 'number' ? { stepMs: item.stepMs } : {}) }].sort((a, b) => a.step - b.step).slice(-CAP);
          const mean = (pts: Point[]) => { const v = pts.map(p => p.loss).filter((x): x is number => typeof x === 'number'); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : undefined; };
          return next.map((p, i) => p.loss === undefined ? p : { ...p, ma5: mean(next.slice(Math.max(0, i - 4), i + 1)) });
        });
      } catch { /* malformed event */ }
    };
    return () => stream.close();
  }, [job?.id]);

  const chartSteps = useMemo(() => {
    const kls: number[] = [];
    return history.map(p => {
      if (typeof p.arKl === 'number') kls.push(p.arKl);
      const reading = typeof p.arKl === 'number' ? klTrend(kls) : null;
      const q = { ...p, loss: p.loss ?? Number.NaN };
      return reading !== null ? { ...q, klStop: reading } : q;
    });
  }, [history]);
  if (history.length < 2) return job ? <p className="text-[11px] text-zinc-500">{t('trainingStudio.refine.waitingSteps', 'Waiting for the first steps…')}</p> : null;
  const last = history[history.length - 1];
  const paced = history.slice(-20).filter(p => p.stepMs !== undefined);
  const pace = paced.length ? paced.reduce((s, p) => s + (p.stepMs ?? 0), 0) / paced.length / 1000 : undefined;
  const kls = history.map(p => p.arKl).filter((v): v is number => typeof v === 'number');
  const reading = klTrend(kls);
  return (
    <div>
      <TrainingChart epochs={[]} steps={chartSteps} milestones={[]} target={0} klTarget={klTarget ?? 0} klStopLabel="KL trend (stop reading)" maxEpochs={totalSteps} />
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] tabular-nums text-zinc-500">
        <span>{last.step} / {totalSteps} steps</span>
        {reading !== null && <span>KL trend {reading.toFixed(3)}{klTarget ? ` of ${klTarget}` : ''}</span>}
        {last.narRecon !== undefined && <span>decoder recon {last.narRecon.toFixed(4)}</span>}
        {last.elapsedMs !== undefined && <span>elapsed {formatDurationMs(last.elapsedMs)}</span>}
        {pace !== undefined && <span>pace {pace.toFixed(2)}s/step</span>}
        {job?.phase && <span>{job.phase}</span>}
      </div>
    </div>
  );
};

export default Yue2JointRunChart;
