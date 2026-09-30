// Yue2OptimisePanel.tsx — the YuE2 Optimise phase: optional measurements of
// the prepared album before training (Dataset-Calibrated Training). Every
// result lands in the dataset folder's _hotstep-optimisation.json, one section
// per measurement, read back here on load.
//
// First measurement: base loss, the base model's loss on every song before any
// training (server/src/services/training/yue2Optimise.ts).
import React, { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle2, Circle, Gauge, Loader2, XCircle } from 'lucide-react';
import { ParamLabel } from '../shared/ParamLabel';
import { useTrainingStore } from '../../stores/trainingStore';
import { getJob, getYue2Optimise, startYue2BaseLoss, type TrainingJobSummary, type Yue2OptimiseStatus } from '../../services/trainingApi';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

export const Yue2OptimisePanel: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const { t } = useTranslation();
  const setPhase = useTrainingStore(s => s.setPhase);
  const [status, setStatus] = useState<Yue2OptimiseStatus | null>(null);
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  const [error, setError] = useState('');

  const load = useCallback(() => { getYue2Optimise(datasetId).then(setStatus).catch(err => setError(err instanceof Error ? err.message : String(err))); }, [datasetId]);
  useEffect(() => { setStatus(null); setJob(null); setError(''); load(); }, [load]);

  // Follow a started job to its end, then read the saved file again.
  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return;
    const timer = window.setInterval(() => {
      getJob(job.id).then(next => {
        setJob(next);
        if (next.status === 'done') load();
        if (next.status === 'failed') setError(next.error || t('trainingStudio.yue2.optimise.failed', 'The base-loss pass failed.'));
      }).catch(() => { /* next tick */ });
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job, load, t]);

  const measure = async () => {
    setError('');
    try { const { jobId } = await startYue2BaseLoss(datasetId); setJob(await getJob(jobId)); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };

  if (!status) return <div className="flex items-center justify-center py-20 text-zinc-500 text-sm"><Loader2 size={18} className="animate-spin mr-2" /> …</div>;
  const ready = status.stages.cache && status.stages.codes && status.stages.sheets;
  const running = job?.status === 'queued' || job?.status === 'running';
  const baseLoss = status.data?.baseLoss;
  const stage = (done: boolean, label: string) => <span className="flex items-center gap-1.5 text-xs text-zinc-600 dark:text-zinc-300">
    {done ? <CheckCircle2 size={14} className="text-emerald-500" /> : <Circle size={14} className="text-zinc-400" />}{label}</span>;
  const rows = baseLoss ? [...baseLoss.items].sort((a, b) => b.arCe - a.arCe) : [];
  const maxAr = Math.max(...rows.map(r => r.arCe), 1e-9), maxNar = Math.max(...rows.map(r => r.narMse), 1e-9);

  return (
    <div className="flex flex-col gap-4">
      {error && <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
        <XCircle size={16} className="mt-0.5 flex-shrink-0" /><span className="min-w-0 break-words">{error}</span></div>}

      <div className={CARD}>
        <p className="text-xs text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.optimise.intro', 'Everything on this page is optional and changes nothing about training on its own. Results are saved with the dataset, in its folder, so they can be read again later:')} <span className="font-mono text-[11px] break-all">{status.file}</span></p>
        <div className="mt-3 flex flex-wrap items-center gap-4">
          {stage(status.stages.cache, t('trainingStudio.yue2.runAllStageName1', 'latent cache'))}
          {stage(status.stages.codes, t('trainingStudio.yue2.runAllStageName2', 'codes'))}
          {stage(status.stages.sheets, t('trainingStudio.yue2.runAllStageName3', 'lead sheets'))}
          {!ready && <button type="button" onClick={() => setPhase('preprocess')} className="text-[11px] font-semibold text-amber-600 dark:text-amber-400 hover:underline">
            {t('trainingStudio.yue2.optimise.goPrepare', 'Run these on the Prepare page first')}</button>}
        </div>
      </div>

      <div className={CARD}>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2"><Gauge size={15} className="text-amber-500" />
              <ParamLabel label={t('trainingStudio.yue2.optimise.baseLoss', 'Base model loss')}
                className="text-sm font-semibold text-zinc-900 dark:text-white"
                info={t('trainingStudio.yue2.optimise.baseLossInfo', 'How far this album is from what YuE2 already knows, measured before any training: the base model\'s loss on every song, with no adapter. Planner CE is how surprised the planner is by the album\'s songs, structure and lead sheets; decoder MSE is how surprised the decoder is by its sound (averaged over three noise levels on the same 60 s window training uses). Higher means further from the base. Dense, busy music reads higher too, so compare albums of a similar style. The same album measures the same every time. Takes about a minute for a 20-song album and runs the same preparation as training, reusing it when nothing changed.')} />
            </div>
            {baseLoss && <p className="text-[11px] text-zinc-500 mt-1">{t('trainingStudio.yue2.optimise.measured', 'Measured {{when}} on {{base}}{{companion}}', { when: new Date(baseLoss.measuredAt).toLocaleString(), base: baseLoss.base, companion: baseLoss.companion ? t('trainingStudio.yue2.optimise.withCompanion', ' with the companion decoder') : '' })}</p>}
          </div>
          <button type="button" onClick={() => void measure()} disabled={!ready || running}
            className="shrink-0 rounded-lg bg-amber-500 px-3 py-2 text-xs font-semibold text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5">
            {running && <Loader2 size={13} className="animate-spin" />}
            {running ? (job?.phase === 'measuring' ? t('trainingStudio.yue2.optimise.progress', 'Measuring {{done}} of {{total}}', { done: job.done, total: job.total }) : t('trainingStudio.yue2.optimise.preparing', 'Preparing…'))
              : baseLoss ? t('trainingStudio.yue2.optimise.remeasure', 'Measure again') : t('trainingStudio.yue2.optimise.measure', 'Measure')}
          </button>
        </div>

        {baseLoss && <>
          <div className="mt-3 flex flex-wrap gap-6">
            <div><div className="text-[10px] uppercase tracking-wider text-zinc-500">{t('trainingStudio.yue2.optimise.plannerCe', 'Planner CE')}</div>
              <div className="text-lg font-bold font-mono text-zinc-900 dark:text-white">{baseLoss.summary.arCeMean.toFixed(3)}</div></div>
            <div><div className="text-[10px] uppercase tracking-wider text-zinc-500">{t('trainingStudio.yue2.optimise.decoderMse', 'Decoder MSE')}</div>
              <div className="text-lg font-bold font-mono text-zinc-900 dark:text-white">{baseLoss.summary.narMseMean.toFixed(3)}</div></div>
            <div><div className="text-[10px] uppercase tracking-wider text-zinc-500">{t('trainingStudio.yue2.optimise.songs', 'Songs')}</div>
              <div className="text-lg font-bold font-mono text-zinc-900 dark:text-white">{baseLoss.summary.items}</div></div>
          </div>
          <p className="mt-3 text-[11px] text-zinc-500">{t('trainingStudio.yue2.optimise.perSong', 'Per song, most unfamiliar planner first. A song far out of line with the rest is worth a listen and a look at its caption and lyrics before training.')}</p>
          <div className="mt-2 flex flex-col gap-1">
            {rows.map(r => <div key={r.file} className="grid grid-cols-[minmax(0,1fr)_120px_120px] items-center gap-3 text-[11px]">
              <span className="truncate text-zinc-700 dark:text-zinc-300" title={r.file}>{r.file}</span>
              <span className="flex items-center gap-1.5"><span className="h-1.5 rounded bg-amber-500/70" style={{ width: `${Math.round(60 * r.arCe / maxAr)}px` }} /><span className="font-mono text-zinc-500">{r.arCe.toFixed(3)}</span></span>
              <span className="flex items-center gap-1.5"><span className="h-1.5 rounded bg-sky-500/70" style={{ width: `${Math.round(60 * r.narMse / maxNar)}px` }} /><span className="font-mono text-zinc-500">{r.narMse.toFixed(3)}</span></span>
            </div>)}
          </div>
        </>}
      </div>
    </div>
  );
};

export default Yue2OptimisePanel;
