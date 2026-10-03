import React, { useCallback, useEffect, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  cancelWorkerUpdate, getWorkerUpdate, listTrainingWorkers, startWorkerUpdate,
  type TrainingWorkerStatus, type WorkerUpdateStatus,
} from '../../services/trainingApi';

export const WorkersPanel: React.FC = () => {
  const { t } = useTranslation();
  const [workers, setWorkers] = useState<TrainingWorkerStatus[]>([]);
  const [updates, setUpdates] = useState<Record<string, WorkerUpdateStatus | null>>({});
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    try {
      const rows = await listTrainingWorkers();
      setWorkers(rows);
      const jobs = await Promise.all(rows.map(w => getWorkerUpdate(w.name).catch(() => null)));
      setUpdates(Object.fromEntries(rows.map((w, i) => [w.name, jobs[i]])));
      setError('');
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  }, []);
  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(id);
  }, [refresh]);
  const update = async (name: string) => {
    try { const job = await startWorkerUpdate(name); setUpdates(old => ({ ...old, [name]: job })); setError(''); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };
  const cancel = async (name: string) => {
    try { await cancelWorkerUpdate(name); await refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };
  return (
    <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-3 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">{t('trainingStudio.workers.panel', 'Workers')}</h3>
        <button type="button" onClick={() => void refresh()} aria-label={t('trainingStudio.workers.refresh', 'Refresh workers')} className="text-zinc-500 hover:text-amber-500"><RefreshCw size={14} /></button>
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}
      {workers.map(w => {
        const updateJob = updates[w.name];
        const busy = updateJob && !['done', 'failed', 'cancelled'].includes(updateJob.status);
        const badge = !w.online ? t('trainingStudio.workers.offlineBadge', 'Offline')
          : !w.commit ? t('trainingStudio.workers.legacyBadge', 'Update API unavailable')
          : w.relation === 'current' ? t('trainingStudio.workers.currentBadge', 'Current')
          : w.relation === 'behind' ? t('trainingStudio.workers.behindBadge', 'Behind by {{count}} commits', { count: w.behind ?? 0 })
          : t('trainingStudio.workers.divergedBadge', 'Diverged');
        return <div key={w.name} className="rounded-lg border border-zinc-200 dark:border-white/10 p-3 text-xs space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <strong className="text-zinc-800 dark:text-zinc-200">{w.name}</strong>
            <span className={w.relation === 'current' ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'}>{badge}</span>
            {w.dirty && <span className="text-amber-600 dark:text-amber-400">{t('trainingStudio.workers.dirty', 'Local changes')}</span>}
            <button type="button" onClick={() => void update(w.name)}
              disabled={!w.online || w.relation !== 'behind' || !w.idle || !!busy}
              className="ml-auto rounded-md bg-amber-600 px-2 py-1 font-semibold text-white disabled:opacity-40">
              {busy ? <Loader2 size={13} className="animate-spin" /> : t('trainingStudio.workers.update', 'Update')}
            </button>
            {updateJob?.cancellable && <button type="button" onClick={() => void cancel(w.name)} className="text-red-500 hover:underline">{t('trainingStudio.workers.cancel', 'Cancel')}</button>}
          </div>
          {w.online && <div className="flex flex-wrap gap-x-4 gap-y-1 text-zinc-600 dark:text-zinc-400">
            <span>Git {w.commit?.slice(0, 8) ?? 'unknown'}</span>
            <span>Engine {w.engineVersion || w.engine || 'unknown'}{w.engineBuiltAt ? ` · built ${new Date(w.engineBuiltAt).toLocaleString()}` : ''}</span>
            <span>GPU {w.gpu ? `${w.gpu.memoryUsedMiB} MiB · ${w.gpu.utilization}%` : 'unavailable'}</span>
            <span>{w.job ? `${w.job.kind}${w.job.dataset ? ` · ${w.job.dataset}` : ''} · ${w.job.status} · ${w.job.done}/${w.job.total}` : t('trainingStudio.workers.idle', 'Idle')}</span>
          </div>}
          {w.error && <p className="text-red-500">{w.error}</p>}
          {updateJob && <div>
            <span className="font-semibold text-zinc-700 dark:text-zinc-300">Update: {updateJob.status}</span>
            {updateJob.error && <span className="ml-2 text-red-500">{updateJob.error}</span>}
            {!!updateJob.lines.length && <pre className="mt-1 max-h-44 overflow-auto whitespace-pre-wrap rounded bg-zinc-100 dark:bg-black/30 p-2 text-[11px] text-zinc-700 dark:text-zinc-300">{updateJob.lines.join('\n')}</pre>}
          </div>}
        </div>;
      })}
    </div>
  );
};
