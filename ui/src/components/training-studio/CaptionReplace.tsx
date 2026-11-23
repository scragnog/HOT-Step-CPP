import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { CaptionReplacePreview } from '../../../../server/src/contracts/captionReplace';
import { replaceCaptions } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { queryClient } from '../../queries/client';
import { TD } from '../../queries/training-data/queries';
import { resourceKey } from '../../queries/keys';
import { keysPlan, writeAndInvalidate } from '../../queries/writes';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';

export const CaptionReplace: React.FC<{
  datasetId: string; worker: string | null; sampleIds: string[]; onClose: () => void;
}> = props => {
  const { t } = useTranslation();
  // This command owns the selection it opened with, including its worker.
  const [target] = useState(() => ({ datasetId: props.datasetId, worker: props.worker, sampleIds: props.sampleIds }));
  const [find, setFind] = useState('');
  const [replacement, setReplacement] = useState('');
  const [preview, setPreview] = useState<CaptionReplacePreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);
  const run = async (apply: boolean) => {
    setBusy(true); setError('');
    try {
      const command = () => replaceCaptions(target.datasetId, {
        sampleIds: target.sampleIds, find, replacement,
        ...(apply && preview ? { previewToken: preview.previewToken } : {}),
      }, { worker: target.worker, silent: true });
      if (apply) {
        const current = useTrainingStore.getState();
        if (current.trainingWorker === target.worker && current.selectedDatasetId === target.datasetId
          && (Object.keys(current.pendingEdits).length || current.savingSampleIds.size)) {
          throw new Error(t('trainingStudio.replace.saveFirst'));
        }
        await writeAndInvalidate(queryClient, command, keysPlan(
          resourceKey(TD.dataset, { worker: target.worker }, target.datasetId),
          resourceKey(TD.sampleYue2, { worker: target.worker }, target.datasetId),
        ));
        setDone(true);
        const state = useTrainingStore.getState();
        if (state.trainingWorker === target.worker && state.selectedDatasetId === target.datasetId) await state.openDataset(target.datasetId);
      } else setPreview(await command());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPreview(null);
    } finally { setBusy(false); }
  };
  const inputClass = 'w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-zinc-900 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100';
  const buttonClass = 'rounded-lg px-3 py-2 text-xs font-semibold bg-amber-500/20 text-amber-700 dark:text-amber-400 disabled:opacity-40';
  return (
    <section aria-label={t('trainingStudio.replace.title')} className="rounded-xl border border-amber-500/30 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">{t('trainingStudio.replace.title')}</h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">{t('trainingStudio.replace.scope', { count: target.sampleIds.length })}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1">
          <ParamLabel label={t('trainingStudio.replace.find')} info={t('trainingStudio.replace.findInfo')} />
          <input value={find} disabled={busy || done} onChange={e => { setFind(e.target.value); setPreview(null); }} className={inputClass} />
        </label>
        <label className="space-y-1">
          <ParamLabel label={t('trainingStudio.replace.with')} info={t('trainingStudio.replace.withInfo')} />
          <input value={replacement} disabled={busy || done} onChange={e => { setReplacement(e.target.value); setPreview(null); }} className={inputClass} />
        </label>
      </div>
      {preview && <div aria-live="polite" className="text-xs text-zinc-700 dark:text-zinc-300">
        <p>{t(done ? 'trainingStudio.replace.done' : 'trainingStudio.replace.count', { tracks: preview.tracks, files: preview.files.length })}</p>
        <ul className="max-h-48 overflow-auto list-disc pl-5 mt-2">{preview.files.map(f => <li key={f.file} className="break-all">{f.file}</li>)}</ul>
      </div>}
      {error && <p role="alert" className="text-xs text-red-600 dark:text-red-400">{error}</p>}
      <div className="flex gap-2">
        {!done && <button className={buttonClass} disabled={busy || !find || find === replacement} onClick={() => void run(false)}>{t('trainingStudio.replace.preview')}</button>}
        {!done && preview && <button className={buttonClass} disabled={busy || !preview.files.length} onClick={() => void run(true)}>{t('trainingStudio.replace.apply')}</button>}
        <button className={buttonClass} disabled={busy} onClick={props.onClose}>{t(done ? 'common.close' : 'common.cancel')}</button>
      </div>
    </section>
  );
};
