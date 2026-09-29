// TrainingWorkerBar.tsx — which machine the Training Studio trains on.
//
// Renders nothing unless Settings lists a training worker (TRAINING_WORKERS).
// "Train on" points the whole studio at the worker through this PC's proxy,
// so its batches, ladders, previews and rung scores are the worker's. The bar
// also shows this PC's dispatch line (captioning here, pushing, queued there)
// and fetches finished adapters back to this PC's presets.

import React, { useEffect, useState } from 'react';
import { Download, Loader2, Server } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { useTrainingStore } from '../../stores/trainingStore';
import {
  getWorkerDispatch, listTrainingWorkers, pullWorkerAdapters,
  type TrainingWorkerStatus, type WorkerDispatch,
} from '../../services/trainingApi';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-3';
const HERE = '__here__';
const mb = (bytes: number) => `${(bytes / 1048576).toFixed(0)} MB`;

export const TrainingWorkerBar: React.FC = () => {
  const { t } = useTranslation();
  const worker = useTrainingStore(s => s.trainingWorker);
  const setWorker = useTrainingStore(s => s.setTrainingWorker);
  const [workers, setWorkers] = useState<TrainingWorkerStatus[]>([]);
  const [dispatches, setDispatches] = useState<WorkerDispatch[]>([]);
  const [pulling, setPulling] = useState(false);
  const [note, setNote] = useState('');

  useEffect(() => { void listTrainingWorkers().then(setWorkers).catch(() => setWorkers([])); }, []);
  // A dispatch captions and pushes here for minutes before the worker's batch
  // shows it, so its line is polled while any dispatch is still running.
  const running = dispatches.some(d => d.running);
  useEffect(() => {
    if (!workers.length) return;
    const tick = () => void Promise.all(workers.map(w => getWorkerDispatch(w.name).catch(() => null)))
      .then(all => setDispatches(all.filter((d): d is WorkerDispatch => !!d)));
    tick();
    const id = window.setInterval(tick, running ? 3000 : 15000);
    return () => window.clearInterval(id);
  }, [workers, running]);

  // The studio switches to the worker as soon as a dispatch starts, before
  // anything has arrived there: reload its datasets and batch as each lands.
  const loadDatasets = useTrainingStore(s => s.loadDatasets);
  const loadBatches = useTrainingStore(s => s.loadYue2Batches);
  const queuedHere = dispatches.find(d => d.worker === worker)?.items.filter(i => i.status === 'queued').length ?? 0;
  useEffect(() => {
    if (!queuedHere) return;
    void loadDatasets(); void loadBatches();
  }, [queuedHere, loadDatasets, loadBatches]);

  if (!workers.length && !worker) return null;

  const current = workers.find(w => w.name === worker);
  const pull = async () => {
    if (!worker) return;
    setPulling(true); setNote('');
    try {
      const got = await pullWorkerAdapters(worker);
      const fetched = got.filter(g => g.status === 'fetched');
      setNote(fetched.length
        ? t('trainingStudio.workers.fetched', 'Fetched {{names}} ({{size}}) and linked them to the album presets here.', { names: fetched.map(g => g.slug).join(', '), size: mb(fetched.reduce((n, g) => n + g.bytes, 0)) })
        : t('trainingStudio.workers.upToDate', 'Nothing new: every linked adapter on {{worker}} is already here.', { worker }));
    } catch (err) { setNote(err instanceof Error ? err.message : String(err)); }
    finally { setPulling(false); }
  };

  return (
    <div className={`${CARD} flex flex-col gap-2`}>
      <div className="flex items-center gap-3 flex-wrap">
        <Server size={15} className="text-amber-500 flex-shrink-0" />
        <ParamLabel
          label={t('trainingStudio.workers.trainOn', 'Train on')}
          className="text-xs font-semibold text-zinc-700 dark:text-zinc-300"
          info={t('trainingStudio.workers.trainOnInfo', 'This PC, or a training worker listed in Settings. On a worker, everything below (datasets, batches, training charts, previews and rung scores) is the worker\'s: you listen and score here while its GPU trains. Datasets reach it through "Run on" when you start a YuE2 batch from this PC.')}
        />
        <StyledSelect
          accent="amber"
          value={worker ?? HERE}
          onChange={value => void setWorker(value === HERE ? null : value)}
          options={[
            { value: HERE, label: t('trainingStudio.workers.thisPc', 'This PC') },
            ...workers.map(w => ({ value: w.name, label: w.online ? w.name : `${w.name} (offline)`, hint: w.url })),
          ]}
          className="w-56"
        />
        {current && !current.online && <span className="text-[11px] text-red-500">{t('trainingStudio.workers.offline', 'Unreachable: {{error}}', { error: current.error ?? '' })}</span>}
        {current?.online && current.versionMatch === false && <span className="text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.workers.versionMismatch', 'Worker runs {{version}}; update it to the same version as this PC.', { version: current.version })}</span>}
        {worker && (
          <button type="button" onClick={() => void pull()} disabled={pulling}
            className="ml-auto flex items-center gap-1 text-[11px] font-semibold text-amber-600 dark:text-amber-400 hover:underline disabled:opacity-50">
            {pulling ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
            {t('trainingStudio.workers.pull', 'Fetch finished adapters')}
          </button>
        )}
      </div>
      {worker && <p className="text-[11px] text-zinc-500">
        {t('trainingStudio.workers.viewHint', 'Showing the datasets on {{worker}}. To send more, switch Train on to This PC, pick them for Train multiple, and choose Run on: {{worker}}.', { worker })}
      </p>}
      {note && <p className="text-[11px] text-zinc-500 break-words">{note}</p>}
      {dispatches.filter(d => d.running || d.items.some(i => i.status === 'failed')).map(d => (
        <div key={d.worker} className="flex flex-col gap-0.5 text-[11px] text-zinc-600 dark:text-zinc-400">
          <span className="font-semibold">{t('trainingStudio.workers.dispatchTitle', 'Sending to {{worker}}', { worker: d.worker })}</span>
          {d.items.map(i => (
            <span key={i.datasetId} className={i.status === 'failed' ? 'text-red-500 break-words' : ''}>
              {i.name}: {i.status}{i.status === 'queued' && i.bytes ? ` · ${mb(i.bytes)} sent` : ''}{i.error ? ` · ${i.error}` : ''}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
};

export default TrainingWorkerBar;
