import React, { useMemo, useState } from 'react';
import { Loader2, Play, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import {
  getJob, getYue2ArStatus, getYue2AitkPrepare,
  startYue2Align, startYue2Preprocess, startYue2Sheet,
  startYue2Stems, startYue2Tokenize, startYue2JointTrain,
  type Yue2JointPreviewOptions,
} from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';

type RowState = { status: 'waiting' | 'running' | 'done' | 'failed' | 'cancelled'; phase?: string; error?: string };

const terminal = new Set(['done', 'failed', 'cancelled']);
const waitFor = async (jobId: string, onPhase: (p: string) => void): Promise<void> => {
  for (;;) {
    const job = await getJob(jobId);
    onPhase(job.phase || job.kind);
    if (terminal.has(job.status)) {
      if (job.status !== 'done') throw new Error(job.error || `Job ${job.status}`);
      return;
    }
    await new Promise(resolve => window.setTimeout(resolve, 1200));
  }
};

interface Props { open: boolean; onClose: () => void }

/** AITK's own multi-dataset queue. It deliberately calls the individual
 * cache endpoints and the joint endpoints directly; the Legacy seven-stage
 * queue has different training semantics and must not be reused here. */
export const Yue2AitkBatchWizard: React.FC<Props> = ({ open, onClose }) => {
  const { t } = useTranslation();
  const datasets = useTrainingStore(s => s.datasets);
  const activeJob = useTrainingStore(s => s.activeJob);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [lyricTiming, setLyricTiming] = useState(true);
  const [steps, setSteps] = useState(400);
  const [saveEvery, setSaveEvery] = useState(50);
  const [seed, setSeed] = useState(42);
  const [device, setDevice] = useState('CUDA0');
  const [running, setRunning] = useState(false);
  const [cancelled, setCancelled] = useState(false);
  const [rows, setRows] = useState<Record<string, RowState>>({});

  const selected = useMemo(() => datasets.filter(d => checked[d.id]), [datasets, checked]);
  const busy = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const signal = React.useRef({ cancelled: false });

  const update = (id: string, next: RowState) => setRows(previous => ({ ...previous, [id]: next }));
  const runStage = async (id: string, phase: string, start: () => Promise<{ jobId: string }>) => {
    if (signal.current.cancelled) throw new Error('Batch cancelled');
    update(id, { status: 'running', phase });
    const result = await start();
    await waitFor(result.jobId, p => update(id, { status: 'running', phase: `${phase}: ${p}` }));
    if (signal.current.cancelled) throw new Error('Batch cancelled');
  };

  const run = async () => {
    if (running || busy || selected.length === 0) return;
    signal.current = { cancelled: false };
    setCancelled(false); setRunning(true);
    for (const dataset of selected) {
      if (signal.current.cancelled) {
        update(dataset.id, { status: 'cancelled', phase: 'cancelled' });
        continue;
      }
      update(dataset.id, { status: 'running', phase: 'checking caches' });
      try {
        if (signal.current.cancelled) throw new Error('Batch cancelled');
        const ar = await getYue2ArStatus(dataset.id);
        if (signal.current.cancelled) throw new Error('Batch cancelled');
        if (!ar.stages.preprocess.done) await runStage(dataset.id, 'latent cache', () => startYue2Preprocess(dataset.id, {}));
        if (!ar.stages.tokenize.done) await runStage(dataset.id, 'codes', () => startYue2Tokenize(dataset.id, {}));
        if (!ar.stages.sheet.done) await runStage(dataset.id, 'lead sheets', () => startYue2Sheet(dataset.id, {}));
        if (lyricTiming && ar.stages.align.stemsReady < ar.stages.align.stemsNeeded) {
          await runStage(dataset.id, 'vocal stems', () => startYue2Stems(dataset.id, {}));
        }
        const refreshed = lyricTiming ? await getYue2ArStatus(dataset.id) : ar;
        if (lyricTiming && !refreshed.stages.align.done) {
          await runStage(dataset.id, 'lyric alignment', () => startYue2Align(dataset.id, {}));
        }
        const prep = await getYue2AitkPrepare(dataset.id);
        if (signal.current.cancelled) throw new Error('Batch cancelled');
        const defaults = prep.defaults;
        if (!defaults?.legacyManifest || !defaults.checkpoint || !defaults.tokenizer || !defaults.models?.vae || !defaults.models.semantic || !defaults.models.sheetsage) {
          throw new Error('Joint Training preparation defaults are incomplete for this dataset');
        }
        update(dataset.id, { status: 'running', phase: 'joint training' });
        const preview: Yue2JointPreviewOptions = { enabled: false, everySteps: saveEvery, seconds: 40, seed: 424242, previewMaxFrames: 0, baseline: false, control: false };
        const train = await startYue2JointTrain(dataset.id, {
          trainingMethod: 'aitk', checkpoint: defaults.checkpoint, dataset: '', autoPrepare: true,
          output: '', steps, saveEvery, seed, device, lyricTiming,
          alignmentEnabled: lyricTiming, cursorWeight: lyricTiming ? 0.08 : 0, preview,
        });
        await waitFor(train.jobId, (p: string) => update(dataset.id, { status: 'running', phase: `joint training: ${p}` }));
        if (signal.current.cancelled) throw new Error('Batch cancelled');
        update(dataset.id, { status: 'done', phase: 'complete' });
      } catch (error) {
        if (signal.current.cancelled) update(dataset.id, { status: 'cancelled', phase: 'cancelled' });
        else update(dataset.id, { status: 'failed', error: error instanceof Error ? error.message : String(error) });
        if (signal.current.cancelled) continue;
      }
    }
    setRunning(false);
  };

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={e => { if (e.target === e.currentTarget && !running) onClose(); }}>
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-card p-5 shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.yue2.aitkBatch.title', 'Train several datasets')}</h3>
          <button type="button" onClick={onClose} disabled={running}><X size={16} /></button>
        </div>
        <p className="text-[11px] text-zinc-500 mb-4">{t('trainingStudio.yue2.aitkBatch.help', 'Each dataset runs its caches, preparation, and joint AR + NAR training in order. Settings are shared; output folders are unique per dataset.')}</p>
        <div className="grid grid-cols-2 gap-2 mb-4">
          <label className="text-xs text-zinc-600 dark:text-zinc-400">Steps<input disabled={running} type="number" min={1} value={steps} onChange={e => setSteps(Math.max(1, Number(e.target.value) || 1))} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
          <label className="text-xs text-zinc-600 dark:text-zinc-400">Save every<input disabled={running} type="number" min={1} value={saveEvery} onChange={e => setSaveEvery(Math.max(1, Number(e.target.value) || 1))} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
          <label className="text-xs text-zinc-600 dark:text-zinc-400">Seed<input disabled={running} type="number" value={seed} onChange={e => setSeed(Number(e.target.value) || 0)} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
          <label className="text-xs text-zinc-600 dark:text-zinc-400">CUDA device<input disabled={running} value={device} onChange={e => setDevice(e.target.value || 'CUDA0')} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
        </div>
        <label className="flex items-center gap-2 text-xs mb-4"><input disabled={running} type="checkbox" checked={lyricTiming} onChange={e => setLyricTiming(e.target.checked)} className="accent-amber-500" /> Include lyric timing stems and alignment</label>
        <div className="rounded-lg border border-zinc-200 dark:border-white/10 divide-y divide-zinc-200 dark:divide-white/10 mb-4">
          {datasets.map(ds => { const row = rows[ds.id]; return <label key={ds.id} className="flex items-center gap-2 p-2 text-xs"><input type="checkbox" checked={!!checked[ds.id]} onChange={e => setChecked(previous => ({ ...previous, [ds.id]: e.target.checked }))} disabled={running} className="accent-amber-500" /><span className="flex-1 truncate">{ds.name}</span>{row && <span className={row.status === 'failed' ? 'text-red-500' : row.status === 'done' ? 'text-emerald-500' : 'text-zinc-500'}>{row.error || row.phase}</span>}</label>; })}
          {datasets.length === 0 && <div className="p-3 text-xs text-zinc-500">No datasets available.</div>}
        </div>
        {cancelled && <p className="text-xs text-amber-600 mb-2">Batch cancellation requested; the current server job may finish before the next dataset stops.</p>}
        <div className="flex justify-end gap-2"><button type="button" onClick={onClose} disabled={running} className="px-3 py-2 text-xs">Close</button>{running ? <button type="button" onClick={() => { signal.current.cancelled = true; setCancelled(true); }} className="px-3 py-2 rounded-lg bg-red-500/15 text-red-600 text-xs">Cancel after current job</button> : <button type="button" onClick={() => void run()} disabled={busy || selected.length === 0} className="px-3 py-2 rounded-lg bg-amber-500 text-black text-xs font-semibold disabled:opacity-40"><Play size={13} className="inline mr-1" />Start batch</button>}</div>
        {running && <Loader2 size={15} className="animate-spin text-amber-500 mt-3" />}
      </div>
    </div>
  );
};
