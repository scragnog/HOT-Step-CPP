import React, { useEffect, useMemo, useState } from 'react';
import { Loader2, Play, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import {
  getJob, getYue2ArStatus, getYue2AitkPrepare,
  startYue2Align, startYue2Preprocess, startYue2Sheet,
  startYue2Stems, startYue2Tokenize, startYue2JointTrain,
  type Yue2JointPreviewOptions, type Yue2JointTrainRequest,
} from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { readYue2JointPresets, type Yue2JointPreset } from './yue2JointPresets';

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
  const [presets, setPresets] = useState<Yue2JointPreset[]>([]);
  const [presetName, setPresetName] = useState('');
  const [formError, setFormError] = useState('');
  const [running, setRunning] = useState(false);
  const [cancelled, setCancelled] = useState(false);
  const [rows, setRows] = useState<Record<string, RowState>>({});

  const selected = useMemo(() => datasets.filter(d => checked[d.id]), [datasets, checked]);
  const busy = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const signal = React.useRef({ cancelled: false });
  const selectedPreset = presets.find(preset => preset.name === presetName);

  useEffect(() => {
    if (open) {
      setPresets(readYue2JointPresets());
      setPresetName('');
      setFormError('');
    }
  }, [open]);

  const choosePreset = (name: string) => {
    setPresetName(name);
    setFormError('');
    const preset = presets.find(item => item.name === name);
    const settings = preset?.settings;
    if (!settings) return;
    if (typeof settings.steps === 'number') setSteps(settings.steps);
    if (typeof settings.saveEvery === 'number') setSaveEvery(settings.saveEvery);
    if (typeof settings.seed === 'number') setSeed(settings.seed);
    if (typeof settings.device === 'string') setDevice(settings.device);
    if (preset.version === 2 && typeof settings.lyricTiming === 'boolean') setLyricTiming(settings.lyricTiming);
  };

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
    if (!Number.isInteger(steps) || steps < 1 || !Number.isInteger(saveEvery) || saveEvery < 1 || saveEvery > steps) {
      setFormError('Save every must be between 1 and the total step count.');
      return;
    }
    const recipe = selectedPreset?.settings ?? {};
    if (recipe.stopMode === 'kl' && !(typeof recipe.targetKl === 'number' && recipe.targetKl > 0)) {
      setFormError('The selected preset needs an AR KL target above 0.');
      return;
    }
    if (recipe.stopMode === 'loss' && !(typeof recipe.targetLoss === 'number' && recipe.targetLoss > 0)) {
      setFormError('The selected preset needs a target loss above 0.');
      return;
    }
    setFormError('');
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
        // Same rule as the Training Studio stage chain: the cache must carry
        // the sidecar captions and lyrics (mode 'ace'), and one built without
        // them is rebuilt, not skipped — latents are cache hits, only the text
        // is refilled. A cache with no lyrics makes the align stage skip every
        // source and the AR train on empty prefixes.
        if (!ar.stages.preprocess.done || ar.stages.preprocess.captionModeOk === false) {
          await runStage(dataset.id, 'latent cache', () => startYue2Preprocess(dataset.id, { captionMode: 'ace' }));
        }
        if (!ar.stages.tokenize.done) await runStage(dataset.id, 'codes', () => startYue2Tokenize(dataset.id, {}));
        if (!ar.stages.sheet.done) await runStage(dataset.id, 'lead sheets', () => startYue2Sheet(dataset.id, {}));
        if (lyricTiming) {
          // Before preprocessing, stemsNeeded is zero. The status captured at
          // the start of this loop cannot decide whether stems may be skipped.
          let refreshed = await getYue2ArStatus(dataset.id);
          const needed = refreshed.stages.align.stemsNeeded;
          if (needed < 1) throw new Error('The latent cache contains no songs to align.');
          if (refreshed.stages.align.stemsReady < needed) {
            await runStage(dataset.id, 'vocal stems', () => startYue2Stems(dataset.id, {}));
            refreshed = await getYue2ArStatus(dataset.id);
          }
          if (refreshed.stages.align.stemsReady < needed) {
            throw new Error(`Vocal stems are incomplete (${refreshed.stages.align.stemsReady}/${needed}); retry separation before alignment.`);
          }
          if (!refreshed.stages.align.done) {
            await runStage(dataset.id, 'lyric alignment', () => startYue2Align(dataset.id, {}));
            refreshed = await getYue2ArStatus(dataset.id);
          }
          if (!refreshed.stages.align.done) throw new Error('Lyric alignment produced no cursor spans.');
        }
        const prep = await getYue2AitkPrepare(dataset.id);
        if (signal.current.cancelled) throw new Error('Batch cancelled');
        const defaults = prep.defaults;
        if (!defaults?.legacyManifest || !defaults.checkpoint || !defaults.tokenizer || !defaults.models?.vae || !defaults.models.semantic || !defaults.models.sheetsage) {
          throw new Error('Joint Training preparation defaults are incomplete for this dataset');
        }
        update(dataset.id, { status: 'running', phase: 'joint training' });
        // Preset preview timing is shared; caption/lyrics/song overrides belong
        // to one dataset and must not leak into the rest of the batch.
        const preview: Yue2JointPreviewOptions = {
          enabled: recipe.preview?.enabled === true, everySteps: saveEvery,
          seconds: recipe.preview?.seconds ?? 40, seed: recipe.preview?.seed ?? 424242,
          previewMaxFrames: 1000, baseline: recipe.preview?.baseline ?? false,
          control: recipe.preview?.control ?? false,
        };
        preview.previewMaxFrames = Math.max(8, Math.min(120, preview.seconds || 40)) * 25;
        const trainOptions: Yue2JointTrainRequest = {
          trainingMethod: 'aitk', checkpoint: defaults.checkpoint, dataset: '', autoPrepare: true,
          output: '', steps, saveEvery, seed, device, lyricTiming,
          alignmentEnabled: lyricTiming, cursorWeight: lyricTiming ? (recipe.cursorWeight ?? 0.08) : 0, preview,
          ...(recipe.optimizer ? { optimizer: recipe.optimizer } : {}),
          ...(recipe.prodigyD0 !== undefined ? { prodigyD0: recipe.prodigyD0 } : {}),
          ...(recipe.muonLrScale !== undefined ? { muonLrScale: recipe.muonLrScale } : {}),
          ...(recipe.muonNsSteps !== undefined ? { muonNsSteps: recipe.muonNsSteps } : {}),
          ...(recipe.rank !== undefined ? { rank: recipe.rank } : {}),
          ...(recipe.alpha !== undefined ? { alpha: recipe.alpha } : {}),
          ...(recipe.stopMode ? { stopMode: recipe.stopMode } : {}),
          ...(recipe.stopMode === 'loss' && recipe.targetLoss !== undefined ? { targetLoss: recipe.targetLoss } : {}),
          ...(recipe.stopMode === 'kl' && recipe.targetKl !== undefined ? { targetKl: recipe.targetKl } : {}),
          // The recipe IS these knobs; without them a batch trains at the
          // engine's flat defaults whatever the preset says.
          ...(recipe.lr !== undefined ? { lr: recipe.lr } : {}),
          ...(recipe.weightDecay !== undefined ? { weightDecay: recipe.weightDecay } : {}),
          ...(recipe.plannerLrScale !== undefined ? { plannerLrScale: recipe.plannerLrScale } : {}),
          ...(recipe.klWeight !== undefined ? { klWeight: recipe.klWeight } : {}),
          ...(recipe.abcDropout !== undefined ? { abcDropout: recipe.abcDropout } : {}),
          ...(recipe.captionDropout !== undefined ? { captionDropout: recipe.captionDropout } : {}),
        };
        const train = await startYue2JointTrain(dataset.id, trainOptions);
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
        <label className="block text-xs text-zinc-600 dark:text-zinc-400 mb-4">
          {t('trainingStudio.yue2.aitkBatch.preset', 'Training preset')}
          <select disabled={running} value={presetName} onChange={event => choosePreset(event.target.value)}
            className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2">
            <option value="">{t('trainingStudio.yue2.aitkBatch.noPreset', 'None — use the settings below')}</option>
            {presets.map(preset => <option key={preset.name} value={preset.name}>{preset.name}</option>)}
          </select>
        </label>
        {selectedPreset && <p className="text-[11px] text-zinc-500 mb-3">
          {t('trainingStudio.yue2.aitkBatch.presetDetails', 'Applied to every dataset: {{optimizer}}, rank {{rank}}, alpha {{alpha}}, {{mode}} stopping, previews {{previews}}.', {
            optimizer: selectedPreset.settings.optimizer ?? 'adamw', rank: selectedPreset.settings.rank ?? 64,
            alpha: selectedPreset.settings.alpha ?? 64, mode: selectedPreset.settings.stopMode ?? 'steps',
            previews: selectedPreset.settings.preview?.enabled ? 'on' : 'off',
          })}
        </p>}
        {selectedPreset && selectedPreset.version !== 2 && <p className="text-[11px] text-amber-600 mb-3">
          {t('trainingStudio.yue2.aitkBatch.oldPresetTiming', 'This older preset did not reliably save lyric timing. Check the timing box below before starting.')}
        </p>}
        <div className="grid grid-cols-2 gap-2 mb-4">
          <label className="text-xs text-zinc-600 dark:text-zinc-400">Steps<input disabled={running} type="number" min={1} value={steps} onChange={e => setSteps(Math.max(1, Number(e.target.value) || 1))} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
          <label className="text-xs text-zinc-600 dark:text-zinc-400">Save every<input disabled={running} type="number" min={1} value={saveEvery} onChange={e => setSaveEvery(Math.max(1, Number(e.target.value) || 1))} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
          <label className="text-xs text-zinc-600 dark:text-zinc-400">Seed<input disabled={running} type="number" value={seed} onChange={e => setSeed(Number(e.target.value) || 0)} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
          <label className="text-xs text-zinc-600 dark:text-zinc-400">CUDA device<input disabled={running} value={device} onChange={e => setDevice(e.target.value || 'CUDA0')} className="mt-1 w-full rounded-lg bg-zinc-100 dark:bg-black/20 p-2" /></label>
        </div>
        <label className="flex items-center gap-2 text-xs mb-4"><input disabled={running} type="checkbox" checked={lyricTiming} onChange={e => setLyricTiming(e.target.checked)} className="accent-amber-500" /> Include lyric timing stems and alignment</label>
        {formError && <p className="text-xs text-red-500 mb-3">{formError}</p>}
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
