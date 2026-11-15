// LabelPanel.tsx — one-click labeling: Essentia + Genius + AI caption
//
// 2026-07-27 pivot: ace-understand is out of the default flow (weak captions,
// hallucinated lyrics). One button runs, per track: local BPM/key (Essentia,
// parallel CPU lane) → Genius lyrics → audio-grounded LLM caption+genre.
// The step controls live in useLabelSteps, shared with the batch import wizard.

import React, { useState } from 'react';
import { Loader2, Play } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { MergePolicy } from '../../services/trainingApi';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { useTrainingStore } from '../../stores/trainingStore';
import { EnhancePanel } from './EnhancePanel';
import { JobProgress } from './JobProgress';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';
import { useLabelSteps } from './useLabelSteps';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

const MERGE_OPTIONS: Array<{ id: MergePolicy; labelKey: string }> = [
  { id: 'fill_missing',      labelKey: 'trainingStudio.label.mergeFill' },
  { id: 'overwrite_caption', labelKey: 'trainingStudio.label.mergeCaption' },
  { id: 'overwrite_lyrics',  labelKey: 'trainingStudio.label.mergeLyrics' },
  { id: 'overwrite_all',     labelKey: 'trainingStudio.label.mergeAll' },
];

export const LabelPanel: React.FC = () => {
  const { t } = useTranslation();
  const activeJob = useActiveTrainingJob();
  const startLabel = useTrainingStore(s => s.startLabel);
  const selectedSampleIds = useTrainingStore(s => s.selectedSampleIds);

  const [scope, setScope] = useState<'unlabeled' | 'all' | 'selected'>('unlabeled');
  const { fields, options, anyStep, needsEngine, invalidTempoBounds } = useLabelSteps();
  const [mergePolicy, setMergePolicy] = useState<MergePolicy>('fill_missing');
  const [starting, setStarting] = useState(false);

  const selCount = selectedSampleIds.size;
  const jobRunning = !!activeJob && (activeJob.status === 'queued' || activeJob.status === 'running');
  // A running TRAINER no longer blocks labelling (server rule, 2026-09-09):
  // cloud captioning, Genius and Essentia run in the network lane beside it.
  // Only MOSS and the legacy /understand step need the engine, and only those
  // keep the button disabled while a trainer runs. Any other active job
  // (preprocess, codes, another label) still blocks, as before.
  // Mirrors the server's own TRAINER_KINDS (routes/training.ts). `yue2-nar-train`
  // joins it and `yue2-preprocess` does not, for the same reason the ACE
  // preprocess step does not: it is a GPU pass over the dataset's own audio, not
  // a trainer that leaves the files alone. The two YuE2 AR cache stages join on
  // that same test rather than on being "training": both take the manifest as
  // their only input and rewrite it in place, so neither can lose a sidecar edit.
  const TRAINER_KINDS = ['train-lm', 'train-dit', 'mm3-train-lm', 'yue2-nar-train',
    'yue2-tokenize', 'yue2-align', 'yue2-ar-train', 'audition', 'lm-calibrate', 'dit-calibrate'];
  const trainerRunning = jobRunning && TRAINER_KINDS.includes(String(activeJob?.kind));
  const blockedByJob = trainerRunning ? needsEngine : jobRunning;

  const handleStart = async () => {
    setStarting(true);
    try {
      await startLabel({
        ...(scope === 'selected' ? { sampleIds: Array.from(selectedSampleIds) } : { scope: scope === 'all' ? 'all' : 'unlabeled' }),
        ...options,
        mergePolicy,
      });
    } finally {
      setStarting(false);
    }
  };

  const radio = 'flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer';

  return (
    <div className="flex flex-col gap-4">
      <div className={`${CARD} flex flex-col gap-4`}>
        <div>
          <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.label.title')}</h2>
          <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">{t('trainingStudio.label.subtitle')}</p>
        </div>

        {/* Scope */}
        <div className="flex flex-col gap-2">
          <ParamLabel
            label={t('trainingStudio.label.scope')}
            info={t('trainingStudio.label.scopeInfo')}
          />
          <div className="flex flex-col gap-1.5">
            <label className={radio}>
              <input type="radio" checked={scope === 'unlabeled'} onChange={() => setScope('unlabeled')} className="accent-amber-500" />
              {t('trainingStudio.label.scopeUnlabeled')}
            </label>
            <label className={radio}>
              <input type="radio" checked={scope === 'all'} onChange={() => setScope('all')} className="accent-amber-500" />
              {t('trainingStudio.label.scopeAll')}
            </label>
            <label className={`${radio} ${selCount === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}>
              <input
                type="radio"
                checked={scope === 'selected'}
                disabled={selCount === 0}
                onChange={() => setScope('selected')}
                className="accent-amber-500"
              />
              {t('trainingStudio.label.scopeSelected', { count: selCount })}
            </label>
          </div>
        </div>

        {fields}

        {/* Merge policy */}
        <div className="flex flex-col gap-1.5">
          <ParamLabel
            label={t('trainingStudio.label.mergePolicy')}
            info={t('trainingStudio.label.mergePolicyInfo')}
            onReset={mergePolicy !== 'fill_missing' ? () => setMergePolicy('fill_missing') : undefined}
          />
          <StyledSelect
            accent="amber"
            size="sm"
            className="max-w-64"
            value={mergePolicy}
            onChange={(v) => setMergePolicy(v as MergePolicy)}
            options={MERGE_OPTIONS.map(o => ({ value: o.id, label: t(o.labelKey) }))}
          />
        </div>

        {/* `error` is rendered once by DatasetDetail so every step sees it. */}

        <button
          onClick={() => void handleStart()}
          disabled={blockedByJob || starting || !anyStep || invalidTempoBounds}
          title={trainerRunning && needsEngine ? t('trainingStudio.label.mossBlocked', 'Local captioning and lyric transcription need the card, which the running training job owns. Pick a cloud captioner, turn off local lyrics, or wait.') : undefined}
          className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {starting ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
          {starting ? t('trainingStudio.label.starting') : t('trainingStudio.label.start')}
        </button>
      </div>

      <JobProgress />

      <EnhancePanel selectedSampleIds={Array.from(selectedSampleIds)} disabled={jobRunning && !trainerRunning} engineBusy={trainerRunning} />
    </div>
  );
};

export default LabelPanel;
