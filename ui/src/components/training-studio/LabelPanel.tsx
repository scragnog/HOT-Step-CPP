// LabelPanel.tsx — one-click labeling: Essentia + Genius + AI caption
//
// 2026-07-27 pivot: ace-understand is out of the default flow (weak captions,
// hallucinated lyrics). One button runs, per track: local BPM/key (Essentia,
// parallel CPU lane) → Genius lyrics → audio-grounded LLM caption+genre.
// No engine/GPU involvement at all. Each step degrades to disabled + an
// explanation when its capability is missing.

import React, { useState } from 'react';
import { Loader2, Play } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { MergePolicy } from '../../services/trainingApi';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { useTrainingStore } from '../../stores/trainingStore';
import { EnhancePanel } from './EnhancePanel';
import { JobProgress } from './JobProgress';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

const MERGE_OPTIONS: Array<{ id: MergePolicy; labelKey: string }> = [
  { id: 'fill_missing',      labelKey: 'trainingStudio.label.mergeFill' },
  { id: 'overwrite_caption', labelKey: 'trainingStudio.label.mergeCaption' },
  { id: 'overwrite_lyrics',  labelKey: 'trainingStudio.label.mergeLyrics' },
  { id: 'overwrite_all',     labelKey: 'trainingStudio.label.mergeAll' },
];

export const LabelPanel: React.FC = () => {
  const { t } = useTranslation();
  const caps = useTrainingStore(s => s.capabilities);
  const activeJob = useTrainingStore(s => s.activeJob);
  const startLabel = useTrainingStore(s => s.startLabel);
  const selectedSampleIds = useTrainingStore(s => s.selectedSampleIds);

  const essentiaOk = caps?.essentia.available !== false;
  const geniusOk = !!caps?.genius.configured;
  // MOSS needs no API key, so an unconfigured cloud provider must not disable
  // captioning outright — that hid the only offline option exactly when it was
  // the only one available.
  const mossOk = !!caps?.moss.available;
  const captionOk = !!caps?.llm.configured || mossOk;

  const [scope, setScope] = useState<'unlabeled' | 'all' | 'selected'>('unlabeled');
  const [useEssentia, setUseEssentia] = useState(true);
  const [useGenius, setUseGenius] = useState(true);
  const [useCaption, setUseCaption] = useState(true);
  // Default to MOSS when present: it is local, free, and describes what it hears
  // rather than rewriting the local analysis. Cloud stays available via Enhance.
  const [captionProvider, setCaptionProvider] = useState('');
  // '' = the provider's default model (config-level). The server has always
  // accepted caption.model; this panel just never offered it, which made the
  // Settings-page default look hardcoded.
  const [captionModel, setCaptionModel] = useState('');
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
  const effectiveEssentia = useEssentia && essentiaOk;
  const effectiveGenius = useGenius && geniusOk;
  const effectiveCaption = useCaption && captionOk;
  const anyStep = effectiveEssentia || effectiveGenius || effectiveCaption;
  // What this run would need the engine for: MOSS captioning. (The legacy
  // /understand step is not offered by this panel.)
  const needsEngine = effectiveCaption && (captionProvider || (mossOk ? 'moss' : '')) === 'moss';
  const blockedByJob = trainerRunning ? needsEngine : jobRunning;

  const handleStart = async () => {
    setStarting(true);
    try {
      await startLabel({
        ...(scope === 'selected' ? { sampleIds: Array.from(selectedSampleIds) } : { scope: scope === 'all' ? 'all' : 'unlabeled' }),
        useEssentia: effectiveEssentia,
        useGenius: effectiveGenius,
        useCaption: effectiveCaption,
        // Without this the server falls back to config.lireek.defaultProvider,
        // which is why labeling always went to Gemini even with MOSS installed.
        caption: {
          provider: captionProvider || (mossOk ? 'moss' : undefined),
          ...(captionModel ? { model: captionModel } : {}),
        },
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

        {/* Steps */}
        <div className="flex flex-col gap-2">
          <Toggle
            accent="amber"
            checked={effectiveEssentia}
            disabled={!essentiaOk}
            onChange={setUseEssentia}
            label={t('trainingStudio.label.useEssentia')}
            info={t('trainingStudio.label.useEssentiaInfo')}
          />
          {!essentiaOk && (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.caps.essentiaMissing')}</div>
          )}

          <Toggle
            accent="amber"
            checked={effectiveGenius}
            disabled={!geniusOk}
            onChange={setUseGenius}
            label={t('trainingStudio.label.useGenius')}
            info={t('trainingStudio.label.useGeniusInfo')}
          />
          {!geniusOk && (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.geniusMissing')}</div>
          )}

          <Toggle
            accent="amber"
            checked={effectiveCaption}
            disabled={!captionOk}
            onChange={setUseCaption}
            label={t('trainingStudio.label.useCaption')}
            info={t('trainingStudio.label.useCaptionInfo')}
          />
          {effectiveCaption && (mossOk || caps?.llm.configured) && (() => {
            const cloud = caps?.llm.providers.filter(pr => pr.available) ?? [];
            const sel = captionProvider || (mossOk ? 'moss' : caps?.llm.defaultProvider || '');
            const active = cloud.find(pr => pr.id === sel);
            return (
              <div className="ml-6 flex flex-wrap items-center gap-2">
                <ParamLabel
                  label={t('trainingStudio.label.captioner')}
                  info={t('trainingStudio.label.captionerInfo')}
                  className="text-[11px] text-zinc-500"
                />
                <StyledSelect
                  accent="amber"
                  size="sm"
                  value={sel}
                  onChange={(v) => { setCaptionProvider(v); setCaptionModel(''); }}
                  className="w-auto"
                  /* Two independent facts, and the label used to get one of
                     them wrong for everyone: WHERE the model runs, which the
                     provider now declares rather than the UI inferring from
                     `id === 'gemini'` (that called a local LM Studio server
                     "cloud"), and WHETHER it receives the audio, which only
                     Gemini and MOSS do. */
                  options={[
                    ...(mossOk ? [{ value: 'moss', label: 'MOSS — local, hears the audio' }] : []),
                    ...cloud.map(pr => ({
                      value: pr.id,
                      label: `${pr.name} (${pr.local ? 'local' : 'cloud'}, ${pr.id === 'gemini' ? 'hears the audio' : 'text only'})`,
                    })),
                  ]}
                />
                {/* Model picker for cloud providers — Gemini's list is fetched
                    live from the API, so new models appear by themselves. */}
                {active && active.models.length > 0 && (
                  <StyledSelect
                    accent="amber"
                    size="sm"
                    value={captionModel || active.defaultModel}
                    onChange={setCaptionModel}
                    className="w-auto"
                    options={active.models.map(m => ({ value: m, label: m }))}
                  />
                )}
              </div>
            );
          })()}
          {/* Say why MOSS is not on offer. The server already works out which
              of the two causes it is — the binary is not there, or the weights
              are not — because the user's next action is completely different
              for each, and then the UI used to drop that on the floor and just
              hide the option. Someone with 9.7 GB of MOSS weights correctly
              installed had no way to discover that the missing piece was
              ace-caption, which no release before v1.4 packaged at all. */}
          {effectiveCaption && !mossOk && caps?.moss.missing && (
            <div className="ml-6 text-[11px] text-zinc-500">
              MOSS (local) unavailable: {caps.moss.missing}
            </div>
          )}
          {!captionOk && (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.captionMissing')}</div>
          )}
        </div>

        {/* Merge policy */}
        <div className="flex flex-col gap-1.5">
          <ParamLabel
            label={t('trainingStudio.label.mergePolicy')}
            info={t('trainingStudio.label.mergePolicyInfo')}
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
          disabled={blockedByJob || starting || !anyStep}
          title={trainerRunning && needsEngine ? t('trainingStudio.label.mossBlocked', 'MOSS needs the engine, which the running training job owns. Pick a cloud captioner or wait.') : undefined}
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
