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
  const effectiveEssentia = useEssentia && essentiaOk;
  const effectiveGenius = useGenius && geniusOk;
  const effectiveCaption = useCaption && captionOk;
  const anyStep = effectiveEssentia || effectiveGenius || effectiveCaption;

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
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.label.scope')}</span>
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
          <label className={`${radio} ${!essentiaOk ? 'opacity-50' : ''}`}>
            <input
              type="checkbox"
              checked={effectiveEssentia}
              disabled={!essentiaOk}
              onChange={(e) => setUseEssentia(e.target.checked)}
              className="accent-amber-500"
            />
            {t('trainingStudio.label.useEssentia')}
          </label>
          {!essentiaOk && (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.caps.essentiaMissing')}</div>
          )}

          <label className={`${radio} ${!geniusOk ? 'opacity-50' : ''}`}>
            <input
              type="checkbox"
              checked={effectiveGenius}
              disabled={!geniusOk}
              onChange={(e) => setUseGenius(e.target.checked)}
              className="accent-amber-500"
            />
            {t('trainingStudio.label.useGenius')}
          </label>
          {!geniusOk && (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.geniusMissing')}</div>
          )}

          <label className={`${radio} ${!captionOk ? 'opacity-50' : ''}`}>
            <input
              type="checkbox"
              checked={effectiveCaption}
              disabled={!captionOk}
              onChange={(e) => setUseCaption(e.target.checked)}
              className="accent-amber-500"
            />
            {t('trainingStudio.label.useCaption')}
          </label>
          {effectiveCaption && (mossOk || caps?.llm.configured) && (() => {
            const cloud = caps?.llm.providers.filter(pr => pr.available) ?? [];
            const sel = captionProvider || (mossOk ? 'moss' : caps?.llm.defaultProvider || '');
            const active = cloud.find(pr => pr.id === sel);
            return (
              <div className="ml-6 flex flex-wrap items-center gap-2">
                <span className="text-[11px] text-zinc-500">Captioner</span>
                <select
                  value={sel}
                  onChange={(e) => { setCaptionProvider(e.target.value); setCaptionModel(''); }}
                  className="text-[11px] rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-1.5 py-0.5"
                >
                  {mossOk && <option value="moss">MOSS — local, hears the audio</option>}
                  {cloud.map(pr => (
                    <option key={pr.id} value={pr.id}>
                      {pr.id === 'gemini' ? `${pr.name} (cloud, hears the audio)` : `${pr.name} (cloud, text only)`}
                    </option>
                  ))}
                </select>
                {/* Model picker for cloud providers — Gemini's list is fetched
                    live from the API, so new models appear by themselves. */}
                {active && active.models.length > 0 && (
                  <select
                    value={captionModel || active.defaultModel}
                    onChange={(e) => setCaptionModel(e.target.value)}
                    className="text-[11px] rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-1.5 py-0.5"
                  >
                    {active.models.map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                )}
              </div>
            );
          })()}
          {captionOk ? (
            <div className="ml-6 text-[11px] text-zinc-500">{t('trainingStudio.label.useCaptionHint')}</div>
          ) : (
            <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.captionMissing')}</div>
          )}
        </div>

        {/* Merge policy */}
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.label.mergePolicy')}</span>
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
          disabled={jobRunning || starting || !anyStep}
          className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {starting ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
          {starting ? t('trainingStudio.label.starting') : t('trainingStudio.label.start')}
        </button>
      </div>

      <JobProgress />

      <EnhancePanel selectedSampleIds={Array.from(selectedSampleIds)} disabled={jobRunning} />
    </div>
  );
};

export default LabelPanel;
