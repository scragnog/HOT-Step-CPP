import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { snapshotOptions } from '../../queries/cover-lyrics/reads';
import type { ProviderInfo } from '../../../../server/src/services/lireek/llm/types';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { QueryError } from '../shared/QueryError';
import { scratchRequest, submitScratchLyrics, followScratchLyrics, type ScratchBrief } from './scratchLyricsClient';

const empty: ScratchBrief = { theme: '', style: '', structure: '', existingLyrics: '' };

export const WriteLyricsTab: React.FC<{
  lyricsSetId: number; onRefresh: () => void; onViewLyrics: () => void;
}> = ({ lyricsSetId, onRefresh, onViewLyrics }) => {
  const { t } = useTranslation();
  const storageKey = `lyric-scratch-${lyricsSetId}`;
  const [brief, setBrief] = useState<ScratchBrief>(() => {
    try { return { ...empty, ...JSON.parse(localStorage.getItem(storageKey) || '{}') }; }
    catch { return empty; }
  });
  const [model, setModel] = useState('');
  const [jobId, setJobId] = useState(() => localStorage.getItem(`${storageKey}-job`) || '');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const providers = useQuery(snapshotOptions<ProviderInfo[]>('lireek', '/lireek/providers'));
  const ollama = providers.data?.find(provider => provider.id === 'ollama');
  const selectedModel = model || ollama?.default_model || '';
  const busy = submitting || !!jobId;

  useEffect(() => { localStorage.setItem(storageKey, JSON.stringify(brief)); }, [storageKey, brief]);
  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    void followScratchLyrics(jobId, controller.signal).then(complete => {
      if (complete) { setSaved(true); onRefresh(); }
    }).catch(err => {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
    }).finally(() => {
      if (!controller.signal.aborted) {
        localStorage.removeItem(`${storageKey}-job`);
        setJobId('');
      }
    });
    return () => controller.abort();
  }, [jobId, storageKey, onRefresh]);

  const generate = async () => {
    setError(''); setSaved(false); setSubmitting(true);
    try {
      const { job } = await submitScratchLyrics(scratchRequest(lyricsSetId, brief, selectedModel));
      localStorage.setItem(`${storageKey}-job`, job.id);
      setJobId(job.id);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setSubmitting(false); }
  };

  const fields: Array<{ key: keyof ScratchBrief; label: string; info: string }> = [
    { key: 'theme', label: t('lyric.scratch.theme', 'Theme'), info: t('lyric.scratch.themeInfo', 'What the song is about. Include the language or point of view you want.') },
    { key: 'style', label: t('lyric.scratch.style', 'Style'), info: t('lyric.scratch.styleInfo', 'Genre, mood, rhyme or vocal direction for the lyric and its musical caption.') },
    { key: 'structure', label: t('lyric.scratch.structure', 'Structure'), info: t('lyric.scratch.structureInfo', 'Optional section order and length, for example two verses, a chorus, a bridge and an outro. Blank lets the model choose.') },
    { key: 'existingLyrics', label: t('lyric.scratch.starter', 'Starter lyrics'), info: t('lyric.scratch.starterInfo', 'Optional lines of your own. The model expands their meaning and idea into a complete song.') },
  ];
  return (
    <div className="p-5 space-y-4 max-w-3xl">
      <p className="text-sm text-zinc-500">{t('lyric.scratch.intro', 'Write a complete song with local Ollama. No source lyrics or learned profile required. The result saves under Generated Lyrics.')}</p>
      <div className="space-y-1">
        <ParamLabel label={t('lyric.scratch.model', 'Ollama model')} info={t('lyric.scratch.modelInfo', 'Choose an installed Ollama model. Configure the Ollama address and default model in Settings, AI Services.')} />
        <StyledSelect accent="pink" value={selectedModel} onChange={setModel} disabled={busy}
          options={(ollama?.models || []).map(value => ({ value, label: value }))} />
        {providers.isError && (
          <QueryError error={providers.error} onRetry={() => void providers.refetch()} compact />
        )}
      </div>
      {fields.map(field => (
        <label key={field.key} className="block space-y-1">
          <ParamLabel label={field.label} info={field.info} />
          <textarea value={brief[field.key]} disabled={busy} rows={field.key === 'existingLyrics' ? 6 : 2}
            maxLength={field.key === 'existingLyrics' ? 12000 : field.key === 'theme' ? 4000 : 2000}
            onChange={event => setBrief(current => ({ ...current, [field.key]: event.target.value }))}
            className="w-full rounded-lg bg-zinc-100 dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 p-3 text-sm" />
        </label>
      ))}
      <button onClick={() => void generate()} disabled={busy || !selectedModel || !(brief.theme.trim() || brief.existingLyrics.trim())}
        className="px-4 py-2 rounded-lg bg-pink-600 text-white disabled:opacity-50">
        {busy ? t('lyric.scratch.writing', 'Writing lyrics...') : t('lyric.scratch.generate', 'Write lyrics')}
      </button>
      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      {saved && <div role="status" className="text-sm text-pink-400">
        {t('lyric.scratch.saved', 'Lyrics saved.')} <button className="underline" onClick={onViewLyrics}>{t('lyric.scratch.view', 'View Generated Lyrics')}</button>
      </div>}
    </div>
  );
};
