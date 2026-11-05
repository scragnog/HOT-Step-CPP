import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '../shared/ParamLabel';
import { MULACOVER_PACK, type HeartmulaCoverTags } from './heartmulaCoverState';

export function MulacoverPackNotice() {
  const { t } = useTranslation();
  return <div role="status" className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-zinc-600 dark:text-zinc-400 space-y-1">
    <p className="font-semibold">{MULACOVER_PACK.name} · {MULACOVER_PACK.license}</p>
    <p>{t('cover.heartmula.packNotice', 'MuLaCover weights and outputs are for non-commercial use. This is a separate cover model pack; the ordinary HeartMuLa models do not include it. Downloads are not available yet.')}</p>
    <p>{t('cover.heartmula.pending', 'HeartMuLa covers are not available yet. You can prepare and save a lead sheet while engine support is pending.')}</p>
  </div>;
}

export function HeartmulaCoverPanel({ tags, onTags, seed, onSeed, canGenerate, generating, error, onGenerate, onCancel, footer }: {
  tags: HeartmulaCoverTags; onTags: (tags: HeartmulaCoverTags) => void;
  seed: number; onSeed: (seed: number) => void; canGenerate: boolean; generating: boolean;
  error: string; onGenerate: () => void; onCancel: () => void; footer: ReactNode;
}) {
  const { t } = useTranslation();
  const inputClass = 'w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-sm text-zinc-900 dark:text-white';
  return <div className="min-w-[300px] flex-1 overflow-y-auto p-4 space-y-4">
    <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('cover.heartmula.title', 'HeartMuLa lead-sheet cover')}</h2>
    <MulacoverPackNotice />
    {(['topic', 'genre', 'instrument', 'mood'] as const).map(field => <label key={field} className="block space-y-2">
      <ParamLabel label={t(`cover.heartmula.${field}`, { topic: 'Topic', genre: 'Genre', instrument: 'Instrument', mood: 'Mood' }[field])}
        info={t('cover.heartmula.tagInfo', 'Style tags for the cover. Each field allows up to 512 characters. Leave blank for no tag; brackets, semicolons and line breaks are not allowed. Tags are independent of the caption and YuE2 adapters.')} />
      <input className={inputClass} maxLength={512} value={tags[field]} onChange={e => onTags({ ...tags, [field]: e.target.value })} />
    </label>)}
    <label className="block space-y-2">
      <ParamLabel label={t('cover.sa3.seed', 'Seed')} info={t('cover.heartmula.seedInfo', 'Use -1 for a random seed, or a fixed whole number to repeat the cover with the same settings.')} />
      <input className={inputClass} type="number" min={-1} max={Number.MAX_SAFE_INTEGER} step={1} value={seed}
        onChange={e => onSeed(Math.max(-1, Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(Number(e.target.value)))))} />
    </label>
    <p className="text-xs text-zinc-500">{t('cover.heartmula.lyricsInfo', 'Confirm the lead sheet and assign lyrics in Prepare. Supported lyric sections: Intro, Verse, Chorus, Bridge and Outro. Unmarked lyrics start as Verse; instrumental uses Intro. Generation is limited to 240 seconds.')}</p>
    {error && <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">{error}</p>}
    <button type="button" onClick={onGenerate} disabled={!canGenerate} className="rounded-lg bg-cyan-500/20 px-4 py-2 text-sm font-medium text-cyan-700 dark:text-cyan-300 disabled:opacity-50">
      {t('cover.heartmula.generate', 'Generate cover')}
    </button>
    {generating && <button type="button" onClick={onCancel} className="ml-3 text-xs text-zinc-500">{t('cover.sa3.cancel', 'Cancel')}</button>}
    {footer}
  </div>;
}
