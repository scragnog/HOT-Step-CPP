// Mm3CoverPanel.tsx — Cover Studio on MiniMax-Music3: encode the source into a
// plank (its RVQ codes), then replay those codes under a new caption and lyrics
// with every forcing lever exposed. Server side: POST /api/mm3/cover/encode,
// the plank block in backends/minimax/generate.ts, coverPattern.ts.
import { useEffect, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../services/httpClient';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import type { Mm3CoverSettings } from './mm3CoverState';

interface PlankRow { file: string; source?: string; caption?: string; duration?: number; created?: string }
interface JobSummary { status: string; phase: string; error: string | null }

export function Mm3CoverPanel({ settings, onSettings, sourceAudioUrl, sourceName, caption, onCaption, lyrics, onLyrics,
  canGenerate, generating, onGenerate, onCancel, footer }: {
  settings: Mm3CoverSettings; onSettings: Dispatch<SetStateAction<Mm3CoverSettings>>; sourceAudioUrl: string; sourceName: string;
  caption: string; onCaption: (v: string) => void; lyrics: string; onLyrics: (v: string) => void;
  canGenerate: boolean; generating: boolean; onGenerate: () => void; onCancel: () => void; footer: ReactNode;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const set = (patch: Partial<Mm3CoverSettings>) => onSettings(s => ({ ...s, ...patch }));
  const planks = useQuery({ queryKey: ['mm3-planks'], queryFn: () => apiClient.get<{ planks: PlankRow[] }>('/mm3/planks') });
  const [encode, setEncode] = useState<{ jobId: string; plank: string; phase: string } | null>(null);
  const [error, setError] = useState('');

  // Poll the encode job until it ends; it runs on the training queue.
  useEffect(() => {
    if (!encode) return;
    const timer = setInterval(async () => {
      try {
        const job = await apiClient.get<JobSummary>(`/training/jobs/${encode.jobId}`);
        if (job.status === 'done') {
          clearInterval(timer);
          setEncode(null);
          await queryClient.invalidateQueries({ queryKey: ['mm3-planks'] });
          set({ plank: encode.plank });
        } else if (job.status === 'failed' || job.status === 'cancelled') {
          clearInterval(timer);
          setEncode(null);
          setError(job.error || t('cover.mm3.encodeFailed', 'Encoding failed'));
        } else {
          setEncode(e => (e && e.phase !== job.phase ? { ...e, phase: job.phase } : e));
        }
      } catch { /* transient; next tick retries */ }
    }, 2000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [encode?.jobId]);

  const startEncode = async () => {
    setError('');
    try {
      const res = await apiClient.post<{ jobId: string; plank: string }>('/mm3/cover/encode', { sourceAudioUrl, sourceName });
      setEncode({ ...res, phase: 'queued' });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const inputClass = 'w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-sm text-zinc-900 dark:text-white';
  const plankLabel = (p: PlankRow) => `${p.source || p.caption?.slice(0, 40) || p.file}${p.duration ? ` · ${Math.round(p.duration)} s` : ''}`;
  const patternInfo: Record<Mm3CoverSettings['patternMode'], string> = {
    all: '',
    prefix: t('cover.mm3.prefixValue', 'Seconds pinned from the start'),
    anchor: t('cover.mm3.anchorValue', 'Pin one frame in every N (25 frames per second)'),
    ranges: t('cover.mm3.rangesValue', 'Pinned ranges in seconds, for example 0-20, 45-60'),
  };

  return <div className="min-w-[300px] flex-1 overflow-y-auto p-4 space-y-4">
    <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('cover.mm3.title', 'MiniMax-Music3 cover')}</h2>
    <p className="text-xs text-zinc-500">{t('cover.mm3.description', "Encodes the source into MM3's own codes, then replays them under your caption and lyrics. The codes carry the song; the levers below set how tightly it is held.")}</p>

    <div className="space-y-2">
      <ParamLabel label={t('cover.mm3.reference', 'Reference')}
        info={t('cover.mm3.referenceInfo', 'Encode the loaded source once; it is saved as a plank and appears in this list for later covers. Encoding stops the engine for a minute or two, so do not start it during a render.')} />
      <button type="button" onClick={startEncode} disabled={!sourceAudioUrl || !!encode}
        className="rounded-lg bg-cyan-500/10 px-3 py-2 text-xs text-cyan-600 dark:text-cyan-300 disabled:opacity-50">
        {encode ? t('cover.mm3.encoding', 'Encoding… ({{phase}})', { phase: encode.phase }) : t('cover.mm3.encode', 'Encode source')}
      </button>
      <StyledSelect accent="cyan" value={settings.plank} onChange={plank => set({ plank })}
        options={[{ value: '', label: t('cover.mm3.noPlank', 'No reference selected') },
          ...(planks.data?.planks ?? []).map(p => ({ value: p.file, label: plankLabel(p) }))]} />
      {error && <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">{error}</p>}
    </div>

    <Toggle size="sm" accent="cyan" checked={settings.semanticOnly} onChange={semanticOnly => set({ semanticOnly })}
      label={t('cover.mm3.semanticOnly', 'Semantic codes only')}
      info={t('cover.mm3.semanticOnlyInfo', 'On: only the semantic codes are forced, so the caption, lyrics and LM adapter shape the sound. Off: full replay of every code, which reproduces the source and ignores the caption and lyrics.')} />
    {!settings.semanticOnly && <p className="text-xs text-amber-600 dark:text-amber-400">{t('cover.mm3.fullReplay', 'Full replay ignores the caption and lyrics, and the bias and pattern below.')}</p>}

    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.forceBias', 'Force bias')}
        info={t('cover.mm3.forceBiasInfo', 'Blank pins every forced frame. A number adds that much to the source code\'s logit and lets the LM choose: below 12 ignores the source, 16 is the only middle, 20 and above follows it.')} />
      <input className={inputClass} type="number" min={0} max={1000} step={1} placeholder={t('cover.mm3.hardPin', 'Hard pin')}
        value={settings.forceBias} disabled={!settings.semanticOnly} onChange={e => set({ forceBias: e.target.value })} />
    </label>

    <div className="space-y-2">
      <ParamLabel label={t('cover.mm3.pattern', 'Forced frames')}
        info={t('cover.mm3.patternInfo', 'Which frames follow the source. Free frames are sampled by the LM and sing the lyrics field.')} />
      <StyledSelect accent="cyan" value={settings.patternMode} onChange={patternMode => set({ patternMode })}
        disabled={!settings.semanticOnly}
        options={[{ value: 'all', label: t('cover.mm3.pinAll', 'Every frame') },
          { value: 'prefix', label: t('cover.mm3.prefix', 'Prefix') },
          { value: 'anchor', label: t('cover.mm3.anchor', 'Anchor every N frames') },
          { value: 'ranges', label: t('cover.mm3.ranges', 'Pinned ranges') }]} />
      {settings.patternMode !== 'all' && settings.semanticOnly && <>
        <input className={inputClass} aria-label={patternInfo[settings.patternMode]} placeholder={patternInfo[settings.patternMode]}
          value={settings.patternValue} onChange={e => set({ patternValue: e.target.value })} />
        <Toggle size="sm" accent="cyan" checked={settings.invert} onChange={invert => set({ invert })}
          label={t('cover.mm3.invert', 'Invert')}
          info={t('cover.mm3.invertInfo', 'Free inside the rule, pinned outside it.')} />
      </>}
    </div>

    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.trim', 'Trim to seconds')}
        info={t('cover.mm3.trimInfo', 'Keep only the first N seconds of the reference. Blank uses all of it. Applied before the forced-frame rule.')} />
      <input className={inputClass} type="number" min={1} step={1} placeholder={t('cover.mm3.wholeReference', 'Whole reference')}
        value={settings.trimSeconds} onChange={e => set({ trimSeconds: e.target.value })} />
    </label>

    <label className="block space-y-2">
      <ParamLabel label={t('cover.sa3.caption', 'Caption')} info={t('cover.mm3.captionInfo', 'The style for the cover, in the MM3 caption format.')} />
      <textarea value={caption} onChange={e => onCaption(e.target.value)} rows={5} className={inputClass} />
    </label>
    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.lyrics', 'Lyrics')} info={t('cover.mm3.lyricsInfo', 'Free and soft-forced frames sing these. Hard-pinned frames keep the source\'s words.')} />
      <textarea value={lyrics} onChange={e => onLyrics(e.target.value)} rows={8} className={inputClass} />
    </label>

    <div className="grid grid-cols-2 gap-3">
      <label className="block space-y-2">
        <ParamLabel label={t('cover.sa3.seed', 'Seed')} info={t('cover.mm3.seedInfo', 'Use -1 for a random seed.')} />
        <input className={inputClass} type="number" min={-1} step={1} value={settings.seed}
          onChange={e => set({ seed: Math.max(-1, Math.trunc(Number(e.target.value))) })} />
      </label>
      <label className="block space-y-2">
        <ParamLabel label={t('cover.mm3.arSeed', 'Planner seed')} info={t('cover.mm3.arSeedInfo', 'Seed for the LM planner. -1 follows the seed.')} />
        <input className={inputClass} type="number" min={-1} step={1} value={settings.arSeed}
          onChange={e => set({ arSeed: Math.max(-1, Math.trunc(Number(e.target.value))) })} />
      </label>
    </div>
    <p className="text-xs text-zinc-500">{t('cover.mm3.globalBar', 'One take per cover. LM sampling (temperature, top-k, top-p, repetition) and the LM adapter and its scales come from the MM3 settings in the top bar, as for any MM3 render.')}</p>

    <button type="button" onClick={onGenerate} disabled={!canGenerate}
      className="rounded-lg bg-cyan-500/20 px-4 py-2 text-sm font-medium text-cyan-700 dark:text-cyan-300 disabled:opacity-50">
      {t('cover.mm3.generate', 'Generate cover')}
    </button>
    {generating && <button type="button" onClick={onCancel} className="ml-3 text-xs text-zinc-500">{t('cover.sa3.cancel', 'Cancel')}</button>}
    {footer}
  </div>;
}
