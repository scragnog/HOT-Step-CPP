// MM3 covers launder the source before encoding and pin its semantic codes.
import { useEffect, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../services/httpClient';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import type { Mm3CoverSettings } from './mm3CoverState';

interface PlankRow { file: string; source?: string; caption?: string; duration?: number; created?: string; laundered?: boolean }
interface JobSummary { status: string; phase: string; error: string | null; startedAt: number | null }

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
  const launderModels = useQuery({ queryKey: ['mm3-launder-models'],
    queryFn: () => apiClient.get<{ missing: string[] }>('/mm3/cover/launder-models') });
  const launderMissing = launderModels.data?.missing ?? [];
  const [encode, setEncode] = useState<{ jobId: string; plank: string; phase: string; seconds: number } | null>(null);
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
          // One file reports no per-frame progress, so show the phase and the
          // elapsed time; a launder runs at roughly the clip's own length.
          const seconds = job.startedAt ? Math.round((Date.now() - job.startedAt) / 1000) : 0;
          setEncode(e => (e ? { ...e, phase: job.phase, seconds } : e));
        }
      } catch { /* transient; next tick retries */ }
    }, 2000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [encode?.jobId]);

  const startEncode = async () => {
    setError('');
    try {
      const res = await apiClient.post<{ jobId: string; plank: string }>('/mm3/cover/encode', { sourceAudioUrl, sourceName, launder: true, launderSteps: settings.launderSteps, launderSeed: settings.launderSeed, launderCaption: settings.launderCaption });
      setEncode({ ...res, phase: 'queued', seconds: 0 });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const inputClass = 'w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-sm text-zinc-900 dark:text-white';
  const plankLabel = (p: PlankRow) => `${p.source || p.caption?.slice(0, 40) || p.file}${p.duration ? ` · ${Math.round(p.duration)} s` : ''}`
    + (p.laundered ? ` · ${t('cover.mm3.launderedTag', 'laundered')}` : '');
  return <div className="min-w-[300px] flex-1 overflow-y-auto p-4 space-y-4">
    <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('cover.mm3.title', 'MiniMax-Music3 cover')}</h2>
    <p className="text-xs text-zinc-500">{t('cover.mm3.description', "Launders the source through MM3 before encoding, then keeps its semantic codes under your target caption.")}</p>

    <fieldset className="space-y-3 rounded-lg border border-zinc-200 dark:border-white/10 p-3">
      <legend className="px-1 text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('cover.mm3.launderGroup', 'Launder')}</legend>
      <p className="text-xs text-zinc-500">{t('cover.mm3.launderDetails', 'Re-renders the source before encoding. Takes about as long as the clip, needs about 14.5 GB of VRAM and uses the Q8_0 LM.')}</p>
      <div className="grid grid-cols-2 gap-3">
        <label className="block space-y-2">
          <ParamLabel label={t('cover.mm3.launderSteps', 'Launder steps')} info={t('cover.mm3.launderStepsInfo', 'Flow steps for the source render. Default 30.')} />
          <input className={inputClass} type="number" min={1} max={1000} step={1} value={settings.launderSteps}
            onChange={e => set({ launderSteps: Math.max(1, Math.min(1000, Math.trunc(Number(e.target.value)))) })} />
        </label>
        <label className="block space-y-2">
          <ParamLabel label={t('cover.mm3.launderSeed', 'Launder seed')} info={t('cover.mm3.launderSeedInfo', 'Seed for the source render. Default 42; -1 chooses a random seed.')} />
          <input className={inputClass} type="number" min={-1} step={1} value={settings.launderSeed}
            onChange={e => set({ launderSeed: Math.max(-1, Math.trunc(Number(e.target.value))) })} />
        </label>
      </div>
      <label className="block space-y-2">
        <ParamLabel label={t('cover.mm3.launderCaption', 'Source caption')} info={t('cover.mm3.launderCaptionInfo', 'Describe the SOURCE as it sounds, not the target style. Blank uses the neutral source caption.')} />
        <textarea className={inputClass} rows={3} value={settings.launderCaption} onChange={e => set({ launderCaption: e.target.value })} />
      </label>
    </fieldset>

    <div className="space-y-2">
      <ParamLabel label={t('cover.mm3.reference', 'Reference')}
        info={t('cover.mm3.referenceInfo', 'Encode the loaded source once; it is saved as a plank for later covers. Encoding stops the engine, so do not start it during a render.')} />
      {launderMissing.length > 0 && <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">
        {t('cover.mm3.launderMissing', 'Laundering needs {{files}}. Download them in Model Manager.', { files: launderMissing.join(', ') })}</p>}
      <button type="button" onClick={startEncode} disabled={!sourceAudioUrl || !!encode || launderModels.isPending || launderModels.isError || launderMissing.length > 0}
        className="rounded-lg bg-cyan-500/10 px-3 py-2 text-xs text-cyan-600 dark:text-cyan-300 disabled:opacity-50">
        {encode ? t('cover.mm3.encodingTimed', 'Encoding… ({{phase}}, {{seconds}} s)', { phase: encode.phase, seconds: encode.seconds }) : t('cover.mm3.encode', 'Encode source')}
      </button>
      <StyledSelect accent="cyan" value={settings.plank} onChange={plank => set({ plank })}
        options={[{ value: '', label: t('cover.mm3.noPlank', 'No reference selected') },
          ...(planks.data?.planks ?? []).map(p => ({ value: p.file, label: plankLabel(p) }))]} />
      {error && <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">{error}</p>}
    </div>

    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.trim', 'Trim to seconds')}
        info={t('cover.mm3.trimInfo', 'Keep only the first N seconds of the reference. Blank uses all of it.')} />
      <input className={inputClass} type="number" min={1} step={1} placeholder={t('cover.mm3.wholeReference', 'Whole reference')}
        value={settings.trimSeconds} onChange={e => set({ trimSeconds: e.target.value })} />
    </label>
    <label className="block space-y-2">
      <ParamLabel label={t('cover.sa3.caption', 'Caption')} info={t('cover.mm3.captionInfo', 'The target style for the cover, in MM3 caption format; do not describe the source.')} />
      <textarea value={caption} onChange={e => onCaption(e.target.value)} rows={5} className={inputClass} />
    </label>
    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.lyrics', 'Lyrics')} info={t('cover.mm3.lyricsInfo', 'The source\'s words are kept; lyrics here give the planner context.')} />
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
