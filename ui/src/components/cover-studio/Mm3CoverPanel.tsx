// Mm3CoverPanel.tsx — Cover Studio on MiniMax-Music3. Two switches: Plan
// encodes the source into a plank (its RVQ codes) and replays them under a new
// caption and lyrics with every forcing lever exposed; Sound blends the
// source's DAV latents into the flow (engine a2a_strength). Server side: POST
// /api/mm3/cover/encode, the plank and a2a blocks in backends/minimax/generate.ts.
import { useEffect, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../services/httpClient';
import { ParamLabel } from '../shared/ParamLabel';
import { Slider } from '../shared/Slider';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
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
  const launderModels = useQuery({ queryKey: ['mm3-launder-models'], enabled: settings.launder,
    queryFn: () => apiClient.get<{ missing: string[] }>('/mm3/cover/launder-models') });
  const launderMissing = settings.launder ? launderModels.data?.missing ?? [] : [];
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
      const res = await apiClient.post<{ jobId: string; plank: string }>('/mm3/cover/encode', { sourceAudioUrl, sourceName, launder: settings.launder, caption });
      setEncode({ ...res, phase: 'queued', seconds: 0 });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const inputClass = 'w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-sm text-zinc-900 dark:text-white';
  const plankLabel = (p: PlankRow) => `${p.source || p.caption?.slice(0, 40) || p.file}${p.duration ? ` · ${Math.round(p.duration)} s` : ''}`
    + (p.laundered ? ` · ${t('cover.mm3.launderedTag', 'laundered')}` : '');
  const patternInfo: Record<Mm3CoverSettings['patternMode'], string> = {
    all: '',
    prefix: t('cover.mm3.prefixValue', 'Seconds pinned from the start'),
    anchor: t('cover.mm3.anchorValue', 'Pin one frame in every N (25 frames per second)'),
    ranges: t('cover.mm3.rangesValue', 'Pinned ranges in seconds, for example 0-20, 45-60'),
  };

  return <div className="min-w-[300px] flex-1 overflow-y-auto p-4 space-y-4">
    <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('cover.mm3.title', 'MiniMax-Music3 cover')}</h2>
    <p className="text-xs text-zinc-500">{t('cover.mm3.description', "Encodes the source into MM3's own codes, then replays them under your caption and lyrics. The codes carry the song; the levers below set how tightly it is held.")}</p>

    <ParamLabel label={t('cover.mm3.influence', 'Source influence')} underline={false}
      className="text-sm font-semibold text-zinc-700 dark:text-zinc-300"
      info={t('cover.mm3.influenceInfo', "Plan steers what the LM writes: the source's codes, held as tightly as the levers say. Sound steers how the render sounds: the source's audio blended into the flow. Use either or both.")} />
    <Toggle size="sm" accent="cyan" checked={settings.plan} onChange={plan => set({ plan })} defaultValue
      label={t('cover.mm3.plan', 'Plan')}
      info={t('cover.mm3.planInfo', "Force the reference's codes on the LM planner. Off: no codes are sent and the planner writes its own song from the caption and lyrics.")} />
    {settings.plan && <div className="space-y-4 border-l border-cyan-500/20 pl-3">
    <div className="space-y-2">
      <ParamLabel label={t('cover.mm3.reference', 'Reference')}
        info={t('cover.mm3.referenceInfo', 'Encode the loaded source once; it is saved as a plank and appears in this list for later covers. Encoding stops the engine for a minute or two, so do not start it during a render.')} />
      <Toggle size="sm" accent="cyan" checked={settings.launder} onChange={launder => set({ launder })} defaultValue={false}
        label={t('cover.mm3.launder', 'Launder source (slower, experimental)')}
        info={t('cover.mm3.launderInfo', "Re-renders the source through MM3's own flow before encoding, so the codes come from MM3-made audio. Takes about as long as the clip, needs about 14.5 GB of VRAM and the rec7 state encoder.")} />
      {launderMissing.length > 0 && <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">
        {t('cover.mm3.launderMissing', 'Laundering needs {{files}}. Download them in Model Manager.', { files: launderMissing.join(', ') })}</p>}
      <button type="button" onClick={startEncode} disabled={!sourceAudioUrl || !!encode || launderMissing.length > 0}
        className="rounded-lg bg-cyan-500/10 px-3 py-2 text-xs text-cyan-600 dark:text-cyan-300 disabled:opacity-50">
        {encode ? t('cover.mm3.encodingTimed', 'Encoding… ({{phase}}, {{seconds}} s)', { phase: encode.phase, seconds: encode.seconds }) : t('cover.mm3.encode', 'Encode source')}
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
    </div>}

    <Toggle size="sm" accent="cyan" checked={settings.sound} onChange={sound => set({ sound })} defaultValue={false}
      label={t('cover.mm3.sound', 'Sound')}
      info={t('cover.mm3.soundInfo', "Blend the source's audio into the flow, so its harmony, rhythm placement and energy steer the render. Without Plan the render is as long as the source.")} />
    {settings.sound && <div className="space-y-3 border-l border-cyan-500/20 pl-3">
      <Slider label={t('cover.mm3.soundStrength', 'Strength')} value={settings.soundStrength}
        onChange={soundStrength => set({ soundStrength })} min={0} max={1} step={0.05} defaultValue={0.5}
        info={t('cover.mm3.soundStrengthInfo', 'How much of the source survives. The flow starts this far along from the noised source and skips the steps before it, so higher is also faster. 1 gives the source back; 0 is off.')} />
      <Slider label={t('cover.mm3.soundHold', 'Hold')} value={settings.soundHold}
        onChange={soundHold => set({ soundHold })} min={0} max={1} step={0.05} defaultValue={0}
        info={t('cover.mm3.soundHoldInfo', 'Fraction of the remaining steps that put the source back at the top of each step. 0 lets the render drift from the source after the start; higher keeps it closer.')} />
      <p className="text-xs text-zinc-500">{t('cover.mm3.soundNote', "Follows the source's timeline. With Plan off the planner writes its own song and the two can fight; start with Plan on.")}</p>
      {settings.plan && settings.trimSeconds.trim() !== '' && <p className="text-xs text-zinc-500">{t('cover.mm3.soundTrim', 'Trim applies to the source audio too.')}</p>}
    </div>}

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
