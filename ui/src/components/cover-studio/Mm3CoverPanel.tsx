// MM3 covers launder the source before encoding and pin its semantic codes.
import { useEffect, useRef, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../services/httpClient';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { mm3MatchBias, type Mm3CoverSettings } from './mm3CoverState';
import { snapshotOptions } from '../../queries/cover-lyrics/reads';
import { waitForQuery } from '../../queries/wait';
import { serverKey } from '../../queries/keys';
import { CoverArtistGrid } from './CoverArtistGrid';
import { EditableSlider } from '../shared/EditableSlider';
import { useGlobalParamsStore } from '../../stores/globalParamsStore';
import { ensureMm3SourceTracks, pickNearestBpmTrack, withSourceAttributes } from '../../utils/mm3CaptionSource';
import type { Artist, LyricsSet, AlbumPreset } from '../../services/lireekApi';

export interface Mm3SourceFacts { bpm: number; key: string; scale?: string; lyrics: string; mm3Caption: string; instrumental: boolean; origin: 'dataset' | 'analysis' }


interface PlankRow { file: string; source?: string; caption?: string; duration?: number; created?: string; laundered?: boolean }
interface JobSummary { status: string; phase: string; error: string | null; startedAt: number | null; result?: { lyrics?: string; mm3Caption?: string } }

export function Mm3CoverPanel({ settings, onSettings, sourceAudioUrl, sourceName, caption, onCaption, lyrics, onLyrics,
  sourceReady, prepareSource, duration, artists, albums, presets, canGenerate, generating, onGenerate, onCancel, footer }: {
  sourceReady: boolean; prepareSource: (signal: AbortSignal) => Promise<Mm3SourceFacts>; duration: number;
  artists: Artist[]; albums: LyricsSet[]; presets: AlbumPreset[];
  settings: Mm3CoverSettings; onSettings: Dispatch<SetStateAction<Mm3CoverSettings>>; sourceAudioUrl: string; sourceName: string;
  caption: string; onCaption: (v: string) => void; lyrics: string; onLyrics: (v: string) => void;
  canGenerate: boolean; generating: boolean; onGenerate: () => void; onCancel: () => void; footer: ReactNode;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const set = (patch: Partial<Mm3CoverSettings>) => onSettings(s => ({ ...s, ...patch }));
  const plankOptions = snapshotOptions<{ planks: PlankRow[] }>('cover', '/mm3/planks');
  const planks = useQuery(plankOptions);
  const models = useQuery(snapshotOptions<{ missing: string[]; flamingo: boolean }>('cover', '/mm3/cover/launder-models'));
  const [facts, setFacts] = useState<Mm3SourceFacts | null>(null);
  const [phase, setPhase] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [retry, setRetry] = useState(0);
  const [artistId, setArtistId] = useState<number | null>(null);
  const [albumId, setAlbumId] = useState<number | null>(null);
  const [trackTitle, setTrackTitle] = useState('custom');
  const [adapterNote, setAdapterNote] = useState('');
  const controller = useRef<AbortController | null>(null);
  const prepareRef = useRef(prepareSource);
  const settingsRef = useRef(settings);
  const onLyricsRef = useRef(onLyrics);
  useEffect(() => { prepareRef.current = prepareSource; settingsRef.current = settings; onLyricsRef.current = onLyrics; });
  const loadedSource = useRef('');
  const catalog = useQuery({ queryKey: serverKey('cover-lyrics/cover', { auth: true, id: 'mm3-caption-catalog', filters: { albums: albums.map(a => a.id).join(',') } }),
    queryFn: async () => Promise.all(albums.map(async album => ({ id: album.id, tracks: await ensureMm3SourceTracks(album.id) }))),
    enabled: albums.length > 0,
  });
  const availableAlbums = albums.filter(a => catalog.data?.some(c => c.id === a.id && c.tracks.length));
  const targetArtists = artists.filter(a => availableAlbums.some(album => album.artist_id === a.id));
  const tracks = catalog.data?.find(a => a.id === albumId)?.tracks || [];
  const chosen = tracks.find(track => track.title === trackTitle);
  useEffect(() => {
    if (chosen && facts) onCaption(withSourceAttributes(chosen.caption, facts.bpm, facts.key, facts.scale));
  }, [chosen, facts, onCaption]);

  const pollJob = async (jobId: string, signal: AbortSignal) => {
    const job = await waitForQuery<JobSummary>(snapshotOptions<JobSummary>('cover', `/training/jobs/${jobId}`),
      data => data.status === 'queued' || data.status === 'running', data => {
        signal.throwIfAborted();
        setElapsed(data.startedAt ? Math.round((Date.now() - data.startedAt) / 1000) : 0);
      }, { signal, timeoutMs: 35 * 60_000, deliverRepeats: true });
    signal.throwIfAborted();
    if (job.status !== 'done') throw new Error(job.error || t('cover.mm3.encodeFailed', 'Encoding failed'));
    return job.result;
  };
  const encodeSource = async (sourceCaption: string, signal: AbortSignal, force = false) => {
    setPhase('encode'); setElapsed(0);
    set({ plank: '', preparedSourceUrl: '' });
    const current = settingsRef.current;
    const res = await apiClient.post<{ jobId?: string; plank: string }>('/mm3/cover/encode', {
      sourceAudioUrl, sourceName, launder: true, launderSteps: current.launderSteps,
      launderSeed: current.launderSeed, launderCaption: sourceCaption, force,
    });
    signal.throwIfAborted();
    if (res.jobId) await pollJob(res.jobId, signal);
    signal.throwIfAborted();
    await queryClient.invalidateQueries({ queryKey: plankOptions.queryKey });
    signal.throwIfAborted();
    set({ plank: res.plank, preparedSourceUrl: sourceAudioUrl });
    setPhase('done');
  };
  const runPrep = async (signal: AbortSignal) => {
    const saved = settingsRef.current;
    const savedCaption = saved.preparedSourceUrl === sourceAudioUrl ? saved.launderCaption : undefined;
    setBusy(true); setError(''); setPhase('metadata'); setFacts(null); setEditing(false);
    set({ plank: '', preparedSourceUrl: '', launderCaption: '' });
    try {
      const source = await prepareRef.current(signal);
      signal.throwIfAborted();
      setFacts(source); setPhase('lyrics');
      onLyricsRef.current(source.lyrics);
      if (!source.lyrics && !source.instrumental && models.data?.flamingo) {
        const { jobId } = await apiClient.post<{ jobId: string }>('/mm3/cover/source-prep', {
          sourceAudioUrl, wantLyrics: true, wantCaption: false,
        });
        signal.throwIfAborted();
        const result = await pollJob(jobId, signal);
        onLyricsRef.current(result?.lyrics || '');
      }
      setPhase('caption');
      const sourceCaption = savedCaption ?? source.mm3Caption;
      set({ launderCaption: sourceCaption });
      if (models.data?.missing.length) throw new Error(t('cover.mm3.launderMissing', 'Laundering needs {{files}}. Download them in Model Manager.', { files: models.data.missing.join(', ') }));
      await encodeSource(sourceCaption, signal);
    } catch (err) {
      if (!signal.aborted) setError(err instanceof Error ? err.message : String(err));
    } finally { if (!signal.aborted) setBusy(false); }
  };
  useEffect(() => {
    if (loadedSource.current !== sourceAudioUrl) {
      controller.current?.abort(); loadedSource.current = ''; setBusy(false); setFacts(null); setError(''); setPhase('metadata');
      set({ plank: '', preparedSourceUrl: '' });
    }
    if (!sourceAudioUrl || !sourceReady || !models.data || loadedSource.current === sourceAudioUrl) return;
    loadedSource.current = sourceAudioUrl;
    const next = new AbortController(); controller.current = next;
    void runPrep(next.signal);
    // Source identity owns this run. Form edits cannot restart a GPU job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceAudioUrl, sourceReady, !!models.data, retry]);
  useEffect(() => () => { controller.current?.abort(); loadedSource.current = ''; }, []);
  const manualRun = async (captionOnly: boolean) => {
    const next = new AbortController(); controller.current?.abort(); controller.current = next;
    setBusy(true); setError('');
    try {
      let sourceCaption = settings.launderCaption;
      if (captionOnly) {
        setPhase('caption');
        const { jobId } = await apiClient.post<{ jobId: string }>('/mm3/cover/source-prep', {
          sourceAudioUrl, wantLyrics: false, wantCaption: true, bpm: facts?.bpm, key: facts?.key,
        });
        next.signal.throwIfAborted();
        const result = await pollJob(jobId, next.signal);
        sourceCaption = result?.mm3Caption || '';
        set({ launderCaption: sourceCaption }); setEditing(false);
      }
      await encodeSource(sourceCaption, next.signal, !captionOnly);
    } catch (err) { if (!next.signal.aborted) setError(err instanceof Error ? err.message : String(err)); }
    finally { if (!next.signal.aborted) setBusy(false); }
  };
  const selectAlbum = (id: number) => {
    setAlbumId(id); setAdapterNote('');
    const available = catalog.data?.find(a => a.id === id)?.tracks || [];
    const nearest = pickNearestBpmTrack(available, facts?.bpm);
    setTrackTitle(nearest?.title || 'custom');
    const adapter = presets.find(p => p.lyrics_set_id === id)?.mm3_adapter_path;
    if (adapter) {
      useGlobalParamsStore.getState().setBackendParam('mm3LmAdapter', adapter);
      setAdapterNote(t('cover.mm3.adapterSelected', 'Album LM adapter selected in the top bar.'));
    }
  };
  const inputClass = 'w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-sm text-zinc-900 dark:text-white';
  const plankLabel = (p: PlankRow) => `${p.source || p.caption?.slice(0, 40) || p.file}${p.duration ? ` · ${Math.round(p.duration)} s` : ''}`
    + (p.laundered ? ` · ${t('cover.mm3.launderedTag', 'laundered')}` : '');
  return <div className="min-w-[300px] flex-1 overflow-y-auto p-4 space-y-4">
    <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('cover.mm3.title', 'MiniMax-Music3 cover')}</h2>
    <p className="text-xs text-zinc-500">{t('cover.mm3.description', "Launders the source through MM3 before encoding, then keeps its semantic codes under your target caption.")}</p>

    <div role="status" className="rounded-lg border border-cyan-500/20 p-3 space-y-2">
      <div className="flex flex-wrap gap-3 text-xs">{['metadata', 'lyrics', 'caption', 'encode', 'done'].map(step =>
        <span key={step} className={phase === step ? 'font-semibold text-cyan-600 dark:text-cyan-300' : 'text-zinc-500'}>{t(`cover.mm3.phase.${step}`, step)}</span>)}</div>
      {(busy || !!sourceAudioUrl && !sourceReady) && <progress className="w-full accent-cyan-500" max={Math.max(duration, 1)} value={phase === 'encode' ? Math.min(elapsed, duration || 1) : undefined} />}
      {phase === 'encode' && busy && <p className="text-xs text-zinc-500">{t('cover.mm3.encodeTime', '{{seconds}} s elapsed / {{duration}} s clip', { seconds: elapsed, duration: Math.round(duration) })}</p>}
      {facts && <p className="text-xs text-zinc-500">{t('cover.mm3.facts', '{{bpm}} BPM · {{key}} · {{scale}} · {{origin}}', {
        bpm: facts.bpm, key: facts.key.split(' ')[0], scale: facts.scale || facts.key.split(' ').slice(1).join(' '),
        origin: facts.origin === 'dataset' ? t('cover.mm3.datasetFacts', 'Dataset sidecar') : t('cover.mm3.analysisFacts', 'Source analysis') })}</p>}
      {(error || models.isError) && <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">{error || String(models.error)}</p>}
      {error && <button type="button" className="text-xs text-cyan-600" onClick={() => { loadedSource.current = ''; setRetry(v => v + 1); }}>{t('cover.mm3.retry', 'Retry')}</button>}
    </div>
    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.launderCaption', 'Source caption')} info={t('cover.mm3.sourceCaptionInfo', 'The dataset MM3 sidecar describes the source. This caption guides the launder render. Edit it, then Re-encode to apply changes.')} />
      <textarea className={inputClass} rows={4} readOnly={!editing || busy} value={settings.launderCaption} onChange={e => set({ launderCaption: e.target.value, preparedSourceUrl: '' })} />
    </label>
    <button type="button" disabled={busy} className="text-xs text-cyan-600 dark:text-cyan-300" onClick={() => setEditing(v => !v)}>{t('cover.mm3.editSourceCaption', 'Edit source caption')}</button>
    {!settings.launderCaption && <div className="space-y-1">
      <button type="button" disabled={busy || !facts || !models.data?.flamingo} className="text-xs text-cyan-600 dark:text-cyan-300 disabled:opacity-50" onClick={() => void manualRun(true)}>{t('cover.mm3.flamingoCaption', 'Caption with Music-Flamingo')}</button>
      {models.data && !models.data.flamingo && <p className="text-xs text-zinc-500">{t('cover.mm3.flamingoMissing', 'Music-Flamingo is not installed. Download it in Model Manager to caption or transcribe this source.')}</p>}
    </div>}
    <details className="rounded-lg border border-zinc-200 dark:border-white/10 p-3 space-y-3">
      <summary className="text-sm text-zinc-700 dark:text-zinc-300 cursor-pointer">{t('cover.mm3.advanced', 'Advanced')}</summary>
      <div className="grid grid-cols-2 gap-3">
        <label className="block space-y-2">
          <ParamLabel label={t('cover.mm3.launderSteps', 'Launder steps')} info={t('cover.mm3.launderStepsInfo', 'Flow steps for the source render. Default 30.')} />
          <input className={inputClass} disabled={busy} type="number" min={1} max={1000} step={1} value={settings.launderSteps}
            onChange={e => set({ launderSteps: Math.max(1, Math.min(1000, Math.trunc(Number(e.target.value)))), preparedSourceUrl: '' })} />
        </label>
        <label className="block space-y-2">
          <ParamLabel label={t('cover.mm3.launderSeed', 'Launder seed')} info={t('cover.mm3.launderSeedInfo', 'Seed for the source render. Default 42; -1 chooses a random seed.')} />
          <input className={inputClass} disabled={busy} type="number" min={-1} step={1} value={settings.launderSeed}
            onChange={e => set({ launderSeed: Math.max(-1, Math.trunc(Number(e.target.value))), preparedSourceUrl: '' })} />
        </label>
      </div>
    </details>
    <div className="space-y-2">
      <button type="button" onClick={() => void manualRun(false)} disabled={!facts || busy || generating || !!models.data?.missing.length}
        className="rounded-lg bg-cyan-500/10 px-3 py-2 text-xs text-cyan-600 dark:text-cyan-300 disabled:opacity-50">{t('cover.mm3.reencode', 'Re-encode')}</button>
      <StyledSelect accent="cyan" value={settings.plank} disabled={busy} onChange={plank => set({ plank, preparedSourceUrl: sourceAudioUrl })}
        options={[{ value: '', label: t('cover.mm3.noPlank', 'No reference selected') }, ...(planks.data?.planks ?? []).map(p => ({ value: p.file, label: plankLabel(p) }))]} />
    </div>
    <ParamLabel label={t('cover.mm3.targetArtist', 'Target artist')} info={t('cover.mm3.targetArtistInfo', 'Choose an artist and album with MM3 training captions. The nearest-tempo caption is selected first.')} />
    <CoverArtistGrid artists={targetArtists} loading={catalog.isPending && albums.length > 0} selectedArtistId={artistId}
      onSelectArtist={artist => { setArtistId(artist.id); const first = availableAlbums.find(a => a.artist_id === artist.id); if (first) selectAlbum(first.id); }} />
    <ParamLabel label={t('cover.mm3.album', 'Album')} info={t('cover.mm3.albumInfo', 'Albums with an MM3 caption. Selecting an album also selects its LM adapter when one is saved.')} />
    <StyledSelect accent="cyan" value={albumId ? String(albumId) : ''} onChange={id => selectAlbum(Number(id))}
      options={availableAlbums.filter(a => a.artist_id === artistId).map(a => ({ value: String(a.id), label: a.album || String(a.id) }))} />
    <ParamLabel label={t('cover.mm3.captionSource', 'Caption source')} info={t('cover.mm3.captionSourceInfo', 'Use a named track or write a Custom caption. Source BPM, key and scale replace the target track facts.')} />
    <StyledSelect accent="cyan" value={trackTitle} onChange={setTrackTitle} options={[{ value: 'custom', label: t('cover.mm3.custom', 'Custom') }, ...tracks.map(track => ({ value: track.title, label: track.title }))]} />
    {adapterNote && <p className="text-xs text-zinc-500">{adapterNote}</p>}
    <EditableSlider label={t('cover.mm3.match', 'Match')} value={settings.match} min={0} max={100} step={1} defaultValue={100}
      onChange={match => set({ match })} tooltip={t('cover.mm3.matchInfo', '100 pins the source semantic codes. Lower values give the target artist more room to freestyle. Try 20–60 for a looser match.')}
      helpText={settings.match >= 100 ? t('cover.mm3.hardPin', 'Hard pin') : t('cover.mm3.biasHelp', 'Semantic bias: {{bias}}', { bias: mm3MatchBias(settings.match) })} />

    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.trim', 'Trim to seconds')}
        info={t('cover.mm3.trimInfo', 'Keep only the first N seconds of the reference. Blank uses all of it.')} />
      <input className={inputClass} type="number" min={1} step={1} placeholder={t('cover.mm3.wholeReference', 'Whole reference')}
        value={settings.trimSeconds} onChange={e => set({ trimSeconds: e.target.value })} />
    </label>
    <label className="block space-y-2">
      <ParamLabel label={t('cover.sa3.caption', 'Caption')} info={t('cover.mm3.captionInfo', 'The target style for the cover, in MM3 caption format; do not describe the source.')} />
      <textarea value={caption} readOnly={trackTitle !== 'custom'} onChange={e => onCaption(e.target.value)} rows={5} className={inputClass} />
    </label>
    <label className="block space-y-2">
      <ParamLabel label={t('cover.mm3.lyrics', 'Lyrics')} info={t('cover.mm3.lyricsInfo', 'The source\'s words are kept; lyrics here give the planner context.')} />
      <textarea value={lyrics} readOnly={busy} onChange={e => onLyrics(e.target.value)} rows={8} className={inputClass} />
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

    <button type="button" onClick={onGenerate} disabled={!canGenerate || busy || !!error}
      className="rounded-lg bg-cyan-500/20 px-4 py-2 text-sm font-medium text-cyan-700 dark:text-cyan-300 disabled:opacity-50">
      {t('cover.mm3.generate', 'Generate cover')}
    </button>
    {generating && <button type="button" onClick={onCancel} className="ml-3 text-xs text-zinc-500">{t('cover.sa3.cancel', 'Cancel')}</button>}
    {footer}
  </div>;
}
