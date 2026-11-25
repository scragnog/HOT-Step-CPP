// MidiStudio.tsx — MIDI Studio orchestrator (audio → MIDI transcription)
//
// Pick any track from the library, transcribe it to multi-instrument MIDI,
// preview the result as a piano roll, and download the .mid.
//
// Transcription runs on the NATIVE ace-midi engine (a GGML port of
// MuScriptor by Kyutai & Mirelo — full attribution in the footer card).
// Model weights are gated on Hugging Face and downloaded in-app with the
// user's read token.

import React, { useCallback, useMemo, useRef, useState } from 'react';
import {
  Piano, FolderOpen, Search, X, Download, Trash2, ChevronDown, ChevronUp,
  CheckCircle2, ExternalLink, Music, KeyRound, Loader2, AlertTriangle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '@/context/AuthContext';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { midiJobsOptions, midiStatusOptions, removeMidiJob } from '@/queries/studios/midi';
import { queryManager } from '@/queries/invalidation';
import { useStudioSongs } from '@/queries/studios/shared';
import type { Song } from '@/types';
import {
  submitTranscription,
  getMidiFileUrl, saveHfToken, startModelDownload,
  HF_MODEL_URLS, HF_TOKEN_SETTINGS_URL, BEAT_MODEL,
  type MuscriptorModel, type MidiModel,
} from '@/services/midiStudioApi';
import { MidiPlayer } from './MidiPlayer';
import { MidiExportPanel } from './MidiExportPanel';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { Toggle } from '@/components/shared/controls/Toggle';
import { useUpload } from '@/queries/writes';
import { formatClock, formatDecimal } from '@/utils/format/format';
import { errorMessage } from '@/utils/format/errorMessage';

const MODEL_INFO: Array<{ id: MidiModel; params: string }> = [
  { id: 'small', params: '103M' },
  { id: 'medium', params: '307M' },
  { id: 'large', params: '1.4B' },
  { id: 'dettato', params: '28M' },
];

export const MidiStudio: React.FC = () => {
  const { t } = useTranslation();
  const { sessionEpoch } = useAuth();
  void sessionEpoch;
  const client = useQueryClient();
  const statusQuery = useQuery(midiStatusOptions());
  const jobsQuery = useQuery(midiJobsOptions());
  const status = statusQuery.data;
  const jobs = jobsQuery.data ?? [];
  const refreshStatus = () => queryManager.invalidate(client, midiStatusOptions().queryKey);
  const refreshJobs = () => queryManager.invalidate(client, midiJobsOptions().queryKey);
  // No per-mutation retry: the client default is already off (queries/client.ts).
  const deletion = useMutation({ mutationFn: (id: string) => removeMidiJob(client, id) });


  // ── HF access token (gated weights) ──
  const [hfTokenInput, setHfTokenInput] = useState('');
  const [hfTokenError, setHfTokenError] = useState('');
  // Once a token is saved the card collapses to a slim row; "Change" reopens it
  const [showTokenEditor, setShowTokenEditor] = useState(false);

  // ── Source selection ──
  const [showLibrary, setShowLibrary] = useState(false);
  const [librarySearch, setLibrarySearch] = useState('');
  const songsQuery = useStudioSongs(showLibrary);
  const librarySongs = (songsQuery.data?.songs ?? []).filter(song => song.audioUrl);
  const [sourceAudioUrl, setSourceAudioUrl] = useState('');
  const [sourceName, setSourceName] = useState('');
  const [sourceSongId, setSourceSongId] = useState<string | undefined>(undefined);
  const [model, setModel] = useState<MidiModel>(() =>
    (localStorage.getItem('hs-midi-model') as MidiModel) || 'small');
  const [repair, setRepair] = useState(() => localStorage.getItem('hs-midi-repair') !== '0');
  const [speculative, setSpeculative] = useState(() => localStorage.getItem('hs-midi-speculative') !== '0');

  // ── Jobs ──
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState('');
  const filteredSongs = useMemo(() => {
    if (!librarySearch.trim()) return librarySongs;
    const q = librarySearch.toLowerCase();
    return librarySongs.filter(s =>
      s.title?.toLowerCase().includes(q) ||
      s.artistName?.toLowerCase().includes(q) ||
      s.style?.toLowerCase().includes(q)
    );
  }, [librarySongs, librarySearch]);

  const handleSelectSong = useCallback((song: Song) => {
    setSourceAudioUrl(song.audioUrl || song.audio_url || '');
    setSourceName(song.title || 'Library Track');
    setSourceSongId(song.id);
    setShowLibrary(false);
    setSubmitError('');
  }, []);

  // ── Upload a track from the user's PC (same endpoint Repaint uses) ──
  const { mutate: uploadAudio, isPending: isUploading } = useUpload<{ audio_url?: string }>();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handleFileUpload = useCallback((file: File) => {
    setSubmitError('');
    uploadAudio({
      path: '/upload/audio',
      files: [{ field: 'audio', file }],
      // The inline submitError line below is the sole notifier for this upload.
      silent: true,
      onSuccess: data => {
        if (!data.audio_url) return;
        setSourceAudioUrl(data.audio_url);
        setSourceName(file.name);
        setSourceSongId(undefined);
        setShowLibrary(false);
      },
      onError: err => setSubmitError(errorMessage(err)),
    });
  }, [uploadAudio]);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileUpload(file);
  }, [handleFileUpload]);

  const handleRepairChange = (v: boolean) => {
    setRepair(v);
    try { localStorage.setItem('hs-midi-repair', v ? '1' : '0'); } catch { /* ignore */ }
  };
  const handleModelChange = (m: MidiModel) => {
    setModel(m);
    try { localStorage.setItem('hs-midi-model', m); } catch { /* ignore */ }
  };
  const handleSpeculativeChange = (v: boolean) => {
    setSpeculative(v);
    try { localStorage.setItem('hs-midi-speculative', v ? '1' : '0'); } catch { /* ignore */ }
  };

  // All three model-download buttons share this mutation: the inline
  // submitError line is the sole notifier (startModelDownload is silent).
  const downloadMut = useMutation({
    mutationFn: (m: MidiModel | typeof BEAT_MODEL) => startModelDownload(m),
    onSuccess: () => refreshStatus(),
    onError: err => setSubmitError(errorMessage(err)),
  });

  const transcribeMut = useMutation({
    mutationFn: () => submitTranscription({ sourceAudioUrl, sourceFileName: sourceName, songId: sourceSongId, model, speculative, repair }),
    onSuccess: () => refreshJobs(),
    onError: err => setSubmitError(errorMessage(err)),
  });
  const handleTranscribe = () => {
    if (!sourceAudioUrl) return;
    setSubmitError('');
    transcribeMut.mutate();
  };

  const hfTokenMut = useMutation({
    mutationFn: (tok: string) => saveHfToken(tok),
    onSuccess: (_data, tok) => {
      setHfTokenInput('');
      if (tok.trim()) setShowTokenEditor(false);
      refreshStatus();
    },
    onError: err => setHfTokenError(errorMessage(err)),
  });
  const handleSaveHfToken = (tok: string) => {
    setHfTokenError('');
    hfTokenMut.mutate(tok);
  };

  const handleDelete = async (jobId: string) => {
    try {
      await deletion.mutateAsync(jobId);
      if (expandedJob === jobId) setExpandedJob(null);
    } catch (err) { setSubmitError(errorMessage(err)); }
  };

  const engineMissing = !!status && !status.engineAvailable;
  const modelState = status?.models?.[model];
  const beatState = status?.models?.[BEAT_MODEL];
  // The transcribe in flight is a duplicate-job guard: the job list only shows
  // it after the status refresh, so the mutation's own pending covers the gap.
  const canTranscribe = !!status?.engineAvailable && !!modelState?.downloaded && !!sourceAudioUrl && !isUploading && !transcribeMut.isPending;

  const fmtGB = (b: number) => `${formatDecimal((b / 1e9), 2)} GB`;

  return (
    <div className="h-full w-full overflow-y-auto">
      <div className="max-w-4xl mx-auto max-[1023px]:px-3 min-[1024px]:px-6 py-8 flex flex-col gap-6">

        {/* ── Header ── */}
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-white flex items-center gap-3">
            <Piano size={26} className="text-purple-500 dark:text-purple-400" />
            {t('midiStudio.title')}
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">{t('midiStudio.subtitle')}</p>
        </div>

        {/* ── Engine missing banner (broken install / old build) ── */}
        {engineMissing && (
          <div className="rounded-xl border border-red-300/60 dark:border-red-500/20 bg-red-50 dark:bg-red-500/5 p-4 flex items-start gap-3">
            <AlertTriangle size={18} className="text-red-500 mt-0.5 flex-shrink-0" />
            <div className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
              <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-1">{t('midiStudio.engineMissingTitle')}</div>
              {t('midiStudio.engineMissingBody')}
            </div>
          </div>
        )}

        {/* ── Hugging Face model access (weights are gated) ──
             Collapses to a slim confirmation row once a token is saved. */}
        {status && status.hfTokenSet && !showTokenEditor && (
          <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card px-4 py-2.5 flex items-center gap-3 flex-wrap text-xs">
            <span className="flex items-center gap-2 font-semibold text-zinc-900 dark:text-white">
              <KeyRound size={14} className="text-purple-500 dark:text-purple-400" />
              {t('midiStudio.hfAccessTitle')}
            </span>
            <span className="flex items-center gap-1 font-medium text-emerald-600 dark:text-emerald-400">
              <CheckCircle2 size={12} /> {t('midiStudio.hfTokenSaved')}
            </span>
            <span className="flex-1" />
            <button
              onClick={() => setShowTokenEditor(true)}
              className="text-zinc-500 dark:text-zinc-400 hover:text-purple-600 dark:hover:text-purple-400 hover:underline"
            >
              {t('midiStudio.hfChangeToken')}
            </button>
          </div>
        )}
        {status && (!status.hfTokenSet || showTokenEditor) && (
          <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4">
            <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-1 flex items-center gap-2">
              <KeyRound size={15} className="text-purple-500 dark:text-purple-400" />
              {t('midiStudio.hfAccessTitle')}
              {status.hfTokenSet && (
                <span className="flex items-center gap-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 size={12} /> {t('midiStudio.hfTokenSaved')}
                </span>
              )}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">{t('midiStudio.hfAccessBody')}</p>
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs">
              {(Object.keys(HF_MODEL_URLS) as MuscriptorModel[]).map(m => (
                <a key={m} href={HF_MODEL_URLS[m]} target="_blank" rel="noreferrer"
                  className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline capitalize">
                  <ExternalLink size={11} /> {m}
                </a>
              ))}
              <a href={HF_TOKEN_SETTINGS_URL} target="_blank" rel="noreferrer"
                className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
                <ExternalLink size={11} /> {t('midiStudio.hfGetToken')}
              </a>
            </div>
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              <input
                type="password"
                value={hfTokenInput}
                onChange={e => setHfTokenInput(e.target.value)}
                placeholder={t('midiStudio.hfTokenPlaceholder')}
                className="flex-1 min-[1024px]:min-w-[220px] min-w-0 px-3 py-2 rounded-lg text-xs bg-zinc-100 dark:bg-black/30 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 outline-none focus:border-purple-500/40"
              />
              <button
                onClick={() => handleSaveHfToken(hfTokenInput)}
                disabled={hfTokenMut.isPending || !hfTokenInput.trim()}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {hfTokenMut.isPending ? <Loader2 size={13} className="animate-spin" /> : t('midiStudio.hfTokenSave')}
              </button>
              {status.hfTokenSet && (
                <button
                  onClick={() => handleSaveHfToken('')}
                  disabled={hfTokenMut.isPending}
                  className="px-3 py-2 rounded-lg text-xs font-medium text-zinc-500 dark:text-zinc-400 hover:text-red-500 border border-zinc-200 dark:border-white/10 transition-colors"
                >
                  {t('midiStudio.hfTokenClear')}
                </button>
              )}
            </div>
            {hfTokenError && <div className="mt-2 text-xs text-red-500 dark:text-red-400">{hfTokenError}</div>}
            <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-2">{t('midiStudio.hfTokenNote')}</p>
          </div>
        )}

        {/* ── New transcription ── */}
        <div
          className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4"
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
        >
          <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-3">{t('midiStudio.newTranscription')}</div>

          {/* Source picker: library track or a file from the user's PC */}
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => setShowLibrary(v => !v)}
              className="px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-2 bg-zinc-100 dark:bg-white/5 hover:bg-zinc-200 dark:hover:bg-white/10 text-zinc-700 dark:text-zinc-300 transition-colors"
            >
              <FolderOpen size={14} /> {t('midiStudio.chooseFromLibrary')}
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="px-3 py-2 rounded-lg text-xs font-medium flex items-center gap-2 bg-zinc-100 dark:bg-white/5 hover:bg-zinc-200 dark:hover:bg-white/10 text-zinc-700 dark:text-zinc-300 disabled:opacity-50 transition-colors"
            >
              {isUploading ? <Loader2 size={14} className="animate-spin" /> : <Music size={14} />}
              {t('midiStudio.uploadFile')}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.flac,audio/wav,audio/mpeg,audio/flac"
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFileUpload(f); e.target.value = ''; }}
            />
            {sourceName ? (
              <span className="flex items-center gap-2 text-sm text-zinc-800 dark:text-zinc-200">
                <Music size={14} className="text-purple-500 dark:text-purple-400" />
                {sourceName}
                <button onClick={() => { setSourceAudioUrl(''); setSourceName(''); setSourceSongId(undefined); }}
                  className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200">
                  <X size={14} />
                </button>
              </span>
            ) : (
              <span className="text-xs text-zinc-400 dark:text-zinc-500">{t('midiStudio.noTrackSelected')}</span>
            )}
          </div>

          {/* Inline library panel */}
          {showLibrary && (
            <div className="mt-3 rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
              <div className="p-2 border-b border-zinc-200 dark:border-white/5 flex items-center gap-2">
                <Search size={14} className="text-zinc-400 flex-shrink-0" />
                <input
                  value={librarySearch}
                  onChange={e => setLibrarySearch(e.target.value)}
                  placeholder={t('midiStudio.searchLibrary')}
                  className="flex-1 bg-transparent text-sm text-zinc-900 dark:text-white placeholder-zinc-400 outline-none"
                />
              </div>
              <div className="max-h-56 overflow-y-auto">
                {filteredSongs.length === 0 && (
                  <div className="px-3 py-4 text-xs text-zinc-400 dark:text-zinc-500">{t('midiStudio.libraryEmpty')}</div>
                )}
                {filteredSongs.map(song => (
                  <button
                    key={song.id}
                    onClick={() => handleSelectSong(song)}
                    className="w-full text-left px-3 py-2 flex items-center justify-between gap-3 hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
                  >
                    <span className="text-sm text-zinc-800 dark:text-zinc-200 truncate">{song.title || t('midiStudio.untitled')}</span>
                    <span className="text-[11px] text-zinc-400 flex-shrink-0">
                      {song.duration ? formatClock(Number(song.duration)) : ''}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Model select + per-model weight download */}
          <div className="mt-4">
            <div className="text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-2">{t('midiStudio.model')}</div>
            <div className="flex gap-2 flex-wrap">
              {MODEL_INFO.map(mi => {
                const ms = status?.models?.[mi.id];
                const pct = ms?.downloading && ms.totalBytes > 0
                  ? Math.round(100 * ms.receivedBytes / ms.totalBytes) : 0;
                return (
                  <div
                    key={mi.id}
                    onClick={() => handleModelChange(mi.id)}
                    className={`px-3 py-2 rounded-lg text-xs border cursor-pointer transition-colors min-w-[150px] ${
                      model === mi.id
                        ? 'border-purple-500/60 bg-purple-500/10 text-purple-600 dark:text-purple-300'
                        : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-400 hover:border-zinc-300 dark:hover:border-white/20'
                    }`}
                  >
                    <span className="font-semibold capitalize">{mi.id}</span>
                    <span className="opacity-60 ml-1.5">{mi.params}</span>
                    <span className="block text-[10px] opacity-60 mt-0.5">{t(`midiStudio.model_${mi.id}`)}</span>
                    {ms?.downloaded ? (
                      <span className="flex items-center gap-1 text-[10px] text-emerald-600 dark:text-emerald-400 mt-1">
                        <CheckCircle2 size={10} /> {t('midiStudio.modelReady')} ({fmtGB(ms.sizeBytes)})
                      </span>
                    ) : ms?.downloading ? (
                      <span className="block mt-1">
                        <span className="flex items-center gap-1 text-[10px] text-purple-500">
                          <Loader2 size={10} className="animate-spin" />
                          {t('midiStudio.downloading')} {pct}% ({fmtGB(ms.receivedBytes)}{ms.totalBytes ? ` / ${fmtGB(ms.totalBytes)}` : ''})
                        </span>
                        <span className="block h-1 mt-1 rounded bg-zinc-200 dark:bg-white/10 overflow-hidden">
                          <span className="block h-full bg-purple-500 transition-all" style={{ width: `${pct}%` }} />
                        </span>
                      </span>
                    ) : (
                      <button
                        onClick={e => { e.stopPropagation(); downloadMut.mutate(mi.id); }}
                        className="flex items-center gap-1 text-[10px] text-purple-600 dark:text-purple-400 hover:underline mt-1"
                      >
                        <Download size={10} /> {t('midiStudio.downloadModel')}
                      </button>
                    )}
                    {ms?.error && !ms.downloading && (
                      <span className="block text-[10px] text-red-500 dark:text-red-400 mt-1 max-w-[200px]">
                        {ms.gated ? t('midiStudio.gatedHint') + ' ' : ''}{ms.error.slice(0, 120)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            <p className="text-[11px] text-zinc-400 dark:text-zinc-500 mt-2">{t('midiStudio.firstRunNote')}</p>

            {/* Beat This!: optional, gives the MIDI tempo, time signature and bar lines */}
            <div className="mt-3 flex items-center gap-2 flex-wrap text-xs">
              <ParamLabel label={t('midiStudio.tempoModel')} info={t('midiStudio.tempoModelInfo')} />
              {beatState?.downloaded ? (
                <span className="flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 size={11} /> {t('midiStudio.tempoModelReady')}
                </span>
              ) : beatState?.downloading ? (
                <span className="flex items-center gap-1 text-[11px] text-purple-500">
                  <Loader2 size={11} className="animate-spin" />
                  {t('midiStudio.downloading')} {beatState.totalBytes ? Math.round(100 * beatState.receivedBytes / beatState.totalBytes) : 0}%
                </span>
              ) : (
                <button
                  onClick={() => downloadMut.mutate(BEAT_MODEL)}
                  className="flex items-center gap-1 text-[11px] text-purple-600 dark:text-purple-400 hover:underline"
                >
                  <Download size={11} /> {t('midiStudio.tempoModelDownload')}
                </button>
              )}
              {beatState?.error && !beatState.downloading && (
                <span className="text-[11px] text-red-500 dark:text-red-400">{beatState.error.slice(0, 120)}</span>
              )}
            </div>

            {/* dettato's repairer: on by default, can be skipped for the raw transcriber notes */}
            {model === 'dettato' && (
              <div className="mt-3 flex items-center gap-2 flex-wrap text-xs">
                <Toggle size="sm" accent="purple" checked={repair} onChange={handleRepairChange}
                  label={t('midiStudio.repair')} info={t('midiStudio.repairInfo')} />
              </div>
            )}

            {/* Speculative decoding: only the large model gains from it */}
            {model === 'large' && (
              <div className="mt-3 flex items-center gap-2 flex-wrap text-xs">
                <Toggle size="sm" accent="purple" checked={speculative} onChange={handleSpeculativeChange}
                  label={t('midiStudio.speculative')} info={t('midiStudio.speculativeInfo')} />
                {speculative && !status?.models?.small?.downloaded && (
                  <span className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('midiStudio.speculativeNeedsSmall')}</span>
                )}
              </div>
            )}
          </div>

          {/* Primary action morphs with the selected model's state:
              download weights -> downloading progress -> transcribe. */}
          <div className="mt-4 flex items-center gap-3 flex-wrap">
            {modelState && !modelState.downloaded ? (
              <button
                onClick={() => { if (modelState.downloading) return; setSubmitError(''); downloadMut.mutate(model); }}
                disabled={modelState.downloading || engineMissing}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-60 disabled:cursor-wait transition-colors flex items-center gap-2"
              >
                {modelState.downloading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    {t('midiStudio.downloadingCta', {
                      model,
                      pct: modelState.totalBytes ? Math.round(100 * modelState.receivedBytes / modelState.totalBytes) : 0,
                    })}
                  </>
                ) : (
                  <>
                    <Download size={16} /> {t('midiStudio.downloadCta', { model })}
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={handleTranscribe}
                disabled={!canTranscribe}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                <Piano size={16} /> {t('midiStudio.transcribe')}
              </button>
            )}
            {modelState && !modelState.downloaded && !modelState.downloading && !status?.hfTokenSet && model !== 'dettato' && (
              <span className="text-xs text-amber-600 dark:text-amber-400">{t('midiStudio.needTokenHint')}</span>
            )}
            {modelState?.downloaded && !sourceAudioUrl && (
              <span className="text-xs text-zinc-400 dark:text-zinc-500">{t('midiStudio.pickTrackHint')}</span>
            )}
            {submitError && <span className="text-xs text-red-500 dark:text-red-400">{submitError}</span>}
          </div>
        </div>

        {/* ── Transcriptions list ── */}
        <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4">
          <div className="text-sm font-semibold text-zinc-900 dark:text-white mb-3">{t('midiStudio.transcriptions')}</div>
          {jobs.length === 0 && (
            <div className="text-xs text-zinc-400 dark:text-zinc-500 py-2">{t('midiStudio.noJobs')}</div>
          )}
          <div className="flex flex-col gap-2">
            {jobs.map(job => {
              const running = job.status === 'queued' || job.status === 'transcribing';
              // A failed job can still hold a partial .mid (the failed badge stays).
              const hasMidi = job.status === 'done' || (job.status === 'failed' && !!job.partial);
              const pct = job.chunksTotal ? Math.round(100 * (job.chunksDone || 0) / job.chunksTotal) : 0;
              return (
                <div key={job.id} className="rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20 px-3 py-2.5">
                  <div className="flex items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate">{job.sourceFileName}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-200 dark:bg-white/10 text-zinc-500 dark:text-zinc-400 capitalize flex-shrink-0">{job.model}</span>
                      </div>
                      {running && (
                        <div className="mt-1">
                          <span className="flex items-center gap-1.5 text-[11px] text-zinc-500 dark:text-zinc-400">
                            <Loader2 size={11} className="animate-spin flex-shrink-0" />
                            {job.status === 'queued'
                              ? t('midiStudio.statusQueued')
                              : t('midiStudio.statusTranscribing', { done: job.chunksDone || 0, total: job.chunksTotal || 0, notes: job.noteCount })}
                          </span>
                          {job.status === 'transcribing' && (
                            <span className="block h-1 mt-1 rounded bg-zinc-200 dark:bg-white/10 overflow-hidden">
                              <span className="block h-full bg-purple-500 transition-all" style={{ width: `${pct}%` }} />
                            </span>
                          )}
                        </div>
                      )}
                      {/* live player appears as soon as transcription starts —
                          play immediately, crossfade original <-> MIDI */}
                      {job.status === 'done' && (
                        <div className="flex items-center gap-1.5 mt-0.5 text-[11px] text-emerald-600 dark:text-emerald-400">
                          <CheckCircle2 size={11} />
                          {t('midiStudio.doneSummary', { notes: job.noteCount, seconds: Math.round(job.durationSec) })}
                        </div>
                      )}
                      {job.status === 'failed' && (
                        <>
                          <div className="mt-0.5 text-[11px] text-red-500 dark:text-red-400 truncate">{job.error || t('midiStudio.statusFailed')}</div>
                          {job.gated && job.model !== 'dettato' && (
                            <div className="mt-1 text-[11px] text-amber-600 dark:text-amber-400 flex items-start gap-1">
                              <AlertTriangle size={11} className="mt-0.5 flex-shrink-0" />
                              <span>
                                {t('midiStudio.gatedHint')}{' '}
                                <a href={HF_MODEL_URLS[job.model]} target="_blank" rel="noreferrer" className="underline">
                                  {t('midiStudio.gatedHintLink', { model: job.model })}
                                </a>
                              </span>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0">
                      {hasMidi && (
                        <>
                          <button
                            onClick={() => setExpandedJob(e => e === job.id ? null : job.id)}
                            className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
                            title={t('midiStudio.preview')}
                          >
                            {expandedJob === job.id ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                          </button>
                          <a
                            href={getMidiFileUrl(job.id)}
                            className="p-1.5 rounded-md text-zinc-500 hover:text-purple-600 dark:hover:text-purple-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
                            title={t('midiStudio.downloadMid')}
                            download
                          >
                            <Download size={15} />
                          </a>
                        </>
                      )}
                      <button
                        onClick={() => handleDelete(job.id)}
                        className="p-1.5 rounded-md text-zinc-500 hover:text-red-500 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
                        title={running ? t('midiStudio.cancel') : t('midiStudio.delete')}
                      >
                        {running ? <X size={15} /> : <Trash2 size={15} />}
                      </button>
                    </div>
                  </div>
                  {job.status === 'transcribing' && (
                    <MidiPlayer key={`${job.id}-live`} jobId={job.id} sourceAudioUrl={job.sourceAudioUrl} live />
                  )}
                  {expandedJob === job.id && hasMidi && (
                    <>
                      <MidiPlayer key={`${job.id}-done`} jobId={job.id} sourceAudioUrl={job.sourceAudioUrl} live={false} />
                      {job.status === 'done' && <MidiExportPanel jobId={job.id} initial={job.exportOptions} />}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Attribution ── */}
        <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20 p-4 text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
          <div className="font-semibold text-zinc-700 dark:text-zinc-300 mb-1">{t('midiStudio.creditsTitle')}</div>
          <p>
            {t('midiStudio.creditsBody')}{' '}
            <span className="text-zinc-700 dark:text-zinc-300">
              Simon Rouard, Michael Krause, Axel Roebel, Carl-Johann Simon-Gabriel, Alexandre Défossez
            </span>.
          </p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
            <a href="https://github.com/muscriptor/muscriptor" target="_blank" rel="noreferrer"
              className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
              <ExternalLink size={11} /> GitHub
            </a>
            <a href="https://arxiv.org/abs/2607.08168" target="_blank" rel="noreferrer"
              className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
              <ExternalLink size={11} /> {t('midiStudio.paper')}
            </a>
            <a href="https://muscriptor.kyutai.org/" target="_blank" rel="noreferrer"
              className="flex items-center gap-1 text-purple-600 dark:text-purple-400 hover:underline">
              <ExternalLink size={11} /> {t('midiStudio.demo')}
            </a>
          </div>
          <p className="mt-2 opacity-80">{t('midiStudio.licenseNote')}</p>
        </div>
      </div>
    </div>
  );
};

export default MidiStudio;
