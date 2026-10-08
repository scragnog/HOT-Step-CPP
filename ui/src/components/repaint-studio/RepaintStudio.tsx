// RepaintStudio.tsx — Main Repaint Studio orchestrator
//
// Composes: RepaintWaveform, RegionLyricsEditor, RepaintSettings
// Allows users to select a region of an existing track and regenerate it
// with optionally modified lyrics.

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Upload, Loader2, X, Music, FolderOpen, AlertTriangle } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { songApi } from '../../services/api';
import { submitStudioRender, followStudioRender } from '../../services/repaintLayerWorkflowApi';
import { workflowApi } from '../../services/workflowApi';
import { useBackendStore } from '../../stores/backendStore';
import { fetchLrc } from '../../utils/lrcUtils';
import { RepaintWaveform } from './RepaintWaveform';
import { RegionLyricsEditor } from './RegionLyricsEditor';
import { RepaintSettings } from './RepaintSettings';
import { BackendCapabilityGate } from '../shared/BackendCapabilityGate';
import {
  addManualQueueItem,
  completeManualQueueItem, failManualQueueItem,
} from '../../stores/audioGenQueueStore';
import type { Song } from '../../types';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';

// ── Persist helpers (same pattern as CoverStudio) ──
function persist(key: string, value: unknown) {
  try { localStorage.setItem(`hs-repaint-${key}`, JSON.stringify(value)); } catch {}
}
function restore<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(`hs-repaint-${key}`);
    if (raw !== null) return JSON.parse(raw);
  } catch {}
  return fallback;
}

export const RepaintStudio: React.FC = () => {
  const { token } = useAuth();
  const gp = useGlobalParamsStore();
  const { disguiseArtist } = useDisguiseMode();

  // ── Source song state ──
  const [sourceSong, setSourceSong] = useState<Song | null>(() => restore('sourceSong', null));
  const [sourceAssetId, setSourceAssetId] = useState(() => restore<string>('sourceAssetId', ''));
  const [sourceAudioUrl, setSourceAudioUrl] = useState(() => restore<string>('sourceAudioUrl', ''));
  const [sourceName, setSourceName] = useState(() => restore<string>('sourceName', ''));
  const [isUploading, setIsUploading] = useState(false);
  const [duration, setDuration] = useState(0);

  // ── LRC state ──
  const [lrcText, setLrcText] = useState<string | null>(null);

  // ── Region state ──
  const [regionStart, setRegionStart] = useState(() => restore<number>('regionStart', 0));
  const [regionEnd, setRegionEnd] = useState(() => restore<number>('regionEnd', 0));

  // ── Lyrics ──
  const [lyrics, setLyrics] = useState(() => restore<string>('lyrics', ''));

  // ── Settings ──
  const [repaintMode, setRepaintMode] = useState(() => restore<string>('repaintMode', 'balanced'));
  const [crossfadeFrames, setCrossfadeFrames] = useState(() => restore<number>('crossfadeFrames', 10));
  const [styleCaption, setStyleCaption] = useState(() => restore<string>('styleCaption', ''));

  // ── Generation state ──
  const [isGenerating, setIsGenerating] = useState(false);
  const [genStage, setGenStage] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [queueItemId, setQueueItemId] = useState<string | null>(null);
  const submissionKeyRef = useRef<string | null>(null);
  const [toast, setToast] = useState('');
  const [wipDismissed, setWipDismissed] = useState(false);

  // ── Sidebar resize ──

  // ── Library picker ──
  const [showLibrary, setShowLibrary] = useState(false);
  const [librarySearch, setLibrarySearch] = useState('');
  const [librarySongs, setLibrarySongs] = useState<Song[]>([]);

  // ── Persist ──
  useEffect(() => { persist('sourceSong', sourceSong); }, [sourceSong]);
  useEffect(() => { persist('sourceAssetId', sourceAssetId); }, [sourceAssetId]);
  useEffect(() => { persist('sourceAudioUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { persist('sourceName', sourceName); }, [sourceName]);
  useEffect(() => { persist('regionStart', regionStart); }, [regionStart]);
  useEffect(() => { persist('regionEnd', regionEnd); }, [regionEnd]);
  useEffect(() => { persist('lyrics', lyrics); }, [lyrics]);
  useEffect(() => { persist('repaintMode', repaintMode); }, [repaintMode]);
  useEffect(() => { persist('crossfadeFrames', crossfadeFrames); }, [crossfadeFrames]);
  useEffect(() => { persist('styleCaption', styleCaption); }, [styleCaption]);

  useEffect(() => {
    const id = localStorage.getItem('hs-repaint-workflowJob');
    if (!id || !token) return;
    let mounted = true;
    setIsGenerating(true);
    setActiveJobId(id);
    void followStudioRender(token, id, stage => { if (mounted) setGenStage(stage); })
      .then(async result => {
        if (!mounted) return;
        const { job } = await workflowApi.get(token, id);
        if (!mounted) return;
        const captured = job.input as { sourceName?: string; styleCaption?: string };
        const qId = addManualQueueItem({ title: captured.sourceName ? `${captured.sourceName} (Repaint)` : 'Repaint',
          artistName: '', caption: captured.styleCaption || '' });
        const audio = result.audio || {};
        completeManualQueueItem(qId, { audioUrl: audio.audioUrls?.[0] || '', songId: audio.songIds?.[0],
          masteredAudioUrl: audio.masteredAudioUrl, noAdapterAudioUrl: audio.noAdapterAudioUrl,
          audioDuration: audio.duration });
        showToast('Repaint complete!');
      }).catch(err => { if (mounted) showToast(`Repaint ${String((err as Error).message)}`); })
      .finally(() => {
        if (localStorage.getItem('hs-repaint-workflowJob') === id) localStorage.removeItem('hs-repaint-workflowJob');
        if (mounted) { setIsGenerating(false); setActiveJobId(null); }
      });
    return () => { mounted = false; };
  }, [token]);

  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 4000); };

  // ── Load LRC when source audio changes ──
  useEffect(() => {
    if (!sourceAudioUrl) { setLrcText(null); return; }
    fetchLrc(sourceAudioUrl).then(setLrcText);
  }, [sourceAudioUrl]);

  // ── Load library songs for picker ──
  useEffect(() => {
    if (showLibrary && token) {
      songApi.list(token)
        .then(({ songs }) => setLibrarySongs(songs))
        .catch(() => {});
    }
  }, [showLibrary, token]);

  const filteredSongs = useMemo(() => {
    if (!librarySearch.trim()) return librarySongs;
    const q = librarySearch.toLowerCase();
    return librarySongs.filter(s =>
      s.title?.toLowerCase().includes(q) ||
      s.artistName?.toLowerCase().includes(q) ||
      s.style?.toLowerCase().includes(q)
    );
  }, [librarySongs, librarySearch]);

  // ── Select song from library ──
  const handleSelectSong = useCallback((song: Song) => {
    setSourceSong(song);
    setSourceAssetId('');
    setSourceAudioUrl(song.audioUrl || song.audio_url || '');
    setSourceName(song.title || 'Library Track');
    setLyrics(song.lyrics || '');
    setStyleCaption(song.style || song.caption || '');
    setRegionStart(0);
    setRegionEnd(0);
    setShowLibrary(false);
  }, []);

  // ── File upload ──
  const handleFileUpload = useCallback(async (file: File) => {
    if (!token) return;
    setIsUploading(true);
    try {
      const fd = new FormData();
      fd.append('audio', file);
      const res = await fetch('/api/upload/audio', { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd });
      if (!res.ok) throw new Error('Upload failed');
      const { audio_url, asset_id } = await res.json();
      setSourceAudioUrl(audio_url);
      setSourceAssetId(asset_id);
      setSourceName(file.name);
      setSourceSong(null);
      setRegionStart(0);
      setRegionEnd(0);
      showToast('Audio uploaded');
    } catch (err: any) {
      showToast(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  }, [token]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) handleFileUpload(file);
  }, [handleFileUpload]);

  // ── Clear source ──
  const handleClear = useCallback(() => {
    setSourceSong(null);
    setSourceAssetId('');
    setSourceAudioUrl('');
    setSourceName('');
    setLyrics('');
    setStyleCaption('');
    setLrcText(null);
    setRegionStart(0);
    setRegionEnd(0);
    setDuration(0);
  }, []);

  // ── Region change ──
  const handleRegionChange = useCallback((start: number, end: number) => {
    setRegionStart(start);
    setRegionEnd(end);
  }, []);

  // ── Generate ──
  const handleGenerate = useCallback(async () => {
    if (submissionKeyRef.current || localStorage.getItem('hs-repaint-workflowJob')) return;
    if (!token || !sourceAudioUrl) { showToast('Load source audio first'); return; }
    const source = sourceSong?.id
      ? { kind: 'song' as const, id: String(sourceSong.id), expectedUrl: sourceAudioUrl }
      : sourceAssetId
        ? { kind: 'asset' as const, id: sourceAssetId, expectedUrl: sourceAudioUrl }
        : null;
    if (!source) { showToast('Select the source again to verify ownership'); return; }
    const effEnd = regionEnd > 0 ? regionEnd : duration;
    if (effEnd <= regionStart) { showToast('Invalid region — end must be after start'); return; }
    const key = crypto.randomUUID();
    submissionKeyRef.current = key;
    setIsGenerating(true);
    let submittedJobId: string | null = null;
    try {
      const engineParams = gp.getGlobalParams();
      const { job } = await submitStudioRender(token, 'repaint-render', {
        source, expectedBackend: useBackendStore.getState().activeBackendId,
        engineParams, regionStart, regionEnd: effEnd, lyrics, styleCaption,
        sourceName, repaintMode: repaintMode as 'conservative' | 'balanced' | 'aggressive',
        crossfadeFrames,
      }, key);
      if (submissionKeyRef.current !== key) { await workflowApi.cancel(token, job.id); return; }
      setActiveJobId(job.id);
      submittedJobId = job.id;
      localStorage.setItem('hs-repaint-workflowJob', job.id);
      showToast('Repaint generation started!');
      const qId = addManualQueueItem({
        title: sourceName ? `${sourceName} (Repaint)` : 'Repaint',
        artistName: sourceSong?.artistName || '',
        caption: styleCaption || String(engineParams.style || ''),
      });
      setQueueItemId(qId);
      await followStudioRender(token, job.id, stage => setGenStage(stage)).then(result => {
        setGenStage('Complete!');
        const audio = result.audio || {};
        completeManualQueueItem(qId, {
          audioUrl: audio.audioUrls?.[0] || '', songId: audio.songIds?.[0],
          masteredAudioUrl: audio.masteredAudioUrl, noAdapterAudioUrl: audio.noAdapterAudioUrl,
          audioDuration: audio.duration,
        });
        showToast('Repaint complete!');
      }).catch(err => {
        failManualQueueItem(qId, (err as Error).message);
        showToast(`Failed: ${(err as Error).message}`);
      });
    } catch (err: any) {
      showToast(`Generation failed: ${err.message}`);
    } finally {
      if (submissionKeyRef.current === key) {
        submissionKeyRef.current = null;
        if (localStorage.getItem('hs-repaint-workflowJob') === submittedJobId) localStorage.removeItem('hs-repaint-workflowJob');
        setIsGenerating(false);
        setActiveJobId(null);
        setQueueItemId(null);
      }
    }
  }, [token, sourceAudioUrl, sourceSong, sourceAssetId, regionStart, regionEnd, duration,
    lyrics, styleCaption, sourceName, repaintMode, crossfadeFrames, gp]);

  const handleCancel = async () => {
    submissionKeyRef.current = null;
    if (activeJobId && token) { try { await workflowApi.cancel(token, activeJobId); } catch {} }
    localStorage.removeItem('hs-repaint-workflowJob');
    if (queueItemId) failManualQueueItem(queueItemId, 'Cancelled by user');
    setIsGenerating(false);
    setActiveJobId(null);
    setQueueItemId(null);
    setGenStage('');
  };

  const canGenerate = !!sourceAudioUrl && !isGenerating && (regionEnd > regionStart || (regionEnd === 0 && duration > 0));


  // ── Render ──
  return (
    <BackendCapabilityGate feature="repaint">
    <div className="flex flex-col w-full h-full bg-zinc-50 dark:bg-suno overflow-hidden">
      {/* Toast */}
      {toast && (
        <div className="absolute top-16 right-6 z-50 px-4 py-2 rounded-xl bg-white dark:bg-zinc-900 text-white text-sm shadow-xl border border-zinc-300 dark:border-white/10 animate-in fade-in slide-in-from-top-2">
          {toast}
        </div>
      )}

      {/* WIP notice banner */}
      {!wipDismissed && (
        <div className="flex-shrink-0 px-4 py-2.5 bg-amber-500/10 border-b border-amber-500/20 flex items-center gap-3">
          <AlertTriangle size={16} className="text-amber-400 flex-shrink-0" />
          <p className="text-xs text-amber-300/90 flex-1">
            <span className="font-semibold">Work in progress</span> — This component is under heavy development and may not work as expected.
          </p>
          <button
            onClick={() => setWipDismissed(true)}
            className="flex-shrink-0 w-5 h-5 rounded flex items-center justify-center text-amber-400/60 hover:text-amber-300 hover:bg-amber-500/10 transition-colors"
            title="Dismiss"
          >
            <X size={12} />
          </button>
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* ── Left Panel: Source + Waveform ── */}
        <div className="w-[360px] flex-shrink-0 flex flex-col border-r border-white/5 overflow-y-auto">
          {/* Source selector */}
          <div className="p-4 border-b border-white/5">
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <Music size={14} className="text-pink-400" />
              Source Track
            </h3>

            {sourceAudioUrl ? (
              /* Source loaded */
              <div className="space-y-3">
                <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/[0.03] border border-white/5">
                  <div className="w-8 h-8 rounded-lg bg-pink-500/10 flex items-center justify-center flex-shrink-0">
                    <Music size={14} className="text-pink-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">{sourceName}</div>
                    {sourceSong?.artistName && (
                      <div className="text-[10px] text-zinc-500 truncate">{disguiseArtist(sourceSong.artistName)}</div>
                    )}
                  </div>
                  <button
                    onClick={handleClear}
                    className="w-6 h-6 rounded-md flex items-center justify-center text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                    title="Clear source"
                  >
                    <X size={12} />
                  </button>
                </div>

                {/* Waveform with region selector */}
                <RepaintWaveform
                  audioUrl={sourceAudioUrl}
                  regionStart={regionStart}
                  regionEnd={regionEnd}
                  onRegionChange={handleRegionChange}
                  onDurationChange={setDuration}
                />
              </div>
            ) : (
              /* Empty state — upload or pick from library */
              <div className="space-y-3">
                {/* Upload drop zone */}
                <div
                  onDragOver={e => e.preventDefault()}
                  onDrop={handleDrop}
                  className="relative border-2 border-dashed border-white/10 rounded-xl p-6 text-center
                    hover:border-pink-500/30 hover:bg-pink-500/5 transition-colors cursor-pointer group"
                  onClick={() => {
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'audio/*';
                    input.onchange = (e) => {
                      const file = (e.target as HTMLInputElement).files?.[0];
                      if (file) handleFileUpload(file);
                    };
                    input.click();
                  }}
                >
                  {isUploading ? (
                    <Loader2 size={24} className="mx-auto text-pink-400 animate-spin" />
                  ) : (
                    <>
                      <Upload size={24} className="mx-auto text-zinc-500 group-hover:text-pink-400 transition-colors mb-2" />
                      <p className="text-xs text-zinc-500 group-hover:text-zinc-400">
                        Drop audio file or click to upload
                      </p>
                    </>
                  )}
                </div>

                {/* Pick from library button */}
                <button
                  onClick={() => setShowLibrary(!showLibrary)}
                  className="w-full px-3 py-2 rounded-xl bg-white/[0.03] border border-white/5
                    text-xs text-zinc-400 hover:text-white hover:border-pink-500/30
                    transition-colors flex items-center justify-center gap-2"
                >
                  <FolderOpen size={12} />
                  {showLibrary ? 'Hide Library' : 'Pick from Library'}
                </button>

                {/* Library picker */}
                {showLibrary && (
                  <div className="border border-white/5 rounded-xl overflow-hidden max-h-[300px] flex flex-col">
                    <div className="p-2 border-b border-white/5">
                      <input
                        value={librarySearch}
                        onChange={e => setLibrarySearch(e.target.value)}
                        className="w-full px-2.5 py-1.5 rounded-lg bg-black/20 border border-white/10
                          text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500"
                        placeholder="Search library..."
                      />
                    </div>
                    <div className="flex-1 overflow-y-auto">
                      {filteredSongs.length === 0 ? (
                        <div className="p-4 text-center text-xs text-zinc-600">No tracks found</div>
                      ) : (
                        filteredSongs.slice(0, 50).map(song => (
                          <button
                            key={song.id}
                            onClick={() => handleSelectSong(song)}
                            className="w-full text-left px-3 py-2 hover:bg-white/5 transition-colors
                              border-b border-white/[0.02] last:border-0 flex items-center gap-2"
                          >
                            <div className="flex-1 min-w-0">
                              <div className="text-xs text-white truncate">{song.title || 'Untitled'}</div>
                              <div className="text-[10px] text-zinc-500 truncate">
                                {disguiseArtist(song.artistName || '')} {song.duration ? `· ${typeof song.duration === 'number' ? Math.round(song.duration) + 's' : song.duration}` : ''}
                              </div>
                            </div>
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Settings panel */}
          <RepaintSettings
            repaintMode={repaintMode}
            onRepaintModeChange={setRepaintMode}
            crossfadeFrames={crossfadeFrames}
            onCrossfadeFramesChange={setCrossfadeFrames}
            styleCaption={styleCaption}
            onStyleCaptionChange={setStyleCaption}
            canGenerate={canGenerate}
            isGenerating={isGenerating}
            genStage={genStage}
            onGenerate={handleGenerate}
            onCancel={handleCancel}
          />
        </div>

        {/* ── Center: Lyrics Editor ── */}
        <div className="flex-1 flex flex-col overflow-hidden border-r border-white/5">
          <RegionLyricsEditor
            lrcText={lrcText}
            fallbackLyrics={lyrics}
            regionStart={regionStart}
            regionEnd={regionEnd}
            duration={duration}
            onLyricsChange={setLyrics}
          />
        </div>

      </div>
    </div>
    </BackendCapabilityGate>
  );
};
