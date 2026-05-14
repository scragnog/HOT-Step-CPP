// RepaintStudio.tsx — Main Repaint Studio orchestrator
//
// Composes: RepaintWaveform, RegionLyricsEditor, RepaintSettings, ActivitySidebar
// Allows users to select a region of an existing track and regenerate it
// with optionally modified lyrics.

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Upload, Loader2, X, Music, FolderOpen, AlertTriangle } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { generateApi, songApi } from '../../services/api';
import { fetchLrc } from '../../utils/lrcUtils';
import { RepaintWaveform } from './RepaintWaveform';
import { RegionLyricsEditor } from './RegionLyricsEditor';
import { RepaintSettings } from './RepaintSettings';
import { ActivitySidebar } from '../shared/ActivitySidebar';
import {
  addManualQueueItem, updateManualQueueItem,
  completeManualQueueItem, failManualQueueItem,
  useAudioGenQueue,
} from '../../stores/audioGenQueueStore';
import type { Song } from '../../types';

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
  const gp = useGlobalParams();
  const queue = useAudioGenQueue();

  // ── Source song state ──
  const [sourceSong, setSourceSong] = useState<Song | null>(() => restore('sourceSong', null));
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
  const [genProgress, setGenProgress] = useState(0);
  const [genStage, setGenStage] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [queueItemId, setQueueItemId] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [toast, setToast] = useState('');
  const [wipDismissed, setWipDismissed] = useState(false);

  // ── Sidebar resize ──
  const [sidebarWidth, setSidebarWidth] = usePersistedState('hs-repaintSidebarWidth', 320);

  // ── Library picker ──
  const [showLibrary, setShowLibrary] = useState(false);
  const [librarySearch, setLibrarySearch] = useState('');
  const [librarySongs, setLibrarySongs] = useState<Song[]>([]);

  // ── Persist ──
  useEffect(() => { persist('sourceSong', sourceSong); }, [sourceSong]);
  useEffect(() => { persist('sourceAudioUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { persist('sourceName', sourceName); }, [sourceName]);
  useEffect(() => { persist('regionStart', regionStart); }, [regionStart]);
  useEffect(() => { persist('regionEnd', regionEnd); }, [regionEnd]);
  useEffect(() => { persist('lyrics', lyrics); }, [lyrics]);
  useEffect(() => { persist('repaintMode', repaintMode); }, [repaintMode]);
  useEffect(() => { persist('crossfadeFrames', crossfadeFrames); }, [crossfadeFrames]);
  useEffect(() => { persist('styleCaption', styleCaption); }, [styleCaption]);

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
      const res = await fetch('/api/upload/audio', { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Upload failed');
      const { audio_url } = await res.json();
      setSourceAudioUrl(audio_url);
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
    if (!token || !sourceAudioUrl) { showToast('Load source audio first'); return; }

    const effEnd = regionEnd > 0 ? regionEnd : duration;
    if (effEnd <= regionStart) { showToast('Invalid region — end must be after start'); return; }

    setIsGenerating(true);
    try {
      // Map repaint mode to injection ratio
      const modeRatios: Record<string, number> = {
        conservative: 0.7,
        balanced: 0.5,
        aggressive: 0.3,
      };

      const engineParams = gp.getGlobalParams();
      const params: Record<string, any> = {
        ...engineParams,
        customMode: true,
        taskType: 'repaint',
        sourceAudioUrl,
        repaintingStart: regionStart,
        repaintingEnd: effEnd,
        lyrics: lyrics || '[Instrumental]',
        style: styleCaption || engineParams.style || '',
        title: sourceName ? `${sourceName} (Repaint)` : 'Repaint',
        duration: 0, // engine determines from source
        source: 'repaint',
        // Repaint-specific params passed to engine via translateParams
        repaintCrossfadeFrames: crossfadeFrames,
        repaintInjectionRatio: modeRatios[repaintMode] ?? 0.5,
      };

      // Carry latent URL if available from source song
      if (sourceSong?.latentUrl || sourceSong?.latent_url) {
        params.sourceLatentUrl = sourceSong.latentUrl || sourceSong.latent_url;
      }

      const res = await generateApi.submit(params as any, token);
      const jobId = res.jobId;
      setActiveJobId(jobId);
      showToast('Repaint generation started!');

      // Add to queue
      const qId = addManualQueueItem({
        title: params.title,
        artistName: sourceSong?.artistName || '',
        caption: params.style,
      });
      setQueueItemId(qId);
      updateManualQueueItem(qId, { jobId });

      pollJob(jobId, qId);
    } catch (err: any) {
      showToast(`Generation failed: ${err.message}`);
      setIsGenerating(false);
    }
  }, [token, sourceAudioUrl, regionStart, regionEnd, duration, lyrics, styleCaption,
    repaintMode, crossfadeFrames, sourceSong, sourceName, gp]);

  const pollJob = (jobId: string, qId: string) => {
    setGenProgress(0);
    setGenStage('Queued...');
    const startTime = Date.now();
    const iv = setInterval(async () => {
      try {
        const s = await generateApi.status(jobId);
        const rawProg = s.progress;
        const pct = rawProg != null
          ? Math.min(100, Math.max(0, Math.round(rawProg > 1 ? rawProg : rawProg * 100)))
          : undefined;
        if (pct != null) setGenProgress(pct);
        if (s.stage) setGenStage(s.stage);

        updateManualQueueItem(qId, {
          progress: pct,
          stage: s.stage || 'Generating...',
          elapsed: Math.round((Date.now() - startTime) / 1000),
        });

        if (s.status === 'succeeded') {
          clearInterval(iv);
          setGenProgress(100);
          setGenStage('Complete!');
          setIsGenerating(false);
          setActiveJobId(null);
          setQueueItemId(null);
          setRefreshTrigger(p => p + 1);
          showToast('Repaint complete!');
          setTimeout(() => { setGenProgress(0); setGenStage(''); }, 3000);

          const audioUrl = s.result?.audioUrls?.[0] || '';
          const songId = s.result?.songIds?.[0];
          completeManualQueueItem(qId, {
            audioUrl,
            songId,
            masteredAudioUrl: s.result?.masteredAudioUrl,
            audioDuration: s.result?.duration,
          });
        } else if (s.status === 'failed') {
          clearInterval(iv);
          setIsGenerating(false);
          setActiveJobId(null);
          setQueueItemId(null);
          setGenProgress(0);
          setGenStage('');
          showToast(`Failed: ${s.error || 'Unknown error'}`);
          failManualQueueItem(qId, s.error || 'Unknown error');
        }
      } catch {}
    }, 2000);
    setTimeout(() => clearInterval(iv), 1_800_000);
  };

  const handleCancel = async () => {
    if (activeJobId) { try { await generateApi.cancel(activeJobId); } catch {} }
    if (queueItemId) failManualQueueItem(queueItemId, 'Cancelled by user');
    setIsGenerating(false);
    setActiveJobId(null);
    setQueueItemId(null);
    setGenProgress(0);
    setGenStage('');
  };

  const canGenerate = !!sourceAudioUrl && !isGenerating && (regionEnd > regionStart || (regionEnd === 0 && duration > 0));

  // ── Sidebar resize handler ──
  const handleSidebarResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = sidebarWidth;
    const onMove = (ev: MouseEvent) => {
      const newW = Math.min(700, Math.max(240, startW + startX - ev.clientX));
      setSidebarWidth(newW);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [sidebarWidth, setSidebarWidth]);

  // ── Render ──
  return (
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
                      <div className="text-[10px] text-zinc-500 truncate">{sourceSong.artistName}</div>
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
                                {song.artistName || ''} {song.duration ? `· ${typeof song.duration === 'number' ? Math.round(song.duration) + 's' : song.duration}` : ''}
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
            genProgress={genProgress}
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

        {/* ── Right: Resize handle + Activity Sidebar ── */}
        <div
          className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
          onMouseDown={handleSidebarResize}
        >
          <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
        </div>
        <div className="h-full flex-shrink-0 border-l border-white/5 overflow-hidden" style={{ width: sidebarWidth }}>
          <ActivitySidebar
            showToast={(msg) => showToast(msg as string)}
            source="repaint"
            refreshKey={refreshTrigger + queue.completionCounter}
            queueCountColor="bg-pink-500/20 text-pink-300"
            compact={sidebarWidth < 380}
          />
        </div>
      </div>
    </div>
  );
};
