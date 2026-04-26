// CoverStudio.tsx — Main Cover Studio orchestrator
// Composes: SourcePanel, ArtistSettingsPanel, RecentCovers
import React, { useState, useEffect } from 'react';
import { Guitar, Search, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { generateApi } from '../../services/api';
import { lireekApi, type Artist, type AlbumPreset } from '../../services/lireekApi';
import { SourcePanel } from './SourcePanel';
import { ArtistSettingsPanel } from './ArtistSettingsPanel';
import { RecentCovers } from './RecentCovers';
import {
  persist, restore, getTrackCache, saveTrackCacheEntry, transposeKey,
  type AudioMetadata, type AudioAnalysis,
} from './coverStudioUtils';

export const CoverStudio: React.FC = () => {
  const { token } = useAuth();
  const gp = useGlobalParams();

  // ── Source audio state ──
  const [sourceFileName, setSourceFileName] = useState(() => restore<string>('sourceFileName', ''));
  const [sourceAudioUrl, setSourceAudioUrl] = useState(() => restore<string>('sourceAudioUrl', ''));
  const [metadata, setMetadata] = useState<AudioMetadata | null>(() => restore('metadata', null));
  const [analysis, setAnalysis] = useState<AudioAnalysis | null>(() => restore('analysis', null));
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // ── Song details ──
  const [songArtist, setSongArtist] = useState(() => restore<string>('songArtist', ''));
  const [songTitle, setSongTitle] = useState(() => restore<string>('songTitle', ''));
  const [lyrics, setLyrics] = useState(() => restore<string>('lyrics', ''));
  const [isSearchingLyrics, setIsSearchingLyrics] = useState(false);

  // ── Target artist ──
  const [artists, setArtists] = useState<Artist[]>([]);
  const [selectedArtistId, setSelectedArtistId] = useState<number | null>(() => restore('selectedArtistId', null));
  const [selectedPreset, setSelectedPreset] = useState<AlbumPreset | null>(() => restore('selectedPreset', null));
  const [artistCaption, setArtistCaption] = useState(() => restore<string>('artistCaption', ''));
  const [artistPresets, setArtistPresets] = useState<{ lsId: number; album: string; preset: AlbumPreset | null }[]>([]);
  const [isLoadingArtists, setIsLoadingArtists] = useState(false);

  // ── Cover settings ──
  const [audioCoverStrength, setAudioCoverStrength] = useState(() => restore<number>('audioCoverStrength', 0.5));
  const [coverNoiseStrength, setCoverNoiseStrength] = useState(() => restore<number>('coverNoiseStrength', 0));
  const [tempoScale, setTempoScale] = useState(() => restore<number>('tempoScale', 1.0));
  const [pitchShift, setPitchShift] = useState(() => restore<number>('pitchShift', 0));

  // ── Generation ──
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStage, setGenStage] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [toast, setToast] = useState('');

  // ── Persist ──
  useEffect(() => { persist('sourceFileName', sourceFileName); }, [sourceFileName]);
  useEffect(() => { persist('sourceAudioUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { persist('metadata', metadata); }, [metadata]);
  useEffect(() => { persist('analysis', analysis); }, [analysis]);
  useEffect(() => { persist('songArtist', songArtist); }, [songArtist]);
  useEffect(() => { persist('songTitle', songTitle); }, [songTitle]);
  useEffect(() => { persist('lyrics', lyrics); }, [lyrics]);
  useEffect(() => { persist('selectedArtistId', selectedArtistId); }, [selectedArtistId]);
  useEffect(() => { persist('selectedPreset', selectedPreset); }, [selectedPreset]);
  useEffect(() => { persist('artistCaption', artistCaption); }, [artistCaption]);
  useEffect(() => { persist('audioCoverStrength', audioCoverStrength); }, [audioCoverStrength]);
  useEffect(() => { persist('coverNoiseStrength', coverNoiseStrength); }, [coverNoiseStrength]);
  useEffect(() => { persist('tempoScale', tempoScale); }, [tempoScale]);
  useEffect(() => { persist('pitchShift', pitchShift); }, [pitchShift]);

  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 4000); };

  // ── Load artists on mount ──
  useEffect(() => {
    setIsLoadingArtists(true);
    lireekApi.listArtists()
      .then(res => {
        setArtists(res.artists);
        if (selectedArtistId && !selectedPreset) {
          const a = res.artists.find(x => x.id === selectedArtistId);
          if (a) loadArtistPresets(a);
        }
      })
      .catch(() => showToast('Failed to load artists'))
      .finally(() => setIsLoadingArtists(false));
  }, []);

  // ── File upload + analysis pipeline ──
  const handleFileSelected = async (file: File) => {
    if (!token) { showToast('Please sign in first'); return; }
    setSourceFileName(file.name);

    // Check track cache
    const cached = getTrackCache()[file.name];
    if (cached) {
      showToast('Loaded from cache!');
      if (cached.artist) setSongArtist(cached.artist);
      if (cached.title) setSongTitle(cached.title);
      if (cached.lyrics) setLyrics(cached.lyrics);
      setMetadata({ artist: cached.artist || '', title: cached.title || '', album: cached.album || '', duration: cached.duration });
      setAnalysis({ bpm: cached.bpm, key: cached.key, scale: cached.scale });
      // Still upload the file
      setIsUploading(true);
      try {
        const fd = new FormData(); fd.append('audio', file);
        const r = await fetch('/api/upload/audio', { method: 'POST', body: fd });
        if (r.ok) { const d = await r.json(); setSourceAudioUrl(d.audio_url || ''); }
      } catch {} finally { setIsUploading(false); }
      return;
    }

    // Full pipeline
    setIsUploading(true);
    let extractedArtist = '', extractedTitle = '', extractedAlbum = '';
    let extractedDuration: number | null = null;
    try {
      // 1. Metadata
      const metaFd = new FormData(); metaFd.append('audio', file);
      const metaRes = await fetch('/api/analyze/metadata', { method: 'POST', body: metaFd });
      if (metaRes.ok) {
        const meta = await metaRes.json();
        setMetadata(meta);
        extractedArtist = meta.artist || '';
        extractedTitle = meta.title || '';
        extractedAlbum = meta.album || '';
        extractedDuration = meta.duration;
        if (meta.artist) setSongArtist(meta.artist);
        if (meta.title) setSongTitle(meta.title);
      }
      // 2. Upload
      const upFd = new FormData(); upFd.append('audio', file);
      const upRes = await fetch('/api/upload/audio', { method: 'POST', body: upFd });
      if (!upRes.ok) throw new Error('Upload failed');
      const upData = await upRes.json();
      const audioUrl = upData.audio_url || '';
      setSourceAudioUrl(audioUrl);

      // 3. Essentia analysis
      setIsUploading(false); setIsAnalyzing(true);
      let bpm = 120, key = 'C major', scale: string | undefined;
      const anRes = await fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audioUrl }),
      });
      if (anRes.ok) {
        const d = await anRes.json();
        bpm = d.bpm || 120; key = `${d.key || 'C'} ${d.scale || 'major'}`; scale = d.scale;
        setAnalysis({ bpm, key, scale });
      }
      // 4. Cache
      saveTrackCacheEntry(file.name, { artist: extractedArtist, title: extractedTitle, album: extractedAlbum, duration: extractedDuration, bpm, key, scale });
    } catch (err: any) {
      showToast(`Error: ${err.message}`);
    } finally { setIsUploading(false); setIsAnalyzing(false); }
  };

  // ── Lyrics search ──
  const handleSearchLyrics = async () => {
    if (!songArtist.trim() || !songTitle.trim()) { showToast('Enter artist and title first'); return; }
    setIsSearchingLyrics(true);
    try {
      const result = await lireekApi.searchSongLyrics(songArtist.trim(), songTitle.trim());
      setLyrics(result.lyrics);
      if (result.title) setSongTitle(result.title);
      showToast('Lyrics found!');
      if (sourceFileName) saveTrackCacheEntry(sourceFileName, { lyrics: result.lyrics, artist: songArtist.trim(), title: result.title || songTitle.trim() });
    } catch (err: any) { showToast(err.message || 'No lyrics found'); }
    finally { setIsSearchingLyrics(false); }
  };

  // ── Artist preset loading ──
  const loadArtistPresets = async (artist: Artist) => {
    try {
      const { lyrics_sets } = await lireekApi.listLyricsSets(artist.id);
      const results: typeof artistPresets = [];
      for (const ls of lyrics_sets) {
        try {
          const { preset } = await lireekApi.getPreset(ls.id);
          results.push({ lsId: ls.id, album: ls.album || ls.id.toString(), preset });
        } catch { results.push({ lsId: ls.id, album: ls.album || ls.id.toString(), preset: null }); }
      }
      setArtistPresets(results);
      // Find caption
      let caption = '';
      for (const ls of lyrics_sets) {
        try {
          const { generations } = await lireekApi.listGenerations(undefined, ls.id);
          const wc = generations.find(g => g.caption?.trim());
          if (wc?.caption) { caption = wc.caption; break; }
        } catch {}
      }
      setArtistCaption(caption);
      // Pick adapter preset
      const withAdapter = results.find(p => p.preset?.adapter_path);
      if (withAdapter?.preset) {
        setSelectedPreset(withAdapter.preset);
        if (withAdapter.preset.audio_cover_strength != null) setAudioCoverStrength(withAdapter.preset.audio_cover_strength);
      } else { setSelectedPreset(results[0]?.preset || null); }
    } catch { setSelectedPreset(null); setArtistPresets([]); setArtistCaption(''); }
  };

  const handleSelectArtist = async (artist: Artist) => {
    setSelectedArtistId(artist.id);
    await loadArtistPresets(artist);
  };

  // ── Generation ──
  const handleGenerate = async () => {
    if (!token || !sourceAudioUrl || !lyrics.trim()) { showToast('Missing source audio or lyrics'); return; }
    setIsGenerating(true);
    try {
      const selectedArtist = artists.find(a => a.id === selectedArtistId);
      const sourceBpm = analysis?.bpm || 120;
      const sourceKey = analysis?.key || 'C major';
      const targetBpm = Math.round(sourceBpm * tempoScale);
      const targetKey = pitchShift !== 0 ? transposeKey(sourceKey, pitchShift) : sourceKey;

      // Start from global engine params
      const engineParams = gp.getGlobalParams();

      // Override with cover-specific params
      const params: Record<string, any> = {
        ...engineParams,
        customMode: true,
        lyrics,
        style: artistCaption || engineParams.style || '',
        title: `${songTitle || 'Cover'} (${selectedArtist?.name || 'Cover'})`,
        taskType: 'cover',
        sourceAudioUrl,
        audioCoverStrength,
        coverNoiseStrength,
        bpm: targetBpm,
        keyScale: targetKey,
        duration: 0,
        instrumental: false,
        source: 'cover-studio',
        artistName: selectedArtist?.name || songArtist || '',
        sourceArtist: songArtist || '',
      };
      if (tempoScale !== 1.0) params.tempoScale = tempoScale;
      if (pitchShift !== 0) params.pitchShift = pitchShift;

      // Apply album preset adapter (overrides global adapter)
      if (selectedPreset?.adapter_path) {
        params.loraPath = selectedPreset.adapter_path;
        params.loraScale = selectedPreset.adapter_scale ?? gp.adapterScale;
        if (selectedPreset.adapter_group_scales) params.adapterGroupScales = selectedPreset.adapter_group_scales;
      }
      // Reference track + matchering
      if (selectedPreset?.reference_track_path) {
        params.referenceAudioUrl = selectedPreset.reference_track_path;
        params.masteringEnabled = true;
        params.masteringReference = selectedPreset.reference_track_path;
      }

      const res = await generateApi.submit(params as any, token);
      const jobId = res.jobId;
      showToast(`Cover generation started!`);
      setActiveJobId(jobId);
      pollJob(jobId);
    } catch (err: any) { showToast(`Generation failed: ${err.message}`); setIsGenerating(false); }
  };

  const pollJob = (jobId: string) => {
    setGenProgress(0); setGenStage('Queued...');
    const iv = setInterval(async () => {
      try {
        const s = await generateApi.status(jobId);
        if (s.progress != null) setGenProgress(Math.round(s.progress * 100));
        if (s.stage) setGenStage(s.stage);
        if (s.status === 'succeeded') {
          clearInterval(iv); setGenProgress(100); setGenStage('Complete!');
          setIsGenerating(false); setActiveJobId(null);
          setRefreshTrigger(p => p + 1); showToast('Cover generated!');
          setTimeout(() => { setGenProgress(0); setGenStage(''); }, 3000);
        } else if (s.status === 'failed') {
          clearInterval(iv); setIsGenerating(false); setActiveJobId(null);
          setGenProgress(0); setGenStage('');
          showToast(`Failed: ${s.error || 'Unknown error'}`);
        }
      } catch {}
    }, 2000);
    setTimeout(() => clearInterval(iv), 1_800_000);
  };

  const handleCancel = async () => {
    if (activeJobId) { try { await generateApi.cancel(activeJobId); } catch {} }
    setIsGenerating(false); setActiveJobId(null); setGenProgress(0); setGenStage('');
  };

  const handleClearSource = () => {
    setSourceFileName(''); setSourceAudioUrl('');
    setMetadata(null); setAnalysis(null);
    setSongArtist(''); setSongTitle(''); setLyrics('');
  };

  const canGenerate = !!sourceAudioUrl && !!lyrics.trim() && !isGenerating;

  // ── Render ──
  return (
    <div className="flex flex-col w-full h-full bg-zinc-50 dark:bg-suno-panel overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-zinc-200 dark:border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Guitar className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-zinc-900 dark:text-white">Cover Studio</h1>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">Create AI covers of existing songs</p>
          </div>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="absolute top-16 right-6 z-50 px-4 py-2 rounded-xl bg-zinc-900 text-white text-sm shadow-xl border border-white/10 animate-in fade-in slide-in-from-top-2">
          {toast}
        </div>
      )}

      {/* Main workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Source Audio */}
        <SourcePanel
          sourceFileName={sourceFileName} metadata={metadata} analysis={analysis}
          isUploading={isUploading} isAnalyzing={isAnalyzing}
          onFileSelected={handleFileSelected} onClear={handleClearSource}
        />

        {/* Center: Lyrics */}
        <div className="flex-1 flex flex-col overflow-hidden border-r border-zinc-200 dark:border-white/5">
          <div className="flex-shrink-0 px-4 py-3 border-b border-zinc-200 dark:border-white/5">
            <div className="flex items-center justify-between gap-2">
              {/* Artist + Title inputs */}
              <div className="flex-1 flex gap-2">
                <input value={songArtist} onChange={e => setSongArtist(e.target.value)}
                  placeholder="Artist" className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 focus:outline-none focus:border-cyan-500" />
                <input value={songTitle} onChange={e => setSongTitle(e.target.value)}
                  placeholder="Song title" className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 focus:outline-none focus:border-cyan-500" />
              </div>
              <button onClick={handleSearchLyrics} disabled={isSearchingLyrics || !songArtist.trim() || !songTitle.trim()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-xs font-medium transition-colors disabled:opacity-50">
                {isSearchingLyrics ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
                Genius
              </button>
            </div>
          </div>
          <div className="flex-1 p-4">
            <textarea value={lyrics} onChange={e => setLyrics(e.target.value)}
              placeholder="Lyrics will appear here after searching Genius, or paste them manually..."
              className="w-full h-full resize-none bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-xl px-4 py-3 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-cyan-500 transition-colors font-mono leading-relaxed" />
          </div>
          {/* Recent Covers below lyrics */}
          <div className="flex-shrink-0 border-t border-zinc-200 dark:border-white/5 p-4 max-h-[280px] overflow-y-auto scrollbar-hide">
            <RecentCovers refreshTrigger={refreshTrigger} />
          </div>
        </div>

        {/* Right: Artist + Settings */}
        <ArtistSettingsPanel
          artists={artists} isLoadingArtists={isLoadingArtists}
          selectedArtistId={selectedArtistId} onSelectArtist={handleSelectArtist}
          artistPresets={artistPresets} selectedPreset={selectedPreset}
          onSelectPreset={setSelectedPreset}
          audioCoverStrength={audioCoverStrength} onAudioCoverStrength={setAudioCoverStrength}
          coverNoiseStrength={coverNoiseStrength} onCoverNoiseStrength={setCoverNoiseStrength}
          tempoScale={tempoScale} onTempoScale={setTempoScale}
          pitchShift={pitchShift} onPitchShift={setPitchShift}
          analysis={analysis} canGenerate={canGenerate}
          isGenerating={isGenerating} genProgress={genProgress} genStage={genStage}
          onGenerate={handleGenerate} onCancel={handleCancel}
        />
      </div>
    </div>
  );
};
