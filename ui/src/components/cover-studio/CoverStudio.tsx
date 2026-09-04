// CoverStudio.tsx — Main Cover Studio orchestrator
// Composes: SourcePanel, ArtistSettingsPanel, ActivitySidebar
import React, { useState, useEffect, useCallback } from 'react';
import type { Song } from '../../types';
import { Search, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { DEFAULT_SETTINGS, type AppSettings } from '../settings/SettingsPanel';
import { generateApi } from '../../services/api';
import { createGenerationTimer, getGenerationTimeoutMinutes } from '../../utils/generationTimer';
import { lireekApi, type Artist, type AlbumPreset } from '../../services/lireekApi';
import {
  startSeparation, waitForCompletion, recombineStems, getStemAudioUrl,
  type SeparationLevel,
} from '../../services/supersepApi';
import { SourcePanel } from './SourcePanel';
import { ArtistSettingsPanel } from './ArtistSettingsPanel';
import { ActivitySidebar } from '../shared/ActivitySidebar';
import { BackendCapabilityGate } from '../shared/BackendCapabilityGate';
import { StemMixer, type StemControl, type MixerStemInfo } from '../shared/StemMixer';
import {
  addManualQueueItem, updateManualQueueItem,
  completeManualQueueItem, failManualQueueItem,
  useAudioGenQueueSelector,
} from '../../stores/audioGenQueueStore';
import {
  persist, restore, getTrackCache, saveTrackCacheEntry, transposeKey,
  type AudioMetadata, type AudioAnalysis,
} from './coverStudioUtils';
import type { LatentMetadata } from '../shared/LatentImport';
import { loadSelections, saveSelections } from '../lyric-studio/ProviderSelector';

// ── Serial cover-generation queue ────────────────────────────────────────────
// Lets the user stack multiple cover generations (different settings) without
// waiting — same pattern as InstaGen's queue. Jobs run one at a time; the engine
// serializes anyway, this keeps the client-side recombine→submit→poll ordered.
const _coverQueue: Array<() => Promise<void>> = [];
let _coverRunning = false;
// Survives Cover Studio remounts (it unmounts when you leave the tab) so a
// previously-sent library track isn't re-applied every time you revisit (#61).
let _lastConsumedCoverTs = 0;
function enqueueCoverJob(fn: () => Promise<void>) {
  _coverQueue.push(fn);
  if (!_coverRunning) _drainCoverQueue();
}
async function _drainCoverQueue() {
  _coverRunning = true;
  while (_coverQueue.length > 0) {
    const job = _coverQueue.shift()!;
    try { await job(); } catch { /* each job handles its own errors */ }
  }
  _coverRunning = false;
}

interface CoverStudioProps {
  /** A library track to load as the cover source (from "Send to Cover Studio", #61). */
  coverSource?: { song: Song; timestamp: number } | null;
}

export const CoverStudio: React.FC<CoverStudioProps> = ({ coverSource }) => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const gp = useGlobalParamsStore();
  const [settings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

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

  // ── Cover caption LLM ──
  const [coverCaptionProvider, setCoverCaptionProvider] = useState(() => loadSelections().coverCaption.provider);
  const [coverCaptionModel, setCoverCaptionModel] = useState(() => loadSelections().coverCaption.model);
  const [isGeneratingCaption, setIsGeneratingCaption] = useState(false);

  // ── Cover settings ──
  const [audioCoverStrength, setAudioCoverStrength] = useState(() => restore<number>('audioCoverStrength', 0.5));
  const [coverNoiseStrength, setCoverNoiseStrength] = useState(() => restore<number>('coverNoiseStrength', 0));
  const [coverNoiseMethod, setCoverNoiseMethod] = useState(() => restore<string>('coverNoiseMethod', ''));
  const [tempoScale, setTempoScale] = useState(() => restore<number>('tempoScale', 1.0));
  const [pitchShift, setPitchShift] = useState(() => restore<number>('pitchShift', 0));
  const [bpmCorrection, setBpmCorrection] = useState(() => restore<number>('bpmCorrection', 1));
  const [bpmOverride, setBpmOverride] = useState<number | null>(() => restore<number | null>('bpmOverride', null));
  const [keyOverride, setKeyOverride] = useState<string | null>(() => restore<string | null>('keyOverride', null));
  const [noFsq, setNoFsq] = useState(() => restore<boolean>('noFsq', false));
  const [instrumental, setInstrumental] = useState(() => restore<boolean>('coverInstrumental', false));
  const [sourceLatentUrl, setSourceLatentUrl] = useState(() => restore<string>('sourceLatentUrl', ''));
  const [vocalLanguage, setVocalLanguage] = useState(() => restore<string>('coverVocalLanguage', 'en'));
  const [timbreOverridePath, setTimbreOverridePath] = useState(() => restore<string>('coverTimbreOverride', ''));

  // ── Generation ──
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStage, setGenStage] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [toast, setToast] = useState('');
  const [queueItemId, setQueueItemId] = useState<string | null>(null);
  const completionCounter = useAudioGenQueueSelector(s => s.completionCounter);

  // ── Sidebar resize ──
  const [sidebarWidth, setSidebarWidth] = usePersistedState('hs-activitySidebarWidth', 320);
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

  // ── Advanced Mode (SuperSep) ──
  const [advancedMode, setAdvancedMode] = useState(false);
  const [sepLevel, setSepLevel] = useState<SeparationLevel>(() => restore('sepLevel', 1) as SeparationLevel);
  const [isSeparating, setIsSeparating] = useState(false);
  const [sepProgress, setSepProgress] = useState(0);
  const [sepMessage, setSepMessage] = useState('');
  const [sepJobId, setSepJobId] = useState<string | null>(null);
  const [sepStems, setSepStems] = useState<MixerStemInfo[] | null>(null);
  const [stemControls, setStemControls] = useState<StemControl[]>([]);
  const [showMixer, setShowMixer] = useState(false);


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
  useEffect(() => { persist('coverNoiseMethod', coverNoiseMethod); }, [coverNoiseMethod]);
  useEffect(() => { persist('tempoScale', tempoScale); }, [tempoScale]);
  useEffect(() => { persist('pitchShift', pitchShift); }, [pitchShift]);
  useEffect(() => { persist('bpmCorrection', bpmCorrection); }, [bpmCorrection]);
  useEffect(() => { persist('bpmOverride', bpmOverride); }, [bpmOverride]);
  useEffect(() => { persist('keyOverride', keyOverride); }, [keyOverride]);
  useEffect(() => { persist('noFsq', noFsq); }, [noFsq]);
  useEffect(() => { persist('coverInstrumental', instrumental); }, [instrumental]);
  useEffect(() => { persist('sourceLatentUrl', sourceLatentUrl); }, [sourceLatentUrl]);
  useEffect(() => { persist('coverVocalLanguage', vocalLanguage); }, [vocalLanguage]);
  useEffect(() => { persist('coverTimbreOverride', timbreOverridePath); }, [timbreOverridePath]);
  useEffect(() => { persist('sepLevel', sepLevel); }, [sepLevel]);

  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 4000); };

  // ── "Send to Cover Studio" — load a library track as the cover source (#61) ──
  useEffect(() => {
    if (!coverSource || coverSource.timestamp === _lastConsumedCoverTs) return;
    _lastConsumedCoverTs = coverSource.timestamp;
    const s = coverSource.song;
    const gpData: any = s.generationParams || s.generation_params || {};
    const audioUrl = s.audioUrl || s.audio_url || '';

    // Source audio — reuse the track's server URL directly (loadSourceAudio
    // resolves /audio/ paths server-side), so no re-upload is needed.
    setSourceAudioUrl(audioUrl);
    setSourceFileName(s.title || 'Library track');
    setMetadata({
      artist: s.artistName || '', title: s.title || '', album: '',
      duration: typeof s.duration === 'number' ? s.duration : null,
    });

    // Text + style descriptions
    setSongTitle(s.title || '');
    if (s.artistName) setSongArtist(s.artistName);
    setLyrics(s.lyrics || gpData.lyrics || '');
    setArtistCaption(s.style || s.caption || gpData.caption || '');

    // Instrumental / vocal intent
    const lyr = (s.lyrics || gpData.lyrics || '').trim().toLowerCase();
    setInstrumental(gpData.instrumental === true || lyr === '' || lyr === '[instrumental]');

    // Reset transforms/overrides from any previous source
    setBpmCorrection(1); setKeyOverride(null); setBpmOverride(null);
    setTempoScale(1.0); setPitchShift(0);

    // BPM/key — (A) use the track's stored metadata, else (B) analyze the source.
    const storedBpm = s.bpm ?? gpData.bpm;
    const storedKey = s.key_scale || gpData.keyScale;
    if (storedBpm && storedKey) {
      setAnalysis({ bpm: Number(storedBpm), key: String(storedKey), scale: String(storedKey).split(' ')[1] });
    } else if (audioUrl) {
      setIsAnalyzing(true);
      fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audioUrl }),
      })
        .then(r => (r.ok ? r.json() : null))
        .then(d => { if (d) setAnalysis({ bpm: d.bpm || 120, key: `${d.key || 'C'} ${d.scale || 'major'}`, scale: d.scale }); })
        .catch(() => { /* leave defaults; user can override BPM/key manually */ })
        .finally(() => setIsAnalyzing(false));
    }
    showToast(t('cover.loadedFromLibrary', 'Loaded source from library'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coverSource?.timestamp]);

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
      .catch(() => showToast(t('cover.failedToLoadArtists')))
      .finally(() => setIsLoadingArtists(false));
  }, []);

  // ── File upload + analysis pipeline ──
  const handleFileSelected = async (file: File) => {
    if (!token) { showToast(t('cover.signInFirst')); return; }
    setSourceFileName(file.name);
    setBpmCorrection(1);
    setKeyOverride(null);

    // Check track cache
    const cached = getTrackCache()[file.name];
    if (cached) {
      showToast(t('cover.loadedFromCache'));
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
      let analysed = false;
      const anRes = await fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audioUrl }),
      });
      if (anRes.ok) {
        const d = await anRes.json();
        bpm = d.bpm || 120; key = `${d.key || 'C'} ${d.scale || 'major'}`; scale = d.scale;
        analysed = true;
        setAnalysis({ bpm, key, scale });
      }
      // 4. Cache. Only a real analysis goes in: with Essentia unavailable the
      // 120 / C major fallback was cached and came back labelled "detected" on
      // every later load of the same file (#131).
      saveTrackCacheEntry(file.name, {
        artist: extractedArtist, title: extractedTitle, album: extractedAlbum, duration: extractedDuration,
        ...(analysed ? { bpm, key, scale } : {}),
      });
    } catch (err: any) {
      showToast(`Error: ${err.message}`);
    } finally { setIsUploading(false); setIsAnalyzing(false); }
  };

  // ── Lyrics search ──
  const handleSearchLyrics = async () => {
    if (!songArtist.trim() || !songTitle.trim()) { showToast(t('cover.enterArtistTitle')); return; }
    setIsSearchingLyrics(true);
    try {
      const result = await lireekApi.searchSongLyrics(songArtist.trim(), songTitle.trim());
      setLyrics(result.lyrics);
      if (result.title) setSongTitle(result.title);
      showToast(t('cover.lyricsFound'));
      if (sourceFileName) saveTrackCacheEntry(sourceFileName, { lyrics: result.lyrics, artist: songArtist.trim(), title: result.title || songTitle.trim() });
    } catch (err: any) { showToast(err.message || t('cover.noLyricsFound')); }
    finally { setIsSearchingLyrics(false); }
  };

  // ── Apply preset to global UI (adapter bar + mastering reference) ──
  const applyPresetToGlobal = (preset: AlbumPreset | null) => {
    // Only sync the adapter path — never override user's manual scale/group-scale settings
    if (preset?.adapter_path) {
      gp.setAdapter(preset.adapter_path);
    }
    if (preset?.reference_track_path) {
      gp.setMasteringReference(preset.reference_track_path);
    }
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
      // Find caption — waterfall: generations → profiles → LLM
      let caption = '';
      // Step 1: Try generation captions
      for (const ls of lyrics_sets) {
        try {
          const { generations } = await lireekApi.listGenerations(undefined, ls.id);
          const wc = generations.find(g => g.caption?.trim());
          if (wc?.caption) { caption = wc.caption; break; }
        } catch {}
      }
      // Step 2: Try profile style_caption
      if (!caption) {
        for (const ls of lyrics_sets) {
          try {
            const profiles = await lireekApi.listProfiles(ls.id);
            for (const p of profiles.profiles) {
              const full = await lireekApi.getProfile(p.id);
              if (full.profile_data?.style_caption) {
                caption = full.profile_data.style_caption;
                break;
              }
            }
            if (caption) break;
          } catch {}
        }
      }
      setArtistCaption(caption);
      // Step 3: On-demand LLM generation (async, non-blocking)
      if (!caption) {
        const sel = loadSelections();
        const { provider, model } = sel.coverCaption;
        if (provider) {
          setIsGeneratingCaption(true);
          lireekApi.generateCaption(artist.id, { provider, model: model || undefined })
            .then(res => { if (res.caption) setArtistCaption(res.caption); })
            .catch(err => console.warn('[CoverStudio] Caption generation failed:', err))
            .finally(() => setIsGeneratingCaption(false));
        }
      }
      // Pick adapter preset — use first album with adapter (don't override cover settings)
      const withAdapter = results.find(p => p.preset?.adapter_path);
      if (withAdapter?.preset) {
        setSelectedPreset(withAdapter.preset);
        applyPresetToGlobal(withAdapter.preset);
      } else { setSelectedPreset(results[0]?.preset || null); }
    } catch { setSelectedPreset(null); setArtistPresets([]); setArtistCaption(''); }
  };

  const handleSelectArtist = async (artist: Artist) => {
    setSelectedArtistId(artist.id);
    await loadArtistPresets(artist);
  };

  // ── Cover caption LLM handlers ──
  const handleCoverProviderChange = (provider: string) => {
    setCoverCaptionProvider(provider);
    const sel = loadSelections();
    sel.coverCaption = { ...sel.coverCaption, provider };
    saveSelections(sel);
  };
  const handleCoverModelChange = (model: string) => {
    setCoverCaptionModel(model);
    const sel = loadSelections();
    sel.coverCaption = { ...sel.coverCaption, model };
    saveSelections(sel);
  };
  const handleRegenerateCaption = async () => {
    if (!selectedArtistId) return;
    const sel = loadSelections();
    const { provider, model } = sel.coverCaption;
    if (!provider) { showToast('Select a Caption LLM provider first'); return; }
    setIsGeneratingCaption(true);
    try {
      const res = await lireekApi.generateCaption(selectedArtistId, { provider, model: model || undefined, force: true });
      if (res.caption) setArtistCaption(res.caption);
    } catch (err: any) { showToast(`Caption generation failed: ${err.message}`); }
    finally { setIsGeneratingCaption(false); }
  };

  // ── Generation ──
  const handleGenerate = () => {
    if (!token || !sourceAudioUrl) { showToast(t('cover.missingSrcOrLyrics')); return; }
    if (!instrumental && !lyrics.trim()) { showToast('Enter lyrics or enable Instrumental mode'); return; }

    // Show a queue item immediately ("Queued…" if a cover is already running),
    // then enqueue the work so the user can stack more without waiting (#62).
    const coverTitle = songArtist
      ? `${songTitle || 'Cover'} (${songArtist} Cover)`
      : (songTitle || 'Cover');
    const qId = addManualQueueItem({
      title: coverTitle,
      artistName: artists.find(a => a.id === selectedArtistId)?.name || '',
      caption: artistCaption || '',
    });
    updateManualQueueItem(qId, { stage: _coverRunning ? 'Queued…' : 'Preparing…' });
    setIsGenerating(true);
    showToast(t('cover.genStarted'));

    // The closure snapshots the current settings, so each queued cover keeps
    // the params it was submitted with.
    enqueueCoverJob(async () => {
      try {
      // Step 0: If advanced mode with stems, auto-recombine before generation
      let effectiveSourceUrl = sourceAudioUrl;
      if (advancedMode && sepStems && sepStems.length > 0 && sepJobId) {
        setGenStage(t('cover.recombiningStems'));
        setGenProgress(2);
        try {
          // Build effective controls — log them for diagnostics
          const effectiveControls = stemControls.map(c => ({
            index: c.index,
            volume: c.muted ? 0 : c.volume,
            muted: c.muted,
          }));
          console.log('[CoverStudio] Auto-recombine controls:', JSON.stringify(effectiveControls));
          console.log('[CoverStudio] Stem names:', sepStems.map(s => `[${s.index}] ${s.name}`).join(', '));
          const blob = await recombineStems(sepJobId, effectiveControls);
          // Upload recombined WAV to get a server-side URL
          const fd = new FormData();
          fd.append('audio', blob, 'recombined-stems.wav');
          const upRes = await fetch('/api/upload/audio', { method: 'POST', body: fd });
          if (upRes.ok) {
            const { audio_url } = await upRes.json();
            effectiveSourceUrl = audio_url;
            console.log('[CoverStudio] Using recombined stems:', audio_url);
          }
        } catch (err: any) {
          console.warn('[CoverStudio] Stem recombine failed, using original:', err.message);
          showToast(`Stem recombine failed: ${err.message}. Using original audio.`);
        }
      }
      const selectedArtist = artists.find(a => a.id === selectedArtistId);
      const sourceBpm = bpmOverride != null ? bpmOverride : ((analysis?.bpm || 120) * bpmCorrection);
      const sourceKey = keyOverride || analysis?.key || 'C major';
      const targetBpm = Math.round(sourceBpm * tempoScale);
      const targetKey = pitchShift !== 0 ? transposeKey(sourceKey, pitchShift) : sourceKey;

      // Start from global engine params
      const engineParams = gp.getGlobalParams();

      // Override with cover-specific params
      const params: Record<string, any> = {
        ...engineParams,
        customMode: true,
        lyrics: instrumental ? '[Instrumental]' : lyrics,
        style: artistCaption || engineParams.style || '',
        title: songArtist
          ? `${songTitle || 'Cover'} (${songArtist} Cover)`
          : (songTitle || 'Cover'),
        taskType: noFsq ? 'cover-nofsq' : 'cover',
        sourceAudioUrl: effectiveSourceUrl,
        audioCoverStrength,
        coverNoiseStrength,
        ...(coverNoiseMethod ? { coverNoiseMethod } : {}),
        bpm: targetBpm,
        keyScale: targetKey,
        duration: 0,
        instrumental: instrumental,
        vocalLanguage,
        source: 'cover-studio',
        artistName: selectedArtist?.name || songArtist || '',
        sourceArtist: songArtist || '',
        ...(sourceLatentUrl ? { sourceLatentUrl } : {}),
      };
      if (tempoScale !== 1.0) params.tempoScale = tempoScale;
      if (pitchShift !== 0) params.pitchShift = pitchShift;

      // Apply album preset adapter (overrides global adapter)
      if (selectedPreset?.adapter_path) {
        params.loraPath = selectedPreset.adapter_path;
        // IMPORTANT: always use the user's manual scale from the adapters dropdown,
        // NOT the preset's stored scale — user's manual overrides take priority.
        // params.loraScale and params.adapterGroupScales already come from
        // engineParams (spread on line 257) and must not be overridden here.
        // Override trigger word to match the preset's adapter, not the global one
        if (settings.triggerUseFilename) {
          const presetFilename = selectedPreset.adapter_path.split(/[\\/]/).pop() || '';
          const presetTrigger = presetFilename.replace(/\.safetensors$/i, '');
          if (presetTrigger) {
            params.triggerWord = presetTrigger;
            params.triggerPlacement = settings.triggerPlacement || 'prepend';
          }
        }
      }
      // Reference track + matchering from album preset
      if (selectedPreset?.reference_track_path) {
        params.referenceAudioUrl = selectedPreset.reference_track_path;
        params.masteringEnabled = true;
        params.masteringReference = selectedPreset.reference_track_path;
        // Default: use preset reference as timbre (can be overridden below)
        params.timbreReference = true;
      }
      // Timbre conditioning — user override takes priority over preset reference
      if (timbreOverridePath) {
        params.timbreReference = timbreOverridePath;
      } else if (typeof engineParams.timbreReference === 'string' && engineParams.timbreReference) {
        params.timbreReference = engineParams.timbreReference;
      }

      const res = await generateApi.submit(params as any, token);
      updateManualQueueItem(qId, { jobId: res.jobId });
      await pollJobAsync(res.jobId, qId);
      } catch (err: any) {
        failManualQueueItem(qId, err.message || 'Generation failed');
      } finally {
        // Reset the inline progress only once the whole queue has drained.
        if (_coverQueue.length === 0) {
          setIsGenerating(false);
          setActiveJobId(null);
          setGenProgress(0);
          setGenStage('');
        }
      }
    });
  };

  // Poll a single cover job to completion. Resolves on any terminal state so
  // the serial queue can advance to the next cover. Updates the inline progress
  // (safe — jobs run one at a time) and the per-job queue item.
  const pollJobAsync = (jobId: string, qId: string): Promise<void> => new Promise<void>((resolve) => {
    setActiveJobId(jobId); setQueueItemId(qId);
    setGenProgress(0); setGenStage('Queued...');
    // Clock ignores server-queue wait — only real generation time counts.
    const timer = createGenerationTimer();
    const iv = setInterval(async () => {
      try {
        const s = await generateApi.status(jobId);
        const tk = timer.tick(s.status);
        // Server sends 0-100; normalise to 0-100 for display
        const rawProg = s.progress;
        const pct = rawProg != null
          ? Math.min(100, Math.max(0, Math.round(rawProg > 1 ? rawProg : rawProg * 100)))
          : undefined;
        if (pct != null) setGenProgress(pct);
        if (s.stage) setGenStage(s.stage);

        // Update shared queue item
        updateManualQueueItem(qId, {
          progress: pct,
          stage: s.stage || 'Generating...',
          elapsed: tk.elapsed,
        });

        if (tk.timedOut) {
          clearInterval(iv);
          showToast(`Generation timed out after ${getGenerationTimeoutMinutes()} minutes`);
          failManualQueueItem(qId, 'Generation timed out');
          resolve();
          return;
        }

        if (s.status === 'succeeded') {
          clearInterval(iv); setGenProgress(100); setGenStage('Complete!');
          setRefreshTrigger(p => p + 1); showToast(t('cover.coverGenerated'));

          // Complete queue item with audio data
          completeManualQueueItem(qId, {
            audioUrl: s.result?.audioUrls?.[0] || '',
            songId: s.result?.songIds?.[0],
            masteredAudioUrl: s.result?.masteredAudioUrl,
            noAdapterAudioUrl: s.result?.noAdapterAudioUrl,
            audioDuration: s.result?.duration,
          });
          resolve();
        } else if (s.status === 'failed' || s.status === 'cancelled') {
          clearInterval(iv);
          showToast(`Failed: ${s.error || 'Unknown error'}`);
          failManualQueueItem(qId, s.error || (s.status === 'cancelled' ? 'Cancelled' : 'Unknown error'));
          resolve();
        }
      } catch { /* transient poll error — keep polling */ }
    }, 2000);
    // Absolute backstop so a wedged job can't block the queue forever. Generous
    // so it never pre-empts the generation-start timer above.
    setTimeout(() => { clearInterval(iv); resolve(); }, (getGenerationTimeoutMinutes() + 30) * 60_000);
  });

  // Cancel the currently-running cover. The poll sees the cancelled status,
  // resolves, and the queue advances to the next item.
  const handleCancel = async () => {
    if (activeJobId) { try { await generateApi.cancel(activeJobId); } catch {} }
    if (queueItemId) failManualQueueItem(queueItemId, 'Cancelled by user');
  };

  const handleClearSource = () => {
    setSourceFileName(''); setSourceAudioUrl('');
    setMetadata(null); setAnalysis(null);
    setSongArtist(''); setSongTitle(''); setLyrics('');
    setBpmCorrection(1); setKeyOverride(null);
    // Clear stems too
    setSepStems(null); setStemControls([]); setSepJobId(null); setShowMixer(false);
  };

  const handleClearArtist = () => {
    setSelectedArtistId(null);
    setSelectedPreset(null);
    setArtistPresets([]);
    setArtistCaption('');
  };

  // Always allow queuing another cover — covers stack and run one at a time (#62).
  const canGenerate = !!sourceAudioUrl && (!!lyrics.trim() || instrumental);

  // ── SuperSep handlers ──
  const handleSeparate = useCallback(async () => {
    if (!sourceAudioUrl) { showToast(t('cover.uploadAudioFirst')); return; }
    setIsSeparating(true);
    setSepProgress(0);
    setSepMessage(t('cover.startingSeparation'));
    setSepStems(null);


    try {
      // Start separation — pass server URL directly (no need to download/re-upload)
      const jobId = await startSeparation(sourceAudioUrl, sepLevel);
      setSepJobId(jobId);

      // Wait for completion with progress updates
      const result = await waitForCompletion(jobId, (progress, message) => {
        setSepProgress(progress);
        setSepMessage(message);
      });

      // Map SuperSep stems to shared MixerStemInfo (add audioUrl)
      const mixerStems: MixerStemInfo[] = result.stems.map(s => ({
        name: s.name,
        category: s.category,
        audioUrl: getStemAudioUrl(jobId, s.index),
        index: s.index,
        stage: s.stage,
      }));
      setSepStems(mixerStems);
      // Initialize stem controls (all at 100%, unmuted)
      setStemControls(mixerStems.map(s => ({ index: s.index, volume: 1.0, muted: false })));
      setShowMixer(true);
      showToast(t('cover.separatedIntoStems', { count: result.stems.length }));
    } catch (err: any) {
      showToast(`Separation failed: ${err.message}`);
    } finally {
      setIsSeparating(false);
    }
  }, [sourceAudioUrl, sepLevel]);



  // ── Render ──
  return (
    <BackendCapabilityGate feature="cover">
    <div className="flex flex-col w-full h-full bg-zinc-50 dark:bg-suno overflow-hidden">
      {/* Toast */}
      {toast && (
        <div className="absolute top-16 right-6 z-50 px-4 py-2 rounded-xl bg-white dark:bg-zinc-900 text-white text-sm shadow-xl border border-zinc-300 dark:border-white/10 animate-in fade-in slide-in-from-top-2">
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
          bpmCorrection={bpmCorrection} onBpmCorrectionChange={setBpmCorrection}
          bpmOverride={bpmOverride} onBpmOverrideChange={setBpmOverride}
          keyOverride={keyOverride} onKeyOverrideChange={setKeyOverride}
          vocalLanguage={vocalLanguage} onVocalLanguageChange={setVocalLanguage}
          advancedMode={advancedMode} onAdvancedModeChange={setAdvancedMode}
          sepLevel={sepLevel} onSepLevelChange={(v) => setSepLevel(v as SeparationLevel)}
          isSeparating={isSeparating} sepProgress={sepProgress} sepMessage={sepMessage}
          sourceAudioUrl={sourceAudioUrl} onSeparate={handleSeparate}
          hasStems={!!(sepStems && sepStems.length > 0 && sepJobId)}
          onConfigureStems={() => setShowMixer(true)}
          sourceLatentUrl={sourceLatentUrl}
          onLatentLoaded={(url: string, meta: LatentMetadata) => {
            setSourceLatentUrl(url);
            // Auto-populate fields from HSLAT metadata
            if (meta.lyrics) setLyrics(meta.lyrics);
            if (meta.caption) setArtistCaption(meta.caption);
            if (meta.bpm && meta.bpm > 0) {
              setAnalysis(prev => prev ? { ...prev, bpm: meta.bpm! } : { bpm: meta.bpm!, key: meta.key || '', scale: undefined });
            }
            if (meta.key) {
              setKeyOverride(meta.key);
            }
          }}
          onLatentClear={() => setSourceLatentUrl('')}
          timbreOverridePath={timbreOverridePath}
          onTimbreOverridePathChange={setTimbreOverridePath}
          token={token}
          coverCaptionProvider={coverCaptionProvider}
          coverCaptionModel={coverCaptionModel}
          onCoverCaptionProviderChange={handleCoverProviderChange}
          onCoverCaptionModelChange={handleCoverModelChange}
        />

        {/* Center: Lyrics */}
        <div className="flex-1 flex flex-col overflow-hidden border-r border-zinc-200 dark:border-white/5">
          <div className="flex-shrink-0 px-4 py-3 border-b border-zinc-200 dark:border-white/5">
            <div className="flex items-center justify-between gap-2">
              {/* Artist + Title inputs */}
              <div className="flex-1 flex gap-2">
                <input value={songArtist} onChange={e => setSongArtist(e.target.value)}
                  placeholder={t('cover.artistPlaceholder')} className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 focus:outline-none focus:border-cyan-500" />
                <input value={songTitle} onChange={e => setSongTitle(e.target.value)}
                  placeholder={t('cover.songTitlePlaceholder')} className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 focus:outline-none focus:border-cyan-500" />
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
              placeholder={t('cover.lyricsPlaceholder')}
              className="w-full h-full resize-none bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-xl px-4 py-3 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-cyan-500 transition-colors font-mono leading-relaxed" />
          </div>
          
          {showMixer && sepStems && sepJobId && (
            <StemMixer jobId={sepJobId} stems={sepStems}
              controls={stemControls} onControlsChange={setStemControls}
              onClose={() => setShowMixer(false)} />
          )}
        </div>

        {/* Right: Artist + Settings */}
        <ArtistSettingsPanel
          artists={artists} isLoadingArtists={isLoadingArtists}
          selectedArtistId={selectedArtistId} onSelectArtist={handleSelectArtist}
          onClearArtist={handleClearArtist}
          artistPresets={artistPresets} selectedPreset={selectedPreset}
          onSelectPreset={(p) => { setSelectedPreset(p); applyPresetToGlobal(p); }}
          audioCoverStrength={audioCoverStrength} onAudioCoverStrength={setAudioCoverStrength}
          coverNoiseStrength={coverNoiseStrength} onCoverNoiseStrength={setCoverNoiseStrength}
          coverNoiseMethod={coverNoiseMethod} onCoverNoiseMethodChange={setCoverNoiseMethod}
          noFsq={noFsq} onNoFsqChange={setNoFsq}
          instrumental={instrumental} onInstrumentalChange={setInstrumental}
          tempoScale={tempoScale} onTempoScale={setTempoScale}
          pitchShift={pitchShift} onPitchShift={setPitchShift}
          analysis={analysis}
          bpmCorrection={bpmCorrection}
          keyOverride={keyOverride}
          artistCaption={artistCaption} onArtistCaptionChange={setArtistCaption}
          canGenerate={canGenerate}
          isGenerating={isGenerating} genProgress={genProgress} genStage={genStage}
          onGenerate={handleGenerate} onCancel={handleCancel}
          isGeneratingCaption={isGeneratingCaption}
          onRegenerateCaption={handleRegenerateCaption}
        />

        {/* Resize handle */}
        <div
          className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
          onMouseDown={handleSidebarResize}
        >
          <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
        </div>
        {/* Right: Recent Covers + Queue */}
        <div className="h-full flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-hidden" style={{ width: sidebarWidth }}>
          <ActivitySidebar
            showToast={showToast}
            source="cover-studio"
            refreshKey={refreshTrigger + completionCounter}
            queueCountColor="bg-cyan-500/20 text-cyan-300"
            compact={sidebarWidth < 380}
          />
        </div>
      </div>
    </div>
    </BackendCapabilityGate>
  );
};
