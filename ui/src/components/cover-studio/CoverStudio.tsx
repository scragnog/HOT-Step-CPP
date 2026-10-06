// CoverStudio.tsx — Main Cover Studio orchestrator
// Composes: SourcePanel, ArtistSettingsPanel
import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  recombineStems,
  type SeparationLevel,
} from '../../services/supersepApi';
import { SourcePanel } from './SourcePanel';
import { ArtistSettingsPanel } from './ArtistSettingsPanel';
import { BackendCapabilityGate } from '../shared/BackendCapabilityGate';
import { StemMixer } from '../shared/StemMixer';
import { useCoverStemsStore } from '../../stores/coverStemsStore';
import {
  addManualQueueItem, updateManualQueueItem,
  completeManualQueueItem, failManualQueueItem,
} from '../../stores/audioGenQueueStore';
import {
  persist, restore, getTrackCache, saveTrackCacheEntry, transposeKey,
  type AudioMetadata, type AudioAnalysis,
} from './coverStudioUtils';
import type { LatentMetadata } from '../shared/LatentImport';
import { loadSelections, saveSelections } from '../lyric-studio/ProviderSelector';
import { useBackendStore } from '../../stores/backendStore';
import { fetchYue2CaptionSource, resolveYue2Caption, yue2PickAtEnqueue, type Yue2SourceTrack } from '../../utils/yue2CaptionSource';
import { yue2CoverApi, type Yue2CoverDatasetMetadata, type Yue2CoverJob, type Yue2CoverReadiness,
  type Yue2ScoreSection, type Yue2SectionReview } from '../../services/yue2CoverApi';
import { Yue2CoverPanel } from './Yue2CoverPanel';
import { Yue2CoverScore } from './Yue2CoverScore';
import { SectionMatchReview } from './SectionMatchReview';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import type { Yue2SectionMatch } from '../../services/yue2CoverApi';

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

function coverScoreKeyLabel(abc: string): string {
  const key = abc.match(/^K:\s*([A-Ga-g][#b]?)(m)?(?:\s+(minor|major|maj))?(?=\s|$)/m);
  if (!key) return '';
  return `${key[1]} ${key[2] || key[3]?.toLowerCase() === 'minor' ? 'minor' : 'major'}`;
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
  const yue2Mode = useBackendStore(s => s.activeBackendId === 'yue2');
  const yue2Models = useBackendStore(s => s.models.yue2);

  // ── Source audio state ──
  const [sourceFileName, setSourceFileName] = useState(() => restore<string>('sourceFileName', ''));
  const [sourceAudioUrl, setSourceAudioUrl] = useState(() => restore<string>('sourceAudioUrl', ''));
  const [sourceSongId, setSourceSongId] = useState(() => restore<string>('sourceSongId', ''));
  const [metadata, setMetadata] = useState<AudioMetadata | null>(() => restore('metadata', null));
  const [analysis, setAnalysis] = useState<AudioAnalysis | null>(() => restore('analysis', null));
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // ── Song details ──
  const [songArtist, setSongArtist] = useState(() => restore<string>('songArtist', ''));
  const [songTitle, setSongTitle] = useState(() => restore<string>('songTitle', ''));
  const [lyrics, setLyrics] = useState(() => restore<string>('lyrics', ''));
  const [lyricsSource, setLyricsSource] = useState<'dataset-sidecar' | null>(() => restore('lyricsSource', null));
  const [datasetAnalysis, setDatasetAnalysis] = useState(() => restore('datasetAnalysis', false));
  const [sheetScoreSource, setSheetScoreSource] = useState<'dataset' | null>(null);
  const sourceLookupRef = useRef(0);
  const lyricEditRef = useRef(0);
  const instrumentalEditRef = useRef(0);
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
  const [pairMode, setPairMode] = useState<'base' | 'pair'>('base');
  const [yue2Ar, setYue2Ar] = useState('');
  const [yue2Nar, setYue2Nar] = useState('');
  const [coverVoices, setCoverVoices] = useState<'vocal' | 'both'>('both');
  const [keepChords, setKeepChords] = useState(false);
  const [coverTempoMode, setCoverTempoMode] = useState<'free' | 'source' | 'set'>('free');
  const [coverBpm, setCoverBpm] = useState(120);
  const [coverKeyShift, setCoverKeyShift] = useState(0);
  const [coverCfgScale, setCoverCfgScale] = useState(1);
  const [captionMode, setCaptionMode] = useState('custom');
  const [captionTracks, setCaptionTracks] = useState<Yue2SourceTrack[]>([]);
  const [readiness, setReadiness] = useState<Yue2CoverReadiness | null>(null);
  const [sheetJob, setSheetJob] = useState<Yue2CoverJob | null>(null);
  const [sheetJobId, setSheetJobId] = useState('');
  const sheetJobRef = useRef('');
  const sheetRequestRef = useRef(0);
  const [sheetPreparing, setSheetPreparing] = useState(false);
  const [sheetError, setSheetError] = useState('');
  const [sheetAbc, setSheetAbc] = useState('');
  const [scoreSections, setScoreSections] = useState<Yue2ScoreSection[]>([]);
  const [sectionLint, setSectionLint] = useState<Yue2SectionReview['lint'] | null>(null);
  // Match sections to score: the proposal and the lyrics it was made from.
  const [sectionMatch, setSectionMatch] = useState<{ match: Yue2SectionMatch; before: string; key: string } | null>(null);
  const [sectionMatching, setSectionMatching] = useState(false);
  const [sectionSaving, setSectionSaving] = useState(false);
  const [sectionSaveConfirm, setSectionSaveConfirm] = useState(false);
  const [scoreDetailsSaving, setScoreDetailsSaving] = useState(false);
  const [scoreDetailsConfirm, setScoreDetailsConfirm] = useState(false);
  const [sheetAudioUrl, setSheetAudioUrl] = useState('');
  const [approvedSheet, setApprovedSheet] = useState<{ abc: string; sourceId: string; sourceLabel: string; audioUrl: string; key: string } | null>(null);

  // ── Generation ──
  const [isGenerating, setIsGenerating] = useState(false);
  const [genProgress, setGenProgress] = useState(0);
  const [genStage, setGenStage] = useState('');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [toast, setToast] = useState('');
  const [queueItemId, setQueueItemId] = useState<string | null>(null);


  // ── Advanced Mode (SuperSep) ──
  // Split result, controls, job id and in-flight progress live in
  // coverStemsStore — component-local state here was destroyed by leaving
  // the tab, since that unmounts CoverStudio (#135).
  const [advancedMode, setAdvancedMode] = useState(false);
  const [sepLevel, setSepLevel] = useState<SeparationLevel>(() => restore('sepLevel', 1) as SeparationLevel);
  const {
    sepJobId, sepStems, stemControls, showMixer, isSeparating, sepProgress, sepMessage,
    setShowMixer, setStemControls, startSeparation: startStemSeparation, clearStems,
  } = useCoverStemsStore();
  const sourceKey = JSON.stringify([sourceAudioUrl, sourceSongId, advancedMode && sepJobId,
    advancedMode && sepStems?.length, advancedMode && stemControls]);
  const sourceKeyRef = useRef(sourceKey);
  sourceKeyRef.current = sourceKey;
  useEffect(() => {
    sheetRequestRef.current += 1;
    setApprovedSheet(null); setSheetAbc(''); setScoreSections([]); setSectionLint(null); setSectionMatch(null);
    setSheetAudioUrl(''); setSheetError(''); setSheetJob(null); setSheetPreparing(false);
    if (sheetJobRef.current && token) void yue2CoverApi.cancel(sheetJobRef.current, token).catch(() => {});
    sheetJobRef.current = ''; setSheetJobId('');
  }, [sourceKey, token]);

  useEffect(() => {
    if (!yue2Mode || !token) return;
    void useBackendStore.getState().fetchModels('yue2');
    void yue2CoverApi.readiness(token).then(setReadiness).catch(err => setSheetError(err.message));
  }, [yue2Mode, token]);

  useEffect(() => {
    if (!yue2Mode || pairMode !== 'pair') { setCaptionTracks([]); return; }
    const adapter = yue2Ar || yue2Nar;
    if (!adapter) { setCaptionTracks([]); return; }
    let live = true;
    void fetchYue2CaptionSource({ adapter }).then(result => { if (live) setCaptionTracks(result.tracks); });
    return () => { live = false; };
  }, [yue2Mode, pairMode, yue2Ar, yue2Nar]);

  useEffect(() => {
    if (!sheetJobId || !token) return;
    let live = true;
    const poll = async () => {
      try {
        const result = await yue2CoverApi.status(sheetJobId, token);
        if (!live || sheetJobRef.current !== sheetJobId) return;
        setSheetJob(result.job ?? null);
        if (result.job?.status === 'done') {
          setSheetAbc(result.abc || ''); setScoreSections(result.sections || []);
          setSheetJobId(''); sheetJobRef.current = '';
        } else if (result.job?.status === 'failed' || result.job?.status === 'cancelled') {
          setSheetError(result.job.error || `Transcription ${result.job.status}`);
          setSheetJobId(''); sheetJobRef.current = '';
        }
      } catch (err: any) { if (live) { setSheetError(err.message); setSheetJobId(''); sheetJobRef.current = ''; } }
    };
    void poll();
    const timer = setInterval(() => { void poll(); }, 1000);
    return () => { live = false; clearInterval(timer); };
  }, [sheetJobId, token]);

  useEffect(() => {
    if (!yue2Mode || !token || !sheetAbc.trim()) { setSectionLint(null); return; }
    let live = true;
    const timer = setTimeout(() => {
      void yue2CoverApi.reviewSections(sheetAbc, lyrics, token).then(review => {
        if (!live) return;
        setScoreSections(review.sections);
        setSectionLint(review.lint);
      }).catch(() => { if (live) setSectionLint(null); });
    }, 250);
    return () => { live = false; clearTimeout(timer); };
  }, [yue2Mode, token, sheetAbc, lyrics]);


  // ── Persist ──
  useEffect(() => { persist('sourceFileName', sourceFileName); }, [sourceFileName]);
  useEffect(() => { persist('sourceAudioUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { persist('sourceSongId', sourceSongId); }, [sourceSongId]);
  useEffect(() => { persist('metadata', metadata); }, [metadata]);
  useEffect(() => { persist('analysis', analysis); }, [analysis]);
  useEffect(() => { persist('songArtist', songArtist); }, [songArtist]);
  useEffect(() => { persist('songTitle', songTitle); }, [songTitle]);
  useEffect(() => { persist('lyrics', lyrics); }, [lyrics]);
  useEffect(() => { persist('lyricsSource', lyricsSource); }, [lyricsSource]);
  useEffect(() => { persist('datasetAnalysis', datasetAnalysis); }, [datasetAnalysis]);
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

  const applyDatasetMetadata = (data: Yue2CoverDatasetMetadata, lyricEdit: number, instrumentalEdit: number): boolean => {
    if (!data.matched) return false;
    if (data.abc) { setSheetAbc(data.abc); setScoreSections(data.sections || []); setSheetScoreSource('dataset'); }
    if (!data.metadataAvailable || data.bpm == null || !data.key) return false;
    if (lyricEditRef.current === lyricEdit) {
      setLyrics(data.lyrics || '');
      setLyricsSource(instrumentalEditRef.current === instrumentalEdit ? 'dataset-sidecar' : null);
    }
    if (instrumentalEditRef.current === instrumentalEdit) setInstrumental(data.isInstrumental === true);
    setAnalysis({ bpm: data.bpm, key: data.key, scale: data.key.split(' ')[1] });
    setDatasetAnalysis(true);
    return true;
  };

  const lookupDatasetMetadata = async (input: { songId?: string; sourceAudioUrl?: string }) => {
    if (!token) return null;
    try { return await yue2CoverApi.sourceMetadata(input, token); }
    catch { return null; }
  };

  // ── "Send to Cover Studio" — load a library track as the cover source (#61) ──
  useEffect(() => {
    if (!coverSource || coverSource.timestamp === _lastConsumedCoverTs) return;
    _lastConsumedCoverTs = coverSource.timestamp;
    const s = coverSource.song;
    const gpData: any = s.generationParams || s.generation_params || {};
    const audioUrl = s.audioUrl || s.audio_url || '';
    setSourceSongId(String(s.id));
    const lookup = ++sourceLookupRef.current;
    const lyricEdit = lyricEditRef.current, instrumentalEdit = instrumentalEditRef.current;
    setLyricsSource(null); setDatasetAnalysis(false); setSheetScoreSource(null);
    clearStems();

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
    void (async () => {
      const dataset = await lookupDatasetMetadata({ songId: String(s.id) });
      if (sourceLookupRef.current !== lookup || applyDatasetMetadata(dataset || { matched: false }, lyricEdit, instrumentalEdit)) return;
      if (storedBpm && storedKey) {
        setAnalysis({ bpm: Number(storedBpm), key: String(storedKey), scale: String(storedKey).split(' ')[1] });
      } else if (audioUrl) {
        setIsAnalyzing(true);
        try {
          const r = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ audioUrl }) });
          const d = r.ok ? await r.json() : null;
          if (d && sourceLookupRef.current === lookup) setAnalysis({ bpm: d.bpm || 120, key: `${d.key || 'C'} ${d.scale || 'major'}`, scale: d.scale });
        } catch { /* leave defaults; user can override */ }
        finally { if (sourceLookupRef.current === lookup) setIsAnalyzing(false); }
      }
    })();
    showToast(t('cover.loadedFromLibrary', 'Loaded source from library'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coverSource?.timestamp]);

  // ── Load artists on mount ──
  useEffect(() => {
    if (yue2Mode) return;
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
  }, [yue2Mode]);

  // ── File upload + analysis pipeline ──
  const handleFileSelected = async (file: File) => {
    if (!token) { showToast(t('cover.signInFirst')); return; }
    const lookup = ++sourceLookupRef.current;
    const lyricEdit = lyricEditRef.current, instrumentalEdit = instrumentalEditRef.current;
    setLyricsSource(null); setDatasetAnalysis(false); setSheetScoreSource(null);
    setSourceAudioUrl(''); setSourceSongId(''); clearStems();
    setSourceFileName(file.name);
    setBpmCorrection(1);
    setBpmOverride(null);
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
        if (r.ok) {
          const d = await r.json();
          if (sourceLookupRef.current !== lookup) return;
          setSourceAudioUrl(d.audio_url || '');
          const dataset = await lookupDatasetMetadata({ sourceAudioUrl: d.audio_url });
          if (sourceLookupRef.current === lookup && dataset) applyDatasetMetadata(dataset, lyricEdit, instrumentalEdit);
        }
      } catch {} finally { if (sourceLookupRef.current === lookup) setIsUploading(false); }
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
        if (sourceLookupRef.current !== lookup) return;
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
      if (sourceLookupRef.current !== lookup) return;
      const audioUrl = upData.audio_url || '';
      setSourceAudioUrl(audioUrl);

      const dataset = await lookupDatasetMetadata({ sourceAudioUrl: audioUrl });
      if (sourceLookupRef.current !== lookup) return;
      if (dataset && applyDatasetMetadata(dataset, lyricEdit, instrumentalEdit)) return;

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
        if (sourceLookupRef.current === lookup) setAnalysis({ bpm, key, scale });
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
    } finally { if (sourceLookupRef.current === lookup) { setIsUploading(false); setIsAnalyzing(false); } }
  };

  // ── Lyrics search ──
  const handleSearchLyrics = async () => {
    if (!songArtist.trim() || !songTitle.trim()) { showToast(t('cover.enterArtistTitle')); return; }
    lyricEditRef.current++;
    setIsSearchingLyrics(true);
    try {
      const result = await lireekApi.searchSongLyrics(songArtist.trim(), songTitle.trim());
      setLyrics(result.lyrics);
      setLyricsSource(null);
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

  const prepareYue2Source = async (): Promise<{ sourceAudioUrl?: string; songId?: string; audioUrl: string }> => {
    if (advancedMode && sepStems?.length && sepJobId) {
      const controls = stemControls.map(c => ({ index: c.index, volume: c.muted ? 0 : c.volume, muted: c.muted }));
      const blob = await recombineStems(sepJobId, controls);
      const fd = new FormData(); fd.append('audio', blob, 'recombined-stems.wav');
      const response = await fetch('/api/upload/audio', { method: 'POST', body: fd });
      if (!response.ok) throw new Error('Could not upload the recombined source');
      const uploaded = await response.json() as { audio_url?: string };
      if (!uploaded.audio_url) throw new Error('Recombined source has no audio URL');
      return { sourceAudioUrl: uploaded.audio_url, audioUrl: uploaded.audio_url };
    }
    if (sourceSongId) return { songId: sourceSongId, audioUrl: sourceAudioUrl };
    return { sourceAudioUrl, audioUrl: sourceAudioUrl };
  };

  const handleTranscribe = async () => {
    if (!token || !sourceAudioUrl) return;
    const key = sourceKey;
    const request = ++sheetRequestRef.current;
    setSheetError(''); setApprovedSheet(null); setSheetAbc(''); setScoreSections([]); setSheetPreparing(true);
    try {
      const source = await prepareYue2Source();
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
      const result = await yue2CoverApi.start({ ...source, sourceLabel: sourceFileName.slice(0, 120), force: !!sheetAbc }, token);
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) {
        if (result.jobId) await yue2CoverApi.cancel(result.jobId, token);
        return;
      }
      setSheetAudioUrl(source.audioUrl);
      if (result.jobId) { sheetJobRef.current = result.jobId; setSheetJobId(result.jobId); }
      else if (result.abc) {
        setSheetAbc(result.abc); setScoreSections(result.sections || []);
        setSheetScoreSource(result.scoreSource === 'dataset' ? 'dataset' : null);
      }
    } catch (err: any) { if (sourceKeyRef.current === key && sheetRequestRef.current === request) setSheetError(err.message); }
    finally { if (sheetRequestRef.current === request) setSheetPreparing(false); }
  };

  const handleCancelSheet = async () => {
    sheetRequestRef.current += 1;
    const id = sheetJobRef.current;
    sheetJobRef.current = ''; setSheetJobId(''); setSheetPreparing(false);
    setSheetJob(null); setSheetError('Transcription cancelled');
    if (id && token) try { await yue2CoverApi.cancel(id, token); } catch (err: any) { setSheetError(err.message); }
  };

  const handleApproveSheet = async () => {
    if (!token || !sheetAbc.trim() || !sourceAudioUrl) return;
    const key = sourceKey;
    const request = ++sheetRequestRef.current;
    setSheetError(''); setSheetPreparing(true);
    try {
      const source = sheetAudioUrl
        ? { audioUrl: sheetAudioUrl, ...(sheetAudioUrl === sourceAudioUrl && sourceSongId ? { songId: sourceSongId } : { sourceAudioUrl: sheetAudioUrl }) }
        : await prepareYue2Source();
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
      const result = await yue2CoverApi.start({ ...source, sourceLabel: sourceFileName.slice(0, 120), abc: sheetAbc }, token);
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
      setApprovedSheet({ abc: result.abc || sheetAbc.trim(), sourceId: result.sourceId,
        sourceLabel: result.sourceLabel, audioUrl: source.audioUrl, key });
      setScoreSections(result.sections || []);
      setSheetAudioUrl(source.audioUrl);
    } catch (err: any) { if (sourceKeyRef.current === key && sheetRequestRef.current === request) setSheetError(err.message); }
    finally { if (sheetRequestRef.current === request) setSheetPreparing(false); }
  };

  // The original source, never a recombined mix: its vocal stem is what the
  // aligner reads, and its path is what finds the dataset song.
  const sectionSource = () => sourceSongId ? { songId: sourceSongId } : { sourceAudioUrl };
  const handleMatchSections = async () => {
    if (!token || !sheetAbc.trim() || !lyrics.trim() || !sourceAudioUrl) return;
    const key = sourceKey, before = lyrics;
    setSheetError(''); setSectionMatch(null); setSectionMatching(true);
    try {
      const match = await yue2CoverApi.matchSections({ ...sectionSource(), abc: sheetAbc, lyrics: before }, token);
      if (sourceKeyRef.current === key) setSectionMatch({ match, before, key });
    } catch (err: any) { if (sourceKeyRef.current === key) setSheetError(`Match sections: ${err.message}`); }
    finally { setSectionMatching(false); }
  };
  // The proposal is for the lyrics it was made from; edits since void it.
  const sectionMatchStale = !!sectionMatch && sectionMatch.before !== lyrics;
  const applySectionMatch = () => {
    if (!sectionMatch || sectionMatchStale) return;
    lyricEditRef.current++;
    setLyrics(sectionMatch.match.lyrics);
    setLyricsSource(null);
    setSectionMatch(null);
  };
  const saveSectionMatch = async () => {
    if (!token || !sectionMatch || sectionMatchStale) return;
    setSectionSaving(true);
    try {
      await yue2CoverApi.saveDatasetDetails({ ...sectionSource(), lyrics: sectionMatch.match.lyrics }, token);
      if (sourceKeyRef.current === sectionMatch.key) applySectionMatch();
      showToast('Lyrics saved to the dataset');
    } catch (err: any) { setSheetError(`Save to dataset: ${err.message}`); }
    finally { setSectionSaving(false); }
  };

  // The score's tempo and key, offered for a dataset song whose .txt differs.
  const scoreBpm = Math.round(Number(sheetAbc.match(/^Q:[^=\r\n]*=\s*(\d+(?:\.\d+)?)/m)?.[1]) || 0);
  const scoreKey = coverScoreKeyLabel(sheetAbc).replace(/(major|minor)$/, m => m[0].toUpperCase() + m.slice(1));
  const scoreDetailsDiffer = yue2Mode && datasetAnalysis && !!analysis && scoreBpm > 0 && !!scoreKey &&
    (Math.round(analysis.bpm) !== scoreBpm || analysis.key.toLowerCase() !== scoreKey.toLowerCase());
  const saveScoreDetails = async () => {
    if (!token || !scoreDetailsDiffer) return;
    const key = sourceKey, bpm = scoreBpm, musicalKey = scoreKey;
    setScoreDetailsSaving(true);
    try {
      await yue2CoverApi.saveDatasetDetails({ ...sectionSource(), bpm, key: musicalKey }, token);
      if (sourceKeyRef.current === key) {
        setAnalysis({ bpm, key: musicalKey, scale: musicalKey.split(' ')[1] });
        setBpmOverride(null); setKeyOverride(null);
      }
      showToast('Tempo and key saved to the dataset');
    } catch (err: any) { setSheetError(`Save to dataset: ${err.message}`); }
    finally { setScoreDetailsSaving(false); }
  };

  // ── Generation ──
  const captionSelection = captionMode === 'auto' ? { mode: 'auto' as const }
    : captionMode.startsWith('track:') ? { mode: 'track' as const, selectedName: captionMode.slice(6) }
    : { mode: 'custom' as const };
  const resolvedCaption = resolveYue2Caption(artistCaption, analysis?.bpm, captionTracks, captionSelection).caption;
  const adapterOptions = (kind: 'ar' | 'nar') => [
    { value: '', label: `Base ${kind.toUpperCase()}` },
    ...(yue2Models?.lmAdapters || []).filter(path =>
      (yue2Models?.lmAdapterMeta?.[path] as { kind?: string } | undefined)?.kind === kind,
    ).map(path => ({ value: path, label: yue2Models?.lmAdapterMeta?.[path]?.label || path.split(/[\\/]/).pop() || path })),
  ];

  const handleGenerate = () => {
    if (!token || !sourceAudioUrl) { showToast(t('cover.missingSrcOrLyrics')); return; }
    if (!instrumental && !lyrics.trim()) { showToast('Enter lyrics or enable Instrumental mode'); return; }
    if (yue2Mode) {
      if (!approvedSheet || approvedSheet.key !== sourceKey) { showToast('Review and approve the score first'); return; }
      if (pairMode === 'pair' && (!yue2Ar || !yue2Nar)) { showToast('Choose both YuE2 adapter halves'); return; }
      // Take the whole request at click time; the serial queue may start it much later.
      const engineParams = gp.getGlobalParams() as Record<string, unknown>;
      const yue2Params = Object.fromEntries(Object.entries(engineParams).filter(([key]) => key.startsWith('yue2')));
      // Create sends these shared settings with its YuE2 requests too. Keep
      // ACE model, adapter, sampler and source-audio controls out of this job.
      const sharedParams = {
        backend: 'yue2', seed: engineParams.seed, randomSeed: engineParams.randomSeed,
        batchSize: engineParams.batchSize, duration: -1,
        postProcessingEnabled: engineParams.postProcessingEnabled,
        masteringEnabled: engineParams.masteringEnabled, masteringReference: engineParams.masteringReference,
        lufsEnabled: engineParams.lufsEnabled, lufsTarget: engineParams.lufsTarget,
        lufsCeilingDb: engineParams.lufsCeilingDb,
        stableStepOn: engineParams.stableStepOn, stableStepStrength: engineParams.stableStepStrength,
        stableStepBackend: engineParams.stableStepBackend, stableStepAdapters: engineParams.stableStepAdapters,
        stableStepSeed: engineParams.stableStepSeed,
        stableStepSeedFollowsDit: engineParams.stableStepSeedFollowsDit,
        stableStepSteps: engineParams.stableStepSteps, stableStepSolver: engineParams.stableStepSolver,
        stableStepScheduler: engineParams.stableStepScheduler,
        stableStepGuidanceMode: engineParams.stableStepGuidanceMode,
        stableStepGuidanceScale: engineParams.stableStepGuidanceScale,
        stableStepPluginParams: engineParams.stableStepPluginParams,
        whisperLyricsEnabled: engineParams.whisperLyricsEnabled, whisperModel: engineParams.whisperModel,
        whisperLanguage: engineParams.whisperLanguage, whisperBeamSize: engineParams.whisperBeamSize,
        whisperIsolateVocals: engineParams.whisperIsolateVocals,
        qualityEvalEnabled: engineParams.qualityEvalEnabled, qualityEvalTarget: engineParams.qualityEvalTarget,
        coverArtEnabled: engineParams.coverArtEnabled, coverArtSubject: engineParams.coverArtSubject,
        coResident: settings.coResident, cacheLmCodes: settings.cacheLmCodes,
        parallelWhisper: settings.parallelWhisper, parallelQualityEval: settings.parallelQualityEval,
        parallelCoverArt: settings.parallelCoverArt,
      };
      const pair = { ...yue2PickAtEnqueue(null),
        lmAdapterAr: pairMode === 'pair' ? yue2Ar : '',
        lmAdapterNar: pairMode === 'pair' ? yue2Nar : '' };
      const title = songArtist ? `${songTitle || 'Cover'} (${songArtist} Cover)` : (songTitle || 'Cover');
      const scoreSourceKey = coverScoreKeyLabel(approvedSheet.abc);
      const movedKey = coverKeyShift && scoreSourceKey ? transposeKey(scoreSourceKey, coverKeyShift) : '';
      const renderKey = movedKey ? movedKey.replace(/ major$/, '').replace(/ minor$/, 'm') : 'source';
      const coverChoices = { voices: coverVoices, keepChords,
        tempo: coverTempoMode === 'set' ? coverBpm : coverTempoMode,
        key: renderKey, cfgScale: coverCfgScale };
      const params = { ...sharedParams, ...yue2Params, yue2CfgScale: coverCfgScale, customMode: true, taskType: 'text2music',
        title, caption: resolvedCaption, style: resolvedCaption, lyrics: instrumental ? '' : lyrics,
        ...(lyricsSource ? { lyricsSource } : {}),
        ...(sheetScoreSource ? { scoreSource: 'dataset-sidecar' } : {}),
        instrumental, source: 'cover-studio', sourceAudioUrl: approvedSheet.audioUrl,
        yue2Cover: { sourceId: approvedSheet.sourceId, sourceLabel: approvedSheet.sourceLabel, ...coverChoices },
        yue2Abc: approvedSheet.abc, yue2Cot: keepChords ? 'full' : 'melody', yue2Pick: pair };
      const qId = addManualQueueItem({ title, artistName: '', caption: resolvedCaption });
      updateManualQueueItem(qId, { stage: _coverRunning ? 'Queued…' : 'Preparing…' });
      setIsGenerating(true);
      enqueueCoverJob(async () => {
        try {
          const res = await generateApi.submit(params as any, token);
          updateManualQueueItem(qId, { jobId: res.jobId });
          await pollJobAsync(res.jobId, qId);
        } catch (err: any) { failManualQueueItem(qId, err.message || 'Generation failed'); }
        finally { if (_coverQueue.length === 0) { setIsGenerating(false); setActiveJobId(null); setGenProgress(0); setGenStage(''); } }
      });
      return;
    }

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
        ...(lyricsSource ? { lyricsSource } : {}),
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
          showToast(t('cover.coverGenerated'));

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
    sourceLookupRef.current++;
    setSourceFileName(''); setSourceAudioUrl(''); setSourceSongId('');
    setMetadata(null); setAnalysis(null);
    setSongArtist(''); setSongTitle(''); setLyrics('');
    setLyricsSource(null); setDatasetAnalysis(false); setSheetScoreSource(null);
    setBpmCorrection(1); setBpmOverride(null); setKeyOverride(null);
    // Clear stems too — releases the split job server-side.
    clearStems();
  };

  const handleClearArtist = () => {
    setSelectedArtistId(null);
    setSelectedPreset(null);
    setArtistPresets([]);
    setArtistCaption('');
  };

  // Always allow queuing another cover — covers stack and run one at a time (#62).
  const canGenerate = !!sourceAudioUrl && (!!lyrics.trim() || instrumental)
    && (!yue2Mode || (!!approvedSheet && approvedSheet.key === sourceKey && (pairMode === 'base' || (!!yue2Ar && !!yue2Nar))));

  // ── SuperSep handlers ──
  // Split + poll now runs inside coverStemsStore.startSeparation, so it keeps
  // going to completion even if the user leaves Cover Studio mid-split (#135).
  const handleSeparate = useCallback(async () => {
    if (!sourceAudioUrl) { showToast(t('cover.uploadAudioFirst')); return; }
    try {
      await startStemSeparation(sourceAudioUrl, sepLevel);
      const count = useCoverStemsStore.getState().sepStems?.length ?? 0;
      showToast(t('cover.separatedIntoStems', { count }));
    } catch (err: any) {
      showToast(`Separation failed: ${err.message}`);
    }
  }, [sourceAudioUrl, sepLevel, startStemSeparation]);



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
      {/* Scrolls sideways when the three panels' minimum widths don't fit. */}
      <div className="flex-1 flex overflow-x-auto overflow-y-hidden">
        {/* Left: Source Audio */}
        <SourcePanel
          yue2Mode={yue2Mode}
          sourceFileName={sourceFileName} metadata={metadata} analysis={analysis}
          fromDataset={datasetAnalysis}
          scoreDetails={scoreDetailsDiffer ? { bpm: scoreBpm, key: scoreKey, saving: scoreDetailsSaving,
            onSave: () => setScoreDetailsConfirm(true) } : null}
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

        {/* Center: Lyrics. The side panels shrink before it does. */}
        <div className="flex-1 min-w-[320px] flex flex-col overflow-hidden border-r border-zinc-200 dark:border-white/5">
          <div className="flex-shrink-0 px-4 py-3 border-b border-zinc-200 dark:border-white/5">
            <div className="flex items-center justify-between gap-2">
              {/* Artist + Title inputs */}
              <div className="flex-1 flex gap-2">
                <input value={songArtist} onChange={e => setSongArtist(e.target.value)}
                  placeholder={t('cover.artistPlaceholder')} className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 focus:outline-none focus:border-cyan-500" />
                <input value={songTitle} onChange={e => setSongTitle(e.target.value)}
                  placeholder={t('cover.songTitlePlaceholder')} className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 focus:outline-none focus:border-cyan-500" />
              </div>
              <button onClick={handleSearchLyrics} disabled={!!lyricsSource || isSearchingLyrics || !songArtist.trim() || !songTitle.trim()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-xs font-medium transition-colors disabled:opacity-50">
                {isSearchingLyrics ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
                Genius
              </button>
            </div>
          </div>
          <div className="flex-1 min-h-0 p-4 flex flex-col gap-1">
            {lyricsSource && <span className="text-[10px] text-cyan-600 dark:text-cyan-300">Lyrics from dataset · editable</span>}
            <textarea value={lyrics} onChange={e => { lyricEditRef.current++; setLyrics(e.target.value); setLyricsSource(null); }}
              placeholder={t('cover.lyricsPlaceholder')}
              className="w-full min-h-0 flex-1 resize-none bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-xl px-4 py-3 text-sm text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-cyan-500 transition-colors font-mono leading-relaxed" />
          </div>
          {yue2Mode && !!sheetAbc && <div className="flex-shrink-0 px-4 pb-3 space-y-1">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">Score sections</span>
              <button type="button" onClick={() => void handleMatchSections()}
                disabled={!scoreSections.length || !lyrics.trim() || instrumental || sectionMatching}
                title="Align the lyrics on the source's vocal stem and retag each block with the score section it is sung in. Shows the changes before applying."
                className="text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">
                {sectionMatching ? 'Matching… (separating vocals the first time)' : 'Match sections to score'}</button>
            </div>
            {sectionMatch && sectionMatch.key === sourceKey && <SectionMatchReview before={sectionMatch.before} match={sectionMatch.match} stale={sectionMatchStale}
              saving={sectionSaving} onApply={applySectionMatch} onSave={() => setSectionSaveConfirm(true)} onDiscard={() => setSectionMatch(null)} />}
            {/* In-page, not window.confirm: a native dialog stalls browser automation. */}
            <ConfirmDialog isOpen={sectionSaveConfirm} title="Save lyrics to dataset"
              message="Write these lyrics into the dataset song’s .txt? The current file is kept as a backup."
              confirmLabel="Save" onCancel={() => setSectionSaveConfirm(false)}
              onConfirm={() => { setSectionSaveConfirm(false); void saveSectionMatch(); }} />
            <ConfirmDialog isOpen={scoreDetailsConfirm} title="Save tempo and key to dataset"
              message={`Replace the dataset song’s tempo and key with the score’s (${scoreBpm} BPM, ${scoreKey})? The current .txt is kept as a backup.`}
              confirmLabel="Save" onCancel={() => setScoreDetailsConfirm(false)}
              onConfirm={() => { setScoreDetailsConfirm(false); void saveScoreDetails(); }} />
            {scoreSections.length > 0
              ? <div className="flex flex-wrap gap-1">{scoreSections.map((section, index) =>
                <span key={`${index}-${section.startBar}`} className="rounded bg-cyan-500/10 px-2 py-0.5 text-[11px] text-cyan-800 dark:text-cyan-200">
                  {section.label} · bar {section.startBar}
                </span>)}</div>
              : <p className="text-xs text-zinc-500">No score section labels found.</p>}
            {!instrumental && sectionLint && !sectionLint.ok &&
              <p role="status" className="text-xs text-amber-700 dark:text-amber-300">{sectionLint.message}</p>}
          </div>}
          {yue2Mode && <Yue2CoverScore
            sourceReady={!!sourceAudioUrl && !isUploading}
            readiness={readiness} job={sheetJob} preparing={sheetPreparing} error={sheetError}
            abc={sheetAbc} approved={!!approvedSheet && approvedSheet.key === sourceKey}
            onAbcChange={value => { sheetRequestRef.current += 1; setSheetPreparing(false); setSheetAbc(value);
              setScoreSections([]); setSectionLint(null); setSheetScoreSource(null); setApprovedSheet(null); }}
            scoreSource={sheetScoreSource}
            onTranscribe={handleTranscribe} onCancel={handleCancelSheet} onApprove={handleApproveSheet}
          />}
          
          {showMixer && sepStems && sepJobId && (
            <StemMixer jobId={sepJobId} stems={sepStems}
              controls={stemControls} onControlsChange={setStemControls}
              onClose={() => setShowMixer(false)} />
          )}
        </div>

        {/* Right: Artist + Settings */}
        {yue2Mode ? <Yue2CoverPanel
          caption={artistCaption} onCaptionChange={setArtistCaption}
          captionMode={captionMode} onCaptionMode={setCaptionMode}
          captionTracks={captionTracks} resolvedCaption={resolvedCaption}
          instrumental={instrumental} onInstrumentalChange={value => { instrumentalEditRef.current++; setInstrumental(value); setLyricsSource(null); }}
          pairMode={pairMode} onPairMode={setPairMode}
          ar={yue2Ar} nar={yue2Nar} onAr={setYue2Ar} onNar={setYue2Nar}
          arOptions={adapterOptions('ar')} narOptions={adapterOptions('nar')}
          keepChords={keepChords} onKeepChords={setKeepChords}
          scoreAbc={sheetAbc} scoreKeyLabel={coverScoreKeyLabel(sheetAbc)}
          voices={coverVoices} onVoices={setCoverVoices}
          tempoMode={coverTempoMode} onTempoMode={setCoverTempoMode}
          bpm={coverBpm} onBpm={setCoverBpm}
          keyShift={coverKeyShift} onKeyShift={setCoverKeyShift}
          cfgScale={coverCfgScale} onCfgScale={setCoverCfgScale}
          canGenerate={canGenerate} isGenerating={isGenerating} genProgress={genProgress} genStage={genStage}
          onGenerate={handleGenerate} onCancel={handleCancel}
        /> : <ArtistSettingsPanel
          artists={artists} isLoadingArtists={isLoadingArtists}
          selectedArtistId={selectedArtistId} onSelectArtist={handleSelectArtist}
          onClearArtist={handleClearArtist}
          artistPresets={artistPresets} selectedPreset={selectedPreset}
          onSelectPreset={(p) => { setSelectedPreset(p); applyPresetToGlobal(p); }}
          audioCoverStrength={audioCoverStrength} onAudioCoverStrength={setAudioCoverStrength}
          coverNoiseStrength={coverNoiseStrength} onCoverNoiseStrength={setCoverNoiseStrength}
          coverNoiseMethod={coverNoiseMethod} onCoverNoiseMethodChange={setCoverNoiseMethod}
          noFsq={noFsq} onNoFsqChange={setNoFsq}
          instrumental={instrumental} onInstrumentalChange={value => { instrumentalEditRef.current++; setInstrumental(value); setLyricsSource(null); }}
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
        />}

      </div>
    </div>
    </BackendCapabilityGate>
  );
};
