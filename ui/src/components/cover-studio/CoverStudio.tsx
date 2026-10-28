// CoverStudio.tsx — Main Cover Studio orchestrator
// Composes: SourcePanel, ArtistSettingsPanel

import { pollWhile } from '../../queries/contract';
import { waitForQuery } from '../../queries/wait';
import { generationStatusOptions } from '../../queries/shell-queue/queue';
import { songsChanged } from '../../queries/cover-lyrics/reads';
import type { Yue2CoverResult } from '../../services/yue2CoverApi';
import { useMutation, useQuery } from '@tanstack/react-query';
import { snapshotOptions } from '../../queries/cover-lyrics/reads';
import type { LyricsSet } from '../../services/lireekApi';
import React, { useState, useEffect, useCallback, useRef } from 'react';
import type { Song } from '../../types';
import { Search, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { sheetTranscriberChoice } from './sheetTranscriber';
import { usePersistedState } from '../../hooks/usePersistedState';
import { DEFAULT_SETTINGS, type AppSettings } from '../settings/SettingsPanel';
import { generateApi } from '../../services/api';
import type { GenerationJob } from '../../types';
import { apiClient, ApiError } from '../../services/httpClient';
import { useUpload } from '../../queries/writes';
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
  persist, restore, getTrackCache, saveTrackCacheEntry, transposeKey, yue2CoverContentMissing,
  type AudioMetadata, type AudioAnalysis,
} from './coverStudioUtils';
import { useStudioDraftMirror } from '../../services/studioDraftMirror';
import { StudioDraftPicker } from '../shared/StudioDraftPicker';
import { Fold } from '../shared/Fold';
import { exportPrepare, structureHash, type PrepareV2 } from '../../utils/yue2Prepare';
import { normalizeAbc } from '../../../../server/src/contracts/yue2Prepare';
import { LyricsSectionHighlight } from './LyricsSectionHighlight';
import type { LatentMetadata } from '../shared/LatentImport';
import { loadSelections, saveSelections } from '../lyric-studio/ProviderSelector';
import { useBackendStore } from '../../stores/backendStore';
import { fetchYue2CaptionSource, resolveYue2Caption, yue2PickAtEnqueue, type Yue2SourceTrack } from '../../utils/yue2CaptionSource';
import { yue2CoverApi, type Yue2CoverDatasetMetadata, type Yue2CoverJob, type Yue2CoverReadiness,
  type Yue2ScoreSection } from '../../services/yue2CoverApi';
import { Yue2CoverPanel, COVER_AR_DEFAULT, clampCoverAr, COVER_GUIDE_DEFAULT, coverGuideBlock, type CoverGuideState } from './Yue2CoverPanel';
import { resolveYue2Render, type Yue2RenderMode } from './yue2RenderMode';
import { Yue2CoverScore } from './Yue2CoverScore';
import { CoverResizeHandle } from './CoverResizeHandle';
import { startDragResize } from './startDragResize';
import { MusicXmlImport } from './MusicXmlImport';
import { PrepareStructure } from './PrepareStructure';
import { mainMeter, type ScoreRange } from './prepareView';
import { PrepareLyrics } from './PrepareLyrics';
import { useYue2Prepare, PREPARE_DRAFT_KEY, type PrepareCut } from './useYue2Prepare';
import { Yue2BuildPage, buildCaption, buildNext, draftText, finishSavedBuild, onBuildSaved, useYue2Build, type BuildSettings } from './Yue2Build';
import { barRanges } from '../../utils/abcSectionMove';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { AbcChart } from './AbcChart';
import { coverWorkflowApi, waitForCoverJob, type CoverDraftResult } from '../../services/coverWorkflowApi';
import { coverLyricWorkflowApi as workflowApi } from '../../queries/cover-lyrics/workflows';

// ── Serial cover-generation queue ────────────────────────────────────────────
// Lets the user stack multiple cover generations (different settings) without
// waiting — same pattern as InstaGen's queue. Jobs run one at a time; the engine
// serializes anyway, this keeps the client-side recombine→submit→poll ordered.
const _coverQueue: Array<() => Promise<void>> = [];
let _coverRunning = false;
/** Builds with a Save song queued or rendering: Delete build waits, because the render reads the build's files. */
const _savingBuilds = new Map<string, number>();
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

// ABC mode suffixes SheetSage2-Refined's key_mode writes (K:Ddor, K:Gmix, ...).
const KEY_MODES: Record<string, string> = { dor: 'dorian', phr: 'phrygian', lyd: 'lydian', mix: 'mixolydian', loc: 'locrian' };

function coverScoreKeyLabel(abc: string): string {
  const key = abc.match(/^K:\s*([A-Ga-g][#b]?)(mix|dor|phr|lyd|loc|m)?(?:\s+(minor|major|maj))?(?=\s|$)/m);
  if (!key) return '';
  if (key[2] && KEY_MODES[key[2]]) return `${key[1]} ${KEY_MODES[key[2]]}`;
  return `${key[1]} ${key[2] || key[3]?.toLowerCase() === 'minor' ? 'minor' : 'major'}`;
}

/** The score choices params.yue2Cover carries for `s`, its key moved from score `abc`'s. */
function coverChoicesFor(abc: string, s: Omit<BuildSettings, 'arStrength'>) {
  const sourceKey = coverScoreKeyLabel(abc);
  const movedKey = s.keyShift && sourceKey ? transposeKey(sourceKey, s.keyShift) : '';
  const abcMode = (label: string) => Object.entries(KEY_MODES).reduce(
    (text, [suffix, name]) => text.replace(new RegExp(` ${name}$`), suffix), label);
  const key = movedKey ? abcMode(movedKey.replace(/ major$/, '').replace(/ minor$/, 'm')) : 'source';
  return { voices: s.voices, tempo: s.tempo, key, ...(movedKey ? { keyShift: s.keyShift } : {}), cfgScale: s.cfgScale };
}

/** A build's settings, read back from its params. */
function buildSettingsOf(params: Record<string, unknown>): BuildSettings {
  const cover = (params.yue2Cover ?? {}) as Record<string, unknown>;
  const ar = Number((params.yue2Pick as Record<string, unknown> | undefined)?.lmAdapterArScale);
  return { voices: cover.voices === 'vocal' ? 'vocal' : 'both',
    tempo: typeof cover.tempo === 'number' || cover.tempo === 'source' ? cover.tempo : 'free',
    keyShift: Number(cover.keyShift ?? 0), cfgScale: Number(params.yue2CfgScale ?? cover.cfgScale ?? 1),
    arStrength: Number.isFinite(ar) ? ar : COVER_AR_DEFAULT };
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
  const [sourceMode, setSourceMode] = useState<'audio' | 'musicxml'>(() => restore('sourceMode', 'audio'));
  const [draftName, setDraftName] = useState(() => restore('draftName', ''));
  const [sourceWidth, setSourceWidth] = usePersistedState('hs-cover-source-width', 320);
  const [scoreWidth, setScoreWidth] = usePersistedState('hs-cover-score-width', 560);
  const [settingsWidth, setSettingsWidth] = usePersistedState('hs-cover-settings-width', 420);
  const [scoreHeight, setScoreHeight] = usePersistedState('hs-cover-score-height', 320);
  const startResize = (event: React.MouseEvent, width: number, setWidth: (value: number) => void, min: number, max: number) =>
    startDragResize(event, 'x', width, setWidth, min, max);
  const startScoreResize = (event: React.MouseEvent) => startDragResize(event, 'y', scoreHeight, setScoreHeight, 160, 900, -1);

  // ── Source audio state ──
  const [sourceFileName, setSourceFileName] = useState(() => restore<string>('sourceFileName', ''));
  const [sourceAudioUrl, setSourceAudioUrl] = useState(() => restore<string>('sourceAudioUrl', ''));
  const [sourceAssetId, setSourceAssetId] = useState(() => restore<string>('sourceAssetId', ''));
  const [coverDraftId, setCoverDraftId] = useState(() => restore<string>('coverDraftId', ''));
  const [coverDraftRevision, setCoverDraftRevision] = useState(() => restore<number>('coverDraftRevision', 0));
  const coverWorkflowJobRef = useRef('');
  const coverRenderJobRef = useRef('');
  const [sourceSongId, setSourceSongId] = useState(() => restore<string>('sourceSongId', ''));
  const [metadata, setMetadata] = useState<AudioMetadata | null>(() => restore('metadata', null));
  const [analysis, setAnalysis] = useState<AudioAnalysis | null>(() => restore('analysis', null));
  const [useDetectedDetails, setUseDetectedDetails] = usePersistedState('hs-cover-use-detected-details', true);
  const uploadMut = useUpload<{ audio_url?: string; asset_id?: string }>();
  const isUploading = uploadMut.isPending;
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

  // ── Target artist ──
  const artistsQuery = useQuery(snapshotOptions<Artist[]>('lireek', '/lireek/artists'));
  const setsQuery = useQuery(snapshotOptions<LyricsSet[]>('lireek', '/lireek/lyrics-sets'));
  const presetsQuery = useQuery(snapshotOptions<{ presets: AlbumPreset[] }>('lireek', '/lireek/presets'));
  const artists = artistsQuery.data ?? [];
  const allSets = setsQuery.data ?? [];
  const allPresets = presetsQuery.data?.presets ?? [];
  const isLoadingArtists = artistsQuery.isPending;
  const isLoadingYue2Artists = setsQuery.isPending || presetsQuery.isPending;
  const [selectedArtistId, setSelectedArtistId] = useState<number | null>(() => restore('selectedArtistId', null));
  const [selectedPreset, setSelectedPreset] = useState<AlbumPreset | null>(() => restore('selectedPreset', null));
  const [artistCaption, setArtistCaption] = useState(() => restore<string>('artistCaption', ''));
  const artistPresets = allSets.filter(ls => ls.artist_id === selectedArtistId).map(ls => ({
    lsId: ls.id, album: ls.album || String(ls.id), preset: allPresets.find(p => p.lyrics_set_id === ls.id) ?? null,
  }));

  // ── Cover caption LLM ──
  const [coverCaptionProvider, setCoverCaptionProvider] = useState(() => loadSelections().coverCaption.provider);
  const [coverCaptionModel, setCoverCaptionModel] = useState(() => loadSelections().coverCaption.model);
  // The background preset waterfall's caption leg; the manual regenerate is a
  // mutation — the button spins on either (see isGeneratingCaption below).
  const [bgCaptionPending, setBgCaptionPending] = useState(false);
  const captionRequestRef = useRef(0);
  const captionJobRef = useRef('');

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
  const [renderMode, setRenderMode] = useState<Yue2RenderMode>(() => {
    const saved = restore<string>('pairMode', 'base');
    return saved === 'pair' ? 'custom' : saved === 'artist' || saved === 'custom' ? saved : 'base';
  });
  const [yue2Ar, setYue2Ar] = useState(() => restore('yue2Ar', ''));
  const [yue2Nar, setYue2Nar] = useState(() => restore('yue2Nar', ''));
  // #304: Base renders with no adapters, so a pair picked in the model picker
  // was silently dropped. Custom's empty slots take the picker's pair, and a
  // studio with no mode saved, loaded or clicked yet opens in Custom when the
  // picker holds both halves. A saved mode always wins.
  const modeChosen = useRef(restore<string | null>('pairMode', null) !== null);
  const chooseRenderMode = (mode: Yue2RenderMode) => { modeChosen.current = true; setRenderMode(mode); };
  const pickerAr = String(yue2Models?.defaults?.lmAdapterAr ?? '');
  const pickerNar = String(yue2Models?.defaults?.lmAdapterNar ?? '');
  useEffect(() => {
    if (pickerAr) setYue2Ar(v => v || pickerAr);
    if (pickerNar) setYue2Nar(v => v || pickerNar);
    if (pickerAr && pickerNar && !modeChosen.current) { modeChosen.current = true; setRenderMode('custom'); }
  }, [pickerAr, pickerNar]);
  const [selectedYue2ArtistId, setSelectedYue2ArtistId] = useState<number | null>(() => restore('selectedYue2ArtistId', null));
  const [selectedYue2LyricsSetId, setSelectedYue2LyricsSetId] = useState<number | null>(() => restore('selectedYue2LyricsSetId', null));
  const eligibleIds = new Set(allPresets.filter(p => p.yue2_ar_adapter_path && p.yue2_nar_adapter_path).map(p => p.lyrics_set_id));
  const yue2Artists = artists.filter(a => allSets.some(ls => ls.artist_id === a.id && eligibleIds.has(ls.id)));
  const [yue2AlbumsForArtistId, setYue2AlbumsForArtistId] = useState<number | null>(null);
  const [yue2Albums, setYue2Albums] = useState<{ lsId: number; album: string; preset: AlbumPreset | null }[]>([]);
  const [coveringBand, setCoveringBand] = useState(() => restore<string>('coveringBand', ''));
  // Build by section: compose a few sections at a time and keep the best take.
  // The Build page (true) or the Prepare page (false).
  const [buildMode, setBuildMode] = useState(() => restore<boolean>('buildMode', false));
  const [buildSessionId, setBuildSessionId] = useState(() => restore<string>('buildSessionId', ''));
  useEffect(() => onBuildSaved(id => setBuildSessionId(cur => cur === id ? '' : cur)), []);
  // The build's own lyrics before its first session, tied to the source they
  // were written for. Prepare's lyrics and the dataset are never changed by it.
  const [buildLyricsDraft, setBuildLyricsDraft] = useState<{ source: string; text: string } | null>(() => restore('buildLyricsDraft', null));
  const [coverVoices, setCoverVoices] = useState<'vocal' | 'both'>(() => restore('coverVoices', 'both'));
  const [keepChords, setKeepChords] = useState(() => restore('keepChords', false));
  const [coverTempoMode, setCoverTempoMode] = useState<'free' | 'source' | 'set'>(() => restore('coverTempoMode', 'free'));
  const [coverBpm, setCoverBpm] = useState(() => restore('coverBpm', 120));
  const [coverKeyShift, setCoverKeyShift] = useState(() => restore('coverKeyShift', 0));
  const [coverCfgScale, setCoverCfgScale] = useState(() => restore('coverCfgScale', 1));
  const [coverGuide, setCoverGuide] = useState<CoverGuideState>(() => ({ ...COVER_GUIDE_DEFAULT, ...restore('coverGuide', {}) }));
  // Cover's own AR strength; it replaces the adapter picker's AR scale in every Cover request.
  const [coverArStrength, setCoverArStrength] = useState(COVER_AR_DEFAULT);
  const [coverLyricSchedule, setCoverLyricSchedule] = useState<'off' | 'bias' | 'mask'>(() => restore('coverLyricSchedule', 'off'));
  const [captionMode, setCaptionMode] = useState(() => restore('captionMode', 'custom'));
  const [captionTracks, setCaptionTracks] = useState<Yue2SourceTrack[]>([]);
  const readinessQuery = useQuery({ ...snapshotOptions<Yue2CoverReadiness>('cover', '/yue2-cover/readiness'), enabled: yue2Mode && !!token });
  const readiness = readinessQuery.data ?? null;
  const [sheetJob, setSheetJob] = useState<Yue2CoverJob | null>(null);
  const [sheetJobId, setSheetJobId] = useState('');
  const sheetJobRef = useRef('');
  const sheetRequestRef = useRef(0);
  const [sheetPreparing, setSheetPreparing] = useState(false);
  const [sheetError, setSheetError] = useState('');
  const [sheetAbc, setSheetAbc] = useState(() => restore('sheetAbc', ''));
  const [scoreSections, setScoreSections] = useState<Yue2ScoreSection[]>(() => restore('scoreSections', []));
  // The score section playing or last picked, marked in Prepare's lyric rows.
  // Kept with the score it indexes, so a new score drops it.
  const [activeSectionOf, setActiveSectionOf] = useState<{ abc: string; index: number } | null>(null);
  const activeSection = activeSectionOf?.abc === sheetAbc ? activeSectionOf.index : null;
  // Prepare's selection and playback in the score view (Hear score).
  const [scoreHighlight, setScoreHighlight] = useState<[number, number] | null>(null);
  const [scorePlay, setScorePlay] = useState<{ range: ScoreRange | null } | null>(null);
  const [scorePlaying, setScorePlaying] = useState(false);
  // Every start of the score player, so the recording can be silenced by any of them.
  const [scoreStarts, setScoreStarts] = useState(0);
  const onScorePlaying = (playing: boolean) => { setScorePlaying(playing); if (playing) setScoreStarts(n => n + 1); };
  // A new object each call, so asking to stop is acted on even when nothing was playing from a range.
  const playScore = (range: ScoreRange | null) => setScorePlay({ range });
  // ½×/2× on the score tempo, for when the transcriber counted half or double time.
  const [scoreTempoScale, setScoreTempoScale] = useState(() => restore('scoreTempoScale', { abc: '', factor: 1 }));
  const [scoreDetailsConfirm, setScoreDetailsConfirm] = useState(false);
  const [sheetAudioUrl, setSheetAudioUrl] = useState(() => restore('sheetAudioUrl', ''));
  const [approvedSheet, setApprovedSheet] = useState<{ abc: string; sourceId: string; sourceLabel: string; audioUrl: string; key: string } | null>(() => restore('approvedSheet', null));

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
  const [advancedMode, setAdvancedMode] = useState(() => restore('advancedMode', false));
  const [sepLevel, setSepLevel] = useState<SeparationLevel>(() => restore('sepLevel', 1) as SeparationLevel);
  const {
    sepJobId, sepStems, stemControls, showMixer, isSeparating, sepProgress, sepMessage,
    setShowMixer, setStemControls, startSeparation: startStemSeparation, clearStems,
  } = useCoverStemsStore();
  const sourceKey = JSON.stringify([sourceMode, sourceAudioUrl, sourceSongId, sourceAssetId, advancedMode && sepJobId,
    advancedMode && sepStems?.length, advancedMode && stemControls]);
  const sourceKeyRef = useRef(sourceKey);
  const sourceResetKeyRef = useRef(sourceKey);
  const skipSourceResetRef = useRef(false);
  sourceKeyRef.current = sourceKey;
  // Prepare's document: the score's structure, then the lyrics assigned to
  // its sections. It owns the score while it is edited (sheetAbc follows it);
  // a score that arrives from elsewhere (a transcription, an import, a
  // dataset sidecar) starts a new document. Cuts live in the document's hook.
  const prep = useYue2Prepare({ sourceKey, abc: sheetAbc, lyrics,
    hasRecording: sourceMode === 'audio' && !!sourceAudioUrl });
  const docAbc = prep.doc.abc;
  // A transcription result becomes the score only when it arrives: a retry that fails or is cancelled
  // leaves the current preparation alone. A different score replaces the document (recoverable below)
  // and needs approving again; the same one changes nothing.
  const acceptTranscript = (abc: string, sections: Yue2ScoreSection[]) => {
    if (normalizeAbc(abc) !== docAbc) setApprovedSheet(null);
    setSheetAbc(abc); setScoreSections(sections);
  };
  // The preparation a replacing score pushed aside, to bring back.
  const [replaced, setReplaced] = useState<{ doc: PrepareV2; cuts: PrepareCut[]; sourceKey: string } | null>(null);
  const lastDocAbc = useRef(docAbc);
  const pendingDraft = useRef<Record<string, unknown> | null>(null);
  useEffect(() => {
    // A score from outside the document replaces it; sheetAbc and the document already agree otherwise.
    if (normalizeAbc(sheetAbc) === docAbc) return;
    if (sheetAbc.trim() && docAbc) setReplaced({ doc: prep.doc, cuts: prep.cuts, sourceKey });
    prep.restoreDraft({}, { sourceKey, abc: sheetAbc, lyrics });
  }, [sheetAbc]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    // An edit, undo or redo of the document's score reaches the rest of the studio.
    if (lastDocAbc.current === docAbc) return;
    lastDocAbc.current = docAbc;
    if (normalizeAbc(sheetAbc) !== docAbc) changeSheetAbc(docAbc);
  }, [docAbc]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    // A restored draft whose source key arrives a render later.
    if (!pendingDraft.current) return;
    const fields = pendingDraft.current;
    pendingDraft.current = null;
    prep.restoreDraft(fields, { sourceKey, abc: sheetAbc, lyrics });
  }, [sourceKey]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (sourceResetKeyRef.current === sourceKey) return;
    sourceResetKeyRef.current = sourceKey;
    if (skipSourceResetRef.current) { skipSourceResetRef.current = false; return; }
    sheetRequestRef.current += 1;
    setApprovedSheet(null); setSheetAbc(''); setScoreSections([]);
    setSheetAudioUrl(''); setSheetError(''); setSheetJob(null); setSheetPreparing(false);
    if (sheetJobRef.current && token) void yue2CoverApi.cancel(sheetJobRef.current).catch(() => {});
    sheetJobRef.current = ''; setSheetJobId('');
  }, [sourceKey, token]);

  useEffect(() => {
    if (!yue2Mode || !token) return;
    void useBackendStore.getState().fetchModels('yue2');
  }, [yue2Mode, token]);

  useEffect(() => {
    if (!yue2Mode || renderMode === 'base') { setCaptionTracks([]); return; }
    const album = yue2Albums.find(a => a.lsId === selectedYue2LyricsSetId) || yue2Albums[0];
    const adapter = renderMode === 'artist' ? album?.preset?.yue2_ar_adapter_path || album?.preset?.yue2_nar_adapter_path || '' : yue2Ar || yue2Nar;
    if (!adapter) { setCaptionTracks([]); return; }
    let live = true;
    void fetchYue2CaptionSource({ adapter }).then(result => { if (live) setCaptionTracks(result.tracks); });
    return () => { live = false; };
  }, [yue2Mode, renderMode, yue2Ar, yue2Nar, yue2Albums, selectedYue2LyricsSetId]);

  const sheetSnapshot = useQuery({
    ...snapshotOptions<Yue2CoverResult>('cover', `/yue2-cover/transcriptions/${encodeURIComponent(sheetJobId)}`),
    enabled: !!sheetJobId && !!token,
    refetchInterval: pollWhile<Yue2CoverResult>(data => !data || !!data.job && ['queued', 'running'].includes(data.job.status)),
  });
  useEffect(() => {
    if (!sheetJobId || sheetJobRef.current !== sheetJobId) return;
    const result = sheetSnapshot.data;
    if (!result) return;
    setSheetJob(result.job ?? null);
    if (result.job?.status === 'done') {
      acceptTranscript(result.abc || '', result.sections || []);
      setSheetJobId(''); sheetJobRef.current = '';
    } else if (result.job?.status === 'failed' || result.job?.status === 'cancelled') {
      setSheetError(result.job.error || `Transcription ${result.job.status}`);
      setSheetJobId(''); sheetJobRef.current = '';
    }
  }, [sheetJobId, sheetSnapshot.data]);

  useEffect(() => {
    if (!yue2Mode || !token || !sheetAbc.trim()) return;
    let live = true;
    const timer = setTimeout(() => {
      // The sections only: the lyrics are paired with them by Prepare's own check.
      void yue2CoverApi.reviewSections(sheetAbc, '').then(review => {
        if (live) setScoreSections(review.sections);
      }).catch(() => {});
    }, 250);
    return () => { live = false; clearTimeout(timer); };
  }, [yue2Mode, token, sheetAbc]);


  // ── Persist ──
  useEffect(() => { persist('sourceFileName', sourceFileName); }, [sourceFileName]);
  useEffect(() => { persist('sourceAudioUrl', sourceAudioUrl); }, [sourceAudioUrl]);
  useEffect(() => { persist('sourceAssetId', sourceAssetId); }, [sourceAssetId]);
  useEffect(() => { persist('coverDraftId', coverDraftId); }, [coverDraftId]);
  useEffect(() => { persist('coverDraftRevision', coverDraftRevision); }, [coverDraftRevision]);
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
  useEffect(() => { persist('sourceMode', sourceMode); }, [sourceMode]);
  useEffect(() => { persist('draftName', draftName); }, [draftName]);
  useEffect(() => { persist('selectedYue2ArtistId', selectedYue2ArtistId); }, [selectedYue2ArtistId]);
  useEffect(() => { persist('selectedYue2LyricsSetId', selectedYue2LyricsSetId); }, [selectedYue2LyricsSetId]);
  useEffect(() => { persist('coveringBand', coveringBand); }, [coveringBand]);
  useEffect(() => { persist('buildMode', buildMode); }, [buildMode]);
  useEffect(() => { persist('buildSessionId', buildSessionId); }, [buildSessionId]);
  useEffect(() => { persist('buildLyricsDraft', buildLyricsDraft); }, [buildLyricsDraft]);
  useEffect(() => { persist('advancedMode', advancedMode); }, [advancedMode]);
  useEffect(() => { persist('pairMode', renderMode); }, [renderMode]);
  useEffect(() => { persist('yue2Ar', yue2Ar); }, [yue2Ar]);
  useEffect(() => { persist('yue2Nar', yue2Nar); }, [yue2Nar]);
  useEffect(() => { persist('coverVoices', coverVoices); }, [coverVoices]);
  useEffect(() => { persist('keepChords', keepChords); }, [keepChords]);
  useEffect(() => { persist('coverTempoMode', coverTempoMode); }, [coverTempoMode]);
  useEffect(() => { persist('coverBpm', coverBpm); }, [coverBpm]);
  useEffect(() => { persist('coverKeyShift', coverKeyShift); }, [coverKeyShift]);
  useEffect(() => { persist('coverCfgScale', coverCfgScale); }, [coverCfgScale]);
  useEffect(() => { persist('coverGuide', coverGuide); }, [coverGuide]);
  useEffect(() => { persist('coverLyricSchedule', coverLyricSchedule); }, [coverLyricSchedule]);
  useEffect(() => { persist('captionMode', captionMode); }, [captionMode]);
  useEffect(() => { persist('sheetAbc', sheetAbc); }, [sheetAbc]);
  useEffect(() => { persist('scoreSections', scoreSections); }, [scoreSections]);
  useEffect(() => { persist('scoreTempoScale', scoreTempoScale); }, [scoreTempoScale]);
  useEffect(() => { persist('sheetAudioUrl', sheetAudioUrl); }, [sheetAudioUrl]);
  useEffect(() => { persist('approvedSheet', approvedSheet); }, [approvedSheet]);
  // The editing fields, mirrored into this browser's revisioned server draft.
  // coverDraftId/Revision stay local: they point at the workflow's cover-draft.
  const draft = useStudioDraftMirror('cover', token, { 'cover-studio-sourceFileName': sourceFileName, 'cover-studio-sourceAudioUrl': sourceAudioUrl, 'cover-studio-sourceAssetId': sourceAssetId, 'cover-studio-sourceSongId': sourceSongId, 'cover-studio-metadata': metadata, 'cover-studio-analysis': analysis, 'cover-studio-songArtist': songArtist, 'cover-studio-songTitle': songTitle, 'cover-studio-lyrics': lyrics, 'cover-studio-lyricsSource': lyricsSource, 'cover-studio-datasetAnalysis': datasetAnalysis, 'cover-studio-selectedArtistId': selectedArtistId, 'cover-studio-selectedPreset': selectedPreset, 'cover-studio-artistCaption': artistCaption, 'cover-studio-audioCoverStrength': audioCoverStrength, 'cover-studio-coverNoiseStrength': coverNoiseStrength, 'cover-studio-coverNoiseMethod': coverNoiseMethod, 'cover-studio-tempoScale': tempoScale, 'cover-studio-pitchShift': pitchShift, 'cover-studio-bpmCorrection': bpmCorrection, 'cover-studio-bpmOverride': bpmOverride, 'cover-studio-keyOverride': keyOverride, 'cover-studio-noFsq': noFsq, 'cover-studio-coverInstrumental': instrumental, 'cover-studio-sourceLatentUrl': sourceLatentUrl, 'cover-studio-coverVocalLanguage': vocalLanguage, 'cover-studio-coverTimbreOverride': timbreOverridePath, 'cover-studio-sepLevel': sepLevel,
    'cover-studio-sourceMode': sourceMode, 'cover-studio-advancedMode': advancedMode, 'cover-studio-pairMode': renderMode, 'cover-studio-yue2Ar': yue2Ar, 'cover-studio-yue2Nar': yue2Nar, 'cover-studio-coverVoices': coverVoices, 'cover-studio-keepChords': keepChords, 'cover-studio-coverTempoMode': coverTempoMode, 'cover-studio-coverBpm': coverBpm, 'cover-studio-coverKeyShift': coverKeyShift, 'cover-studio-coverCfgScale': coverCfgScale, 'cover-studio-coverGuide': coverGuide, 'cover-studio-coverLyricSchedule': coverLyricSchedule, 'cover-studio-captionMode': captionMode, 'cover-studio-sheetAbc': sheetAbc, 'cover-studio-scoreSections': scoreSections, ...prep.draftFields, 'cover-studio-scoreTempoScale': scoreTempoScale, 'cover-studio-sheetAudioUrl': sheetAudioUrl, 'cover-studio-approvedSheet': approvedSheet, 'cover-studio-selectedYue2ArtistId': selectedYue2ArtistId, 'cover-studio-selectedYue2LyricsSetId': selectedYue2LyricsSetId, 'cover-studio-coveringBand': coveringBand, 'cover-studio-buildMode': buildMode, 'cover-studio-buildSessionId': buildSessionId, 'cover-studio-buildLyricsDraft': buildLyricsDraft, ...(draftName ? { 'cover-studio-draftName': draftName } : {}) },
    { ...(sourceAssetId ? { sourceAssetId } : {}), ...(sourceSongId ? { sourceSongId } : {}) });
  // Loading a draft sets the same state the controls edit. Its uploaded source
  // gets a fresh cover-draft from the stored analysis (no re-analysis, no job
  // beyond opening the source); nothing is generated.
  const applyDraft = (body: { fields: Record<string, unknown> }) => {
    const f = body.fields;
    setDraftName(typeof f['cover-studio-draftName'] === 'string' ? f['cover-studio-draftName'] : '');
    const nextSourceKey = JSON.stringify([
      f['cover-studio-sourceMode'] ?? 'audio',
      f['cover-studio-sourceAudioUrl'] ?? sourceAudioUrl,
      f['cover-studio-sourceSongId'] ?? sourceSongId,
      f['cover-studio-sourceAssetId'] ?? sourceAssetId,
      (f['cover-studio-advancedMode'] ?? advancedMode) && sepJobId,
      (f['cover-studio-advancedMode'] ?? advancedMode) && sepStems?.length,
      (f['cover-studio-advancedMode'] ?? advancedMode) && stemControls,
    ]);
    skipSourceResetRef.current = nextSourceKey !== sourceKey;
    setSourceMode(Object.hasOwn(f, 'cover-studio-sourceMode') ? f['cover-studio-sourceMode'] as 'audio' | 'musicxml' : 'audio');
    sourceLookupRef.current++;
    captionRequestRef.current++;
    setCoverDraftId(''); setCoverDraftRevision(0);
    if (Object.hasOwn(f, 'cover-studio-sourceFileName')) setSourceFileName(f['cover-studio-sourceFileName'] as never);
    if (Object.hasOwn(f, 'cover-studio-sourceAudioUrl')) setSourceAudioUrl(f['cover-studio-sourceAudioUrl'] as never);
    if (Object.hasOwn(f, 'cover-studio-sourceAssetId')) setSourceAssetId(f['cover-studio-sourceAssetId'] as never);
    if (Object.hasOwn(f, 'cover-studio-sourceSongId')) setSourceSongId(f['cover-studio-sourceSongId'] as never);
    if (Object.hasOwn(f, 'cover-studio-metadata')) setMetadata(f['cover-studio-metadata'] as never);
    if (Object.hasOwn(f, 'cover-studio-analysis')) setAnalysis(f['cover-studio-analysis'] as never);
    if (Object.hasOwn(f, 'cover-studio-songArtist')) setSongArtist(f['cover-studio-songArtist'] as never);
    if (Object.hasOwn(f, 'cover-studio-songTitle')) setSongTitle(f['cover-studio-songTitle'] as never);
    if (Object.hasOwn(f, 'cover-studio-lyrics')) setLyrics(f['cover-studio-lyrics'] as never);
    if (Object.hasOwn(f, 'cover-studio-lyricsSource')) setLyricsSource(f['cover-studio-lyricsSource'] as never);
    if (Object.hasOwn(f, 'cover-studio-datasetAnalysis')) setDatasetAnalysis(f['cover-studio-datasetAnalysis'] as never);
    if (Object.hasOwn(f, 'cover-studio-selectedArtistId')) setSelectedArtistId(f['cover-studio-selectedArtistId'] as never);
    if (Object.hasOwn(f, 'cover-studio-selectedPreset')) setSelectedPreset(f['cover-studio-selectedPreset'] as never);
    if (Object.hasOwn(f, 'cover-studio-artistCaption')) setArtistCaption(f['cover-studio-artistCaption'] as never);
    if (Object.hasOwn(f, 'cover-studio-audioCoverStrength')) setAudioCoverStrength(f['cover-studio-audioCoverStrength'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverNoiseStrength')) setCoverNoiseStrength(f['cover-studio-coverNoiseStrength'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverNoiseMethod')) setCoverNoiseMethod(f['cover-studio-coverNoiseMethod'] as never);
    if (Object.hasOwn(f, 'cover-studio-tempoScale')) setTempoScale(f['cover-studio-tempoScale'] as never);
    if (Object.hasOwn(f, 'cover-studio-pitchShift')) setPitchShift(f['cover-studio-pitchShift'] as never);
    if (Object.hasOwn(f, 'cover-studio-bpmCorrection')) setBpmCorrection(f['cover-studio-bpmCorrection'] as never);
    if (Object.hasOwn(f, 'cover-studio-bpmOverride')) setBpmOverride(f['cover-studio-bpmOverride'] as never);
    if (Object.hasOwn(f, 'cover-studio-keyOverride')) setKeyOverride(f['cover-studio-keyOverride'] as never);
    if (Object.hasOwn(f, 'cover-studio-noFsq')) setNoFsq(f['cover-studio-noFsq'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverInstrumental')) setInstrumental(f['cover-studio-coverInstrumental'] as never);
    if (Object.hasOwn(f, 'cover-studio-sourceLatentUrl')) setSourceLatentUrl(f['cover-studio-sourceLatentUrl'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverVocalLanguage')) setVocalLanguage(f['cover-studio-coverVocalLanguage'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverTimbreOverride')) setTimbreOverridePath(f['cover-studio-coverTimbreOverride'] as never);
    if (Object.hasOwn(f, 'cover-studio-sepLevel')) setSepLevel(f['cover-studio-sepLevel'] as never);
    if (Object.hasOwn(f, 'cover-studio-advancedMode')) setAdvancedMode(f['cover-studio-advancedMode'] as never);
    if (Object.hasOwn(f, 'cover-studio-pairMode')) {
      const mode = f['cover-studio-pairMode'];
      modeChosen.current = true;
      setRenderMode(mode === 'pair' ? 'custom' : mode === 'artist' || mode === 'custom' ? mode : 'base');
    }
    if (Object.hasOwn(f, 'cover-studio-yue2Ar')) setYue2Ar(f['cover-studio-yue2Ar'] as never);
    if (Object.hasOwn(f, 'cover-studio-yue2Nar')) setYue2Nar(f['cover-studio-yue2Nar'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverVoices')) setCoverVoices(f['cover-studio-coverVoices'] as never);
    if (Object.hasOwn(f, 'cover-studio-keepChords')) setKeepChords(f['cover-studio-keepChords'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverTempoMode')) setCoverTempoMode(f['cover-studio-coverTempoMode'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverBpm')) setCoverBpm(f['cover-studio-coverBpm'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverKeyShift')) setCoverKeyShift(f['cover-studio-coverKeyShift'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverCfgScale')) setCoverCfgScale(f['cover-studio-coverCfgScale'] as never);
    if (Object.hasOwn(f, 'cover-studio-coverGuide')) setCoverGuide({ ...COVER_GUIDE_DEFAULT, ...(f['cover-studio-coverGuide'] as object) });
    if (Object.hasOwn(f, 'cover-studio-coverLyricSchedule')) setCoverLyricSchedule(f['cover-studio-coverLyricSchedule'] as never);
    if (Object.hasOwn(f, 'cover-studio-captionMode')) setCaptionMode(f['cover-studio-captionMode'] as never);
    if (Object.hasOwn(f, 'cover-studio-sheetAbc')) setSheetAbc(f['cover-studio-sheetAbc'] as never);
    if (Object.hasOwn(f, 'cover-studio-scoreSections')) setScoreSections(f['cover-studio-scoreSections'] as never);
    // The Prepare document and its cuts come back with the draft. A draft from
    // before version 2 has none and migrates from its saved score and lyrics;
    // a cut saved before merge, relabel and split has no removedMusic flag and
    // counts as removing music. Their source key may arrive a render later.
    const restoring = { ...f, 'cover-studio-sheetAbc': Object.hasOwn(f, 'cover-studio-sheetAbc') ? f['cover-studio-sheetAbc'] : sheetAbc,
      'cover-studio-lyrics': Object.hasOwn(f, 'cover-studio-lyrics') ? f['cover-studio-lyrics'] : lyrics };
    if (nextSourceKey === sourceKey) prep.restoreDraft(restoring, { sourceKey, abc: sheetAbc, lyrics });
    else pendingDraft.current = restoring;
    if (Object.hasOwn(f, 'cover-studio-scoreTempoScale')) setScoreTempoScale(f['cover-studio-scoreTempoScale'] as never);
    if (Object.hasOwn(f, 'cover-studio-sheetAudioUrl')) setSheetAudioUrl(f['cover-studio-sheetAudioUrl'] as never);
    if (Object.hasOwn(f, 'cover-studio-approvedSheet')) setApprovedSheet(f['cover-studio-approvedSheet'] as never);
    if (Object.hasOwn(f, 'cover-studio-selectedYue2ArtistId')) setSelectedYue2ArtistId(f['cover-studio-selectedYue2ArtistId'] as never);
    if (Object.hasOwn(f, 'cover-studio-selectedYue2LyricsSetId')) setSelectedYue2LyricsSetId(f['cover-studio-selectedYue2LyricsSetId'] as never);
    if (Object.hasOwn(f, 'cover-studio-coveringBand')) setCoveringBand(f['cover-studio-coveringBand'] as never);
    setBuildMode((f['cover-studio-buildMode'] ?? false) as never);
    setBuildSessionId((f['cover-studio-buildSessionId'] ?? '') as never);
    setBuildLyricsDraft((f['cover-studio-buildLyricsDraft'] ?? null) as never);
    const assetId = f['cover-studio-sourceAssetId'];
    if (typeof assetId === 'string' && assetId) {
      const meta = f['cover-studio-metadata'] as AudioMetadata | null, found = f['cover-studio-analysis'] as AudioAnalysis | null;
      void openUploadedAsset(assetId, sourceLookupRef.current, meta && found ? { metadata: meta, analysis: found } : undefined)
        // Transport failures already toast via the global handler.
        .catch(err => { if (!(err instanceof ApiError)) showToast(String((err as Error).message)); });
    }
  };

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
    if (useDetectedDetails) {
      setAnalysis({ bpm: data.bpm, key: data.key, scale: data.key.split(' ')[1] });
      setDatasetAnalysis(true);
    }
    return true;
  };

  const lookupDatasetMetadata = async (input: { songId?: string; sourceAudioUrl?: string }) => {
    if (!token) return null;
    try { return await yue2CoverApi.sourceMetadata(input); }
    catch { return null; }
  };

  // ── "Send to Cover Studio" — load a library track as the cover source (#61) ──
  useEffect(() => {
    if (!coverSource || coverSource.timestamp === _lastConsumedCoverTs) return;
    setSourceMode('audio');
    _lastConsumedCoverTs = coverSource.timestamp;
    const s = coverSource.song;
    captionRequestRef.current++;
    if (captionJobRef.current && token) void workflowApi.cancel(captionJobRef.current).catch(() => {});
    if (coverWorkflowJobRef.current && token) void workflowApi.cancel(coverWorkflowJobRef.current).catch(() => {});
    const gpData: any = s.generationParams || s.generation_params || {};
    const audioUrl = s.audioUrl || s.audio_url || '';
    setSourceAssetId(''); setCoverDraftId(''); setCoverDraftRevision(0);
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
    // Detection off: the new source starts at the 120 / C major default, never the old song's values.
    setAnalysis(useDetectedDetails ? null : { bpm: 120, key: 'C major' });
    setBpmCorrection(1); setKeyOverride(null); setBpmOverride(null);
    setTempoScale(1.0); setPitchShift(0);

    // BPM/key — (A) use the track's stored metadata, else (B) analyze the source.
    const storedBpm = s.bpm ?? gpData.bpm;
    const storedKey = s.key_scale || gpData.keyScale;
    void (async () => {
      const dataset = await lookupDatasetMetadata({ songId: String(s.id) });
      if (sourceLookupRef.current !== lookup || applyDatasetMetadata(dataset || { matched: false }, lyricEdit, instrumentalEdit)) return;
      if (!useDetectedDetails) return;
      if (storedBpm && storedKey) {
        setAnalysis({ bpm: Number(storedBpm), key: String(storedKey), scale: String(storedKey).split(' ')[1] });
      } else if (audioUrl) {
        setIsAnalyzing(true);
        try {
          // Silent: a failed analysis just leaves the 120 / C major defaults.
          const d = await apiClient.post<{ bpm?: number; key?: string; scale?: string }>('/analyze', { audioUrl }, { silent: true });
          if (sourceLookupRef.current === lookup) setAnalysis({ bpm: d.bpm || 120, key: `${d.key || 'C'} ${d.scale || 'major'}`, scale: d.scale });
        } catch { /* leave defaults; user can override */ }
        // Unconditional: a guarded reset stuck the spinner forever when the
        // source switched to one that needed no analysis (stored bpm/key).
        finally { setIsAnalyzing(false); }
      }
    })();
    showToast(t('cover.loadedFromLibrary', 'Loaded source from library'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [coverSource?.timestamp]);

  // ── Load artists on mount ──
  // Bulk snapshots avoid a per-artist and per-album request fan-out.
  const restoredPresetLoaded = useRef(false);
  useEffect(() => {
    if (restoredPresetLoaded.current || !setsQuery.data || !presetsQuery.data) return;
    restoredPresetLoaded.current = true;
    if (!yue2Mode && selectedArtistId && !selectedPreset) {
      const ids = new Set(setsQuery.data.filter(ls => ls.artist_id === selectedArtistId).map(ls => ls.id));
      const preset = presetsQuery.data.presets.find(p => ids.has(p.lyrics_set_id) && p.adapter_path);
      if (preset) setSelectedPreset(preset);
    }
  }, [yue2Mode, setsQuery.data, presetsQuery.data, selectedArtistId]);

  useEffect(() => {
    if (!yue2Mode || renderMode !== 'artist' || selectedYue2ArtistId == null || !setsQuery.data || !presetsQuery.data) return;
    const presets = new Map(presetsQuery.data.presets.map(p => [p.lyrics_set_id, p]));
    const available = setsQuery.data.filter(ls => ls.artist_id === selectedYue2ArtistId)
      .map(ls => ({ lsId: ls.id, album: ls.album || String(ls.id), preset: presets.get(ls.id) ?? null }))
      .filter(a => a.preset?.yue2_ar_adapter_path && a.preset?.yue2_nar_adapter_path);
    setYue2Albums(available);
    setYue2AlbumsForArtistId(selectedYue2ArtistId);
    const chosen = available.find(a => a.lsId === selectedYue2LyricsSetId) || available[0];
    if (chosen && chosen.lsId !== selectedYue2LyricsSetId) setSelectedYue2LyricsSetId(chosen.lsId);
  }, [yue2Mode, renderMode, selectedYue2ArtistId, setsQuery.data, presetsQuery.data, selectedYue2LyricsSetId]);

  const openUploadedAsset = async (assetId: string, lookup: number, cached?: {
    metadata: AudioMetadata; analysis: AudioAnalysis;
  }): Promise<CoverDraftResult | null> => {
    if (!token || !assetId) return null;
    const { job } = await coverWorkflowApi.open(assetId, cached);
    coverWorkflowJobRef.current = job.id;
    try {
      const done = await waitForCoverJob(job.id, () => sourceLookupRef.current !== lookup);
      if (sourceLookupRef.current !== lookup) return null;
      const result = done.result as CoverDraftResult;
      setCoverDraftId(result.documentId);
      setCoverDraftRevision(result.revision);
      return result;
    } finally {
      if (coverWorkflowJobRef.current === job.id) coverWorkflowJobRef.current = '';
    }
  };

  const handleFileSelected = async (file: File) => {
    setSourceMode('audio');
    if (!token) { showToast(t('cover.signInFirst')); return; }
    const lookup = ++sourceLookupRef.current;
    if (coverWorkflowJobRef.current) void workflowApi.cancel(coverWorkflowJobRef.current).catch(() => {});
    captionRequestRef.current++;
    if (captionJobRef.current) void workflowApi.cancel(captionJobRef.current).catch(() => {});
    setSourceAssetId(''); setCoverDraftId(''); setCoverDraftRevision(0);
    const lyricEdit = lyricEditRef.current, instrumentalEdit = instrumentalEditRef.current;
    // The last source's lyrics never carry over: the sidecar, cache or a search fills them.
    setLyrics(''); setLyricsSource(null); setDatasetAnalysis(false); setSheetScoreSource(null);
    setSourceAudioUrl(''); setSourceSongId(''); clearStems();
    setSourceFileName(file.name);
    setAnalysis(useDetectedDetails ? null : { bpm: 120, key: 'C major' });
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
      if (useDetectedDetails && Number.isFinite(cached.bpm) && cached.key) setAnalysis({ bpm: cached.bpm, key: cached.key, scale: cached.scale });
      // Still upload the file
      try {
        const d = await uploadMut.mutateAsync({ path: '/upload/audio', files: [{ field: 'audio', file }] });
        if (sourceLookupRef.current !== lookup) return;
        setSourceAudioUrl(d.audio_url || '');
        setSourceAssetId(d.asset_id || '');
        setIsAnalyzing(true);
        const opened = await openUploadedAsset(d.asset_id || '', lookup,
          Number.isFinite(cached.bpm) && cached.key ? {
          metadata: { artist: cached.artist || '', title: cached.title || '',
            album: cached.album || '', duration: cached.duration ?? null },
          analysis: { bpm: cached.bpm, key: cached.key, scale: cached.scale },
        } : undefined);
        if (useDetectedDetails && opened?.analysis) setAnalysis(opened.analysis);
        const dataset = await lookupDatasetMetadata({ sourceAudioUrl: d.audio_url });
        if (sourceLookupRef.current === lookup && dataset) applyDatasetMetadata(dataset, lyricEdit, instrumentalEdit);
      } catch (err) {
        // Transport failures already toast via the global handler; surface only
        // domain errors (a failed cover-draft job throws a plain Error).
        if (!(err instanceof ApiError)) showToast(err instanceof Error ? err.message : String(err));
      } finally { if (sourceLookupRef.current === lookup) { setIsAnalyzing(false); } }
      return;
    }

    // Full pipeline
    let extractedArtist = '', extractedTitle = '', extractedAlbum = '';
    let extractedDuration: number | null = null;
    try {
      // Upload once; the Node workflow reads metadata and analyses that asset.
      const upData = await uploadMut.mutateAsync({ path: '/upload/audio', files: [{ field: 'audio', file }] });
      if (sourceLookupRef.current !== lookup) return;
      const audioUrl = upData.audio_url || '';
      setSourceAudioUrl(audioUrl);
      setSourceAssetId(upData.asset_id || '');
      setIsAnalyzing(true);

      const opened = await openUploadedAsset(upData.asset_id || '', lookup);
      if (sourceLookupRef.current !== lookup) return;
      if (opened?.metadata) {
        setMetadata(opened.metadata);
        extractedArtist = opened.metadata.artist;
        extractedTitle = opened.metadata.title;
        extractedAlbum = opened.metadata.album;
        extractedDuration = opened.metadata.duration;
        if (extractedArtist) setSongArtist(extractedArtist);
        if (extractedTitle) setSongTitle(extractedTitle);
      }

      const dataset = await lookupDatasetMetadata({ sourceAudioUrl: audioUrl });
      if (sourceLookupRef.current !== lookup) return;
      if (dataset && applyDatasetMetadata(dataset, lyricEdit, instrumentalEdit)) return;

      // 3. Essentia analysis
      let bpm = 120, key = 'C major', scale: string | undefined;
      let analysed = false;
      if (opened?.analysis) {
        const d = opened.analysis;
        bpm = d.bpm; key = d.key; scale = d.scale;
        analysed = true;
        if (useDetectedDetails && sourceLookupRef.current === lookup) setAnalysis({ bpm, key, scale });
      }
      // 4. Cache. Only a real analysis goes in: with Essentia unavailable the
      // 120 / C major fallback was cached and came back labelled "detected" on
      // every later load of the same file (#131).
      saveTrackCacheEntry(file.name, {
        artist: extractedArtist, title: extractedTitle, album: extractedAlbum, duration: extractedDuration,
        ...(analysed ? { bpm, key, scale } : {}),
      });
    } catch (err) {
      // Transport failures already toast via the global handler; surface only
      // domain errors (a failed cover-draft job throws a plain Error).
      if (!(err instanceof ApiError)) showToast(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally { if (sourceLookupRef.current === lookup) { setIsAnalyzing(false); } }
  };

  // ── Lyrics search ──
  const searchLyricsMut = useMutation({
    mutationFn: ({ artist, title }: { artist: string; title: string }) => lireekApi.searchSongLyrics(artist, title),
    onSuccess: (result, vars) => {
      setLyrics(result.lyrics);
      setLyricsSource(null);
      if (result.title) setSongTitle(result.title);
      showToast(t('cover.lyricsFound'));
      if (sourceFileName) saveTrackCacheEntry(sourceFileName, { lyrics: result.lyrics, artist: vars.artist, title: result.title || vars.title });
    },
    onError: (err: any) => {
      // Transport failures already toast via the global handler; surface only
      // domain errors (no lyrics found, etc.).
      if (!(err instanceof ApiError)) showToast(err.message || t('cover.noLyricsFound'));
    },
  });
  const isSearchingLyrics = searchLyricsMut.isPending;
  const handleSearchLyrics = () => {
    if (!songArtist.trim() || !songTitle.trim()) { showToast(t('cover.enterArtistTitle')); return; }
    lyricEditRef.current++;
    searchLyricsMut.mutate({ artist: songArtist.trim(), title: songTitle.trim() });
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

  const resolveCaptionOnNode = async (artistId: number, force: boolean): Promise<string | null> => {
    if (!token || !coverDraftId || !coverDraftRevision) return null;
    const request = ++captionRequestRef.current;
    const key = sourceKey;
    const { provider, model } = loadSelections().coverCaption;
    const { job } = await coverWorkflowApi.caption(coverDraftId, coverDraftRevision,
      artistId, provider, model || '', force);
    captionJobRef.current = job.id;
    try {
      const done = await waitForCoverJob(job.id, () =>
        captionRequestRef.current !== request || sourceKeyRef.current !== key);
      if (captionRequestRef.current !== request || sourceKeyRef.current !== key) return null;
      const result = done.result as CoverDraftResult;
      setCoverDraftRevision(result.revision);
      return result.caption || '';
    } finally {
      if (captionJobRef.current === job.id) captionJobRef.current = '';
    }
  };

  // ── Artist preset loading ──
  const presetRequestRef = useRef(0);
  useEffect(() => () => { presetRequestRef.current++; }, []);
  const loadArtistPresets = async (artist: Artist) => {
    const request = ++presetRequestRef.current;
    const source = sourceKeyRef.current;
    const current = () => request === presetRequestRef.current && sourceKeyRef.current === source;
    try {
      const [{ lyrics_sets: allSets }, { presets }] = await Promise.all([lireekApi.listLyricsSets(), lireekApi.listAllPresets()]);
      const lyrics_sets = allSets.filter(ls => ls.artist_id === artist.id);
      const bySet = new Map(presets.map(p => [p.lyrics_set_id, p]));
      const results = lyrics_sets.map(ls => ({ lsId: ls.id, album: ls.album || String(ls.id), preset: bySet.get(ls.id) ?? null }));
      if (!current()) return;
      // Find caption — waterfall: generations → profiles → LLM
      if (token && sourceAssetId && coverDraftId && coverDraftRevision && !advancedMode) {
        setBgCaptionPending(true);
        const pending = resolveCaptionOnNode(artist.id, false);
        const request = captionRequestRef.current;
        void pending
          .then(caption => { if (caption !== null) setArtistCaption(caption); })
          .catch(err => console.warn('[CoverStudio] Caption generation failed:', err))
          .finally(() => { if (captionRequestRef.current === request) setBgCaptionPending(false); });
      } else {
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
      if (!current()) return;
      setArtistCaption(caption);
      // Step 3: On-demand LLM generation (async, non-blocking)
      if (!caption) {
        const sel = loadSelections();
        const { provider, model } = sel.coverCaption;
        if (provider) {
          setBgCaptionPending(true);
          lireekApi.generateCaption(artist.id, { provider, model: model || undefined })
            .then(res => { if (current() && res.caption) setArtistCaption(res.caption); })
            .catch(err => console.warn('[CoverStudio] Caption generation failed:', err))
            .finally(() => setBgCaptionPending(false));
        }
      }
      }
      if (!current()) return;
      // Pick adapter preset — use first album with adapter (don't override cover settings)
      const withAdapter = results.find(p => p.preset?.adapter_path);
      if (withAdapter?.preset) {
        setSelectedPreset(withAdapter.preset);
        applyPresetToGlobal(withAdapter.preset);
      } else { setSelectedPreset(results[0]?.preset || null); }
    } catch { if (current()) { setSelectedPreset(null); setArtistCaption(''); } }
  };

  const handleSelectArtist = async (artist: Artist) => {
    captionRequestRef.current++;
    if (captionJobRef.current && token) void workflowApi.cancel(captionJobRef.current).catch(() => {});
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
  const captionMut = useMutation({
    mutationFn: async ({ artistId, provider, model }: { artistId: number; provider: string; model: string }) => {
      if (token && sourceAssetId && coverDraftId && coverDraftRevision && !advancedMode) {
        return (await resolveCaptionOnNode(artistId, true)) || '';
      }
      const res = await lireekApi.generateCaption(artistId, { provider, model: model || undefined, force: true });
      return res.caption;
    },
    onSuccess: caption => { if (caption) setArtistCaption(caption); },
    onError: (err: any) => {
      // Transport failures already toast via the global handler; surface only
      // domain errors (a failed caption job throws a plain Error).
      if (!(err instanceof ApiError)) showToast(`Caption generation failed: ${err.message}`);
    },
  });
  const isGeneratingCaption = bgCaptionPending || captionMut.isPending;
  const handleRegenerateCaption = () => {
    if (!selectedArtistId) return;
    const sel = loadSelections();
    const { provider, model } = sel.coverCaption;
    if (!provider) { showToast('Select a Caption LLM provider first'); return; }
    captionMut.mutate({ artistId: selectedArtistId, provider, model });
  };

  const prepareYue2Source = async (): Promise<{ sourceAudioUrl?: string; songId?: string; audioUrl: string }> => {
    if (advancedMode && sepStems?.length && sepJobId) {
      const controls = stemControls.map(c => ({ index: c.index, volume: c.muted ? 0 : c.volume, muted: c.muted }));
      const blob = await recombineStems(sepJobId, controls);
      const uploaded = await uploadMut.mutateAsync({ path: '/upload/audio', files: [{ field: 'audio', file: blob, name: 'recombined-stems.wav' }] });
      if (!uploaded.audio_url) throw new Error('Recombined source has no audio URL');
      return { sourceAudioUrl: uploaded.audio_url, audioUrl: uploaded.audio_url };
    }
    if (sourceSongId) return { songId: sourceSongId, audioUrl: sourceAudioUrl };
    return { sourceAudioUrl, audioUrl: sourceAudioUrl };
  };

  const handleTranscribe = async () => {
    if (!token || !sourceAudioUrl) return;
    if (sourceAssetId && (!coverDraftId || !coverDraftRevision)) {
      setSheetError('Source is still being prepared'); return;
    }
    const key = sourceKey;
    const request = ++sheetRequestRef.current;
    let workflowId = '';
    setSheetError(''); setSheetPreparing(true);
    try {
      if (sourceAssetId && coverDraftId && coverDraftRevision && !advancedMode) {
        const { job } = await coverWorkflowApi.transcribe(coverDraftId, coverDraftRevision, !!sheetAbc);
        workflowId = job.id;
        coverWorkflowJobRef.current = job.id;
        const done = await waitForCoverJob(job.id, () =>
          sourceKeyRef.current !== key || sheetRequestRef.current !== request);
        if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
        const result = done.result as CoverDraftResult;
        setCoverDraftRevision(result.revision);
        acceptTranscript(result.abc || '', []);
        setSheetScoreSource(result.scoreSource === 'dataset' ? 'dataset' : null);
        setSheetAudioUrl(sourceAudioUrl);
        return;
      }
      const source = await prepareYue2Source();
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
      const result = await yue2CoverApi.start({ ...source, sourceLabel: sourceFileName.slice(0, 120), force: !!sheetAbc,
        ...sheetTranscriberChoice() });
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) {
        if (result.jobId) await yue2CoverApi.cancel(result.jobId);
        return;
      }
      setSheetAudioUrl(source.audioUrl);
      if (result.jobId) { sheetJobRef.current = result.jobId; setSheetJobId(result.jobId); }
      else if (result.abc) {
        acceptTranscript(result.abc, result.sections || []);
        setSheetScoreSource(result.scoreSource === 'dataset' ? 'dataset' : null);
      }
    } catch (err: any) { if (sourceKeyRef.current === key && sheetRequestRef.current === request) setSheetError(err.message); }
    finally {
      if (workflowId && coverWorkflowJobRef.current === workflowId) coverWorkflowJobRef.current = '';
      if (sheetRequestRef.current === request) setSheetPreparing(false);
    }
  };

  // A newly loaded recording is transcribed by itself, once: there is nothing
  // else to do with it first. It waits for the upload, the analysis and the
  // dataset lookup (which may bring a score of its own), and never runs for
  // the source a page load restores.
  const autoTranscribedRef = useRef(sourceKey);
  useEffect(() => {
    if (autoTranscribedRef.current === sourceKey) return;
    if (!yue2Mode || !token || sourceMode !== 'audio' || !sourceAudioUrl || isUploading || isAnalyzing) return;
    if (sheetAbc.trim() || sheetPreparing || sheetJobId || !readiness?.ready) return;
    if (sourceAssetId && (!coverDraftId || !coverDraftRevision)) return;
    autoTranscribedRef.current = sourceKey;
    void handleTranscribe();
  }); // Every render: it acts once per source, when that source is ready.

  const handleCancelSheet = async () => {
    sheetRequestRef.current += 1;
    if (coverWorkflowJobRef.current && token) {
      const id = coverWorkflowJobRef.current;
      coverWorkflowJobRef.current = '';
      await workflowApi.cancel(id).catch(() => {});
    }
    const id = sheetJobRef.current;
    sheetJobRef.current = ''; setSheetJobId(''); setSheetPreparing(false);
    setSheetJob(null); setSheetError('Transcription cancelled');
    if (id && token) try { await yue2CoverApi.cancel(id); } catch (err: any) { setSheetError(err.message); }
  };

  const handleApproveSheet = async () => {
    if (!token || !sheetAbc.trim() || (sourceMode === 'audio' && !sourceAudioUrl)) return;
    if (sourceMode === 'musicxml') {
      setSheetError('');
      setApprovedSheet({ abc: sheetAbc.trim(), sourceId: '', sourceLabel: '', audioUrl: '', key: sourceKey });
      return;
    }
    if (sourceAssetId && (!coverDraftId || !coverDraftRevision)) {
      setSheetError('Source is still being prepared'); return;
    }
    const key = sourceKey;
    const request = ++sheetRequestRef.current;
    setSheetError(''); setSheetPreparing(true);
    try {
      if (sourceAssetId && coverDraftId && coverDraftRevision && !advancedMode) {
        const result = await coverWorkflowApi.approveScore(coverDraftId, coverDraftRevision, sheetAbc);
        if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
        setCoverDraftRevision(result.document.revision);
        setApprovedSheet({ abc: result.abc || sheetAbc.trim(), sourceId: result.sourceId,
          sourceLabel: result.sourceLabel, audioUrl: sourceAudioUrl, key });
        setSheetAudioUrl(sourceAudioUrl);
        return;
      }
      const source = sheetAudioUrl
        ? { audioUrl: sheetAudioUrl, ...(sheetAudioUrl === sourceAudioUrl && sourceSongId ? { songId: sourceSongId } : { sourceAudioUrl: sheetAudioUrl }) }
        : await prepareYue2Source();
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
      const result = await yue2CoverApi.start({ ...source, sourceLabel: sourceFileName.slice(0, 120), abc: sheetAbc });
      if (sourceKeyRef.current !== key || sheetRequestRef.current !== request) return;
      setApprovedSheet({ abc: result.abc || sheetAbc.trim(), sourceId: result.sourceId,
        sourceLabel: result.sourceLabel, audioUrl: source.audioUrl, key });
      setScoreSections(result.sections || []);
      setSheetAudioUrl(source.audioUrl);
    } catch (err: any) { if (sourceKeyRef.current === key && sheetRequestRef.current === request) setSheetError(err.message); }
    finally { if (sheetRequestRef.current === request) setSheetPreparing(false); }
  };

  // The score change, as the Prepare document's edits make it: the rest of the
  // studio follows (the score needs approving again, its sections are re-read).
  const changeSheetAbc = (value: string) => {
    sheetRequestRef.current += 1; setSheetPreparing(false); setSheetAbc(value);
    setScoreSections([]); setSheetScoreSource(null); setApprovedSheet(null);
  };

  // 2/4 to 4/4: the server rewrites the score for a preview. Applying it is one
  // more edit of the Prepare document (undoable there), so the structure has
  // to be confirmed again.
  const [remeter, setRemeter] = useState<{ from: string; pickup: boolean; abc?: string; twoFourBars?: number; error?: string } | null>(null);
  const remeterOpen = remeter?.from === sheetAbc ? remeter : null;
  // No silent filter here: remeterScore is silent, so this popover's error line
  // is the sole notifier and must show every failure.
  const remeterMut = useMutation({
    mutationFn: ({ from, pickup }: { from: string; pickup: boolean }) => yue2CoverApi.remeterScore(from, pickup),
    onSuccess: (data, vars) => setRemeter(current =>
      current?.from === vars.from && current.pickup === vars.pickup ? { ...current, ...data } : current),
    onError: (err, vars) => setRemeter(current =>
      current?.from === vars.from && current.pickup === vars.pickup
        ? { ...current, error: err instanceof Error ? err.message : String(err) } : current),
  });
  const previewRemeter = (pickup: boolean) => {
    if (!token) return;
    setRemeter({ from: sheetAbc, pickup });
    remeterMut.mutate({ from: sheetAbc, pickup });
  };
  const applyRemeter = () => {
    if (remeterOpen?.abc && prep.applyStructuralOp({ kind: 'replace', abc: remeterOpen.abc })) setRemeter(null);
  };
  // Bar lines a beat or more off the song's downbeats: the same preview, then one edit.
  const [barShift, setBarShift] = useState<{ from: string; beats: number; abc?: string; error?: string } | null>(null);
  const barShiftOpen = barShift?.from === sheetAbc ? barShift : null;
  const previewBarShift = async (beats: number) => {
    if (!token) return;
    const from = sheetAbc;
    const settle = (result: object) => setBarShift(current =>
      current?.from === from && current.beats === beats ? { ...current, ...result } : current);
    setBarShift({ from, beats });
    try { settle(await yue2CoverApi.shiftBars(from, beats)); }
    catch (err) { settle({ error: (err as Error).message }); }
  };
  const applyBarShift = () => {
    if (barShiftOpen?.abc && prep.applyStructuralOp({ kind: 'replace', abc: barShiftOpen.abc })) setBarShift(null);
  };

  // The original source, never a recombined mix: its vocal stem is what the
  // aligner reads, and its path is what finds the dataset song.
  const sectionSource = () => sourceSongId ? { songId: sourceSongId } : { sourceAudioUrl };
  /** Lyric assignment proposals for the current document; the recording only when asked for. */
  const suggestAssignments = (useRecording: boolean) => token
    ? yue2CoverApi.suggestAssignments({ ...(useRecording ? sectionSource() : {}), prepare: prep.doc, useRecording })
    : Promise.reject(new Error('Sign in first.'));

  // The score's tempo and key, offered for a dataset song whose .txt differs.
  const scoreBpm = Math.round(Number(sheetAbc.match(/^Q:[^=\r\n]*=\s*(\d+(?:\.\d+)?)/m)?.[1]) || 0);
  const scoreTempoFactor = scoreTempoScale.abc === sheetAbc ? scoreTempoScale.factor : 1;
  const scaledScoreBpm = Math.round(scoreBpm * scoreTempoFactor);
  const scaleScoreTempo = (by: number) => setScoreTempoScale({ abc: sheetAbc, factor: scoreTempoFactor * by });
  const scoreKey = coverScoreKeyLabel(sheetAbc).replace(/(major|minor|dorian|phrygian|lydian|mixolydian|locrian)$/, m => m[0].toUpperCase() + m.slice(1));
  const scoreDetailsDiffer = yue2Mode && datasetAnalysis && !!analysis && scoreBpm > 0 && !!scoreKey &&
    (Math.round(analysis.bpm) !== scoreBpm || analysis.key.toLowerCase() !== scoreKey.toLowerCase());
  // Both legs are silent — the sheet's "Save to dataset:" line is the sole
  // notifier, so onError shows every failure. The source key travels as a
  // variable: if the source changed while the save was in flight, the write
  // still succeeded but its bpm/key belong to the old source, so only the
  // toast applies — not the analysis state.
  const scoreDetailsMut = useMutation({
    mutationFn: async (key: string) => {
      let bpm = scaledScoreBpm, musicalKey = scoreKey;
      if (sourceAssetId && coverDraftId && coverDraftRevision && !advancedMode) {
        const saved = await coverWorkflowApi.saveScoreDetails(coverDraftId,
          coverDraftRevision, sheetAbc, scoreTempoFactor);
        bpm = saved.bpm; musicalKey = saved.key;
      } else {
        await yue2CoverApi.saveDatasetDetails({ ...sectionSource(), bpm, key: musicalKey });
      }
      return { bpm, key: musicalKey, stale: sourceKeyRef.current !== key };
    },
    onSuccess: ({ bpm, key, stale }) => {
      if (!stale) {
        setAnalysis({ bpm, key, scale: key.split(' ')[1] });
        setBpmOverride(null); setKeyOverride(null);
      }
      showToast('Tempo and key saved to the dataset');
    },
    onError: (err: any) => setSheetError(`Save to dataset: ${err.message}`),
  });
  const scoreDetailsSaving = scoreDetailsMut.isPending;
  const saveScoreDetails = () => {
    if (!token || !scoreDetailsDiffer) return;
    scoreDetailsMut.mutate(sourceKey);
  };

  // ── Generation ──
  // A YuE2 render sends the Prepare document's confirmed pair: the score and the
  // lyrics tagged in its order. Instrumental mode comes from the document too
  // (every section marked Instrumental); the server refuses a mismatch.
  const preparedResult = yue2Mode ? exportPrepare(prep.doc) : null;
  const confirmedPair = preparedResult && !('error' in preparedResult) ? preparedResult : null;
  const prepareBlock = preparedResult && 'error' in preparedResult ? preparedResult.error : '';
  const renderInstrumental = yue2Mode ? !!confirmedPair?.instrumental : instrumental;
  const renderLyrics = yue2Mode ? confirmedPair?.lyrics ?? '' : lyrics;
  // The dataset sidecar's lyrics are marked as such only while the pair still sends them as they are.
  const renderLyricsSource = !yue2Mode || (confirmedPair && confirmedPair.lyrics === normalizeAbc(lyrics)) ? lyricsSource : null;
  const resolvedCaption = artistCaption.trim();
  const selectCaptionSource = (mode: string) => {
    setCaptionMode(mode);
    if (mode !== 'custom') {
      const selection = mode === 'auto' ? { mode: 'auto' as const }
        : { mode: 'track' as const, selectedName: mode.slice(6) };
      setArtistCaption(resolveYue2Caption(artistCaption, analysis?.bpm, captionTracks, selection).caption);
    }
  };
  const selectYue2Album = (lyricsSetId: number) => {
    const album = availableYue2Albums.find(a => a.lsId === lyricsSetId);
    if (!album?.preset) return;
    setSelectedYue2LyricsSetId(lyricsSetId);

  };
  const availableYue2Albums = yue2AlbumsForArtistId === selectedYue2ArtistId ? yue2Albums : [];
  const selectedYue2Album = availableYue2Albums.find(a => a.lsId === selectedYue2LyricsSetId) || availableYue2Albums[0];
  const renderSelection = resolveYue2Render(renderMode,
    { ar: yue2Ar, nar: yue2Nar, artist: coveringBand },
    { ar: selectedYue2Album?.preset?.yue2_ar_adapter_path, nar: selectedYue2Album?.preset?.yue2_nar_adapter_path,
      artist: yue2Artists.find(a => a.id === selectedYue2ArtistId)?.name });
  const activeYue2Ar = renderSelection.lmAdapterAr;
  const activeYue2Nar = renderSelection.lmAdapterNar;
  const targetYue2Artist = renderSelection.artist;
  const adapterOptions = (kind: 'ar' | 'nar') => [
    { value: '', label: `Base ${kind.toUpperCase()}` },
    ...(yue2Models?.lmAdapters || []).filter(path =>
      (yue2Models?.lmAdapterMeta?.[path] as { kind?: string } | undefined)?.kind === kind,
    ).map(path => ({ value: path, label: yue2Models?.lmAdapterMeta?.[path]?.label || path.split(/[\\/]/).pop() || path })),
    ...((kind === 'ar' ? yue2Ar : yue2Nar) && !(yue2Models?.lmAdapters || []).includes(kind === 'ar' ? yue2Ar : yue2Nar)
      ? [{ value: kind === 'ar' ? yue2Ar : yue2Nar, label: (kind === 'ar' ? yue2Ar : yue2Nar).split(/[\\/]/).pop() || 'Album adapter' }]
      : []),
  ];

  const submitCoverWorkflow = () => {
    if (!token || !coverDraftId || !coverDraftRevision) return;
    const engineParams = gp.getGlobalParams() as Record<string, unknown>;
    const backend = useBackendStore.getState().activeBackendId;
    const input = {
      documentId: coverDraftId, revision: coverDraftRevision,
      expectedBackend: backend, engineParams,
      title: songTitle, artistName: songArtist,
      targetArtistName: yue2Mode ? targetYue2Artist : artists.find(a => a.id === selectedArtistId)?.name || '',
      lyrics: renderLyrics, caption: artistCaption,
      instrumental: renderInstrumental, lyricsSource: renderLyricsSource, scoreSource: sheetScoreSource, analysis,
      ...(yue2Mode ? { prepare: prep.doc } : {}),
      settings: { coResident: settings.coResident, cacheLmCodes: settings.cacheLmCodes,
        parallelWhisper: settings.parallelWhisper, parallelQualityEval: settings.parallelQualityEval,
        parallelCoverArt: settings.parallelCoverArt },
      controls: {
        bpmOverride, bpmCorrection, keyOverride, tempoScale, pitchShift, noFsq,
        audioCoverStrength, coverNoiseStrength, coverNoiseMethod, vocalLanguage,
        sourceLatentUrl, timbreOverridePath,
        presetAdapterPath: selectedPreset?.adapter_path || '',
        presetReferencePath: selectedPreset?.reference_track_path || '',
        triggerUseFilename: settings.triggerUseFilename,
        triggerPlacement: settings.triggerPlacement || 'prepend',
        voices: coverVoices, keepChords, tempoMode: coverTempoMode, coverBpm,
        keyShift: coverKeyShift, cfgScale: coverCfgScale, lyricSchedule: coverLyricSchedule,
        lmAdapterAr: activeYue2Ar, lmAdapterNar: activeYue2Nar,
        pairMode: renderMode === 'base' ? 'base' : 'pair',
        yue2Pick: { ...yue2PickAtEnqueue(null), lmAdapterArScale: clampCoverAr(coverArStrength) },
        captionMode: yue2Mode ? 'custom' : captionMode, captionTracks: renderMode === 'base' ? [] : captionTracks,
      },
    };
    const qId = addManualQueueItem({ title: songArtist
      ? `${songTitle || 'Cover'} (${songArtist} Cover)` : (songTitle || 'Cover'),
      artistName: yue2Mode ? targetYue2Artist : artists.find(a => a.id === selectedArtistId)?.name || '', caption: artistCaption });
    updateManualQueueItem(qId, { stage: 'Preparing…' });
    setIsGenerating(true);
    void (async () => {
    let workflowId = '';
    try {
      const { job } = await coverWorkflowApi.render(input);
      workflowId = job.id;
      coverRenderJobRef.current = job.id;
      setActiveJobId(job.id); setQueueItemId(qId);
      updateManualQueueItem(qId, { workflowJob: { id: job.id, token }, stage: 'Queued…', status: 'pending' });
      // Name the workflow's state only when it changes: the queue store follows
      // the render itself and fills in the engine's stage and progress.
      let shown = '';
      const done = await waitForCoverJob(job.id, () => false, current => {
        const running = current.status === 'running';
        const stage = running ? 'Generating…' : 'Queued…';
        if (stage !== shown) updateManualQueueItem(qId, { stage: shown = stage, status: running ? 'generating' : 'pending' });
      });
      const result = done.result as { audio?: { audioUrls?: string[]; songIds?: string[];
        masteredAudioUrl?: string; noAdapterAudioUrl?: string; duration?: number };
        elapsedSeconds?: number | null };
      const audio = result.audio || {};
      completeManualQueueItem(qId, { audioUrl: audio.audioUrls?.[0] || '', songId: audio.songIds?.[0],
        masteredAudioUrl: audio.masteredAudioUrl, noAdapterAudioUrl: audio.noAdapterAudioUrl,
        audioDuration: audio.duration, elapsedSeconds: result.elapsedSeconds });
      showToast(t('cover.coverGenerated'));
    } catch (err: any) {
      failManualQueueItem(qId, err.message || 'Generation failed');
    } finally {
      if (coverRenderJobRef.current === workflowId) {
        setIsGenerating(false); setActiveJobId(null); setQueueItemId(null);
      }
      if (workflowId && coverRenderJobRef.current === workflowId) coverRenderJobRef.current = '';
    }
    })();
  };

  // The approved score is part of the source: a MusicXML import has no other identity.
  const buildLyricsSource = JSON.stringify([sourceMode, sourceAudioUrl, sourceSongId, sourceAssetId, approvedSheet?.abc ?? '']);
  // A build starts from the confirmed pair, so a lyrics draft written before its first session has nowhere to go
  // (the server refuses lyrics that differ from the confirmed ones); the session's own lyrics are edited after it starts.
  const buildLyrics = renderInstrumental || yue2Mode ? null : draftText(buildLyricsDraft, buildLyricsSource);
  const build = useYue2Build(token, buildSessionId, setBuildSessionId, { text: buildLyrics,
    set: text => setBuildLyricsDraft(text === null || text === lyrics ? null : { source: buildLyricsSource, text }) });
  const buildSession = build.session && build.session.id === buildSessionId ? build.session : null;
  const nextBuild = buildNext(buildSession);
  /** Build-page lyric edits go to the session; before one exists the build follows the confirmed pair. */
  const applyBuildLyrics = async (text: string) => {
    if (!buildSession && !build.loading) { showToast('Start the build first, then edit its lyrics. Until then it uses the lyrics confirmed on the Prepare page.'); return false; }
    return build.applyLyrics(text);
  };
  /** The params a build starts from: the session's own, or the Prepare page's before there is one. */
  const buildParams = (): Record<string, unknown> | null => {
    if (buildSession) return buildSession.params;
    const built = yue2CoverParams();
    if (typeof built === 'string') { showToast(built); return null; }
    return built.params;
  };
  const generateBuildTakes = (to: number, count: number, caption: string, lyrics: Record<string, string>) => {
    const params = buildParams();
    return params ? build.generate(params, to, count, caption, lyrics).then(Boolean) : Promise.resolve(false);
  };
  const setPrepareSettings = (s: BuildSettings) => {
    setCoverVoices(s.voices);
    if (typeof s.tempo === 'number') { setCoverTempoMode('set'); setCoverBpm(s.tempo); } else setCoverTempoMode(s.tempo);
    setCoverKeyShift(s.keyShift); setCoverCfgScale(s.cfgScale); setCoverArStrength(clampCoverAr(s.arStrength));
  };
  // The Build page's settings row: the Prepare page's settings until a build
  // exists, the build's until its first keep, then a fresh build. The Prepare
  // page follows every applied change, so it and a fresh build agree.
  const applyBuildSettings = async (s: BuildSettings) => {
    if (buildSession && !buildSession.kept.length) {
      const choices = coverChoicesFor(String(buildSession.params.yue2Abc ?? ''), s);
      if (!await build.settings({ ...choices, arStrength: clampCoverAr(s.arStrength) })) return;
    }
    setPrepareSettings(s);
    if (buildSession?.kept.length) build.startOver();
  };
  const planBuild = (upcoming: string[], finished: boolean) => {
    const params = buildParams();
    if (params) void build.plan(params, upcoming, finished);
  };
  // Save song: an ordinary render of the kept codec stream, so it gets the
  // usual post-processing and library save.
  const saveBuild = () => {
    if (!token || !buildSession?.kept.length) return;
    const sessionId = buildSession.id;
    const params = { ...buildSession.params, yue2Build: { session: buildSession.id },
      yue2BatchSize: 1, yue2Variations: 1, batchSize: 1, randomSeed: true };
    const title = String(buildSession.params.title || 'Cover');
    const qId = addManualQueueItem({ title, artistName: String(buildSession.params.artistName || ''), caption: String(buildSession.params.caption || '') });
    updateManualQueueItem(qId, { stage: _coverRunning ? 'Queued…' : 'Preparing…' });
    setIsGenerating(true);
    _savingBuilds.set(sessionId, (_savingBuilds.get(sessionId) ?? 0) + 1);
    enqueueCoverJob(async () => {
      try {
        const res = await generateApi.submit(params as any);
        updateManualQueueItem(qId, { jobId: res.jobId });
        const outcome = await pollJobAsync(res.jobId, qId);
        // Only a saved song starts the next build, and only if this build is
        // still the open one; any other outcome keeps it for a retry.
        if (outcome === 'succeeded') finishSavedBuild(sessionId);
      } catch (err: any) { failManualQueueItem(qId, err.message || 'Generation failed'); }
      finally {
        const left = (_savingBuilds.get(sessionId) ?? 1) - 1;
        if (left > 0) _savingBuilds.set(sessionId, left); else _savingBuilds.delete(sessionId);
        if (_coverQueue.length === 0) { setIsGenerating(false); setActiveJobId(null); setGenProgress(0); setGenStage(''); }
      }
    });
    showToast('Saving the build as a song: it is in the queue');
  };
  // The Build page's view of what is being built: the session's own settings
  // once it exists, the Prepare page's until then.
  const renderBuildPage = () => {
    const draftParams = buildSession ? null : yue2CoverParams();
    const params = buildSession?.params ?? (typeof draftParams === 'string' || !draftParams ? null : draftParams.params);
    const abc = String(params?.yue2Abc ?? sheetAbc);
    const sections = buildSession?.sections ?? scoreSections.map(x => ({ label: x.label, startBar: x.startBar }));
    return <Yue2BuildPage session={buildSession} sections={sections}
      totalBars={barRanges(abc).reduce((n, r) => Math.max(n, r.bar), 0)}
      lyrics={buildSession ? String(buildSession.params.lyrics ?? '') : buildLyrics ?? renderLyrics}
      prepareLyrics={renderLyrics} onApplyLyrics={applyBuildLyrics} loading={build.loading}
      settings={buildSession ? buildSettingsOf(buildSession.params) : { voices: coverVoices,
        tempo: coverTempoMode === 'set' ? coverBpm : coverTempoMode, keyShift: coverKeyShift, cfgScale: coverCfgScale, arStrength: coverArStrength }}
      onSettings={s => void applyBuildSettings(s)} keyLabel={coverScoreKeyLabel(abc)}
      sourceBpm={Number(abc.match(/^Q:[^=\r\n]*=\s*(\d+(?:\.\d+)?)/m)?.[1] ?? 120)}
      title={[params?.artist, params?.title ?? (songTitle || 'Cover')].filter(Boolean).join(' - ')}
      scoreAbc={abc} recordingUrl={sourceMode === 'audio' && prep.recordingSeekAvailable ? sourceAudioUrl : ''}
      blocked={buildSession ? '' : typeof draftParams === 'string' ? `${draftParams} on the Prepare page.` : !canGenerate ? 'Finish the Prepare page first.' : ''}
      busy={build.busy} error={build.error} caption={buildCaption(params ?? undefined)}
      onOccurrenceLyrics={edits => void build.occurrenceLyrics(edits)}
      onGenerate={generateBuildTakes} onPlan={planBuild} onCancel={() => void build.cancel()} onKeep={(take, crop, count) => void build.keep(take, crop, count)} onDelete={take => void build.remove(take)}
      onUndo={section => void build.undo(section)} onStartOver={build.startOver} onDeleteBuild={() => buildSession && _savingBuilds.has(buildSession.id) ? showToast('A Save song render is queued for this build; delete it once that finishes.') : void build.deleteBuild()} onSave={saveBuild}
      onCrop={(take, frame, mode) => void build.crop(take, frame, mode)} onTrim={(take, seconds) => void build.trim(take, seconds)} onPreview={() => void build.preview()}
      onBack={() => setBuildMode(false)} />;
  };

  /** The /api/generate params for a YuE2 cover of the approved score, as the
   *  one-shot render and Build by section both submit them, or why not. */
  const yue2CoverParams = (): { params: Record<string, unknown>; title: string } | string => {
    if (!confirmedPair) return prepareBlock || 'Confirm the score structure and lyrics first';
    if (!approvedSheet || approvedSheet.key !== sourceKey || structureHash(approvedSheet.abc) !== confirmedPair.structureHash)
      return 'Approve the confirmed score first';
    const missing = yue2CoverContentMissing({ caption: resolvedCaption, lyrics: renderLyrics, instrumental: renderInstrumental });
    if (missing) return missing;
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
    lmAdapterArScale: clampCoverAr(coverArStrength),
    lmAdapterAr: activeYue2Ar,
    lmAdapterNar: activeYue2Nar };
  const title = songArtist ? `${songTitle || 'Cover'} (${songArtist} Cover)` : (songTitle || 'Cover');
  const coverChoices = { keepChords, ...coverChoicesFor(approvedSheet.abc, { voices: coverVoices,
    tempo: coverTempoMode === 'set' ? coverBpm : coverTempoMode, keyShift: coverKeyShift, cfgScale: coverCfgScale }) };
  const params = { ...sharedParams, ...yue2Params, yue2CfgScale: coverCfgScale,
    yue2LyricSchedule: coverLyricSchedule, customMode: true, taskType: 'text2music',
    title, artist: targetYue2Artist, artistName: targetYue2Artist || songArtist, sourceArtist: songArtist, caption: artistCaption, style: artistCaption, lyrics: renderInstrumental ? '' : renderLyrics,
    ...(renderLyricsSource ? { lyricsSource: renderLyricsSource } : {}),
    ...(sheetScoreSource ? { scoreSource: 'dataset-sidecar' } : {}),
    instrumental: renderInstrumental, source: 'cover-studio',
    ...(sourceMode === 'audio' ? { sourceAudioUrl: approvedSheet.audioUrl } : {}),
    yue2Cover: sourceMode === 'audio'
      ? { sourceId: approvedSheet.sourceId, sourceLabel: approvedSheet.sourceLabel, ...coverChoices,
          ...(coverGuide.on && !coverGuideBlock(coverKeyShift, coverTempoMode)
            ? { guide: (({ on: _on, ...rest }) => rest)(coverGuide) } : {}) }
      : { sourceKind: 'musicxml', ...coverChoices },
    yue2Abc: approvedSheet.abc, yue2Prepare: prep.doc, yue2Cot: keepChords ? 'full' : 'melody', yue2Pick: pair };
    return { params, title };
  };

  const handleGenerate = () => {
    if (!token || (sourceMode === 'audio' && !sourceAudioUrl)) { showToast(t('cover.missingSrcOrLyrics')); return; }
    if (sourceMode === 'audio' && sourceAssetId && (!coverDraftId || !coverDraftRevision)) {
      showToast('Source is still being prepared'); return;
    }
    if (yue2Mode && !confirmedPair) { showToast(prepareBlock || 'Confirm the score structure and lyrics first'); return; }
    if (!yue2Mode && !instrumental && !lyrics.trim()) { showToast('Enter lyrics or enable Instrumental mode'); return; }
    if (yue2Mode && renderMode !== 'base' && (!activeYue2Ar || !activeYue2Nar)) {
      showToast('Choose both YuE2 adapter halves'); return;
    }
    if (sourceMode === 'audio' && sourceAssetId && coverDraftId && coverDraftRevision && !advancedMode) {
      if (yue2Mode && (!approvedSheet || approvedSheet.key !== sourceKey)) {
        showToast('Approve the confirmed score first'); return;
      }
      submitCoverWorkflow();
      return;
    }
    if (yue2Mode) {
      const built = yue2CoverParams();
      if (typeof built === 'string') { showToast(built); return; }
      const { params, title } = built;
      const qId = addManualQueueItem({ title, artistName: targetYue2Artist, caption: artistCaption });
      updateManualQueueItem(qId, { stage: _coverRunning ? 'Queued…' : 'Preparing…' });
      setIsGenerating(true);
      enqueueCoverJob(async () => {
        try {
          const res = await generateApi.submit(params as any);
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
          // Upload recombined WAV to get a server-side URL; on failure keep the
          // original. Silent — that fallback is deliberate, not an error worth
          // a global toast.
          try {
            const { audio_url } = await uploadMut.mutateAsync({ path: '/upload/audio', files: [{ field: 'audio', file: blob, name: 'recombined-stems.wav' }], silent: true });
            if (audio_url) {
              effectiveSourceUrl = audio_url;
              console.log('[CoverStudio] Using recombined stems:', audio_url);
            }
          } catch { /* upload failed — fall back to the original source */ }
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

      const res = await generateApi.submit(params as any);
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
  const pollJobAsync = async (jobId: string, qId: string): Promise<'succeeded' | 'failed' | 'cancelled' | 'timeout'> => {
    setActiveJobId(jobId); setQueueItemId(qId);
    setGenProgress(0); setGenStage('Queued...');
    const timer = createGenerationTimer();
    let timedOut = false;
    let outcome: 'succeeded' | 'failed' | 'cancelled' | 'timeout' = 'failed';
    try {
      await waitForQuery<GenerationJob>(generationStatusOptions(jobId),
        s => !timedOut && !['succeeded', 'failed', 'cancelled'].includes(s.status), s => {
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
            timedOut = true; outcome = 'timeout';
            showToast(`Generation timed out after ${getGenerationTimeoutMinutes()} minutes`);
            failManualQueueItem(qId, 'Generation timed out');
            return;
          }

          if (s.status === 'succeeded') {
            outcome = 'succeeded';
            setGenProgress(100); setGenStage('Complete!');
            showToast(t('cover.coverGenerated'));

            // Complete queue item with audio data
            completeManualQueueItem(qId, {
              audioUrl: s.result?.audioUrls?.[0] || '',
              songId: s.result?.songIds?.[0],
              masteredAudioUrl: s.result?.masteredAudioUrl,
              noAdapterAudioUrl: s.result?.noAdapterAudioUrl,
              audioDuration: s.result?.duration,
            });
          } else if (s.status === 'failed' || s.status === 'cancelled') {
            outcome = s.status;
            showToast(`Failed: ${s.error || 'Unknown error'}`);
            failManualQueueItem(qId, s.error || (s.status === 'cancelled' ? 'Cancelled' : 'Unknown error'));
          }
        }, { timeoutMs: (getGenerationTimeoutMinutes() + 30) * 60_000, deliverRepeats: true });
      await songsChanged();
    } catch (err) {
      failManualQueueItem(qId, err instanceof Error ? err.message : 'Generation status failed');
      return 'failed';
    }
    return outcome;
  };

  // Cancel the currently-running cover. The poll sees the cancelled status,
  // resolves, and the queue advances to the next item.
  const handleCancel = async () => {
    let cancelled = false;
    if (coverRenderJobRef.current && token) {
      try { await workflowApi.cancel(coverRenderJobRef.current); cancelled = true; } catch {}
    } else if (activeJobId) { try { await generateApi.cancel(activeJobId); cancelled = true; } catch {} }
    // Only claim cancellation when the server confirmed it — a failed cancel
    // leaves the job running, and the queue item must stay in flight.
    if (queueItemId && cancelled) failManualQueueItem(queueItemId, 'Cancelled by user');
  };

  const handleClearSource = () => {
    sourceLookupRef.current++;
    captionRequestRef.current++;
    if (captionJobRef.current && token) void workflowApi.cancel(captionJobRef.current).catch(() => {});
    if (coverWorkflowJobRef.current && token) void workflowApi.cancel(coverWorkflowJobRef.current).catch(() => {});
    setSourceAssetId(''); setCoverDraftId(''); setCoverDraftRevision(0);
    setSourceFileName(''); setSourceAudioUrl(''); setSourceSongId('');
    setMetadata(null); setAnalysis(null);
    setSongArtist(''); setSongTitle(''); setLyrics('');
    setLyricsSource(null); setDatasetAnalysis(false); setSheetScoreSource(null);
    setBpmCorrection(1); setBpmOverride(null); setKeyOverride(null);
    // Clear stems too — releases the split job server-side.
    clearStems();
  };

  const selectSourceMode = (mode: 'audio' | 'musicxml') => {
    if (mode === sourceMode) return;
    handleClearSource();
    sheetRequestRef.current++;
    setApprovedSheet(null); setSheetAbc(''); setScoreSections([]);
    setSheetAudioUrl(''); setSheetError('');
    if (sheetJobRef.current && token) void yue2CoverApi.cancel(sheetJobRef.current).catch(() => {});
    sheetJobRef.current = ''; setSheetJobId(''); setSheetJob(null); setSheetPreparing(false);
    setSourceMode(mode);
  };

  const handleClearArtist = () => {
    presetRequestRef.current++;
    setSelectedArtistId(null);
    setSelectedPreset(null);

    setArtistCaption('');
  };

  // Always allow queuing another cover — covers stack and run one at a time (#62).
  const canGenerate = (!yue2Mode || sourceMode === 'audio' ? !!sourceAudioUrl && (!sourceAssetId || (!!coverDraftId && !!coverDraftRevision)) : true)
    && (yue2Mode ? !!confirmedPair : (!!lyrics.trim() || instrumental))
    && (!yue2Mode || (!!approvedSheet && approvedSheet.key === sourceKey && (renderMode === 'base' || (!!activeYue2Ar && !!activeYue2Nar))));

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



  // Drafts sit under Generate in whichever settings column is showing.
  const draftsFold = (
    <Fold title="Drafts" storageKey="cover-studio-fold-drafts">
      <StudioDraftPicker control={draft} textKey="cover-studio-songTitle" apply={applyDraft} namedSave={yue2Mode} inline />
    </Fold>
  );

  // ── Render ──
  return (
    <BackendCapabilityGate feature="cover">
    <div className="flex flex-col w-full h-full bg-zinc-50 dark:bg-suno overflow-hidden">
      {/* Toast */}
      {toast && (
        <div className="absolute top-16 right-6 z-50 max-w-[500px] px-4 py-2 rounded-xl bg-white dark:bg-zinc-900 text-white text-sm shadow-xl border border-zinc-300 dark:border-white/10 animate-in fade-in slide-in-from-top-2 whitespace-normal break-words">
          {toast}
        </div>
      )}

      {yue2Mode && <div className="flex items-center gap-1 border-b border-zinc-200 dark:border-white/10 px-4 pt-2">
        {([[false, 'Prepare'], [true, 'Build']] as const).map(([on, label]) =>
          <button key={label} type="button" aria-pressed={buildMode === on} onClick={() => setBuildMode(on)}
            className={`-mb-px border-b-2 px-3 py-1.5 text-xs font-semibold ${buildMode === on
              ? 'border-pink-500 text-pink-700 dark:text-pink-300' : 'border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'}`}>{label}</button>)}
      </div>}
      {yue2Mode && buildMode && renderBuildPage()}

      {/* Main workspace */}
      {/* Scrolls sideways when the three panels' minimum widths don't fit.
          Hidden, not unmounted, on the Build page, so the lead sheet keeps its state. */}
      <div className={`flex-1 flex overflow-x-auto overflow-y-hidden ${yue2Mode && buildMode ? 'hidden' : ''}`}>
        {/* Left: source choice */}
        <div className="min-w-[240px] flex-shrink-0 flex flex-col min-h-0" style={{ width: sourceWidth }}>
        {yue2Mode && <div className="border-r border-zinc-200 dark:border-white/5 p-4 space-y-3 overflow-y-auto">
          <div className="flex gap-2" role="group" aria-label={t('cover.sourceMode.label')}>
            <button type="button" onClick={() => selectSourceMode('audio')} aria-pressed={sourceMode === 'audio'}
              className={`rounded-lg px-3 py-1.5 text-xs ${sourceMode === 'audio' ? 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-300' : 'text-zinc-600 dark:text-zinc-400'}`}>
              {t('cover.sourceMode.audio')}</button>
            <button type="button" onClick={() => selectSourceMode('musicxml')} aria-pressed={sourceMode === 'musicxml'}
              className={`rounded-lg px-3 py-1.5 text-xs ${sourceMode === 'musicxml' ? 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-300' : 'text-zinc-600 dark:text-zinc-400'}`}>
              {t('cover.sourceMode.musicxml')}</button>
          </div>
          {sourceMode === 'musicxml' && <MusicXmlImport key={sourceKey} abc={sheetAbc} disabled={sheetPreparing}
            onAbcChange={(value, imported) => { sheetRequestRef.current++; setSheetAbc(value); setApprovedSheet(null);
              setScoreSections([]); setSheetScoreSource(null);
              if (!songTitle.trim()) setSongTitle(imported?.title ?? '');
              if (imported?.lyrics && !lyrics.trim()) setLyrics(imported.lyrics);
              // A score with lyrics is a vocal cover: Instrumental left on from an earlier
              // score would drop the lyrics and move the melody to Ins (gate render c2efb9a5).
              if (imported?.lyrics.trim()) setInstrumental(false);
              // A score with chord symbols renders with them (Chain of Thought full); one without, melody only.
              if (imported) setKeepChords(!!imported.chords); }}>
            <div className="rounded-lg border border-dashed border-zinc-300 dark:border-white/20 p-6 text-center text-xs text-zinc-500">
              {t('cover.sourceMode.dropScore')}
            </div>
          </MusicXmlImport>}
        </div>}
        {(!yue2Mode || sourceMode === 'audio') && <>
        <div className="px-4 pt-4 space-y-2">
          <Toggle size="sm" accent="cyan" checked={useDetectedDetails}
            disabled={isUploading || isAnalyzing}
            onChange={enabled => {
              if (!enabled && !analysis) setAnalysis({ bpm: bpmOverride ?? 120, key: keyOverride || 'C major' });
              setUseDetectedDetails(enabled);
            }}
            label={t('cover.useDetectedDetails', 'Use detected BPM/key')}
            info={t('cover.useDetectedDetailsInfo', 'On: loading audio uses detected, cached or dataset BPM and key. Off: ignore those results; loading new audio resets BPM and key to the defaults. Source analysis may still run. This does not edit the ABC score. Default manual values are 120 BPM and C major.')} />
          <ParamLabel label={t('cover.sourceKeyBehavior', 'Source key behavior')}
            info={yue2Mode
              ? t('cover.sourceKeyBehaviorYue2', 'The source Key field is metadata. It does not transpose the ABC notes or pitch-shift the recording. YuE2 composes from the reviewed ABC score; use Pitch Shift in the cover settings to transpose that score for generation.')
              : t('cover.sourceKeyBehaviorAce', 'The source Key field corrects the key sent to generation as conditioning. Changing it does not apply a DSP pitch shift to the recording. Use Pitch Shift in the cover settings to change the audio pitch.')}
            className="text-xs text-zinc-600 dark:text-zinc-400" />
          {yue2Mode && analysis && <div className="flex items-center gap-2">
            <ParamLabel label={t('cover.tempoFix')} info={t('cover.sourceTempoFixInfo', 'Halve or double the current source BPM, including a manual override. This corrects source metadata; it does not change the ABC score tempo.')} className="text-xs text-zinc-600 dark:text-zinc-400" />
            {([0.5, 2] as const).map(factor => <button key={factor} type="button"
              onClick={() => setBpmOverride(Math.max(20, Math.min(300, Math.round((bpmOverride ?? analysis.bpm) * factor))))}
              className="rounded border border-zinc-300 dark:border-white/10 px-2 py-1 text-xs text-zinc-700 dark:text-zinc-300">
              {factor === 0.5 ? '÷2' : '×2'}
            </button>)}
          </div>}
        </div>
        <SourcePanel
          yue2Mode={yue2Mode}
          sourceFileName={sourceFileName} metadata={metadata} analysis={analysis}
          fromDataset={datasetAnalysis}
          scoreDetails={scoreDetailsDiffer ? { bpm: scaledScoreBpm, key: scoreKey, saving: scoreDetailsSaving,
            onScale: scaleScoreTempo, onSave: () => setScoreDetailsConfirm(true) } : null}
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
        /></>}
        </div>

        <CoverResizeHandle label="Resize source column" direction="horizontal" onMouseDown={e => startResize(e, sourceWidth, setSourceWidth, 240, 900)} />
        {/* Center: Lyrics and score */}
        <div className="min-w-[320px] flex-shrink-0 flex flex-col overflow-hidden border-r border-zinc-200 dark:border-white/5" style={{ width: scoreWidth }}>
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
          {!yue2Mode && <div className="flex-1 min-h-0 p-4 flex flex-col gap-1">
            {lyricsSource && <span className="text-[10px] text-cyan-600 dark:text-cyan-300">Lyrics from dataset · editable</span>}
            <LyricsSectionHighlight value={lyrics} onChange={value => { lyricEditRef.current++; setLyrics(value); setLyricsSource(null); }}
              placeholder={t('cover.lyricsPlaceholder')} highlight={null} />
          </div>}
          {yue2Mode && <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
            {lyricsSource && <span className="text-[10px] text-cyan-600 dark:text-cyan-300">Lyrics from dataset · editable</span>}
            <div className="flex flex-wrap items-center gap-1" role="group" aria-label={t('cover.prepare.steps')}>
              {([['structure', t('cover.prepare.step.structure'), true],
                ['lyrics', t('cover.prepare.step.lyrics'), !prep.editingStructure && !!prep.doc.confirmed.structure && structureHash(prep.doc.abc) === prep.doc.confirmed.structure.structureHash],
                ['ready', t('cover.prepare.step.ready'), !!confirmedPair]] as const).map(([step, label, enabled]) =>
                <button key={step} type="button" aria-pressed={prep.step === step} disabled={!enabled} onClick={() => prep.showStep(step)}
                  className={`rounded-md px-3 py-1 text-xs font-semibold disabled:opacity-40 ${prep.step === step
                    ? 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-300' : 'text-zinc-600 dark:text-zinc-400'}`}>{label}</button>)}
              {prep.doc.confirmed.structure && !prep.editingStructure && !(approvedSheet && approvedSheet.key === sourceKey) &&
                <button type="button" onClick={() => void handleApproveSheet()} disabled={sheetPreparing || !sheetAbc.trim()}
                  title={t('cover.prepare.approveInfo')} className="ml-2 text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">{t('cover.prepare.approve')}</button>}
            </div>
            {replaced && replaced.sourceKey === sourceKey && <p role="status" className="flex flex-wrap items-center gap-2 text-xs text-amber-700 dark:text-amber-300">
              {t('cover.prepare.replaced')}
              <button type="button" className="text-cyan-700 dark:text-cyan-300" onClick={() => {
                prep.restoreDraft({ [PREPARE_DRAFT_KEY]: replaced.doc, 'cover-studio-scoreCuts': replaced.cuts, 'cover-studio-sheetAbc': replaced.doc.abc,
                  'cover-studio-lyrics': lyrics }, { sourceKey, abc: replaced.doc.abc, lyrics });
                setReplaced(null);
              }}>{t('cover.prepare.restore')}</button>
              <button type="button" className="text-zinc-500" onClick={() => setReplaced(null)}>{t('cover.prepare.dismiss')}</button>
            </p>}
            {prep.step === 'structure'
              ? <PrepareStructure prep={prep} sourceAudioUrl={sourceMode === 'audio' ? sourceAudioUrl : ''} scorePlaying={scorePlaying}
                onPlayScore={playScore} scoreStarts={scoreStarts} onHighlight={setScoreHighlight}
                planFromLyrics={token && lyrics.trim() && sourceMode === 'audio' && sourceAudioUrl
                  ? () => yue2CoverApi.planScoreSections({ ...sectionSource(), abc: prep.doc.abc, lyrics }) : undefined}
                onConfirmed={() => void handleApproveSheet()}>
                {mainMeter(sheetAbc) === '2/4' && !remeterOpen &&
                  <button type="button" onClick={() => void previewRemeter(false)} disabled={sheetPreparing}
                    title="Rewrite the score in 4/4 by pairing its 2/4 bars. Every note keeps its pitch and timing. Shows a preview before applying."
                    className="rounded-md border border-cyan-500/50 px-2 py-1 text-xs text-cyan-700 dark:text-cyan-300 hover:bg-cyan-500/10 disabled:opacity-40">Convert 2/4 to 4/4</button>}
                {remeterOpen && <div className="space-y-2 rounded-lg border border-cyan-500/30 p-2">
                  <Toggle size="sm" accent="cyan" checked={remeterOpen.pickup} label="First bar is a pickup"
                    info="Keep bar 1 as a 2/4 bar and pair the bars from bar 2. Turn it on when the preview puts the downbeats in the middle of the 4/4 bars."
                    onChange={pickup => void previewRemeter(pickup)} />
                  {remeterOpen.error ? <p role="alert" className="text-xs text-red-500">This score cannot be converted: {remeterOpen.error}</p>
                    : !remeterOpen.abc ? <p className="text-xs text-zinc-500">Converting…</p>
                    : <>
                      <p className="text-xs text-zinc-500">
                        {remeterOpen.twoFourBars
                          ? `${remeterOpen.twoFourBars} bar${remeterOpen.twoFourBars > 1 ? 's stay' : ' stays'} in 2/4: a pickup, a bar before a new section or meter, or the last bar. `
                          : ''}
                        Applying replaces this cover's score (the dataset files are not changed) and can be undone above; the structure then needs confirming again.</p>
                      <AbcChart abc={remeterOpen.abc} disabled height={240} playingBar={null} onAbcChange={() => {}} />
                    </>}
                  <div className="flex gap-3">
                    <button type="button" onClick={applyRemeter} disabled={!remeterOpen.abc}
                      className="text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">Apply 4/4</button>
                    <button type="button" onClick={() => setRemeter(null)} className="text-xs text-zinc-500">Cancel</button>
                  </div>
                </div>}
                {sheetAbc.trim() && !barShiftOpen && <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="text-zinc-500">Bar lines off?</span>
                  {([[-1, 'A beat earlier', 'after'], [1, 'A beat later', 'before']] as const).map(([beats, label, when]) =>
                    <button key={beats} type="button" onClick={() => void previewBarShift(beats)} disabled={sheetPreparing}
                      title={`Move every bar line ${label.toLowerCase()}, keeping each note where it sounds. Use it when the bar-1 click lands a beat ${when} the song's downbeat. Shows a preview before applying.`}
                      className="rounded-md border border-cyan-500/50 px-2 py-1 text-cyan-700 dark:text-cyan-300 hover:bg-cyan-500/10 disabled:opacity-40">{label}</button>)}
                </div>}
                {barShiftOpen && <div className="space-y-2 rounded-lg border border-cyan-500/30 p-2">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-600 dark:text-zinc-300">
                    <button type="button" onClick={() => void previewBarShift(barShiftOpen.beats === 1 ? -1 : barShiftOpen.beats - 1)}
                      className="rounded-md border border-zinc-300 dark:border-white/10 px-2">−</button>
                    <span>Bar lines {Math.abs(barShiftOpen.beats)} beat{Math.abs(barShiftOpen.beats) > 1 ? 's' : ''} {barShiftOpen.beats < 0 ? 'earlier' : 'later'}</span>
                    <button type="button" onClick={() => void previewBarShift(barShiftOpen.beats === -1 ? 1 : barShiftOpen.beats + 1)}
                      className="rounded-md border border-zinc-300 dark:border-white/10 px-2">+</button>
                  </div>
                  {barShiftOpen.error ? <p role="alert" className="text-xs text-red-500">The bar lines cannot be moved: {barShiftOpen.error}</p>
                    : !barShiftOpen.abc ? <p className="text-xs text-zinc-500">Moving…</p>
                    : <>
                      <p className="text-xs text-zinc-500">Every note keeps its time; notes across a new bar line are tied and section starts move with their bar lines.
                        Applying replaces this cover's score (the dataset files are not changed) and can be undone above; the structure then needs confirming again.</p>
                      <AbcChart abc={barShiftOpen.abc} disabled height={240} playingBar={null} onAbcChange={() => {}} />
                    </>}
                  <div className="flex gap-3">
                    <button type="button" onClick={applyBarShift} disabled={!barShiftOpen.abc}
                      className="text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">Apply</button>
                    <button type="button" onClick={() => setBarShift(null)} className="text-xs text-zinc-500">Cancel</button>
                  </div>
                </div>}
              </PrepareStructure>
              : <PrepareLyrics prep={prep} lyrics={lyrics} wide={scoreWidth > 560}
                onLyricsText={value => { lyricEditRef.current++; setLyrics(value); setLyricsSource(null); }}
                suggest={suggestAssignments} scorePlaying={scorePlaying} activeSection={activeSection}
                onPlayScore={playScore} />}
          </div>}
          {/* In-page, not window.confirm: a native dialog stalls browser automation. */}
          <ConfirmDialog isOpen={scoreDetailsConfirm} title="Save tempo and key to dataset"
            message={`Replace the dataset song’s tempo and key with the score’s (${scaledScoreBpm} BPM, ${scoreKey})? The current .txt is kept as a backup.`}
            confirmLabel="Save" onCancel={() => setScoreDetailsConfirm(false)}
            onConfirm={() => { setScoreDetailsConfirm(false); void saveScoreDetails(); }} />
          {yue2Mode && <CoverResizeHandle label="Resize lyrics and lead sheet" direction="vertical" onMouseDown={startScoreResize} />}
          {yue2Mode && <Yue2CoverScore
            height={scoreHeight}
            wide={scoreWidth > 400}
            sourceReady={(sourceMode === 'musicxml' ? !!sheetAbc.trim() : !!sourceAudioUrl) && !isUploading}
            sourceMode={sourceMode}
            readiness={readiness} job={sheetJob} preparing={sheetPreparing} error={sheetError}
            abc={prep.doc.abc} sourceKey={sourceKey} approved={!!approvedSheet && approvedSheet.key === sourceKey}
            onAbcChange={value => { prep.applyStructuralOp({ kind: 'replace', abc: value }); }}
            readOnly={!(prep.editingStructure && prep.step === 'structure')}
            onMoveSection={(index, bar) => { prep.applyStructuralOp({ kind: 'move', index, bar }); }}
            highlightBars={prep.step === 'structure' ? scoreHighlight : null}
            play={scorePlay} onPlaying={onScorePlaying}
            scoreSource={sheetScoreSource} onSection={index => setActiveSectionOf({ abc: sheetAbc, index })}
            onTranscribe={handleTranscribe} onCancel={handleCancelSheet}
          />}

          {showMixer && sepStems && sepJobId && (
            <StemMixer jobId={sepJobId} stems={sepStems}
              controls={stemControls} onControlsChange={setStemControls}
              onClose={() => setShowMixer(false)} />
          )}
        </div>

        <CoverResizeHandle label="Resize lyrics and score column" direction="horizontal" onMouseDown={e => startResize(e, scoreWidth, setScoreWidth, 320, 1200)} />
        {/* Right: Artist + Settings */}
        {yue2Mode ? <div className="min-w-[300px] flex-1 flex flex-col min-h-0" style={{ flexBasis: settingsWidth }}>
          {renderMode === 'base' && (pickerAr || pickerNar) && <p className="px-4 pt-3 text-xs text-amber-600 dark:text-amber-400">
            {t('cover.yue2BaseIgnoresLora')}</p>}
          <Yue2CoverPanel
          caption={artistCaption} onCaptionChange={value => { setArtistCaption(value); setCaptionMode('custom'); }}
          captionMode={captionMode} onCaptionMode={selectCaptionSource}
          captionTracks={captionTracks}
          artists={yue2Artists} loadingArtists={isLoadingArtists || isLoadingYue2Artists} selectedArtistId={selectedYue2ArtistId}
          onSelectArtist={artist => { setYue2AlbumsForArtistId(null); setYue2Albums([]); setSelectedYue2ArtistId(artist.id); setSelectedYue2LyricsSetId(null); }}
          albums={availableYue2Albums} selectedLyricsSetId={selectedYue2LyricsSetId} onSelectAlbum={selectYue2Album}
          coveringBand={coveringBand} onCoveringBand={setCoveringBand}
          onOpenBuild={() => setBuildMode(true)}
          buildStatus={buildSession ? `Build in progress: ${Math.min(nextBuild, buildSession.sections.length)} of ${buildSession.sections.length} sections` : ''}
          onStartOverBuild={buildSession ? build.startOver : undefined}
          instrumental={renderInstrumental} onInstrumentalChange={() => showToast('Instrumental follows the lyrics: mark every section Instrumental in Assign lyrics, or give a section words.')}
          renderMode={renderMode} onRenderMode={chooseRenderMode}
          ar={yue2Ar} nar={yue2Nar} onAr={setYue2Ar} onNar={setYue2Nar}
          arOptions={adapterOptions('ar')} narOptions={adapterOptions('nar')}
          keepChords={keepChords} onKeepChords={setKeepChords}
          scoreAbc={sheetAbc} scoreKeyLabel={coverScoreKeyLabel(sheetAbc)}
          voices={coverVoices} onVoices={setCoverVoices}
          tempoMode={coverTempoMode} onTempoMode={setCoverTempoMode}
          bpm={coverBpm} onBpm={setCoverBpm}
          keyShift={coverKeyShift} onKeyShift={setCoverKeyShift}
          cfgScale={coverCfgScale} onCfgScale={setCoverCfgScale}
          arStrength={coverArStrength} onArStrength={v => setCoverArStrength(clampCoverAr(v))}
          lyricSchedule={coverLyricSchedule} onLyricSchedule={setCoverLyricSchedule}
          guide={coverGuide} onGuide={setCoverGuide} hasSourceAudio={sourceMode === 'audio'}
          canGenerate={canGenerate} isGenerating={isGenerating} onGenerate={handleGenerate}
          footer={draftsFold}
        />
        </div> : <ArtistSettingsPanel
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
          onGenerate={handleGenerate} onCancel={handleCancel} footer={draftsFold}
          isGeneratingCaption={isGeneratingCaption}
          onRegenerateCaption={handleRegenerateCaption}
        />}
        {yue2Mode && <CoverResizeHandle label="Resize generation settings column" direction="horizontal"
          onMouseDown={e => startResize(e, settingsWidth, setSettingsWidth, 300, 900)} />}

      </div>
    </div>
    </BackendCapabilityGate>
  );
};
