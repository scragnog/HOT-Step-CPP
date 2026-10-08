// CreatePanel.tsx — The composition panel (Content + Metadata only)
//
// Global engine parameters (Models, Adapters, Generation Settings, LM, Mastering)
// have been moved to the GlobalParamBar. This panel now only handles
// per-song content and metadata.

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Zap, ListPlus, Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useGlobalParams, useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { useAuth } from '../../context/AuthContext';
import { generateApi } from '../../services/api';
import { Yue2ScorePreviewModal, type Yue2ScorePreviewData } from './Yue2ScorePreviewModal';
import { ContentSection } from './ContentSection';
import { MetadataSection } from './MetadataSection';
import { LatentImport } from '../shared/LatentImport';
import { CoverArtSubjectSection } from '../shared/CoverArtSubjectSection';
import { AiGenerateModal, type AiGenerateResult } from './AiGenerateModal';
import { Mm3ComposeButton } from './Mm3ComposeButton';
import { TrainingDraftBar } from './TrainingDraftBar';
import { createContentParams } from './createContent';
import { useStudioDraftMirror } from '../../services/studioDraftMirror';
import { StudioDraftPicker } from '../shared/StudioDraftPicker';
import { useBackendStore } from '../../stores/backendStore';
import { expandWildcards, hasWildcards, randomWildcardSeed } from '../../utils/wildcardUtils';
import {
  MM3_CAPTION_SOURCES_KEY, clearMm3CaptionSources, pickNearestBpmTrack,
  readMm3CaptionSources, resolveMm3Caption,
  type Mm3CaptionSourcesHandoff,
} from '../../utils/mm3CaptionSource';
import {
  YUE2_BACKEND_ID, YUE2_CAPTION_DATASET_KEY, YUE2_CAPTION_SOURCES_KEY,
  ensureYue2CaptionSource, pickNearestBpmTrack as pickNearestYue2Track,
  hasStoredYue2CaptionSelection, readYue2CaptionDataset, readYue2CaptionSelection,
  readYue2CaptionSources, resolveYue2Caption,
  writeYue2CaptionSelection, yue2CaptionAdapterPath,
  yue2TrackBpm,
  type Yue2CaptionSelection, type Yue2SourceTrack,
} from '../../utils/yue2CaptionSource';
import { listDatasets, type TrainingDatasetSummary } from '../../services/trainingApi';
import { extractCaptionTags } from '../../utils/captionTags';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { writePersistedState } from '../../hooks/usePersistedState';
import type { GenerationParams, Song } from '../../types';
import type { CreateIntent } from '../../../../server/src/contracts/resolution';

interface CreatePanelProps {
  onGenerate: (params: Partial<GenerationParams>, intent?: CreateIntent) => void;
  activeJobCount: number;
  reuseData?: { song: Song; timestamp: number } | null;
}

export const CreatePanel: React.FC<CreatePanelProps> = ({ onGenerate, activeJobCount, reuseData }) => {
  const { t } = useTranslation();
  const mm3Mode = useBackendStore(s => s.activeBackendId === 'minimax-m3'
    && (s.capabilities[s.activeBackendId]?.core.captionSource ?? 'mm3-tracks') === 'mm3-tracks');

  // ── AI Generate modal ──
  const [aiModalOpen, setAiModalOpen] = useState(false);

  // ── Content (per-song) ──
  const [caption, setCaption] = usePersistedState('hs-caption', '');
  const [lyrics, setLyrics] = usePersistedState('hs-lyrics', '');
  const [negativePrompt, setNegativePrompt] = usePersistedState('hs-negative-prompt', '');
  const [instrumental, setInstrumental] = usePersistedState('hs-instrumental', false);

  // ── Compose-time caption helpers (MDMAchine) ──
  const [loraTrigger, setLoraTrigger] = usePersistedState('hs-lora-trigger', '');
  const [beatIntro, setBeatIntro] = usePersistedState('hs-beat-intro', false);
  const [introBars, setIntroBars] = usePersistedState('hs-intro-bars', 2);
  const [autoExpand, setAutoExpand] = usePersistedState('hs-main-auto-expand', false);


  // ── Song Info (optional, auto-populated from Lyric Studio Send to Create) ──
  const [title, setTitle] = usePersistedState('hs-title', '');
  const [artist, setArtist] = usePersistedState('hs-artist', '');
  const [subject, setSubject] = usePersistedState('hs-subject', '');

  // ── Metadata (per-song) ──
  const [bpm, setBpm] = usePersistedState('hs-bpm', 0);
  const [keyScale, setKeyScale] = usePersistedState('hs-keyScale', '');
  const [timeSignature, setTimeSignature] = usePersistedState('hs-timeSignature', '');
  const [duration, setDuration] = usePersistedState('hs-duration', -1);
  const [vocalLanguage, setVocalLanguage] = usePersistedState('hs-vocalLanguage', 'en');
  // '' = Any. Reaches MiniMax-Music3 only through the composed caption's Vocal
  // Details section — there is no wire field for it on either backend.
  const [vocalGender, setVocalGender] = usePersistedState('hs-vocalGender', '');
  const [sourceLatentUrl, setSourceLatentUrl] = usePersistedState('hs-sourceLatentUrl', '');

  // ── MM3 caption source (songs sent here from Lyric Studio) ──
  // Send-to-Create leaves the album's captioned source tracks and the song's
  // own caption in one localStorage key, so the same three-way control can be
  // offered here with no server call. Absent key, or a non-MM3 backend, and
  // none of this exists — the caption box is the plain editable one.
  const [mm3Sources, setMm3Sources] = useState<Mm3CaptionSourcesHandoff | null>(() => readMm3CaptionSources());
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === MM3_CAPTION_SOURCES_KEY) setMm3Sources(readMm3CaptionSources());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const mm3SourcesActive = mm3Mode && !!mm3Sources && mm3Sources.tracks.length > 0;
  const mm3Resolved = mm3SourcesActive
    ? resolveMm3Caption({ bpm, caption_mm3: mm3Sources!.customCaption }, mm3Sources!.tracks, mm3Sources!)
    : null;
  const mm3CaptionLocked = !!mm3Resolved && mm3Resolved.mode !== 'custom';

  // The resolved caption IS the caption: the request path sends `caption`
  // unchanged, so nothing server-side needs to know a source was picked. Also
  // re-runs when the tempo changes, since Automatic is a function of it.
  useEffect(() => {
    if (mm3CaptionLocked && mm3Resolved && mm3Resolved.caption !== caption) setCaption(mm3Resolved.caption);
  }, [mm3CaptionLocked, mm3Resolved?.caption]);

  const setMm3Mode = useCallback((value: string) => {
    if (!mm3Sources) return;
    const next: Mm3CaptionSourcesHandoff = value === 'auto' || value === 'custom'
      ? { ...mm3Sources, mode: value, selectedTitle: undefined }
      : { ...mm3Sources, mode: 'track', selectedTitle: value.slice('track:'.length) };
    try { localStorage.setItem(MM3_CAPTION_SOURCES_KEY, JSON.stringify(next)); } catch { /* ignore */ }
    setMm3Sources(next);
    // Switching to Custom hands the box back with the song's own caption in it —
    // otherwise it would be left holding the dataset track's.
    if (next.mode === 'custom') setCaption(mm3Sources.customCaption);
  }, [mm3Sources, setCaption]);

  const dismissMm3Sources = useCallback(() => {
    clearMm3CaptionSources();
    setMm3Sources(null);
  }, []);

  // ── YuE2 caption source ──
  // Keyed by training DATASET id, not by adapter path: a run folder can be
  // moved by hand and the cleanup job can sweep a run's prepared cache, and
  // neither has anything to do with the dataset the captions actually belong
  // to (see utils/yue2CaptionSource.ts header). `hs-yue2CaptionDataset` is the
  // dataset in force here — set by a Send-to-Create handoff or by the Dataset
  // dropdown below — and when it is empty this falls back to whatever dataset
  // the engine's resident adapter was trained on, without writing that guess
  // back, so a later handoff still wins outright.
  const yue2Mode = useBackendStore(s => s.activeBackendId === YUE2_BACKEND_ID
    && (s.capabilities[s.activeBackendId]?.core.captionSource ?? 'yue2-dataset') === 'yue2-dataset');
  const yue2AdapterPath = useBackendStore(
    s => yue2CaptionAdapterPath(s.models[YUE2_BACKEND_ID]?.defaults as Record<string, unknown> | undefined));
  const yue2HasCatalogue = useBackendStore(s => !!s.models[YUE2_BACKEND_ID]);
  const fetchBackendModels = useBackendStore(s => s.fetchModels);
  const [yue2Ds, setYue2Ds] = useState<{ datasetId: string; datasetName: string; tracks: Yue2SourceTrack[] }>(
    { datasetId: '', datasetName: '', tracks: [] });
  const [yue2SourcePending, setYue2SourcePending] = useState(false);
  const [yue2DatasetChoice, setYue2DatasetChoice] = useState<string>(() => readYue2CaptionDataset());
  const [yue2DatasetOptions, setYue2DatasetOptions] = useState<TrainingDatasetSummary[]>([]);
  const [yue2Selection, setYue2Selection] = useState<Yue2CaptionSelection>({ mode: 'custom' });
  const yue2ResolveSeq = useRef(0);
  // The caption as it stands right now, for the effect below — which must not
  // re-run on every keystroke, and so cannot have it as a dependency. Synced in
  // its own effect rather than during render, and declared FIRST so it is
  // already current by the time the effect below reads it.
  const captionRef = useRef(caption);
  useEffect(() => { captionRef.current = caption; });

  // The Adapters cluster is what normally loads the catalogue, and it only
  // mounts while that dropdown is open — so ask for it here rather than have
  // this control depend on the user having opened an unrelated panel first.
  useEffect(() => {
    if (yue2Mode && !yue2HasCatalogue) void fetchBackendModels(YUE2_BACKEND_ID);
  }, [yue2Mode, yue2HasCatalogue, fetchBackendModels]);

  // The catalogue for the Dataset dropdown — every dataset, not just the ones
  // with a YuE2 adapter, since the block is meant to be reachable even for a
  // base-model render whose dataset just happens to carry captions too.
  useEffect(() => {
    if (!yue2Mode) return;
    let live = true;
    void listDatasets().then(ds => { if (live) setYue2DatasetOptions(ds); }).catch(() => {});
    return () => { live = false; };
  }, [yue2Mode]);

  const resolveYue2Dataset = useCallback(async () => {
    const seq = ++yue2ResolveSeq.current;
    if (!yue2Mode) {
      setYue2SourcePending(false);
      setYue2Ds({ datasetId: '', datasetName: '', tracks: [] });
      return;
    }
    setYue2SourcePending(true);
    const explicit = readYue2CaptionDataset();
    setYue2DatasetChoice(explicit);
    const src = explicit
      ? await ensureYue2CaptionSource({ dataset: explicit })
      : yue2AdapterPath
        ? await ensureYue2CaptionSource({ adapter: yue2AdapterPath })
        : { datasetId: '', datasetName: '', tracks: [] };
    if (seq === yue2ResolveSeq.current) {
      setYue2Ds(src);
      setYue2SourcePending(false);
    }
  }, [yue2Mode, yue2AdapterPath]);

  useEffect(() => { void resolveYue2Dataset(); }, [resolveYue2Dataset]);

  // A Send-to-Create handoff writes both keys with the same-tab StorageEvent
  // `write()` fires (useAudioGeneration.sendToCreate), so this fires in the
  // same tab too, not only cross-tab. The dataset itself is re-resolved either
  // way; the handoff additionally carries a mode/track/custom pick that has to
  // land on THAT dataset's stored selection before the resolve reads it.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === YUE2_CAPTION_SOURCES_KEY) {
        const handoff = readYue2CaptionSources();
        if (handoff) {
          writeYue2CaptionSelection(handoff.datasetId, {
            mode: handoff.mode, selectedName: handoff.selectedName, customCaption: handoff.customCaption,
          });
        }
      }
      if (e.key === YUE2_CAPTION_SOURCES_KEY || e.key === YUE2_CAPTION_DATASET_KEY) void resolveYue2Dataset();
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [resolveYue2Dataset]);

  useEffect(() => {
    const datasetId = yue2Ds.datasetId;
    if (!yue2Mode || !datasetId) {
      setYue2Selection({ mode: 'custom' });
      return;
    }
    // Automatic is the default, so resolving a captioned dataset can take over
    // a caption box the user has already typed into. Stash what is in it the
    // first time that happens, so Custom hands their own words back instead of
    // the dataset track's. Only when nothing is stashed yet: a later visit must
    // not overwrite the stash with the dataset caption now sitting in the box.
    // ...but not over a prompt that is already written. Automatic being the
    // default meant switching the backend to YuE2 silently replaced the
    // caption the user was looking at with a dataset track's and greyed the
    // box out, with the picker that undoes it sitting below the fold (#163).
    // Automatic still wins on an empty box — the case it is the default for.
    const sel = hasStoredYue2CaptionSelection(datasetId)
      ? readYue2CaptionSelection(datasetId)
      : (captionRef.current || '').trim()
        ? { mode: 'custom' as const }
        : readYue2CaptionSelection(datasetId);
    if (sel.mode !== 'custom' && sel.customCaption === undefined) {
      sel.customCaption = captionRef.current;
      writeYue2CaptionSelection(datasetId, sel);
    }
    setYue2Selection(sel);
  }, [yue2Mode, yue2Ds.datasetId]);

  const yue2Tracks = yue2Ds.tracks;
  const yue2CaptionTags = useMemo(() => extractCaptionTags(yue2Tracks), [yue2Tracks]);
  const yue2TagSourceName = yue2Ds.datasetName
    || (() => {
      const dataset = yue2DatasetOptions.find(d => d.id === yue2Ds.datasetId);
      return dataset?.name || dataset?.albumName || dataset?.slug || yue2Ds.datasetId;
    })();
  const yue2SourcesActive = yue2Mode && yue2Tracks.length > 0;
  const yue2Resolved = yue2SourcesActive
    ? resolveYue2Caption(yue2Selection.customCaption ?? caption, bpm, yue2Tracks, yue2Selection)
    : null;
  const yue2CaptionLocked = !!yue2Resolved && yue2Resolved.mode !== 'custom';
  // The Dataset dropdown is reachable whenever there is a dataset to name —
  // chosen, adapter-resolved, or simply available to pick — even before any
  // dataset has captioned tracks to offer.
  const yue2DatasetBlockVisible = yue2Mode
    && (!!yue2Ds.datasetId || !!yue2AdapterPath || yue2DatasetOptions.length > 0);

  // Same contract as the MM3 effect: the resolved caption IS the caption, so
  // nothing on the request path needs to know a dataset track was picked.
  useEffect(() => {
    if (yue2CaptionLocked && yue2Resolved && yue2Resolved.caption !== caption) setCaption(yue2Resolved.caption);
  }, [yue2CaptionLocked, yue2Resolved?.caption]);

  const setYue2CaptionMode = useCallback((value: string) => {
    const datasetId = yue2Ds.datasetId;
    if (!datasetId) return;
    const prev = readYue2CaptionSelection(datasetId);
    // Leaving Custom is the last moment the user's own caption is still in the
    // box, so that is where it has to be stashed.
    const customCaption = prev.mode === 'custom' ? caption : prev.customCaption;
    const next: Yue2CaptionSelection =
      value === 'auto' ? { mode: 'auto', customCaption }
      : value === 'custom' ? { mode: 'custom', customCaption }
      : { mode: 'track', selectedName: value.slice('track:'.length), customCaption };
    writeYue2CaptionSelection(datasetId, next);
    setYue2Selection(next);
    if (next.mode === 'custom') setCaption(customCaption ?? '');
  }, [yue2Ds.datasetId, caption, setCaption]);

  const customTagCaption = yue2CaptionLocked ? yue2Selection.customCaption ?? '' : caption;
  const captionTagIndex = (parts: string[], phrase: string) => parts.findIndex((part, index) =>
    index % 2 === 0 && part.trim().toLocaleLowerCase() === phrase.toLocaleLowerCase());
  const hasCaptionTag = (phrase: string) =>
    captionTagIndex(customTagCaption.split(/([,.])/), phrase) >= 0;
  const toggleCaptionTag = (phrase: string) => {
    const parts = customTagCaption.split(/([,.])/);
    const index = captionTagIndex(parts, phrase);
    let next: string;
    if (index >= 0) {
      parts[index] = '';
      if (index + 1 < parts.length) parts[index + 1] = '';
      else if (index > 0) parts[index - 1] = '';
      next = parts.join('').trim().replace(/[,.]+$/, '').trim();
    } else {
      const base = customTagCaption.trim().replace(/[,.]\s*$/, '');
      next = base ? `${base}, ${phrase}` : phrase;
    }
    if (yue2Resolved?.mode !== 'custom') setYue2CaptionMode('custom');
    setCaption(next);
  };

  // The Dataset dropdown itself — writes the explicit choice ('' for None,
  // which then falls back to the adapter-resolved dataset above) and lets the
  // storage listener do the re-resolve, the same as every other write here.
  const setYue2DatasetPick = useCallback((value: string) => {
    writePersistedState(YUE2_CAPTION_DATASET_KEY, value);
  }, []);

  // Global params context — for reuse data
  const gp = useGlobalParams();

  // ── YuE2 score preview ──
  // The "Preview the score first" toggle is a YuE2 backend extension, so it
  // lives in backendParams like the other YuE2 knobs. When it is on, Generate
  // plans the lead sheet first, shows it, and only enqueues the render once
  // the user says Continue — with THAT score and THAT seed pinned.
  const { token } = useAuth();
  // The form and its caption choices, mirrored into this browser's server draft.
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const draft = useStudioDraftMirror('create', token, {
    'hs-caption': caption, 'hs-lyrics': lyrics, 'hs-negative-prompt': negativePrompt, 'hs-instrumental': instrumental,
    'hs-lora-trigger': loraTrigger, 'hs-beat-intro': beatIntro, 'hs-intro-bars': introBars,
    'hs-title': title, 'hs-artist': artist, 'hs-subject': subject, 'hs-bpm': bpm, 'hs-keyScale': keyScale,
    'hs-timeSignature': timeSignature, 'hs-duration': duration, 'hs-vocalLanguage': vocalLanguage,
    'hs-vocalGender': vocalGender, 'hs-sourceLatentUrl': sourceLatentUrl,
    'hs-mm3CaptionSources': mm3Sources, 'hs-yue2CaptionDataset': yue2DatasetChoice,
    ...(yue2Ds.datasetId ? { [`hs-yue2CaptionSource:ds:${yue2Ds.datasetId}`]: yue2Selection } : {}),
  }, activeBackendId ? { backendId: activeBackendId } : {});
  // A loaded draft goes through the same persisted keys the form reads, so
  // every field and caption choice updates as if typed. Nothing is generated.
  const applyDraft = useCallback((body: { fields: Record<string, unknown> }) => {
    for (const [key, value] of Object.entries(body.fields)) if (key.startsWith('hs-')) writePersistedState(key, value);
    const selection = yue2Ds.datasetId ? body.fields[`hs-yue2CaptionSource:ds:${yue2Ds.datasetId}`] : undefined;
    if (selection && typeof selection === 'object') setYue2Selection(selection as Yue2CaptionSelection);
  }, [yue2Ds.datasetId]);
  const yue2PreviewScore = useGlobalParamsStore((s: any) => !!s.backendParams?.yue2PreviewScore);
  const yue2AbcSupplied = useGlobalParamsStore((s: any) => typeof s.backendParams?.yue2Abc === 'string' && !!s.backendParams.yue2Abc.trim());
  const [scorePreview, setScorePreview] = useState<{ open: boolean; params: Partial<GenerationParams> | null; data: Yue2ScorePreviewData | null; error: string | null }>(
    { open: false, params: null, data: null, error: null });
  const planScore = useCallback(async (params: Partial<GenerationParams>, freshSeed: boolean) => {
    if (!token) return;
    setScorePreview({ open: true, params, data: null, error: null });
    try {
      // The plan and the render must share a seed for the render to be
      // reproducible from the sheet the user approved. Retry asks for a new
      // draw; the first attempt honours the global seed control.
      const seedParams = freshSeed
        ? { ...params, randomSeed: true, seed: -1 }
        : { ...params, randomSeed: gp.randomSeed, seed: gp.seed };
      const data = await generateApi.yue2Plan(seedParams, token);
      setScorePreview(prev => (prev.open ? { ...prev, data } : prev));
    } catch (err) {
      setScorePreview(prev => (prev.open ? { ...prev, error: err instanceof Error ? err.message : String(err) } : prev));
    }
  }, [token, gp.randomSeed, gp.seed]);

  // ── Reuse data (Edit) — restores ALL generation params for full reproducibility ──
  useEffect(() => {
    if (!reuseData) return;
    const gpData = reuseData.song.generationParams;
    if (!gpData) return;

    // Style Description field ← user's original style input from generation_params
    // Priority: gpData.caption (original user input) → song.style (DB column)
    setCaption(gpData.caption || reuseData.song.style || '');
    // Lyrics
    setLyrics(gpData.lyrics || reuseData.song.lyrics || '');
    // Song info metadata
    if (gpData.title || reuseData.song.title) setTitle(gpData.title || reuseData.song.title || '');
    if (gpData.artist) setArtist(gpData.artist);
    if (gpData.subject) setSubject(gpData.subject);
    // Metadata
    if (gpData.bpm) setBpm(gpData.bpm);
    if (gpData.keyScale) setKeyScale(gpData.keyScale);
    if (gpData.timeSignature) setTimeSignature(gpData.timeSignature);
    if (gpData.duration) setDuration(typeof gpData.duration === 'string' ? parseFloat(gpData.duration) : gpData.duration);
    if (gpData.vocalLanguage) setVocalLanguage(gpData.vocalLanguage);
    // Engine params — full reproducibility
    if (gpData.inferenceSteps) gp.setInferenceSteps(gpData.inferenceSteps);
    if (gpData.guidanceScale !== undefined) gp.setGuidanceScale(gpData.guidanceScale);
    if (gpData.cfgCutoffRatio !== undefined) gp.setCfgCutoffRatio(gpData.cfgCutoffRatio);
    if (gpData.lmCfgCutoffRatio !== undefined) gp.setLmCfgCutoffRatio(gpData.lmCfgCutoffRatio);
    if (gpData.cacheRatio !== undefined) gp.setCacheRatio(gpData.cacheRatio);
    if (gpData.seed !== undefined) gp.setSeed(gpData.seed);
    if (gpData.randomSeed !== undefined) gp.setRandomSeed(gpData.randomSeed);
    if (gpData.lmSeed !== undefined) gp.setLmSeed(gpData.lmSeed);
    if (gpData.lmSeedFollowsDit !== undefined) gp.setLmSeedFollowsDit(gpData.lmSeedFollowsDit);
    if (gpData.shift !== undefined) gp.setShift(gpData.shift);
    if (gpData.inferMethod) gp.setInferMethod(gpData.inferMethod);
    if (gpData.scheduler) gp.setScheduler(gpData.scheduler);
    if (gpData.guidanceMode) gp.setGuidanceMode(gpData.guidanceMode);
    if (gpData.batchSize) gp.setBatchSize(gpData.batchSize);
    if (gpData.useCotCaption !== undefined) gp.setUseCotCaption(gpData.useCotCaption);
    if (gpData.skipLm !== undefined) gp.setSkipLm(gpData.skipLm);
    // Adapter
    if (gpData.adapter || gpData.loraPath) gp.setAdapter(gpData.adapter || gpData.loraPath);
    if (gpData.adapterScale ?? gpData.loraScale) gp.setAdapterScale(gpData.adapterScale ?? gpData.loraScale);
    if (gpData.adapterGroupScales) gp.setAdapterGroupScales(gpData.adapterGroupScales);
    if (gpData.adapterMode) gp.setAdapterMode(gpData.adapterMode);
    // Model selection
    if (gpData.ditModel) gp.setDitModel(gpData.ditModel);
    if (gpData.lmModel) gp.setLmModel(gpData.lmModel);
    if (gpData.vaeModel) gp.setVaeModel(gpData.vaeModel);
    // DCW
    if (gpData.dcwEnabled !== undefined) gp.setDcwEnabled(gpData.dcwEnabled);
    if (gpData.dcwMode) gp.setDcwMode(gpData.dcwMode);
    if (gpData.dcwLowScaler !== undefined) gp.setDcwLowScaler(gpData.dcwLowScaler);
    if (gpData.dcwHighScaler !== undefined) gp.setDcwHighScaler(gpData.dcwHighScaler);
    // Post-processing
    if (gpData.postProcessingEnabled !== undefined) gp.setPostProcessingEnabled(gpData.postProcessingEnabled);
    if (gpData.masteringEnabled !== undefined) gp.setMasteringEnabled(gpData.masteringEnabled);
    if (gpData.masteringReference !== undefined) gp.setMasteringReference(gpData.masteringReference);
  }, [reuseData?.timestamp]);

  // ── AI generation result handler ──
  const handleAiResult = useCallback((result: AiGenerateResult) => {
    if (result.caption) setCaption(result.caption);
    if (result.lyrics) setLyrics(result.lyrics);
    if (result.title) setTitle(result.title);
    if (result.subject) setSubject(result.subject);
    if (result.bpm) setBpm(result.bpm);
    if (result.keyScale) setKeyScale(result.keyScale);
    if (result.timeSignature) setTimeSignature(result.timeSignature);
    if (result.duration) setDuration(result.duration);
    if (result.vocalLanguage) setVocalLanguage(result.vocalLanguage);
    // Disable instrumental mode since AI generated lyrics
    setInstrumental(false);
  }, [setCaption, setLyrics, setTitle, setSubject, setBpm, setKeyScale, setTimeSignature, setDuration, setVocalLanguage, setInstrumental]);

  const handleGenerate = () => {
    // Wildcard auto-expand: reproducible from the DiT seed when it's locked,
    // fresh randomness when the seed is random anyway
    const wcSeed = gp.randomSeed ? randomWildcardSeed() : gp.seed;
    const resolvedCaption = autoExpand && hasWildcards(caption)
      ? expandWildcards(caption, wcSeed, 0) : caption;
    const resolvedLyrics = autoExpand && hasWildcards(lyrics)
      ? expandWildcards(lyrics, wcSeed, 0) : lyrics;

    const params = createContentParams({
      caption, lyrics, negativePrompt, instrumental, loraTrigger, beatIntro, introBars,
      title, artist, subject, bpm, keyScale, timeSignature, duration, vocalLanguage, sourceLatentUrl,
    }, { mm3Mode, resolvedCaption, resolvedLyrics });
    // A pasted lead sheet (#181) is the score; previewing would plan a new one
    // and throw it away.
    if (yue2Mode && yue2PreviewScore && !yue2AbcSupplied) {
      void planScore(params, false);
      return;
    }
    onGenerate(params, createIntent(params));
  };

  const createIntent = (legacyParams: Partial<GenerationParams>): CreateIntent => ({
    kind: 'create',
    params: { ...legacyParams, caption, lyrics },
    compose: { autoExpand, loraTrigger, beatIntro, introBars },
    ...(mm3SourcesActive ? { captionSource: {
      engine: 'minimax-m3' as const,
      customCaption: mm3Sources!.customCaption,
      selection: { mode: mm3Sources!.mode, selectedTitle: mm3Sources!.selectedTitle },
      tracks: mm3Sources!.tracks,
    } } : yue2SourcesActive ? { captionSource: {
      engine: 'yue2' as const,
      datasetId: yue2Ds.datasetId,
      adapterInForce: !!yue2AdapterPath,
      selection: { mode: yue2Selection.mode, selectedName: yue2Selection.selectedName },
    } } : {}),
  });

  return (
    <div className="h-full flex flex-col bg-zinc-50 dark:bg-suno">
      <TrainingDraftBar />
      <StudioDraftPicker control={draft} textKey="hs-title" apply={applyDraft} />
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">{t('createPanel.title')}</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAiModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-violet-400 hover:text-violet-300 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 hover:border-violet-500/30 transition-all duration-200"
            title="Generate all fields using an external AI model"
          >
            <Sparkles size={13} />
            Generate with AI
          </button>
        </div>
      </div>

      {/* Scrollable body — now much slimmer */}
      <div className="flex-1 overflow-y-auto hide-scrollbar px-4 py-3 space-y-1">
        <ContentSection
          caption={caption} onCaptionChange={setCaption}
          lyrics={lyrics} onLyricsChange={setLyrics}
          instrumental={instrumental} onInstrumentalChange={setInstrumental}
          title={title} onTitleChange={setTitle}
          artist={artist} onArtistChange={setArtist}
          subject={subject} onSubjectChange={setSubject}
          negativePrompt={negativePrompt} onNegativePromptChange={setNegativePrompt}
          loraTrigger={loraTrigger} onLoraTriggerChange={setLoraTrigger}
          beatIntro={beatIntro} onBeatIntroChange={setBeatIntro}
          introBars={introBars} onIntroBarsChange={setIntroBars}
          autoExpand={autoExpand} onAutoExpandChange={setAutoExpand}
          wildcardSeed={gp.randomSeed ? undefined : gp.seed}
          captionReadOnly={mm3CaptionLocked || yue2CaptionLocked}
        />

        {/* MM3 only, and only for a song sent here from Lyric Studio: which
            caption renders. Reusing a training track's own Structured Caption
            verbatim is what reliably lands in the artist's style and reaches a
            natural ending, so it is the default and this song's own caption is
            the opt-in. */}
        {mm3SourcesActive && mm3Resolved && (
          <div className="pt-2 space-y-1">
            <div className="flex items-center justify-between">
              <ParamLabel
                label={t('createPanel.mm3CaptionSource', 'Caption source')}
                className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
                info={t('createPanel.mm3CaptionSourceInfo', "Chooses which caption this song actually sends to MiniMax-Music3. Automatic from dataset picks the album track whose BPM is closest to this song's, and uses that track's own Structured Caption verbatim. A named Track locks in that specific track's caption instead. Custom uses this song's own caption, editable in the Style Description box above.")}
              />
              <button
                onClick={dismissMm3Sources}
                title={t('createPanel.mm3CaptionSourceDismissHint', 'Stop using the album’s captions for this panel')}
                className="text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors px-1"
              >
                {t('createPanel.mm3CaptionSourceDismiss', 'Dismiss')}
              </button>
            </div>
            <StyledSelect
              accent="pink"
              size="sm"
              className="w-full"
              value={mm3Resolved.mode === 'track' && mm3Resolved.fromTitle ? `track:${mm3Resolved.fromTitle}` : mm3Resolved.mode}
              onChange={setMm3Mode}
              options={[
                {
                  value: 'auto',
                  label: `${t('createPanel.mm3CaptionAuto', 'Automatic from dataset')}${(() => {
                    const auto = pickNearestBpmTrack(mm3Sources!.tracks, bpm);
                    return auto ? ` (${t('createPanel.mm3CaptionNearestTempo', 'nearest tempo')}: ${auto.title})` : '';
                  })()}`,
                },
                ...mm3Sources!.tracks.map(track => ({
                  value: `track:${track.title}`,
                  label: `${t('createPanel.mm3CaptionTrack', 'Track')}: ${track.title}${track.bpm ? ` · ${track.bpm} BPM` : ''}`,
                })),
                { value: 'custom', label: t('createPanel.mm3CaptionCustom', "Custom (this song's own caption)") },
              ]}
            />
            {mm3CaptionLocked && (
              <p className="text-[10px] text-cyan-400/70">
                {t('createPanel.mm3CaptionFromTrack', 'From dataset track')}: {mm3Resolved.fromTitle}
              </p>
            )}
          </div>
        )}

        {/* YuE2 only: which training DATASET the caption picker below offers,
            and — once that dataset has captioned tracks — which caption
            conditions the render. An AR adapter was trained on whole songs
            under their own captions with half of them dropped, so every
            dataset caption is a prompt it has actually seen — picking one
            steers towards that track rather than the album's average. The
            trigger word is not shown or typed here: generate.ts wraps
            whatever this box holds in the adapter's own style template.
            The Dataset row is always reachable in YuE2 mode, even before any
            dataset is chosen, so switching backends never hides the control
            below the fold the way #163 did for the caption source itself. */}
        {yue2DatasetBlockVisible && (
          <div className="pt-2 space-y-1">
            <ParamLabel
              label={t('createPanel.yue2Dataset', 'Dataset')}
              className="text-xs font-medium text-zinc-500 uppercase tracking-wider block"
              info={t('createPanel.yue2DatasetInfo', "Picks which training dataset's captions the Caption source picker below offers. Defaults to the dataset the active YuE2 adapter was trained on. None turns the picker off and leaves the caption box as plain free text.")}
            />
            <StyledSelect
              value={yue2DatasetChoice}
              onChange={setYue2DatasetPick}
              accent="pink"
              size="sm"
              placeholder={t('createPanel.yue2DatasetNone', 'None')}
              options={[
                { value: '', label: t('createPanel.yue2DatasetNone', 'None') },
                ...yue2DatasetOptions.map(d => ({
                  value: d.id, label: d.name || d.albumName || d.slug, hint: d.slug,
                })),
              ]}
            />

            {!!yue2AdapterPath && !yue2Ds.datasetId && !yue2SourcePending && (
              <p className="text-[11px] text-zinc-500">
                {t('createPanel.yue2NoLinkedCaptions', 'This adapter has no linked training captions.')}
              </p>
            )}

            {yue2SourcesActive && yue2Resolved && (
              <>
                <ParamLabel
                  label={t('createPanel.yue2CaptionSource', 'Caption source')}
                  className="text-xs font-medium text-zinc-500 uppercase tracking-wider block pt-1"
                  info={t('createPanel.yue2CaptionSourceInfo', "Chooses which caption this song actually sends to YuE2. Custom uses the caption typed in the Style Description box above. Automatic from dataset picks the dataset track whose BPM is closest to this song's and uses that track's own caption. A named Track locks in that specific track's caption instead — every dataset caption is one the adapter actually trained on, so picking one steers towards that track rather than the album's average.")}
                />
                <StyledSelect
                  accent="pink"
                  size="sm"
                  className="w-full"
                  value={yue2Resolved.mode === 'track' && yue2Resolved.fromName ? `track:${yue2Resolved.fromName}` : yue2Resolved.mode}
                  onChange={setYue2CaptionMode}
                  options={[
                    { value: 'custom', label: t('createPanel.yue2CaptionCustom', 'Custom (the caption above)') },
                    {
                      value: 'auto',
                      label: `${t('createPanel.yue2CaptionAuto', 'Automatic from dataset')}${(() => {
                        const auto = pickNearestYue2Track(yue2Tracks, bpm);
                        return auto ? ` (${t('createPanel.yue2CaptionNearestTempo', 'nearest tempo')}: ${auto.name})` : '';
                      })()}`,
                    },
                    ...yue2Tracks.map(track => ({
                      value: `track:${track.name}`,
                      label: `${t('createPanel.yue2CaptionTrack', 'Track')}: ${track.name}${yue2TrackBpm(track) ? ` · ${yue2TrackBpm(track)} BPM` : ''}`,
                    })),
                  ]}
                />
                {!yue2SourcePending && yue2CaptionTags.length > 0 && (
                  <div className="space-y-1.5 pt-1">
                    <ParamLabel
                      label={t('createPanel.yue2TagsFrom', { source: yue2TagSourceName, defaultValue: 'Tags from {{source}}' })}
                      className="text-xs font-medium text-zinc-500 uppercase tracking-wider block"
                      info={t('createPanel.yue2TagsInfo', 'These phrases come from the selected training dataset. Click a tag to add it to your Custom caption; click it again to remove it.')}
                    />
                    <div className="flex flex-wrap gap-1.5">
                      {yue2CaptionTags.map(phrase => {
                        const selected = hasCaptionTag(phrase);
                        return (
                          <button
                            key={phrase.toLocaleLowerCase()}
                            type="button"
                            aria-pressed={selected}
                            onClick={() => toggleCaptionTag(phrase)}
                            className={`rounded-full border px-2.5 py-1 text-[11px] transition-colors ${selected
                              ? 'border-pink-400/60 bg-pink-500/20 text-pink-200'
                              : 'border-pink-500/25 bg-pink-500/5 text-zinc-400 hover:border-pink-400/50 hover:text-pink-200'}`}
                          >
                            {phrase}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
                {yue2CaptionLocked && (
                  <p className="text-[10px] text-emerald-400/70">
                    {t('createPanel.yue2CaptionFromTrack', 'From dataset track')}: {yue2Resolved.fromName}
                  </p>
                )}
              </>
            )}
          </div>
        )}

        {/* MM3 only: turn the plain-English caption into a Structured Caption.
            Hidden while a dataset caption is locked in — composing would only
            overwrite a box the user cannot edit. */}
        {mm3Mode && !mm3CaptionLocked && (
          <div className="pt-2">
            <Mm3ComposeButton
              brief={caption}
              controls={{ bpm, keyScale, timeSignature, duration, vocalLanguage, vocalGender }}
              onComposed={setCaption}
            />
          </div>
        )}

        <MetadataSection
          bpm={bpm} onBpmChange={setBpm}
          keyScale={keyScale} onKeyScaleChange={setKeyScale}
          timeSignature={timeSignature} onTimeSignatureChange={setTimeSignature}
          duration={duration} onDurationChange={setDuration}
          vocalLanguage={vocalLanguage} onVocalLanguageChange={setVocalLanguage}
          vocalGender={vocalGender} onVocalGenderChange={setVocalGender}
        />

        {/* Latent import */}
        <LatentImport
          latentUrl={sourceLatentUrl}
          onLatentLoaded={(url, meta) => {
            setSourceLatentUrl(url);
            if (meta.bpm && meta.bpm > 0) setBpm(meta.bpm);
            if (meta.key) setKeyScale(meta.key);
            if (meta.lyrics) setLyrics(meta.lyrics);
            if (meta.caption) setCaption(meta.caption);
          }}
          onClear={() => setSourceLatentUrl('')}
        />

        {/* Cover Art prompt override (only when enabled) */}
        <CoverArtSubjectSection />
      </div>

      {/* Generate button */}
      <div className="px-4 py-3 border-t border-zinc-200 dark:border-white/5 space-y-2">
        <button
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 hover:shadow-lg hover:shadow-pink-500/20 text-white"
          onClick={handleGenerate}
          disabled={!caption.trim() && !lyrics.trim() && !instrumental}
        >
          {activeJobCount > 0 ? (
            <>
              <ListPlus size={18} />
              {t('createPanel.queueGeneration')}
              <span className="ml-1 inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full bg-white/20 text-xs font-bold tabular-nums">
                {activeJobCount}
              </span>
            </>
          ) : (
            <>
              <Zap size={18} />
              {t('createPanel.generate')}
            </>
          )}
        </button>
      </div>

      {/* AI Generate Modal */}
      <AiGenerateModal
        isOpen={aiModalOpen}
        onClose={() => setAiModalOpen(false)}
        onResult={handleAiResult}
      />

      {/* YuE2 lead-sheet preview: continue renders the approved score with
          its seed pinned; retry plans again with a fresh seed. */}
      <Yue2ScorePreviewModal
        open={scorePreview.open}
        data={scorePreview.data}
        error={scorePreview.error}
        onCancel={() => setScorePreview({ open: false, params: null, data: null, error: null })}
        onRetry={() => { if (scorePreview.params) void planScore(scorePreview.params, true); }}
        onContinue={() => {
          const { params, data } = scorePreview;
          setScorePreview({ open: false, params: null, data: null, error: null });
          if (!params || !data) return;
          const approved = { ...params, yue2Abc: data.abc, seed: data.seed, randomSeed: false } as Partial<GenerationParams>;
          onGenerate(approved, createIntent(approved));
        }}
      />
    </div>
  );
};
