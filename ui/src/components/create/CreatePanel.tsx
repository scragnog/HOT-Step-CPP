// CreatePanel.tsx — The composition panel (Content + Metadata only)
//
// Global engine parameters (Models, Adapters, Generation Settings, LM, Mastering)
// have been moved to the GlobalParamBar. This panel now only handles
// per-song content and metadata.

import React, { useState, useEffect, useCallback } from 'react';
import { Zap, ListPlus, Sparkles, Radio } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { ContentSection } from './ContentSection';
import { MetadataSection } from './MetadataSection';
import { LatentImport } from '../shared/LatentImport';
import { CoverArtSubjectSection } from '../shared/CoverArtSubjectSection';
import { AiGenerateModal, type AiGenerateResult } from './AiGenerateModal';
import { Mm3ComposeButton } from './Mm3ComposeButton';
import { useStreamGeneration } from '../../hooks/useStreamGeneration';
import { useBackendStore } from '../../stores/backendStore';
import { StreamPlayer } from '../player/StreamPlayer';
import { expandWildcards, hasWildcards, randomWildcardSeed } from '../../utils/wildcardUtils';
import {
  MM3_CAPTION_SOURCES_KEY, clearMm3CaptionSources, pickNearestBpmTrack,
  readMm3CaptionSources, resolveMm3Caption,
  type Mm3CaptionSourcesHandoff,
} from '../../utils/mm3CaptionSource';
import {
  YUE2_BACKEND_ID, ensureYue2SourceTracks, pickNearestBpmTrack as pickNearestYue2Track,
  readYue2CaptionSelection, resolveYue2Caption, writeYue2CaptionSelection, yue2CaptionAdapterPath,
  yue2TrackBpm,
  type Yue2CaptionSelection, type Yue2SourceTrack,
} from '../../utils/yue2CaptionSource';
import type { GenerationParams, Song } from '../../types';

interface CreatePanelProps {
  onGenerate: (params: Partial<GenerationParams>) => void;
  activeJobCount: number;
  reuseData?: { song: Song; timestamp: number } | null;
  /** Currently active streaming job ID (for SSE connection) */
  streamJobId?: string | null;
}

export const CreatePanel: React.FC<CreatePanelProps> = ({ onGenerate, activeJobCount, reuseData, streamJobId }) => {
  const { t } = useTranslation();
  const mm3Mode = useBackendStore(s => s.activeBackendId) === 'minimax-m3';

  // ── Stream mode ──
  const [streamMode, setStreamMode] = usePersistedState('hs-streamMode', false);
  const stream = useStreamGeneration(streamJobId || null);

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

  // LoRA trigger word prepended, beat intro/outro request appended
  const buildCaption = useCallback((base: string) => {
    const loraText = loraTrigger.trim() ? `${loraTrigger.trim()}, ` : '';
    const beatText = beatIntro ? `, with a clean ${introBars}-bar percussive intro and outro for DJ mixing` : '';
    return `${loraText}${base}${beatText}`;
  }, [loraTrigger, beatIntro, introBars]);

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
  // The same choice as MM3's above, keyed by the selected AR adapter instead of
  // by a song: the adapter is merged into the resident LM, so it is in force for
  // every render until it is switched, and its training captions are the ones
  // that are in distribution for it. Default is Custom — unlike the MM3 control,
  // this one can appear over a caption the user typed.
  const yue2Mode = useBackendStore(s => s.activeBackendId) === YUE2_BACKEND_ID;
  const yue2Adapter = useBackendStore(
    s => yue2CaptionAdapterPath(s.models[YUE2_BACKEND_ID]?.defaults as Record<string, unknown> | undefined));
  const yue2HasCatalogue = useBackendStore(s => !!s.models[YUE2_BACKEND_ID]);
  const fetchBackendModels = useBackendStore(s => s.fetchModels);
  const [yue2Tracks, setYue2Tracks] = useState<Yue2SourceTrack[]>([]);
  const [yue2Selection, setYue2Selection] = useState<Yue2CaptionSelection>({ mode: 'custom' });

  // The Adapters cluster is what normally loads the catalogue, and it only
  // mounts while that dropdown is open — so ask for it here rather than have
  // this control depend on the user having opened an unrelated panel first.
  useEffect(() => {
    if (yue2Mode && !yue2HasCatalogue) void fetchBackendModels(YUE2_BACKEND_ID);
  }, [yue2Mode, yue2HasCatalogue, fetchBackendModels]);

  useEffect(() => {
    if (!yue2Mode || !yue2Adapter) {
      setYue2Tracks([]);
      setYue2Selection({ mode: 'custom' });
      return;
    }
    setYue2Selection(readYue2CaptionSelection(yue2Adapter));
    let live = true;
    void ensureYue2SourceTracks(yue2Adapter).then(tracks => { if (live) setYue2Tracks(tracks); });
    return () => { live = false; };
  }, [yue2Mode, yue2Adapter]);

  const yue2SourcesActive = yue2Mode && yue2Tracks.length > 0;
  const yue2Resolved = yue2SourcesActive
    ? resolveYue2Caption(yue2Selection.customCaption ?? caption, bpm, yue2Tracks, yue2Selection)
    : null;
  const yue2CaptionLocked = !!yue2Resolved && yue2Resolved.mode !== 'custom';

  // Same contract as the MM3 effect: the resolved caption IS the caption, so
  // nothing on the request path needs to know a dataset track was picked.
  useEffect(() => {
    if (yue2CaptionLocked && yue2Resolved && yue2Resolved.caption !== caption) setCaption(yue2Resolved.caption);
  }, [yue2CaptionLocked, yue2Resolved?.caption]);

  const setYue2CaptionMode = useCallback((value: string) => {
    if (!yue2Adapter) return;
    const prev = readYue2CaptionSelection(yue2Adapter);
    // Leaving Custom is the last moment the user's own caption is still in the
    // box, so that is where it has to be stashed.
    const customCaption = prev.mode === 'custom' ? caption : prev.customCaption;
    const next: Yue2CaptionSelection =
      value === 'auto' ? { mode: 'auto', customCaption }
      : value === 'custom' ? { mode: 'custom', customCaption }
      : { mode: 'track', selectedName: value.slice('track:'.length), customCaption };
    writeYue2CaptionSelection(yue2Adapter, next);
    setYue2Selection(next);
    if (next.mode === 'custom') setCaption(customCaption ?? '');
  }, [yue2Adapter, caption, setCaption]);

  // Global params context — for reuse data
  const gp = useGlobalParams();

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

    const params: Partial<GenerationParams> = {
      caption: buildCaption(resolvedCaption),
      lyrics: instrumental ? '[Instrumental]' : resolvedLyrics,
      ...(negativePrompt.trim() ? { negative_prompt: negativePrompt.trim() } : {}),
      instrumental,
      bpm, keyScale, timeSignature, vocalLanguage,
      // MM3 has no length input — a duration there is a frame cap that can only
      // truncate the planner's own ending, so every MM3 render is auto. The
      // control is hidden in MM3 mode (MetadataSection), and this stops the
      // persisted ACE value riding along behind it. The backend enforces the
      // same thing, so a stale row or a direct API call cannot reinstate a cap.
      duration: mm3Mode ? -1 : duration,
      // vocalGender is deliberately NOT sent: neither backend has a wire field
      // for it. It reaches the model only by being written into the caption's
      // Vocal Details section by Mm3ComposeButton, and the caption is what
      // travels. Adding it to the request would create another dead knob.
      taskType: 'text2music',
    };
    // Optional song info fields — only include if populated
    if (title.trim()) params.title = title.trim();
    if (artist.trim()) params.artist = artist.trim();
    if (subject.trim()) params.subject = subject.trim();
    if (sourceLatentUrl) params.sourceLatentUrl = sourceLatentUrl;
    // Stream mode — SHELVED
    // if (streamMode) {
    //   (params as any).streamMode = true;
    // }
    onGenerate(params);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-50 dark:bg-suno">
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
              <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
                {t('createPanel.mm3CaptionSource', 'Caption source')}
              </label>
              <button
                onClick={dismissMm3Sources}
                title={t('createPanel.mm3CaptionSourceDismissHint', 'Stop using the album’s captions for this panel')}
                className="text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors px-1"
              >
                {t('createPanel.mm3CaptionSourceDismiss', 'Dismiss')}
              </button>
            </div>
            <select
              value={mm3Resolved.mode === 'track' && mm3Resolved.fromTitle ? `track:${mm3Resolved.fromTitle}` : mm3Resolved.mode}
              onChange={e => setMm3Mode(e.target.value)}
              className="w-full px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-xs text-zinc-700 dark:text-zinc-300 outline-none focus:border-cyan-500/50 transition-colors"
            >
              <option value="auto">
                {t('createPanel.mm3CaptionAuto', 'Automatic from dataset')}
                {(() => {
                  const auto = pickNearestBpmTrack(mm3Sources!.tracks, bpm);
                  return auto ? ` (${t('createPanel.mm3CaptionNearestTempo', 'nearest tempo')}: ${auto.title})` : '';
                })()}
              </option>
              {mm3Sources!.tracks.map(track => (
                <option key={track.title} value={`track:${track.title}`}>
                  {t('createPanel.mm3CaptionTrack', 'Track')}: {track.title}{track.bpm ? ` · ${track.bpm} BPM` : ''}
                </option>
              ))}
              <option value="custom">{t('createPanel.mm3CaptionCustom', "Custom (this song's own caption)")}</option>
            </select>
            {mm3CaptionLocked && (
              <p className="text-[10px] text-cyan-400/70">
                {t('createPanel.mm3CaptionFromTrack', 'From dataset track')}: {mm3Resolved.fromTitle}
              </p>
            )}
          </div>
        )}

        {/* YuE2 only, and only when the selected adapter's dataset has captions:
            which caption conditions the render. An AR adapter was trained on
            whole songs under their own captions with half of them dropped, so
            every dataset caption is a prompt it has actually seen — picking one
            steers towards that track rather than the album's average. The
            trigger word is not shown or typed here: generate.ts wraps whatever
            this box holds in the adapter's own style template. */}
        {yue2SourcesActive && yue2Resolved && (
          <div className="pt-2 space-y-1">
            <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider block">
              {t('createPanel.yue2CaptionSource', 'Caption source')}
            </label>
            <select
              value={yue2Resolved.mode === 'track' && yue2Resolved.fromName ? `track:${yue2Resolved.fromName}` : yue2Resolved.mode}
              onChange={e => setYue2CaptionMode(e.target.value)}
              className="w-full px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-xs text-zinc-700 dark:text-zinc-300 outline-none focus:border-emerald-500/50 transition-colors"
            >
              <option value="custom">{t('createPanel.yue2CaptionCustom', 'Custom (the caption above)')}</option>
              <option value="auto">
                {t('createPanel.yue2CaptionAuto', 'Automatic from dataset')}
                {(() => {
                  const auto = pickNearestYue2Track(yue2Tracks, bpm);
                  return auto ? ` (${t('createPanel.yue2CaptionNearestTempo', 'nearest tempo')}: ${auto.name})` : '';
                })()}
              </option>
              {yue2Tracks.map(track => (
                <option key={track.name} value={`track:${track.name}`}>
                  {t('createPanel.yue2CaptionTrack', 'Track')}: {track.name}
                  {yue2TrackBpm(track) ? ` · ${yue2TrackBpm(track)} BPM` : ''}
                </option>
              ))}
            </select>
            {yue2CaptionLocked && (
              <p className="text-[10px] text-emerald-400/70">
                {t('createPanel.yue2CaptionFromTrack', 'From dataset track')}: {yue2Resolved.fromName}
              </p>
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

      {/* Stream Player — SHELVED: streaming not yet production-ready */}
      {false && streamJobId && (
        <div className="px-4 py-2 border-t border-zinc-200 dark:border-white/5">
          <StreamPlayer
            connected={stream.connected}
            status={stream.status}
            previews={stream.previews}
            playing={stream.playing}
            done={stream.done}
            error={stream.error}
            onPlay={stream.play}
            onPause={stream.pause}
            onStop={stream.stop}
          />
        </div>
      )}

      {/* Generate button + Stream toggle */}
      <div className="px-4 py-3 border-t border-zinc-200 dark:border-white/5 space-y-2">
        {/* Stream mode toggle — SHELVED: streaming not yet production-ready */}
        {false && <div className="flex items-center justify-between">
          <button
            onClick={() => setStreamMode(!streamMode)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
              streamMode
                ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
                : 'text-zinc-500 hover:text-zinc-300 bg-zinc-800/50 border border-zinc-700/50'
            }`}
            title="Enable streaming preview — hear audio as it generates"
          >
            <Radio size={12} />
            Stream
          </button>
          {streamMode && (
            <span className="text-[10px] text-zinc-600 italic">Preview audio during generation</span>
          )}
        </div>}

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
    </div>
  );
};
