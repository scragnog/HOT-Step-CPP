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
import { useStreamGeneration } from '../../hooks/useStreamGeneration';
import { StreamPlayer } from '../player/StreamPlayer';
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
  const [sourceLatentUrl, setSourceLatentUrl] = usePersistedState('hs-sourceLatentUrl', '');

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
    const params: Partial<GenerationParams> = {
      caption,
      lyrics: instrumental ? '[Instrumental]' : lyrics,
      ...(negativePrompt.trim() ? { negative_prompt: negativePrompt.trim() } : {}),
      instrumental,
      bpm, duration, keyScale, timeSignature, vocalLanguage,
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
        />

        <MetadataSection
          bpm={bpm} onBpmChange={setBpm}
          keyScale={keyScale} onKeyScaleChange={setKeyScale}
          timeSignature={timeSignature} onTimeSignatureChange={setTimeSignature}
          duration={duration} onDurationChange={setDuration}
          vocalLanguage={vocalLanguage} onVocalLanguageChange={setVocalLanguage}
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
