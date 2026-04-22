// CreatePanel.tsx — The composition panel (Content + Metadata only)
//
// Global engine parameters (Models, Adapters, Generation Settings, LM, Mastering)
// have been moved to the GlobalParamBar. This panel now only handles
// per-song content and metadata.

import React, { useEffect } from 'react';
import { Zap, ListPlus } from 'lucide-react';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { ContentSection } from './ContentSection';
import { MetadataSection } from './MetadataSection';
import type { GenerationParams, Song } from '../../types';

interface CreatePanelProps {
  onGenerate: (params: Partial<GenerationParams>) => void;
  activeJobCount: number;
  reuseData?: { song: Song; timestamp: number } | null;
}

export const CreatePanel: React.FC<CreatePanelProps> = ({ onGenerate, activeJobCount, reuseData }) => {
  // ── Content (per-song) ──
  const [caption, setCaption] = usePersistedState('hs-caption', '');
  const [lyrics, setLyrics] = usePersistedState('hs-lyrics', '');
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

  // Global params context — for reuse data
  const gp = useGlobalParams();

  // ── Reuse data ──
  useEffect(() => {
    if (!reuseData) return;
    const gpData = reuseData.song.generationParams;
    if (!gpData) return;

    if (reuseData.song.caption || gpData.caption) setCaption(reuseData.song.caption || gpData.caption || '');
    if (reuseData.song.lyrics || gpData.lyrics) setLyrics(reuseData.song.lyrics || gpData.lyrics || '');
    if (reuseData.song.style || gpData.style) setCaption(reuseData.song.style || gpData.style || '');
    if (gpData.bpm) setBpm(gpData.bpm);
    if (gpData.keyScale) setKeyScale(gpData.keyScale);
    if (gpData.timeSignature) setTimeSignature(gpData.timeSignature);
    if (gpData.duration) setDuration(typeof gpData.duration === 'string' ? parseFloat(gpData.duration) : gpData.duration);
    if (gpData.inferenceSteps) gp.setInferenceSteps(gpData.inferenceSteps);
    if (gpData.guidanceScale !== undefined) gp.setGuidanceScale(gpData.guidanceScale);
    if (gpData.seed !== undefined) gp.setSeed(gpData.seed);
  }, [reuseData?.timestamp]);

  const handleGenerate = () => {
    const params: Partial<GenerationParams> = {
      caption,
      lyrics: instrumental ? '[Instrumental]' : lyrics,
      instrumental,
      bpm, duration, keyScale, timeSignature, vocalLanguage,
      taskType: 'text2music',
    };
    // Optional song info fields — only include if populated
    if (title.trim()) params.title = title.trim();
    if (artist.trim()) params.artist = artist.trim();
    if (subject.trim()) params.subject = subject.trim();
    onGenerate(params);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-50 dark:bg-suno-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Create</h2>
        <span className="text-xs text-zinc-500 font-medium">text2music</span>
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
        />

        <MetadataSection
          bpm={bpm} onBpmChange={setBpm}
          keyScale={keyScale} onKeyScaleChange={setKeyScale}
          timeSignature={timeSignature} onTimeSignatureChange={setTimeSignature}
          duration={duration} onDurationChange={setDuration}
          vocalLanguage={vocalLanguage} onVocalLanguageChange={setVocalLanguage}
        />
      </div>

      {/* Generate button */}
      <div className="px-4 py-3 border-t border-zinc-200 dark:border-white/5">
        <button
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white font-semibold text-sm transition-all duration-200 hover:shadow-lg hover:shadow-pink-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={handleGenerate}
          disabled={!caption.trim() && !lyrics.trim() && !instrumental}
        >
          {activeJobCount > 0 ? (
            <>
              <ListPlus size={18} />
              Queue Generation
              <span className="ml-1 inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full bg-white/20 text-xs font-bold tabular-nums">
                {activeJobCount}
              </span>
            </>
          ) : (
            <>
              <Zap size={18} />
              Generate
            </>
          )}
        </button>
      </div>
    </div>
  );
};
