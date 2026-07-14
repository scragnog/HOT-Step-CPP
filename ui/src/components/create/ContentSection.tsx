// ContentSection.tsx — Caption + Lyrics input area with optional metadata fields
// Ported to Tailwind styling matching hot-step-9000.

import React from 'react';
import { Music, ChevronDown, ChevronRight, Plug, Drum } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { hasWildcards, expandInPlace, randomWildcardSeed } from '../../utils/wildcardUtils';

interface ContentSectionProps {
  caption: string;
  onCaptionChange: (v: string) => void;
  lyrics: string;
  onLyricsChange: (v: string) => void;
  instrumental: boolean;
  onInstrumentalChange: (v: boolean) => void;
  // Optional metadata fields (auto-populated from Lyric Studio Send to Create)
  title: string;
  onTitleChange: (v: string) => void;
  artist: string;
  onArtistChange: (v: string) => void;
  subject: string;
  onSubjectChange: (v: string) => void;
  negativePrompt: string;
  onNegativePromptChange: (v: string) => void;
  // Compose-time caption helpers (MDMAchine)
  loraTrigger: string;
  onLoraTriggerChange: (v: string) => void;
  beatIntro: boolean;
  onBeatIntroChange: (v: boolean) => void;
  introBars: number;
  onIntroBarsChange: (v: number) => void;
  autoExpand: boolean;
  onAutoExpandChange: (v: boolean) => void;
  /** Seed for manual wildcard expansion; undefined = random per click */
  wildcardSeed?: number;
}

export const ContentSection: React.FC<ContentSectionProps> = ({
  caption, onCaptionChange, lyrics, onLyricsChange,
  instrumental, onInstrumentalChange,
  title, onTitleChange, artist, onArtistChange, subject, onSubjectChange,
  negativePrompt, onNegativePromptChange,
  loraTrigger, onLoraTriggerChange,
  beatIntro, onBeatIntroChange,
  introBars, onIntroBarsChange,
  autoExpand, onAutoExpandChange,
  wildcardSeed,
}) => {
  const { t } = useTranslation();
  const hasMetadata = !!(title || artist || subject);
  const [showMetadata, setShowMetadata] = React.useState(hasMetadata);
  const styleRef = React.useRef<HTMLTextAreaElement>(null);
  const lyricsRef = React.useRef<HTMLTextAreaElement>(null);

  // Expand {a|b|c} wildcards in place, preserving the caret position
  const expandField = (
    ref: React.RefObject<HTMLTextAreaElement | null>,
    onChange: (v: string) => void,
  ) => {
    if (!ref.current) return;
    const seed = wildcardSeed ?? randomWildcardSeed();
    const { value, selectionStart, selectionEnd } = expandInPlace(ref.current, seed, 0);
    onChange(value);
    requestAnimationFrame(() => {
      ref.current?.setSelectionRange(selectionStart, selectionEnd);
      ref.current?.focus();
    });
  };

  const autoExpandBtn = (
    <button
      onClick={() => onAutoExpandChange(!autoExpand)}
      title={autoExpand ? t('contentSection.autoExpandOn') : t('contentSection.autoExpandOff')}
      className={`text-[9px] px-1.5 py-0.5 rounded transition-colors ${
        autoExpand
          ? 'bg-amber-500/20 text-amber-600 dark:bg-amber-600/30 dark:text-amber-300 hover:bg-amber-500/30 dark:hover:bg-amber-600/50'
          : 'text-zinc-400 dark:text-zinc-600 hover:text-zinc-600 dark:hover:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-800'
      }`}
    >
      ⚄ {t('contentSection.autoExpand')}
    </button>
  );

  // Auto-expand when metadata is populated (e.g. from Send to Create)
  React.useEffect(() => {
    if (hasMetadata && !showMetadata) setShowMetadata(true);
  }, [hasMetadata]);

  return (
    <div className="space-y-3">
      {/* Style / Caption */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
            {t('contentSection.styleDescription')}
          </label>
          <div className="flex items-center gap-1.5">
            {hasWildcards(caption) && (
              <button
                onClick={() => expandField(styleRef, onCaptionChange)}
                className="text-[9px] px-1.5 py-0.5 rounded font-mono bg-pink-500/10 text-pink-600 dark:bg-pink-900/40 dark:text-pink-300 hover:bg-pink-500/20 dark:hover:bg-pink-700/60 transition-colors"
              >
                {'{·}'} {t('contentSection.expand')}
              </button>
            )}
            {autoExpandBtn}
          </div>
        </div>
        <textarea
          ref={styleRef}
          className="w-full px-3 py-2.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 dark:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none resize-none transition-colors"
          placeholder="Dreamy indie folk, warm acoustic guitar, soft female vocals, intricate fingerpicking..."
          value={caption}
          onChange={e => onCaptionChange(e.target.value)}
          rows={3}
        />
      </div>

      {/* LoRA trigger word + Beat intro/outro — compose-time caption helpers */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 text-[10px] text-zinc-500 shrink-0 w-20">
            <Plug size={11} /> {t('contentSection.loraTrigger')}
          </span>
          <input
            type="text"
            value={loraTrigger}
            onChange={e => onLoraTriggerChange(e.target.value)}
            placeholder={t('contentSection.loraTriggerPlaceholder')}
            className="flex-1 px-2 py-1 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-xs text-zinc-700 dark:text-zinc-300 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 outline-none focus:border-pink-500/30 transition-colors"
          />
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onBeatIntroChange(!beatIntro)}
            title={t('contentSection.beatIOTooltip')}
            className={`flex items-center gap-1 text-[10px] px-2 py-0.5 rounded font-medium transition-colors shrink-0 border ${
              beatIntro
                ? 'bg-orange-500/15 text-orange-600 border-orange-500/40 dark:bg-orange-600/30 dark:text-orange-300 dark:border-orange-600/40'
                : 'text-zinc-500 border-zinc-300 dark:border-zinc-700 hover:bg-zinc-200 dark:hover:bg-zinc-800 hover:text-zinc-700 dark:hover:text-zinc-300'
            }`}
          >
            <Drum size={11} /> {t('contentSection.beatIO')}
          </button>
          {beatIntro && (
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-zinc-500 dark:text-zinc-600">{t('contentSection.bars')}:</span>
              {([1, 2, 4, 8] as const).map(n => (
                <button
                  key={n}
                  onClick={() => onIntroBarsChange(n)}
                  className={`text-[9px] w-5 h-4 rounded font-medium transition-colors ${
                    introBars === n
                      ? 'bg-orange-500 text-white'
                      : 'text-zinc-500 dark:text-zinc-600 hover:bg-zinc-200 dark:hover:bg-zinc-800'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Song Metadata (Title / Artist / Subject) — collapsible */}
      <div>
        <button
          onClick={() => setShowMetadata(!showMetadata)}
          className="flex items-center gap-1.5 text-xs font-medium text-zinc-500 uppercase tracking-wider hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors mb-1.5"
        >
          {showMetadata
            ? <ChevronDown size={12} className="text-zinc-500" />
            : <ChevronRight size={12} className="text-zinc-500" />}
          {t('contentSection.songInfo')}
          {hasMetadata && (
            <span className="text-[9px] text-pink-400/80 font-normal normal-case ml-1">{t('contentSection.populated')}</span>
          )}
        </button>

        {showMetadata && (
          <div className="space-y-2 pl-0.5">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] text-zinc-600 mb-0.5">{t('contentSection.artist')}</label>
                <input
                  type="text"
                  className="w-full px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 dark:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors"
                  placeholder={t('contentSection.artistPlaceholder')}
                  value={artist}
                  onChange={e => onArtistChange(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-[10px] text-zinc-600 mb-0.5">{t('contentSection.title')}</label>
                <input
                  type="text"
                  className="w-full px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 dark:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors"
                  placeholder={t('contentSection.titlePlaceholder')}
                  value={title}
                  onChange={e => onTitleChange(e.target.value)}
                />
              </div>
            </div>
            <div>
              <label className="block text-[10px] text-zinc-600 mb-0.5">{t('contentSection.subject')}</label>
              <input
                type="text"
                className="w-full px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 dark:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors"
                placeholder={t('contentSection.subjectPlaceholder')}
                value={subject}
                onChange={e => onSubjectChange(e.target.value)}
              />
            </div>
          </div>
        )}
      </div>

      {/* Instrumental toggle */}
      <label className="flex items-center gap-2.5 cursor-pointer group">
        <div className="relative">
          <input
            type="checkbox"
            checked={instrumental}
            onChange={e => onInstrumentalChange(e.target.checked)}
            className="sr-only peer"
          />
          <div className="w-8 h-4.5 bg-zinc-200 dark:bg-zinc-700 rounded-full peer-checked:bg-pink-500 transition-colors" />
          <div className="absolute top-0.5 left-0.5 w-3.5 h-3.5 bg-white rounded-full transition-transform peer-checked:translate-x-3.5" />
        </div>
        <div className="flex items-center gap-1.5">
          <Music size={14} className="text-zinc-500" />
          <span className="text-sm text-zinc-600 dark:text-zinc-400 group-hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors">
            {t('contentSection.instrumental')}
          </span>
        </div>
      </label>

      {/* Lyrics */}
      {!instrumental && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
              {t('contentSection.lyrics')}
            </label>
            <div className="flex items-center gap-1.5">
              {hasWildcards(lyrics) && (
                <button
                  onClick={() => expandField(lyricsRef, onLyricsChange)}
                  className="text-[9px] px-1.5 py-0.5 rounded font-mono bg-purple-500/10 text-purple-600 dark:bg-purple-900/40 dark:text-purple-300 hover:bg-purple-500/20 dark:hover:bg-purple-700/60 transition-colors"
                >
                  {'{·}'} {t('contentSection.expand')}
                </button>
              )}
              {autoExpandBtn}
            </div>
          </div>
          <textarea
            ref={lyricsRef}
            className="w-full px-3 py-2.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 dark:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none resize-vertical transition-colors font-mono leading-relaxed"
            placeholder={`[Verse 1]\nWalking through the morning light\nEvery shadow fading bright\n\n[Chorus]\nWe're alive, we're alive tonight...`}
            value={lyrics}
            onChange={e => onLyricsChange(e.target.value)}
            rows={8}
          />
        </div>
      )}

      {/* Negative Prompt */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          Negative Prompt
        </label>
        <textarea
          className="w-full px-3 py-2.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-red-500/50 focus:ring-1 focus:ring-red-500/20 outline-none resize-none transition-colors"
          placeholder="jazz, acoustic, slow, ambient, piano, soft, classical..."
          value={negativePrompt}
          onChange={e => onNegativePromptChange(e.target.value)}
          rows={2}
        />
      </div>
    </div>
  );
};
