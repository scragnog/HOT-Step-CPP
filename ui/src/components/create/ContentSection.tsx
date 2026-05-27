// ContentSection.tsx — Caption + Lyrics input area with optional metadata fields
// Ported to Tailwind styling matching hot-step-9000.

import React from 'react';
import { Music, ChevronDown, ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';

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
}

export const ContentSection: React.FC<ContentSectionProps> = ({
  caption, onCaptionChange, lyrics, onLyricsChange,
  instrumental, onInstrumentalChange,
  title, onTitleChange, artist, onArtistChange, subject, onSubjectChange,
  negativePrompt, onNegativePromptChange,
}) => {
  const { t } = useTranslation();
  const hasMetadata = !!(title || artist || subject);
  const [showMetadata, setShowMetadata] = React.useState(hasMetadata);

  // Auto-expand when metadata is populated (e.g. from Send to Create)
  React.useEffect(() => {
    if (hasMetadata && !showMetadata) setShowMetadata(true);
  }, [hasMetadata]);

  return (
    <div className="space-y-3">
      {/* Style / Caption */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          {t('contentSection.styleDescription')}
        </label>
        <textarea
          className="w-full px-3 py-2.5 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 dark:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none resize-none transition-colors"
          placeholder="Dreamy indie folk, warm acoustic guitar, soft female vocals, intricate fingerpicking..."
          value={caption}
          onChange={e => onCaptionChange(e.target.value)}
          rows={3}
        />
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
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
            {t('contentSection.lyrics')}
          </label>
          <textarea
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
