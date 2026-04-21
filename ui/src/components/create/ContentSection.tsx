// ContentSection.tsx — Caption + Lyrics input area
// Ported to Tailwind styling matching hot-step-9000.

import React from 'react';
import { Music } from 'lucide-react';
import { useLanguage } from '../../context/LanguageContext';
import { ToggleSwitch } from '../shared/ToggleSwitch';

interface ContentSectionProps {
  title: string;
  onTitleChange: (v: string) => void;
  caption: string;
  onCaptionChange: (v: string) => void;
  lyrics: string;
  onLyricsChange: (v: string) => void;
  instrumental: boolean;
  onInstrumentalChange: (v: boolean) => void;
}

export const ContentSection: React.FC<ContentSectionProps> = ({
  title, onTitleChange, caption, onCaptionChange, lyrics, onLyricsChange,
  instrumental, onInstrumentalChange,
}) => {
  const { t } = useLanguage();
  return (
    <div className="space-y-4 px-3 pb-3">
      {/* Song Name */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          {t('create_song_title_label')}
        </label>
        <input
          type="text"
          className="w-full px-3 py-2.5 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors"
          placeholder={t('create_song_title_placeholder')}
          value={title}
          onChange={e => onTitleChange(e.target.value)}
        />
      </div>

      {/* Style / Caption */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          {t('create_description_label')}
        </label>
        <textarea
          className="w-full px-3 py-2.5 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none resize-vertical transition-colors"
          placeholder={t('create_description_placeholder')}
          value={caption}
          onChange={e => onCaptionChange(e.target.value)}
          rows={3}
        />
      </div>

      {/* Instrumental toggle */}
      <ToggleSwitch
        checked={instrumental}
        onChange={onInstrumentalChange}
        label={t('create_instrumental')}
        icon={<Music size={14} />}
      />

      {/* Lyrics */}
      {!instrumental && (
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
            {t('create_lyrics_label')}
          </label>
          <textarea
            className="w-full px-3 py-2.5 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none resize-vertical transition-colors font-mono leading-relaxed"
            placeholder={t('create_lyrics_placeholder')}
            value={lyrics}
            onChange={e => onLyricsChange(e.target.value)}
            rows={8}
          />
        </div>
      )}
    </div>
  );
};
