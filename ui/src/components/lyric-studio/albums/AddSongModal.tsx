import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { X, FileText, Type } from 'lucide-react';

interface AddSongModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (title: string, lyrics: string) => Promise<void>;
  albumName?: string;
}

export const AddSongModal: React.FC<AddSongModalProps> = ({ isOpen, onClose, onSubmit, albumName }) => {
  const [title, setTitle] = useState('');
  const { t } = useTranslation();
  const [lyrics, setLyrics] = useState('');
  const [submitting, setSubmitting] = useState(false);

  React.useEffect(() => {
    if (isOpen) { setTitle(''); setLyrics(''); }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !lyrics.trim()) return;
    setSubmitting(true);
    try {
      await onSubmit(title.trim(), lyrics);
      onClose();
    } finally { setSubmitting(false); }
  };

  return (
    <div className="fixed inset-0 bg-black/30 dark:bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center">
              <FileText className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">{t('lyric.addSong')}</h2>
              {albumName && <p className="text-xs text-zinc-500">{albumName}</p>}
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Type className="w-3.5 h-3.5 text-zinc-600 dark:text-zinc-400" />
                {t('lyric.songTitle')}
              </span>
            </label>
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. My Song Title"
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-zinc-300 dark:border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 transition-all"
              autoFocus
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-zinc-600 dark:text-zinc-400" />
                {t('lyric.lyrics')}
              </span>
            </label>
            <textarea value={lyrics} onChange={(e) => setLyrics(e.target.value)}
              placeholder={"[Verse 1]\nYour lyrics here...\n\n[Chorus]\nChorus lyrics here..."}
              className="w-full h-80 px-4 py-3 rounded-xl bg-white/5 border border-zinc-300 dark:border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 transition-all resize-y font-mono text-sm leading-relaxed"
            />
            <p className="text-xs text-zinc-500 mt-1">
              Use [Verse], [Chorus], [Bridge] section headers for best profiling results
            </p>
          </div>
          <button type="submit" disabled={!title.trim() || !lyrics.trim() || submitting}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-200 dark:disabled:bg-zinc-200 dark:bg-zinc-700 disabled:text-zinc-500 text-white font-semibold transition-all"
          >
            {submitting ? t('lyric.adding') : t('lyric.addSong')}
          </button>
        </form>
      </div>
    </div>
  );
};
