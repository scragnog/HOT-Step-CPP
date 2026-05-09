import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { X, Search, Music, Disc3 } from 'lucide-react';

interface FetchLyricsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onFetch: (artist: string, album: string, maxSongs: number) => Promise<void>;
  prefillArtist?: string;
}

export const FetchLyricsModal: React.FC<FetchLyricsModalProps> = ({
  isOpen, onClose, onFetch, prefillArtist,
}) => {
  const [artist, setArtist] = useState(prefillArtist || '');
  const { t } = useTranslation();
  const [album, setAlbum] = useState('');
  const [maxSongs, setMaxSongs] = useState(50);

  // Reset form when modal opens with a new prefill
  React.useEffect(() => {
    if (isOpen) {
      setArtist(prefillArtist || '');
      setAlbum('');
    }
  }, [isOpen, prefillArtist]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!artist.trim()) return;
    // Close immediately — fetch runs in background with toast notification
    onClose();
    onFetch(artist.trim(), album.trim(), maxSongs);
  };

  return (
    <div className="fixed inset-0 bg-black/30 dark:bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-pink-500/10 flex items-center justify-center">
              <Search className="w-4 h-4 text-pink-400" />
            </div>
            <h2 className="text-lg font-bold text-white">{t('lyric.fetchLyrics')}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Music className="w-3.5 h-3.5 text-zinc-600 dark:text-zinc-400" />
                {t('lyric.artistName')}
              </span>
            </label>
            <input
              type="text"
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              placeholder="e.g. Steel Panther"
              disabled={!!prefillArtist}
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-zinc-300 dark:border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all disabled:opacity-50"
              autoFocus={!prefillArtist}
            />
            <p className="text-xs text-zinc-500 mt-1">
              Supports artist names or Genius URLs
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Disc3 className="w-3.5 h-3.5 text-zinc-600 dark:text-zinc-400" />
                {t('lyric.albumName')}
              </span>
            </label>
            <input
              type="text"
              value={album}
              onChange={(e) => setAlbum(e.target.value)}
              placeholder="e.g. Feel the Steel (or leave empty for top songs)"
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-zinc-300 dark:border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 transition-all disabled:opacity-50"
              autoFocus={!!prefillArtist}
            />
            <p className="text-xs text-zinc-500 mt-1">
              Also supports Genius album URLs
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1.5">
              {t('lyric.maxSongs')}
            </label>
            <input
              type="number"
              value={maxSongs}
              onChange={(e) => setMaxSongs(Math.max(1, Math.min(100, parseInt(e.target.value) || 50)))}
              min={1}
              max={100}
              className="w-24 px-3 py-2 rounded-xl bg-white/5 border border-zinc-300 dark:border-white/10 text-white text-center font-mono focus:outline-none focus:border-pink-500/50 transition-all disabled:opacity-50"
            />
          </div>

          <button
            type="submit"
            disabled={!artist.trim()}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-pink-600 hover:bg-pink-500 disabled:bg-zinc-200 dark:bg-zinc-700 disabled:text-zinc-500 text-white font-semibold transition-all"
          >
            <Search className="w-4 h-4" />
            {t('lyric.fetchLyrics')}
          </button>
        </form>
      </div>
    </div>
  );
};
