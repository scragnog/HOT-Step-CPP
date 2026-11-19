import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { X, User, Link2 } from 'lucide-react';

interface AddArtistModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (name: string, imageUrl?: string) => Promise<void>;
}

export const AddArtistModal: React.FC<AddArtistModalProps> = ({ isOpen, onClose, onSubmit }) => {
  const [name, setName] = useState('');
  const { t } = useTranslation();
  const [imageUrl, setImageUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    try {
      await onSubmit(name.trim(), imageUrl.trim() || undefined);
      setName('');
      setImageUrl('');
      onClose();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 dark:bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <User className="w-4 h-4 text-emerald-400" />
            </div>
            <h2 className="text-lg font-bold text-white">{t('lyric.addArtist')}</h2>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <User className="w-3.5 h-3.5 text-zinc-600 dark:text-zinc-400" />
                {t('lyric.artistName')}
              </span>
            </label>
            <input
              type="text" value={name} onChange={(e) => setName(e.target.value)}
              placeholder="e.g. My Custom Artist"
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-zinc-300 dark:border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 transition-all"
              autoFocus
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Link2 className="w-3.5 h-3.5 text-zinc-600 dark:text-zinc-400" />
                {t('lyric.imageUrl')} <span className="text-zinc-500 font-normal">{t('lyric.optional')}</span>
              </span>
            </label>
            <input
              type="text" value={imageUrl} onChange={(e) => setImageUrl(e.target.value)}
              placeholder="https://example.com/artist.jpg"
              className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-zinc-300 dark:border-white/10 text-white placeholder:text-zinc-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 transition-all"
            />
          </div>
          <button type="submit" disabled={!name.trim() || submitting}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-200 dark:disabled:bg-zinc-200 dark:bg-zinc-700 disabled:text-zinc-500 text-white font-semibold transition-all"
          >
            {submitting ? t('lyric.adding') : t('lyric.addArtist')}
          </button>
        </form>
      </div>
    </div>
  );
};
