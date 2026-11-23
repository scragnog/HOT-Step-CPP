// MetadataEditorModal.tsx — edit a track's embed metadata + cover (#60).
//
// Shows the auto-populated metadata and lets the user change it. Title / genre /
// bpm / key / lyrics update the song columns (so the library reflects them);
// artist / album / year / comment are stored as verbatim embed overrides. The
// cover can be replaced with an uploaded image. All of it is embedded into
// exported audio files on download (handled server-side by gatherSongMetadata).

import { artworkSrc } from '../../utils/artworkSrc';
import React, { useState, useRef } from 'react';
import ReactDOM from 'react-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { X, Image as ImageIcon, Loader2, Save } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song } from '@/types';
import { songApi } from '@/services/api';
import { events, useUpload, writeAndInvalidate } from '@/queries/writes';

interface MetadataEditorModalProps {
  song: Song;
  onClose: () => void;
  onSaved: (song: Song) => void;
}

const Field: React.FC<{ label: string; value: string; onChange: (v: string) => void; type?: string }> = ({
  label, value, onChange, type = 'text',
}) => (
  <div>
    <label className="block text-xs font-medium text-zinc-500 mb-1">{label}</label>
    <input
      type={type}
      value={value}
      onChange={e => onChange(e.target.value)}
      className="w-full text-sm rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-white/10 px-3 py-2 text-zinc-800 dark:text-zinc-200 outline-none focus:border-cyan-500"
    />
  </div>
);

export const MetadataEditorModal: React.FC<MetadataEditorModalProps> = ({ song, onClose, onSaved }) => {
  const { t } = useTranslation();
  const gp: any = song.generationParams || {};
  const overrides: any = (() => {
    try { return song.metadata_overrides ? JSON.parse(song.metadata_overrides) : {}; } catch { return {}; }
  })();
  const initialYear = (() => {
    try { return song.created_at ? String(new Date(song.created_at).getFullYear()) : ''; } catch { return ''; }
  })();

  const [title, setTitle] = useState(song.title || '');
  const [artist, setArtist] = useState<string>(overrides.artist ?? gp.artist ?? gp.artistName ?? '');
  const [album, setAlbum] = useState<string>(overrides.album ?? gp.album ?? '');
  const [genre, setGenre] = useState(song.caption || gp.caption || '');
  const [year, setYear] = useState<string>(overrides.year ?? initialYear);
  const [comment, setComment] = useState<string>(overrides.comment ?? '');
  const [bpm, setBpm] = useState(song.bpm ? String(song.bpm) : (gp.bpm ? String(gp.bpm) : ''));
  const [key, setKey] = useState(song.key_scale || gp.keyScale || '');
  const [lyrics, setLyrics] = useState(song.lyrics || '');
  const [coverUrl, setCoverUrl] = useState(song.coverUrl || song.cover_url || '');

  const queryClient = useQueryClient();
  const { mutate: uploadCover, isPending: uploading } = useUpload<{ cover_url?: string }>();
  const fileRef = useRef<HTMLInputElement>(null);

  // Save and the cover upload both fail through the transport's one notifier
  // (the interceptor toasts command failures), so the modal keeps no error
  // state of its own — an inline copy of the same message was a double toast.
  const saveMut = useMutation({
    mutationFn: async () => {
      const payload: any = {
        title,
        caption: genre,
        bpm: bpm ? (parseInt(bpm, 10) || 0) : 0,
        key_scale: key,
        lyrics,
        cover_url: coverUrl,
        metadata_overrides: {
          artist: artist.trim() || undefined,
          album: album.trim() || undefined,
          year: year.trim() || undefined,
          comment: comment.trim() || undefined,
        },
      };
      return writeAndInvalidate(queryClient, () => songApi.update(song.id, payload), events('song:written'));
    },
    onSuccess: ({ song: updated }) => { onSaved(updated); onClose(); },
  });
  // Named zero-arg handler so the button can pass it straight to onClick —
  // mutate itself takes `void` variables, which a click event is not.
  const save = () => saveMut.mutate();

  const handleCoverFile = (file: File) => {
    uploadCover({
      path: '/upload/cover-image',
      files: [{ field: 'image', file }],
      onSuccess: d => { if (d.cover_url) setCoverUrl(d.cover_url); },
    });
  };

  return ReactDOM.createPortal(
    <div
      className="fixed inset-0 z-[150] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg max-h-[88vh] overflow-y-auto rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-white/10">
          <h2 className="text-base font-bold text-zinc-900 dark:text-white">{t('metadata.editTitle', 'Edit Metadata')}</h2>
          <button onClick={onClose} className="p-1 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10">
            <X size={18} />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Cover */}
          <div className="flex items-center gap-4">
            <div className="relative w-24 h-24 rounded-xl overflow-hidden bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center flex-shrink-0">
              {coverUrl
                ? <img src={artworkSrc(coverUrl)} alt="" className="w-full h-full object-cover" />
                : <ImageIcon size={28} className="text-zinc-400" />}
              {uploading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                  <Loader2 size={20} className="animate-spin text-white" />
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) handleCoverFile(f); e.currentTarget.value = ''; }}
              />
              <button
                onClick={() => fileRef.current?.click()}
                disabled={uploading}
                className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/15 text-cyan-400 hover:bg-cyan-500/25 transition-colors disabled:opacity-50"
              >
                {t('metadata.replaceCover', 'Replace cover image')}
              </button>
              <p className="text-[11px] text-zinc-500 mt-1.5 leading-relaxed">
                {t('metadata.coverHint', 'PNG/JPG. Embedded into MP3/FLAC downloads.')}
              </p>
            </div>
          </div>

          <Field label={t('metadata.title', 'Title')} value={title} onChange={setTitle} />
          <Field label={t('metadata.artist', 'Artist')} value={artist} onChange={setArtist} />
          <div className="grid grid-cols-2 gap-3">
            <Field label={t('metadata.album', 'Album')} value={album} onChange={setAlbum} />
            <Field label={t('metadata.year', 'Year')} value={year} onChange={setYear} />
          </div>
          <Field label={t('metadata.genre', 'Genre / Style')} value={genre} onChange={setGenre} />
          <div className="grid grid-cols-2 gap-3">
            <Field label={t('metadata.bpm', 'BPM')} value={bpm} onChange={setBpm} type="number" />
            <Field label={t('metadata.key', 'Key')} value={key} onChange={setKey} />
          </div>
          <Field label={t('metadata.comment', 'Comment')} value={comment} onChange={setComment} />
          <div>
            <label className="block text-xs font-medium text-zinc-500 mb-1">{t('metadata.lyrics', 'Lyrics')}</label>
            <textarea
              value={lyrics}
              onChange={e => setLyrics(e.target.value)}
              rows={5}
              className="w-full text-sm rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-white/10 px-3 py-2 text-zinc-800 dark:text-zinc-200 outline-none focus:border-cyan-500 resize-y"
            />
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-zinc-200 dark:border-white/10">
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
            {t('common.cancel', 'Cancel')}
          </button>
          <button
            onClick={save}
            disabled={saveMut.isPending || uploading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-cyan-500 hover:bg-cyan-400 text-white transition-colors disabled:opacity-50"
          >
            {saveMut.isPending ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
            {t('common.save', 'Save')}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
};
