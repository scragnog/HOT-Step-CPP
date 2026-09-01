// SendToLyricStudio.tsx — export a labeled dataset into Lyric Studio
//
// One card + one confirm modal. The server detects artist/album (audio-tag
// majority vote → dataset defaults → folder name); both fields are editable
// before committing. An existing artist+album set is updated in place, and the
// dataset's trained adapters plus one of its own tracks (the timbre /
// mastering reference) can be linked as the album preset.

import React, { useState } from 'react';
import { AlertTriangle, CheckCircle2, Loader2, Mic2, RefreshCw, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  exportToLyricStudio, getLyricStudioPreview,
  type LyricStudioExportPreview, type LyricStudioExportResult,
} from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500';

const lastFolder = (p: string) => p.split(/[\\/]/).filter(Boolean).slice(-2).join('/');
const fileName = (p: string) => p.split(/[\\/]/).filter(Boolean).pop() ?? p;

export const SendToLyricStudio: React.FC = () => {
  const { t } = useTranslation();
  const detail = useTrainingStore(s => s.detail);

  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<LyricStudioExportPreview | null>(null);
  const [artist, setArtist] = useState('');
  const [album, setAlbum] = useState('');
  const [linkAdapters, setLinkAdapters] = useState(true);
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState<LyricStudioExportResult | null>(null);
  const [error, setError] = useState('');

  if (!detail) return null;
  const datasetId = detail.id;

  const openModal = async () => {
    setOpen(true);
    setLoading(true);
    setPreview(null);
    setResult(null);
    setError('');
    try {
      const p = await getLyricStudioPreview(datasetId);
      setPreview(p);
      setArtist(p.artist);
      setAlbum(p.album);
      setLinkAdapters(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'preview failed');
    } finally {
      setLoading(false);
    }
  };

  const send = async () => {
    setSending(true);
    setError('');
    try {
      const r = await exportToLyricStudio(datasetId, {
        artist: artist.trim(),
        album: album.trim(),
        linkAdapters,
      });
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'export failed');
    } finally {
      setSending(false);
    }
  };

  const hasPresetAssets = !!(preview?.ditAdapter || preview?.lmAdapter || preview?.referenceTrack);
  const canSend = !!preview && preview.songs.length > 0 && !!artist.trim() && !!album.trim() && !sending;

  return (
    <>
      <div className={`${CARD} flex items-center gap-3`}>
        <Mic2 size={16} className="text-amber-500 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.lyricStudio.title')}</div>
          <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.lyricStudio.subtitle')}</p>
        </div>
        <button
          onClick={() => void openModal()}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 text-zinc-800 dark:text-zinc-200 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors flex-shrink-0"
        >
          {t('trainingStudio.lyricStudio.open')}
        </button>
      </div>

      {open && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-panel shadow-2xl flex flex-col max-h-[85vh]">
            {/* Header */}
            <div className="flex items-center gap-2 px-5 py-4 border-b border-zinc-200 dark:border-white/5">
              <Mic2 size={15} className="text-amber-500" />
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white flex-1">{t('trainingStudio.lyricStudio.title')}</h3>
              <button
                onClick={() => setOpen(false)}
                className="p-1 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              >
                <X size={15} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
              {loading && (
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <Loader2 size={13} className="animate-spin" /> {t('trainingStudio.lyricStudio.loading')}
                </div>
              )}

              {error && (
                <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{error}</div>
              )}

              {/* Success view */}
              {result && (
                <div className="flex flex-col gap-3">
                  <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-xs text-emerald-600 dark:text-emerald-400">
                    <CheckCircle2 size={14} className="mt-0.5 flex-shrink-0" />
                    <span>
                      {t(result.updatedExisting ? 'trainingStudio.lyricStudio.doneUpdated' : 'trainingStudio.lyricStudio.doneCreated',
                        { artist: artist.trim(), album: album.trim(), count: result.songCount })}
                      {result.presetUpdated && <> {t('trainingStudio.lyricStudio.donePreset')}</>}
                    </span>
                  </div>
                  {result.imageUrl && (
                    <img src={result.imageUrl} alt="" className="w-24 h-24 rounded-lg object-cover self-center" />
                  )}
                </div>
              )}

              {/* Form view */}
              {preview && !result && (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <label className="flex flex-col gap-1.5">
                      <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.lyricStudio.artist')}</span>
                      <input type="text" value={artist} onChange={e => setArtist(e.target.value)} className={INPUT} />
                      <span className="text-[10px] text-zinc-500">{t(`trainingStudio.lyricStudio.source.${preview.artistSource}`)}</span>
                    </label>
                    <label className="flex flex-col gap-1.5">
                      <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.lyricStudio.album')}</span>
                      <input type="text" value={album} onChange={e => setAlbum(e.target.value)} className={INPUT} />
                      <span className="text-[10px] text-zinc-500">{t(`trainingStudio.lyricStudio.source.${preview.albumSource}`)}</span>
                    </label>
                  </div>

                  {preview.songs.length === 0 ? (
                    <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
                      <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                      {t('trainingStudio.lyricStudio.noSongs')}
                    </div>
                  ) : (
                    <div className="text-xs text-zinc-600 dark:text-zinc-400">
                      {t('trainingStudio.lyricStudio.songs', { count: preview.songs.length })}
                      {(preview.skippedExcluded + preview.skippedInstrumental + preview.skippedNoLyrics) > 0 && (
                        <span className="block text-[11px] text-zinc-500 mt-0.5">
                          {t('trainingStudio.lyricStudio.skipped', {
                            excluded: preview.skippedExcluded,
                            instrumental: preview.skippedInstrumental,
                            noLyrics: preview.skippedNoLyrics,
                          })}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Song list */}
                  {preview.songs.length > 0 && (
                    <div className="rounded-lg border border-zinc-200 dark:border-white/5 max-h-40 overflow-y-auto divide-y divide-zinc-100 dark:divide-white/5">
                      {preview.songs.map(s => (
                        <div key={s.sampleId} className="px-3 py-1.5 text-xs text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
                          <span className="truncate flex-1">{s.title}</span>
                          {!s.hasCaption && (
                            <span className="text-[10px] text-zinc-400 dark:text-zinc-600 flex-shrink-0">{t('trainingStudio.grid.statusUnlabeled')}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Update-in-place notice (based on the detected names; the
                      server re-resolves after overrides) */}
                  {preview.existingLyricsSetId !== null ? (
                    <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
                      <RefreshCw size={13} className="mt-0.5 flex-shrink-0" />
                      {t('trainingStudio.lyricStudio.updateNotice', { count: preview.existingSongCount })}
                    </div>
                  ) : (
                    <div className="text-[11px] text-zinc-500">{t('trainingStudio.lyricStudio.createNotice')}</div>
                  )}

                  {/* Adapters → album preset */}
                  {hasPresetAssets ? (
                    <label className="flex items-start gap-2.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={linkAdapters}
                        onChange={e => setLinkAdapters(e.target.checked)}
                        className="mt-0.5 accent-amber-500"
                      />
                      <span className="text-xs text-zinc-700 dark:text-zinc-300">
                        <span className="font-semibold">{t('trainingStudio.lyricStudio.linkAdapters')}</span>
                        {preview.ditAdapter && (
                          <span className="block text-[11px] font-mono text-zinc-500 mt-0.5" title={preview.ditAdapter.path}>
                            {t('trainingStudio.lyricStudio.adapterDit', { name: lastFolder(preview.ditAdapter.path), detail: preview.ditAdapter.detail })}
                          </span>
                        )}
                        {preview.lmAdapter && (
                          <span className="block text-[11px] font-mono text-zinc-500 mt-0.5" title={preview.lmAdapter.path}>
                            {t('trainingStudio.lyricStudio.adapterLm', { name: lastFolder(preview.lmAdapter.path), detail: preview.lmAdapter.detail })}
                          </span>
                        )}
                        {preview.referenceTrack && (
                          <span className="block text-[11px] font-mono text-zinc-500 mt-0.5" title={preview.referenceTrack}>
                            {t('trainingStudio.lyricStudio.adapterReference', { name: fileName(preview.referenceTrack) })}
                          </span>
                        )}
                      </span>
                    </label>
                  ) : (
                    <div className="text-[11px] text-zinc-500">{t('trainingStudio.lyricStudio.noAdapters')}</div>
                  )}

                  {preview.geniusConfigured && (
                    <div className="text-[11px] text-zinc-500">{t('trainingStudio.lyricStudio.artHint')}</div>
                  )}
                </>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-zinc-200 dark:border-white/5">
              {result ? (
                <button
                  onClick={() => setOpen(false)}
                  className="px-4 py-2 rounded-xl text-xs font-bold bg-amber-500 text-black hover:bg-amber-400 transition-colors"
                >
                  {t('trainingStudio.lyricStudio.close')}
                </button>
              ) : (
                <>
                  <button
                    onClick={() => setOpen(false)}
                    className="px-4 py-2 rounded-xl text-xs font-semibold text-zinc-600 dark:text-zinc-400 hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
                  >
                    {t('trainingStudio.lyricStudio.cancel')}
                  </button>
                  <button
                    onClick={() => void send()}
                    disabled={!canSend}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    {sending && <Loader2 size={13} className="animate-spin" />}
                    {sending ? t('trainingStudio.lyricStudio.sending') : t('trainingStudio.lyricStudio.send')}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default SendToLyricStudio;
