// CoverArtPromptModal.tsx — per-track cover art with a custom prompt (#67).
//
// Manual "Generate / Regenerate Cover Art" from a track's dropdown menu opens
// this modal instead of firing immediately. It pre-fills the textarea with the
// exact prompt the engine would auto-build (fetched from /prompt-preview), lets
// the user edit it freely, then POSTs the verbatim prompt to /cover-art/generate
// and follows the job with a Query poll — dispatching `cover-art-updated` so
// App.tsx refreshes the song. Auto-generate-after-creation is unaffected (it never opens this).
//
// Mounted once at the SongList root; driven by the `open-cover-art-prompt`
// window CustomEvent dispatched from the context-menu items.

import React, { useState, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { QueryObserver, useQueryClient } from '@tanstack/react-query';
import { X, Image as ImageIcon, Loader2, Sparkles, RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song } from '@/types';
import { apiClient } from '@/services/httpClient';
import { useCoverArtGeneration, songPromptInputs } from '@/hooks/useCoverArtGeneration';
import { pollWhile } from '@/queries/contract';
import { coverArtJobOptions, type CoverArtJob } from '@/queries/shell-queue/queue';

const coverArtEnded = (job: CoverArtJob | undefined) => job?.status === 'succeeded' || job?.status === 'failed';
/** Give up following a job after this long; the server keeps working. */
const COVER_ART_FOLLOW_MS = 300_000;

/** Detail payload for the `open-cover-art-prompt` window event. */
export interface OpenCoverArtPromptDetail {
  song: Song;
}

/** Open the prompt modal for a song from anywhere in the tree. */
export function openCoverArtPrompt(song: Song): void {
  window.dispatchEvent(new CustomEvent<OpenCoverArtPromptDetail>('open-cover-art-prompt', {
    detail: { song },
  }));
}

export const CoverArtPromptModal: React.FC = () => {
  const { t } = useTranslation();
  const [song, setSong] = useState<Song | null>(null);
  const [prompt, setPrompt] = useState('');
  const [defaultPrompt, setDefaultPrompt] = useState('');
  const [loadingPreview, setLoadingPreview] = useState(false);
  const { jobId, error, generating, generate, clearError, resetJob, fail } =
    useCoverArtGeneration(song, prompt);
  const queryClient = useQueryClient();

  const close = useCallback(() => {
    resetJob();
    clearError();
    setSong(null);
    setPrompt('');
    setDefaultPrompt('');
  }, [resetJob, clearError]);

  // Listen for open requests
  useEffect(() => {
    const onOpen = async (e: Event) => {
      const detail = (e as CustomEvent<OpenCoverArtPromptDetail>).detail;
      if (!detail?.song) return;
      setSong(detail.song);
      clearError();
      // One-shot read that seeds the textarea — never cached, so it goes
      // straight through the transport (the interceptor reports a failed
      // preview; the modal opens with an empty prompt either way).
      setLoadingPreview(true);
      setPrompt('');
      try {
        const d = await apiClient.post<{ prompt?: string }>('/cover-art/prompt-preview', songPromptInputs(detail.song));
        setPrompt(d.prompt || '');
        setDefaultPrompt(d.prompt || '');
      } catch {
        setPrompt('');
        setDefaultPrompt('');
      } finally {
        setLoadingPreview(false);
      }
    };
    window.addEventListener('open-cover-art-prompt', onOpen);
    return () => window.removeEventListener('open-cover-art-prompt', onOpen);
  }, [clearError]);

  // Follow the job: Query polls it, one request at a time, and stops at a
  // terminal status. Errors and non-OK replies keep polling (with backoff)
  // until the job ends or the safety stop fires. Unsubscribing on the first
  // terminal answer acts on it once.
  useEffect(() => {
    if (!jobId || !song) return;
    const observer = new QueryObserver(queryClient, {
      ...coverArtJobOptions(jobId),
      refetchInterval: pollWhile<CoverArtJob>(j => !coverArtEnded(j), 2000),
    });
    const timer = setTimeout(() => resetJob(), COVER_ART_FOLLOW_MS);
    const unsubscribe = observer.subscribe(({ data }) => {
      if (!coverArtEnded(data)) return;
      unsubscribe();
      if (data!.status === 'succeeded') {
        window.dispatchEvent(new CustomEvent('cover-art-updated', {
          detail: { songId: song.id, coverUrl: data!.result?.coverUrl },
        }));
        close();
      } else {
        fail(data!.error || 'Generation failed');
      }
    });
    return () => { unsubscribe(); clearTimeout(timer); };
  }, [jobId, song, queryClient, close, resetJob, fail]);

  if (!song) return null;

  const hasCover = !!(song.coverUrl || (song as any).cover_url);

  return ReactDOM.createPortal(
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={generating ? undefined : close}
    >
      <div
        className="w-full max-w-lg rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-2 px-5 py-4 border-b border-zinc-200 dark:border-white/10">
          <ImageIcon size={16} className="text-pink-400 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
              {hasCover
                ? t('coverArt.regenerateTitle', 'Regenerate Cover Art')
                : t('coverArt.generateTitle', 'Generate Cover Art')}
            </h3>
            <p className="text-xs text-zinc-500 truncate">{song.title || t('library.untitled', 'Untitled')}</p>
          </div>
          <button
            onClick={close}
            disabled={generating}
            className="p-1 rounded text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors disabled:opacity-40"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-zinc-500">
              {t('coverArt.promptLabel', 'Image prompt')}
            </label>
            {prompt !== defaultPrompt && !!defaultPrompt && (
              <button
                onClick={() => setPrompt(defaultPrompt)}
                disabled={generating}
                className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-pink-400 transition-colors disabled:opacity-40"
                title={t('coverArt.promptReset', 'Reset to auto-generated prompt')}
              >
                <RotateCcw size={10} />
                {t('coverArt.promptReset', 'Reset')}
              </button>
            )}
          </div>

          {loadingPreview ? (
            <div className="flex items-center gap-2 text-xs text-zinc-500 py-6 justify-center">
              <Loader2 size={14} className="animate-spin" />
              {t('coverArt.promptLoading', 'Preparing prompt…')}
            </div>
          ) : (
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              disabled={generating}
              rows={6}
              autoFocus
              placeholder={t('coverArt.promptPlaceholder', 'Describe the artwork you want…')}
              className="w-full px-3 py-2 rounded-lg bg-zinc-100 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/10
                         text-sm text-zinc-800 dark:text-zinc-200 placeholder-zinc-500
                         focus:border-pink-500/40 focus:ring-1 focus:ring-pink-500/20
                         outline-none transition-colors resize-none disabled:opacity-60"
            />
          )}

          <p className="text-[10px] text-zinc-500 leading-relaxed">
            {t('coverArt.promptHelp', 'Edit the prompt to steer the artwork. 1024×1024 via FLUX.2-klein-4B. Text/lettering is automatically suppressed.')}
          </p>

          {error && (
            <div className="text-xs text-red-400 bg-red-500/5 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-zinc-200 dark:border-white/10">
          <button
            onClick={close}
            disabled={generating}
            className="px-3 py-2 text-xs font-medium rounded-lg text-zinc-600 dark:text-zinc-400 hover:bg-black/5 dark:hover:bg-white/5 transition-colors disabled:opacity-40"
          >
            {t('common.cancel', 'Cancel')}
          </button>
          <button
            onClick={generate}
            disabled={generating || loadingPreview || !prompt.trim()}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg
                       bg-gradient-to-r from-pink-500 to-purple-500 text-white
                       hover:from-pink-400 hover:to-purple-400 transition-all
                       disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {generating
              ? <><Loader2 size={14} className="animate-spin" /> {t('coverArt.generating', 'Generating…')}</>
              : <><Sparkles size={14} /> {t('coverArt.generate', 'Generate')}</>}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
};
