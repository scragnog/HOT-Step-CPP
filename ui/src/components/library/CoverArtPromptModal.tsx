// CoverArtPromptModal.tsx — per-track cover art with a custom prompt (#67).
//
// Manual "Generate / Regenerate Cover Art" from a track's dropdown menu opens
// this modal instead of firing immediately. It pre-fills the textarea with the
// exact prompt the engine would auto-build (fetched from /prompt-preview), lets
// the user edit it freely, then POSTs the verbatim prompt to /cover-art/generate
// and polls to completion — dispatching `cover-art-updated` so App.tsx refreshes
// the song. Auto-generate-after-creation is unaffected (it never opens this).
//
// Mounted once at the SongList root; driven by the `open-cover-art-prompt`
// window CustomEvent dispatched from the context-menu items.

import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { X, Image as ImageIcon, Loader2, Sparkles, RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song } from '../../types';

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

/** Build the metadata payload used for both preview and generation. */
function songPromptInputs(song: Song) {
  const params: any = song.generationParams || (song as any).generation_params || {};
  return {
    title: song.title || '',
    style: song.style || params?.style || '',
    lyrics: song.lyrics || params?.lyrics || '',
    subject: (song as any).cover_art_subject || params?.coverArtSubject || params?.subject || '',
  };
}

export const CoverArtPromptModal: React.FC = () => {
  const { t } = useTranslation();
  const [song, setSong] = useState<Song | null>(null);
  const [prompt, setPrompt] = useState('');
  const [defaultPrompt, setDefaultPrompt] = useState('');
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const close = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    setSong(null);
    setPrompt('');
    setDefaultPrompt('');
    setError('');
    setGenerating(false);
  }, []);

  // Listen for open requests
  useEffect(() => {
    const onOpen = (e: Event) => {
      const detail = (e as CustomEvent<OpenCoverArtPromptDetail>).detail;
      if (!detail?.song) return;
      setSong(detail.song);
      setError('');
      setGenerating(false);
      // Fetch the auto-assembled prompt to pre-fill
      setLoadingPreview(true);
      setPrompt('');
      fetch('/api/cover-art/prompt-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(songPromptInputs(detail.song)),
      })
        .then(r => (r.ok ? r.json() : Promise.reject()))
        .then(d => { setPrompt(d.prompt || ''); setDefaultPrompt(d.prompt || ''); })
        .catch(() => { setPrompt(''); setDefaultPrompt(''); })
        .finally(() => setLoadingPreview(false));
    };
    window.addEventListener('open-cover-art-prompt', onOpen);
    return () => window.removeEventListener('open-cover-art-prompt', onOpen);
  }, []);

  // Cleanup poll on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const generate = useCallback(() => {
    if (!song || !prompt.trim()) return;
    setGenerating(true);
    setError('');
    fetch('/api/cover-art/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        songId: song.id,
        ...songPromptInputs(song),
        prompt: prompt.trim(),
      }),
    })
      .then(async r => {
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          // 503 means the weights are absent, not that anything went wrong.
          // "Cover art not installed" on its own sent people hunting through
          // Settings and the Model Manager for a download that lives in the
          // Post-Processing menu instead (issue #108), so say where it is and
          // which files are missing.
          if (r.status === 503) {
            const missing: string[] = Array.isArray(d.missingFiles) ? d.missingFiles : [];
            throw new Error(
              'Cover art models are not downloaded yet. Open the Cover Art section '
              + 'of the Post-Processing menu in the top bar and download a model there.'
              + (missing.length ? ` Missing: ${missing.join(', ')}.` : '')
            );
          }
          throw new Error(d.error || 'Failed to start generation');
        }
        const { jobId } = await r.json();
        pollRef.current = setInterval(async () => {
          try {
            const jr = await fetch(`/api/cover-art/generate/${jobId}`);
            if (!jr.ok) return;
            const job = await jr.json();
            if (job.status === 'succeeded') {
              if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
              window.dispatchEvent(new CustomEvent('cover-art-updated', {
                detail: { songId: song.id, coverUrl: job.result?.coverUrl },
              }));
              close();
            } else if (job.status === 'failed') {
              if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
              setError(job.error || 'Generation failed');
              setGenerating(false);
            }
          } catch { /* network blip — keep polling */ }
        }, 2000);
        // Safety stop after 5 minutes
        setTimeout(() => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; setGenerating(false); } }, 300_000);
      })
      .catch(err => { setError(err.message); setGenerating(false); });
  }, [song, prompt, close]);

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
