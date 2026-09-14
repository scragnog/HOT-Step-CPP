// ImportTracksModal.tsx — bring audio you already have into the library.
//
// The post-processing chain only runs on a song row backed by a raw WAV, so a
// FLAC sitting in a folder had no way of reaching it. This is that way in:
// pick files, they are converted into the shape a render has and inserted as
// songs. Nothing here involves the generation engine — an imported track is
// just a track, and everything the library offers (post-processing, stems,
// export, playlists) works on it.
//
// Mounted once at the App root; opened from anywhere with openImportTracks().

import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactDOM from 'react-dom';
import { X, Upload, Loader2, FileAudio, AlertTriangle, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { songApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { startPostProcessing } from '../../stores/postProcessStore';

/** Mirrors IMPORT_EXTENSIONS in server/src/services/library/importTrack.ts —
 *  the server is the authority and re-checks every upload. */
const ACCEPTED = ['.wav', '.mp3', '.flac', '.m4a', '.mp4', '.aac', '.ogg', '.opus', '.webm', '.aiff', '.aif'];

/** Whether to run post-processing on what was imported. Remembered because it
 *  is a per-user habit, not a per-import decision. */
const PP_PREF_KEY = 'hs-import-runPostProcess';

/** Open the import modal from anywhere in the tree. */
export function openImportTracks(): void {
  window.dispatchEvent(new CustomEvent('open-import-tracks'));
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function hasAcceptedExtension(name: string): boolean {
  const dot = name.lastIndexOf('.');
  return dot >= 0 && ACCEPTED.includes(name.slice(dot).toLowerCase());
}

type Phase = 'picking' | 'uploading' | 'converting' | 'done';

export const ImportTracksModal: React.FC = () => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [phase, setPhase] = useState<Phase>('picking');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [failures, setFailures] = useState<{ file: string; error: string }[]>([]);
  const [importedCount, setImportedCount] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [description, setDescription] = useState('');
  const [runPp, setRunPp] = useState(() => {
    try { return localStorage.getItem(PP_PREF_KEY) === '1'; } catch { return false; }
  });
  const inputRef = useRef<HTMLInputElement | null>(null);

  const busy = phase === 'uploading' || phase === 'converting';

  const close = useCallback(() => {
    setOpen(false);
    setFiles([]);
    setPhase('picking');
    setProgress(0);
    setError('');
    setFailures([]);
    setImportedCount(0);
    setDragging(false);
  }, []);

  useEffect(() => {
    const onOpen = () => {
      setOpen(true);
      setPhase('picking');
      setFiles([]);
      setError('');
      setFailures([]);
      setImportedCount(0);
      setProgress(0);
    };
    window.addEventListener('open-import-tracks', onOpen);
    return () => window.removeEventListener('open-import-tracks', onOpen);
  }, []);

  // Escape closes, but never mid-import: the upload would carry on invisibly.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) close(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, busy, close]);

  const addFiles = useCallback((incoming: FileList | File[] | null) => {
    if (!incoming) return;
    const picked = Array.from(incoming).filter(f => hasAcceptedExtension(f.name));
    const rejected = Array.from(incoming).length - picked.length;
    setError(rejected > 0
      ? t('library.importSkippedTypes', {
          defaultValue: '{{n}} file(s) skipped — not an audio type this can read.',
          n: rejected,
        })
      : '');
    setFiles(prev => {
      // Same name and size twice over is the same file picked twice.
      const seen = new Set(prev.map(f => `${f.name}:${f.size}`));
      return [...prev, ...picked.filter(f => !seen.has(`${f.name}:${f.size}`))];
    });
  }, [t]);

  // StableStep refines towards a text description. A generated track gets one
  // from the prompt it came from; an imported one has only this field and the
  // file's own genre tag, and with neither the server refuses that stage
  // rather than refining towards a generic "Instrumental track".
  const ppNeedsDescription = runPp && !description.trim();

  const doImport = useCallback(async () => {
    if (!token || files.length === 0) return;
    setError('');
    setFailures([]);
    setPhase('uploading');
    setProgress(0);

    try {
      const { songs, errors } = await songApi.importTracks(files, token, {
        description,
        onProgress: (fraction) => {
          setProgress(fraction);
          // The upload finishing is the server starting work, and conversion is
          // the part with no progress to report.
          if (fraction >= 1) setPhase('converting');
        },
      });

      // App keeps the library list; it already listens for this.
      for (const song of songs) {
        window.dispatchEvent(new CustomEvent('song-created', { detail: { song } }));
      }

      if (runPp) {
        // Sequential on purpose: the server runs one pass at a time on the GPU
        // lane anyway, and starting them in order keeps the activity dock
        // readable. A refusal shows up there per track, so it is not swallowed.
        for (const song of songs) {
          try { await startPostProcessing(song); } catch { /* reported on the dock */ }
        }
      }

      setImportedCount(songs.length);
      setFailures(errors);
      if (errors.length === 0) { close(); return; }
      setPhase('done');
    } catch (err: any) {
      setError(err.message || 'Import failed');
      setPhase('picking');
    }
  }, [token, files, runPp, description, close]);

  if (!open) return null;

  const totalBytes = files.reduce((sum, f) => sum + f.size, 0);

  return ReactDOM.createPortal(
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={busy ? undefined : close}
    >
      <div
        className="w-full max-w-lg rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-2 px-5 py-4 border-b border-zinc-200 dark:border-white/10">
          <Upload size={16} className="text-pink-400 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
              {t('library.importTitle', 'Upload into Library')}
            </h3>
            <p className="text-xs text-zinc-500">
              {t('library.importSubtitle', 'Add tracks you already have — no generation involved.')}
            </p>
          </div>
          <button
            onClick={close}
            disabled={busy}
            className="p-1 rounded text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors disabled:opacity-40"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-3">
          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => {
              e.preventDefault();
              setDragging(false);
              if (!busy) addFiles(e.dataTransfer?.files || null);
            }}
            onClick={() => { if (!busy) inputRef.current?.click(); }}
            className={`
              rounded-xl border border-dashed px-4 py-6 text-center cursor-pointer transition-colors
              ${dragging
                ? 'border-pink-400 bg-pink-500/10'
                : 'border-zinc-300 dark:border-white/15 hover:border-pink-400/60 hover:bg-black/5 dark:hover:bg-white/5'}
              ${busy ? 'opacity-50 pointer-events-none' : ''}
            `}
          >
            <FileAudio size={22} className="mx-auto mb-2 text-zinc-400" />
            <div className="text-sm text-zinc-700 dark:text-zinc-300">
              {t('library.importDrop', 'Drop audio files here, or click to choose')}
            </div>
            <div className="text-[11px] text-zinc-500 mt-1">{ACCEPTED.join('  ')}</div>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={ACCEPTED.join(',')}
              className="hidden"
              onChange={e => { addFiles(e.target.files); e.target.value = ''; }}
            />
          </div>

          {/* Picked files */}
          {files.length > 0 && (
            <div className="max-h-44 overflow-y-auto rounded-xl border border-zinc-200 dark:border-white/10 divide-y divide-zinc-200 dark:divide-white/5">
              {files.map((f, i) => (
                <div key={`${f.name}:${f.size}:${i}`} className="flex items-center gap-2 px-3 py-2">
                  <span className="flex-1 min-w-0 truncate text-xs text-zinc-700 dark:text-zinc-300">{f.name}</span>
                  <span className="text-[10px] text-zinc-500 flex-shrink-0">{formatSize(f.size)}</span>
                  <button
                    onClick={() => setFiles(prev => prev.filter((_, idx) => idx !== i))}
                    disabled={busy}
                    className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-30"
                    title={t('library.importRemove', 'Remove')}
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* What the track sounds like */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-zinc-500">
              {t('library.importDescription', 'Description (optional)')}
            </label>
            <input
              type="text"
              value={description}
              disabled={busy}
              onChange={e => setDescription(e.target.value)}
              placeholder={t('library.importDescriptionPlaceholder', 'dark synthwave, analog bass, driving 4/4')}
              className="w-full px-3 py-2 rounded-lg text-xs bg-transparent border border-zinc-200 dark:border-white/10 text-zinc-800 dark:text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-pink-400/60 disabled:opacity-50"
            />
            <p className="text-[11px] text-zinc-500">
              {t('library.importDescriptionHint',
                'Applied to every file in this batch, and used as the genre tag when the file has none. '
                + 'StableStep refines towards it — without one, that stage is refused.')}
            </p>
          </div>

          {/* Post-processing opt-in */}
          <label className={`flex items-start gap-2 text-xs ${busy ? 'opacity-50' : 'cursor-pointer'}`}>
            <input
              type="checkbox"
              checked={runPp}
              disabled={busy}
              onChange={e => {
                setRunPp(e.target.checked);
                try { localStorage.setItem(PP_PREF_KEY, e.target.checked ? '1' : '0'); } catch { /* ignore */ }
              }}
              className="mt-0.5 accent-pink-500"
            />
            <span className="text-zinc-600 dark:text-zinc-400">
              {t('library.importRunPp', 'Run post-processing after import')}
              <span className="block text-[11px] text-zinc-500">
                {t('library.importRunPpHint',
                  'Uses the chain configured in the Post-Processing menu, one track at a time. '
                  + 'You can also run it later from a track’s menu.')}
              </span>
            </span>
          </label>

          {ppNeedsDescription && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-500">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('library.importPpNoDescription',
                  'No description given. If StableStep is on in your chain it will be refused for any '
                  + 'file without a genre tag of its own — every other stage still runs.')}
              </span>
            </div>
          )}

          {/* Progress */}
          {busy && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
                <Loader2 size={13} className="animate-spin text-pink-400" />
                {phase === 'uploading'
                  ? t('library.importUploading', 'Uploading {{size}}…', { size: formatSize(totalBytes) })
                  : t('library.importConverting', 'Converting and adding to the library…')}
              </div>
              <div className="h-1 rounded-full bg-zinc-200 dark:bg-white/10 overflow-hidden">
                <div
                  className={`h-full bg-pink-500 transition-all ${phase === 'converting' ? 'animate-pulse' : ''}`}
                  style={{ width: `${Math.round((phase === 'converting' ? 1 : progress) * 100)}%` }}
                />
              </div>
            </div>
          )}

          {/* Errors */}
          {!!error && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-500">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {failures.length > 0 && (
            <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-[11px] text-red-400 space-y-1">
              <div className="font-semibold">
                {t('library.importPartial', '{{ok}} imported, {{bad}} failed', {
                  ok: importedCount, bad: failures.length,
                })}
              </div>
              {failures.map(f => (
                <div key={f.file} className="text-red-300/90">
                  <span className="font-medium">{f.file}</span> — {f.error}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-zinc-200 dark:border-white/10">
          <button
            onClick={close}
            disabled={busy}
            className="px-3 py-1.5 rounded-lg text-xs text-zinc-600 dark:text-zinc-400 hover:bg-black/5 dark:hover:bg-white/10 transition-colors disabled:opacity-40"
          >
            {phase === 'done' ? t('common.close', 'Close') : t('common.cancel', 'Cancel')}
          </button>
          <button
            onClick={doImport}
            disabled={busy || files.length === 0 || !token}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-pink-500/15 text-pink-400 border border-pink-500/25 hover:bg-pink-500/25 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
            {files.length === 1
              ? t('library.importActionOne', 'Import 1 track')
              : t('library.importActionMany', 'Import {{n}} tracks', { n: files.length })}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
};
