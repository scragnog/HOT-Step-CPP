// FolderPicker.tsx — folder browse dialog for the dataset wizard
//
// Reuses the existing GET /api/adapters/browse endpoint with the
// `trainingAudio` filter so audio files show up as context while the user
// picks the containing directory. The last-visited dir is remembered.

import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { createPortal } from 'react-dom';
import { ChevronUp, Folder, FileAudio, Loader2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { trainingQueries as tq } from '@/queries/training-data/queries';
import { readString } from '@/utils/storage/storageJson';

const LAST_DIR_KEY = 'hs-training-lastDir';

function getLastDir(): string {
  return readString(LAST_DIR_KEY);
}
function saveLastDir(dir: string): void {
  try { localStorage.setItem(LAST_DIR_KEY, dir); } catch { /* ignore */ }
}

interface FolderPickerProps {
  open: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
}

export const FolderPicker: React.FC<FolderPickerProps> = ({ open, onClose, onSelect }) => {
  const { t } = useTranslation();
  // The directory the listing is for. It starts at the last-visited dir
  // (remembered in localStorage) and changes as you navigate — which changes
  // the query key, so useQuery fetches the new dir and shows a cached result
  // instantly when you revisit one (no hand-rolled loading/error/cancel).
  const [currentDir, setCurrentDir] = useState(getLastDir);
  const [pathInput, setPathInput] = useState(getLastDir);

  // An empty dir asks the server for its default folder, so a first open with
  // nothing remembered still lists something.
  const browseQuery = useQuery({
    ...tq.browse(currentDir, 'trainingAudio'),
    enabled: open,
  });

  // Remember where the user was once a listing lands (localStorage is an
  // external system — the sanctioned use of an effect).
  useEffect(() => {
    const cur = browseQuery.data?.current;
    if (cur) saveLastDir(cur);
  }, [browseQuery.data]);

  if (!open) return null;

  const navigate = (dir: string) => { setCurrentDir(dir); setPathInput(dir); };
  const resolvedDir = browseQuery.data?.current ?? currentDir;
  const entries = browseQuery.data?.entries ?? [];
  // Spinner only while a not-yet-cached dir is loading (a cached revisit shows
  // instantly and refetches in the background).
  const loading = browseQuery.isFetching && !browseQuery.data;
  const error = browseQuery.error
    ? (browseQuery.error instanceof Error ? browseQuery.error.message : 'Failed to list directory')
    : null;

  const dirs = entries.filter(e => e.type === 'dir');
  const files = entries.filter(e => e.type === 'file');

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="w-[560px] max-h-[72vh] flex flex-col overflow-hidden rounded-xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-card shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.wizard.folder')}</h3>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>

        <form
          onSubmit={(e) => { e.preventDefault(); if (pathInput.trim()) navigate(pathInput.trim()); }}
          className="px-4 py-2 border-b border-zinc-200 dark:border-white/5"
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              placeholder={t('trainingStudio.wizard.pathPlaceholder')}
              className="flex-1 rounded-lg px-3 py-1.5 text-xs font-mono bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500"
            />
            <button
              type="submit"
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
            >
              {t('trainingStudio.wizard.go')}
            </button>
          </div>
        </form>

        {error && <div className="px-4 py-2 text-xs text-red-500 bg-red-500/10">{error}</div>}

        <div className="flex-1 overflow-y-auto min-h-[220px]">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-zinc-500 text-xs">
              <Loader2 size={18} className="animate-spin mr-2" /> {t('trainingStudio.wizard.scanning')}
            </div>
          ) : (
            <>
              {dirs.map(entry => (
                <button
                  key={entry.path}
                  onClick={() => navigate(entry.path)}
                  className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                >
                  {entry.name === '..'
                    ? <ChevronUp size={15} className="text-zinc-500 flex-shrink-0" />
                    : <Folder size={15} className="text-amber-500 flex-shrink-0" />}
                  <span className="text-xs truncate text-zinc-700 dark:text-zinc-300 font-medium">{entry.name}</span>
                </button>
              ))}
              {files.slice(0, 50).map(entry => (
                <div key={entry.path} className="w-full flex items-center gap-3 px-4 py-1.5 opacity-60">
                  <FileAudio size={14} className="text-zinc-500 flex-shrink-0" />
                  <span className="text-[11px] truncate text-zinc-500">{entry.name}</span>
                </div>
              ))}
              {entries.length === 0 && (
                <div className="py-10 text-center text-xs text-zinc-500">{t('trainingStudio.wizard.foundNone')}</div>
              )}
            </>
          )}
        </div>

        <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
          <span className="text-[10px] font-mono text-zinc-500 truncate max-w-[55%]" title={resolvedDir}>{resolvedDir}</span>
          <div className="flex gap-2">
            <button
              onClick={() => resolvedDir && onSelect(resolvedDir)}
              disabled={!resolvedDir}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 transition-colors"
            >
              {t('trainingStudio.wizard.useFolder')}
            </button>
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
            >
              {t('trainingStudio.wizard.cancel')}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
};

export default FolderPicker;
