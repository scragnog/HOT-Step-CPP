// FileBrowserModal.tsx — Reusable file/folder browser modal
//
// Navigates the server filesystem via /api/adapters/browse.
// Supports both file-selection and folder-selection modes.
// Remembers last-visited directory per filter type in localStorage.

import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { createPortal } from 'react-dom';
import { Folder, FileText, ChevronUp, X, Loader2 } from 'lucide-react';
import { browseOptions } from '@/queries/shell-queue/settings';
import type { BrowseEntry } from '@/types';
import { formatBytes } from '@/utils/format/format';
import { readString } from '@/utils/storage/storageJson';

interface FileBrowserModalProps {
  /** Controls visibility */
  open: boolean;
  /** Called when modal is dismissed without selection */
  onClose: () => void;
  /** Called with the selected path (file or folder) */
  onSelect: (path: string) => void;
  /** 'file' = only filtered files selectable, 'folder' = only directories */
  mode: 'file' | 'folder';
  /** Optional starting directory (overrides remembered dir) */
  startPath?: string;
  /** File filter: 'adapters' shows .safetensors, 'audio' shows audio files */
  filter?: string;
  /** Custom title for the modal header */
  title?: string;
}

/** localStorage key per filter type */
const STORAGE_KEY_PREFIX = 'hs-fileBrowser-lastDir';

function getLastDir(filter?: string): string {
  const key = filter ? `${STORAGE_KEY_PREFIX}-${filter}` : STORAGE_KEY_PREFIX;
  return readString(key);
}

function saveLastDir(dir: string, filter?: string): void {
  const key = filter ? `${STORAGE_KEY_PREFIX}-${filter}` : STORAGE_KEY_PREFIX;
  try { localStorage.setItem(key, dir); } catch { /* ignore */ }
}

export const FileBrowserModal: React.FC<FileBrowserModalProps> = ({
  open, onClose, onSelect, mode, startPath, filter, title,
}) => {
  // The directory being browsed; changing it drives the query. Query handles
  // cancellation of stale navigations and loading/error state automatically.
  // null until the open effect picks the start dir; '' then means the server's
  // default folder.
  const [currentDir, setCurrentDir] = useState<string | null>(null);
  const [pathInput, setPathInput] = useState('');

  const browseQuery = useQuery({
    ...browseOptions(currentDir ?? '', filter),
    enabled: open && currentDir !== null,
  });

  // Sync the displayed path and persist it when a listing lands.
  useEffect(() => {
    if (browseQuery.data) {
      setPathInput(browseQuery.data.current);
      saveLastDir(browseQuery.data.current, filter);
    }
  }, [browseQuery.data, filter]);

  // Load initial directory when modal opens
  useEffect(() => {
    if (open) {
      const dir = startPath || getLastDir(filter) || '';
      setCurrentDir(dir);
      setPathInput(dir);
    }
  }, [open, startPath, filter]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!open) return null;

  const entries = browseQuery.data?.entries ?? [];
  const loading = browseQuery.isPending;
  const error = browseQuery.error?.message ?? null;
  const displayDir = browseQuery.data?.current ?? currentDir ?? '';

  const navigateTo = (dir: string) => { setCurrentDir(dir); setPathInput(dir); };

  const handleEntryClick = (entry: BrowseEntry) => {
    if (entry.type === 'dir') {
      navigateTo(entry.path);
    } else if (mode === 'file') {
      onSelect(entry.path);
    }
  };

  const handleSelectFolder = () => {
    if (mode === 'folder' && displayDir) {
      onSelect(displayDir);
    }
  };

  const handlePathSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (pathInput.trim()) {
      navigateTo(pathInput.trim());
    }
  };

  return createPortal(
    <div
      data-portal-layer
      className="fixed inset-0 z-[70] flex items-center justify-center"
      style={{ background: 'rgba(0, 0, 0, 0.5)', backdropFilter: 'blur(4px)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="flex flex-col overflow-hidden"
        style={{
          width: '540px',
          maxHeight: '70vh',
          background: '#18181b',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-4 py-3"
          style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}
        >
          <h3 className="text-sm font-semibold text-white">
            {title || (mode === 'file' ? 'Select File' : 'Select Folder')}
          </h3>
          <button
            onClick={onClose}
            className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-700 dark:text-zinc-300 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Path bar */}
        <form
          onSubmit={handlePathSubmit}
          className="px-4 py-2"
          style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              placeholder="Enter path..."
              className="flex-1 bg-zinc-200 dark:bg-black/20 border border-zinc-300 dark:border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-pink-500 font-mono"
            />
            <button
              type="submit"
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:bg-zinc-700 transition-colors"
            >
              Go
            </button>
          </div>
        </form>

        {/* Error */}
        {error && (
          <div className="px-4 py-2 text-xs text-red-400 bg-red-900/20">
            {error}
          </div>
        )}

        {/* File list */}
        <div className="flex-1 overflow-y-auto" style={{ minHeight: '200px', maxHeight: '400px' }}>
          {loading ? (
            <div className="flex items-center justify-center py-12 text-zinc-600 dark:text-zinc-400">
              <Loader2 size={20} className="animate-spin mr-2" />
              <span className="text-xs">Loading...</span>
            </div>
          ) : (
            <div>
              {entries.map((entry) => (
                <button
                  key={entry.path}
                  onClick={() => handleEntryClick(entry)}
                  className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-white/5 transition-colors group"
                  style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.02)' }}
                >
                  {entry.type === 'dir' ? (
                    entry.name === '..' ? (
                      <ChevronUp size={16} className="text-zinc-600 dark:text-zinc-400 flex-shrink-0" />
                    ) : (
                      <Folder size={16} className="text-amber-500 flex-shrink-0" />
                    )
                  ) : (
                    <FileText size={16} className="text-pink-500 flex-shrink-0" />
                  )}
                  <span className={`text-xs truncate flex-1 ${
                    entry.type === 'dir'
                      ? 'text-zinc-700 dark:text-zinc-300 font-medium'
                      : 'text-zinc-600 dark:text-zinc-400'
                  }`}>
                    {entry.name}
                  </span>
                  {entry.type === 'file' && entry.size != null && (
                    <span className="text-zinc-500 flex-shrink-0" style={{ fontSize: '10px' }}>
                      {formatBytes(entry.size, { zero: '' })}
                    </span>
                  )}
                  {entry.type === 'file' && mode === 'file' && (
                    <span
                      className="font-semibold text-pink-500 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                      style={{ fontSize: '10px' }}
                    >
                      Select
                    </span>
                  )}
                </button>
              ))}
              {entries.length === 0 && !loading && (
                <div className="py-8 text-center text-xs text-zinc-600 dark:text-zinc-400">
                  {filter === 'adapters' ? 'No .safetensors files found' : 'No matching files found'}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          className="flex items-center justify-between px-4 py-3"
          style={{
            borderTop: '1px solid rgba(255, 255, 255, 0.05)',
            background: 'rgba(0, 0, 0, 0.2)',
          }}
        >
          <span
            className="text-zinc-500 truncate font-mono"
            style={{ fontSize: '10px', maxWidth: '60%' }}
            title={displayDir}
          >
            {displayDir}
          </span>
          <div className="flex gap-2">
            {mode === 'folder' && (
              <button
                onClick={handleSelectFolder}
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-pink-600 text-white hover:bg-pink-700 transition-colors"
              >
                Select This Folder
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-600 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
};
