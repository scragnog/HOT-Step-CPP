// FileBrowserModal.tsx — Reusable file/folder browser modal
//
// Navigates the server filesystem via /api/adapters/browse.
// Supports both file-selection and folder-selection modes.
// Remembers last-visited directory per filter type in localStorage.

import React, { useState, useEffect, useCallback } from 'react';
import { Folder, FileText, ChevronUp, X, Loader2 } from 'lucide-react';
import { adapterApi } from '../../services/api';
import type { BrowseEntry } from '../../types';

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

/** Format byte size to human-readable string */
function formatSize(bytes?: number): string {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** localStorage key per filter type */
const STORAGE_KEY_PREFIX = 'hs-fileBrowser-lastDir';

function getLastDir(filter?: string): string {
  const key = filter ? `${STORAGE_KEY_PREFIX}-${filter}` : STORAGE_KEY_PREFIX;
  try { return localStorage.getItem(key) || ''; } catch { return ''; }
}

function saveLastDir(dir: string, filter?: string): void {
  const key = filter ? `${STORAGE_KEY_PREFIX}-${filter}` : STORAGE_KEY_PREFIX;
  try { localStorage.setItem(key, dir); } catch { /* ignore */ }
}

export const FileBrowserModal: React.FC<FileBrowserModalProps> = ({
  open, onClose, onSelect, mode, startPath, filter, title,
}) => {
  const [currentDir, setCurrentDir] = useState('');
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pathInput, setPathInput] = useState('');

  const loadDir = useCallback(async (dir: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await adapterApi.browse(dir, filter);
      setCurrentDir(result.current);
      setPathInput(result.current);
      setEntries(result.entries);
      saveLastDir(result.current, filter);
    } catch (err: any) {
      setError(err?.message || 'Failed to list directory');
    } finally {
      setLoading(false);
    }
  }, [filter]);

  // Load initial directory when modal opens
  useEffect(() => {
    if (open) {
      loadDir(startPath || getLastDir(filter) || '');
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!open) return null;

  const handleEntryClick = (entry: BrowseEntry) => {
    if (entry.type === 'dir') {
      loadDir(entry.path);
    } else if (mode === 'file') {
      onSelect(entry.path);
    }
  };

  const handleSelectFolder = () => {
    if (mode === 'folder' && currentDir) {
      onSelect(currentDir);
    }
  };

  const handlePathSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (pathInput.trim()) {
      loadDir(pathInput.trim());
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
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
            className="text-zinc-400 hover:text-zinc-300 transition-colors"
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
              className="flex-1 bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500 font-mono"
            />
            <button
              type="submit"
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors"
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
            <div className="flex items-center justify-center py-12 text-zinc-400">
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
                      <ChevronUp size={16} className="text-zinc-400 flex-shrink-0" />
                    ) : (
                      <Folder size={16} className="text-amber-500 flex-shrink-0" />
                    )
                  ) : (
                    <FileText size={16} className="text-pink-500 flex-shrink-0" />
                  )}
                  <span className={`text-xs truncate flex-1 ${
                    entry.type === 'dir'
                      ? 'text-zinc-300 font-medium'
                      : 'text-zinc-400'
                  }`}>
                    {entry.name}
                  </span>
                  {entry.type === 'file' && entry.size != null && (
                    <span className="text-zinc-500 flex-shrink-0" style={{ fontSize: '10px' }}>
                      {formatSize(entry.size)}
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
                <div className="py-8 text-center text-xs text-zinc-400">
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
            title={currentDir}
          >
            {currentDir}
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
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-zinc-700 text-zinc-300 hover:bg-zinc-600 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
