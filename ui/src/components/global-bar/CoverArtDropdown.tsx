// CoverArtDropdown.tsx — AI cover art toggle + download status
//
// Renders as an accordion section inside PostProcessingDropdown.
// Shows: toggle, installation status, download progress, model info.

import React, { useState, useEffect, useCallback } from 'react';
import { Image, Download, X, Check, Loader2 } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { useAuth } from '../../context/AuthContext';
import { ToggleSwitch } from './BarSection';

// ── Types ────────────────────────────────────────────────────────

interface FileProgress {
  filename: string;
  description: string;
  status: string;
  bytesDownloaded: number;
  totalBytes: number;
  speed: number;
}

interface CoverArtStatusResponse {
  installed: boolean;
  missingFiles: string[];
  download: {
    phase: string;
    files: FileProgress[];
    totalBytes: number;
    downloadedBytes: number;
    overallProgress: number;
  };
}

// ── Helpers ──────────────────────────────────────────────────────

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

const formatSpeed = (bytesPerSec: number): string => {
  if (bytesPerSec < 1024 * 1024) return `${(bytesPerSec / 1024).toFixed(0)} KB/s`;
  return `${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`;
};

// ── Component ───────────────────────────────────────────────────

export const CoverArtContent: React.FC = () => {
  const gp = useGlobalParams();
  const { token } = useAuth();
  const [status, setStatus] = useState<CoverArtStatusResponse | null>(null);
  const [polling, setPolling] = useState(false);

  // Poll status
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/cover-art/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        return data;
      }
    } catch {}
    return null;
  }, []);

  // Initial fetch + polling during download
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (!polling) return;
    const interval = setInterval(async () => {
      const data = await fetchStatus();
      if (data && data.download.phase !== 'downloading') {
        setPolling(false);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [polling, fetchStatus]);

  // Start download
  const handleDownload = useCallback(async () => {
    if (!token) return;
    try {
      await fetch('/api/cover-art/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      });
      setPolling(true);
      fetchStatus();
    } catch (err) {
      console.error('[CoverArt] Download failed:', err);
    }
  }, [token, fetchStatus]);

  // Cancel download
  const handleCancel = useCallback(async () => {
    if (!token) return;
    try {
      await fetch('/api/cover-art/download/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      });
      setPolling(false);
      fetchStatus();
    } catch {}
  }, [token, fetchStatus]);

  const isDownloading = status?.download?.phase === 'downloading';
  const isInstalled = status?.installed ?? false;

  return (
    <div className="space-y-3 mt-2">
      {/* Auto-generate toggle */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-zinc-600 dark:text-zinc-400">Auto-generate after creation</span>
        <ToggleSwitch
          checked={gp.coverArtEnabled}
          onChange={gp.setCoverArtEnabled}
          accentColor="pink"
        />
      </div>

      {/* Status indicator */}
      {!status ? (
        <div className="text-xs text-zinc-500 italic text-center py-2">
          Checking status...
        </div>
      ) : isInstalled ? (
        <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-emerald-500/5 border border-emerald-500/10">
          <Check size={14} className="text-emerald-400 flex-shrink-0" />
          <span className="text-xs text-emerald-400">Ready — FLUX.2-klein-4B</span>
        </div>
      ) : isDownloading ? (
        /* Download progress */
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Loader2 size={14} className="text-pink-400 animate-spin" />
              <span className="text-xs text-zinc-400">
                Downloading... {status.download.overallProgress}%
              </span>
            </div>
            <button
              onClick={handleCancel}
              className="p-1 rounded hover:bg-red-500/10 text-zinc-500 hover:text-red-400 transition-colors"
              title="Cancel download"
            >
              <X size={12} />
            </button>
          </div>

          {/* Overall progress bar */}
          <div className="w-full bg-zinc-200 dark:bg-zinc-800 rounded-full h-1.5">
            <div
              className="bg-gradient-to-r from-pink-500 to-purple-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${status.download.overallProgress}%` }}
            />
          </div>

          {/* Per-file progress */}
          <div className="space-y-1">
            {status.download.files.map(f => (
              <div key={f.filename} className="flex items-center gap-2 text-[10px]">
                <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
                  f.status === 'completed' ? 'bg-emerald-400' :
                  f.status === 'downloading' ? 'bg-pink-400 animate-pulse' :
                  f.status === 'failed' ? 'bg-red-400' :
                  'bg-zinc-600'
                }`} />
                <span className="text-zinc-500 truncate flex-1">{f.description}</span>
                {f.status === 'downloading' && f.speed > 0 && (
                  <span className="text-zinc-600 font-mono flex-shrink-0">
                    {formatSpeed(f.speed)}
                  </span>
                )}
                <span className="text-zinc-600 font-mono flex-shrink-0">
                  {f.status === 'completed' ? '✓' :
                   f.status === 'downloading' ? `${formatBytes(f.bytesDownloaded)} / ${formatBytes(f.totalBytes)}` :
                   f.status === 'failed' ? '✗' :
                   formatBytes(f.totalBytes)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Not installed — show download button */
        <div className="space-y-2">
          <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800/50 border border-zinc-300 dark:border-white/5">
            <Image size={14} className="text-zinc-500 flex-shrink-0" />
            <span className="text-xs text-zinc-500">
              Not installed — one-click download (~5.9 GB)
            </span>
          </div>

        <button
            onClick={handleDownload}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-semibold rounded-xl
                       bg-gradient-to-r from-pink-500/10 to-purple-500/10
                       border border-pink-500/20 text-pink-400
                       hover:from-pink-500/20 hover:to-purple-500/20 hover:border-pink-500/30
                       transition-all"
          >
            <Download size={14} />
            Download Cover Art Models + Engine
          </button>
        </div>
      )}

      <p className="text-[10px] text-zinc-600 leading-relaxed">
        Generate 1024×1024 album cover art using FLUX.2-klein-4B.
        Uses the song&apos;s subject or lyrics to create relevant artwork.
        Runs after audio generation completes.
      </p>
    </div>
  );
};

// ── Badge ────────────────────────────────────────────────────────

export const CoverArtBadge: React.FC = () => {
  const { coverArtEnabled } = useGlobalParams();
  if (!coverArtEnabled) return null;
  return (
    <span className="text-[10px] text-pink-400/60 font-mono">Cover Art</span>
  );
};
