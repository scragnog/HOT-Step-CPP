import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { coverArtStatusOptions, type CoverArtStatus } from '@/queries/shell-queue/queue';
import { pollWhile } from '@/queries/contract';
import { queryManager } from '@/queries/invalidation';
import { apiClient } from '@/services/httpClient';
// CoverArtDropdown.tsx — AI cover art toggle + download status
//
// Renders as an accordion section inside PostProcessingDropdown.
// Shows: toggle, installation status, download progress, model info.

import React from 'react';
import { Image, Download, X, Check, Loader2 } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { useGlobalParamsStore } from '@/stores/globalParamsStore';
import { ToggleSwitch } from './BarSection';
import { QueryError } from '@/components/shared/feedback/QueryError';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { HintButton } from '@/components/shared/controls/HintButton';
import { GLOBAL_PARAM_DEFAULTS } from '@/stores/globalParamsStore';
import { formatBytes } from '@/utils/format/format';

// ── Helpers ──────────────────────────────────────────────────────

const formatSpeed = (bytesPerSec: number): string => {
  if (bytesPerSec < 1024 * 1024) return `${(bytesPerSec / 1024).toFixed(0)} KB/s`;
  return `${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`;
};

// ── Component ───────────────────────────────────────────────────

export const CoverArtContent: React.FC = () => {
  const gp = useGlobalParamsStore(useShallow(s => ({
    coverArtEnabled: s.coverArtEnabled, setCoverArtEnabled: s.setCoverArtEnabled,
  })));
  const client = useQueryClient();
  const options = coverArtStatusOptions();
  const { data: status, isError, error, refetch } = useQuery({ ...options,
    refetchInterval: pollWhile<CoverArtStatus>(data => data?.download.phase === 'downloading'),
  });
  // Start download (the write dirties the status it polls)
  const downloadMut = useMutation({
    mutationFn: () => apiClient.post('/cover-art/download'),
    onSuccess: () => { void queryManager.invalidate(client, options.queryKey); },
  });

  // Cancel download
  const cancelMut = useMutation({
    mutationFn: () => apiClient.post('/cover-art/download/cancel'),
    onSuccess: () => { void queryManager.invalidate(client, options.queryKey); },
  });

  const isDownloading = status?.download?.phase === 'downloading';
  const isInstalled = status?.installed ?? false;

  return (
    <div className="space-y-3 mt-2">
      {isError && <QueryError error={error} onRetry={() => void refetch()} compact />}
      {/* Auto-generate toggle */}
      <div className="flex items-center justify-between">
        <ParamLabel label="Auto-generate after creation"
          className="text-sm text-zinc-600 dark:text-zinc-400"
          info="Generates 1024x1024 album cover art with FLUX.2-klein-4B, using the song's subject or lyrics as the brief. Runs once audio generation finishes."
          onReset={gp.coverArtEnabled !== GLOBAL_PARAM_DEFAULTS.coverArtEnabled ? () => gp.setCoverArtEnabled(GLOBAL_PARAM_DEFAULTS.coverArtEnabled) : undefined} />
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
            <HintButton
              onClick={() => cancelMut.mutate()} disabled={cancelMut.isPending}
              label="Cancel download"
              info="Stops the cover-art model download. Files already fetched stay cached, so a later download resumes."
              ariaLabel="Cancel download"
              className="p-1 rounded hover:bg-red-500/10 text-zinc-500 hover:text-red-400 transition-colors"
            >
              <X size={12} />
            </HintButton>
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
            onClick={() => downloadMut.mutate()} disabled={downloadMut.isPending}
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

    </div>
  );
};

// ── Badge ────────────────────────────────────────────────────────

export const CoverArtBadge: React.FC = () => {
  const coverArtEnabled = useGlobalParamsStore(s => s.coverArtEnabled);
  if (!coverArtEnabled) return null;
  return (
    <span className="text-[10px] text-pink-400/60 font-mono">Cover Art</span>
  );
};
