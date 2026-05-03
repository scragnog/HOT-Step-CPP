// DownloadProgressBar.tsx — Reusable download progress bar with speed/ETA

import React from 'react';
import { X, Play } from 'lucide-react';
import type { DownloadJob } from '../../types';

interface Props {
  job: DownloadJob;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  compact?: boolean;
}

/** Format bytes to human-readable string */
function formatSize(bytes: number): string {
  if (bytes >= 1_073_741_824) return (bytes / 1_073_741_824).toFixed(2) + ' GB';
  if (bytes >= 1_048_576) return (bytes / 1_048_576).toFixed(1) + ' MB';
  if (bytes >= 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return bytes + ' B';
}

/** Format speed to human-readable string */
function formatSpeed(bytesPerSec: number): string {
  if (bytesPerSec >= 1_073_741_824) return (bytesPerSec / 1_073_741_824).toFixed(1) + ' GB/s';
  if (bytesPerSec >= 1_048_576) return (bytesPerSec / 1_048_576).toFixed(1) + ' MB/s';
  if (bytesPerSec >= 1024) return (bytesPerSec / 1024).toFixed(0) + ' KB/s';
  return bytesPerSec.toFixed(0) + ' B/s';
}

/** Format ETA in human-readable form */
function formatEta(bytes: number, speed: number): string {
  if (speed <= 0) return '—';
  const secs = bytes / speed;
  if (secs < 60) return `${Math.ceil(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.ceil(secs % 60)}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

const statusColors: Record<string, string> = {
  queued: 'bg-zinc-600 text-zinc-300',
  downloading: 'bg-sky-500/20 text-sky-400',
  paused: 'bg-amber-500/20 text-amber-400',
  completed: 'bg-emerald-500/20 text-emerald-400',
  failed: 'bg-red-500/20 text-red-400',
  cancelled: 'bg-zinc-600 text-zinc-400',
};

export const DownloadProgressBar: React.FC<Props> = ({ job, onCancel, onResume, compact }) => {
  const pct = job.totalBytes > 0 ? Math.min(100, (job.bytesDownloaded / job.totalBytes) * 100) : 0;
  const remaining = job.totalBytes - job.bytesDownloaded;
  const isActive = job.status === 'downloading' || job.status === 'queued';
  const canResume = job.status === 'paused' || job.status === 'failed';

  return (
    <div className={`rounded-xl border border-white/5 bg-zinc-800/50 ${compact ? 'p-2' : 'p-3'}`}>
      {/* Header row */}
      <div className="flex items-center gap-2 mb-1.5">
        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${statusColors[job.status] || statusColors.queued}`}>
          {job.status === 'downloading' ? `${pct.toFixed(0)}%` : job.status.toUpperCase()}
        </span>
        <span className={`${compact ? 'text-xs' : 'text-sm'} text-zinc-300 font-medium truncate flex-1`}>
          {job.filename}
        </span>
        <div className="flex items-center gap-1 flex-shrink-0">
          {canResume && (
            <button onClick={() => onResume(job.jobId)} title="Resume"
              className="p-1 rounded-lg hover:bg-white/10 text-amber-400 hover:text-amber-300 transition-colors">
              <Play size={12} />
            </button>
          )}
          {isActive && (
            <button onClick={() => onCancel(job.jobId)} title="Cancel"
              className="p-1 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-red-400 transition-colors">
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 rounded-full bg-zinc-700/50 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            isActive ? 'bg-gradient-to-r from-sky-500 to-sky-400' :
            job.status === 'completed' ? 'bg-emerald-500' :
            job.status === 'paused' ? 'bg-amber-500' :
            'bg-zinc-600'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Stats row */}
      <div className="flex items-center justify-between mt-1 text-[10px] text-zinc-500">
        <span>{formatSize(job.bytesDownloaded)} / {formatSize(job.totalBytes)}</span>
        {isActive && job.speed > 0 && (
          <span>{formatSpeed(job.speed)} · ETA {formatEta(remaining, job.speed)}</span>
        )}
        {job.status === 'failed' && job.error && (
          <span className="text-red-400 truncate ml-2">{job.error}</span>
        )}
      </div>
    </div>
  );
};
