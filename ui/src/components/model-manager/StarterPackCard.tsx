// StarterPackCard.tsx — Starter pack card with smart download

import React from 'react';
import { Download, CheckCircle, Package } from 'lucide-react';
import type { RegistryFile, DownloadJob } from '../../types';

interface Props {
  pack: { id: string; name: string; description: string; fileIds: string[] };
  files: RegistryFile[];
  downloadJobs: DownloadJob[];
  onDownloadPack: (fileIds: string[]) => void;
}

function formatSize(bytes: number): string {
  if (bytes >= 1_073_741_824) return (bytes / 1_073_741_824).toFixed(1) + ' GB';
  if (bytes >= 1_048_576) return (bytes / 1_048_576).toFixed(0) + ' MB';
  return (bytes / 1024).toFixed(0) + ' KB';
}

export const StarterPackCard: React.FC<Props> = ({ pack, files, downloadJobs, onDownloadPack }) => {
  const installed = files.filter(f => f.installed);
  const missing = files.filter(f => !f.installed);
  const allInstalled = missing.length === 0;
  const totalSize = files.reduce((a, f) => a + f.sizeBytes, 0);
  const remainingSize = missing.reduce((a, f) => a + f.sizeBytes, 0);

  // Check if any pack files are currently downloading
  const activeJobCount = downloadJobs.filter(j =>
    files.some(f => f.fileId === j.fileId || f.id === j.fileId) &&
    (j.status === 'downloading' || j.status === 'queued')
  ).length;

  const isDownloading = activeJobCount > 0;

  return (
    <div className={`rounded-2xl border p-5 transition-all ${
      allInstalled
        ? 'bg-emerald-500/5 border-emerald-500/20'
        : 'bg-zinc-800/50 border-white/5 hover:border-pink-500/20 hover:bg-zinc-800/80'
    }`}>
      {/* Header */}
      <div className="flex items-start gap-3 mb-3">
        <div className={`p-2 rounded-xl ${allInstalled ? 'bg-emerald-500/10' : 'bg-pink-500/10'}`}>
          <Package size={18} className={allInstalled ? 'text-emerald-400' : 'text-pink-400'} />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-bold text-zinc-200">{pack.name}</h3>
          <p className="text-xs text-zinc-500 mt-0.5 leading-relaxed">{pack.description}</p>
        </div>
      </div>

      {/* File list */}
      <div className="space-y-1 mb-4">
        {files.map(f => (
          <div key={f.id} className="flex items-center gap-2 text-xs">
            {f.installed ? (
              <CheckCircle size={11} className="text-emerald-400 flex-shrink-0" />
            ) : (
              <Download size={11} className="text-zinc-600 flex-shrink-0" />
            )}
            <span className={`truncate ${f.installed ? 'text-zinc-400' : 'text-zinc-300'}`}>
              {f.displayName}
            </span>
            <span className="text-zinc-600 font-mono text-[10px] ml-auto flex-shrink-0">
              {formatSize(f.sizeBytes)}
            </span>
          </div>
        ))}
      </div>

      {/* Size summary */}
      <div className="text-[10px] text-zinc-600 mb-3 font-mono">
        {formatSize(totalSize)} total
        {!allInstalled && <span className="text-zinc-500"> · {formatSize(remainingSize)} remaining</span>}
      </div>

      {/* Download button */}
      {allInstalled ? (
        <div className="flex items-center justify-center gap-1.5 py-2 rounded-xl bg-emerald-500/10 text-emerald-400 text-xs font-medium">
          <CheckCircle size={14} />
          All Installed
        </div>
      ) : isDownloading ? (
        <div className="flex items-center justify-center gap-1.5 py-2 rounded-xl bg-sky-500/10 text-sky-400 text-xs font-medium animate-pulse">
          Downloading {activeJobCount} file{activeJobCount > 1 ? 's' : ''}...
        </div>
      ) : (
        <button
          onClick={() => onDownloadPack(missing.map(f => f.id))}
          className="w-full flex items-center justify-center gap-1.5 py-2 rounded-xl bg-gradient-to-r from-pink-500 to-pink-600 text-white text-xs font-medium hover:from-pink-400 hover:to-pink-500 transition-all shadow-lg shadow-pink-500/10"
        >
          <Download size={14} />
          Download {missing.length} model{missing.length > 1 ? 's' : ''} ({formatSize(remainingSize)})
        </button>
      )}
    </div>
  );
};
