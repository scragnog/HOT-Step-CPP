// ModelRow.tsx — Single model entry in the catalogue

import React, { useState } from 'react';
import { Download, Trash2, CheckCircle } from 'lucide-react';
import { DownloadProgressBar } from './DownloadProgressBar';
import type { RegistryFile, DownloadJob } from '../../types';

interface Props {
  file: RegistryFile;
  downloadJob?: DownloadJob;
  onDownload: (fileId: string) => void;
  onCancel: (jobId: string) => void;
  onResume: (jobId: string) => void;
  onDelete: (filename: string) => void;
}

/** Format bytes to human-readable string */
function formatSize(bytes: number): string {
  if (bytes >= 1_073_741_824) return (bytes / 1_073_741_824).toFixed(1) + ' GB';
  if (bytes >= 1_048_576) return (bytes / 1_048_576).toFixed(0) + ' MB';
  return (bytes / 1024).toFixed(0) + ' KB';
}

const quantColors: Record<string, string> = {
  BF16: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
  F32: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
  F16: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
  Q8_0: 'bg-sky-500/15 text-sky-400 border-sky-500/20',
  Q6_K: 'bg-violet-500/15 text-violet-400 border-violet-500/20',
  Q5_K_M: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
  Q4_K_M: 'bg-orange-500/15 text-orange-400 border-orange-500/20',
  MXFP4: 'bg-pink-500/15 text-pink-400 border-pink-500/20',
};

export const ModelRow: React.FC<Props> = ({ file, downloadJob, onDownload, onCancel, onResume, onDelete }) => {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const isDownloading = downloadJob && (downloadJob.status === 'downloading' || downloadJob.status === 'queued' || downloadJob.status === 'paused' || downloadJob.status === 'failed');

  return (
    <div className="group rounded-xl border border-white/5 bg-zinc-800/30 hover:bg-zinc-800/60 hover:border-white/10 transition-all p-3">
      {/* Main row */}
      <div className="flex items-center gap-3">
        {/* Name + quant badge */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-sm font-medium text-zinc-200">{file.displayName}</span>
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${quantColors[file.quant] || 'bg-zinc-700 text-zinc-400 border-zinc-600'}`}>
              {file.quant}
            </span>
            {file.tags.includes('recommended') && (
              <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-pink-500/15 text-pink-400 border border-pink-500/20 uppercase tracking-wider">
                Recommended
              </span>
            )}
          </div>
          <p className="text-xs text-zinc-500 leading-relaxed truncate">{file.description}</p>
        </div>

        {/* Size */}
        <span className="text-xs text-zinc-500 font-mono flex-shrink-0 w-16 text-right">
          {formatSize(file.sizeBytes)}
        </span>

        {/* Actions */}
        <div className="flex items-center gap-1.5 flex-shrink-0 w-28 justify-end">
          {file.installed && !isDownloading && (
            <>
              {confirmDelete ? (
                <div className="flex items-center gap-1">
                  <button onClick={() => { onDelete(file.filename); setConfirmDelete(false); }}
                    className="text-[10px] px-2 py-1 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors">
                    Delete
                  </button>
                  <button onClick={() => setConfirmDelete(false)}
                    className="text-[10px] px-2 py-1 rounded bg-zinc-700 text-zinc-400 hover:bg-zinc-600 transition-colors">
                    Cancel
                  </button>
                </div>
              ) : (
                <>
                  <span className="flex items-center gap-1 text-xs text-emerald-400">
                    <CheckCircle size={12} />
                    <span className="text-[10px]">Installed</span>
                  </span>
                  <button onClick={() => setConfirmDelete(true)} title="Delete model"
                    className="p-1 rounded-lg text-zinc-600 hover:text-red-400 hover:bg-white/5 transition-colors opacity-0 group-hover:opacity-100">
                    <Trash2 size={12} />
                  </button>
                </>
              )}
            </>
          )}
          {!file.installed && !isDownloading && (
            <button onClick={() => onDownload(file.id)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-pink-500 to-pink-600 text-white text-xs font-medium hover:from-pink-400 hover:to-pink-500 transition-all shadow-lg shadow-pink-500/10">
              <Download size={12} />
              Download
            </button>
          )}
        </div>
      </div>

      {/* Download progress (inline) */}
      {isDownloading && downloadJob && (
        <div className="mt-2">
          <DownloadProgressBar job={downloadJob} onCancel={onCancel} onResume={onResume} compact />
        </div>
      )}
    </div>
  );
};
