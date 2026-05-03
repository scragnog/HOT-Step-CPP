// ModelManagerModal.tsx — Full-screen model manager modal
//
// Entry point for browsing, downloading, and managing GGUF models.
// Features starter packs at the top and a full tabbed catalogue below.

import React, { useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { X, HardDrive, FolderOpen, Download } from 'lucide-react';
import { useModelRegistry } from './useModelRegistry';
import { useDownloadStream } from './useDownloadStream';
import { StarterPackCard } from './StarterPackCard';
import { ModelCatalogueTab } from './ModelCatalogueTab';
import { DownloadProgressBar } from './DownloadProgressBar';
import { modelManagerApi } from '../../services/api';

interface Props {
  onClose: () => void;
}

function formatSize(bytes: number): string {
  if (bytes >= 1_073_741_824) return (bytes / 1_073_741_824).toFixed(1) + ' GB';
  if (bytes >= 1_048_576) return (bytes / 1_048_576).toFixed(0) + ' MB';
  return (bytes / 1024).toFixed(0) + ' KB';
}

export const ModelManagerModal: React.FC<Props> = ({ onClose }) => {
  const { registry, loading, error, refresh, getPackFiles, installedFiles } = useModelRegistry();

  const { jobs, hasActiveDownloads } = useDownloadStream({
    onComplete: refresh,
  });

  // ── Actions ─────────────────────────────────────────────────

  const handleDownload = useCallback(async (fileId: string) => {
    try {
      await modelManagerApi.download(fileId);
    } catch (err: any) {
      console.error('[ModelManager] Download failed:', err);
    }
  }, []);

  const handleDownloadPack = useCallback(async (fileIds: string[]) => {
    for (const id of fileIds) {
      try {
        await modelManagerApi.download(id);
      } catch (err: any) {
        console.error('[ModelManager] Pack download failed:', err);
      }
    }
  }, []);

  const handleCancel = useCallback(async (jobId: string) => {
    try {
      await modelManagerApi.cancel(jobId);
    } catch (err: any) {
      console.error('[ModelManager] Cancel failed:', err);
    }
  }, []);

  const handleResume = useCallback(async (jobId: string) => {
    try {
      await modelManagerApi.resume(jobId);
    } catch (err: any) {
      console.error('[ModelManager] Resume failed:', err);
    }
  }, []);

  const handleDelete = useCallback(async (filename: string) => {
    try {
      await modelManagerApi.deleteFile(filename);
      refresh();
    } catch (err: any) {
      console.error('[ModelManager] Delete failed:', err);
    }
  }, [refresh]);

  // ── Computed ────────────────────────────────────────────────

  const activeJobs = useMemo(() =>
    jobs.filter(j => j.status === 'downloading' || j.status === 'queued' || j.status === 'paused' || j.status === 'failed'),
    [jobs]
  );

  const totalInstalled = installedFiles.size;
  const totalDiskUsage = useMemo(() => {
    if (!registry) return 0;
    return registry.files
      .filter(f => f.installed)
      .reduce((a, f) => a + f.sizeBytes, 0);
  }, [registry]);

  // ── Render ──────────────────────────────────────────────────

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative w-full max-w-5xl max-h-[90vh] mt-[5vh] mx-4 rounded-2xl bg-zinc-900 border border-white/10 shadow-2xl flex flex-col overflow-hidden">
        {/* ── Header ───────────────────────────────────────── */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 flex-shrink-0">
          <div>
            <h2 className="text-lg font-bold text-zinc-100 flex items-center gap-2">
              <HardDrive size={18} className="text-pink-400" />
              Model Manager
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5">Download and manage GGUF models from HuggingFace</p>
          </div>
          <button onClick={onClose}
            className="p-2 rounded-xl hover:bg-white/5 text-zinc-500 hover:text-zinc-300 transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* ── Scrollable content ───────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6 scrollbar-thin">

          {/* Loading/Error states */}
          {loading && (
            <div className="flex items-center justify-center py-20">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-pink-500 border-t-transparent" />
            </div>
          )}

          {error && (
            <div className="rounded-xl bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
              Failed to load model registry: {error}
            </div>
          )}

          {registry && !loading && (
            <>
              {/* ── Active Downloads Banner ──────────────── */}
              {activeJobs.length > 0 && (
                <div className="rounded-2xl border border-sky-500/20 bg-sky-500/5 p-4">
                  <h3 className="text-xs font-semibold text-sky-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <Download size={13} className="animate-bounce" />
                    {activeJobs.length} download{activeJobs.length > 1 ? 's' : ''} in progress
                  </h3>
                  <div className="space-y-2">
                    {activeJobs.map(job => (
                      <DownloadProgressBar
                        key={job.jobId}
                        job={job}
                        onCancel={handleCancel}
                        onResume={handleResume}
                        compact
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* ── Starter Packs ────────────────────────── */}
              <div>
                <h3 className="text-sm font-semibold text-zinc-300 mb-1">Starter Packs</h3>
                <p className="text-xs text-zinc-600 mb-4">Get up and running quickly — each pack includes everything you need to generate music.</p>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                  {registry.packs.map(pack => (
                    <StarterPackCard
                      key={pack.id}
                      pack={pack}
                      files={getPackFiles(pack.id)}
                      downloadJobs={jobs}
                      onDownloadPack={handleDownloadPack}
                    />
                  ))}
                </div>
              </div>

              {/* ── Divider ──────────────────────────────── */}
              <div className="border-t border-white/5" />

              {/* ── Full Catalogue ────────────────────────── */}
              <div>
                <h3 className="text-sm font-semibold text-zinc-300 mb-1">All Models</h3>
                <p className="text-xs text-zinc-600 mb-4">Browse the complete model catalogue — {registry.files.length} models across 5 categories.</p>
                <ModelCatalogueTab
                  files={registry.files}
                  downloadJobs={jobs}
                  onDownload={handleDownload}
                  onCancel={handleCancel}
                  onResume={handleResume}
                  onDelete={handleDelete}
                />
              </div>
            </>
          )}
        </div>

        {/* ── Footer ──────────────────────────────────────── */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-white/5 bg-zinc-900/80 flex-shrink-0">
          <div className="flex items-center gap-4 text-[10px] text-zinc-600 font-mono">
            <span className="flex items-center gap-1">
              <FolderOpen size={11} />
              {registry?.modelsDir || '—'}
            </span>
            <span>{totalInstalled} models installed</span>
            <span>{formatSize(totalDiskUsage)} on disk</span>
          </div>
          {hasActiveDownloads && (
            <span className="text-[10px] text-sky-400 animate-pulse">Downloads in progress...</span>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
};
