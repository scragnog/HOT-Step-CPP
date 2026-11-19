import { useMutation, useQueryClient } from '@tanstack/react-query';
import { invalidateModels } from '@/queries/global-models/options';
// ModelManagerModal.tsx — Full-screen model manager modal
//
// Entry point for browsing, downloading, and managing GGUF models.
// Features starter packs at the top and a full tabbed catalogue below.

import React, { useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { X, HardDrive, FolderOpen, Download } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useModelRegistry } from './useModelRegistry';
import { useDownloadStream } from './useDownloadStream';
import { StarterPackCard } from './StarterPackCard';
import { ModelCatalogueTab, FAMILY_TABS, familyForPack, type FamilyTab } from './ModelCatalogueTab';
import { DownloadProgressBar } from './DownloadProgressBar';
import { modelManagerApi } from '@/services/api';
import { usePersistedState } from '@/hooks/usePersistedState';
import { formatBytes } from '@/utils/format/format';
import { plural } from '@/utils/format/plural';

interface Props {
  onClose: () => void;
}

/** Read the persisted Hugging Face token (set on the StableStep tab).
 *  Forwarded with every download request; the server only sends it to
 *  huggingface.co and only when non-empty (gated repos). */
function getStoredHfToken(): string | undefined {
  try {
    const raw = localStorage.getItem('hs-hfToken');
    const token = raw !== null ? JSON.parse(raw) : '';
    return typeof token === 'string' && token.trim() ? token.trim() : undefined;
  } catch {
    return undefined;
  }
}

export const ModelManagerModal: React.FC<Props> = ({ onClose }) => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const { registry, loading, error, getPackFiles, installedFiles } = useModelRegistry();

  // Selected family tab (ACE-Step 1.5 / MiniMax-Music3 / YuE2 / Shared) —
  // owned here (not inside ModelCatalogueTab) so the Starter Packs section
  // above the catalogue can filter to the same family.
  const [activeFamily, setActiveFamily] = usePersistedState<FamilyTab>('hs-modelManagerFamily', 'as1.5');
  const [sa3LicenseAccepted] = usePersistedState('hs-stablestepLicenseAccepted', false);

  const { jobs, hasActiveDownloads } = useDownloadStream();

  // ── Actions ─────────────────────────────────────────────────

  const handleDownload = useCallback(async (fileId: string) => {
    try {
      await modelManagerApi.download(fileId, getStoredHfToken());
    } catch (err: any) {
      console.error('[ModelManager] Download failed:', err);
    }
  }, []);

  const handleDownloadPack = useCallback(async (fileIds: string[]) => {
    for (const id of fileIds) {
      try {
        await modelManagerApi.download(id, getStoredHfToken());
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

  // Deleting a model file is one command: one mutation. deleteFile is
  // non-silent, so the global toast is the notifier (no inline error line here).
  const deleteMut = useMutation({
    mutationFn: (filename: string) => modelManagerApi.deleteFile(filename),
    onSuccess: () => { void invalidateModels(client); },
  });

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

  // Starter packs for the currently selected family tab only.
  const visiblePacks = useMemo(() => {
    if (!registry) return [];
    return registry.packs.filter(p => familyForPack(p, registry.files) === activeFamily);
  }, [registry, activeFamily]);

  // ── Render ──────────────────────────────────────────────────

  return createPortal(
    <div data-portal-layer className="fixed inset-0 z-[70] flex items-start justify-center">
      {/* Backdrop — does NOT close on click; only the X button closes the modal */}
      <div className="absolute inset-0 bg-black/20 dark:bg-black/40 dark:bg-black/70 backdrop-blur-sm" />

      {/* Modal */}
      <div className="relative w-full max-w-5xl max-h-[90vh] mt-[5vh] mx-4 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 shadow-2xl flex flex-col overflow-hidden">
        {/* ── Header ───────────────────────────────────────── */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5 flex-shrink-0">
          <div>
            <h2 className="text-lg font-bold text-zinc-100 flex items-center gap-2">
              <HardDrive size={18} className="text-pink-400" />
              Model Manager
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5">{t('models.subtitle')}</p>
          </div>
          <button onClick={onClose}
            className="p-2 rounded-xl hover:bg-white/5 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors">
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
                    {activeJobs.length} {plural(activeJobs.length, 'download')} in progress
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

              {/* ── Family tabs ──────────────────────────── */}
              {/* Selects the backend (ACE-Step 1.5 / MiniMax-Music3 / YuE2 /
                  Shared) both the Starter Packs and the full catalogue below
                  filter to — one selection drives both sections. */}
              <div className="flex gap-1 border-b border-zinc-200 dark:border-white/5 overflow-x-auto">
                {FAMILY_TABS.map(fam => (
                  <button
                    key={fam.id}
                    onClick={() => setActiveFamily(fam.id)}
                    className={`px-4 py-2.5 text-sm font-semibold whitespace-nowrap transition-colors border-b-2 -mb-px ${
                      activeFamily === fam.id
                        ? 'text-pink-400 border-pink-500'
                        : 'text-zinc-500 border-transparent hover:text-zinc-700 dark:hover:text-zinc-300'
                    }`}
                  >
                    {fam.label}
                  </button>
                ))}
              </div>

              {/* ── Starter Packs ────────────────────────── */}
              {visiblePacks.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-1">{t('models.starterPacks')}</h3>
                  <p className="text-xs text-zinc-600 mb-4">{t('models.starterPacksDesc')}</p>
                  {activeFamily === 'sa3' && !sa3LicenseAccepted && (
                    <p className="text-xs text-amber-400 mb-3">{t('models.sa3.acceptPackLicense', 'Accept the Stability AI license below to enable pack downloads.')}</p>
                  )}
                  <fieldset disabled={activeFamily === 'sa3' && !sa3LicenseAccepted} className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                    {visiblePacks.map(pack => (
                      <StarterPackCard
                        key={pack.id}
                        pack={pack}
                        files={getPackFiles(pack.id)}
                        downloadJobs={jobs}
                        onDownloadPack={handleDownloadPack}
                      />
                    ))}
                  </fieldset>
                </div>
              )}

              {/* ── Divider ──────────────────────────────── */}
              <div className="border-t border-zinc-200 dark:border-white/5" />

              {/* ── Full Catalogue ────────────────────────── */}
              <div>
                <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-1">{t('models.allModels')}</h3>
                <p className="text-xs text-zinc-600 mb-4">{t('models.allModelsDesc', { count: registry.files.length })}</p>
                <ModelCatalogueTab
                  files={registry.files}
                  downloadJobs={jobs}
                  onDownload={handleDownload}
                  onCancel={handleCancel}
                  onResume={handleResume}
                  onDelete={deleteMut.mutate}
                  activeFamily={activeFamily}
                />
              </div>
            </>
          )}
        </div>

        {/* ── Footer ──────────────────────────────────────── */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-zinc-200 dark:border-white/5 bg-white dark:bg-zinc-900/80 flex-shrink-0">
          <div className="flex items-center gap-4 text-[10px] text-zinc-600 font-mono">
            <span className="flex items-center gap-1">
              <FolderOpen size={11} />
              {registry?.modelsDir || '—'}
            </span>
            <span>{totalInstalled} {t('models.installed').toLowerCase()}</span>
            <span>{formatBytes(totalDiskUsage)} {t('models.onDisk', { size: '' }).trim()}</span>
          </div>
          {hasActiveDownloads && (
            <span className="text-[10px] text-sky-400 animate-pulse">{t('models.downloadsActive')}</span>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
};
