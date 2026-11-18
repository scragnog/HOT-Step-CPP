import { apiClient, type ReadOptions } from './httpClient';

// Mirrors server/src/routes/settingsWorkspaceArchive.ts (the UI does not import
// server code). Keep the two in step.
export type ArchiveCategoryId = 'library' | 'training' | 'adapters' | 'models' | 'secrets';
export type ArchiveChoice = Record<Exclude<ArchiveCategoryId, 'library'>, boolean>;

export interface ArchivePreview {
  dataRoot: string | null;
  categories: Record<ArchiveCategoryId, { files: number; bytes: number; paths: string[] }>;
  defaults: ArchiveChoice;
  destination: { path: string; freeBytes: number | null; errors: string[] } | null;
  skipped: Array<{ path: string; reason: string }>;
  skippedNotListed: number;
  blockers: string[];
}

export interface ArchiveJob {
  id: string;
  state: 'holding' | 'running' | 'done' | 'failed' | 'cancelled';
  outFile: string;
  categories: ArchiveChoice;
  progress: { phase: 'database' | 'files' | 'finishing'; filesDone: number; filesTotal: number; bytesDone: number; bytesTotal: number } | null;
  bytes: number | null;
  skipped: number;
  externals: number;
  error: string | null;
  blockers: string[];
  startedAt: string;
  finishedAt: string | null;
}

const BASE = '/settings/workspace-archive';

// Writes are silent at the transport: WorkspaceArchivePanel shows each failure inline.
export const workspaceArchiveApi = {
  preview: (destination: string, opts?: ReadOptions) =>
    apiClient.get<ArchivePreview>(`${BASE}/preview${destination ? `?destination=${encodeURIComponent(destination)}` : ''}`, opts),
  job: (opts?: ReadOptions) => apiClient.get<{ job: ArchiveJob | null }>(`${BASE}/export`, opts),
  start: (destination: string, categories: ArchiveChoice) =>
    apiClient.post<{ job: ArchiveJob }>(`${BASE}/export`, { destination, categories }, { silent: true }),
  cancel: () => apiClient.post<{ job: ArchiveJob }>(`${BASE}/export/cancel`, {}, { silent: true }),
};
