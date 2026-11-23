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

export interface ArchiveImportProgress {
  phase: 'checking' | 'extracting' | 'rewriting' | 'waiting' | 'finishing';
  filesDone: number; filesTotal: number; bytesDone: number; bytesTotal: number;
  /** Set while waiting for running work to stop before the switch. */
  blockers?: string[];
}
export interface ArchiveExternal { path: string; refs: number; existsHere: boolean }
export interface ArchiveImportReport {
  importedAt: string; zipFile: string; target: string;
  files: number; bytes: number; integrity: 'ok' | 'no database';
  externals: ArchiveExternal[]; skipped: number; scheduled: boolean;
}
export interface ArchiveImportJob {
  id: string;
  state: 'running' | 'done' | 'failed' | 'cancelled';
  zipFile: string;
  target: string;
  progress: ArchiveImportProgress | null;
  report: ArchiveImportReport | null;
  error: string | null;
  errors: string[];
  startedAt: string;
  finishedAt: string | null;
}
export interface ArchiveImportPreview {
  archive: { createdAt: string; appVersion: string; platform: string; dataRoot: string | null; categories: unknown };
  sources: Array<{ category: ArchiveCategoryId; prefix: string; files: number; bytes: number }>;
  files: number;
  bytes: number;
  externals: ArchiveExternal[];
  skipped: unknown[];
  target: { path: string; freeBytes: number | null; errors: string[] } | null;
  blockers: string[];
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
  importPreview: (zip: string, target: string) =>
    apiClient.post<ArchiveImportPreview>(`${BASE}/import/preview`, { zip, target: target || undefined }, { silent: true }),
  importJob: (opts?: ReadOptions) => apiClient.get<{ job: ArchiveImportJob | null }>(`${BASE}/import`, opts),
  importStart: (zip: string, target: string) => apiClient.post<{ job: ArchiveImportJob }>(`${BASE}/import`, { zip, target }, { silent: true }),
  importCancel: () => apiClient.post<{ job: ArchiveImportJob }>(`${BASE}/import/cancel`, {}, { silent: true }),
};
