import { normalizeSong } from './api';
import { apiClient } from './httpClient';
import type { Song } from '@/types';

export type AudioFormat = 'wav' | 'flac' | 'opus' | 'mp3';
export type AudioVariant = 'original' | 'mastered' | 'noadapter' | 'latent';

// Silent: ImportTracksModal collects per-file failures into its own summary,
// so a toast per failed file would be a second copy of the same error.
export async function uploadImportAsset(file: File, onProgress?: (fraction: number) => void): Promise<string> {
  const data = await apiClient.upload<{ assetId: string }>('/export-import/assets', [{ field: 'audio', file }], { onProgress, silent: true });
  return data.assetId;
}

export interface ImportResult {
  index: number;
  assetId: string;
  song?: Song;
  error?: string;
}

// Silent: ImportTracksModal renders a batch failure itself (modal error line).
export async function importAssets(items: { assetId: string; description?: string }[]): Promise<ImportResult[]> {
  const results: ImportResult[] = [];
  for (let offset = 0; offset < items.length; offset += 50) {
    const body = await apiClient.post<{ items: ImportResult[] }>(
      '/export-import/imports', { items: items.slice(offset, offset + 50) }, { silent: true });
    results.push(...(body.items).map((item) => ({
      ...item,
      index: item.index + offset,
      song: item.song ? normalizeSong(item.song) : undefined,
    })));
  }
  return results;
}

export async function resolveExports(
  items: { songId: string; variant?: AudioVariant; audioUrl?: string; srcUrl?: string }[],
  options: { format?: AudioFormat; bitrate?: number; artist?: string; prepend?: string; downloadVersion?: 'original' | 'mastered' | 'both'; includeLatent?: boolean },
): Promise<{ index: number; songId: string; variant?: AudioVariant; filename?: string; url?: string; error?: string }[]> {
  const body = await apiClient.post<{ items: { index: number; songId: string; variant?: AudioVariant; filename?: string; url?: string; error?: string }[] }>(
    '/export-import/exports/resolve', { items, ...options });
  return body.items;
}

export async function validateProfileImport(filename: string, profile: Record<string, unknown>) {
  return apiClient.post<{ name: string; data: Record<string, unknown> }>('/export-import/profiles/validate', { filename, profile });
}
