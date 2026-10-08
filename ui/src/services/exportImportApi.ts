import { normalizeSong } from './api';
import type { Song } from '../types';

export type AudioFormat = 'wav' | 'flac' | 'opus' | 'mp3';
export type AudioVariant = 'original' | 'mastered' | 'noadapter' | 'latent';

async function readJson(response: Response) {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

export async function uploadImportAsset(file: File, token: string, onProgress?: (fraction: number) => void): Promise<string> {
  const form = new FormData();
  form.append('audio', file);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/export-import/assets');
    xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body: { assetId?: string; error?: string } = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error */ }
      if (xhr.status >= 200 && xhr.status < 300 && body.assetId) resolve(body.assetId);
      else reject(new Error(body.error || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('Upload failed — the server closed the connection'));
    xhr.send(form);
  });
}

export interface ImportResult {
  index: number;
  assetId: string;
  song?: Song;
  error?: string;
}

export async function importAssets(items: { assetId: string; description?: string }[], token: string): Promise<ImportResult[]> {
  const results: ImportResult[] = [];
  for (let offset = 0; offset < items.length; offset += 50) {
    const body = await readJson(await fetch('/api/export-import/imports', {
      method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ items: items.slice(offset, offset + 50) }),
    }));
    results.push(...(body.items as ImportResult[]).map((item) => ({
      ...item,
      index: item.index + offset,
      song: item.song ? normalizeSong(item.song) : undefined,
    })));
  }
  return results;
}

export async function resolveExports(
  items: { songId: string; variant?: AudioVariant; audioUrl?: string; srcUrl?: string }[],
  options: { format?: AudioFormat; bitrate?: number; artist?: string; prepend?: string; downloadVersion?: 'original' | 'mastered' | 'both'; includeLatent?: boolean }, token: string,
): Promise<{ index: number; songId: string; variant?: AudioVariant; filename?: string; url?: string; error?: string }[]> {
  const body = await readJson(await fetch('/api/export-import/exports/resolve', {
    method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ items, ...options }),
  }));
  return body.items;
}

export async function validateProfileImport(filename: string, profile: Record<string, unknown>, token: string) {
  return readJson(await fetch('/api/export-import/profiles/validate', {
    method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, profile }),
  })) as Promise<{ name: string; data: Record<string, unknown> }>;
}
