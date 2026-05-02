// api.ts — Frontend API client
//
// Thin wrapper around fetch() for all server endpoints.
// Each method is standalone — import only what you need.

import type { Song, GenerationParams, GenerationJob, AuthState, AceModels, BrowseEntry, AdapterFile } from '../types';

const BASE = '/api';

async function get<T>(path: string, token?: string | null): Promise<T> {
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

async function post<T>(path: string, body?: unknown, token?: string | null): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

async function patch<T>(path: string, body: unknown, token?: string | null): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

async function del<T>(path: string, token?: string | null): Promise<T> {
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { method: 'DELETE', headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

// ── Auth ────────────────────────────────────────────────────
export const authApi = {
  autoLogin: () => get<AuthState>('/auth/auto'),
  getMe: (token: string) => get<{ user: AuthState['user'] }>('/auth/me', token),
  updateUsername: (username: string, token: string) =>
    patch<AuthState>('/auth/username', { username }, token),
};

// ── Song Normalizer ─────────────────────────────────────────
/** Map DB snake_case fields to camelCase for component consumption */
function normalizeSong(s: any): Song {
  const gp = (() => {
    if (s.generationParams) return s.generationParams;
    if (s.generation_params) {
      return typeof s.generation_params === 'string'
        ? JSON.parse(s.generation_params) : s.generation_params;
    }
    return undefined;
  })();

  return {
    id: s.id,
    title: s.title || '',
    lyrics: s.lyrics || '',
    style: s.style || '',
    caption: s.caption || s.style || '',
    audioUrl: s.audio_url || s.audioUrl || '',
    audio_url: s.audio_url,
    coverUrl: s.cover_url || s.coverUrl,
    cover_url: s.cover_url,
    duration: s.duration || 0,
    bpm: s.bpm || gp?.bpm,
    key_scale: s.key_scale,
    time_signature: s.time_signature,
    tags: s.tags || [],
    is_public: s.is_public,
    dit_model: s.dit_model,
    generation_params: s.generation_params,
    generationParams: gp,
    created_at: s.created_at,
    createdAt: s.created_at ? new Date(s.created_at) : undefined,
    masteredAudioUrl: s.mastered_audio_url || s.masteredAudioUrl || '',
    mastered_audio_url: s.mastered_audio_url,
  };
}

// ── Songs ────────────────────────────────────────────────────
export const songApi = {
  list: async (token: string) => {
    const data = await get<{ songs: any[] }>('/songs', token);
    return { songs: data.songs.map(normalizeSong) };
  },
  get: async (id: string) => {
    const data = await get<{ song: any }>(`/songs/${id}`);
    return { song: normalizeSong(data.song) };
  },
  create: (song: Partial<Song>, token: string) => post<{ song: Song }>('/songs', song, token),
  update: async (id: string, data: Partial<Song>, token: string) => {
    const resp = await patch<{ song: any }>(`/songs/${id}`, data, token);
    return { song: normalizeSong(resp.song) };
  },
  delete: (id: string, token: string) => del<{ success: boolean }>(`/songs/${id}`, token),
  deleteAll: (token: string) => del<{ success: boolean; deletedCount: number }>('/songs', token),
  bulkDelete: (ids: string[], token: string) =>
    post<{ success: boolean; deletedCount: number }>('/songs/bulk-delete', { ids }, token),
  nukeGenerations: (token: string) =>
    post<{ success: boolean; songsDeleted: number; filesDeleted: number; lireekAudioGensDeleted: number }>(
      '/songs/nuke-generations', {}, token
    ),
  crop: (id: string, inPoint: number, outPoint: number, token: string, audioUrl?: string) =>
    post<{ cropped: boolean; newDuration: number }>(`/songs/${id}/crop`, { inPoint, outPoint, audioUrl }, token),
};

// ── Generation ──────────────────────────────────────────────
export const generateApi = {
  submit: (params: GenerationParams, token: string) =>
    post<{ jobId: string; status: string }>('/generate', params, token),
  status: (jobId: string) => get<GenerationJob>(`/generate/status/${jobId}`),
  cancel: (jobId: string) => post<{ success: boolean }>(`/generate/cancel/${jobId}`),
  cancelAll: () => post<{ success: boolean; cancelled: number }>('/generate/cancel-all'),
};

// ── Models ──────────────────────────────────────────────────
export const modelApi = {
  list: () => get<AceModels>('/models'),
  health: () => get<{ aceServer: string }>('/models/health'),
};

// ── Health ──────────────────────────────────────────────────
export const healthApi = {
  check: () => get<{
    status: string;
    aceServer: { status: string; url: string; version: string };
    server: { port: number; uptime: number };
  }>('/health'),
};

// ── Shutdown ────────────────────────────────────────────────
export const shutdownApi = {
  quit: () => post<{ success: boolean; message: string }>('/shutdown'),
};
// ── Mastering ───────────────────────────────────────────────
export const masteringApi = {
  /** Upload a reference track */
  uploadReference: async (file: File, token: string): Promise<{ name: string; path: string; url: string }> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/mastering/upload-reference`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    if (!res.ok) throw new Error('Upload failed');
    return res.json();
  },
  /** List uploaded reference tracks */
  listReferences: () => get<{ references: Array<{ name: string; size: number; url: string }> }>('/mastering/references'),
  /** Delete a reference track */
  deleteReference: (name: string, token: string) => del<{ ok: boolean }>(`/mastering/references/${name}`, token),
  /** Run mastering on an existing song */
  run: (songId: string, referenceName: string, token: string) =>
    post<{ ok: boolean; masteredUrl: string; songId: string }>('/mastering/run', { songId, referenceName }, token),
};

// ── Adapters ────────────────────────────────────────────────
export const adapterApi = {
  /** Browse directory — returns entries (dirs + filtered files) */
  browse: (dirPath: string, filter?: string) =>
    get<{ current: string; entries: BrowseEntry[] }>(
      `/adapters/browse?path=${encodeURIComponent(dirPath)}${filter ? `&filter=${encodeURIComponent(filter)}` : ''}`
    ),
  /** Scan folder for .safetensors files */
  scan: (folder: string) =>
    post<{ files: AdapterFile[] }>('/adapters/scan', { folder }),
};

// ── VST3 Post-Processing ────────────────────────────────────
export interface VstPlugin {
  name: string;
  vendor: string;
  version: string;
  path: string;
  uid: string;
  subcategories: string;
}

export interface VstChainEntry {
  uid: string;
  name: string;
  vendor: string;
  path: string;
  enabled: boolean;
  statePath: string;
}

export const vstApi = {
  /** Scan for installed VST3 plugins */
  scan: () => get<{ plugins: VstPlugin[] }>('/vst/scan'),
  /** Get current chain config */
  getChain: () => get<{ plugins: VstChainEntry[] }>('/vst/chain'),
  /** Update chain config */
  updateChain: (plugins: VstChainEntry[]) =>
    fetch(`${BASE}/vst/chain`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plugins }),
    }).then(r => r.json()) as Promise<{ plugins: VstChainEntry[] }>,
  /** Launch plugin GUI */
  openGui: (pluginPath: string, uid?: string) =>
    post<{ ok: boolean; pid: number }>('/vst/gui', { pluginPath, uid }),
  /** Start real-time monitor */
  monitorStart: (trackPath: string) =>
    post<{ ok: boolean; pid: number; plugins: number }>('/vst/monitor/start', { trackPath }),
  /** Stop monitor */
  monitorStop: () =>
    post<{ ok: boolean; wasRunning: boolean }>('/vst/monitor/stop', {}),
  /** Switch monitor to a different track */
  monitorSwitch: (trackPath: string) =>
    post<{ ok: boolean }>('/vst/monitor/switch', { trackPath }),
  /** Get monitor status */
  monitorStatus: () =>
    get<{ running: boolean; pid: number | null; position: number; duration: number }>('/vst/monitor/status'),
  /** Seek monitor to a position */
  monitorSeek: (position: number) =>
    post<{ ok: boolean }>('/vst/monitor/seek', { position }),
};

// ── Settings / .env ─────────────────────────────────────────
export const settingsApi = {
  /** Read current .env values for exposed keys */
  getEnv: () => get<{ values: Record<string, string>; restartKeys: string[] }>('/settings/env'),
  /** Update .env values (partial — only send changed keys) */
  updateEnv: (values: Record<string, string>) =>
    post<{ updated: string[]; restartRequired: boolean }>('/settings/env', values),
};
