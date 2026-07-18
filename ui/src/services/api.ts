// api.ts — Frontend API client
//
// Thin wrapper around fetch() for all server endpoints.
// Each method is standalone — import only what you need.

import type { Song, UnifiedRecentSong, GenerationParams, GenerationJob, AuthState, AceModels, BrowseEntry, AdapterFile, ModelRegistry } from '../types';

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
    metadata_overrides: s.metadata_overrides,
    created_at: s.created_at,
    createdAt: s.created_at ? new Date(s.created_at) : undefined,
    masteredAudioUrl: s.mastered_audio_url || s.masteredAudioUrl || '',
    mastered_audio_url: s.mastered_audio_url,
    latentUrl: s.latent_url || s.latentUrl || '',
    latent_url: s.latent_url,
    quality_scores: s.quality_scores,
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
  listIds: (token: string) => get<{ ids: string[] }>('/songs/ids', token),
  getRecentSongs: (token: string, source?: string, limit = 50) =>
    get<{ songs: UnifiedRecentSong[] }>(
      `/songs/recent?limit=${limit}${source && source !== 'all' ? `&source=${source}` : ''}`, token
    ),
};

// ── Generation ──────────────────────────────────────────────
export const generateApi = {
  submit: (params: GenerationParams, token: string) =>
    post<{ jobId: string; status: string }>('/generate', params, token),
  status: (jobId: string) => get<GenerationJob>(`/generate/status/${jobId}`),
  cancel: (jobId: string) => post<{ success: boolean }>(`/generate/cancel/${jobId}`),
  cancelAll: () => post<{ success: boolean; cancelled: number }>('/generate/cancel-all'),
  queueStatus: () => get<{
    depth: number;
    running: boolean;
    current: {
      id: string;
      status: string;
      stage: string;
      progress: number;
      age: number;
      aceJobId?: string;
    } | null;
    pending: number;
  }>('/generate/queue'),
  resetQueue: () => post<{ success: boolean; cancelled: number; drained: number }>('/generate/reset-queue'),
};

// ── Song Builder (Udio-style section-by-section generation) ──
export interface BuilderProject {
  id: string;
  user_id: string;
  title: string;
  style: string;
  bpm: number;
  key_scale: string;
  time_signature: string;
  vocal_language: string;
  section_length: number;
  variant_count: number;
  gen_params: string;
  created_at: string;
  updated_at: string;
  section_count?: number;
}

export type BuilderDirection = 'first' | 'append' | 'prepend';
export type BuilderSectionStatus = 'pending' | 'generating' | 'ready' | 'chosen' | 'failed';

export interface BuilderSection {
  id: string;
  project_id: string;
  position: number;
  label: string;
  lyrics: string;
  direction: BuilderDirection;
  section_length: number;
  candidate_song_ids: string[];
  candidates: Song[];
  chosen: Song | null;
  chosen_song_id: string | null;
  job_id: string | null;
  status: BuilderSectionStatus;
  created_at: string;
  updated_at: string;
}

function normalizeSection(s: any): BuilderSection {
  return {
    ...s,
    candidate_song_ids: s.candidate_song_ids || [],
    candidates: (s.candidates || []).map(normalizeSong),
    chosen: s.chosen ? normalizeSong(s.chosen) : null,
  };
}

export interface BuilderSectionInput {
  position?: number;
  label?: string;
  lyrics?: string;
  direction?: BuilderDirection;
  sectionLength?: number;
  candidateSongIds?: string[];
  chosenSongId?: string | null;
  jobId?: string | null;
  status?: BuilderSectionStatus;
}

export const builderApi = {
  listProjects: (token: string) =>
    get<{ projects: BuilderProject[] }>('/builder/projects', token),
  getProject: async (id: string, token: string) => {
    const data = await get<{ project: BuilderProject; sections: any[] }>(`/builder/projects/${id}`, token);
    return { project: data.project, sections: data.sections.map(normalizeSection) };
  },
  createProject: (body: Partial<BuilderProject> & { keyScale?: string; timeSignature?: string; vocalLanguage?: string; sectionLength?: number; variantCount?: number; genParams?: unknown }, token: string) =>
    post<{ project: BuilderProject; sections: BuilderSection[] }>('/builder/projects', body, token),
  updateProject: async (id: string, body: Record<string, unknown>, token: string) => {
    const data = await patch<{ project: BuilderProject; sections: any[] }>(`/builder/projects/${id}`, body, token);
    return { project: data.project, sections: data.sections.map(normalizeSection) };
  },
  deleteProject: (id: string, token: string) => del<{ ok: boolean }>(`/builder/projects/${id}`, token),

  createSection: async (projectId: string, body: BuilderSectionInput, token: string) => {
    const data = await post<{ section: any }>(`/builder/projects/${projectId}/sections`, body, token);
    return { section: normalizeSection(data.section) };
  },
  updateSection: async (sectionId: string, body: BuilderSectionInput, token: string) => {
    const data = await patch<{ section: any }>(`/builder/sections/${sectionId}`, body, token);
    return { section: normalizeSection(data.section) };
  },
  deleteSection: (sectionId: string, token: string) => del<{ ok: boolean }>(`/builder/sections/${sectionId}`, token),
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
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(errBody.error || `Upload failed (${res.status})`);
    }
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
    post<{ updated: string[]; restartRequired: boolean }>('/settings/env', { values }),
  /** Detect available GPUs via nvidia-smi */
  getGpus: () => get<{ gpus: Array<{ index: number; name: string; memoryMB: number }> }>('/settings/gpus'),
};

// ── Parameter Profiles ──────────────────────────────────────
export interface ParamProfile {
  name: string;
  saved_at: string;
  data: Record<string, unknown>;
}

export const profileApi = {
  /** List all saved profiles (full data inline) */
  list: () => get<{ profiles: ParamProfile[]; count: number }>('/profiles'),
  /** Save or overwrite a named profile */
  save: (name: string, data: Record<string, unknown>) =>
    post<{ ok: boolean; name: string; saved_at: string }>('/profiles', { name, data }),
  /** Rename a profile (data unchanged) */
  rename: (name: string, newName: string) =>
    patch<{ ok: boolean; name: string }>(`/profiles/${encodeURIComponent(name)}`, { newName }),
  /** Delete a profile */
  remove: (name: string) =>
    del<{ ok: boolean; deleted: string }>(`/profiles/${encodeURIComponent(name)}`),
};

// ── Model Manager ───────────────────────────────────────────
export const modelManagerApi = {
  /** Get full model registry with installed status */
  registry: () => get<ModelRegistry>('/model-manager/registry'),
  /** Start downloading a model file */
  download: (fileId: string) => post<{ jobId: string }>('/model-manager/download', { fileId }),
  /** Cancel an active download */
  cancel: (jobId: string) => post<{ ok: boolean }>(`/model-manager/download/${jobId}/cancel`),
  /** Resume a paused/failed download */
  resume: (jobId: string) => post<{ jobId: string }>(`/model-manager/download/${jobId}/resume`),
  /** Delete an installed model file */
  deleteFile: (filename: string) => del<{ ok: boolean }>(`/model-manager/files/${encodeURIComponent(filename)}`),
  /** SSE endpoint URL for download progress */
  downloadsStreamUrl: `${BASE}/model-manager/downloads`,
};

// ── Retranscribe Lyrics ─────────────────────────────────────
export async function retranscribeLyrics(
  songId: string,
  options?: { model?: string; language?: string; beamSize?: number }
): Promise<{ success: boolean; lineCount: number; wordCount: number }> {
  const res = await fetch(`/api/songs/${songId}/retranscribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options || {}),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return res.json();
}
