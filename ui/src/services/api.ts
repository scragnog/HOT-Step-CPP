// api.ts — Frontend API client
//
// Thin wrapper over the shared client (httpClient.ts → axiosClient.ts) for all
// server endpoints; each method is standalone — import only what you need.
// The bearer comes from the auth session accessor, so no function here takes
// a token.

import type { Song, UnifiedRecentSong, GenerationParams, GenerationJob, AuthState, AceModels, ModelRegistry } from '../types';
import { getGenerationTimeoutMinutes } from '../utils/generationTimer';
import type { ResolveIntent, ResolvePreviewResponse } from '../../../server/src/contracts/resolution';
import type { DownloadStartResponse } from '../../../server/src/contracts/modelManager';
import type { HealthResponse } from '../../../server/src/contracts/health';
import type { RestartResponse, ShutdownResponse } from '../../../server/src/contracts/shutdown';
import type {
  UploadReferenceResponse, ListReferencesResponse, DeleteReferenceResponse, RunMasteringResponse,
} from '../../../server/src/contracts/mastering';
import type { BrowseResponse, ScanResponse, LmAdaptersResponse } from '../../../server/src/contracts/adapters';
import type {
  VstPlugin, ChainEntry, ScanPluginsResponse, GetChainResponse, UpdateChainResponse,
  GuiResponse, MonitorStartResponse, MonitorStopResponse, MonitorSwitchResponse,
  MonitorStatusResponse, MonitorSeekResponse,
} from '../../../server/src/contracts/vst';
import type { ReportLogsResponse } from '../../../server/src/contracts/reportLogs';
import type { EngineWorkersResponse, EnvResponse, EnvUpdateResponse, GpusResponse } from '../../../server/src/contracts/settings';
import type {
  ProfileFile, ListProfilesResponse, SaveProfileResponse, RenameProfileResponse, DeleteProfileResponse,
} from '../../../server/src/contracts/profiles';
import { apiClient, ApiError, HttpStatus, type ClientOptions, type ReadOptions } from './httpClient';
import { SLOW_ROUTE_TIMEOUT_MS } from './axiosClient';

const BASE = '/api';

// get/post/patch/del delegate to the shared client (httpClient.ts) — same
// signatures and error shape (ApiError, a plain Error subclass with status +
// body) every caller in this file already expects. Every read takes a trailing
// `opts?: ReadOptions` so a query function can pass its AbortSignal; writes
// take the wider ClientOptions when they need silent/timeout overrides.
const get = <T>(path: string, opts?: ReadOptions): Promise<T> => apiClient.get<T>(path, { signal: opts?.signal });
const post = <T>(path: string, body?: unknown, opts?: ClientOptions): Promise<T> => apiClient.post<T>(path, body, opts);
const patch = <T>(path: string, body: unknown, opts?: ClientOptions): Promise<T> => apiClient.patch<T>(path, body, opts);
const del = <T>(path: string, opts?: ClientOptions): Promise<T> => apiClient.delete<T>(path, opts);

// ── Auth ────────────────────────────────────────────────────
export const authApi = {
  autoLogin: (opts?: ReadOptions) => get<AuthState>('/auth/auto', opts),
  getMe: (opts?: ReadOptions) => get<{ user: AuthState['user'] }>('/auth/me', opts),
  updateUsername: (username: string) => patch<AuthState>('/auth/username', { username }),
};

// ── Song Normalizer ─────────────────────────────────────────
/** Map DB snake_case fields to camelCase for component consumption */
export function normalizeSong(s: any): Song {
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
    noAdapterAudioUrl: s.noadapter_audio_url || s.noAdapterAudioUrl || '',
    noadapter_audio_url: s.noadapter_audio_url,
    latentUrl: s.latent_url || s.latentUrl || '',
    latent_url: s.latent_url,
    quality_scores: s.quality_scores,
  };
}

// ── Songs ────────────────────────────────────────────────────
export const songApi = {
  list: async (opts?: ReadOptions) => {
    const data = await get<{ songs: any[] }>('/songs', opts);
    return { songs: data.songs.map(normalizeSong) };
  },
  get: async (id: string, opts?: ReadOptions) => {
    const data = await get<{ song: any }>(`/songs/${id}`, opts);
    return { song: normalizeSong(data.song) };
  },
  create: (song: Partial<Song>) => post<{ song: Song }>('/songs', song),
  update: async (id: string, data: Partial<Song>) => {
    const resp = await patch<{ song: any }>(`/songs/${id}`, data);
    return { song: normalizeSong(resp.song) };
  },
  delete: (id: string) => del<{ success: boolean }>(`/songs/${id}`),
  deleteAll: () => del<{ success: boolean; deletedCount: number }>('/songs'),
  bulkDelete: (ids: string[]) =>
    post<{ success: boolean; deletedCount: number }>('/songs/bulk-delete', { ids }),
  nukeGenerations: () =>
    post<{ success: boolean; songsDeleted: number; filesDeleted: number; lireekAudioGensDeleted: number }>(
      '/songs/nuke-generations', {}
    ),
  // Trim mode's crop (player/TrimControls) renders its own inline error line, so
  // it is silent — that line is the sole notifier on failure (R2). Sole consumer.
  crop: (id: string, inPoint: number, outPoint: number, audioUrl?: string) =>
    post<{ cropped: boolean; newDuration: number }>(`/songs/${id}/crop`, { inPoint, outPoint, audioUrl }, { silent: true }),
  listIds: (opts?: ReadOptions) => get<{ ids: string[] }>('/songs/ids', opts),
  getRecentSongs: (source?: string, limit = 50, opts?: ReadOptions) =>
    get<{ songs: UnifiedRecentSong[] }>(
      `/songs/recent?limit=${limit}${source && source !== 'all' ? `&source=${source}` : ''}`, opts
    ),
  /**
   * Import audio files from disk into the library. Each becomes a song row
   * backed by a raw WAV, which is what post-processing, stems and export all
   * expect — nothing here touches the generation engine.
   *
   * A file that cannot be read comes back in `errors` while the rest still
   * import; only an all-failed request rejects.
   */
  /** XHR under the hood (apiClient.upload): an album of FLACs is a long
   *  upload, and a progress bar is the difference between "working" and
   *  "hung". */
  importTracks: async (
    files: File[],
    opts?: { description?: string; onProgress?: (fraction: number) => void },
  ): Promise<{ songs: Song[]; errors: { file: string; error: string }[] }> => {
    const raw = await apiClient.upload<any>('/songs/import', files.map(file => ({ field: 'audio', file })), {
      // Before the files, so it is parsed early whatever the server does with it.
      fields: opts?.description ? { description: opts.description } : undefined,
      onProgress: opts?.onProgress,
    });
    return {
      songs: (raw.songs || []).map(normalizeSong),
      errors: raw.errors || [],
    };
  },
};

// ── Generation ──────────────────────────────────────────────

/** Stamp the user's configured generation timeout onto an outbound request.
 *
 *  The server clamps `generationTimeoutMinutes` and falls back to 45 min when
 *  the field is absent (routes/generate.ts pollUntilDone). Only the Create
 *  queue used to set it, so Cover Studio, Repaint Studio, InstaGen and Song
 *  Builder all silently ran on the 45 min default no matter what Settings said
 *  (issue #97). Stamping it here means every caller of submit() gets it, and a
 *  new studio can't reintroduce the same gap.
 *
 *  A caller that already set the field keeps its own value. */
function withTimeout(params: GenerationParams): GenerationParams {
  if (typeof (params as any).generationTimeoutMinutes === 'number') return params;
  return { ...params, generationTimeoutMinutes: getGenerationTimeoutMinutes() } as GenerationParams;
}

/** The same default for an intent the server resolves, in its params and, for
 *  a written song, its app settings. Values the intent already has stay. */
export function withIntentTimeout<T extends ResolveIntent>(intent: T): T {
  const timeout = getGenerationTimeoutMinutes();
  const params = typeof intent.params.generationTimeoutMinutes === 'number'
    ? intent.params : { ...intent.params, generationTimeoutMinutes: timeout };
  return intent.kind === 'written-song'
    ? { ...intent, params, settings: { ...intent.settings,
        app: { ...intent.settings?.app, generationTimeoutMinutes:
          typeof intent.settings?.app?.generationTimeoutMinutes === 'number'
            ? intent.settings.app.generationTimeoutMinutes : timeout } } }
    : { ...intent, params };
}

export const generateApi = {
  preview: (intent: ResolveIntent) => post<ResolvePreviewResponse>('/resolve/preview', intent),
  /** A missing switch endpoint is an older server and keeps the legacy path. */
  selectedPath: async (opts?: ReadOptions): Promise<'old' | 'resolved'> => {
    try {
      const { path } = await apiClient.get<{ path: string }>('/resolve/path', { signal: opts?.signal });
      return path === 'resolved' ? 'resolved' : 'old';
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === HttpStatus.notFound) return 'old';
        throw new Error((err.body as { error?: string } | null)?.error || 'Cannot read generation path');
      }
      throw err;
    }
  },
  previewIntent: (intent: ResolveIntent): Promise<ResolvePreviewResponse> =>
    generateApi.preview(withIntentTimeout(intent)),
  resolveIfSelected: async (intent: ResolveIntent): Promise<ResolvePreviewResponse | null> =>
    (await generateApi.selectedPath()) === 'resolved' ? generateApi.previewIntent(intent) : null,
  submit: (params: GenerationParams) => post<{ jobId: string; status: string }>('/generate', withTimeout(params)),
  /** YuE2 score preview: plan the lead sheet only and return it, no audio.
   *  `seed` is the one the plan used, so the follow-up render can pin it. */
  yue2Plan: (params: Partial<GenerationParams>) =>
    post<{ abc: string; seed: number; end_reason: string; health: { verdict: string; reason: string; bars: number; estSeconds?: number; sections: string[] } }>(
      '/generate/yue2/plan', params),
  status: (jobId: string, opts?: ReadOptions) => get<GenerationJob>(`/generate/status/${jobId}`, opts),
  cancel: (jobId: string) => post<{ success: boolean }>(`/generate/cancel/${jobId}`),
  cancelAll: () => post<{ success: boolean; cancelled: number }>('/generate/cancel-all'),
  queueStatus: (opts?: ReadOptions) => get<{
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
  }>('/generate/queue', opts),
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
  /** Bumped by every edit; section edits send the one they were based on. */
  revision: number;
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

/** The composer's captured state for the next section; Node works out the
 *  geometry and the request (server/src/services/songBuilder/sectionOps.ts). */
export interface GenerateSectionInput {
  idempotencyKey: string;
  expectedRevision: number;
  /** The engine active when the user pressed Generate. */
  expectedBackend: string;
  direction: BuilderDirection;
  label: string;
  lyrics: string;
  length: { bars: number } | { seconds: number };
  overlap: number;
  clipPoint: number | null;
  seedSectionId: string | null;
  seedStrength: number;
  previewMastering: boolean;
  coResident: boolean;
  headDuration?: number;
  engineParams: Record<string, unknown>;
}

type BuilderView = { project: BuilderProject; sections: BuilderSection[] };

/** Section operations throw ApiError, so a stale revision (409 with
 *  currentRevision) can be told apart and the project reloaded. */
async function builderOp<T extends { project: BuilderProject; sections: any[] }>(path: string, method: 'POST' | 'PATCH' | 'DELETE', body?: unknown, opts?: ClientOptions): Promise<T> {
  const data = await apiClient.request<T>(method, `/builder${path}`, body, opts);
  return { ...data, sections: data.sections.map(normalizeSection) };
}

export const builderApi = {
  listProjects: (opts?: ReadOptions) => get<{ projects: BuilderProject[] }>('/builder/projects', opts),
  getProject: async (id: string, opts?: ReadOptions) => {
    const data = await get<{ project: BuilderProject; sections: any[] }>(`/builder/projects/${id}`, opts);
    return { project: data.project, sections: data.sections.map(normalizeSection) };
  },
  createProject: (body: Partial<BuilderProject> & { keyScale?: string; timeSignature?: string; vocalLanguage?: string; sectionLength?: number; variantCount?: number; genParams?: unknown }) =>
    post<{ project: BuilderProject; sections: BuilderSection[] }>('/builder/projects', body),
  updateProject: async (id: string, body: Record<string, unknown>) => {
    const data = await patch<{ project: BuilderProject; sections: any[] }>(`/builder/projects/${id}`, body);
    return { project: data.project, sections: data.sections.map(normalizeSection) };
  },
  deleteProject: (id: string) => del<{ ok: boolean }>(`/builder/projects/${id}`),

  /** Start the next section; follow `jobId` with followJob. The section ops
   *  take ClientOptions because SongBuilder displays their failures itself
   *  (409 → reload, everything else a prefixed line) and passes { silent: true }. */
  generateSection: (projectId: string, body: GenerateSectionInput, opts?: ClientOptions) =>
    builderOp<BuilderView & { jobId: string; sectionId: string }>(`/projects/${projectId}/sections/generate`, 'POST', body, opts),
  chooseCandidate: (sectionId: string, songId: string, expectedRevision: number, opts?: ClientOptions) =>
    builderOp<BuilderView>(`/sections/${sectionId}/choose`, 'POST', { songId, expectedRevision }, opts),
  stopSection: (sectionId: string, expectedRevision: number, opts?: ClientOptions) =>
    builderOp<BuilderView>(`/sections/${sectionId}/stop`, 'POST', { expectedRevision }, opts),
  editSection: (sectionId: string, edit: { label?: string; lyrics?: string }, expectedRevision: number, opts?: ClientOptions) =>
    builderOp<BuilderView>(`/sections/${sectionId}`, 'PATCH', { ...edit, expectedRevision }, opts),
  deleteSection: (sectionId: string, expectedRevision: number, opts?: ClientOptions) =>
    builderOp<BuilderView>(`/sections/${sectionId}?expectedRevision=${expectedRevision}`, 'DELETE', undefined, opts),
};

// ── Models ──────────────────────────────────────────────────
export const modelApi = {
  list: (opts?: ReadOptions) => get<AceModels>('/models', opts),
  health: (opts?: ReadOptions) => get<{ aceServer: string }>('/models/health', opts),
};

// ── Health ──────────────────────────────────────────────────
export const healthApi = {
  check: (opts?: ReadOptions) => get<HealthResponse>('/health', opts),
};

// ── Shutdown ────────────────────────────────────────────────
export const shutdownApi = {
  quit: () => post<ShutdownResponse>('/shutdown', {}, { silent: true }),
  /** Ask the server to restart itself (applies restart-required .env keys).
   *  The connection is torn down as the server goes, so a rejected promise is
   *  the expected outcome — silent, and callers swallow it. */
  restart: () => post<RestartResponse>('/shutdown/restart', {}, { silent: true }),
};
// ── Mastering ───────────────────────────────────────────────
export const masteringApi = {
  /** Upload a reference track (XHR under the hood, for progress). */
  uploadReference: (file: File) => apiClient.upload<UploadReferenceResponse>('/mastering/upload-reference', [{ field: 'file', file }]),
  /** List uploaded reference tracks */
  listReferences: (opts?: ReadOptions) => get<ListReferencesResponse>('/mastering/references', opts),
  /** Delete a reference track */
  deleteReference: (name: string) => del<DeleteReferenceResponse>(`/mastering/references/${name}`),
  /** Run mastering on an existing song — blocks server-side for minutes. */
  run: (songId: string, referenceName: string) =>
    post<RunMasteringResponse>('/mastering/run', { songId, referenceName }, { timeout: SLOW_ROUTE_TIMEOUT_MS }),
};

// ── Adapters ────────────────────────────────────────────────
export const adapterApi = {
  /** Browse directory — returns entries (dirs + filtered files). Slow on
   *  network filesystems, so it gets the slow-route timeout. */
  browse: (dirPath: string, filter?: string, opts?: ReadOptions) =>
    apiClient.get<BrowseResponse>(
      `/adapters/browse?path=${encodeURIComponent(dirPath)}${filter ? `&filter=${encodeURIComponent(filter)}` : ''}`,
      { ...opts, timeout: SLOW_ROUTE_TIMEOUT_MS },
    ),
  /** Scan folder for .safetensors files. A POST, but a pure read; a recursive
   *  scan of a large library can take minutes. Silent: AdaptersDropdown's scan
   *  command renders the failure in its own inline line, so the transport must
   *  not also toast. */
  scan: (folder: string, opts?: ReadOptions) =>
    apiClient.post<ScanResponse>('/adapters/scan', { folder }, { signal: opts?.signal, timeout: SLOW_ROUTE_TIMEOUT_MS, silent: true }),
  /** Planner-LM adapters (local HOT-Step feature). Scans `folder` when given,
   *  else every per-size root (lm-06b/lm-17b/lm-4b) plus the legacy flat lm/.
   *  `lmSize` comes from the parent folder (new layout) or the legacy -<size>
   *  name suffix; `trigger` is the embedded trigger word when present. */
  lmList: (folder?: string, opts?: ReadOptions) =>
    get<LmAdaptersResponse>(
      `/adapters/lm${folder ? `?folder=${encodeURIComponent(folder)}` : ''}`, opts),
};

// ── VST3 Post-Processing ────────────────────────────────────
// VstPlugin and ChainEntry are shared with the server
// (server/src/contracts/vst.ts); re-exported under VstChainEntry, this
// file's existing public name for it.
export type { VstPlugin };
export type VstChainEntry = ChainEntry;

export const vstApi = {
  /** Scan for installed VST3 plugins */
  scan: (opts?: ReadOptions) => get<ScanPluginsResponse>('/vst/scan', opts),
  /** Get current chain config */
  getChain: (opts?: ReadOptions) => get<GetChainResponse>('/vst/chain', opts),
  /** Update chain config */
  updateChain: (plugins: VstChainEntry[]) => apiClient.put<UpdateChainResponse>('/vst/chain', { plugins }),
  /** Launch plugin GUI */
  openGui: (pluginPath: string, uid?: string) =>
    post<GuiResponse>('/vst/gui', { pluginPath, uid }),
  /** Chain uids whose editor window process is still running */
  openGuis: (opts?: ReadOptions) => get<{ uids: string[] }>('/vst/gui/open', opts),
  /** Start real-time monitor */
  monitorStart: (trackPath: string) =>
    post<MonitorStartResponse>('/vst/monitor/start', { trackPath }),
  /** Stop monitor */
  monitorStop: () =>
    post<MonitorStopResponse>('/vst/monitor/stop', {}),
  /** Switch monitor to a different track */
  monitorSwitch: (trackPath: string) =>
    post<MonitorSwitchResponse>('/vst/monitor/switch', { trackPath }),
  /** Get monitor status */
  monitorStatus: (opts?: ReadOptions) =>
    get<MonitorStatusResponse>('/vst/monitor/status', opts),
  /** Seek monitor to a position */
  monitorSeek: (position: number) =>
    post<MonitorSeekResponse>('/vst/monitor/seek', { position }),
};

// ── Settings / .env ─────────────────────────────────────────
export const settingsApi = {
  /** Read current .env values for exposed keys */
  getEnv: (opts?: ReadOptions) => get<EnvResponse>('/settings/env', opts),
  /** Update .env values (partial — only send changed keys). Silent: the
   *  settings env form renders this failure itself (its status line), so a
   *  transport toast would be a second copy of the same error. */
  updateEnv: (values: Record<string, string>) =>
    post<EnvUpdateResponse>('/settings/env', { values }, { silent: true }),
  /** Detect available GPUs via nvidia-smi. `uuid` is what the picker stores in
   *  CUDA_VISIBLE_DEVICES — the index is display-only, because nvidia-smi and
   *  CUDA number GPUs differently (issue #153). */
  getGpus: (opts?: ReadOptions) => get<GpusResponse>('/settings/gpus', opts),
  /** Per-worker engine status for the experimental multi-GPU section. */
  getEngineWorkers: (opts?: ReadOptions) => get<EngineWorkersResponse>('/settings/engine-workers', opts),
};

export const reportLogsApi = {
  snapshot: (opts?: ReadOptions) => get<ReportLogsResponse>('/logs/snapshot', opts),
};

// ── Parameter Profiles ──────────────────────────────────────
// ParamProfile is shared with the server (server/src/contracts/profiles.ts),
// re-exported under this file's existing public name for it.
export type ParamProfile = ProfileFile;

// Profiles are per engine: `backend` is the backend id they belong to.
export const profileApi = {
  /** List one engine's saved profiles (full data inline) */
  list: (backend: string, opts?: ReadOptions) =>
    get<ListProfilesResponse>(`/profiles?backend=${encodeURIComponent(backend)}`, opts),
  /** Save or overwrite a named profile. Silent: ProfilesModal renders the
   *  mutation failure in its own error line. */
  save: (backend: string, name: string, data: Record<string, unknown>) =>
    post<SaveProfileResponse>('/profiles', { name, data, backend }, { silent: true }),
  /** Rename a profile (data unchanged). Silent: ProfilesModal's error line. */
  rename: (backend: string, name: string, newName: string) =>
    patch<RenameProfileResponse>(`/profiles/${encodeURIComponent(name)}?backend=${encodeURIComponent(backend)}`, { newName }, { silent: true }),
  /** Delete a profile. Silent: ProfilesModal's error line. */
  remove: (backend: string, name: string) =>
    del<DeleteProfileResponse>(`/profiles/${encodeURIComponent(name)}?backend=${encodeURIComponent(backend)}`, { silent: true }),
};

// ── Model Manager ───────────────────────────────────────────
export const modelManagerApi = {
  /** Get full model registry with installed status */
  registry: (opts?: ReadOptions) => get<ModelRegistry>('/model-manager/registry', opts),
  /** Start downloading a model file. Optional hfToken is forwarded to
   *  huggingface.co as a Bearer token (gated repos only). */
  download: (fileId: string, hfToken?: string) =>
    post<DownloadStartResponse>('/model-manager/download', hfToken ? { fileId, hfToken } : { fileId }),
  /** Cancel an active download */
  cancel: (jobId: string) => post<{ ok: boolean }>(`/model-manager/download/${jobId}/cancel`),
  /** Resume a paused/failed download */
  resume: (jobId: string) => post<DownloadStartResponse>(`/model-manager/download/${jobId}/resume`),
  /** Delete an installed model file */
  deleteFile: (filename: string) => del<{ ok: boolean }>(`/model-manager/files/${encodeURIComponent(filename)}`),
  /** SSE endpoint URL for download progress */
  downloadsStreamUrl: `${BASE}/model-manager/downloads`,
};

// ── Retranscribe Lyrics ─────────────────────────────────────
export interface PostProcessJobStatus {
  jobId: string;
  songId?: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  stage: string;
  error?: string;
  masteredAudioUrl?: string;
}

/**
 * Run the post-processing chain over an already-rendered song.
 *
 * `params` is the full getGlobalParams() payload built with the master PP
 * toggle forced on — deliberately not a hand-picked subset of PP keys, so a
 * knob added later cannot go quietly missing on this path. The server forces
 * the master switch itself and replaces the fields that belong to the song
 * (instrumental, caption, seed) rather than trusting the current form.
 *
 * Refused with 409 when the song already has a mastered version: the chain is
 * not idempotent and a second pass overcooks it.
 */
export async function runSongPostProcessing(
  songId: string,
  params: Record<string, any>
): Promise<{ jobId: string; status: string; stage: string; stages: string[] }> {
  try {
    // Silent: the post-process dock mirror renders its own inline error line.
    return await apiClient.post(`/songs/${songId}/postprocess`, params, { silent: true });
  } catch (err) {
    if (err instanceof ApiError) {
      const data = (err.body ?? {}) as { alreadyProcessed?: boolean; masteredAudioUrl?: string };
      // The already-processed refusal carries the truth about this song, and the
      // caller uses it to fix its own stale copy. Losing that detail to a bare
      // Error string is what left the UI insisting a mastered track was
      // unmastered — so it rides along on the ApiError itself.
      throw Object.assign(err, {
        alreadyProcessed: !!data.alreadyProcessed,
        masteredAudioUrl: data.masteredAudioUrl as string | undefined,
      });
    }
    throw err;
  }
}

/** Poll a post-processing re-run. Omit jobId to find whatever is in flight for
 *  the song, which is how a reloaded tab re-attaches to a running pass. */
export async function getSongPostProcessingStatus(
  songId: string,
  jobId?: string,
  opts?: ReadOptions,
): Promise<PostProcessJobStatus | null> {
  const qs = jobId ? `?jobId=${encodeURIComponent(jobId)}` : '';
  try {
    return await apiClient.get<PostProcessJobStatus>(`/songs/${songId}/postprocess${qs}`, { signal: opts?.signal });
  } catch (err) {
    if (err instanceof ApiError && err.status === HttpStatus.notFound) return null; // nothing in flight
    throw err;
  }
}

/**
 * Throw away a song's post-processed version — the raw render is untouched.
 *
 * The escape hatch for the no-double-cook rule: refusing a second pass is
 * right, but a track has to be able to get out of a bad first one.
 */
export function revertSongPostProcessing(songId: string): Promise<{ removed: boolean; songId: string }> {
  // Silent: same owner as runSongPostProcessing — the dock mirror's inline line.
  return apiClient.delete(`/songs/${songId}/postprocess`, { silent: true });
}

/** Re-runs Whisper transcription — the server runs it inline, so a full track
 *  can take minutes. */
export function retranscribeLyrics(
  songId: string,
  options?: { model?: string; language?: string; beamSize?: number; signal?: AbortSignal; force?: boolean }
): Promise<{ success: boolean; lineCount: number; wordCount: number; model?: string }> {
  const { signal, ...body } = options || {};
  // Silent: the retranscribe store owns the single toast for this route, so the
  // global interceptor must not add a second one with the same sentence.
  return apiClient.post(`/songs/${songId}/retranscribe`, body, {
    timeout: SLOW_ROUTE_TIMEOUT_MS,
    silent: true,
    ...(signal ? { signal } : {}),
  });
}

/**
 * Move a track's lyric timing by hand, in milliseconds (positive is later).
 * Both sidecars move together on the server, so exports follow.
 */
export function nudgeLyrics(audioUrl: string, deltaMs: number): Promise<{
  success: boolean;
  source: 'lyrics.json' | 'lrc';
  lines: number;
  appliedSec: number;
}> {
  return post('/songs/lyrics/offset', { audioUrl, deltaMs });
}

/**
 * Split the lyric line under the playhead in two at that moment. Word times are
 * left alone; the new line boundary is what makes the second half light up when
 * it is actually sung. The server rebuilds the `.lrc` to match.
 */
export function splitLyrics(audioUrl: string, atSec: number): Promise<{
  success: boolean;
  source: 'lyrics.json' | 'lrc';
  atSec: number;
  left: string;
  right: string;
  lines: number;
}> {
  return post('/songs/lyrics/split', { audioUrl, atSec });
}

/**
 * Replace a track's word timings with hand-placed ones from the lyric editor.
 * The server sorts them, recomputes each line's start/end from its words, and
 * rebuilds the `.lrc`.
 */
export function saveLyricTimings(audioUrl: string, lines: unknown[], expectedRevision?: number): Promise<{
  success: boolean;
  lines: number;
  words: number;
  revision: number;
}> {
  return post('/songs/lyrics/timings', { audioUrl, lines, expectedRevision });
}

/** Which Whisper model a retranscription would use, plus the installed ones. */
export function whisperModels(): Promise<{
  model: string | null;
  label: string | null;
  models: { file: string; label: string }[];
}> {
  return get('/songs/retranscribe/model');
}
