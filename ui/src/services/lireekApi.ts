/**
 * Lyric Studio API client — typed wrappers for /api/lireek/ endpoints.
 * Ported from hot-step-9000 lyricStudioApi.ts for the cpp engine.
 */

const API_BASE = '';

async function api<T>(endpoint: string, options: {
  method?: string;
  body?: unknown;
  timeoutMs?: number;
} = {}): Promise<T> {
  const { method = 'GET', body, timeoutMs = 10000 } = options;
  const headers: HeadersInit = { 'Content-Type': 'application/json' };

  // Abort controller with timeout to prevent connection pool hangs
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Request failed' }));
      throw new Error(error.detail || error.error || error.message || `Request failed (${response.status})`);
    }

    return response.json();
  } finally {
    clearTimeout(timer);
  }
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface Artist {
  id: number;
  name: string;
  image_url?: string;
  genius_id?: number;
  lyrics_set_count?: number;
  created_at: string;
}

export interface LyricsSet {
  id: number;
  artist_id: number;
  artist_name: string;
  album: string | null;
  image_url?: string;
  songs: SongLyric[] | string;
  max_songs: number;
  total_songs?: number;
  created_at: string;
}

export interface SongLyric {
  title: string;
  lyrics: string;
  url?: string;
}

export interface Profile {
  id: number;
  lyrics_set_id: number;
  provider: string;
  model: string;
  profile_data: Record<string, any>;
  created_at: string;
}

export interface Generation {
  id: number;
  profile_id: number;
  provider: string;
  model: string;
  lyrics: string;
  title?: string;
  subject?: string;
  caption?: string;
  bpm?: number;
  key?: string;
  duration?: number;
  parent_generation_id?: number;
  system_prompt?: string;
  user_prompt?: string;
  extra_instructions?: string;
  created_at: string;
  // Context fields (from /generations/all)
  artist_name?: string;
  album?: string;
}

export interface AlbumPreset {
  id: number;
  lyrics_set_id: number;
  adapter_path?: string;
  adapter_scale?: number;
  adapter_group_scales?: { self_attn: number; cross_attn: number; mlp: number; cond_embed: number };
  reference_track_path?: string;
  audio_cover_strength?: number;
  created_at: string;
}

export interface AudioGeneration {
  id: number;
  generation_id: number;
  hotstep_job_id: string;
  audio_url?: string;
  cover_url?: string;
  duration?: number;
  created_at: string;
}

export interface RecentSong {
  ag_id: number;
  hotstep_job_id: string;
  audio_url?: string;
  cover_url?: string;
  ag_created_at: string;
  generation_id: number;
  song_title: string;
  subject?: string;
  caption?: string;
  lyrics?: string;
  duration?: number;
  lyrics_set_id: number;
  album?: string;
  album_image?: string;
  artist_id: number;
  artist_name: string;
  artist_image?: string;
  mastered_audio_url?: string;
}

// ── API ─────────────────────────────────────────────────────────────────────

export const lireekApi = {
  // ── Artists ──────────────────────────────────────────────────────────────
  listArtists: (): Promise<{ artists: Artist[] }> =>
    api<Artist[]>('/api/lireek/artists').then(artists => ({ artists })),

  deleteArtist: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/artists/${id}`, { method: 'DELETE' }),

  refreshArtistImage: (id: number): Promise<{ image_url: string }> =>
    api(`/api/lireek/artists/${id}/refresh-image`, { method: 'POST' }),

  setArtistImage: (id: number, imageUrl: string): Promise<{ image_url: string }> =>
    api(`/api/lireek/artists/${id}/set-image`, { method: 'POST', body: { image_url: imageUrl } }),

  createArtist: (params: { name: string; image_url?: string }): Promise<{ artist: Artist }> =>
    api<Artist>('/api/lireek/artists/create', { method: 'POST', body: params }).then(artist => ({ artist })),

  // ── Lyrics Sets ─────────────────────────────────────────────────────────
  listLyricsSets: (artistId?: number): Promise<{ lyrics_sets: LyricsSet[] }> => {
    const params = new URLSearchParams();
    if (artistId != null) params.set('artist_id', String(artistId));
    const qs = params.toString();
    return api<LyricsSet[]>(`/api/lireek/lyrics-sets${qs ? `?${qs}` : ''}`).then(lyrics_sets => ({ lyrics_sets }));
  },

  getLyricsSet: (id: number): Promise<LyricsSet> =>
    api(`/api/lireek/lyrics-sets/${id}`),

  /** Fetch everything for the album detail page in a single call. */
  getAlbumFullDetail: (lyricsSetId: number): Promise<{
    lyrics_set: LyricsSet;
    profiles: Profile[];
    generations: Generation[];
    preset: AlbumPreset | null;
  }> => api(`/api/lireek/lyrics-sets/${lyricsSetId}/full-detail`),

  deleteLyricsSet: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/lyrics-sets/${id}`, { method: 'DELETE' }),

  removeSong: (lyricsSetId: number, songIndex: number): Promise<any> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/songs/${songIndex}`, { method: 'DELETE' }),

  editSong: (lyricsSetId: number, songIndex: number, lyrics: string): Promise<LyricsSet> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/songs/${songIndex}`, { method: 'PUT', body: { lyrics } }),

  addSongToSet: (lyricsSetId: number, params: { title: string; lyrics: string }): Promise<LyricsSet> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/add-song`, { method: 'POST', body: params }),

  refreshAlbumImage: (id: number): Promise<{ image_url: string }> =>
    api(`/api/lireek/lyrics-sets/${id}/refresh-image`, { method: 'POST' }),

  setAlbumImage: (id: number, imageUrl: string): Promise<{ image_url: string }> =>
    api(`/api/lireek/lyrics-sets/${id}/set-image`, { method: 'POST', body: { image_url: imageUrl } }),

  createLyricsSet: (params: {
    artist_id: number;
    album?: string;
    image_url?: string;
    songs?: { title: string; lyrics: string }[];
  }): Promise<{ lyrics_set: LyricsSet }> =>
    api('/api/lireek/lyrics-sets/create', { method: 'POST', body: params }),

  // ── Genius Fetch ────────────────────────────────────────────────────────
  fetchLyrics: (params: {
    artist: string;
    album?: string;
    max_songs?: number;
  }): Promise<{ artist: Artist; lyrics_set: LyricsSet; songs_fetched: number }> =>
    api('/api/lireek/fetch-lyrics', { method: 'POST', body: params, timeoutMs: 120_000 }),

  searchSongLyrics: (artist: string, title: string): Promise<{ title: string; lyrics: string }> =>
    api('/api/lireek/search-song-lyrics', { method: 'POST', body: { artist, title }, timeoutMs: 30_000 }),

  // ── Profiles ────────────────────────────────────────────────────────────
  listProfiles: (lyricsSetId?: number): Promise<{ profiles: Profile[] }> => {
    const params = new URLSearchParams();
    if (lyricsSetId != null) params.set('lyrics_set_id', String(lyricsSetId));
    const qs = params.toString();
    return api<Profile[]>(`/api/lireek/profiles${qs ? `?${qs}` : ''}`).then(profiles => ({ profiles }));
  },

  getProfile: (id: number): Promise<Profile> =>
    api(`/api/lireek/profiles/${id}`),

  deleteProfile: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/profiles/${id}`, { method: 'DELETE' }),

  buildProfile: (lyricsSetId: number, params: {
    provider: string;
    model?: string;
  }): Promise<Profile> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/build-profile`, { method: 'POST', body: params, timeoutMs: 300_000 }),

  // ── Generations ─────────────────────────────────────────────────────────
  listGenerations: (profileId?: number, lyricsSetId?: number): Promise<{ generations: Generation[] }> => {
    const params = new URLSearchParams();
    if (profileId != null) params.set('profile_id', String(profileId));
    if (lyricsSetId != null) params.set('lyrics_set_id', String(lyricsSetId));
    const qs = params.toString();
    return api<Generation[]>(`/api/lireek/generations${qs ? `?${qs}` : ''}`).then(generations => ({ generations }));
  },

  listAllGenerations: (): Promise<{ generations: Generation[] }> =>
    api('/api/lireek/generations/all'),

  getGeneration: (id: number): Promise<Generation> =>
    api(`/api/lireek/generations/${id}`),

  generateLyrics: (profileId: number, params: {
    profile_id: number;
    provider: string;
    model?: string;
    extra_instructions?: string;
  }): Promise<Generation> =>
    api(`/api/lireek/profiles/${profileId}/generate`, { method: 'POST', body: params, timeoutMs: 300_000 }),

  refineLyrics: (generationId: number, params: {
    provider: string;
    model?: string;
  }): Promise<Generation> =>
    api(`/api/lireek/generations/${generationId}/refine`, { method: 'POST', body: params, timeoutMs: 300_000 }),

  updateMetadata: (generationId: number, updates: {
    title?: string;
    caption?: string;
    bpm?: number;
    key?: string;
    duration?: number;
    subject?: string;
    lyrics?: string;
  }): Promise<any> =>
    api(`/api/lireek/generations/${generationId}`, { method: 'PATCH', body: updates }),

  deleteGeneration: (id: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/generations/${id}`, { method: 'DELETE' }),

  // ── Export ──────────────────────────────────────────────────────────────
  exportGeneration: (generationId: number): Promise<{ exported: boolean; path: string }> =>
    api(`/api/lireek/generations/${generationId}/export`, { method: 'POST' }),

  // ── Album Presets ───────────────────────────────────────────────────────
  getPreset: (lyricsSetId: number): Promise<{ preset: AlbumPreset | null }> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`),

  upsertPreset: (lyricsSetId: number, params: {
    adapter_path?: string;
    adapter_scale?: number;
    adapter_group_scales?: { self_attn: number; cross_attn: number; mlp: number; cond_embed: number };
    reference_track_path?: string;
    audio_cover_strength?: number;
  }): Promise<{ preset: AlbumPreset }> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`, { method: 'PUT', body: params }),

  deletePreset: (lyricsSetId: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/lyrics-sets/${lyricsSetId}/preset`, { method: 'DELETE' }),

  listAllPresets: (): Promise<{ presets: AlbumPreset[] }> =>
    api('/api/lireek/presets'),

  // ── Slop Detection ──────────────────────────────────────────────────────
  slopScan: (text: string): Promise<any> =>
    api('/api/lireek/slop-scan', { method: 'POST', body: { text } }),

  // ── Bulk Operations ─────────────────────────────────────────────────────
  purgeAll: (): Promise<any> =>
    api('/api/lireek/purge', { method: 'POST' }),

  // ── Audio Generations ───────────────────────────────────────────────────
  linkAudio: (generationId: number, jobId: string): Promise<any> =>
    api(`/api/lireek/generations/${generationId}/audio`, { method: 'POST', body: { job_id: jobId } }),

  getAudioGenerations: (generationId: number): Promise<{ audio_generations: AudioGeneration[] }> =>
    api(`/api/lireek/generations/${generationId}/audio`),

  deleteAudioGeneration: (agId: number): Promise<{ deleted: boolean }> =>
    api(`/api/lireek/audio-generations/${agId}`, { method: 'DELETE' }),

  resolveAudioGeneration: (jobId: string, audioUrl: string, coverUrl?: string): Promise<{ updated: boolean }> =>
    api('/api/lireek/audio-generations/resolve', {
      method: 'PATCH',
      body: { job_id: jobId, audio_url: audioUrl, cover_url: coverUrl || null },
    }),

  // ── Direct Audio Generation (via cpp engine) ────────────────────────────
  submitAudioGeneration: (params: {
    lyrics: string;
    prompt: string;
    bpm?: number;
    key_scale?: string;
    audio_duration?: number;
    lireek_adapter_path?: string;
    lireek_adapter_scale?: number;
    lireek_group_scales?: { self_attn: number; cross_attn: number; mlp: number };
    mastering_params?: { mode: string; reference_file?: string };
  }): Promise<{ job_id: string }> =>
    api('/api/generate', { method: 'POST', body: params }),

  // ── Prompts ───────────────────────────────────────────────────────────
  listPrompts: (): Promise<{ prompts: { name: string; custom: string | null }[] }> =>
    api('/api/lireek/prompts'),

  savePrompt: (name: string, value: string): Promise<{ success: boolean }> =>
    api(`/api/lireek/prompts/${name}`, { method: 'PUT', body: { value } }),

  resetPrompt: (name: string): Promise<{ success: boolean }> =>
    api(`/api/lireek/prompts/${name}`, { method: 'DELETE' }),

  // ── Recent Songs ────────────────────────────────────────────────────────
  getRecentSongs: (limit = 30): Promise<{ songs: RecentSong[] }> =>
    api(`/api/lireek/recent-songs?limit=${limit}`),

  // ── Curated Profile ─────────────────────────────────────────────────────
  buildCuratedProfile: (artistId: number, params: {
    selections: { lyrics_set_id: number; song_indices: number[] }[];
    provider: string;
    model?: string;
  }): Promise<{ lyrics_set: LyricsSet; profile: Profile }> =>
    api(`/api/lireek/artists/${artistId}/curated-profile`, {
      method: 'POST', body: params, timeoutMs: 300_000,
    }),

  // ── LLM Providers ───────────────────────────────────────────────────────
  getProviders: (): Promise<{ id: string; name: string; available: boolean; models: string[]; default_model: string }[]> =>
    api('/api/lireek/providers'),
};

// ── SSE Streaming ─────────────────────────────────────────────────────────

export interface StreamCallbacks {
  onChunk?: (text: string) => void;
  onPhase?: (phase: string) => void;
  onResult?: (data: any) => void;
  onError?: (message: string) => void;
}

async function consumeSSE(url: string, body: any, callbacks: StreamCallbacks): Promise<void> {
  const resp = await fetch(`${API_BASE}${url}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }

  const reader = resp.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';
  let currentEventType = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();

      // Named event lines: "event: chunk"
      if (trimmed.startsWith('event: ')) {
        currentEventType = trimmed.slice(7).trim();
        continue;
      }

      if (!trimmed.startsWith('data: ')) continue;
      const jsonStr = trimmed.slice(6).trim();
      if (!jsonStr) continue;

      try {
        const data = JSON.parse(jsonStr);

        // Use named event type if available, else fallback to inline type
        const eventType = currentEventType || data.type;
        currentEventType = ''; // Reset after use

        switch (eventType) {
          case 'chunk': callbacks.onChunk?.(data.text ?? data); break;
          case 'phase': callbacks.onPhase?.(data.phase ?? data.text ?? data); break;
          case 'complete':
          case 'result': callbacks.onResult?.(data); break;
          case 'error': callbacks.onError?.(data.error ?? data.message ?? data); break;
        }
      } catch { /* skip malformed lines */ }
    }
  }
}

export const streamBuildProfile = (
  lyricsSetId: number,
  req: { provider: string; model?: string },
  callbacks: StreamCallbacks,
): Promise<void> =>
  consumeSSE(`/api/lireek/lyrics-sets/${lyricsSetId}/build-profile-stream`, req, callbacks);

export const streamGenerate = (
  profileId: number,
  req: { profile_id: number; provider: string; model?: string; extra_instructions?: string },
  callbacks: StreamCallbacks,
): Promise<void> =>
  consumeSSE(`/api/lireek/profiles/${profileId}/generate-stream`, req, callbacks);

export const streamRefine = (
  generationId: number,
  req: { provider: string; model?: string },
  callbacks: StreamCallbacks,
): Promise<void> =>
  consumeSSE(`/api/lireek/generations/${generationId}/refine-stream`, req, callbacks);

export const skipThinking = (): Promise<void> =>
  api('/api/lireek/skip-thinking', { method: 'POST' });

export const streamBuildCuratedProfile = (
  artistId: number,
  req: {
    selections: { lyrics_set_id: number; song_indices: number[] }[];
    provider: string;
    model?: string;
  },
  callbacks: StreamCallbacks,
): Promise<void> =>
  consumeSSE(`/api/lireek/artists/${artistId}/curated-profile-stream`, req, callbacks);
