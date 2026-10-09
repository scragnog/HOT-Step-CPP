// contracts/songs.ts — the library's Song as /api/songs sends it
// (docs/dev/frontend-library.md).
//
// A Song on the wire is its database row, with two columns decoded: `tags`
// becomes an array and `is_public` a boolean. Every other column, including
// the JSON-in-a-string `generation_params`, is sent as stored. Display
// normalisation (camelCase, fallbacks, titles) belongs to each client.

/** Every field of a Song, in column order. A new column appears here first. */
export const SONG_FIELDS = [
  'id', 'user_id', 'title', 'lyrics', 'style', 'caption', 'audio_url', 'cover_url', 'duration', 'bpm', 'key_scale',
  'time_signature', 'tags', 'is_public', 'like_count', 'view_count', 'dit_model', 'generation_params', 'mastered_audio_url',
  'backend', 'created_at', 'latent_url', 'quality_scores', 'cover_art_subject', 'kick_stem_url', 'snare_stem_url',
  'hihat_stem_url', 'disco_data_url', 'metadata_overrides', 'noadapter_audio_url',
] as const;

export interface Song {
  id: string;
  user_id: string;
  title: string;
  lyrics: string;
  style: string;
  caption: string;
  /** Relative media URL, e.g. /audio/<id>.wav. '' when the song has none. */
  audio_url: string;
  cover_url: string;
  /** Seconds. 0 when unknown. */
  duration: number;
  bpm: number;
  key_scale: string;
  time_signature: string;
  tags: string[];
  is_public: boolean;
  like_count: number;
  view_count: number;
  dit_model: string;
  /** The generation request as a JSON string ('{}' when none). */
  generation_params: string;
  mastered_audio_url: string;
  /** The engine that rendered it, e.g. 'ace', 'yue2', 'minimax-m3'. */
  backend: string;
  /** SQLite datetime text, UTC: 'YYYY-MM-DD HH:MM:SS'. */
  created_at: string;
  latent_url: string;
  quality_scores: string;
  cover_art_subject: string;
  kick_stem_url: string;
  snare_stem_url: string;
  hihat_stem_url: string;
  disco_data_url: string;
  /** Embed-tag overrides as a JSON string; '' when none. */
  metadata_overrides: string;
  noadapter_audio_url: string;
}

/** GET /api/songs[?source=], newest first (created_at DESC), unpaginated. */
export interface SongListResponse { songs: Song[] }
/** GET /api/songs/ids. */
export interface SongIdsResponse { ids: string[] }
/** GET /api/songs/:id, POST /api/songs and PATCH /api/songs/:id. */
export interface SongResponse { song: Song }

/** GET /api/songs/recent[?source=create|lyric-studio|cover-studio|all&limit=50]:
 *  a normalised shape, newest first, at most `limit` (default 50). */
export const RECENT_SONG_FIELDS = [
  'id', 'title', 'audio_url', 'mastered_audio_url', 'noadapter_audio_url', 'latent_url', 'kick_stem_url', 'snare_stem_url',
  'hihat_stem_url', 'disco_data_url', 'cover_url', 'duration', 'lyrics', 'caption', 'style', 'bpm', 'key_scale',
  'time_signature', 'metadata_overrides', 'source', 'created_at', 'artist_name', 'artist_image', 'album', 'generation_id',
] as const;
export type RecentSong = Pick<Song, 'id' | 'title' | 'audio_url' | 'mastered_audio_url' | 'noadapter_audio_url' | 'latent_url'
  | 'kick_stem_url' | 'snare_stem_url' | 'hihat_stem_url' | 'disco_data_url' | 'cover_url' | 'duration' | 'lyrics' | 'caption'
  | 'style' | 'bpm' | 'key_scale' | 'time_signature' | 'metadata_overrides' | 'created_at'> & {
  /** generation_params.source, default 'create'. */
  source: string;
  artist_name: string;
  artist_image: string;
  album: string;
  generation_id: number | null;
};
export interface RecentSongsResponse { songs: RecentSong[] }

/** Fields PATCH /api/songs/:id writes. Others in the body are ignored. `tags`
 *  (array) and `metadata_overrides` (object; null or '' clears) are also
 *  accepted. */
export const SONG_EDITABLE_FIELDS = ['title', 'lyrics', 'style', 'caption', 'cover_url', 'is_public',
  'bpm', 'key_scale', 'time_signature', 'dit_model', 'cover_art_subject'] as const;
export type SongPatch = Partial<Pick<Song, typeof SONG_EDITABLE_FIELDS[number]>> & {
  tags?: string[];
  metadata_overrides?: Record<string, unknown> | null | '';
};

/** POST /api/songs/import (multipart `audio`, up to 50 files, optional
 *  `description`). 200 when at least one file imported; 400 with the same
 *  `errors` when none did. */
export interface SongImportResponse {
  songs: Song[];
  errors: Array<{ file: string; error: string }>;
  /** The accepted extensions. */
  accepted: string[];
}

/** DELETE /api/songs/:id. Also removes the song's files. */
export interface SongDeleteResponse { success: true }
/** DELETE /api/songs (all of the user's songs) and POST /api/songs/bulk-delete { ids }. */
export interface SongBulkDeleteResponse { success: true; deletedCount: number }
