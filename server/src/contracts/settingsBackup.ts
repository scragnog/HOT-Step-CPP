import { z } from 'zod/v4';

const text = z.string();
const id = z.number().int().positive();
const integer = z.number().int();
const nullableText = text.nullable();
const nullableNumber = z.number().nullable();
const LYRIC_PROMPT_PREFIX = 'prompt_';
const INSTAGEN_PROMPT_KEY = 'instagen_system_prompt';
const lyricSettingKey = text.refine(key => key.startsWith(LYRIC_PROMPT_PREFIX) || key === INSTAGEN_PROMPT_KEY,
  'Only Lyric Studio prompt settings can be imported.');
const jsonText = (kind: 'array' | 'object') => text.refine(value => {
  try {
    const parsed: unknown = JSON.parse(value);
    return kind === 'array' ? Array.isArray(parsed) : parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed);
  } catch { return false; }
}, `Expected a JSON ${kind}.`);
export const appSchema = z.object({
  coResident: z.boolean(), cacheLmCodes: z.boolean(), speculativeDecoding: z.boolean(),
  downloadFormat: z.enum(['wav', 'flac', 'opus', 'mp3']),
  downloadMp3Bitrate: z.number().min(128).max(320),
  downloadOpusBitrate: z.number().min(96).max(256),
  downloadVersion: z.enum(['original', 'mastered', 'both']), downloadIncludeLatent: z.boolean(),
  triggerUseFilename: z.boolean(), triggerPlacement: z.enum(['prepend', 'append', 'replace']),
  discoKickExtract: z.boolean(), parallelWhisper: z.boolean(), parallelQualityEval: z.boolean(),
  parallelCoverArt: z.boolean(), generationTimeoutMinutes: z.number().positive(),
}).strict();
// The version fixes the portable columns; never accept table or column names from a file.
export const backupTables = {
  artists: z.object({ id, name: text, created_at: text, image_url: nullableText, genius_id: integer.nullable() }).strict(),
  lyrics_sets: z.object({ id, artist_id: id, album: nullableText, max_songs: integer, songs: jsonText('array'), fetched_at: text, image_url: nullableText }).strict(),
  profiles: z.object({ id, lyrics_set_id: id, provider: text, model: text, profile_data: jsonText('object'), created_at: text }).strict(),
  generations: z.object({
    id, profile_id: id, provider: text, model: text, extra_instructions: nullableText,
    title: text, subject: text, lyrics: text, system_prompt: text, user_prompt: text, created_at: text,
    bpm: integer, key: text, caption: text, caption_mm3: text, duration: integer, caption_yue2: text,
    parent_generation_id: id.nullable(), audio_generated_count: integer, first_generated_at: nullableText,
    download_count: integer, first_downloaded_at: nullableText,
  }).strict(),
  album_presets: z.object({
    id, lyrics_set_id: id, adapter_path: nullableText, adapter_scale: nullableNumber,
    adapter_group_scales: nullableText, reference_track_path: nullableText, audio_cover_strength: nullableNumber,
    lm_adapter_path: nullableText, lm_adapter_scale: nullableNumber, created_at: nullableText,
    mm3_adapter_path: nullableText, yue2_ar_adapter_path: nullableText, yue2_nar_adapter_path: nullableText,
  }).strict(),
  audio_generations: z.object({
    id, generation_id: id, hotstep_job_id: text, audio_url: nullableText, cover_url: nullableText,
    created_at: nullableText, downloaded_at: nullableText, lyrics_set_id: id.nullable(),
  }).strict(),
};

const lyricSchema = z.object({
  artists: z.array(backupTables.artists), lyrics_sets: z.array(backupTables.lyrics_sets),
  profiles: z.array(backupTables.profiles), generations: z.array(backupTables.generations),
  album_presets: z.array(backupTables.album_presets), audio_generations: z.array(backupTables.audio_generations),
}).strict();
export const backupSchema = z.object({
  version: z.literal(1), exportedAt: z.iso.datetime(),
  settings: z.object({ app: appSchema, lyricStudio: z.record(lyricSettingKey, text) }).strict(),
  lyricStudio: lyricSchema,
}).strict();

export type SettingsBackup = z.infer<typeof backupSchema>;
export type BackupCounts = Record<keyof typeof backupTables | 'settings', number>;
export interface BackupPreview { current: BackupCounts; incoming: BackupCounts; exportedAt: string }
