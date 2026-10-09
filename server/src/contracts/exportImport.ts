// contracts/exportImport.ts — /api/export-import: uploads, imports, export
// resolution and profile-import validation (docs/dev/frontend-media.md).
//
// Clients send ids and options, never local paths: a file reaches the server
// as a multipart upload that becomes an asset id, and an export is resolved to
// relative download URLs the client fetches itself.

import { z } from 'zod/v4';

export const audioFormat = z.enum(['wav', 'flac', 'opus', 'mp3']);
export const audioVariant = z.enum(['original', 'mastered', 'noadapter', 'latent']);
export type AudioFormat = z.infer<typeof audioFormat>;
export type AudioVariant = z.infer<typeof audioVariant>;

/** Extensions POST /assets accepts (case-insensitive). Anything else answers 400. */
export const IMPORT_AUDIO_EXTENSIONS = [
  '.wav', '.mp3', '.flac', '.m4a', '.mp4', '.aac', '.ogg', '.opus', '.webm', '.aiff', '.aif',
] as const;
/** Largest upload POST /assets accepts, in bytes. */
export const IMPORT_MAX_UPLOAD_BYTES = 500 * 1024 * 1024;

/** POST /exports/resolve. */
export const exportRequest = z.strictObject({
  items: z.array(z.strictObject({
    songId: z.string().min(1).max(128).regex(/^[a-zA-Z0-9_-]+$/),
    variant: audioVariant.optional(),
    audioUrl: z.string().max(1024).optional(),
    srcUrl: z.string().max(1024).optional(),
  })).min(1).max(100),
  format: audioFormat.default('flac'),
  downloadVersion: z.enum(['original', 'mastered', 'both']).default('mastered'),
  includeLatent: z.boolean().default(false),
  bitrate: z.number().int().min(32).max(512).optional(),
  artist: z.string().max(200).optional(),
  prepend: z.string().max(200).optional(),
});

/** POST /imports. */
export const importRequest = z.strictObject({
  items: z.array(z.strictObject({ assetId: z.uuid(), description: z.string().max(1000).optional() })).min(1).max(50),
});

/** POST /profiles/validate. */
export const profileImportRequest = z.strictObject({
  filename: z.string().min(1).max(255),
  profile: z.record(z.string(), z.unknown()),
});

/** Unwrap a saved profile wrapper ({ name, saved_at, data }) or take bare
 *  preset JSON, and name it from the file. Throws on an empty profile. */
export function validateProfileImport(input: unknown): ProfileImportResult {
  const parsed = profileImportRequest.parse(input);
  const wrapper = parsed.profile;
  const data = wrapper._format === undefined && typeof wrapper.data === 'object' && wrapper.data !== null
    && !Array.isArray(wrapper.data) && (wrapper.data as Record<string, unknown>)._format === 'hot-step-preset'
    ? wrapper.data as Record<string, unknown> : wrapper;
  if (Object.keys(data).length === 0) throw new Error('Profile data is empty');
  return { name: parsed.filename.replace(/\.json$/i, '').replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').slice(0, 100).trim() || 'imported', data };
}

export type ExportRequest = z.infer<typeof exportRequest>;
export type ImportRequest = z.infer<typeof importRequest>;

// ── Responses ──

/** POST /assets (multipart field `audio`). */
export interface AssetUploadResponse { assetId: string }

/** One entry of POST /exports/resolve's `items`. An input item can produce
 *  several entries (both versions, a latent), all with its `index`. A failed
 *  entry has `error` and no `url`; the others still resolve. */
export interface ResolvedExport {
  index: number;
  songId: string;
  variant?: string;
  /** The name the download route will give the file. */
  filename?: string;
  /** Relative: GET it on the same origin. */
  url?: string;
  error?: string;
}
export interface ExportResolveResponse { items: ResolvedExport[] }

/** One entry of POST /imports' `items`, in request order. `song` is the new
 *  library row; a failed entry has `error` and no `song`. */
export interface ImportResultItem {
  index: number;
  assetId: string;
  song?: Record<string, unknown> & { id: string; tags: unknown[]; is_public: boolean };
  error?: string;
}
export interface ImportResponse { items: ImportResultItem[] }

/** POST /profiles/validate. Save `data` under `name` with the profile routes. */
export interface ProfileImportResult { name: string; data: Record<string, unknown> }

/** A whole-request refusal. `issues` comes from schema validation (400). */
export interface ExportImportError {
  error: string;
  issues?: Array<{ path: Array<string | number>; message: string; code?: string }>;
  details?: string;
}
