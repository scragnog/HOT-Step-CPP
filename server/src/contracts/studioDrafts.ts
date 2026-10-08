// Typed document contracts for ordered playlist snapshots and studio drafts.
import { z } from 'zod/v4';

export const playlistItemSchema = z.object({
  id: z.string().min(1).max(200),
  title: z.string(),
  audioUrl: z.string(),
  masteredAudioUrl: z.string().optional(),
  noAdapterAudioUrl: z.string().optional(),
  artistName: z.string().optional(),
  coverUrl: z.string().optional(),
  duration: z.number().finite().nonnegative().optional(),
  style: z.string().optional(),
  generationParams: z.unknown().optional(),
}).passthrough();
export type PlaylistItem = z.infer<typeof playlistItemSchema>;

export const playlistSchema = z.object({ items: z.array(playlistItemSchema).max(2000) }).superRefine((body, ctx) => {
  const ids = new Set<string>();
  for (const [index, item] of body.items.entries()) {
    if (ids.has(item.id)) ctx.addIssue({ code: 'custom', path: ['items', index, 'id'], message: 'Duplicate playlist item' });
    ids.add(item.id);
  }
});
export type PlaylistBody = z.infer<typeof playlistSchema>;

export const studioKindSchema = z.enum(['create', 'cover', 'repaint', 'storm', 'stem-studio', 'stem-builder']);
export type StudioKind = z.infer<typeof studioKindSchema>;

// Fields retain their original storage names, including false, zero and null.
// Selection pointers, playback state and device handles never enter a draft.
export const studioDraftSchema = z.object({
  studio: studioKindSchema,
  fields: z.record(z.string(), z.unknown()),
  backendId: z.string().optional(),
  sourceAssetId: z.string().optional(),
  sourceSongId: z.string().optional(),
  sourceRevision: z.string().optional(),
}).passthrough();
export type StudioDraftBody = z.infer<typeof studioDraftSchema>;

export const saveDraftSchema = z.object({ body: studioDraftSchema, expectedRevision: z.number().int().positive().optional() });
export const importDraftSchema = z.object({
  storageKey: z.string().min(1).max(500),
  raw: z.string().max(1024 * 1024),
  sourceHash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  schemaVersion: z.literal(1),
  expectedRevision: z.number().int().nonnegative(),
  resolution: z.enum(['keep-both', 'replace']).optional(),
});
