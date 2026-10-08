import { z } from 'zod/v4';

export const audioFormat = z.enum(['wav', 'flac', 'opus', 'mp3']);
export const audioVariant = z.enum(['original', 'mastered', 'noadapter', 'latent']);

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

export const importRequest = z.strictObject({
  items: z.array(z.strictObject({ assetId: z.uuid(), description: z.string().max(1000).optional() })).min(1).max(50),
});

export const profileImportRequest = z.strictObject({
  filename: z.string().min(1).max(255),
  profile: z.record(z.string(), z.unknown()),
});

export function validateProfileImport(input: unknown) {
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
