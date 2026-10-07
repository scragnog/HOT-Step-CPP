import { z } from 'zod/v4';

/** Compact generation submission; legacy flat bodies remain accepted. */
export const generationIntentSchema = z.object({
  contract: z.literal('generation-intent/1'),
  params: z.record(z.string(), z.unknown()),
  input: z.record(z.string(), z.unknown()).optional(),
  settings: z.object({
    triggerUseFilename: z.boolean().optional(),
    triggerPlacement: z.enum(['prepend', 'append', 'replace']).optional(),
  }).optional(),
});

export type GenerationIntent = z.infer<typeof generationIntentSchema>;

export function isGenerationIntent(value: unknown): value is { contract: 'generation-intent/1'; input?: unknown } {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && (value as Record<string, unknown>).contract === 'generation-intent/1';
}
