import { z } from 'zod/v4';
import type { GenerationControls } from './generationControls.js';

/**
 * Compact generation submission; legacy flat bodies remain accepted.
 * `params` stays `record(unknown)` deliberately — see
 * docs/dev/frontend-create-controls.md for the documented Create control
 * dictionary (generationControls.ts). That schema is permissive by
 * construction (every field optional, unknown keys pass through), so it
 * never narrows what this route accepts; `GenerationIntentParams` just gives
 * callers the documented shape to type against.
 */
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
export type GenerationIntentParams = GenerationControls;

export function isGenerationIntent(value: unknown): value is { contract: 'generation-intent/1'; input?: unknown } {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && (value as Record<string, unknown>).contract === 'generation-intent/1';
}
