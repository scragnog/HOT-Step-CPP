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
/**
 * Opt-in per-job backend override. When present it names the registered
 * backend this one job runs on; the active backend is not changed. It is a
 * separate field from the legacy `backend`, which stays log-only.
 */
export const backendOverrideSchema = z.string().min(1);

export const generationIntentSchema = z.object({
  contract: z.literal('generation-intent/1'),
  backendOverride: backendOverrideSchema.optional(),
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

/** Read and validate `backendOverride` from a flat body or an intent. Returns
 *  undefined when absent; throws on a present but malformed value. */
export function readBackendOverride(raw: unknown): string | undefined {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) return undefined;
  const value = (raw as Record<string, unknown>).backendOverride;
  if (value === undefined) return undefined;
  const parsed = backendOverrideSchema.safeParse(value);
  if (!parsed.success) throw new Error('backendOverride must be a non-empty string when supplied');
  return parsed.data;
}

/** POST /api/generate/preflight: would this request fit in VRAM now?
 *  fits: runs as sent without unloading anything. passes: runs, but only after
 *  the engine unloads another backend's idle models or plans in a smaller shape
 *  (`detail` says which). refused: the engine would refuse it now. unknown: no
 *  admission model for this backend, no measurement, or the GPU is busy. */
export type GenerationPreflightVerdict = 'fits' | 'passes' | 'refused' | 'unknown';

export interface GenerationPreflight {
  verdict: GenerationPreflightVerdict;
  /** What the request needs at its largest stage, GiB. */
  estimateGb?: number;
  /** What it can use without unloading another backend, GiB. */
  freeGb?: number;
  detail?: string;
}
